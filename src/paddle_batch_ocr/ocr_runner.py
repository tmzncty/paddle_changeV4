"""Serial, manifest-aware OCR execution built on the lazy PaddleX adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .manifest import ManifestStore
from .paddlex_adapter import (
    PipelineRef,
    PaddleXResultError,
    create_ocr_pipeline,
    default_ocr_predict_kwargs,
    parse_paddlex_ocr_result,
    predict_one_to_json,
)
from .safety import is_within


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class OcrRunnerError(RuntimeError):
    """Raised when OCR batch setup is unsafe or inconsistent."""


@dataclass(frozen=True)
class OcrTask:
    source: Path
    output_json: Path


@dataclass(frozen=True)
class OcrTaskResult:
    source: Path
    output_json: Path
    status: str
    error: Optional[str] = None


@dataclass(frozen=True)
class OcrBatchResult:
    tasks: Tuple[OcrTaskResult, ...]

    @property
    def success_count(self) -> int:
        return sum(result.status == "success" for result in self.tasks)

    @property
    def skipped_count(self) -> int:
        return sum(result.status == "skipped" for result in self.tasks)

    @property
    def failed_count(self) -> int:
        return sum(result.status == "failed" for result in self.tasks)


def _safe_output_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise OcrRunnerError(f"refusing symlinked OCR output directory: {raw}")
    return raw.resolve(strict=False)


def discover_ocr_tasks(input_path: Path, output_dir: Path) -> Tuple[OcrTask, ...]:
    """Discover image inputs and map them to deterministic ``*_result.json`` paths."""

    raw_input = Path(input_path).expanduser()
    if raw_input.is_symlink():
        raise OcrRunnerError(f"refusing symlinked OCR input: {raw_input}")
    source = raw_input.resolve(strict=True)
    target_root = _safe_output_root(output_dir)

    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise OcrRunnerError(f"unsupported OCR image extension: {source}")
        return (
            OcrTask(
                source=source,
                output_json=target_root / f"{source.stem}_result.json",
            ),
        )

    if not source.is_dir():
        raise OcrRunnerError(f"OCR input is neither a file nor directory: {source}")
    if target_root == source or is_within(target_root, source):
        raise OcrRunnerError("OCR output directory must not be inside the input directory")

    tasks: List[OcrTask] = []
    for root, dirs, files in os.walk(source):
        dirs.sort()
        files.sort()
        root_path = Path(root)
        for filename in files:
            image = root_path / filename
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image.is_symlink():
                raise OcrRunnerError(f"refusing symlinked OCR image: {image}")
            resolved = image.resolve(strict=True)
            if not is_within(resolved, source):
                raise OcrRunnerError(f"OCR image resolves outside input root: {image}")
            relative = resolved.relative_to(source)
            output_relative = relative.with_name(f"{relative.stem}_result.json")
            tasks.append(
                OcrTask(
                    source=resolved,
                    output_json=target_root / output_relative,
                )
            )

    return tuple(tasks)


def _validate_existing_result(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        parse_paddlex_ocr_result(payload)
    except (OSError, json.JSONDecodeError, PaddleXResultError) as exc:
        raise OcrRunnerError(f"existing OCR result is invalid: {path}: {exc}") from exc


def run_ocr_batch(
    input_path: Path,
    output_dir: Path,
    *,
    pipeline_ref: PipelineRef = "OCR",
    device: Optional[str] = None,
    engine: Optional[str] = None,
    use_hpip: Optional[bool] = None,
    manifest_path: Optional[Path] = None,
    resume: bool = True,
    overwrite: bool = False,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    create_pipeline_fn: Optional[Callable[..., object]] = None,
) -> OcrBatchResult:
    """Run OCR serially with one lazily-created pipeline instance."""

    tasks = discover_ocr_tasks(input_path, output_dir)
    if not tasks:
        raise OcrRunnerError(f"no supported OCR images found under {input_path}")

    predict_kwargs = default_ocr_predict_kwargs(
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )

    pipeline: Optional[object] = None
    pipeline_error: Optional[Exception] = None

    def get_pipeline() -> object:
        nonlocal pipeline, pipeline_error
        if pipeline is not None:
            return pipeline
        if pipeline_error is not None:
            raise OcrRunnerError(
                "PaddleX pipeline initialization previously failed: "
                f"{type(pipeline_error).__name__}: {pipeline_error}"
            ) from pipeline_error
        try:
            pipeline = create_ocr_pipeline(
                pipeline_ref,
                device=device,
                engine=engine,
                use_hpip=use_hpip,
                create_pipeline_fn=create_pipeline_fn,
            )
        except Exception as exc:
            # Cache a batch-wide initialization failure. A broken runtime/model
            # setup must not repeat expensive pipeline construction once per
            # image in a very large input tree.
            pipeline_error = exc
            raise
        return pipeline

    store = ManifestStore(manifest_path) if manifest_path is not None else None
    results: List[OcrTaskResult] = []
    try:
        for task in tasks:
            try:
                previous_record = (
                    store.get_job(task.source, "ocr") if store is not None else None
                )
                manifest_needs_run = (
                    store.needs_run(task.source, "ocr") if store is not None else True
                )

                if task.output_json.exists():
                    can_adopt_existing = (
                        resume
                        and (
                            store is None
                            or previous_record is None
                            or not manifest_needs_run
                        )
                    )
                    if can_adopt_existing:
                        _validate_existing_result(task.output_json)
                        if store is not None:
                            store.mark_success(
                                task.source,
                                "ocr",
                                result_path=task.output_json,
                            )
                        results.append(
                            OcrTaskResult(
                                source=task.source,
                                output_json=task.output_json,
                                status="skipped",
                            )
                        )
                        continue

                    if not overwrite:
                        reason = (
                            "existing result is stale according to manifest; use --overwrite"
                            if store is not None and previous_record is not None and manifest_needs_run
                            else "OCR output already exists; enable resume or overwrite"
                        )
                        raise OcrRunnerError(f"{reason}: {task.output_json}")

                if store is not None:
                    store.mark_started(
                        task.source,
                        "ocr",
                        worker=f"pid-{os.getpid()}",
                        device=device or "auto",
                    )

                predict_one_to_json(
                    get_pipeline(),
                    task.source,
                    task.output_json,
                    overwrite=overwrite,
                    predict_kwargs=predict_kwargs,
                )

                if store is not None:
                    store.mark_success(
                        task.source,
                        "ocr",
                        result_path=task.output_json,
                    )
                results.append(
                    OcrTaskResult(
                        source=task.source,
                        output_json=task.output_json,
                        status="success",
                    )
                )
            except Exception as exc:
                if store is not None:
                    try:
                        store.mark_failure(task.source, "ocr", exc)
                    except Exception:
                        pass
                results.append(
                    OcrTaskResult(
                        source=task.source,
                        output_json=task.output_json,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    finally:
        if store is not None:
            store.close()

    return OcrBatchResult(tasks=tuple(results))
