"""Safety-first targeted retries driven by failed manifest provenance.

The retry layer treats manifest rows as untrusted input that must be validated
before any execution. Planning is read-only. Execution is explicit and
sequential in the first public version so retry behavior remains auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from .config import ProjectConfig
from .manifest import JobRecord, ManifestStore
from .manifest_reporting import query_manifest_job_page
from .naming import find_matching_json
from .ocr_runner import OcrBatchResult, discover_ocr_tasks, run_ocr_batch
from .ocr_schema import OcrSchemaError, parse_ocr_page
from .pdf_render import RenderResult, render_pdf
from .safety import UnsafePathError, is_within, validate_destructive_target
from .searchable_pdf import (
    SearchablePdfResult,
    build_searchable_pdf,
    discover_numbered_page_images,
)


SUPPORTED_RETRY_STAGES = {"ocr", "render", "searchable_pdf"}
_OCR_PREDICT_KEYS = {
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
}


class RetryError(RuntimeError):
    """Raised when retry selection or execution cannot proceed safely."""


@dataclass(frozen=True)
class RetryCandidate:
    source: Path
    stage: str
    intended_result: Optional[Path]
    execution_profile: Optional[Dict[str, object]]
    eligible: bool
    reason: Optional[str]
    source_size: int
    source_mtime_ns: int
    execution_profile_json: Optional[str]


@dataclass(frozen=True)
class RetryPlan:
    manifest_path: Path
    total_matching: int
    candidates: Tuple[RetryCandidate, ...]
    overwrite: bool

    @property
    def eligible_count(self) -> int:
        return sum(candidate.eligible for candidate in self.candidates)

    @property
    def ineligible_count(self) -> int:
        return len(self.candidates) - self.eligible_count


@dataclass(frozen=True)
class RetryItemResult:
    source: Path
    stage: str
    intended_result: Optional[Path]
    status: str
    error: Optional[str] = None


@dataclass(frozen=True)
class RetryExecutionResult:
    items: Tuple[RetryItemResult, ...]

    @property
    def success_count(self) -> int:
        return sum(item.status == "success" for item in self.items)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.items)

    @property
    def ineligible_count(self) -> int:
        return sum(item.status == "ineligible" for item in self.items)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_profile(value: object) -> Dict[str, object]:
    if not isinstance(value, str) or not value:
        raise RetryError("execution profile is missing")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RetryError(f"execution profile JSON is invalid: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RetryError("execution profile must decode to an object")
    return decoded


def _lexical_absolute(path: Path) -> Path:
    """Normalize ``.``/``..`` and make absolute without resolving symlinks."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_existing_symlink_components(path: Path, *, label: str) -> Path:
    """Reject existing symlinks anywhere in the lexical path.

    ``Path.resolve()`` cannot be used before this check: doing so would erase
    the very symlink component that the retry trust boundary promises to reject.
    Missing future components are fine and are checked again immediately before
    execution.
    """

    lexical = _lexical_absolute(path)
    anchor = Path(lexical.anchor)
    current = anchor
    parts = lexical.parts[1:] if lexical.anchor else lexical.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise RetryError(f"{label} contains symlink component: {current}")
    return lexical


def _source_path(row: Mapping[str, object]) -> Path:
    raw = row.get("source_path")
    if not isinstance(raw, str) or not raw:
        raise RetryError("source_path is missing")
    path = _reject_existing_symlink_components(Path(raw), label="source path")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RetryError(f"source no longer exists: {path}") from exc
    if not resolved.is_file():
        raise RetryError(f"source is not a regular file: {resolved}")

    stat = resolved.stat()
    if stat.st_size != int(row["source_size"]):
        raise RetryError("source size changed since the failed attempt")
    if stat.st_mtime_ns != int(row["source_mtime_ns"]):
        raise RetryError("source mtime changed since the failed attempt")
    return resolved


def _validate_source_scope(source: Path, config: ProjectConfig) -> None:
    """Limit automatic reads to configured inputs or project-produced artifacts."""

    allowed_roots = [config.output_root]
    allowed_roots.extend(item.path for item in config.input_sources)
    if any(source == root or is_within(source, root) for root in allowed_roots):
        return
    raise RetryError(
        f"source is outside configured input sources and output_root: {source}"
    )


def _reject_symlink_components(path: Path, root: Path) -> None:
    """Reject lexical symlink components and enforce output-root containment."""

    root = Path(root).expanduser().resolve(strict=False)
    lexical = _reject_existing_symlink_components(path, label="retry path")

    try:
        lexical.relative_to(root)
    except ValueError as exc:
        raise RetryError(
            f"target is outside configured output_root: {lexical}"
        ) from exc
    if lexical == root:
        raise RetryError("retry target cannot be output_root itself")

    resolved = lexical.resolve(strict=False)
    if not is_within(resolved, root):
        raise RetryError(f"target is outside configured output_root: {resolved}")
    if resolved == root:
        raise RetryError("retry target cannot be output_root itself")


def _target_path(
    row: Mapping[str, object],
    config: ProjectConfig,
    *,
    overwrite: bool,
) -> Path:
    raw = row.get("intended_result_path")
    if not isinstance(raw, str) or not raw:
        raise RetryError("intended_result_path is missing")

    target = Path(raw).expanduser()
    _reject_symlink_components(target, config.output_root)
    resolved = target.resolve(strict=False)

    # Every automatic retry is confined to the configured output tree. This
    # intentionally excludes older direct `ocr --manifest` runs that wrote to
    # arbitrary locations outside the project output root.
    try:
        validate_destructive_target(resolved, config.output_root)
    except UnsafePathError as exc:
        raise RetryError(str(exc)) from exc

    if resolved.exists() and not overwrite:
        raise RetryError(
            f"retry target already exists; pass --overwrite to replace it: {resolved}"
        )
    return resolved


def _validate_ocr_profile(
    profile: Mapping[str, object],
    source: Path,
    target: Path,
) -> Dict[str, object]:
    if profile.get("schema") != 2 or profile.get("kind") != "paddlex_ocr":
        raise RetryError("unsupported OCR execution profile schema")

    pipeline = profile.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise RetryError("OCR profile pipeline identity is missing")
    if pipeline.get("type") != "file":
        raise RetryError(
            "automatic retry requires a local pipeline file with SHA-256 provenance; "
            "named pipelines are not reproducible enough"
        )

    raw_pipeline = pipeline.get("path")
    stored_sha = pipeline.get("sha256")
    stored_size = pipeline.get("size")
    if not isinstance(raw_pipeline, str) or not raw_pipeline:
        raise RetryError("OCR pipeline path is missing")
    if not isinstance(stored_sha, str) or len(stored_sha) != 64:
        raise RetryError("OCR pipeline SHA-256 provenance is invalid")
    if isinstance(stored_size, bool) or not isinstance(stored_size, int) or stored_size < 0:
        raise RetryError("OCR pipeline size provenance is invalid")

    pipeline_path = _reject_existing_symlink_components(
        Path(raw_pipeline), label="OCR pipeline path"
    )
    try:
        pipeline_path = pipeline_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RetryError(f"OCR pipeline file no longer exists: {raw_pipeline}") from exc
    if not pipeline_path.is_file():
        raise RetryError(f"OCR pipeline path is not a file: {pipeline_path}")
    if pipeline_path.stat().st_size != stored_size:
        raise RetryError("OCR pipeline file size no longer matches failed-attempt provenance")
    if _sha256_file(pipeline_path) != stored_sha:
        raise RetryError("OCR pipeline SHA-256 no longer matches failed-attempt provenance")

    device = profile.get("device")
    if not isinstance(device, str) or not device:
        raise RetryError("OCR profile device is invalid")
    if device in {"auto", "gpu"}:
        raise RetryError(
            f"OCR retry requires an explicit reproducible device; {device!r} is ambiguous"
        )
    if device != "cpu" and not (
        device.startswith("gpu:") and device[4:].isdigit()
    ):
        raise RetryError(f"unsupported OCR retry device: {device}")

    engine = profile.get("engine")
    if engine is not None and not isinstance(engine, str):
        raise RetryError("OCR profile engine must be string or null")
    use_hpip = profile.get("use_hpip")
    if use_hpip is not None and not isinstance(use_hpip, bool):
        raise RetryError("OCR profile use_hpip must be bool or null")

    predict = profile.get("predict")
    if not isinstance(predict, Mapping):
        raise RetryError("OCR profile predict options are missing")
    if set(predict) != _OCR_PREDICT_KEYS:
        raise RetryError("OCR profile contains unsupported or missing predict options")
    for key in _OCR_PREDICT_KEYS:
        if not isinstance(predict[key], bool):
            raise RetryError(f"OCR predict option {key} must be bool")

    tasks = discover_ocr_tasks(source, target.parent)
    if len(tasks) != 1 or tasks[0].output_json.resolve(strict=False) != target:
        raise RetryError(
            "OCR intended result path does not match the deterministic output mapping"
        )

    return {
        "pipeline_ref": os.fspath(pipeline_path),
        "device": device,
        "engine": engine,
        "use_hpip": use_hpip,
        "use_doc_orientation_classify": bool(predict["use_doc_orientation_classify"]),
        "use_doc_unwarping": bool(predict["use_doc_unwarping"]),
        "use_textline_orientation": bool(predict["use_textline_orientation"]),
    }


def _validate_render_profile(
    profile: Mapping[str, object],
    target: Path,
) -> Dict[str, object]:
    expected_keys = {"schema", "kind", "dpi", "format", "alpha"}
    if set(profile) != expected_keys:
        raise RetryError("render profile contains unsupported or missing fields")
    if profile.get("schema") != 1 or profile.get("kind") != "pdf_render":
        raise RetryError("unsupported render execution profile schema")
    dpi = profile.get("dpi")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 36 <= dpi <= 1200:
        raise RetryError("render profile DPI is invalid")
    if profile.get("format") != "png" or profile.get("alpha") is not False:
        raise RetryError("retry only supports the current PNG/non-alpha render profile")
    if target.exists() and not target.is_dir():
        raise RetryError("render retry target exists but is not a directory")
    return {"dpi": dpi}


def _validate_searchable_profile(
    profile: Mapping[str, object],
    config: ProjectConfig,
) -> Dict[str, object]:
    expected_keys = {
        "schema",
        "kind",
        "fontname",
        "y_offset",
        "layout",
        "images_dir",
        "ocr_json_dir",
        "expected_page_count",
    }
    if set(profile) != expected_keys:
        raise RetryError("searchable-PDF profile contains unsupported or missing fields")
    if profile.get("schema") != 2 or profile.get("kind") != "searchable_pdf":
        raise RetryError("unsupported searchable-PDF execution profile schema")
    if profile.get("layout") != "legacy-v7":
        raise RetryError("retry only supports the frozen legacy-v7 searchable layout")

    fontname = profile.get("fontname")
    y_offset = profile.get("y_offset")
    expected_page_count = profile.get("expected_page_count")
    if not isinstance(fontname, str) or not fontname:
        raise RetryError("searchable-PDF fontname is invalid")
    if isinstance(y_offset, bool) or not isinstance(y_offset, (int, float)):
        raise RetryError("searchable-PDF y_offset is invalid")
    if (
        isinstance(expected_page_count, bool)
        or not isinstance(expected_page_count, int)
        or expected_page_count < 1
    ):
        raise RetryError("searchable-PDF expected_page_count is invalid")

    raw_images = profile.get("images_dir")
    raw_json = profile.get("ocr_json_dir")
    if not isinstance(raw_images, str) or not raw_images:
        raise RetryError("searchable-PDF images_dir is missing")
    if not isinstance(raw_json, str) or not raw_json:
        raise RetryError("searchable-PDF ocr_json_dir is missing")

    images_dir = Path(raw_images).expanduser()
    json_dir = Path(raw_json).expanduser()
    for label, path in (("images_dir", images_dir), ("ocr_json_dir", json_dir)):
        _reject_symlink_components(path, config.output_root)
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise RetryError(f"{label} is not a directory: {resolved}")
        if not is_within(resolved, config.output_root):
            raise RetryError(f"{label} is outside configured output_root")
        if label == "images_dir":
            images_dir = resolved
        else:
            json_dir = resolved

    images = discover_numbered_page_images(images_dir)
    if len(images) != expected_page_count:
        raise RetryError(
            "searchable-PDF page count no longer matches failed-attempt provenance"
        )

    for image in images:
        json_path = find_matching_json(image.name, json_dir)
        if json_path is None:
            raise RetryError(f"missing OCR JSON for {image.name}")
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise OcrSchemaError("OCR JSON root must be a mapping")
            parse_ocr_page(payload)
        except (OSError, json.JSONDecodeError, OcrSchemaError) as exc:
            raise RetryError(f"invalid OCR JSON {json_path}: {exc}") from exc

    return {
        "images_dir": images_dir,
        "json_dir": json_dir,
        "fontname": fontname,
        "y_offset": float(y_offset),
        "expected_page_count": expected_page_count,
    }


def _candidate_from_row(
    row: Mapping[str, object],
    config: ProjectConfig,
    *,
    overwrite: bool,
) -> RetryCandidate:
    source_size = int(row.get("source_size") or 0)
    source_mtime_ns = int(row.get("source_mtime_ns") or 0)
    profile_json = row.get("execution_profile_json")
    profile_value = profile_json if isinstance(profile_json, str) else None
    stage_value = row.get("stage")
    stage = str(stage_value) if stage_value is not None else ""

    source = Path(str(row.get("source_path") or ""))
    intended: Optional[Path] = None
    profile: Optional[Dict[str, object]] = None

    try:
        if stage not in SUPPORTED_RETRY_STAGES:
            raise RetryError(f"unsupported retry stage: {stage!r}")
        source = _source_path(row)
        _validate_source_scope(source, config)
        target = _target_path(row, config, overwrite=overwrite)
        intended = target
        profile = _decode_profile(profile_json)

        if stage == "ocr":
            _validate_ocr_profile(profile, source, target)
        elif stage == "render":
            _validate_render_profile(profile, target)
        else:
            _validate_searchable_profile(profile, config)
    except Exception as exc:
        if isinstance(exc, RetryError):
            reason = str(exc)
        else:
            reason = f"{type(exc).__name__}: {exc}"
        return RetryCandidate(
            source=source,
            stage=stage,
            intended_result=intended,
            execution_profile=profile,
            eligible=False,
            reason=reason,
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
            execution_profile_json=profile_value,
        )

    return RetryCandidate(
        source=source,
        stage=stage,
        intended_result=intended,
        execution_profile=profile,
        eligible=True,
        reason=None,
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
        execution_profile_json=profile_value,
    )


def plan_failed_retries(
    config: ProjectConfig,
    *,
    stage: Optional[str] = None,
    error_class: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    overwrite: bool = False,
) -> RetryPlan:
    """Build a read-only retry plan from failed manifest rows."""

    if stage is not None and stage not in SUPPORTED_RETRY_STAGES:
        raise ValueError(
            "retry stage must be one of ocr, render, searchable_pdf"
        )

    if not config.manifest_path.exists():
        return RetryPlan(
            manifest_path=config.manifest_path,
            total_matching=0,
            candidates=(),
            overwrite=overwrite,
        )

    page = query_manifest_job_page(
        config.manifest_path,
        status="failed",
        stage=stage,
        error_class=error_class,
        limit=limit,
        offset=offset,
    )
    candidates = tuple(
        _candidate_from_row(row, config, overwrite=overwrite)
        for row in page.jobs
    )
    return RetryPlan(
        manifest_path=config.manifest_path,
        total_matching=page.total_matching,
        candidates=candidates,
        overwrite=overwrite,
    )


def _current_record_matches(candidate: RetryCandidate, store: ManifestStore) -> JobRecord:
    record = store.get_job(candidate.source, candidate.stage)
    if record is None:
        raise RetryError("manifest row disappeared after retry planning")
    if record.status != "failed":
        raise RetryError(f"manifest row is no longer failed: {record.status}")
    if record.source_size != candidate.source_size or record.source_mtime_ns != candidate.source_mtime_ns:
        raise RetryError("manifest source fingerprint changed after retry planning")
    if record.intended_result_path != (
        os.fspath(candidate.intended_result) if candidate.intended_result is not None else None
    ):
        raise RetryError("manifest intended target changed after retry planning")
    if record.execution_profile_json != candidate.execution_profile_json:
        raise RetryError("manifest execution profile changed after retry planning")

    _reject_existing_symlink_components(candidate.source, label="source path")
    stat = candidate.source.stat()
    if stat.st_size != record.source_size or stat.st_mtime_ns != record.source_mtime_ns:
        raise RetryError("source changed after retry planning")
    return record


def execute_retry_plan(
    config: ProjectConfig,
    plan: RetryPlan,
    *,
    create_pipeline_fn: Optional[Callable[..., object]] = None,
    render_fn: Callable[..., RenderResult] = render_pdf,
    searchable_fn: Callable[..., SearchablePdfResult] = build_searchable_pdf,
) -> RetryExecutionResult:
    """Execute eligible retry candidates sequentially after revalidation."""

    items: List[RetryItemResult] = []
    for candidate in plan.candidates:
        if not candidate.eligible:
            items.append(
                RetryItemResult(
                    source=candidate.source,
                    stage=candidate.stage,
                    intended_result=candidate.intended_result,
                    status="ineligible",
                    error=candidate.reason,
                )
            )
            continue

        assert candidate.intended_result is not None
        assert candidate.execution_profile is not None
        target = candidate.intended_result
        profile = candidate.execution_profile

        try:
            # Revalidate the manifest row and filesystem just before acting.
            with ManifestStore(config.manifest_path) as store:
                _current_record_matches(candidate, store)
            _validate_source_scope(candidate.source, config)

            # A target or symlink component may have appeared after planning.
            _reject_symlink_components(target, config.output_root)
            if target.exists() and not plan.overwrite:
                raise RetryError(
                    f"retry target appeared after planning; pass --overwrite: {target}"
                )

            if candidate.stage == "ocr":
                kwargs = _validate_ocr_profile(profile, candidate.source, target)
                batch: OcrBatchResult = run_ocr_batch(
                    candidate.source,
                    target.parent,
                    manifest_path=config.manifest_path,
                    resume=False,
                    overwrite=plan.overwrite,
                    workers=1,
                    create_pipeline_fn=create_pipeline_fn,
                    **kwargs,
                )
                if batch.failed_count or batch.success_count != 1:
                    detail = batch.tasks[0].error if batch.tasks else "unexpected OCR retry result"
                    raise RetryError(detail or "OCR retry failed")

            elif candidate.stage == "render":
                kwargs = _validate_render_profile(profile, target)
                with ManifestStore(config.manifest_path) as store:
                    store.mark_started(
                        candidate.source,
                        "render",
                        worker=f"retry-pid-{os.getpid()}",
                        intended_result_path=target,
                        execution_profile=profile,
                    )
                try:
                    result = render_fn(
                        candidate.source,
                        target,
                        dpi=int(kwargs["dpi"]),
                        overwrite=plan.overwrite,
                    )
                except Exception as exc:
                    with ManifestStore(config.manifest_path) as store:
                        store.mark_failure(
                            candidate.source,
                            "render",
                            exc,
                            intended_result_path=target,
                            execution_profile=profile,
                        )
                    raise
                with ManifestStore(config.manifest_path) as store:
                    store.mark_success(
                        candidate.source,
                        "render",
                        result_path=result.output_dir,
                    )

            else:
                kwargs = _validate_searchable_profile(profile, config)
                with ManifestStore(config.manifest_path) as store:
                    store.mark_started(
                        candidate.source,
                        "searchable_pdf",
                        worker=f"retry-pid-{os.getpid()}",
                        intended_result_path=target,
                        execution_profile=profile,
                    )
                try:
                    result = searchable_fn(
                        kwargs["images_dir"],
                        kwargs["json_dir"],
                        target,
                        overwrite=plan.overwrite,
                        y_offset=float(kwargs["y_offset"]),
                        fontname=str(kwargs["fontname"]),
                    )
                    if result.page_count != int(kwargs["expected_page_count"]):
                        raise RetryError(
                            "searchable-PDF retry produced an unexpected page count"
                        )
                except Exception as exc:
                    with ManifestStore(config.manifest_path) as store:
                        store.mark_failure(
                            candidate.source,
                            "searchable_pdf",
                            exc,
                            intended_result_path=target,
                            execution_profile=profile,
                        )
                    raise
                with ManifestStore(config.manifest_path) as store:
                    store.mark_success(
                        candidate.source,
                        "searchable_pdf",
                        result_path=result.output_pdf,
                    )

            items.append(
                RetryItemResult(
                    source=candidate.source,
                    stage=candidate.stage,
                    intended_result=target,
                    status="success",
                )
            )
        except Exception as exc:
            items.append(
                RetryItemResult(
                    source=candidate.source,
                    stage=candidate.stage,
                    intended_result=target,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return RetryExecutionResult(items=tuple(items))
