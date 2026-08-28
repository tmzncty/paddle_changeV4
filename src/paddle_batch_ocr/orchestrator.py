"""Project-level orchestration for PDF and image input sources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import InputSource, ProjectConfig
from .manifest import ManifestStore
from .ocr_runner import OcrBatchResult, run_ocr_batch
from .pdf_render import RenderResult, render_pdf, validate_rendered_pages
from .searchable_pdf import (
    SearchablePdfResult,
    build_searchable_pdf,
    validate_searchable_pdf,
)


class ProjectRunError(RuntimeError):
    """Raised when project-level orchestration cannot proceed safely."""


@dataclass(frozen=True)
class ProjectItemResult:
    source: Path
    kind: str
    status: str
    pages_dir: Optional[Path] = None
    ocr_dir: Optional[Path] = None
    searchable_pdf: Optional[Path] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ProjectRunResult:
    items: Tuple[ProjectItemResult, ...]

    @property
    def success_count(self) -> int:
        return sum(item.status == "success" for item in self.items)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.items)


def _discover_source_files(source: InputSource) -> Tuple[Path, ...]:
    """Discover configured PDF/image files deterministically."""

    root = source.path
    suffixes = (
        {".pdf"}
        if source.kind == "pdf"
        else {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if root.is_file():
        if root.suffix.lower() not in suffixes:
            raise ProjectRunError(
                f"configured {source.kind} input has unsupported extension: {root}"
            )
        return (root.resolve(strict=True),)

    if not root.is_dir():
        raise ProjectRunError(f"configured input does not exist: {root}")

    found: List[Path] = []
    for walk_root, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        base = Path(walk_root)

        for filename in files:
            candidate = base / filename
            if candidate.suffix.lower() not in suffixes:
                continue
            if candidate.is_symlink():
                raise ProjectRunError(f"refusing symlinked input file: {candidate}")
            found.append(candidate.resolve(strict=True))

    return tuple(found)


def _pdf_artifact_root(
    config: ProjectConfig,
    source_index: int,
    source_root: Path,
    pdf_path: Path,
) -> Path:
    relative = (
        Path(pdf_path.name)
        if source_root.is_file()
        else pdf_path.relative_to(source_root)
    )
    return (
        config.output_root
        / f"source-{source_index:03d}"
        / "pdf"
        / relative.with_suffix("")
    ).resolve(strict=False)


def _run_render_stage(
    config: ProjectConfig,
    pdf_path: Path,
    pages_dir: Path,
    *,
    dpi: int,
    render_fn: Callable[..., RenderResult],
    validate_fn: Callable[[Path, Path], Tuple[Path, ...]],
) -> Tuple[str, Tuple[Path, ...]]:
    """Run or safely adopt the render stage for one PDF."""

    with ManifestStore(config.manifest_path) as store:
        previous = store.get_job(pdf_path, "render")
        needs_run = store.needs_run(pdf_path, "render")

        if pages_dir.exists():
            can_adopt = (
                config.resume
                and (
                    previous is None
                    or not needs_run
                )
            )
            if can_adopt:
                try:
                    pages = validate_fn(pdf_path, pages_dir)
                except Exception:
                    if not config.overwrite:
                        raise
                else:
                    store.mark_success(
                        pdf_path,
                        "render",
                        result_path=pages_dir,
                    )
                    return "skipped", pages

            if not config.overwrite:
                raise ProjectRunError(
                    "render output already exists or is stale; "
                    f"enable overwrite: {pages_dir}"
                )

        store.mark_started(
            pdf_path,
            "render",
            worker=f"pid-{os.getpid()}",
        )
        try:
            result = render_fn(
                pdf_path,
                pages_dir,
                dpi=dpi,
                overwrite=config.overwrite,
            )
        except Exception as exc:
            store.mark_failure(pdf_path, "render", exc)
            raise

        store.mark_success(
            pdf_path,
            "render",
            result_path=result.output_dir,
        )
        return "success", result.page_paths


def _run_searchable_stage(
    config: ProjectConfig,
    pdf_path: Path,
    pages_dir: Path,
    ocr_dir: Path,
    output_pdf: Path,
    *,
    expected_page_count: int,
    force_rebuild: bool,
    searchable_fn: Callable[..., SearchablePdfResult],
    validate_fn: Callable[[Path, int], Path],
) -> str:
    """Run/adopt searchable-PDF output while respecting upstream OCR changes."""

    with ManifestStore(config.manifest_path) as store:
        previous = store.get_job(pdf_path, "searchable_pdf")
        needs_run = store.needs_run(pdf_path, "searchable_pdf")

        if output_pdf.exists():
            if force_rebuild:
                if not config.overwrite:
                    raise ProjectRunError(
                        "searchable PDF is stale because OCR produced new results; "
                        f"enable overwrite to rebuild it: {output_pdf}"
                    )
            else:
                can_adopt = (
                    config.resume
                    and (
                        previous is None
                        or not needs_run
                    )
                )
                if can_adopt:
                    try:
                        validate_fn(output_pdf, expected_page_count)
                    except Exception:
                        if not config.overwrite:
                            raise
                    else:
                        store.mark_success(
                            pdf_path,
                            "searchable_pdf",
                            result_path=output_pdf,
                        )
                        return "skipped"

                if not config.overwrite:
                    raise ProjectRunError(
                        "searchable PDF already exists or is stale; "
                        f"enable overwrite: {output_pdf}"
                    )

        store.mark_started(
            pdf_path,
            "searchable_pdf",
            worker=f"pid-{os.getpid()}",
        )
        try:
            result = searchable_fn(
                pages_dir,
                ocr_dir,
                output_pdf,
                overwrite=config.overwrite,
            )
        except Exception as exc:
            store.mark_failure(pdf_path, "searchable_pdf", exc)
            raise

        store.mark_success(
            pdf_path,
            "searchable_pdf",
            result_path=result.output_pdf,
        )
        return "success"


def _run_pdf_document(
    config: ProjectConfig,
    source_index: int,
    source_root: Path,
    pdf_path: Path,
    *,
    dpi: int,
    render_fn: Callable[..., RenderResult],
    validate_render_fn: Callable[[Path, Path], Tuple[Path, ...]],
    ocr_fn: Callable[..., OcrBatchResult],
    searchable_fn: Callable[..., SearchablePdfResult],
    validate_searchable_fn: Callable[[Path, int], Path],
) -> ProjectItemResult:
    """Execute render -> OCR -> searchable-PDF for one document."""

    artifact_root = _pdf_artifact_root(
        config,
        source_index,
        source_root,
        pdf_path,
    )
    pages_dir = artifact_root / "pages"
    ocr_dir = artifact_root / "ocr"
    searchable_path = artifact_root / "searchable.pdf"
    pipeline_ref = config.paddle_config if config.paddle_config is not None else "OCR"

    try:
        _, page_paths = _run_render_stage(
            config,
            pdf_path,
            pages_dir,
            dpi=dpi,
            render_fn=render_fn,
            validate_fn=validate_render_fn,
        )

        ocr_result = ocr_fn(
            pages_dir,
            ocr_dir,
            pipeline_ref=pipeline_ref,
            device=config.runtime.device,
            manifest_path=config.manifest_path,
            resume=config.resume,
            overwrite=config.overwrite,
            workers=config.runtime.ocr_workers,
        )
        if ocr_result.failed_count:
            raise ProjectRunError(
                f"OCR failed for {ocr_result.failed_count} page(s) in {pdf_path}"
            )

        # A source-PDF fingerprint cannot prove that an existing text layer
        # still corresponds to newly produced OCR results. If any page actually
        # ran this time, require a downstream rebuild.
        ocr_changed = ocr_result.success_count > 0

        _run_searchable_stage(
            config,
            pdf_path,
            pages_dir,
            ocr_dir,
            searchable_path,
            expected_page_count=len(page_paths),
            force_rebuild=ocr_changed,
            searchable_fn=searchable_fn,
            validate_fn=validate_searchable_fn,
        )

        return ProjectItemResult(
            source=pdf_path,
            kind="pdf",
            status="success",
            pages_dir=pages_dir,
            ocr_dir=ocr_dir,
            searchable_pdf=searchable_path,
        )
    except Exception as exc:
        return ProjectItemResult(
            source=pdf_path,
            kind="pdf",
            status="failed",
            pages_dir=pages_dir,
            ocr_dir=ocr_dir,
            searchable_pdf=searchable_path,
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_image_source(
    config: ProjectConfig,
    source_index: int,
    source: InputSource,
    *,
    ocr_fn: Callable[..., OcrBatchResult],
) -> ProjectItemResult:
    output_dir = (
        config.output_root
        / f"source-{source_index:03d}"
        / "image"
        / "ocr"
    ).resolve(strict=False)
    pipeline_ref = config.paddle_config if config.paddle_config is not None else "OCR"

    try:
        result = ocr_fn(
            source.path,
            output_dir,
            pipeline_ref=pipeline_ref,
            device=config.runtime.device,
            manifest_path=config.manifest_path,
            resume=config.resume,
            overwrite=config.overwrite,
            workers=config.runtime.ocr_workers,
        )
        if result.failed_count:
            raise ProjectRunError(
                f"OCR failed for {result.failed_count} image(s) under {source.path}"
            )

        return ProjectItemResult(
            source=source.path,
            kind="image",
            status="success",
            ocr_dir=output_dir,
        )
    except Exception as exc:
        return ProjectItemResult(
            source=source.path,
            kind="image",
            status="failed",
            ocr_dir=output_dir,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_project(
    config: ProjectConfig,
    *,
    dpi: int = 144,
    render_fn: Callable[..., RenderResult] = render_pdf,
    validate_render_fn: Callable[[Path, Path], Tuple[Path, ...]] = validate_rendered_pages,
    ocr_fn: Callable[..., OcrBatchResult] = run_ocr_batch,
    searchable_fn: Callable[..., SearchablePdfResult] = build_searchable_pdf,
    validate_searchable_fn: Callable[[Path, int], Path] = validate_searchable_pdf,
) -> ProjectRunResult:
    """Execute configured sources deterministically and isolate failures per item."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    items: List[ProjectItemResult] = []
    for source_index, source in enumerate(config.input_sources, start=1):
        if source.kind == "image":
            items.append(
                _run_image_source(
                    config,
                    source_index,
                    source,
                    ocr_fn=ocr_fn,
                )
            )
            continue

        try:
            pdfs = _discover_source_files(source)
        except Exception as exc:
            items.append(
                ProjectItemResult(
                    source=source.path,
                    kind="pdf",
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if not pdfs:
            items.append(
                ProjectItemResult(
                    source=source.path,
                    kind="pdf",
                    status="failed",
                    error="ProjectRunError: no PDF files found",
                )
            )
            continue

        for pdf_path in pdfs:
            items.append(
                _run_pdf_document(
                    config,
                    source_index,
                    source.path,
                    pdf_path,
                    dpi=dpi,
                    render_fn=render_fn,
                    validate_render_fn=validate_render_fn,
                    ocr_fn=ocr_fn,
                    searchable_fn=searchable_fn,
                    validate_searchable_fn=validate_searchable_fn,
                )
            )

    return ProjectRunResult(items=tuple(items))
