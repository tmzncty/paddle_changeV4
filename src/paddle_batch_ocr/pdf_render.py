"""Transactional PDF-to-image rendering with lazy PyMuPDF loading."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


class PdfRenderError(RuntimeError):
    """Raised when PDF rendering cannot complete safely."""


class PdfDependencyError(PdfRenderError):
    """Raised when the optional PDF dependencies are not installed."""


@dataclass(frozen=True)
class RenderResult:
    source: Path
    output_dir: Path
    page_paths: Tuple[Path, ...]
    page_count: int
    dpi: int


def _require_fitz():
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise PdfDependencyError(
                "PDF rendering requires PyMuPDF; install paddle-batch-ocr[pdf]"
            ) from exc
    return fitz


def _validate_dpi(dpi: int) -> int:
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 36 <= dpi <= 1200:
        raise ValueError("dpi must be an integer between 36 and 1200")
    return dpi


def _replace_directory(staging: Path, target: Path, *, overwrite: bool) -> None:
    """Publish a fully-rendered staging directory into its final location."""

    if target.exists():
        if not overwrite:
            raise FileExistsError(f"render output directory already exists: {target}")
        if not target.is_dir() or target.is_symlink():
            raise PdfRenderError(f"refusing to replace non-directory output: {target}")

        backup = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.backup-", dir=str(target.parent))
        )
        backup.rmdir()
        os.replace(str(target), str(backup))
        try:
            os.replace(str(staging), str(target))
        except BaseException:
            os.replace(str(backup), str(target))
            raise
        else:
            # Publication succeeded. A stale backup is preferable to reporting a
            # false execution failure after the new output is already live.
            shutil.rmtree(backup, ignore_errors=True)
    else:
        os.replace(str(staging), str(target))


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    overwrite: bool = False,
) -> RenderResult:
    """Render every PDF page into a newly published PNG directory.

    Pages are rendered into a hidden sibling staging directory first. The final
    output directory is published only after every page succeeds, so callers do
    not mistake a half-rendered directory for a completed stage.
    """

    fitz = _require_fitz()
    dpi = _validate_dpi(dpi)
    source = Path(pdf_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise PdfRenderError(f"PDF source is not a file: {source}")

    target = Path(output_dir).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"render output directory already exists: {target}")
    if target.is_symlink():
        raise PdfRenderError(f"refusing symlinked render output directory: {target}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.render-", dir=str(target.parent))
    )

    page_count = 0
    try:
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        with fitz.open(str(source)) as document:
            page_count = document.page_count
            if page_count < 1:
                raise PdfRenderError(f"PDF contains no pages: {source}")

            for page_index in range(page_count):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                page_path = staging / f"page_{page_index + 1:05d}.png"
                pixmap.save(str(page_path))

        published_paths = tuple(target / f"page_{i + 1:05d}.png" for i in range(page_count))
        _replace_directory(staging, target, overwrite=overwrite)
        staging = None  # type: ignore[assignment]
        return RenderResult(
            source=source,
            output_dir=target,
            page_paths=published_paths,
            page_count=page_count,
            dpi=dpi,
        )
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
