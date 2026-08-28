"""Searchable-PDF reconstruction using frozen legacy-v7 compatibility rules."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from .io_utils import atomic_publish_file
from .layout import legacy_text_rect, order_two_columns
from .naming import find_matching_json
from .ocr_schema import OcrSchemaError, parse_ocr_page
from .pdf_render import PdfDependencyError


_PAGE_RE = re.compile(r"page_(\d+)", re.IGNORECASE)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SearchablePdfError(RuntimeError):
    """Raised when searchable-PDF reconstruction cannot complete safely."""


@dataclass(frozen=True)
class SearchablePdfResult:
    output_pdf: Path
    page_count: int
    text_line_count: int
    image_paths: Tuple[Path, ...]
    json_paths: Tuple[Path, ...]


def _require_pdf_dependencies():
    try:
        import pymupdf as fitz  # type: ignore
    except ImportError:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise PdfDependencyError(
                "searchable-PDF reconstruction requires PyMuPDF and Pillow; "
                "install paddle-batch-ocr[pdf]"
            ) from exc

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise PdfDependencyError(
            "searchable-PDF reconstruction requires PyMuPDF and Pillow; "
            "install paddle-batch-ocr[pdf]"
        ) from exc

    return fitz, Image


def validate_searchable_pdf(
    pdf_path: Path,
    expected_page_count: int,
) -> Path:
    """Validate an existing searchable-PDF stage before resume adoption."""

    if (
        isinstance(expected_page_count, bool)
        or not isinstance(expected_page_count, int)
        or expected_page_count < 1
    ):
        raise ValueError("expected_page_count must be an integer >= 1")

    fitz, _ = _require_pdf_dependencies()

    raw = Path(pdf_path).expanduser()
    if raw.is_symlink():
        raise SearchablePdfError(
            f"refusing symlinked searchable PDF: {raw}"
        )

    target = raw.resolve(strict=True)
    if not target.is_file() or target.stat().st_size < 1:
        raise SearchablePdfError(
            f"searchable PDF is missing or empty: {target}"
        )

    try:
        with fitz.open(str(target)) as document:
            actual_page_count = document.page_count
    except Exception as exc:
        raise SearchablePdfError(
            f"cannot open searchable PDF {target}: {exc}"
        ) from exc

    if actual_page_count != expected_page_count:
        raise SearchablePdfError(
            "searchable PDF page count mismatch: "
            f"{actual_page_count} != {expected_page_count}"
        )

    return target


def discover_numbered_page_images(
    image_dir: Path,
) -> Tuple[Path, ...]:
    """Discover a complete ``page_00001..N`` image sequence."""

    root = Path(image_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SearchablePdfError(
            f"image directory does not exist: {root}"
        )

    by_page: Dict[int, Path] = {}
    for path in root.iterdir():
        if (
            not path.is_file()
            or path.suffix.lower() not in _IMAGE_SUFFIXES
        ):
            continue

        match = _PAGE_RE.search(path.name)
        if not match:
            continue

        page_number = int(match.group(1))
        if page_number < 1:
            raise SearchablePdfError(
                f"page numbers must start at 1: {path.name}"
            )

        previous = by_page.get(page_number)
        if previous is not None:
            raise SearchablePdfError(
                f"multiple images claim page {page_number}: "
                f"{previous.name}, {path.name}"
            )

        by_page[page_number] = path

    if not by_page:
        raise SearchablePdfError(
            f"no page_<number> images found in {root}"
        )

    last_page = max(by_page)
    missing = [
        number
        for number in range(1, last_page + 1)
        if number not in by_page
    ]
    if missing:
        preview = ", ".join(str(number) for number in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise SearchablePdfError(
            "page image sequence has gaps; missing page(s): "
            f"{preview}{suffix}"
        )

    return tuple(
        by_page[number]
        for number in range(1, last_page + 1)
    )


def _load_ocr_json(path: Path) -> Mapping[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchablePdfError(
            f"cannot read OCR JSON {path}: {exc}"
        ) from exc

    if not isinstance(data, Mapping):
        raise SearchablePdfError(
            f"OCR JSON root must be an object: {path}"
        )
    return data


def build_searchable_pdf(
    image_dir: Path,
    json_dir: Path,
    output_pdf: Path,
    *,
    overwrite: bool = False,
    y_offset: float = 0.0,
    fontname: str = "china-s",
) -> SearchablePdfResult:
    """Build a searchable PDF from numbered page images and OCR JSON.

    Ordering and text geometry intentionally preserve the frozen version-7
    behavior. Missing page JSON remains a hard error.
    """

    fitz, Image = _require_pdf_dependencies()
    images = discover_numbered_page_images(image_dir)

    json_root = Path(json_dir).expanduser().resolve(strict=True)
    if not json_root.is_dir():
        raise SearchablePdfError(
            f"JSON directory does not exist: {json_root}"
        )

    raw_target = Path(output_pdf).expanduser()
    if raw_target.is_symlink():
        raise SearchablePdfError(
            f"refusing symlinked output PDF: {raw_target}"
        )

    target = raw_target.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"searchable PDF already exists: {target}"
        )

    matched_json: List[Path] = []
    for image_path in images:
        json_path = find_matching_json(
            image_path.name,
            json_root,
        )
        if json_path is None:
            raise SearchablePdfError(
                f"no OCR JSON matches image {image_path.name}"
            )
        matched_json.append(json_path)

    document = fitz.open()
    line_count = 0
    temp_path = None

    try:
        for image_path, json_path in zip(images, matched_json):
            try:
                ocr_page = parse_ocr_page(
                    _load_ocr_json(json_path)
                )
            except OcrSchemaError as exc:
                raise SearchablePdfError(
                    f"invalid OCR JSON {json_path}: {exc}"
                ) from exc

            with Image.open(str(image_path)) as image:
                width, height = image.size

            if width < 1 or height < 1:
                raise SearchablePdfError(
                    f"invalid image dimensions for {image_path}"
                )

            page = document.new_page(
                width=width,
                height=height,
            )
            page.insert_image(
                page.rect,
                filename=str(image_path),
            )

            for line in order_two_columns(
                ocr_page.lines,
                page_width=width,
            ):
                if not line.text.strip():
                    continue

                legacy_rect = legacy_text_rect(
                    line,
                    y_offset=y_offset,
                )
                if (
                    legacy_rect.width <= 0
                    or legacy_rect.height <= 0
                ):
                    raise SearchablePdfError(
                        "non-positive legacy text rectangle in "
                        f"{json_path}: {legacy_rect}"
                    )

                rect = fitz.Rect(
                    legacy_rect.x0,
                    legacy_rect.y0,
                    legacy_rect.x1,
                    legacy_rect.y1,
                )
                fontsize = rect.height * 0.9
                while fontsize > 1:
                    text_width = fitz.get_text_length(
                        line.text,
                        fontname=fontname,
                        fontsize=fontsize,
                    )
                    if text_width <= rect.width:
                        break
                    fontsize -= 1

                fontsize = max(
                    1.0,
                    min(float(fontsize), 100.0),
                )
                insertion_point = fitz.Point(
                    rect.x0,
                    rect.y0 + (fontsize * 0.85),
                )
                page.insert_text(
                    insertion_point,
                    line.text,
                    fontname=fontname,
                    fontsize=fontsize,
                    color=(0, 0, 0),
                    fill=(1, 1, 1),
                    render_mode=3,
                )
                line_count += 1

        if document.page_count != len(images):
            raise SearchablePdfError(
                "internal page-count mismatch: "
                f"{document.page_count} != {len(images)}"
            )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp.pdf",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

        document.save(
            str(temp_path),
            garbage=4,
            deflate=True,
        )
        document.close()
        document = None

        atomic_publish_file(
            temp_path,
            target,
            overwrite=overwrite,
            fsync=True,
        )
        temp_path = None

        return SearchablePdfResult(
            output_pdf=target,
            page_count=len(images),
            text_line_count=line_count,
            image_paths=images,
            json_paths=tuple(matched_json),
        )
    finally:
        if document is not None:
            document.close()
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
