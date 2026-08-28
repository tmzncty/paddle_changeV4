"""Deterministic image-to-OCR-JSON naming compatibility helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple


_PAGE_RE = re.compile(r"page_(\d+)", re.IGNORECASE)


def candidate_json_names(image_name: str) -> Tuple[str, ...]:
    """Return JSON filename candidates in historical precedence order.

    The first four names preserve the lookup order used by
    ``pdf_creator_with_text_layer7.py`` for page-numbered images. Generic
    basename fallbacks are appended for non-standard historical filenames.
    """

    basename = Path(image_name).stem
    names = []
    match = _PAGE_RE.search(image_name)
    if match:
        page_num = int(match.group(1))
        names.extend(
            [
                f"page_{page_num:05}.json",
                f"page_{page_num:04}_result.json",
                f"page_{page_num:04}.json",
                f"page_{page_num:03}.json",
            ]
        )

    names.extend(
        [
            f"{basename}_result.json",
            f"{basename}.json",
            f"{basename}_ocr.json",
        ]
    )

    return tuple(dict.fromkeys(names))


def find_matching_json(image_name: str, json_dir: Path) -> Optional[Path]:
    root = Path(json_dir)
    for candidate in candidate_json_names(image_name):
        path = root / candidate
        if path.is_file():
            return path
    return None
