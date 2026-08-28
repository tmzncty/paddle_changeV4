"""Adapters for OCR JSON variants produced by historical PaddleX runs."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence, Tuple


class OcrSchemaError(ValueError):
    """Raised when OCR JSON cannot be normalized safely."""


@dataclass(frozen=True)
class OcrLine:
    polygon: Tuple[Tuple[float, float], ...]
    text: str


@dataclass(frozen=True)
class OcrPage:
    lines: Tuple[OcrLine, ...]
    text_field: str


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _normalize_polygon(value: object, *, index: int) -> Tuple[Tuple[float, float], ...]:
    if not _is_sequence(value) or len(value) < 4:
        raise OcrSchemaError(f"dt_polys[{index}] must contain at least four points")

    points = []
    for point_index, point in enumerate(value):
        if not _is_sequence(point) or len(point) < 2:
            raise OcrSchemaError(
                f"dt_polys[{index}][{point_index}] must contain x/y coordinates"
            )
        x, y = point[0], point[1]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, Real) or not isinstance(y, Real):
            raise OcrSchemaError(
                f"dt_polys[{index}][{point_index}] coordinates must be numeric"
            )
        points.append((float(x), float(y)))
    return tuple(points)


def parse_ocr_page(data: Mapping[str, object]) -> OcrPage:
    """Normalize historical ``dt_polys`` + text-field variants.

    ``pdf_creator_with_text_layer6.py`` expects ``rec_text`` while version 7
    expects ``rec_texts``.  The adapter deliberately supports both spellings
    and exposes which field was consumed for provenance/debugging.
    """

    if not isinstance(data, Mapping):
        raise OcrSchemaError("OCR JSON root must be an object")

    polygons = data.get("dt_polys")
    if "rec_texts" in data:
        text_field = "rec_texts"
        texts = data.get("rec_texts")
    elif "rec_text" in data:
        text_field = "rec_text"
        texts = data.get("rec_text")
    else:
        raise OcrSchemaError("OCR JSON is missing rec_texts/rec_text")

    if not _is_sequence(polygons):
        raise OcrSchemaError("OCR JSON is missing dt_polys or it is not a list")
    if not _is_sequence(texts):
        raise OcrSchemaError(f"{text_field} must be a list")
    if len(polygons) != len(texts):
        raise OcrSchemaError(
            f"polygon/text count mismatch: {len(polygons)} polygons vs {len(texts)} texts"
        )

    lines = []
    for index, (polygon, text) in enumerate(zip(polygons, texts)):
        if not isinstance(text, str):
            raise OcrSchemaError(f"{text_field}[{index}] must be a string")
        lines.append(
            OcrLine(
                polygon=_normalize_polygon(polygon, index=index),
                text=text,
            )
        )

    return OcrPage(lines=tuple(lines), text_field=text_field)
