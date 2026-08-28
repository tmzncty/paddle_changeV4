"""Adapters for historical and current PaddleX/PaddleOCR JSON result shapes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Optional, Sequence, Tuple


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
    polygon_field: str


def _as_sequence(value: object) -> Optional[Sequence[object]]:
    """Return a Python sequence for list/tuple or NumPy-like array values.

    PaddleX's in-memory OCR Result uses NumPy arrays for ``dt_polys`` and
    ``rec_polys``. NumPy ndarray deliberately does not register as a
    ``collections.abc.Sequence``, so accepting only ``Sequence`` rejects the
    documented PaddleX result shape. Duck-typing ``tolist()`` keeps this core
    module dependency-free while normalizing array-like values to ordinary
    Python containers before validation.
    """

    if isinstance(value, (str, bytes, bytearray)):
        return None
    if isinstance(value, Sequence):
        return value

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, Sequence) and not isinstance(
            converted, (str, bytes, bytearray)
        ):
            return converted
    return None


def _normalize_polygon(
    value: object,
    *,
    field_name: str,
    index: int,
) -> Tuple[Tuple[float, float], ...]:
    polygon = _as_sequence(value)
    if polygon is None or len(polygon) < 4:
        raise OcrSchemaError(f"{field_name}[{index}] must contain at least four points")

    points = []
    for point_index, point_value in enumerate(polygon):
        point = _as_sequence(point_value)
        if point is None or len(point) < 2:
            raise OcrSchemaError(
                f"{field_name}[{index}][{point_index}] must contain x/y coordinates"
            )
        x, y = point[0], point[1]
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, Real)
            or not isinstance(y, Real)
        ):
            raise OcrSchemaError(
                f"{field_name}[{index}][{point_index}] coordinates must be numeric"
            )
        points.append((float(x), float(y)))
    return tuple(points)


def parse_ocr_page(data: Mapping[str, object]) -> OcrPage:
    """Normalize modern and historical OCR polygon/text variants.

    Current PaddleOCR 3.x results expose ``rec_polys`` and ``rec_texts`` after
    recognition-confidence filtering, so those fields are preferred as a pair.
    Historical repository outputs used ``dt_polys`` together with either
    ``rec_texts`` (newer legacy) or ``rec_text`` (older legacy).

    Preferring ``rec_polys`` prevents a current 3.x detection box that was later
    filtered out from being paired with the wrong recognized text.
    """

    if not isinstance(data, Mapping):
        raise OcrSchemaError("OCR JSON root must be an object")

    if "rec_texts" in data:
        text_field = "rec_texts"
        texts_value = data.get("rec_texts")
    elif "rec_text" in data:
        text_field = "rec_text"
        texts_value = data.get("rec_text")
    else:
        raise OcrSchemaError("OCR JSON is missing rec_texts/rec_text")

    if text_field == "rec_texts" and "rec_polys" in data:
        polygon_field = "rec_polys"
        polygons_value = data.get("rec_polys")
    else:
        polygon_field = "dt_polys"
        polygons_value = data.get("dt_polys")

    polygons = _as_sequence(polygons_value)
    texts = _as_sequence(texts_value)
    if polygons is None:
        raise OcrSchemaError(
            f"OCR JSON is missing {polygon_field} or it is not an array/list"
        )
    if texts is None:
        raise OcrSchemaError(f"{text_field} must be an array/list")
    if len(polygons) != len(texts):
        raise OcrSchemaError(
            f"polygon/text count mismatch using {polygon_field}/{text_field}: "
            f"{len(polygons)} polygons vs {len(texts)} texts"
        )

    lines = []
    for index, (polygon, text) in enumerate(zip(polygons, texts)):
        if not isinstance(text, str):
            raise OcrSchemaError(f"{text_field}[{index}] must be a string")
        lines.append(
            OcrLine(
                polygon=_normalize_polygon(
                    polygon,
                    field_name=polygon_field,
                    index=index,
                ),
                text=text,
            )
        )

    return OcrPage(
        lines=tuple(lines),
        text_field=text_field,
        polygon_field=polygon_field,
    )
