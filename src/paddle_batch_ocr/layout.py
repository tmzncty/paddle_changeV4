"""Pure geometry helpers that freeze historical searchable-PDF behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .ocr_schema import OcrLine


@dataclass(frozen=True)
class TextRect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def order_two_columns(lines: Iterable[OcrLine], *, page_width: float) -> Tuple[OcrLine, ...]:
    """Preserve the version-7 two-column heuristic.

    Empty text is dropped. A line is classified as left-column when the minimum
    polygon x is strictly less than half the page width. Left-column lines are
    emitted before right-column lines; each column is ordered only by minimum y.
    """

    center_x = page_width / 2.0
    decorated = []
    for line in lines:
        if not line.text.strip():
            continue
        min_x = min(point[0] for point in line.polygon)
        min_y = min(point[1] for point in line.polygon)
        is_left = min_x < center_x
        decorated.append((not is_left, min_y, line))

    decorated.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in decorated)


def legacy_text_rect(
    line: OcrLine,
    *,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    y_offset: float = 0.0,
) -> TextRect:
    """Reproduce the rectangle construction used by legacy version 7.

    This intentionally uses polygon points 0 and 2 instead of calculating a
    geometric bounding box. Keeping it explicit lets future code compare a
    corrected bbox implementation against historical output.
    """

    if len(line.polygon) < 3:
        raise ValueError("legacy text rectangle requires polygon points 0 and 2")

    p0 = line.polygon[0]
    p2 = line.polygon[2]
    return TextRect(
        x0=int(p0[0] * x_scale),
        y0=int(p0[1] * y_scale) + y_offset,
        x1=int(p2[0] * x_scale),
        y1=int(p2[1] * y_scale) + y_offset,
    )
