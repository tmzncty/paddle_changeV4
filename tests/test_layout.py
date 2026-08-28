import unittest

from paddle_batch_ocr.layout import legacy_text_rect, order_two_columns
from paddle_batch_ocr.ocr_schema import OcrLine


class LayoutTests(unittest.TestCase):
    def _line(self, x0, y0, x1, y1, text):
        return OcrLine(
            polygon=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
            text=text,
        )

    def test_two_column_order_matches_legacy_heuristic(self):
        lines = [
            self._line(60, 10, 90, 20, "right-top"),
            self._line(10, 30, 40, 40, "left-bottom"),
            self._line(10, 5, 40, 15, "left-top"),
            self._line(60, 40, 90, 50, "right-bottom"),
        ]

        ordered = order_two_columns(lines, page_width=100)
        self.assertEqual(
            [line.text for line in ordered],
            ["left-top", "left-bottom", "right-top", "right-bottom"],
        )

    def test_two_column_order_drops_blank_text(self):
        ordered = order_two_columns(
            [
                self._line(10, 5, 40, 15, "   "),
                self._line(10, 20, 40, 30, "visible"),
            ],
            page_width=100,
        )
        self.assertEqual([line.text for line in ordered], ["visible"])

    def test_legacy_text_rect_uses_points_zero_and_two(self):
        line = OcrLine(
            polygon=((10.9, 20.9), (80, 18), (70.2, 50.8), (5, 55)),
            text="x",
        )
        rect = legacy_text_rect(line, y_offset=3)
        self.assertEqual((rect.x0, rect.y0, rect.x1, rect.y1), (10, 23, 70, 53))


if __name__ == "__main__":
    unittest.main()
