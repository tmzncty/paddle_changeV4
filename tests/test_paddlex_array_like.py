import unittest

from paddle_batch_ocr.ocr_schema import parse_ocr_page
from paddle_batch_ocr.paddlex_adapter import normalize_ocr_result


class ArrayLike:
    """Dependency-free stand-in for NumPy ndarray/scalar .tolist() behavior."""

    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class PaddleXArrayLikeTests(unittest.TestCase):
    def test_schema_accepts_numpy_like_rec_polys(self):
        payload = {
            "rec_texts": ["中文"],
            "rec_polys": [
                ArrayLike([[1, 2], [11, 2], [11, 8], [1, 8]]),
            ],
        }

        page = parse_ocr_page(payload)

        self.assertEqual(page.polygon_field, "rec_polys")
        self.assertEqual(page.lines[0].polygon[0], (1.0, 2.0))
        self.assertEqual(page.lines[0].polygon[2], (11.0, 8.0))
        self.assertEqual(page.lines[0].text, "中文")

    def test_schema_accepts_numpy_like_top_level_polygon_array(self):
        payload = {
            "rec_texts": ["one", "two"],
            "rec_polys": ArrayLike(
                [
                    [[0, 0], [4, 0], [4, 3], [0, 3]],
                    [[5, 5], [9, 5], [9, 8], [5, 8]],
                ]
            ),
        }

        page = parse_ocr_page(payload)

        self.assertEqual(len(page.lines), 2)
        self.assertEqual(page.lines[1].text, "two")

    def test_normalization_converts_nested_array_like_values_to_json_types(self):
        payload = {
            "input_path": "page.png",
            "dt_polys": ArrayLike([[[1, 2], [11, 2], [11, 8], [1, 8]]]),
            "rec_polys": [
                ArrayLike([[1, 2], [11, 2], [11, 8], [1, 8]]),
            ],
            "rec_texts": ["中文"],
            "rec_scores": ArrayLike([0.95]),
            "rec_boxes": ArrayLike([[1, 2, 11, 8]]),
            "page_index": None,
        }

        normalized = normalize_ocr_result(payload)

        self.assertEqual(
            normalized["rec_polys"],
            [[[1, 2], [11, 2], [11, 8], [1, 8]]],
        )
        self.assertEqual(normalized["rec_scores"], [0.95])
        self.assertEqual(normalized["rec_boxes"], [[1, 2, 11, 8]])
        self.assertEqual(normalized["_paddle_batch_ocr"]["polygon_field"], "rec_polys")


if __name__ == "__main__":
    unittest.main()
