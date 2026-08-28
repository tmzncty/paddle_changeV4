import unittest

from paddle_batch_ocr.ocr_schema import OcrSchemaError, parse_ocr_page


class OcrSchemaTests(unittest.TestCase):
    def test_accepts_rec_texts_variant(self):
        page = parse_ocr_page(
            {
                "dt_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                "rec_texts": ["测试"],
            }
        )
        self.assertEqual(page.text_field, "rec_texts")
        self.assertEqual(page.lines[0].text, "测试")
        self.assertEqual(page.lines[0].polygon[2], (10.0, 5.0))

    def test_accepts_legacy_rec_text_variant(self):
        page = parse_ocr_page(
            {
                "dt_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
                "rec_text": ["legacy"],
            }
        )
        self.assertEqual(page.text_field, "rec_text")
        self.assertEqual(page.lines[0].text, "legacy")

    def test_prefers_rec_texts_when_both_exist(self):
        page = parse_ocr_page(
            {
                "dt_polys": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                "rec_texts": ["new"],
                "rec_text": ["old"],
            }
        )
        self.assertEqual(page.text_field, "rec_texts")
        self.assertEqual(page.lines[0].text, "new")

    def test_rejects_polygon_text_count_mismatch(self):
        with self.assertRaises(OcrSchemaError):
            parse_ocr_page(
                {
                    "dt_polys": [
                        [[0, 0], [1, 0], [1, 1], [0, 1]],
                        [[2, 2], [3, 2], [3, 3], [2, 3]],
                    ],
                    "rec_texts": ["one"],
                }
            )

    def test_rejects_malformed_polygon(self):
        with self.assertRaises(OcrSchemaError):
            parse_ocr_page(
                {
                    "dt_polys": [[[0, 0], [1, 0], [1, 1]]],
                    "rec_texts": ["bad"],
                }
            )


if __name__ == "__main__":
    unittest.main()
