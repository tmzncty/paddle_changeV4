import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.naming import candidate_json_names, find_matching_json


class NamingTests(unittest.TestCase):
    def test_page_number_candidates_preserve_legacy_precedence(self):
        names = candidate_json_names("page_0001.png")
        self.assertEqual(
            names[:4],
            (
                "page_00001.json",
                "page_0001_result.json",
                "page_0001.json",
                "page_001.json",
            ),
        )

    def test_generic_filename_fallbacks(self):
        names = candidate_json_names("cover-front.jpg")
        self.assertEqual(
            names,
            (
                "cover-front_result.json",
                "cover-front.json",
                "cover-front_ocr.json",
            ),
        )

    def test_find_matching_json_uses_precedence_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lower_priority = root / "page_0001_result.json"
            higher_priority = root / "page_00001.json"
            lower_priority.write_text("{}", encoding="utf-8")
            higher_priority.write_text("{}", encoding="utf-8")

            self.assertEqual(
                find_matching_json("page_0001.png", root),
                higher_priority,
            )

    def test_find_matching_json_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(find_matching_json("page_0012.png", Path(temp_dir)))


if __name__ == "__main__":
    unittest.main()
