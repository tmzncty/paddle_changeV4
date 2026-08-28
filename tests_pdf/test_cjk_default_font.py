import json
import tempfile
import unittest
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF releases before the modern import alias
    import fitz

from paddle_batch_ocr.pdf_render import render_pdf
from paddle_batch_ocr.searchable_pdf import build_searchable_pdf


class DefaultCjkFontTests(unittest.TestCase):
    def test_default_china_s_font_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pdf"
            images = root / "pages"
            json_dir = root / "ocr"
            output = root / "searchable.pdf"
            json_dir.mkdir()

            document = fitz.open()
            try:
                document.new_page(width=240, height=120)
                document.save(str(source))
            finally:
                document.close()

            render_pdf(source, images, dpi=72)
            payload = {
                "dt_polys": [
                    [[20, 20], [210, 20], [210, 55], [20, 55]],
                ],
                "rec_texts": ["中文测试"],
            }
            (json_dir / "page_00001.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            result = build_searchable_pdf(images, json_dir, output)
            self.assertEqual(result.page_count, 1)
            self.assertEqual(result.text_line_count, 1)

            with fitz.open(str(output)) as rebuilt:
                extracted = rebuilt.load_page(0).get_text()

            self.assertIn("中文测试", extracted)


if __name__ == "__main__":
    unittest.main()
