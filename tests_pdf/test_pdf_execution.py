import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import fitz
from PIL import Image

from paddle_batch_ocr.cli import main
from paddle_batch_ocr.pdf_render import render_pdf
from paddle_batch_ocr.searchable_pdf import (
    SearchablePdfError,
    build_searchable_pdf,
    discover_numbered_page_images,
)


class PdfExecutionTests(unittest.TestCase):
    def _make_source_pdf(self, path: Path, pages: int = 2) -> None:
        document = fitz.open()
        try:
            for index in range(pages):
                page = document.new_page(width=200, height=100)
                page.insert_text((20, 50), f"source page {index + 1}", fontname="helv", fontsize=12)
            document.save(str(path))
        finally:
            document.close()

    def _write_ocr_json(self, path: Path, text: str, *, field: str = "rec_texts") -> None:
        payload = {
            "dt_polys": [
                [[20, 20], [170, 20], [170, 45], [20, 45]],
            ],
            field: [text],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_render_and_searchable_pdf_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pdf"
            images = root / "pages"
            json_dir = root / "ocr"
            output = root / "searchable.pdf"
            json_dir.mkdir()
            self._make_source_pdf(source, pages=2)

            render_result = render_pdf(source, images, dpi=72)
            self.assertEqual(render_result.page_count, 2)
            self.assertEqual(
                [path.name for path in render_result.page_paths],
                ["page_00001.png", "page_00002.png"],
            )
            self.assertTrue(all(path.is_file() for path in render_result.page_paths))

            # Exercise both historical text-field spellings and two filename conventions.
            self._write_ocr_json(json_dir / "page_00001.json", "alpha one", field="rec_texts")
            self._write_ocr_json(
                json_dir / "page_0002_result.json",
                "beta two",
                field="rec_text",
            )

            result = build_searchable_pdf(
                images,
                json_dir,
                output,
                fontname="helv",
            )
            self.assertEqual(result.page_count, 2)
            self.assertEqual(result.text_line_count, 2)
            self.assertTrue(output.is_file())

            with fitz.open(str(output)) as document:
                self.assertEqual(document.page_count, 2)
                self.assertIn("alpha one", document.load_page(0).get_text())
                self.assertIn("beta two", document.load_page(1).get_text())

    def test_missing_json_is_hard_error_and_does_not_publish_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pdf"
            images = root / "pages"
            json_dir = root / "ocr"
            output = root / "searchable.pdf"
            json_dir.mkdir()
            self._make_source_pdf(source, pages=2)
            render_pdf(source, images, dpi=72)
            self._write_ocr_json(json_dir / "page_00001.json", "only first page")

            with self.assertRaises(SearchablePdfError):
                build_searchable_pdf(images, json_dir, output, fontname="helv")

            self.assertFalse(output.exists())

    def test_page_image_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for number in (1, 3):
                image_path = root / f"page_{number:05d}.png"
                with Image.new("RGB", (32, 32), "white") as image:
                    image.save(image_path)

            with self.assertRaises(SearchablePdfError):
                discover_numbered_page_images(root)

    def test_render_refuses_existing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pdf"
            output = root / "pages"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            self._make_source_pdf(source, pages=1)

            with self.assertRaises(FileExistsError):
                render_pdf(source, output, dpi=72)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_cli_render_and_searchable_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pdf"
            images = root / "pages"
            json_dir = root / "ocr"
            output = root / "searchable.pdf"
            json_dir.mkdir()
            self._make_source_pdf(source, pages=1)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "render",
                        str(source),
                        "--output",
                        str(images),
                        "--dpi",
                        "72",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            render_payload = json.loads(stdout.getvalue())
            self.assertEqual(render_payload["page_count"], 1)

            self._write_ocr_json(json_dir / "page_00001.json", "cli searchable")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "searchable-pdf",
                        "--images",
                        str(images),
                        "--ocr-json",
                        str(json_dir),
                        "--output",
                        str(output),
                        "--fontname",
                        "helv",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            searchable_payload = json.loads(stdout.getvalue())
            self.assertEqual(searchable_payload["page_count"], 1)
            self.assertEqual(searchable_payload["text_line_count"], 1)

            with fitz.open(str(output)) as document:
                self.assertIn("cli searchable", document.load_page(0).get_text())


if __name__ == "__main__":
    unittest.main()
