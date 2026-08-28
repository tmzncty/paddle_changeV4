import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.cli import main


class CliTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        (root / "pdfs").mkdir()
        (root / "images").mkdir()
        (root / "pdfs" / "a.pdf").write_bytes(b"%PDF-placeholder")
        (root / "pdfs" / "ignore.txt").write_text("x", encoding="utf-8")
        (root / "images" / "a.png").write_bytes(b"png")
        (root / "images" / "b.JPG").write_bytes(b"jpg")

        config = {
            "input_sources": [
                {"path": "pdfs", "type": "pdf"},
                {"path": "images", "type": "image"},
            ],
            "output_root": "work/output",
            "log_dir": "work/logs",
            "cache_root": "work/cache",
        }
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_scan_counts_only_matching_file_types(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = main(["scan", "--config", str(config_path)])

            self.assertEqual(rc, 0)
            text = output.getvalue()
            self.assertIn("pdf", text)
            self.assertIn("image", text)
            self.assertIn("total", text)
            self.assertIn("3", text)

    def test_cache_clean_is_dry_run_without_execute(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self._write_config(root)
            cache_temp = root / "work" / "cache" / "temp"
            cache_temp.mkdir(parents=True)
            marker = cache_temp / "marker"
            marker.write_text("keep", encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = main(["cache", "clean", "--config", str(config_path)])

            self.assertEqual(rc, 0)
            self.assertTrue(marker.exists())
            self.assertIn("dry-run", output.getvalue())


if __name__ == "__main__":
    unittest.main()
