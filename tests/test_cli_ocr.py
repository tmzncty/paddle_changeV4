import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paddle_batch_ocr.cli import main
from paddle_batch_ocr.ocr_runner import OcrBatchResult, OcrTaskResult

class OcrCliTests(unittest.TestCase):
    def test_cli_forwards_current_paddlex_execution_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); source = root / "page.png"; output_dir = root / "json"; manifest = root / "manifest.sqlite3"; source.write_bytes(b"x"); expected_output = output_dir / "page_result.json"
            fake_result = OcrBatchResult(tasks=(OcrTaskResult(source.resolve(), expected_output.resolve(), "success"),))
            stdout = io.StringIO()
            with patch("paddle_batch_ocr.cli.run_ocr_batch", return_value=fake_result) as run:
                with contextlib.redirect_stdout(stdout):
                    rc = main(["ocr", str(source), "--output", str(output_dir), "--pipeline", "OCR", "--device", "gpu:0", "--engine", "paddle_static", "--use-hpip", "--use-doc-orientation-classify", "--use-doc-unwarping", "--use-textline-orientation", "--manifest", str(manifest), "--overwrite", "--json"])
            self.assertEqual(rc, 0)
            run.assert_called_once_with(source, output_dir, pipeline_ref="OCR", device="gpu:0", engine="paddle_static", use_hpip=True, manifest_path=manifest, resume=True, overwrite=True, use_doc_orientation_classify=True, use_doc_unwarping=True, use_textline_orientation=True, workers=1)
            payload = json.loads(stdout.getvalue()); self.assertEqual(payload["success"], 1); self.assertEqual(payload["failed"], 0)
    def test_cli_defaults_optional_preprocessors_and_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); source = root / "page.png"; output_dir = root / "json"; source.write_bytes(b"x"); fake_result = OcrBatchResult(tasks=(OcrTaskResult(source.resolve(), (output_dir / "page_result.json").resolve(), "success"),))
            with patch("paddle_batch_ocr.cli.run_ocr_batch", return_value=fake_result) as run:
                with contextlib.redirect_stdout(io.StringIO()): rc = main(["ocr", str(source), "--output", str(output_dir)])
            self.assertEqual(rc, 0); _, kwargs = run.call_args; self.assertFalse(kwargs["use_doc_orientation_classify"]); self.assertFalse(kwargs["use_doc_unwarping"]); self.assertFalse(kwargs["use_textline_orientation"]); self.assertEqual(kwargs["workers"], 1)
    def test_cli_forwards_worker_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); source = root / "page.png"; output_dir = root / "json"; source.write_bytes(b"x"); fake_result = OcrBatchResult(tasks=(OcrTaskResult(source.resolve(), (output_dir / "page_result.json").resolve(), "success"),))
            with patch("paddle_batch_ocr.cli.run_ocr_batch", return_value=fake_result) as run:
                with contextlib.redirect_stdout(io.StringIO()): rc = main(["ocr", str(source), "--output", str(output_dir), "--device", "cpu", "--workers", "3"])
            self.assertEqual(rc, 0); self.assertEqual(run.call_args.kwargs["workers"], 3)
    def test_cli_returns_one_when_batch_contains_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); source = root / "page.png"; output_dir = root / "json"; source.write_bytes(b"x"); fake_result = OcrBatchResult(tasks=(OcrTaskResult(source.resolve(), (output_dir / "page_result.json").resolve(), "failed", "PaddleXResultError: no result"),))
            with patch("paddle_batch_ocr.cli.run_ocr_batch", return_value=fake_result):
                with contextlib.redirect_stdout(io.StringIO()): rc = main(["ocr", str(source), "--output", str(output_dir)])
            self.assertEqual(rc, 1)

if __name__ == "__main__": unittest.main()
