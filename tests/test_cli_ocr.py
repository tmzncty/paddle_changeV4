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
            root = Path(temp_dir)
            source = root / "page.png"
            output_dir = root / "json"
            manifest = root / "manifest.sqlite3"
            source.write_bytes(b"x")
            expected_output = output_dir / "page_result.json"
            fake_result = OcrBatchResult(
                tasks=(
                    OcrTaskResult(
                        source=source.resolve(),
                        output_json=expected_output.resolve(),
                        status="success",
                    ),
                )
            )

            stdout = io.StringIO()
            with patch("paddle_batch_ocr.cli.run_ocr_batch", return_value=fake_result) as run:
                with contextlib.redirect_stdout(stdout):
                    rc = main(
                        [
                            "ocr",
                            str(source),
                            "--output",
                            str(output_dir),
                            "--pipeline",
                            "OCR",
                            "--device",
                            "gpu:0",
                            "--engine",
                            "paddle_static",
                            "--use-hpip",
                            "--use-doc-orientation-classify",
                            "--use-doc-unwarping",
                            "--use-textline-orientation",
                            "--manifest",
                            str(manifest),
                            "--overwrite",
                            "--json",
                        ]
                    )

            self.assertEqual(rc, 0)
            run.assert_called_once_with(
                source,
                output_dir,
                pipeline_ref="OCR",
                device="gpu:0",
                engine="paddle_static",
                use_hpip=True,
                manifest_path=manifest,
                resume=True,
                overwrite=True,
                use_doc_orientation_classify=True,
                use_doc_unwarping=True,
                use_textline_orientation=True,
            )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["success"], 1)
            self.assertEqual(payload["failed"], 0)

    def test_cli_defaults_optional_preprocessors_off(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            output_dir = root / "json"
            source.write_bytes(b"x")
            fake_result = OcrBatchResult(
                tasks=(
                    OcrTaskResult(
                        source=source.resolve(),
                        output_json=(output_dir / "page_result.json").resolve(),
                        status="success",
                    ),
                )
            )

            with patch("paddle_batch_ocr.cli.run_ocr_batch", return_value=fake_result) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = main(["ocr", str(source), "--output", str(output_dir)])

            self.assertEqual(rc, 0)
            _, kwargs = run.call_args
            self.assertFalse(kwargs["use_doc_orientation_classify"])
            self.assertFalse(kwargs["use_doc_unwarping"])
            self.assertFalse(kwargs["use_textline_orientation"])

    def test_cli_returns_one_when_batch_contains_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            output_dir = root / "json"
            source.write_bytes(b"x")
            fake_result = OcrBatchResult(
                tasks=(
                    OcrTaskResult(
                        source=source.resolve(),
                        output_json=(output_dir / "page_result.json").resolve(),
                        status="failed",
                        error="PaddleXResultError: no result",
                    ),
                )
            )

            with patch("paddle_batch_ocr.cli.run_ocr_batch", return_value=fake_result):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = main(["ocr", str(source), "--output", str(output_dir)])

            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
