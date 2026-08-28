import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paddle_batch_ocr.cli import main
from paddle_batch_ocr.ocr_runner import OcrBatchResult, OcrTaskResult


class OcrJsonOutputTests(unittest.TestCase):
    def test_json_mode_routes_runtime_stdout_to_stderr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            output = root / "json"
            source.write_bytes(b"x")

            result = OcrBatchResult(
                tasks=(
                    OcrTaskResult(
                        source=source.resolve(),
                        output_json=(output / "page_result.json").resolve(),
                        status="success",
                    ),
                )
            )

            def noisy_runner(*args, **kwargs):
                print("third-party model download chatter")
                return result

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("paddle_batch_ocr.cli.run_ocr_batch", side_effect=noisy_runner):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    rc = main(
                        [
                            "ocr",
                            str(source),
                            "--output",
                            str(output),
                            "--json",
                        ]
                    )

            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["success"], 1)
            self.assertNotIn("third-party", stdout.getvalue())
            self.assertIn("third-party model download chatter", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
