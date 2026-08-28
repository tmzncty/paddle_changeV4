import contextlib
import io
import json
import os
import subprocess
import sys
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

    def test_process_redirect_catches_native_fd1_writes(self):
        script = r'''
import os
from paddle_batch_ocr.stdio import redirect_process_stdout_to_stderr

with redirect_process_stdout_to_stderr():
    os.write(1, b"native-oneDNN-noise\n")
    print("python-runtime-noise", flush=True)
print('{"ok": true}', flush=True)
'''
        env = os.environ.copy()
        src = str((Path(__file__).resolve().parents[1] / "src").resolve())
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src if not existing else os.pathsep.join((src, existing))

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(json.loads(completed.stdout), {"ok": True})
        self.assertNotIn("native-oneDNN-noise", completed.stdout)
        self.assertNotIn("python-runtime-noise", completed.stdout)
        self.assertIn("native-oneDNN-noise", completed.stderr)
        self.assertIn("python-runtime-noise", completed.stderr)


if __name__ == "__main__":
    unittest.main()
