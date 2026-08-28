import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import load_config
from paddle_batch_ocr.retry import plan_failed_retries


class RetryMissingManifestTests(unittest.TestCase):
    def test_plan_missing_manifest_is_empty_and_no_side_effect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = root / "inputs"
            output = root / "output"
            logs = root / "logs"
            cache = root / "cache"
            for path in (inputs, output, logs, cache):
                path.mkdir()

            manifest = logs / "manifest.sqlite3"
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_sources": [{"path": str(inputs), "type": "image"}],
                        "output_root": str(output),
                        "log_dir": str(logs),
                        "cache_root": str(cache),
                        "manifest_path": str(manifest),
                    }
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertFalse(manifest.exists())
            plan = plan_failed_retries(config)
            self.assertEqual(plan.total_matching, 0)
            self.assertEqual(plan.candidates, ())
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
