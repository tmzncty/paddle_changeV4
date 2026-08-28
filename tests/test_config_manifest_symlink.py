import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import load_config


class ConfigManifestSymlinkTests(unittest.TestCase):
    def test_explicit_manifest_symlink_is_not_resolved_away(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = root / "inputs"
            output = root / "output"
            logs = root / "logs"
            cache = root / "cache"
            for path in (inputs, output, logs, cache):
                path.mkdir()

            real = root / "real.sqlite3"
            real.write_bytes(b"db")
            link = logs / "manifest.sqlite3"
            try:
                link.symlink_to(real)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input_sources": [{"path": str(inputs), "type": "image"}],
                        "output_root": str(output),
                        "log_dir": str(logs),
                        "cache_root": str(cache),
                        "manifest_path": str(link),
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            self.assertEqual(config.manifest_path, link.absolute())
            self.assertTrue(config.manifest_path.is_symlink())


if __name__ == "__main__":
    unittest.main()
