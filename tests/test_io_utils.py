import json
import os
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.io_utils import atomic_write_json


class AtomicJsonTests(unittest.TestCase):
    def test_writes_unicode_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "result.json"
            atomic_write_json(target, {"text": "中文"})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"text": "中文"},
            )

    def test_no_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "result.json"
            atomic_write_json(target, {"value": 1})
            with self.assertRaises(FileExistsError):
                atomic_write_json(target, {"value": 2})
            self.assertEqual(json.loads(target.read_text()), {"value": 1})

    def test_explicit_overwrite_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "result.json"
            atomic_write_json(target, {"value": 1})
            atomic_write_json(target, {"value": 2}, overwrite=True)
            self.assertEqual(json.loads(target.read_text()), {"value": 2})

    @unittest.skipIf(os.name == "nt", "symlink creation may require elevated privileges on Windows")
    def test_symlink_target_is_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real = root / "real.json"
            link = root / "result.json"
            real.write_text('{"value": 1}', encoding="utf-8")
            link.symlink_to(real)

            with self.assertRaises(ValueError):
                atomic_write_json(link, {"value": 2}, overwrite=True)

            self.assertEqual(json.loads(real.read_text(encoding="utf-8")), {"value": 1})


if __name__ == "__main__":
    unittest.main()
