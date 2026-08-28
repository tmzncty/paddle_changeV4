import json
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


if __name__ == "__main__":
    unittest.main()
