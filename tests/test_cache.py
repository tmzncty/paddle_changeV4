import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.cache import clean_temp_cache
from paddle_batch_ocr.safety import UnsafePathError


class CacheTests(unittest.TestCase):
    def test_dry_run_is_default_and_does_not_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            target = root / "temp"
            target.mkdir(parents=True)
            marker = target / "keep.txt"
            marker.write_text("still here", encoding="utf-8")

            result = clean_temp_cache(root)

            self.assertFalse(result.executed)
            self.assertTrue(result.existed)
            self.assertTrue(marker.exists())

    def test_execute_deletes_contents_and_recreates_temp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            target = root / "temp"
            target.mkdir(parents=True)
            (target / "remove.txt").write_text("remove", encoding="utf-8")

            result = clean_temp_cache(root, execute=True)

            self.assertTrue(result.executed)
            self.assertTrue(result.recreated)
            self.assertTrue(target.is_dir())
            self.assertEqual(list(target.iterdir()), [])

    def test_execute_can_leave_temp_absent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            target = root / "temp"
            target.mkdir(parents=True)

            result = clean_temp_cache(root, execute=True, recreate=False)

            self.assertTrue(result.executed)
            self.assertFalse(result.recreated)
            self.assertFalse(target.exists())

    def test_rejects_symlinked_temp_even_when_target_is_inside_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            actual = root / "actual-temp"
            actual.mkdir(parents=True)
            marker = actual / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            (root / "temp").symlink_to(actual, target_is_directory=True)

            with self.assertRaises(UnsafePathError):
                clean_temp_cache(root, execute=True)

            self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
