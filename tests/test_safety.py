import os
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.safety import (
    UnsafePathError,
    is_within,
    validate_destructive_target,
)


class PathSafetyTests(unittest.TestCase):
    def test_is_within_accepts_child_and_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            child = root / "temp" / "job"
            child.mkdir(parents=True)

            self.assertTrue(is_within(child, root))
            self.assertTrue(is_within(root, root))

    def test_is_within_rejects_prefix_sibling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            sibling = Path(temp_dir) / "cache-evil"
            root.mkdir()
            sibling.mkdir()

            self.assertFalse(is_within(sibling, root))

    def test_validate_accepts_child_inside_safe_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            target = root / "temp"
            target.mkdir(parents=True)

            self.assertEqual(validate_destructive_target(target, root), target.resolve())

    def test_validate_rejects_target_outside_allowed_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            outside = Path(temp_dir) / "other"
            root.mkdir()
            outside.mkdir()

            with self.assertRaises(UnsafePathError):
                validate_destructive_target(outside, root)

    def test_validate_rejects_allowed_root_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            root.mkdir()

            with self.assertRaises(UnsafePathError):
                validate_destructive_target(root, root)

    def test_validate_can_explicitly_allow_safe_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            root.mkdir()

            self.assertEqual(
                validate_destructive_target(root, root, allow_root=True),
                root.resolve(),
            )

    def test_validate_rejects_home_as_allowed_root(self):
        home = Path.home().resolve()
        with self.assertRaises(UnsafePathError):
            validate_destructive_target(home / "temp", home)

    def test_validate_rejects_cwd_as_allowed_root(self):
        cwd = Path.cwd().resolve()
        with self.assertRaises(UnsafePathError):
            validate_destructive_target(cwd / "temp", cwd)

    @unittest.skipIf(os.name == "nt", "filesystem-root assertion differs on Windows")
    def test_validate_rejects_filesystem_root_as_allowed_root(self):
        with self.assertRaises(UnsafePathError):
            validate_destructive_target("/tmp", "/", allow_root=True)


if __name__ == "__main__":
    unittest.main()
