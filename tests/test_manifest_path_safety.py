import os
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.manifest import ManifestStore


class ManifestPathSafetyTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "symlink permissions vary on Windows CI")
    def test_rejects_symlinked_manifest_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_db = root / "real.sqlite3"
            link_db = root / "manifest.sqlite3"
            link_db.symlink_to(real_db)

            with self.assertRaises(ValueError):
                ManifestStore(link_db)

            self.assertFalse(real_db.exists())


if __name__ == "__main__":
    unittest.main()
