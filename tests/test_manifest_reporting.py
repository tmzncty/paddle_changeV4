import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.manifest_reporting import (
    ManifestReportingError,
    query_manifest_jobs,
    read_manifest_report,
)


class ManifestReportingTests(unittest.TestCase):
    def _build_manifest(self, root: Path) -> Path:
        manifest = root / "manifest.sqlite3"
        a = root / "a.png"
        b = root / "b.png"
        c = root / "c.pdf"
        for path in (a, b, c):
            path.write_bytes(b"x")

        with ManifestStore(manifest) as store:
            store.mark_started(a, "ocr", worker="pid-1", device="cpu")
            store.mark_success(a, "ocr", result_path=root / "a.json", duration_s=1.5)

            store.mark_started(b, "ocr", worker="pid-2", device="cpu")
            store.mark_failure(b, "ocr", ValueError("bad page"), duration_s=2.0)
            store.mark_failure(b, "ocr", ValueError("bad page again"), duration_s=2.5)

            store.mark_started(c, "render", worker="pid-3")

        return manifest

    def test_report_aggregates_status_stage_errors_and_retries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._build_manifest(root)

            report = read_manifest_report(manifest)

            self.assertEqual(report.total, 3)
            self.assertEqual(
                report.status,
                {"failed": 1, "running": 1, "success": 1},
            )
            self.assertEqual(report.stages["ocr"], {"failed": 1, "success": 1})
            self.assertEqual(report.stages["render"], {"running": 1})
            self.assertEqual(report.error_classes, {"ValueError": 1})
            self.assertEqual(report.retry_total, 2)
            self.assertAlmostEqual(report.duration_total_s, 4.0)

    def test_query_filters_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._build_manifest(root)

            failed = query_manifest_jobs(
                manifest,
                status="failed",
                stage="ocr",
                error_class="ValueError",
            )
            self.assertEqual(len(failed), 1)
            self.assertTrue(str(failed[0]["source_path"]).endswith("b.png"))
            self.assertEqual(failed[0]["retry_count"], 2)
            self.assertEqual(failed[0]["error_message"], "bad page again")

            first = query_manifest_jobs(manifest, limit=2, offset=0)
            second = query_manifest_jobs(manifest, limit=2, offset=2)
            paths = [row["source_path"] for row in first + second]
            self.assertEqual(paths, sorted(paths))
            self.assertEqual(len(paths), 3)

    def test_reporting_connection_is_actually_read_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._build_manifest(root)

            before = manifest.stat().st_mtime_ns
            report = read_manifest_report(manifest)
            self.assertEqual(report.total, 3)
            after = manifest.stat().st_mtime_ns
            self.assertEqual(before, after)

            # Independent SQLite proof of the mode this module relies on.
            uri = "file:{}?mode=ro".format(manifest.as_posix())
            connection = sqlite3.connect(uri, uri=True)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("DELETE FROM jobs")
            finally:
                connection.close()

    def test_missing_manifest_does_not_get_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.sqlite3"
            with self.assertRaises(ManifestReportingError):
                read_manifest_report(path)
            self.assertFalse(path.exists())

    @unittest.skipIf(os.name == "nt", "symlink creation may require elevated privileges on Windows")
    def test_symlink_manifest_is_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._build_manifest(root)
            link = root / "link.sqlite3"
            link.symlink_to(manifest)

            with self.assertRaisesRegex(ManifestReportingError, "symlinked"):
                read_manifest_report(link)


if __name__ == "__main__":
    unittest.main()
