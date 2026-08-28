import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.manifest_reporting import (
    query_manifest_job_page,
    read_manifest_report,
)


LEGACY_SCHEMA = """
CREATE TABLE jobs (
    source_path TEXT NOT NULL,
    stage TEXT NOT NULL,
    source_size INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    result_path TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_class TEXT,
    error_message TEXT,
    worker TEXT,
    device TEXT,
    started_at TEXT,
    finished_at TEXT,
    duration_s REAL,
    PRIMARY KEY (source_path, stage)
)
"""


class ManifestProvenanceTests(unittest.TestCase):
    def _legacy_database(
        self,
        root: Path,
        *,
        with_success: bool = True,
    ):
        source = root / "page.png"
        result = root / "page_result.json"
        source.write_bytes(b"source")
        if with_success:
            result.write_text("{}", encoding="utf-8")

        stat = source.stat()
        manifest = root / "manifest.sqlite3"
        connection = sqlite3.connect(manifest)
        try:
            connection.execute(LEGACY_SCHEMA)
            connection.execute(
                """
                INSERT INTO jobs (
                    source_path, stage, source_size, source_mtime_ns,
                    status, result_path, retry_count
                ) VALUES (?, 'ocr', ?, ?, ?, ?, 0)
                """,
                (
                    str(source.resolve()),
                    stat.st_size,
                    stat.st_mtime_ns,
                    "success" if with_success else "failed",
                    str(result.resolve()) if with_success else None,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return manifest, source, result

    def test_legacy_success_migrates_without_forcing_profile_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source, result = self._legacy_database(root)

            profile = {
                "schema": 1,
                "kind": "paddlex_ocr",
                "device": "cpu",
            }
            with ManifestStore(manifest) as store:
                record = store.get_job(source, "ocr")
                self.assertIsNotNone(record)
                self.assertEqual(record.intended_result_path, str(result.resolve()))
                self.assertIsNone(record.execution_profile_json)

                self.assertFalse(
                    store.needs_run(
                        source,
                        "ocr",
                        intended_result_path=result,
                        execution_profile=profile,
                    )
                )
                after = store.get_job(source, "ocr")
                self.assertEqual(after.status, "success")
                # Do not pretend a historical success was produced by the new profile.
                self.assertIsNone(after.execution_profile_json)

    def test_failed_attempt_keeps_target_and_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.sqlite3"
            source = root / "page.png"
            target = root / "json" / "page_result.json"
            source.write_bytes(b"x")
            profile = {
                "schema": 1,
                "kind": "paddlex_ocr",
                "pipeline_ref": "OCR",
                "device": "cpu",
                "predict": {"use_doc_unwarping": False},
            }

            with ManifestStore(manifest) as store:
                store.mark_started(
                    source,
                    "ocr",
                    worker="pid-1",
                    device="cpu",
                    intended_result_path=target,
                    execution_profile=profile,
                )
                failed = store.mark_failure(
                    source,
                    "ocr",
                    RuntimeError("model failed"),
                )

                self.assertEqual(failed.status, "failed")
                self.assertIsNone(failed.result_path)
                self.assertEqual(failed.intended_result_path, str(target.resolve()))
                self.assertEqual(failed.execution_profile, profile)

    def test_known_profile_change_invalidates_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.sqlite3"
            source = root / "page.png"
            target = root / "page_result.json"
            source.write_bytes(b"x")
            target.write_text("{}", encoding="utf-8")
            profile_a = {"schema": 1, "kind": "ocr", "device": "cpu", "a": 1}
            profile_a_reordered = {"a": 1, "device": "cpu", "kind": "ocr", "schema": 1}
            profile_b = {"schema": 1, "kind": "ocr", "device": "cpu", "a": 2}

            with ManifestStore(manifest) as store:
                store.mark_started(
                    source,
                    "ocr",
                    intended_result_path=target,
                    execution_profile=profile_a,
                )
                store.mark_success(source, "ocr", result_path=target)

                self.assertFalse(
                    store.needs_run(
                        source,
                        "ocr",
                        intended_result_path=target,
                        execution_profile=profile_a_reordered,
                    )
                )
                self.assertTrue(
                    store.needs_run(
                        source,
                        "ocr",
                        intended_result_path=target,
                        execution_profile=profile_b,
                    )
                )
                changed = store.get_job(source, "ocr")
                self.assertEqual(changed.status, "pending")
                self.assertIsNone(changed.result_path)
                self.assertEqual(changed.execution_profile, profile_b)

    def test_known_intended_output_change_invalidates_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.sqlite3"
            source = root / "page.png"
            first = root / "a.json"
            second = root / "b.json"
            source.write_bytes(b"x")
            first.write_text("{}", encoding="utf-8")
            profile = {"schema": 1, "kind": "ocr"}

            with ManifestStore(manifest) as store:
                store.mark_started(
                    source,
                    "ocr",
                    intended_result_path=first,
                    execution_profile=profile,
                )
                store.mark_success(source, "ocr", result_path=first)
                self.assertTrue(
                    store.needs_run(
                        source,
                        "ocr",
                        intended_result_path=second,
                        execution_profile=profile,
                    )
                )
                changed = store.get_job(source, "ocr")
                self.assertEqual(changed.intended_result_path, str(second.resolve()))
                self.assertEqual(changed.status, "pending")

    def test_concurrent_first_open_migrates_legacy_schema_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source, result = self._legacy_database(root)

            def open_and_read(_):
                with ManifestStore(manifest) as store:
                    record = store.get_job(source, "ocr")
                    return (
                        record.intended_result_path,
                        record.execution_profile_json,
                    )

            with ThreadPoolExecutor(max_workers=4) as executor:
                values = list(executor.map(open_and_read, range(8)))

            self.assertEqual(
                values,
                [(str(result.resolve()), None)] * 8,
            )
            connection = sqlite3.connect(manifest)
            try:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(jobs)")
                }
            finally:
                connection.close()
            self.assertIn("intended_result_path", columns)
            self.assertIn("execution_profile_json", columns)

    def test_read_only_reporting_handles_unmigrated_legacy_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source, _ = self._legacy_database(root)

            report = read_manifest_report(manifest)
            self.assertEqual(report.total, 1)
            self.assertEqual(report.intended_result_count, 0)
            self.assertEqual(report.execution_profile_count, 0)

            page = query_manifest_job_page(manifest)
            self.assertEqual(page.total_matching, 1)
            self.assertEqual(page.jobs[0]["source_path"], str(source.resolve()))
            self.assertIsNone(page.jobs[0]["intended_result_path"])
            self.assertIsNone(page.jobs[0]["execution_profile_json"])

            connection = sqlite3.connect(manifest)
            try:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(jobs)")
                }
            finally:
                connection.close()
            self.assertNotIn("intended_result_path", columns)
            self.assertNotIn("execution_profile_json", columns)

    def test_execution_profile_must_be_json_serializable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.sqlite3"
            source = root / "page.png"
            source.write_bytes(b"x")

            with ManifestStore(manifest) as store:
                with self.assertRaisesRegex(TypeError, "JSON-serializable"):
                    store.mark_started(
                        source,
                        "ocr",
                        execution_profile={"bad": object()},
                    )


if __name__ == "__main__":
    unittest.main()
