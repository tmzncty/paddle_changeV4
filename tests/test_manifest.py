import concurrent.futures
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.manifest import ManifestStore


class ManifestTests(unittest.TestCase):
    def test_successful_result_skips_when_source_and_result_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            result = root / "page.json"
            source.write_bytes(b"image")
            result.write_text("{}", encoding="utf-8")

            with ManifestStore(root / "manifest.sqlite3") as store:
                self.assertTrue(store.needs_run(source, "ocr"))
                store.mark_started(source, "ocr", worker="pid-1", device="gpu:0")
                store.mark_success(source, "ocr", result_path=result, duration_s=1.5)
                self.assertFalse(store.needs_run(source, "ocr"))

    def test_missing_result_requires_rerun_even_after_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            result = root / "page.json"
            source.write_bytes(b"image")
            result.write_text("{}", encoding="utf-8")

            with ManifestStore(root / "manifest.sqlite3") as store:
                store.mark_success(source, "ocr", result_path=result)
                result.unlink()
                self.assertTrue(store.needs_run(source, "ocr"))

    def test_source_change_resets_success_and_retry_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            result = root / "page.json"
            source.write_bytes(b"first")
            result.write_text("{}", encoding="utf-8")

            with ManifestStore(root / "manifest.sqlite3") as store:
                store.mark_failure(source, "ocr", ValueError("old failure"))
                store.mark_success(source, "ocr", result_path=result)
                self.assertFalse(store.needs_run(source, "ocr"))

                source.write_bytes(b"changed-size")
                self.assertTrue(store.needs_run(source, "ocr"))
                record = store.get_job(source, "ocr")
                self.assertIsNotNone(record)
                self.assertEqual(record.status, "pending")
                self.assertEqual(record.retry_count, 0)
                self.assertIsNone(record.result_path)

    def test_failure_records_error_and_retry_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            source.write_bytes(b"image")

            with ManifestStore(root / "manifest.sqlite3") as store:
                record = store.mark_failure(source, "ocr", ValueError("bad image"))
                self.assertEqual(record.status, "failed")
                self.assertEqual(record.retry_count, 1)
                self.assertEqual(record.error_class, "ValueError")
                self.assertIn("bad image", record.error_message)
                self.assertTrue(store.needs_run(source, "ocr"))

    def test_concurrent_first_registration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            database = root / "manifest.sqlite3"
            source.write_bytes(b"image")

            # Materialize the schema before creating competing worker connections.
            with ManifestStore(database):
                pass

            def register(_):
                with ManifestStore(database) as store:
                    return store.ensure_job(source, "ocr").status

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                statuses = list(executor.map(register, range(12)))

            self.assertEqual(statuses, ["pending"] * 12)
            with ManifestStore(database) as store:
                self.assertEqual(store.summary(), {"pending": 1})

    def test_stages_are_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "book.pdf"
            rendered = root / "rendered.done"
            source.write_bytes(b"pdf")
            rendered.write_text("ok", encoding="utf-8")

            with ManifestStore(root / "manifest.sqlite3") as store:
                store.mark_success(source, "render", result_path=rendered)
                self.assertFalse(store.needs_run(source, "render"))
                self.assertTrue(store.needs_run(source, "ocr"))
                self.assertEqual(store.summary(), {"pending": 1, "success": 1})


if __name__ == "__main__":
    unittest.main()
