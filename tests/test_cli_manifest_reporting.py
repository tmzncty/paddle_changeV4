import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.cli import main
from paddle_batch_ocr.manifest import ManifestStore


class ManifestReportingCliTests(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        inputs = root / "inputs"
        inputs.mkdir()
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "input_sources": [
                        {"path": str(inputs), "type": "image"}
                    ],
                    "output_root": str(root / "output"),
                    "log_dir": str(root / "logs"),
                    "cache_root": str(root / "cache"),
                    "manifest_path": str(root / "logs" / "manifest.sqlite3"),
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def _populate(self, root: Path) -> Path:
        manifest = root / "logs" / "manifest.sqlite3"
        source_ok = root / "ok.png"
        source_bad = root / "bad.png"
        source_ok.write_bytes(b"x")
        source_bad.write_bytes(b"x")

        with ManifestStore(manifest) as store:
            store.mark_success(source_ok, "ocr", result_path=root / "ok.json")
            store.mark_failure(source_bad, "ocr", RuntimeError("decode failed"))
        return manifest

    def test_report_json_and_jobs_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._write_config(root)
            self._populate(root)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(["manifest", "report", "--config", str(config), "--json"])
            self.assertEqual(rc, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["total"], 2)
            self.assertEqual(report["status"], {"failed": 1, "success": 1})
            self.assertEqual(report["error_classes"], {"RuntimeError": 1})

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "manifest",
                        "jobs",
                        "--config",
                        str(config),
                        "--status",
                        "failed",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            jobs = json.loads(stdout.getvalue())
            self.assertEqual(jobs["count"], 1)
            self.assertEqual(jobs["jobs"][0]["error_class"], "RuntimeError")
            self.assertEqual(jobs["jobs"][0]["error_message"], "decode failed")

    def test_jobs_csv_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._write_config(root)
            self._populate(root)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "manifest",
                        "jobs",
                        "--config",
                        str(config),
                        "--stage",
                        "ocr",
                        "--csv",
                    ]
                )
            self.assertEqual(rc, 0)
            rows = list(csv.DictReader(io.StringIO(stdout.getvalue())))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["status"] for row in rows}, {"success", "failed"})

    def test_missing_manifest_report_and_jobs_do_not_create_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._write_config(root)
            manifest = root / "logs" / "manifest.sqlite3"
            self.assertFalse(manifest.exists())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(["manifest", "report", "--config", str(config), "--json"])
            self.assertEqual(rc, 0)
            self.assertFalse(manifest.exists())
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["exists"])
            self.assertEqual(payload["total"], 0)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(["manifest", "jobs", "--config", str(config), "--json"])
            self.assertEqual(rc, 0)
            self.assertFalse(manifest.exists())
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["jobs"], [])

    def test_invalid_limit_returns_cli_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = self._write_config(root)
            self._populate(root)

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "manifest",
                            "jobs",
                            "--config",
                            str(config),
                            "--limit",
                            "-1",
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("limit must be an integer", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
