import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.cli import main
from paddle_batch_ocr.config import load_config
from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.manifest_reporting import query_manifest_job_page
from paddle_batch_ocr.ocr_runner import build_ocr_execution_profile


class TargetedRetryCliTests(unittest.TestCase):
    def _fixture(self, root: Path):
        inputs = root / "inputs"
        output = root / "output"
        logs = root / "logs"
        cache = root / "cache"
        inputs.mkdir()
        output.mkdir()
        logs.mkdir()
        cache.mkdir()
        pipeline = root / "ocr.yaml"
        pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
        source = inputs / "page.png"
        source.write_bytes(b"source")
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "input_sources": [{"path": str(inputs), "type": "image"}],
                    "output_root": str(output),
                    "log_dir": str(logs),
                    "cache_root": str(cache),
                    "manifest_path": str(logs / "manifest.sqlite3"),
                    "paddle_config": str(pipeline),
                    "runtime": {"device": "cpu", "ocr_workers": 1},
                    "overwrite": False,
                    "resume": True,
                }
            ),
            encoding="utf-8",
        )
        config = load_config(config_path)
        target = output / "ocr" / "page_result.json"
        profile = build_ocr_execution_profile(
            pipeline_ref=pipeline,
            device="cpu",
            engine=None,
            use_hpip=None,
            predict_kwargs={
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            },
        )
        with ManifestStore(config.manifest_path) as store:
            store.mark_failure(
                source,
                "ocr",
                RuntimeError("synthetic failure"),
                intended_result_path=target,
                execution_profile=profile,
            )
        return config_path, config, source, target, pipeline

    def _empty_config(self, root: Path):
        inputs = root / "inputs"
        output = root / "output"
        logs = root / "logs"
        cache = root / "cache"
        for path in (inputs, output, logs, cache):
            path.mkdir()
        config_path = root / "config.json"
        manifest = logs / "manifest.sqlite3"
        config_path.write_text(
            json.dumps(
                {
                    "input_sources": [{"path": str(inputs), "type": "image"}],
                    "output_root": str(output),
                    "log_dir": str(logs),
                    "cache_root": str(cache),
                    "manifest_path": str(manifest),
                }
            ),
            encoding="utf-8",
        )
        return config_path, manifest

    def test_json_dry_run_is_read_only_and_reports_eligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, config, source, target, _ = self._fixture(root)
            before = query_manifest_job_page(config.manifest_path, status="failed").jobs[0]

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "manifest",
                        "retry-failed",
                        "--config",
                        str(config_path),
                        "--stage",
                        "ocr",
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(payload["eligible"], 1)
            self.assertEqual(payload["blocked"], 0)
            self.assertEqual(payload["ineligible"], 0)
            self.assertEqual(payload["candidates"][0]["state"], "eligible")
            self.assertEqual(payload["candidates"][0]["source"], str(source.resolve()))
            self.assertFalse(target.exists())

            after = query_manifest_job_page(config.manifest_path, status="failed").jobs[0]
            self.assertEqual(before, after)

    def test_existing_target_is_reported_as_blocked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _, _, target, _ = self._fixture(root)
            target.parent.mkdir(parents=True)
            target.write_text("existing", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "manifest",
                        "retry-failed",
                        "--config",
                        str(config_path),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["eligible"], 0)
            self.assertEqual(payload["blocked"], 1)
            self.assertEqual(payload["candidates"][0]["state"], "blocked")

    def test_overwrite_turns_existing_target_into_eligible_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _, _, target, _ = self._fixture(root)
            target.parent.mkdir(parents=True)
            target.write_text("existing", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "manifest",
                        "retry-failed",
                        "--config",
                        str(config_path),
                        "--overwrite",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["eligible"], 1)
            self.assertTrue(payload["overwrite"])

    def test_hash_drift_is_reported_as_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, _, _, _, pipeline = self._fixture(root)
            pipeline.write_text("changed pipeline configuration\n", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "manifest",
                        "retry-failed",
                        "--config",
                        str(config_path),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["eligible"], 0)
            self.assertEqual(payload["ineligible"], 1)
            self.assertEqual(payload["candidates"][0]["state"], "ineligible")

    def test_missing_manifest_returns_empty_dry_run_without_creating_db(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, manifest = self._empty_config(root)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "manifest",
                        "retry-failed",
                        "--config",
                        str(config_path),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["total_matching"], 0)
            self.assertEqual(payload["candidates"], [])
            self.assertFalse(manifest.exists())

    def test_invalid_limit_is_rejected_even_without_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, manifest = self._empty_config(root)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "manifest",
                            "retry-failed",
                            "--config",
                            str(config_path),
                            "--limit",
                            "-1",
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("limit must be an integer", stderr.getvalue())
            self.assertFalse(manifest.exists())

    def test_symlinked_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path, manifest = self._empty_config(root)
            real = root / "real.sqlite3"
            real.write_bytes(b"not a database")
            try:
                manifest.symlink_to(real)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "manifest",
                            "retry-failed",
                            "--config",
                            str(config_path),
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("symlinked manifest", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
