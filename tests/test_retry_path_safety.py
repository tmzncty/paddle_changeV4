import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import InputSource, ProjectConfig, RuntimeConfig
from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import build_ocr_execution_profile
from paddle_batch_ocr.retry import plan_failed_retries


class RetryPathSafetyTests(unittest.TestCase):
    def _project(self, root: Path):
        inputs = root / "inputs"
        output = root / "output"
        logs = root / "logs"
        cache = root / "cache"
        for path in (inputs, output, logs, cache):
            path.mkdir()
        config = ProjectConfig(
            input_sources=(InputSource(inputs, "image"),),
            output_root=output,
            log_dir=logs,
            cache_root=cache,
            manifest_path=logs / "manifest.sqlite3",
            runtime=RuntimeConfig(device="cpu"),
        )
        pipeline = root / "ocr.yaml"
        pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
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
        return config, inputs, output, profile

    def test_symlink_component_escaping_output_root_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output, profile = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            real_dir = root / "elsewhere"
            real_dir.mkdir()
            linked_dir = output / "linked"
            try:
                linked_dir.symlink_to(real_dir, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            target = linked_dir / "page_result.json"
            with ManifestStore(config.manifest_path) as store:
                store.mark_failure(
                    source,
                    "ocr",
                    RuntimeError("failed"),
                    intended_result_path=target,
                    execution_profile=profile,
                )

            plan = plan_failed_retries(config, stage="ocr")
            self.assertEqual(plan.eligible_count, 0)
            self.assertIn("outside configured output_root", plan.candidates[0].reason)

    def test_symlink_component_created_after_failure_is_rejected_even_if_it_stays_inside_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output, profile = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")

            # At failure time this is an ordinary, not-yet-created lexical path,
            # so the manifest stores /output/linked/page_result.json unchanged.
            linked_dir = output / "linked"
            target = linked_dir / "page_result.json"
            with ManifestStore(config.manifest_path) as store:
                record = store.mark_failure(
                    source,
                    "ocr",
                    RuntimeError("failed"),
                    intended_result_path=target,
                    execution_profile=profile,
                )
            self.assertEqual(record.intended_result_path, str(target.resolve(strict=False)))

            real_dir = output / "real"
            real_dir.mkdir()
            try:
                linked_dir.symlink_to(real_dir, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")

            plan = plan_failed_retries(config, stage="ocr")
            self.assertEqual(plan.eligible_count, 0)
            self.assertIn("symlink component", plan.candidates[0].reason)

    def test_manifest_cannot_redirect_retry_source_to_arbitrary_external_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, _, output, profile = self._project(root)
            external = root / "external"
            external.mkdir()
            source = external / "page.png"
            source.write_bytes(b"source")
            target = output / "ocr" / "page_result.json"

            with ManifestStore(config.manifest_path) as store:
                store.mark_failure(
                    source,
                    "ocr",
                    RuntimeError("failed"),
                    intended_result_path=target,
                    execution_profile=profile,
                )

            plan = plan_failed_retries(config, stage="ocr")
            self.assertEqual(plan.eligible_count, 0)
            self.assertIn(
                "outside configured input sources and output_root",
                plan.candidates[0].reason,
            )


if __name__ == "__main__":
    unittest.main()
