import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import InputSource, ProjectConfig, RuntimeConfig
from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import build_ocr_execution_profile
from paddle_batch_ocr.retry import plan_failed_retries


class RetryPathSafetyTests(unittest.TestCase):
    @unittest.skipIf(not hasattr(Path, "symlink_to"), "symlinks unavailable")
    def test_symlink_component_under_output_root_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
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
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
            real_dir = root / "elsewhere"
            real_dir.mkdir()
            linked_dir = output / "linked"
            try:
                linked_dir.symlink_to(real_dir, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            target = linked_dir / "page_result.json"
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
                    RuntimeError("failed"),
                    intended_result_path=target,
                    execution_profile=profile,
                )

            plan = plan_failed_retries(config, stage="ocr")
            self.assertEqual(plan.eligible_count, 0)
            self.assertRegex(plan.candidates[0].reason, "outside configured output_root|symlink")


if __name__ == "__main__":
    unittest.main()
