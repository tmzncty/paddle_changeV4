import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import InputSource, ProjectConfig, RuntimeConfig
from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import build_ocr_execution_profile
from paddle_batch_ocr.pdf_render import RenderResult
from paddle_batch_ocr.retry import execute_retry_plan, plan_failed_retries
from paddle_batch_ocr.searchable_pdf import SearchablePdfResult


class FakeOcrPipeline:
    def predict_iter(self, *, input, **kwargs):
        del input, kwargs
        yield {
            "rec_polys": [
                [[0, 0], [30, 0], [30, 12], [0, 12]],
            ],
            "rec_texts": ["retry ok"],
        }


def fake_create_pipeline(*, pipeline, **kwargs):
    del pipeline, kwargs
    return FakeOcrPipeline()


class TargetedRetryTests(unittest.TestCase):
    def _project(self, root: Path):
        inputs = root / "inputs"
        output = root / "output"
        logs = root / "logs"
        cache = root / "cache"
        inputs.mkdir()
        output.mkdir()
        logs.mkdir()
        cache.mkdir()
        config = ProjectConfig(
            input_sources=(InputSource(inputs, "image"),),
            output_root=output,
            log_dir=logs,
            cache_root=cache,
            manifest_path=logs / "manifest.sqlite3",
            runtime=RuntimeConfig(device="cpu"),
        )
        return config, inputs, output

    def _ocr_failure(self, config, source: Path, target: Path, pipeline: Path):
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
            store.mark_started(
                source,
                "ocr",
                worker="test",
                device="cpu",
                intended_result_path=target,
                execution_profile=profile,
            )
            store.mark_failure(
                source,
                "ocr",
                RuntimeError("synthetic failure"),
                intended_result_path=target,
                execution_profile=profile,
            )
        return profile

    def test_plan_is_read_only_and_eligible_with_matching_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
            target = output / "ocr" / "page_result.json"
            self._ocr_failure(config, source, target, pipeline)

            before = config.manifest_path.read_bytes()
            plan = plan_failed_retries(config, stage="ocr")
            after = config.manifest_path.read_bytes()

            self.assertEqual(before, after)
            self.assertEqual(plan.total_matching, 1)
            self.assertEqual(plan.eligible_count, 1)
            self.assertTrue(plan.candidates[0].eligible)
            self.assertEqual(plan.candidates[0].intended_result, target.resolve())
            self.assertFalse(target.exists())

    def test_named_pipeline_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            target = output / "ocr" / "page_result.json"
            profile = build_ocr_execution_profile(
                pipeline_ref="OCR",
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
            self.assertIn("local pipeline file", plan.candidates[0].reason)

    def test_pipeline_hash_drift_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("one\n", encoding="utf-8")
            target = output / "ocr" / "page_result.json"
            self._ocr_failure(config, source, target, pipeline)

            pipeline.write_text("two but changed\n", encoding="utf-8")
            plan = plan_failed_retries(config, stage="ocr")
            self.assertEqual(plan.eligible_count, 0)
            self.assertRegex(plan.candidates[0].reason, "size|SHA-256")

    def test_source_drift_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
            target = output / "ocr" / "page_result.json"
            self._ocr_failure(config, source, target, pipeline)

            source.write_bytes(b"changed source")
            plan = plan_failed_retries(config, stage="ocr")
            self.assertEqual(plan.eligible_count, 0)
            self.assertIn("source size changed", plan.candidates[0].reason)

    def test_target_outside_output_root_is_ineligible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, _ = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
            target = root / "outside" / "page_result.json"
            self._ocr_failure(config, source, target, pipeline)

            plan = plan_failed_retries(config, stage="ocr")
            self.assertEqual(plan.eligible_count, 0)
            self.assertIn("outside configured output_root", plan.candidates[0].reason)

    def test_existing_target_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
            target = output / "ocr" / "page_result.json"
            target.parent.mkdir(parents=True)
            target.write_text("old", encoding="utf-8")
            self._ocr_failure(config, source, target, pipeline)

            blocked = plan_failed_retries(config, stage="ocr")
            self.assertFalse(blocked.candidates[0].eligible)
            self.assertIn("target already exists", blocked.candidates[0].reason)

            allowed = plan_failed_retries(config, stage="ocr", overwrite=True)
            self.assertTrue(allowed.candidates[0].eligible)

    def test_execute_ocr_retry_repairs_manifest_and_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
            target = output / "ocr" / "page_result.json"
            self._ocr_failure(config, source, target, pipeline)

            plan = plan_failed_retries(config, stage="ocr")
            result = execute_retry_plan(
                config,
                plan,
                create_pipeline_fn=fake_create_pipeline,
            )

            self.assertEqual(result.success_count, 1)
            self.assertEqual(result.failed_count, 0)
            self.assertTrue(target.is_file())
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["rec_texts"], ["retry ok"])

            with ManifestStore(config.manifest_path) as store:
                record = store.get_job(source, "ocr")
            self.assertEqual(record.status, "success")
            self.assertEqual(record.result_path, str(target.resolve()))
            self.assertEqual(record.intended_result_path, str(target.resolve()))

    def test_execute_revalidates_manifest_after_planning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "page.png"
            source.write_bytes(b"source")
            pipeline = root / "ocr.yaml"
            pipeline.write_text("pipeline_name: OCR\n", encoding="utf-8")
            target = output / "ocr" / "page_result.json"
            self._ocr_failure(config, source, target, pipeline)
            plan = plan_failed_retries(config, stage="ocr")

            with ManifestStore(config.manifest_path) as store:
                store.mark_success(source, "ocr", result_path=output / "other.json")

            result = execute_retry_plan(
                config,
                plan,
                create_pipeline_fn=fake_create_pipeline,
            )
            self.assertEqual(result.success_count, 0)
            self.assertEqual(result.failed_count, 1)
            self.assertIn("no longer failed", result.items[0].error)
            self.assertFalse(target.exists())

    def test_render_retry_uses_recorded_dpi(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "doc.pdf"
            source.write_bytes(b"pdf")
            target = output / "doc" / "pages"
            profile = {
                "schema": 1,
                "kind": "pdf_render",
                "dpi": 216,
                "format": "png",
                "alpha": False,
            }
            with ManifestStore(config.manifest_path) as store:
                store.mark_failure(
                    source,
                    "render",
                    RuntimeError("render failed"),
                    intended_result_path=target,
                    execution_profile=profile,
                )

            called = {}

            def fake_render(pdf_path, output_dir, *, dpi, overwrite):
                called.update(pdf_path=pdf_path, output_dir=output_dir, dpi=dpi, overwrite=overwrite)
                output_dir.mkdir(parents=True)
                page = output_dir / "page_00001.png"
                page.write_bytes(b"png")
                return RenderResult(pdf_path, output_dir, (page,), 1, dpi)

            plan = plan_failed_retries(config, stage="render")
            result = execute_retry_plan(config, plan, render_fn=fake_render)
            self.assertEqual(result.success_count, 1)
            self.assertEqual(called["dpi"], 216)

    def test_searchable_retry_uses_recorded_intermediate_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, inputs, output = self._project(root)
            source = inputs / "doc.pdf"
            source.write_bytes(b"pdf")
            artifact = output / "doc"
            pages = artifact / "pages"
            ocr = artifact / "ocr"
            target = artifact / "searchable.pdf"
            pages.mkdir(parents=True)
            ocr.mkdir(parents=True)
            image = pages / "page_00001.png"
            image.write_bytes(b"png")
            json_path = ocr / "page_00001.json"
            json_path.write_text(
                json.dumps(
                    {
                        "rec_polys": [
                            [[0, 0], [30, 0], [30, 12], [0, 12]],
                        ],
                        "rec_texts": ["hello"],
                    }
                ),
                encoding="utf-8",
            )
            profile = {
                "schema": 2,
                "kind": "searchable_pdf",
                "fontname": "china-s",
                "y_offset": 0.0,
                "layout": "legacy-v7",
                "images_dir": str(pages.resolve()),
                "ocr_json_dir": str(ocr.resolve()),
                "expected_page_count": 1,
            }
            with ManifestStore(config.manifest_path) as store:
                store.mark_failure(
                    source,
                    "searchable_pdf",
                    RuntimeError("pdf failed"),
                    intended_result_path=target,
                    execution_profile=profile,
                )

            called = {}

            def fake_searchable(images_dir, json_dir, output_pdf, *, overwrite, y_offset, fontname):
                called.update(
                    images_dir=images_dir,
                    json_dir=json_dir,
                    output_pdf=output_pdf,
                    overwrite=overwrite,
                    y_offset=y_offset,
                    fontname=fontname,
                )
                output_pdf.parent.mkdir(parents=True, exist_ok=True)
                output_pdf.write_bytes(b"pdf")
                return SearchablePdfResult(output_pdf, 1, 1, (image,), (json_path,))

            plan = plan_failed_retries(config, stage="searchable_pdf")
            result = execute_retry_plan(config, plan, searchable_fn=fake_searchable)
            self.assertEqual(result.success_count, 1)
            self.assertEqual(called["images_dir"], pages.resolve())
            self.assertEqual(called["json_dir"], ocr.resolve())
            self.assertEqual(called["fontname"], "china-s")


if __name__ == "__main__":
    unittest.main()
