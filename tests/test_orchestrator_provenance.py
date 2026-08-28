import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import InputSource, ProjectConfig, RuntimeConfig
from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import OcrBatchResult, OcrTaskResult
from paddle_batch_ocr.orchestrator import run_project
from paddle_batch_ocr.pdf_render import RenderResult


class ProjectProvenanceTests(unittest.TestCase):
    def _config(self, root: Path, source: Path) -> ProjectConfig:
        config = ProjectConfig(
            input_sources=(InputSource(path=source, kind="pdf"),),
            output_root=root / "work" / "output",
            log_dir=root / "work" / "logs",
            cache_root=root / "work" / "cache",
            manifest_path=root / "work" / "logs" / "manifest.sqlite3",
            runtime=RuntimeConfig(ocr_workers=1, device="cpu"),
        )
        config.validate_paths()
        return config

    def test_render_failure_records_retry_target_and_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "book.pdf"
            source.write_bytes(b"%PDF-fake")
            config = self._config(root, source)

            def failing_render(*args, **kwargs):
                raise RuntimeError("renderer failed")

            result = run_project(
                config,
                dpi=222,
                render_fn=failing_render,
            )
            self.assertEqual(result.failed_count, 1)

            expected = config.output_root / "source-001" / "pdf" / "book" / "pages"
            with ManifestStore(config.manifest_path) as store:
                record = store.get_job(source, "render")

            self.assertIsNotNone(record)
            self.assertEqual(record.status, "failed")
            self.assertEqual(record.intended_result_path, str(expected.resolve()))
            self.assertEqual(record.execution_profile["kind"], "pdf_render")
            self.assertEqual(record.execution_profile["dpi"], 222)

    def test_searchable_failure_records_intermediate_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "book.pdf"
            source.write_bytes(b"%PDF-fake")
            config = self._config(root, source)

            def fake_render(pdf_path, output_dir, *, dpi, overwrite):
                output_dir.mkdir(parents=True, exist_ok=True)
                page = output_dir / "page_00001.png"
                page.write_bytes(b"image")
                return RenderResult(
                    source=pdf_path,
                    output_dir=output_dir,
                    page_paths=(page,),
                    page_count=1,
                    dpi=dpi,
                )

            def fake_ocr(input_path, output_dir, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                output = output_dir / "page_00001_result.json"
                output.write_text("{}", encoding="utf-8")
                return OcrBatchResult(
                    tasks=(
                        OcrTaskResult(
                            source=(input_path / "page_00001.png").resolve(),
                            output_json=output.resolve(),
                            status="success",
                        ),
                    )
                )

            def failing_searchable(*args, **kwargs):
                raise RuntimeError("pdf build failed")

            result = run_project(
                config,
                dpi=144,
                render_fn=fake_render,
                ocr_fn=fake_ocr,
                searchable_fn=failing_searchable,
            )
            self.assertEqual(result.failed_count, 1)

            artifact_root = config.output_root / "source-001" / "pdf" / "book"
            pages = artifact_root / "pages"
            ocr = artifact_root / "ocr"
            final_pdf = artifact_root / "searchable.pdf"

            with ManifestStore(config.manifest_path) as store:
                record = store.get_job(source, "searchable_pdf")

            self.assertIsNotNone(record)
            self.assertEqual(record.status, "failed")
            self.assertEqual(record.intended_result_path, str(final_pdf.resolve()))
            profile = record.execution_profile
            self.assertEqual(profile["schema"], 2)
            self.assertEqual(profile["kind"], "searchable_pdf")
            self.assertEqual(profile["images_dir"], str(pages.resolve()))
            self.assertEqual(profile["ocr_json_dir"], str(ocr.resolve()))
            self.assertEqual(profile["expected_page_count"], 1)
            self.assertEqual(profile["layout"], "legacy-v7")


if __name__ == "__main__":
    unittest.main()
