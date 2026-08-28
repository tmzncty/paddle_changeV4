import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.config import InputSource, ProjectConfig, RuntimeConfig
from paddle_batch_ocr.ocr_runner import OcrBatchResult, OcrTaskResult
from paddle_batch_ocr.orchestrator import run_project
from paddle_batch_ocr.pdf_render import RenderResult
from paddle_batch_ocr.searchable_pdf import SearchablePdfResult

class ProjectOrchestratorTests(unittest.TestCase):
    def test_pdf_pipeline_uses_configured_workers_and_stable_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); inputs = root / "inputs"; inputs.mkdir(); pdf = inputs / "book.pdf"; pdf.write_bytes(b"%PDF-fake")
            config = ProjectConfig(input_sources=(InputSource(path=inputs, kind="pdf"),), output_root=root / "work" / "output", log_dir=root / "work" / "logs", cache_root=root / "work" / "cache", manifest_path=root / "work" / "logs" / "manifest.sqlite3", runtime=RuntimeConfig(ocr_workers=3, device="cpu")); config.validate_paths(); seen = {}
            def fake_render(source, output_dir, *, dpi, overwrite):
                output_dir.mkdir(parents=True, exist_ok=True); page = output_dir / "page_00001.png"; page.write_bytes(b"image")
                return RenderResult(source, output_dir, (page,), 1, dpi)
            def fake_ocr(input_path, output_dir, **kwargs):
                seen["workers"] = kwargs["workers"]; seen["device"] = kwargs["device"]; output_dir.mkdir(parents=True, exist_ok=True); result_path = output_dir / "page_00001_result.json"; result_path.write_text('{"rec_polys": [[[0,0],[1,0],[1,1],[0,1]]], "rec_texts": ["x"]}', encoding="utf-8")
                return OcrBatchResult(tasks=(OcrTaskResult((input_path / "page_00001.png").resolve(), result_path.resolve(), "success"),))
            def fake_searchable(images, ocr_json, output_pdf, *, overwrite):
                output_pdf.parent.mkdir(parents=True, exist_ok=True); output_pdf.write_bytes(b"pdf")
                return SearchablePdfResult(output_pdf, 1, 1, (images / "page_00001.png",), (ocr_json / "page_00001_result.json",))
            result = run_project(config, dpi=200, render_fn=fake_render, ocr_fn=fake_ocr, searchable_fn=fake_searchable)
            self.assertEqual(result.success_count, 1); self.assertEqual(result.failed_count, 0); self.assertEqual(seen, {"workers": 3, "device": "cpu"})
            item = result.items[0]; expected_root = config.output_root / "source-001" / "pdf" / "book"
            self.assertEqual(item.pages_dir, expected_root / "pages"); self.assertEqual(item.ocr_dir, expected_root / "ocr"); self.assertEqual(item.searchable_pdf, expected_root / "searchable.pdf")
    def test_image_source_runs_ocr_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); image = root / "scan.png"; image.write_bytes(b"x")
            config = ProjectConfig(input_sources=(InputSource(path=image, kind="image"),), output_root=root / "work" / "output", log_dir=root / "work" / "logs", cache_root=root / "work" / "cache", manifest_path=root / "work" / "logs" / "manifest.sqlite3", runtime=RuntimeConfig(ocr_workers=1, device="cpu")); config.validate_paths()
            def fake_ocr(input_path, output_dir, **kwargs):
                return OcrBatchResult(tasks=(OcrTaskResult(input_path.resolve(), (output_dir / "scan_result.json").resolve(), "success"),))
            result = run_project(config, ocr_fn=fake_ocr); self.assertEqual(result.success_count, 1); item = result.items[0]
            self.assertEqual(item.kind, "image"); self.assertIsNone(item.pages_dir); self.assertIsNone(item.searchable_pdf); self.assertEqual(item.ocr_dir, config.output_root / "source-001" / "image" / "ocr")

if __name__ == "__main__": unittest.main()
