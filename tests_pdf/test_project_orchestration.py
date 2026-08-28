import json
import tempfile
import unittest
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from paddle_batch_ocr.config import InputSource, ProjectConfig, RuntimeConfig
from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import OcrBatchResult, OcrTaskResult
from paddle_batch_ocr.orchestrator import run_project


class RealProjectOrchestrationTests(unittest.TestCase):
    def _make_pdf(self, path: Path, pages: int = 2) -> None:
        document = fitz.open()
        try:
            for index in range(pages):
                page = document.new_page(width=240, height=120)
                page.insert_text(
                    (20, 60),
                    f"source page {index + 1}",
                    fontname="helv",
                    fontsize=12,
                )
            document.save(str(path))
        finally:
            document.close()

    def test_real_render_and_searchable_with_fake_ocr(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = root / "inputs"
            inputs.mkdir()
            source = inputs / "book.pdf"
            self._make_pdf(source, pages=2)

            config = ProjectConfig(
                input_sources=(InputSource(path=inputs, kind="pdf"),),
                output_root=root / "work" / "output",
                log_dir=root / "work" / "logs",
                cache_root=root / "work" / "cache",
                manifest_path=root / "work" / "logs" / "manifest.sqlite3",
                runtime=RuntimeConfig(ocr_workers=1, device="cpu"),
            )
            config.validate_paths()
            texts = ["项目文本一", "项目文本二"]

            def fake_ocr(input_path, output_dir, **kwargs):
                output_dir.mkdir(parents=True, exist_ok=True)
                tasks = []

                for index, image_path in enumerate(
                    sorted(input_path.glob("page_*.png"))
                ):
                    output_json = output_dir / f"{image_path.stem}_result.json"
                    output_json.write_text(
                        json.dumps(
                            {
                                "rec_polys": [
                                    [[20, 20], [210, 20], [210, 50], [20, 50]]
                                ],
                                "rec_texts": [texts[index]],
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    tasks.append(
                        OcrTaskResult(
                            source=image_path.resolve(),
                            output_json=output_json.resolve(),
                            status="success",
                        )
                    )

                return OcrBatchResult(tasks=tuple(tasks))

            result = run_project(config, dpi=72, ocr_fn=fake_ocr)
            self.assertEqual(result.success_count, 1)
            self.assertEqual(result.failed_count, 0)

            item = result.items[0]
            expected_root = config.output_root / "source-001" / "pdf" / "book"
            self.assertEqual(item.pages_dir, expected_root / "pages")
            self.assertEqual(item.ocr_dir, expected_root / "ocr")
            self.assertEqual(item.searchable_pdf, expected_root / "searchable.pdf")
            self.assertTrue(item.searchable_pdf.is_file())

            with fitz.open(str(item.searchable_pdf)) as document:
                self.assertEqual(document.page_count, 2)
                self.assertIn(texts[0], document.load_page(0).get_text())
                self.assertIn(texts[1], document.load_page(1).get_text())

            with ManifestStore(config.manifest_path) as store:
                render_record = store.get_job(source, "render")
                searchable_record = store.get_job(source, "searchable_pdf")

            self.assertEqual(render_record.status, "success")
            self.assertEqual(searchable_record.status, "success")


if __name__ == "__main__":
    unittest.main()
