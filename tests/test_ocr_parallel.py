import json
import os
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import OcrRunnerError, discover_ocr_tasks, run_ocr_batch

class FakePipeline:
    def predict_iter(self, input, **kwargs):
        stem = Path(input).stem
        yield {"rec_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]], "rec_texts": [stem]}

def fake_pipeline_factory(*, pipeline, **kwargs):
    log_path = os.environ.get("PADDLE_BATCH_OCR_TEST_FACTORY_LOG")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
    return FakePipeline()

class ParallelOcrTests(unittest.TestCase):
    def test_spawn_workers_reuse_one_pipeline_per_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); images = root / "images"; output = root / "json"; manifest = root / "manifest.sqlite3"; factory_log = root / "factory.log"
            images.mkdir()
            for index in range(8):
                (images / f"page_{index:05d}.png").write_bytes(b"x")
            old = os.environ.get("PADDLE_BATCH_OCR_TEST_FACTORY_LOG"); os.environ["PADDLE_BATCH_OCR_TEST_FACTORY_LOG"] = str(factory_log)
            try:
                result = run_ocr_batch(images, output, pipeline_ref="fake", device="cpu", workers=2, manifest_path=manifest, resume=False, create_pipeline_fn=fake_pipeline_factory)
            finally:
                if old is None: os.environ.pop("PADDLE_BATCH_OCR_TEST_FACTORY_LOG", None)
                else: os.environ["PADDLE_BATCH_OCR_TEST_FACTORY_LOG"] = old
            self.assertEqual(result.success_count, 8); self.assertEqual(result.failed_count, 0)
            pids = factory_log.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(set(pids)), 1); self.assertLessEqual(len(set(pids)), 2); self.assertEqual(len(pids), len(set(pids)))
            for index in range(8):
                payload = json.loads((output / f"page_{index:05d}_result.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["rec_texts"], [f"page_{index:05d}"])
            with ManifestStore(manifest) as store:
                self.assertEqual(store.summary().get("success"), 8)
    def test_parallel_requires_explicit_cpu_device(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); source = root / "page.png"; source.write_bytes(b"x")
            with self.assertRaisesRegex(OcrRunnerError, "device='cpu'"):
                run_ocr_batch(source, root / "json", device="auto", workers=2, create_pipeline_fn=fake_pipeline_factory)
    def test_output_name_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir); images = root / "images"; images.mkdir(); (images / "same.png").write_bytes(b"x"); (images / "same.jpg").write_bytes(b"x")
            with self.assertRaisesRegex(OcrRunnerError, "same output JSON"):
                discover_ocr_tasks(images, root / "json")

if __name__ == "__main__": unittest.main()
