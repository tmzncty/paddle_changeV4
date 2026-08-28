import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.ocr_runner import run_ocr_batch


class OcrPipelineInitializationTests(unittest.TestCase):
    def test_failed_pipeline_initialization_is_attempted_once_per_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            source.mkdir()
            for name in ("a.png", "b.png", "c.png"):
                (source / name).write_bytes(b"x")

            calls = []

            def broken_factory(**kwargs):
                calls.append(kwargs)
                raise RuntimeError("model runtime unavailable")

            result = run_ocr_batch(
                source,
                root / "json",
                device="cpu",
                create_pipeline_fn=broken_factory,
            )

            self.assertEqual(len(calls), 1)
            self.assertEqual(result.success_count, 0)
            self.assertEqual(result.failed_count, 3)
            self.assertIn("model runtime unavailable", result.tasks[0].error)
            self.assertIn("previously failed", result.tasks[1].error)
            self.assertIn("previously failed", result.tasks[2].error)


if __name__ == "__main__":
    unittest.main()
