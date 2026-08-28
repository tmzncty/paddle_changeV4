import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import (
    OcrRunnerError,
    discover_ocr_tasks,
    run_ocr_batch,
)


class FakeResult:
    def __init__(self, text):
        self.json = {
            "res": {
                "dt_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
                "rec_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
                "rec_texts": [text],
            }
        }


class RecordingPipeline:
    def __init__(self, fail_name=None):
        self.calls = []
        self.fail_name = fail_name

    def predict_iter(self, *, input):
        path = Path(input)
        self.calls.append(path.name)
        if path.name == self.fail_name:
            return iter([])
        return iter([FakeResult(path.stem)])


class OcrRunnerTests(unittest.TestCase):
    def test_discovery_preserves_relative_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            nested = source / "a" / "b"
            nested.mkdir(parents=True)
            (source / "page_1.png").write_bytes(b"x")
            (nested / "page_2.JPG").write_bytes(b"x")
            (nested / "ignore.txt").write_text("x", encoding="utf-8")

            tasks = discover_ocr_tasks(source, root / "json")

            self.assertEqual([task.source.name for task in tasks], ["page_1.png", "page_2.JPG"])
            self.assertEqual(
                [task.output_json.relative_to(root / "json").as_posix() for task in tasks],
                ["page_1_result.json", "a/b/page_2_result.json"],
            )

    def test_rejects_output_inside_input_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            source.mkdir()
            (source / "page.png").write_bytes(b"x")
            with self.assertRaises(OcrRunnerError):
                discover_ocr_tasks(source, source / "json")

    def test_pipeline_is_created_once_for_multiple_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            source.mkdir()
            for name in ("a.png", "b.png"):
                (source / name).write_bytes(b"x")

            pipeline = RecordingPipeline()
            factory_calls = []

            def factory(**kwargs):
                factory_calls.append(kwargs)
                return pipeline

            result = run_ocr_batch(
                source,
                root / "json",
                create_pipeline_fn=factory,
                device="cpu",
                engine="paddle",
            )

            self.assertEqual(result.success_count, 2)
            self.assertEqual(result.failed_count, 0)
            self.assertEqual(len(factory_calls), 1)
            self.assertEqual(pipeline.calls, ["a.png", "b.png"])
            self.assertTrue((root / "json" / "a_result.json").is_file())
            self.assertTrue((root / "json" / "b_result.json").is_file())

    def test_full_resume_does_not_initialize_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            output = root / "json"
            source.mkdir()
            output.mkdir()
            image = source / "a.png"
            image.write_bytes(b"x")
            (output / "a_result.json").write_text(
                json.dumps(
                    {
                        "dt_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                        "rec_texts": ["existing"],
                    }
                ),
                encoding="utf-8",
            )

            def forbidden_factory(**kwargs):
                raise AssertionError("pipeline should not be initialized during full resume")

            result = run_ocr_batch(
                source,
                output,
                create_pipeline_fn=forbidden_factory,
            )

            self.assertEqual(result.skipped_count, 1)
            self.assertEqual(result.failed_count, 0)

    def test_manifest_can_adopt_existing_valid_result_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            output = root / "json"
            manifest = root / "manifest.sqlite3"
            source.mkdir()
            output.mkdir()
            image = source / "a.png"
            image.write_bytes(b"x")
            result_path = output / "a_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "dt_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                        "rec_texts": ["existing"],
                    }
                ),
                encoding="utf-8",
            )

            result = run_ocr_batch(
                source,
                output,
                manifest_path=manifest,
                create_pipeline_fn=lambda **kwargs: (_ for _ in ()).throw(
                    AssertionError("pipeline should not load")
                ),
            )

            self.assertEqual(result.skipped_count, 1)
            with ManifestStore(manifest) as store:
                record = store.get_job(image, "ocr")
                self.assertIsNotNone(record)
                self.assertEqual(record.status, "success")
                self.assertEqual(Path(record.result_path), result_path.resolve())

    def test_manifest_source_change_marks_existing_result_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            output = root / "json"
            manifest = root / "manifest.sqlite3"
            source.mkdir()
            image = source / "a.png"
            image.write_bytes(b"first")

            first_pipeline = RecordingPipeline()
            first = run_ocr_batch(
                source,
                output,
                manifest_path=manifest,
                create_pipeline_fn=lambda **kwargs: first_pipeline,
            )
            self.assertEqual(first.success_count, 1)

            image.write_bytes(b"changed-size")
            second = run_ocr_batch(
                source,
                output,
                manifest_path=manifest,
                create_pipeline_fn=lambda **kwargs: RecordingPipeline(),
                overwrite=False,
            )

            self.assertEqual(second.failed_count, 1)
            self.assertIn("stale", second.tasks[0].error)

    def test_failures_are_isolated_per_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "images"
            source.mkdir()
            for name in ("bad.png", "good.png"):
                (source / name).write_bytes(b"x")
            pipeline = RecordingPipeline(fail_name="bad.png")

            result = run_ocr_batch(
                source,
                root / "json",
                create_pipeline_fn=lambda **kwargs: pipeline,
            )

            self.assertEqual(result.success_count, 1)
            self.assertEqual(result.failed_count, 1)
            self.assertTrue((root / "json" / "good_result.json").is_file())
            self.assertFalse((root / "json" / "bad_result.json").exists())


if __name__ == "__main__":
    unittest.main()
