import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.manifest import ManifestStore
from paddle_batch_ocr.ocr_runner import (
    build_ocr_execution_profile,
    run_ocr_batch,
)


class FakePipeline:
    def predict_iter(self, input, **kwargs):
        yield {
            "rec_polys": [
                [[0, 0], [20, 0], [20, 10], [0, 10]],
            ],
            "rec_texts": [Path(input).stem],
        }


def fake_pipeline_factory(*, pipeline, **kwargs):
    return FakePipeline()


def failing_pipeline_factory(*, pipeline, **kwargs):
    raise AssertionError("pipeline must not be initialized during adoption")


class OcrExecutionProfileTests(unittest.TestCase):
    def test_named_pipeline_identity_is_stable(self):
        profile = build_ocr_execution_profile(
            pipeline_ref="OCR",
            device="cpu",
            engine=None,
            use_hpip=None,
            predict_kwargs={"use_doc_unwarping": False},
        )

        self.assertEqual(profile["schema"], 2)
        self.assertEqual(profile["kind"], "paddlex_ocr")
        self.assertEqual(
            profile["pipeline"],
            {"type": "name", "value": "OCR"},
        )

    def test_local_pipeline_profile_hashes_file_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Path(temp_dir) / "ocr.yaml"
            pipeline.write_text("model: A\n", encoding="utf-8")

            first = build_ocr_execution_profile(
                pipeline_ref=str(pipeline),
                device="cpu",
                engine=None,
                use_hpip=None,
                predict_kwargs={},
            )
            same = build_ocr_execution_profile(
                pipeline_ref=str(pipeline),
                device="cpu",
                engine=None,
                use_hpip=None,
                predict_kwargs={},
            )

            pipeline.write_text("model: B\n", encoding="utf-8")
            second = build_ocr_execution_profile(
                pipeline_ref=str(pipeline),
                device="cpu",
                engine=None,
                use_hpip=None,
                predict_kwargs={},
            )

            self.assertEqual(first, same)
            self.assertEqual(first["pipeline"]["type"], "file")
            self.assertEqual(first["pipeline"]["path"], str(pipeline.resolve()))
            self.assertEqual(
                first["pipeline"]["sha256"],
                hashlib.sha256(b"model: A\n").hexdigest(),
            )
            self.assertEqual(first["pipeline"]["size"], second["pipeline"]["size"])
            self.assertNotEqual(
                first["pipeline"]["sha256"],
                second["pipeline"]["sha256"],
            )

    def test_serial_ocr_persists_target_and_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            output_dir = root / "json"
            manifest = root / "manifest.sqlite3"
            pipeline = root / "ocr.yaml"
            source.write_bytes(b"x")
            pipeline.write_text("pipeline: OCR\n", encoding="utf-8")

            result = run_ocr_batch(
                source,
                output_dir,
                pipeline_ref=str(pipeline),
                device="cpu",
                manifest_path=manifest,
                create_pipeline_fn=fake_pipeline_factory,
            )
            self.assertEqual(result.success_count, 1)

            output = output_dir / "page_result.json"
            with ManifestStore(manifest) as store:
                record = store.get_job(source, "ocr")

            self.assertIsNotNone(record)
            self.assertEqual(record.status, "success")
            self.assertEqual(record.result_path, str(output.resolve()))
            self.assertEqual(record.intended_result_path, str(output.resolve()))
            self.assertEqual(record.execution_profile["device"], "cpu")
            self.assertEqual(record.execution_profile["pipeline"]["type"], "file")
            self.assertEqual(
                record.execution_profile["pipeline"]["sha256"],
                hashlib.sha256(b"pipeline: OCR\n").hexdigest(),
            )

    def test_first_time_adoption_keeps_historical_profile_unknown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            output_dir = root / "json"
            output_dir.mkdir()
            output = output_dir / "page_result.json"
            manifest = root / "manifest.sqlite3"
            pipeline = root / "ocr.yaml"
            source.write_bytes(b"x")
            pipeline.write_text("pipeline: current\n", encoding="utf-8")
            output.write_text(
                json.dumps(
                    {
                        "rec_polys": [
                            [[0, 0], [20, 0], [20, 10], [0, 10]],
                        ],
                        "rec_texts": ["historical"],
                    }
                ),
                encoding="utf-8",
            )

            result = run_ocr_batch(
                source,
                output_dir,
                pipeline_ref=str(pipeline),
                device="cpu",
                manifest_path=manifest,
                resume=True,
                create_pipeline_fn=failing_pipeline_factory,
            )
            self.assertEqual(result.skipped_count, 1)

            with ManifestStore(manifest) as store:
                record = store.get_job(source, "ocr")

            self.assertEqual(record.intended_result_path, str(output.resolve()))
            self.assertIsNone(record.execution_profile_json)

    def test_pipeline_content_change_invalidates_known_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "page.png"
            output_dir = root / "json"
            manifest = root / "manifest.sqlite3"
            pipeline = root / "ocr.yaml"
            source.write_bytes(b"x")
            pipeline.write_text("model: A\n", encoding="utf-8")

            first = run_ocr_batch(
                source,
                output_dir,
                pipeline_ref=str(pipeline),
                device="cpu",
                manifest_path=manifest,
                create_pipeline_fn=fake_pipeline_factory,
            )
            self.assertEqual(first.success_count, 1)

            pipeline.write_text("model: B\n", encoding="utf-8")
            second = run_ocr_batch(
                source,
                output_dir,
                pipeline_ref=str(pipeline),
                device="cpu",
                manifest_path=manifest,
                resume=True,
                overwrite=False,
                create_pipeline_fn=fake_pipeline_factory,
            )
            self.assertEqual(second.failed_count, 1)
            self.assertIn("stale according to manifest", second.tasks[0].error)

            with ManifestStore(manifest) as store:
                record = store.get_job(source, "ocr")

            self.assertEqual(record.status, "failed")
            self.assertEqual(record.intended_result_path, str((output_dir / "page_result.json").resolve()))
            self.assertEqual(
                record.execution_profile["pipeline"]["sha256"],
                hashlib.sha256(b"model: B\n").hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
