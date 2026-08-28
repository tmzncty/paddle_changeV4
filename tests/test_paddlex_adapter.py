import json
import tempfile
import unittest
from pathlib import Path

from paddle_batch_ocr.paddlex_adapter import (
    PaddleXResultError,
    create_ocr_pipeline,
    create_pipeline_kwargs,
    normalize_ocr_result,
    parse_paddlex_ocr_result,
    predict_one,
    predict_one_to_json,
    result_mapping,
)


MODERN_RESULT = {
    "res": {
        "input_path": "page.png",
        "dt_polys": [
            [[0, 0], [10, 0], [10, 5], [0, 5]],
            [[20, 0], [30, 0], [30, 5], [20, 5]],
        ],
        "rec_polys": [
            [[20, 0], [30, 0], [30, 5], [20, 5]],
        ],
        "rec_texts": ["保留"],
        "rec_scores": [0.99],
    }
}


class FakeResult:
    def __init__(self, payload):
        self.json = payload


class FakePipeline:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict_iter(self, *, input):
        self.calls.append(input)
        return iter(self.results)


class PaddleXAdapterTests(unittest.TestCase):
    def test_unwraps_documented_res_envelope(self):
        mapping = result_mapping(FakeResult(MODERN_RESULT))
        self.assertEqual(mapping["rec_texts"], ["保留"])
        self.assertNotIn("res", mapping)

    def test_modern_result_uses_rec_polys_pairing(self):
        page = parse_paddlex_ocr_result(FakeResult(MODERN_RESULT))
        self.assertEqual(page.polygon_field, "rec_polys")
        self.assertEqual(page.lines[0].text, "保留")
        self.assertEqual(page.lines[0].polygon[0], (20.0, 0.0))

    def test_normalized_result_preserves_raw_fields_and_adds_provenance(self):
        normalized = normalize_ocr_result(FakeResult(MODERN_RESULT))
        self.assertEqual(normalized["rec_scores"], [0.99])
        self.assertEqual(
            normalized["_paddle_batch_ocr"],
            {
                "schema": 1,
                "polygon_field": "rec_polys",
                "text_field": "rec_texts",
            },
        )

    def test_create_pipeline_kwargs_omits_auto_and_none(self):
        self.assertEqual(create_pipeline_kwargs(device="auto"), {})
        self.assertEqual(
            create_pipeline_kwargs(
                device="gpu:2",
                engine="hpi",
                use_hpip=True,
                hpi_config={"backend": "trt"},
            ),
            {
                "device": "gpu:2",
                "engine": "hpi",
                "use_hpip": True,
                "hpi_config": {"backend": "trt"},
            },
        )

    def test_create_pipeline_uses_current_keyword_contract(self):
        calls = []

        def fake_create_pipeline(**kwargs):
            calls.append(kwargs)
            return "pipeline"

        pipeline = create_ocr_pipeline(
            "OCR",
            device="cpu",
            engine="paddle",
            create_pipeline_fn=fake_create_pipeline,
        )
        self.assertEqual(pipeline, "pipeline")
        self.assertEqual(
            calls,
            [{"pipeline": "OCR", "device": "cpu", "engine": "paddle"}],
        )

    def test_predict_one_prefers_predict_iter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "page.png"
            image.write_bytes(b"fake")
            expected = FakeResult(MODERN_RESULT)
            pipeline = FakePipeline([expected])

            result = predict_one(pipeline, image)

            self.assertIs(result, expected)
            self.assertEqual(pipeline.calls, [str(image.resolve())])

    def test_predict_one_rejects_zero_or_multiple_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "page.png"
            image.write_bytes(b"fake")
            with self.assertRaises(PaddleXResultError):
                predict_one(FakePipeline([]), image)
            with self.assertRaises(PaddleXResultError):
                predict_one(
                    FakePipeline([FakeResult(MODERN_RESULT), FakeResult(MODERN_RESULT)]),
                    image,
                )

    def test_predict_one_to_json_validates_and_atomically_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "page.png"
            output = root / "page.json"
            image.write_bytes(b"fake")

            predict_one_to_json(
                FakePipeline([FakeResult(MODERN_RESULT)]),
                image,
                output,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["rec_texts"], ["保留"])
            self.assertEqual(payload["_paddle_batch_ocr"]["polygon_field"], "rec_polys")


if __name__ == "__main__":
    unittest.main()
