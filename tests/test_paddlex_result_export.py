import unittest

from paddle_batch_ocr.paddlex_adapter import normalize_ocr_result, result_mapping


class RuntimeFont:
    pass


class MappingLikePaddleXResult(dict):
    """Stand-in for PaddleX OCRResult: Mapping runtime state + curated .json."""

    def __init__(self):
        super().__init__(
            {
                "input_path": "page.png",
                "rec_texts": ["中文"],
                "rec_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                "vis_fonts": [RuntimeFont()],
                "doc_preprocessor_res": {"output_img": object()},
            }
        )
        self._export = {
            "res": {
                "input_path": "page.png",
                "page_index": None,
                "model_settings": {
                    "use_doc_preprocessor": False,
                    "use_textline_orientation": False,
                },
                "dt_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                "text_det_params": {"limit_side_len": 64},
                "text_type": "general",
                "text_rec_score_thresh": 0.0,
                "return_word_box": False,
                "rec_texts": ["中文"],
                "rec_scores": [0.99],
                "rec_polys": [[[0, 0], [10, 0], [10, 5], [0, 5]]],
                "rec_boxes": [[0, 0, 10, 5]],
            }
        }

    @property
    def json(self):
        return self._export


class PaddleXResultExportTests(unittest.TestCase):
    def test_result_mapping_prefers_documented_json_export(self):
        result = MappingLikePaddleXResult()

        mapping = result_mapping(result)

        self.assertEqual(mapping["rec_texts"], ["中文"])
        self.assertNotIn("vis_fonts", mapping)
        self.assertNotIn("doc_preprocessor_res", mapping)

    def test_normalization_does_not_serialize_runtime_font_or_image_state(self):
        normalized = normalize_ocr_result(MappingLikePaddleXResult())

        self.assertNotIn("vis_fonts", normalized)
        self.assertNotIn("doc_preprocessor_res", normalized)
        self.assertEqual(normalized["rec_texts"], ["中文"])
        self.assertEqual(
            normalized["_paddle_batch_ocr"],
            {
                "schema": 1,
                "polygon_field": "rec_polys",
                "text_field": "rec_texts",
            },
        )


if __name__ == "__main__":
    unittest.main()
