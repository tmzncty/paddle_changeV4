# Legacy behavior contract

This document records behavior that the refactor must preserve or deliberately replace with a documented migration. It is not a recommendation to keep the historical implementation unchanged.

## OCR orchestration reference: `highocr4_f1_pdf_img.py`

The current main legacy reference combines PDF and image OCR in one script.

Observed stages:

1. Define machine-specific PaddleX home, Paddle config, input roots, output root and log/error directories.
2. Set `PADDLEX_HOME` early and create its temp directory.
3. Scan multiple input sources, each tagged as `pdf` or `image`.
4. Skip image OCR when the expected JSON result already exists.
5. Build a PDF preparation queue separately from direct image OCR tasks.
6. Prepare multiple PDFs concurrently; page rendering inside an individual PDF was changed back to serial execution to avoid nested multiprocessing instability.
7. Initialize one PaddleX pipeline per OCR worker process.
8. Validate images with Pillow before inference.
9. Accept PaddleX `predict()` output as generator, list, or a single object exposing `save_to_json`.
10. Persist page-level JSON and copy failed source images to an error directory.
11. Report stage timing, throughput, errors and ETA.
12. Optionally remove temporary rendered PDF pages after processing.

### Legacy risks that must not become new defaults

- absolute `/media/tmzn/...` paths;
- high fixed OCR / PDF preparation / render worker counts;
- high fixed batch sizes;
- cache management coupled to global `~/.paddlex/temp` state;
- recursive deletion before a new run;
- environment assumptions embedded directly in source.

The refactor preserves the useful orchestration ideas but moves machine policy into `ProjectConfig` and destructive behavior behind `paddle_batch_ocr.safety` / `paddle_batch_ocr.cache`.

## Searchable PDF reference: `pdf_creator_with_text_layer7.py`

### Image → JSON matching

For filenames containing `page_<number>`, version 7 tries these names in this exact order:

1. `page_{number:05}.json`
2. `page_{number:04}_result.json`
3. `page_{number:04}.json`
4. `page_{number:03}.json`

For non-page filenames in the enhanced-image path it also tries basename forms:

- `<basename>_result.json`
- `<basename>.json`
- `<basename>_ocr.json`

This precedence is now frozen in `paddle_batch_ocr.naming` tests.

### OCR JSON schema drift

Two adjacent historical searchable-PDF versions expect different text field names:

- `pdf_creator_with_text_layer6.py`: `dt_polys` + `rec_text`
- `pdf_creator_with_text_layer7.py`: `dt_polys` + `rec_texts`

Both versions reject polygon/text length mismatches. The new `paddle_batch_ocr.ocr_schema` adapter accepts both field names, prefers `rec_texts` when both are present, validates polygons and returns one normalized representation.

### Text layer behavior in version 7

For each OCR page, version 7:

1. creates a PDF page at the source image pixel dimensions;
2. inserts the source page image as the visible layer;
3. combines each polygon with its recognized text;
4. ignores empty text entries;
5. calculates a rough two-column ordering from the polygon minimum X relative to page center, then sorts vertically inside each column;
6. creates a PyMuPDF rectangle using polygon points 0 and 2;
7. chooses font size from rectangle height and shrinks until text fits width;
8. inserts text using `china-s` with `render_mode=3` so text is invisible but searchable;
9. writes intermediate chunk PDFs and finally concatenates them.

This behavior is not yet declared correct for every layout. In particular, two-column ordering, rectangle construction and baseline adjustment are heuristic behavior that need fixtures before migration.

## Compatibility rules for the refactor

Until golden fixtures exist, new code should follow these rules:

- accept both `rec_text` and `rec_texts`;
- preserve historical page JSON filename precedence;
- reject malformed or polygon/text mismatched OCR JSON explicitly;
- do not silently overwrite existing OCR JSON or final PDFs by default;
- do not delete global PaddleX cache locations implicitly;
- retain Unicode / Chinese path support;
- keep legacy scripts available until an equivalent tested replacement exists.

## Next fixtures to capture

The most valuable public fixtures are:

1. a one-page image + valid `rec_text` JSON;
2. the same shape with `rec_texts`;
3. one polygon/text mismatch fixture;
4. page naming variants for 3/4/5-digit numbering and `_result`;
5. a two-column page that exposes ordering behavior;
6. a tiny multi-page PDF whose searchable output can be validated for page count and text extraction.
