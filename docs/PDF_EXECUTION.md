# PDF execution layer

The refactored package now has an execution path that can be validated without
Paddle, CUDA, or OCR model downloads.

## Install

```bash
python -m pip install '.[pdf]'
```

The `pdf` extra intentionally uses broad compatible ranges. Current PyMuPDF and
Pillow releases require Python 3.10+, while pip can still resolve the last
Python-3.9-compatible releases inside those ranges for the project's 3.9 core
compatibility line.

PyMuPDF is distributed under the GNU AGPL or a commercial license. Review the
upstream license obligations before redistributing a combined application.

## Render a PDF transactionally

```bash
paddle-batch-ocr render input.pdf --output pages/
```

Important behavior:

- all pages are rendered into a hidden sibling staging directory first;
- the final output directory appears only after the whole PDF renders;
- output page names are deterministic: `page_00001.png`, `page_00002.png`, ...;
- existing output is refused unless `--overwrite` is explicit;
- overwrite uses a backup/replace/rollback sequence rather than deleting the
  old directory first;
- DPI is bounded to 36–1200.

## Build a searchable PDF

```bash
paddle-batch-ocr searchable-pdf \
  --images pages/ \
  --ocr-json json/ \
  --output book_searchable.pdf
```

The public execution path is stricter than the historical script:

- page images must be a complete `page_00001..N` sequence;
- duplicate page numbers or sequence gaps are errors;
- every image must resolve to OCR JSON;
- both historical `rec_text` and `rec_texts` fields are accepted;
- historical JSON filename precedence is preserved;
- invalid polygon/text counts are errors;
- final PDF publication is atomic and no-overwrite by default.

The text ordering and geometry currently use the frozen `pdf_creator_with_text_layer7.py`
compatibility rules. This is intentional: future geometry corrections should
be visible behavior changes backed by golden tests, not accidental side effects
of refactoring.

## CI contract

The `pdf-smoke` job creates a synthetic PDF at test time, renders it, writes
mixed historical OCR JSON variants, rebuilds a searchable PDF, and extracts the
hidden text back with PyMuPDF. No binary fixture is committed to the repository.

The smoke matrix covers Python 3.9 and 3.12 separately from the dependency-free
core baseline.
