# OCR execution layer

The public OCR execution layer targets the current PaddleX 3.x pipeline API while
keeping Paddle/PaddleX optional for the core package.

## Current upstream baseline

At the time this document was written (2026-08):

- PaddleX latest release: 3.7.2;
- PaddleOCR latest release: 3.7.0, including PP-OCRv6;
- PaddleX 3.7 supports Python 3.8–3.13 and PaddlePaddle 3.0+;
- current PaddleX documentation demonstrates PaddlePaddle 3.3.0 installation;
- OCR pipelines are created with `paddlex.create_pipeline(pipeline="OCR")` or a
  local pipeline YAML path;
- current `create_pipeline` options include `device`, `engine`, `engine_config`,
  `use_hpip`, and `hpi_config`;
- pipeline prediction returns iterable Result objects; Result objects expose a
  `json` mapping in addition to `save_to_json()`;
- current OCR results expose `rec_polys` / `rec_texts` after recognition-score
  filtering.

Upstream references:

- https://paddlepaddle.github.io/PaddleX/3.7/en/installation/installation.html
- https://paddlepaddle.github.io/PaddleX/3.7/en/pipeline_usage/instructions/pipeline_python_API.html
- https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html

## Installation policy

PaddlePaddle is deliberately not a normal project dependency because CPU and GPU
runtimes use hardware-specific official wheel indexes.

Install the appropriate PaddlePaddle runtime first according to the current
PaddleX/PaddlePaddle documentation. Then install the OCR integration layer:

```bash
python -m pip install '.[ocr]'
```

The `ocr` extra currently provides `paddlex[ocr]>=3.7,<4`; it does not choose a
PaddlePaddle CPU/GPU runtime for you.

## Serial execution command

```bash
paddle-batch-ocr ocr INPUT \
  --output json/
```

`INPUT` may be one image or a directory scanned recursively. Relative directory
structure is preserved under the output root and files are written as
`<stem>_result.json`.

Useful options:

```bash
paddle-batch-ocr ocr images/ --output json/ --device cpu
paddle-batch-ocr ocr images/ --output json/ --device gpu:0
paddle-batch-ocr ocr images/ --output json/ --pipeline ./OCR.yaml
paddle-batch-ocr ocr images/ --output json/ --engine paddle_static
paddle-batch-ocr ocr images/ --output json/ --manifest work/manifest.sqlite3
```

The first public execution contract is intentionally serial. One pipeline is
created lazily and reused for every image that actually needs inference. A full
resume that only adopts/skips valid existing JSON does not initialize the model.

## Result normalization

Current PaddleOCR 3.x can produce more detection boxes than recognized text
entries because recognition-score filtering happens after detection. The public
adapter therefore prefers:

```text
rec_polys + rec_texts
```

when both are present. Historical repository JSON continues to support:

```text
dt_polys + rec_texts
dt_polys + rec_text
```

This prevents filtered detection boxes from being paired with the wrong text.

PaddleX Result objects are read through their documented `json` attribute. A
common outer `{"res": {...}}` envelope is unwrapped before schema validation.
The original result fields are preserved and a small `_paddle_batch_ocr`
provenance record is added before atomic JSON publication.

## Resume and failure semantics

- output JSON is atomically published and no-overwrite by default;
- valid pre-existing JSON is adopted during normal resume;
- if a manifest already knows a source and its size/mtime changes, the old
  result is stale and is not silently adopted;
- stale existing output requires explicit overwrite;
- one image failure is recorded without aborting every later image;
- the command exits non-zero when any task fails;
- symlinked input/output paths are rejected at discovery/publication boundaries.

## Concurrency

The current public engine does **not** consume `ocr_workers > 1` yet. This is
intentional. The next concurrency milestone will add a process initializer so
each worker creates exactly one PaddleX pipeline and then executes the same task
contract defined here.

Batch sizes for OCR submodules belong in the current PaddleX pipeline YAML (for
example the TextRecognition module's `batch_size`). The historical global
`OCR_BATCH_SIZE` / `hpi_params` approach is not treated as the current PaddleX
API contract.
