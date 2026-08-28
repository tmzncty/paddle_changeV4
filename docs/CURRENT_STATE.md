# Current repository state

This document records the repository after the first public-refactor foundation. It distinguishes the proven legacy processing code from the new package being built around it.

## What the repository actually is

The project is not a fork of PaddlePaddle itself. It is a high-throughput document digitization workflow built around PaddleX / PaddleOCR plus PDF tooling.

The historical code has been used for workloads involving large directory trees, hundreds of books and hundreds of thousands of pages. Its most valuable characteristics are operational rather than architectural: skipping existing results, separating PDF preparation from OCR, process-local model initialization, preserving errors, rebuilding searchable PDFs, and surviving long jobs under CPU/GPU/memory/disk pressure.

## Legacy surface

The root scripts are preserved as reference implementations while their behavior is frozen with tests and migrated behind a package API.

Key files:

- `highocr4_f1_pdf_img.py`: current main reference for mixed PDF/image OCR orchestration;
- `pdf_creator_with_text_layer7.py`: current main reference for searchable-PDF reconstruction;
- older `highocr3_*`, `pdf_creator_with_text_layer5/6.py` and `pdf_searchable*` files: historical behavior variants;
- `del_10min_cache.py`: historical cache maintenance with destructive behavior;
- `OCR.yaml` / `OCR2.yaml`: historical PaddleX pipeline examples.

Known legacy hazards include hard-coded `/media/tmzn/...` paths, aggressive fixed worker counts, global PaddleX-cache assumptions and recursive deletion. Those are retained only as history; they are not new defaults.

## New package surface

The refactor now has an installable `src/paddle_batch_ocr/` package with:

- `config.py`: JSON / optional YAML project configuration and path-conflict validation;
- `safety.py`: realpath-based destructive-boundary validation;
- `cache.py`: dry-run-first cleanup limited to `<cache_root>/temp`;
- `doctor.py`: dependency-light runtime, package, GPU and disk diagnostics;
- `cli.py`: `doctor`, `scan`, `cache clean`, and `manifest status`;
- `discovery.py`: deterministic JSON-directory discovery;
- `naming.py`: historical image-to-JSON filename matching precedence;
- `ocr_schema.py`: normalized `rec_text` / `rec_texts` OCR JSON handling;
- `layout.py`: frozen v7 searchable-PDF ordering and rectangle heuristics;
- `io_utils.py`: atomic JSON publication with no-overwrite default;
- `manifest.py`: SQLite/WAL job state keyed by source and stage.

The package is intentionally useful without importing or installing Paddle. This allows public CI, tooling and safety checks to stay lightweight while the GPU execution layer is migrated separately.

## Current safety defaults

New code currently enforces:

- worker counts and batch size default to `1`;
- output overwrite defaults to disabled;
- cache deletion defaults to dry-run;
- deletion is restricted to `<cache_root>/temp`;
- filesystem root, user home and current working directory cannot be destructive roots;
- input roots cannot contain output/log/cache/manifest paths;
- output and cache cannot overlap;
- logs and cache cannot overlap;
- the manifest cannot be stored inside cache;
- containment is based on resolved paths, not string prefixes.

These safeguards do not retroactively modify the root legacy scripts.

## Resume/state baseline

The new SQLite manifest records source size/mtime, stage, status, result path, retry count, errors, worker/device and timing. It uses WAL plus a busy timeout for short multi-worker writes. It invalidates successful state when the source changes and re-runs a successful task if its recorded output disappears.

The OCR/render/searchable-PDF execution engines still need to be wired to this store before end-to-end resume is complete.

## Compatibility baseline

The first refactor also freezes several historical behaviors so later cleanup cannot silently change output:

- v7 page-image to JSON filename precedence;
- v6 `rec_text` and v7 `rec_texts` schema variants;
- polygon/text count validation;
- v7 two-column ordering heuristic;
- v7 polygon point 0/2 text rectangle construction.

These behaviors are compatibility facts, not declarations that every heuristic is correct. Golden fixtures will decide which behaviors remain compatibility modes and which become explicit bug fixes.

## CI baseline

Public CI runs on Python 3.9 and 3.12 and currently validates:

- compilation of all new package modules and tests;
- syntax compilation of the legacy scripts;
- dependency-free unit tests;
- `pip install --no-deps .`;
- console-script version and `doctor --json` smoke tests;
- `manifest status` smoke test against the example config.

Paddle/CUDA smoke tests are intentionally separate future work; GitHub-hosted runners should not be treated as representative GPU deployment hosts.

## What is deliberately not done yet

- no new PaddleX/PaddleOCR execution engine is wired into the CLI;
- no new PDF-render command is wired into the CLI;
- no new searchable-PDF writer has replaced the legacy writer;
- no CPU/GPU Paddle version matrix is claimed;
- no legacy script has been deleted merely to make the tree look clean;
- no historical commits have been rewritten.

See `ROADMAP.md` for staged migration and `docs/LEGACY_BEHAVIOR.md` for the frozen compatibility contract.
