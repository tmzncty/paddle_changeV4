# Refactor Roadmap

目标不是把历史脚本重写一遍，而是把真实大规模 OCR 任务中已经积累的逻辑逐步收束为一个安全、可复现、可维护的工具。

## M0 — Public project baseline

- [x] README 明确项目定位与 legacy 状态
- [x] destructive behavior / 高负载风险说明
- [x] roadmap / contributing
- [x] Python 3.9 / 3.12 baseline CI
- [ ] 打开并维护 issue tracker
- [ ] 补充 repository topics / description

## M1 — Freeze legacy behavior

- [ ] 每个 legacy 脚本完整用途 / I/O / replacement 表
- [x] 记录主要 OCR legacy 处理阶段和风险默认值
- [x] 冻结 searchable-PDF schema / naming / geometry assumptions
- [x] historical page JSON naming priority
- [x] `rec_text` / `rec_texts`
- [x] v7 two-column ordering and polygon geometry
- [x] synthetic PDF / OCR JSON fixtures
- [x] searchable-PDF round-trip smoke
- [ ] richer geometry golden fixtures

原则：没有行为基线前不删除旧脚本。

## M2 — Configuration and safety

- [x] input / output / log / cache / manifest / Paddle config path model
- [x] OCR / PDF / render worker and batch fields
- [x] `auto / cpu / gpu / gpu:N`
- [x] JSON + optional YAML config
- [x] conservative defaults (`worker=1`, `batch=1`, `overwrite=false`)
- [x] cache cleanup dry-run by default
- [x] realpath containment
- [x] root / home / cwd destructive-root rejection
- [x] cache temp symlink rejection
- [x] input/output/log/cache/manifest overlap validation
- [x] manifest symlink rejection before SQLite open
- [x] preserve explicit manifest symlink identity through config loading
- [x] PDF final staging / atomic publication
- [x] config overwrite/resume wired into project `run`
- [ ] migrate all legacy `clear_cache()` calls to the safe layer

## M3 — Package and unified CLI

Available:

```bash
paddle-batch-ocr doctor
paddle-batch-ocr scan --config CONFIG
paddle-batch-ocr cache clean --config CONFIG
paddle-batch-ocr manifest status --config CONFIG
paddle-batch-ocr manifest report --config CONFIG
paddle-batch-ocr manifest jobs --config CONFIG
paddle-batch-ocr manifest retry-failed --config CONFIG
paddle-batch-ocr render INPUT.pdf --output DIR
paddle-batch-ocr ocr INPUT --output DIR
paddle-batch-ocr searchable-pdf --images DIR --ocr-json DIR --output FILE.pdf
paddle-batch-ocr run --config CONFIG
```

- [x] installable `src/` package
- [x] core does not force Paddle/CUDA
- [x] `pdf` / `yaml` / `ocr` extras
- [x] machine-readable JSON modes
- [x] manifest jobs CSV export
- [x] targeted retry dry-run / execute JSON surface
- [x] fd-level native stdout isolation for OCR/run/retry execute JSON modes
- [x] config-driven project orchestration
- [x] deterministic project artifact layout

## M4 — OCR engine cleanup

### Serial execution

- [x] PaddleX 3.7 `create_pipeline` adapter
- [x] one lazy pipeline per serial batch
- [x] batch-wide pipeline-init failure cache
- [x] `predict_iter` / `predict`
- [x] modern `rec_polys + rec_texts`
- [x] historical `dt_polys + rec_texts` / `rec_text`
- [x] official Result `.json` before runtime Mapping
- [x] NumPy array/scalar -> JSON-safe
- [x] atomic result publication
- [x] no-overwrite by default
- [x] per-image failure isolation
- [x] resume adoption without model initialization
- [x] manifest stale detection
- [x] symlink safety
- [x] strict JSON stdout
- [x] real PP-OCRv6 CPU smoke
- [x] execution profile provenance for actual OCR attempts
- [x] local pipeline YAML content fingerprint (SHA-256)

### CPU process workers

- [x] `workers > 1` process pool
- [x] one lazy pipeline per participating process
- [x] explicit `spawn` start method
- [x] one SQLite connection per process
- [x] parent-side resume/stale preflight before submission
- [x] deterministic output order in batch summary
- [x] output-name collision rejection
- [x] cross-Python fake spawn test
- [x] real two-worker PaddleX CPU smoke
- [x] public safety boundary: multi-worker requires explicit CPU device
- [x] worker/preflight failure provenance preservation
- [ ] GPU worker/device map
- [ ] automatic retry policy
- [ ] richer damaged-image / empty-result / model-failure statuses
- [ ] GPU self-hosted/manual smoke
- [ ] throughput benchmark and thread/worker tuning guide

## M5 — PDF pipeline cleanup

### Rendering

- [x] independent `pdf_render.py`
- [x] one PDF open per render
- [x] 36–1200 DPI
- [x] deterministic `page_00001.png`
- [x] sibling staging directory
- [x] no-overwrite by default
- [x] overwrite backup / replace / rollback
- [x] Python 3.9 / 3.12 real PDF smoke
- [x] existing-render validation for resume
- [x] render stage manifest integration
- [x] render intended-result / DPI execution profile provenance
- [x] provenance-validated targeted render retry
- [ ] colorspace / alpha / format config
- [ ] segmented huge-PDF recovery

### Searchable PDF

- [x] OCR schema adapter
- [x] historical naming compatibility
- [x] legacy layout heuristic freeze
- [x] complete page sequence requirement
- [x] duplicate / gap / missing JSON hard errors
- [x] sibling temp + atomic publication
- [x] hidden-text round-trip
- [x] Python 3.9 / 3.12 validation
- [x] existing PDF page-count validation for resume
- [x] searchable-PDF stage manifest integration
- [x] downstream invalidation when OCR actually produces new results
- [x] searchable target + pages/OCR input-directory execution provenance
- [x] provenance-validated targeted searchable-PDF retry
- [ ] richer dependency fingerprint beyond source-PDF mtime/size
- [ ] Chinese long-text / rotation / column geometry goldens
- [ ] legacy vs corrected geometry mode

## M6 — Resume, manifest and observability

- [x] `(source_path, stage)` task key
- [x] source size / mtime
- [x] `pending / running / success / failed`
- [x] result path
- [x] retry count
- [x] error class / message
- [x] worker / device
- [x] timing fields
- [x] WAL + busy timeout
- [x] concurrent registration idempotence
- [x] source-change invalidation
- [x] missing-success-result invalidation
- [x] serial OCR adoption/stale semantics
- [x] multi-process OCR manifest lifecycle
- [x] render stage lifecycle
- [x] searchable-PDF stage lifecycle
- [x] prevent searchable adoption after upstream OCR changed
- [x] read-only `manifest report` aggregate statistics
- [x] read-only filtered/paged `manifest jobs`
- [x] JSON / CSV job export
- [x] snapshot-consistent reporting during active writes
- [x] intended-result provenance
- [x] canonical execution-profile provenance
- [x] backward-compatible in-place manifest migration
- [x] old-schema read-only reporting without migration
- [x] profile-aware stale detection only when historical profile is known
- [x] local OCR pipeline config content fingerprint
- [x] targeted rerun of failed OCR/render/searchable items
- [x] dry-run by default + explicit `--execute`
- [x] source/profile/target revalidation immediately before retry action
- [x] retry target confined to configured `output_root`
- [x] local OCR pipeline SHA-256 revalidation before automatic retry
- [ ] full dependency graph / content fingerprints
- [ ] automatic retry policy / backoff
- [ ] attempt/event history for every retry
- [ ] stale-running detection / recovery policy
- [ ] large-manifest query indexing + benchmark

当前 provenance / retry 设计刻意不把 legacy success 的未知执行配置猜成当前配置。历史 `result_path` 可以安全回填 intended target；execution profile 仍保持 unknown。只有能够重新验证的失败项才进入 automatic targeted retry，无法证明可复现的项保持 ineligible 并交给人工处理。

## M7 — Tests and CI

- [x] Python 3.9 / 3.12 compile
- [x] dependency-free unit tests
- [x] package install + CLI smoke
- [x] Python 3.9 / 3.12 real PDF execution
- [x] real project render + fake OCR + real searchable round-trip
- [x] real serial PaddleX CPU OCR
- [x] spawn worker lifecycle test on Python 3.9 / 3.12
- [x] real two-worker PaddleX CPU OCR
- [x] real serial/parallel PaddleX manifest-provenance assertions
- [x] real PP-OCRv6 targeted retry fixture / dry-run / execute / manifest-repair gate
- [x] pip cache for heavy jobs
- [x] PaddleX official-model cache
- [x] current official Actions major versions
- [x] no duplicate feature-branch push + PR OCR jobs
- [ ] GPU manual / self-hosted smoke
- [ ] expanded geometry golden tests

Validated real OCR / targeted-retry matrix:

```text
Ubuntu 24.04
Python 3.12.14
PaddlePaddle CPU 3.2.2
PaddleX 3.7.2
PP-OCRv6_small_det
PP-OCRv6_small_rec
```

Targeted retry gate currently proves:

```text
targeted_retry_fixture=PASS
targeted_retry_dry_run=PASS
targeted_retry_recognized_lines=4
targeted_retry_real_ocr=PASS
```

PaddlePaddle 3.3.0 CPU oneDNN/PIR regression remains documented; CI uses 3.2.2 until upstream changes are revalidated。

## M8 — Reproducible environments

- [x] `pyproject.toml`
- [x] YAML extra
- [x] PDF extra
- [x] `paddlex[ocr-core]>=3.7,<3.8`
- [x] PaddlePaddle hardware-specific runtime kept separate
- [x] Python 3.9 / 3.12 PDF dependency resolution
- [x] Paddle/PaddleX CPU matrix
- [x] CI pip/model cache
- [ ] CUDA/GPU matrix
- [ ] optional container / Conda recipe

## M9 — Release and legacy archive

After new CLI coverage and real-data comparison are sufficient:

- [ ] move replaced scripts into `legacy/`
- [ ] mark replacement for each script
- [ ] first stable semantic version
- [ ] changelog / release notes
- [ ] README becomes current-path only

Current `render`, OCR (serial + CPU process workers), `searchable-pdf`, project `run`, manifest observability and provenance-validated targeted retry all have public execution paths. Legacy remains until broader real-data and geometry validation is complete.

## Non-goals

Short term:

- no custom OCR model implementation;
- no general GUI;
- no pretend support for every CUDA/driver combination;
- no blind multi-process replication onto one GPU;
- no arbitrary-path retry driven only by a mutable SQLite row;
- no automatic retry of unverifiable named-pipeline history;
- no deleting historical code merely for aesthetics;
- no abstraction that sacrifices large-batch throughput without evidence.
