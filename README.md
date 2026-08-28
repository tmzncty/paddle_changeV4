# paddle_changeV4

面向**大规模中文 OCR、PDF 批处理与可搜索 PDF 重建**的 PaddleOCR / PaddleX 流水线项目。

> 这个仓库最初是一组在真实大批量文献数字化任务中长期迭代的个人脚本。现在正在把已经验证过的处理经验逐步收束为可安装、可配置、可测试、默认安全的公开工具；历史脚本继续保留，直到新实现有足够的行为基线和真实数据验证后再归档。

## 项目定位

这不是 PaddlePaddle 核心代码的修改版。项目关注 PaddleX / PaddleOCR 的**上层批处理与文档工作流**：

1. 扫描 PDF / 图片目录；
2. PDF 页面事务式渲染；
3. 执行 PaddleX OCR；
4. 原子保存逐页 OCR JSON；
5. 从 OCR 坐标重建 searchable PDF；
6. 用 SQLite manifest 管理长时间任务的 resume / stale / failure；
7. 在安全边界明确的前提下扩展 CPU 多进程吞吐。

根目录 legacy 脚本保留了过去几十万页级任务中的吞吐量、缓存、并发和异常页处理经验，但仍含机器专属路径和激进参数，**不能当作公开项目的通用默认值**。

## 当前状态

**Status: public PDF execution + validated PaddleX OCR + CPU spawn workers + project orchestration**

当前新的 package 位于 `src/paddle_batch_ocr/`，已经具备：

```text
PDF -> PNG pages
image(s) -> PaddleX OCR -> atomic JSON
PNG + OCR JSON -> searchable PDF

configured PDF source
  -> render
  -> OCR
  -> searchable PDF
```

OCR 有两条执行路径：

- `workers=1`：已用真实 PaddleX / PP-OCRv6 CPU 模型验证的串行路径；
- `workers>1`：`spawn` 多进程，每个 process 惰性初始化一个 pipeline。当前只允许显式 `device=cpu`；GPU worker pool 仍未开放。

公共 CI 同时使用 fake picklable pipeline 做跨 Python 的 spawn 生命周期测试，并使用真实 PaddleX 模型做 CPU OCR smoke。

## 安装

### Core

```bash
python -m pip install --no-deps .
paddle-batch-ocr --version
paddle-batch-ocr doctor
```

core 不强制安装 Paddle/CUDA。

### PDF

```bash
python -m pip install '.[pdf]'
```

### YAML 配置

```bash
python -m pip install '.[yaml]'
```

### OCR

先按 PaddlePaddle 官方方式安装与你硬件匹配的 CPU / GPU runtime，再安装项目 OCR extra：

```bash
python -m pip install '.[ocr]'
```

`.[ocr]` 当前使用：

```text
paddlex[ocr-core]>=3.7,<3.8
```

PaddlePaddle 本身刻意不放进统一 dependency，因为 CPU / CUDA wheel 的来源和环境约束不同。

### 当前真实 CPU 验证矩阵

| Component | Version / model |
| --- | --- |
| OS | Ubuntu 24.04 |
| Python | 3.12.14 |
| PaddlePaddle CPU | 3.2.2 |
| PaddleX | 3.7.2 |
| Detection | PP-OCRv6_small_det |
| Recognition | PP-OCRv6_small_rec |

PaddlePaddle 3.3.0 在当前 CPU oneDNN/PIR 路径存在上游回归；公共 CI 暂时使用 3.2.2 作为上游 issue 给出的 workaround。详见 [`docs/OCR_EXECUTION.md`](docs/OCR_EXECUTION.md)。

## CLI

### Diagnostics / manifest / cache

```bash
paddle-batch-ocr doctor
paddle-batch-ocr doctor --json
paddle-batch-ocr doctor --config examples/config.json

paddle-batch-ocr scan --config examples/config.json

paddle-batch-ocr manifest status --config examples/config.json
paddle-batch-ocr manifest status --config examples/config.json --json
paddle-batch-ocr manifest report --config examples/config.json --json
paddle-batch-ocr manifest jobs --config examples/config.json --status failed --json
paddle-batch-ocr manifest jobs --config examples/config.json --status failed --csv > failed.csv

# dry-run
paddle-batch-ocr cache clean --config examples/config.json

# explicit destructive action
paddle-batch-ocr cache clean --config examples/config.json --execute
```

`manifest report/jobs` 使用 read-only SQLite connection，不会为了查看状态创建或迁移数据库。详细语义见 [`docs/MANIFEST_OPERATIONS.md`](docs/MANIFEST_OPERATIONS.md)。

### PDF -> PNG

```bash
paddle-batch-ocr render input.pdf --output pages/
paddle-batch-ocr render input.pdf --output pages/ --dpi 144
```

`render` 先写 sibling staging directory，整本成功后才发布最终目录；默认拒绝覆盖。

### OCR

单图 / 串行：

```bash
paddle-batch-ocr ocr page.png \
  --output json/ \
  --device cpu
```

递归图片目录：

```bash
paddle-batch-ocr ocr images/ \
  --output json/ \
  --device gpu:0
```

CPU 多进程：

```bash
paddle-batch-ocr ocr images/ \
  --output json/ \
  --device cpu \
  --workers 4
```

`workers>1` 当前强制要求 `--device cpu`。这是有意的安全边界：GPU 并发必须先有明确 device map，不能让多个进程默认把同一套模型复制到 `gpu:0`。

指定 pipeline YAML：

```bash
paddle-batch-ocr ocr images/ \
  --output json/ \
  --pipeline ./OCR.yaml
```

接入 manifest：

```bash
paddle-batch-ocr ocr images/ \
  --output json/ \
  --manifest work/manifest.sqlite3
```

OCR 的公共合同包括：

- `workers=1` 时一个 pipeline 惰性初始化并复用；
- `workers>1` 使用 `spawn`，每个参与进程最多初始化一次 pipeline；
- 每个 worker 独立打开 SQLite connection；
- resume / stale preflight 在任务进入 worker pool 前完成；
- 已有合法 JSON 可以 adopt，不需要模型初始化；
- source size/mtime 变化使旧 manifest result 失效；
- 对已有可信 execution profile 的任务，result target / profile 变化也会使旧 success 失效；
- 本地 pipeline YAML 使用绝对路径 + size + SHA-256 指纹，因此同一路径内容变化可以被识别；
- 历史 adopt 的结果不会被伪造为“由当前 pipeline 生成”，未知 profile 保持 unknown；
- JSON 先验证 schema，再 temp + fsync + atomic publish；
- 默认 no-overwrite；
- 单页失败不会吞掉其他任务；
- 任一任务 failed 时 CLI 返回非零；
- 输入 / 输出 / manifest symlink 边界被拒绝；
- 同目录 `same.png` + `same.jpg` 这种会映射到同一个 result filename 的输入在执行前直接拒绝。

`--json` 保持 stdout 为严格 JSON。Paddle / oneDNN 的 Python 与 native fd 1 输出在 inference 期间被临时路由到 stderr。

### Searchable PDF

```bash
paddle-batch-ocr searchable-pdf \
  --images pages/ \
  --ocr-json json/ \
  --output book_searchable.pdf
```

公开实现要求完整页序列和完整 OCR JSON，并在最终文件发布前完成整本构建。

### Project `run`

```bash
paddle-batch-ocr run \
  --config examples/config.json

paddle-batch-ocr run \
  --config examples/config.json \
  --dpi 144 \
  --json
```

PDF source 的稳定 artifact layout：

```text
<output_root>/
  source-001/
    pdf/
      <relative-document-without-.pdf>/
        pages/
        ocr/
        searchable.pdf
```

image source 当前运行 OCR-only：

```text
<output_root>/
  source-002/
    image/
      ocr/
```

`run` 当前**只把 OCR stage 的 `runtime.ocr_workers` 接入 process pool**。PDF document 调度、render 和 searchable-PDF stage 仍按确定性顺序执行；`render_workers` / `pdf_prep_workers` 尚未被当作已实现能力。

更多语义见 [`docs/PROJECT_RUN.md`](docs/PROJECT_RUN.md)。

## Project configuration

从 [`examples/config.json`](examples/config.json) 开始：

```json
{
  "input_sources": [
    {"path": "./input-pdfs", "type": "pdf"},
    {"path": "./input-images", "type": "image"}
  ],
  "output_root": "./work/output",
  "log_dir": "./work/logs",
  "cache_root": "./work/cache",
  "paddle_config": "./OCR2.yaml",
  "runtime": {
    "device": "cpu",
    "ocr_workers": 1,
    "pdf_prep_workers": 1,
    "render_workers": 1,
    "batch_size": 1
  },
  "delete_temp_images": false,
  "overwrite": false,
  "resume": true
}
```

新代码默认保守：

- worker 默认 `1`；
- batch 默认 `1`；
- `overwrite=false`；
- destructive cache cleanup 默认 dry-run；
- output / log / cache / manifest 不允许危险重叠；
- cache root 不能是 filesystem root / home / cwd；
- cache temp symlink 拒绝；
- manifest symlink 在 SQLite open 前拒绝；
- destructive boundary 使用真实路径 containment。

若你使用 `workers>1`，当前配置必须显式：

```json
{
  "runtime": {
    "device": "cpu",
    "ocr_workers": 4
  }
}
```

## Manifest / resume

SQLite manifest 以 `(source_path, stage)` 为键，记录：

- source size / mtime fingerprint；
- `pending / running / success / failed`；
- successfully published `result_path`；
- intended result path（失败时也保留）；
- canonical execution profile；
- retry count；
- error class / message；
- worker / device；
- started / finished / duration；
- WAL + busy timeout。

当前集成：

- OCR：逐图片 / 逐页记录；
- render：逐 PDF stage 记录；
- searchable-PDF：逐 PDF stage 记录。

writer 打开旧 manifest 时会向后兼容迁移 provenance 列；历史 success 可以从 `result_path` 安全回填 intended target，但不会猜测历史 execution profile。只读 report/jobs 可以直接查看未迁移旧库而不修改它。

一个重要依赖规则是：**只要这一轮 OCR 真正产生了新 JSON，已有 searchable PDF 就不能仅凭源 PDF fingerprint 被 adopt。** 在已有 searchable PDF 的情况下，需要显式 `overwrite=true` 才允许重建，避免“新 OCR JSON + 旧文本层 PDF”这种跨 stage 不一致。

更完整的 manifest / provenance / reporting 说明见 [`docs/MANIFEST_OPERATIONS.md`](docs/MANIFEST_OPERATIONS.md)。

## OCR Result compatibility

当前 adapter：

- 当前 PaddleX：优先官方 Result `.json`；
- 当前 geometry：`rec_polys + rec_texts`；
- 历史：`dt_polys + rec_texts`；
- 更早历史：`dt_polys + rec_text`；
- NumPy ndarray / scalar 转成普通 JSON values；
- runtime-only `vis_fonts` / image 等对象不进入稳定结果；
- 新输出增加 `_paddle_batch_ocr` provenance。

## CI

当前公共 CI：

1. Python 3.9 / 3.12 dependency-free compile + tests；
2. package install + CLI smoke；
3. Python 3.9 / 3.12 real PDF execution smoke；
4. project orchestration 的 real render + fake OCR + real searchable-PDF round-trip；
5. Python 3.12 real PaddleX CPU serial OCR；
6. Python 3.12 real PaddleX CPU two-worker smoke。

CI 使用当前 major 的 official Actions，并缓存 pip downloads 与 `~/.paddlex/official_models`。PR branch 不再同时触发“branch push + PR”两份昂贵 OCR；main 由 push 验证，feature branch 由 PR 验证。

真实 serial / two-worker gates 不只检查 OCR 输出：还验证 manifest 的 intended target、本地 pipeline YAML SHA-256 execution profile，以及 two-worker PID 数量。因此 provenance 也有真实 PaddleX CPU 证据。

## Legacy implementation index

| File | Role | Status |
| --- | --- | --- |
| `pdf_to_png.py` | PDF 多进程拆图 | legacy / new `render` exists |
| `highocr3_f2.py` | image directory OCR | legacy |
| `highocr3_f2_pdf.py` / `highocr3_f2_pdf2.py` | early PDF OCR | legacy |
| `highocr4_f1_pdf_img.py` | combined PDF/image batch OCR | legacy / main historical reference |
| `pdf_creator_with_text_layer5.py` / `6.py` / `7.py` | searchable PDF | legacy / replacement exists |
| `pdf_searchable2.5.py` / `pdf_searchable3.py` | searchable-PDF experiments | legacy |
| `del_10min_cache.py` | PaddleX cache experiment | legacy / destructive |
| `OCR.yaml` / `OCR2.yaml` | historical PaddleX configs | legacy configuration |

legacy 数字文件名只是迭代痕迹，不是稳定 API。

## Third-party licenses

仓库本身使用 **GNU GPL-3.0**。PDF extra 使用 PyMuPDF 与 Pillow；PyMuPDF 上游当前采用 GNU AGPL / commercial 双许可边界，分发组合应用时请自行核对相应义务。

## Next

下一阶段重点：

- 基于 manifest provenance 的 targeted retry：先 dry-run，再 explicit execute；
- retry 时校验本地 pipeline 历史 SHA-256，未知 profile 不猜；
- retry policy 与更细 failure classes；
- 对 render / searchable stage 建立更完整 dependency fingerprints；
- GPU worker device map + manual/self-hosted validation；
- PDF geometry golden fixtures（中文长文本、旋转、双栏）；
- benchmark / worker-thread tuning guide；
- 在新 CLI 覆盖和真实数据对比充分后归档 legacy。

详细进度见 [`ROADMAP.md`](ROADMAP.md)。
