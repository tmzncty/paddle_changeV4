# paddle_changeV4

面向**大规模中文 OCR、PDF 批处理与可搜索 PDF 重建**的 PaddleOCR / PaddleX 流水线项目。

> 这个仓库最初是一组在真实大批量文献数字化任务中长期迭代的个人脚本。现在正在把已经验证过的处理经验逐步收束为可安装、可配置、可测试、默认安全的公开工具；历史脚本继续保留，直到新实现有足够的行为基线和真实数据验证后再归档。

## 项目定位

这不是 PaddlePaddle 核心代码的修改版。项目关注的是 PaddleX / PaddleOCR **上层批处理与文档工作流**：

1. 扫描 PDF / 图片目录；
2. PDF 页面事务式渲染；
3. 执行 PaddleX OCR；
4. 原子保存逐页 OCR JSON；
5. 从 OCR 坐标重建 searchable PDF；
6. 为长时间、大规模任务提供 resume、manifest、错误隔离和安全边界。

根目录的 legacy 脚本保存了过去几十万页级任务中的吞吐量、缓存、并发和异常页处理经验，但其中仍有机器专属路径和激进并发值，**不能视为公开项目的安全默认配置**。

## 当前状态

**Status: public PDF execution + serial PaddleX OCR execution active**

当前仓库同时存在两层：

- `src/paddle_batch_ocr/`：新的 package、安全层、配置、diagnostics、manifest、PDF execution、serial OCR execution 和统一 CLI；
- 根目录历史 `.py`：legacy 参考实现，暂不删除。

已经真实验证的主链路包括：

```text
PDF -> PNG pages
image -> PaddleX OCR -> atomic JSON
PNG + OCR JSON -> searchable PDF
```

OCR 当前故意保持**串行**：一个进程惰性初始化一个 PaddleX pipeline，并在整批图片上复用。多进程 worker lifecycle 是下一阶段，不在尚未验证时假装支持。

## 安装

### 核心工具

核心 package 不强制安装 Paddle/CUDA：

```bash
python -m pip install --no-deps .
paddle-batch-ocr --version
paddle-batch-ocr doctor
```

### PDF 功能

```bash
python -m pip install '.[pdf]'
```

### YAML 项目配置

```bash
python -m pip install '.[yaml]'
```

### OCR 功能

项目的 `ocr` extra 只安装 PaddleX OCR core，不替你选择 CPU/GPU PaddlePaddle wheel：

```bash
# 先按 PaddlePaddle 官方方式安装与你硬件匹配的 runtime
python -m pip install '.[ocr]'
```

`.[ocr]` 当前约束为 `paddlex[ocr-core]>=3.7,<3.8`。PaddlePaddle 本身刻意不写进统一 dependency，因为 CPU / CUDA wheel 来自不同官方索引和硬件组合。

当前公共 CPU smoke 的已验证组合：

| Component | Version / model |
| --- | --- |
| Python | 3.12.14 |
| PaddlePaddle CPU | 3.2.2 |
| PaddleX | 3.7.2 |
| Text detection | PP-OCRv6_small_det |
| Text recognition | PP-OCRv6_small_rec |
| Runner | Ubuntu 24.04 GitHub Actions |

PaddlePaddle 3.3.0 在当前 CPU oneDNN/PIR 路径存在已知上游回归；项目 CI 暂时使用 3.2.2 作为上游 issue 给出的 workaround。详见 [`docs/OCR_EXECUTION.md`](docs/OCR_EXECUTION.md)。GPU 仍需要独立 self-hosted/manual 验证。

## CLI

### Diagnostics / manifest / cache

```bash
paddle-batch-ocr doctor
paddle-batch-ocr doctor --json
paddle-batch-ocr doctor --config examples/config.json

paddle-batch-ocr scan --config examples/config.json

paddle-batch-ocr manifest status --config examples/config.json
paddle-batch-ocr manifest status --config examples/config.json --json

# 默认 dry-run
paddle-batch-ocr cache clean --config examples/config.json

# 只有显式 --execute 才删除 <cache_root>/temp
paddle-batch-ocr cache clean --config examples/config.json --execute
```

### PDF -> PNG

```bash
paddle-batch-ocr render input.pdf --output pages/
paddle-batch-ocr render input.pdf --output pages/ --dpi 144
```

`render` 先写 sibling staging directory，整本成功后才发布最终目录；默认拒绝覆盖已有输出。

### PaddleX OCR

单图：

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

指定本地 PaddleX pipeline YAML：

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

`--json` 会把项目 summary 保持为严格机器可读 stdout。Paddle / oneDNN 的 Python 日志以及直接写 OS fd 1 的原生日志会在 OCR 执行期间被临时路由到 stderr，执行结束后恢复 stdout，再输出唯一 JSON summary。

当前 serial OCR contract：

- 一个 pipeline 惰性初始化并复用；
- 已有合法 JSON 在 resume 时可直接采用，不必加载模型；
- source size/mtime 变化会使 manifest 中旧结果失效；
- JSON 先验证 schema，再同目录临时写入 + fsync + atomic publish；
- 默认不覆盖；
- 单图失败不阻断后续图片；
- 任一任务失败时 CLI 返回非零；
- symlinked input/output/manifest 边界会被拒绝。

### searchable PDF

```bash
paddle-batch-ocr searchable-pdf \
  --images pages/ \
  --ocr-json json/ \
  --output book_searchable.pdf
```

公开实现要求完整页序列、每页 OCR JSON 都存在，并在最终 PDF 发布前完成整本构建。详细说明见 [`docs/PDF_EXECUTION.md`](docs/PDF_EXECUTION.md)。

## OCR Result 兼容

新的 adapter 同时覆盖历史和当前 PaddleX 输出：

- 当前 PaddleX：优先官方 Result `.json` 导出合同；
- 当前 geometry：`rec_polys + rec_texts`；
- 历史兼容：`dt_polys + rec_texts`；
- 更早历史兼容：`dt_polys + rec_text`；
- NumPy ndarray / scalar 在 package 边界转换成普通 JSON 值；
- 不把 `vis_fonts`、原始图像等 PaddleX runtime visualization 对象误写进稳定 OCR JSON；
- 每份新 JSON 增加 `_paddle_batch_ocr` provenance metadata。

优先 `rec_polys` 很重要：当前 OCR pipeline 在 recognition score 过滤后，`dt_polys` 与最终识别文本数量可能不再一一对应。

## 项目配置与安全默认值

推荐从 [`examples/config.json`](examples/config.json) 开始。

新的默认原则是保守而不是追求 benchmark：

- worker 默认 `1`；
- batch size 默认 `1`；
- `overwrite=false`；
- cache 删除默认 dry-run；
- output / log / cache / manifest 不允许危险重叠；
- cache root 不能是 filesystem root、用户 home 或当前工作目录；
- destructive path 使用 realpath containment；
- cache temp symlink 直接拒绝；
- manifest symlink 在 SQLite 打开前拒绝。

这些保护**不自动覆盖根目录 legacy 脚本**。不要直接把 legacy 中的 `/media/tmzn/...` 路径、高并发和 cache cleanup 当成通用默认设置。

## Manifest / resume

SQLite manifest 以 `(source_path, stage)` 为任务键，当前记录：

- source size / mtime fingerprint；
- `pending / running / success / failed`；
- result path；
- retry count；
- error class / message；
- worker / device；
- started / finished / duration；
- WAL + busy timeout。

OCR serial execution 已接入 manifest 的 adoption / stale-result 语义；完整 `run` orchestration 与所有 PDF stage 的统一 manifest 生命周期仍在 roadmap。

## CI

当前公共 CI 分层：

1. Python 3.9 / 3.12 dependency-free compile + unit tests；
2. package install + CLI smoke；
3. Python 3.9 / 3.12 real PDF execution smoke；
4. Python 3.12 real PaddleX CPU OCR smoke。

CPU OCR smoke 会真实：

```text
install PaddlePaddle 3.2.2 CPU
install paddlex[ocr-core] 3.7.x
download official PaddleX demo image
download PP-OCRv6 small det/rec
run paddle-batch-ocr ocr --json
parse the summary as strict JSON
open the produced OCR JSON
parse it again through paddle_batch_ocr.ocr_schema
assert recognized lines > 0
```

因此当前 OCR execution layer 不只是 fake Result / mock pipeline 测试通过，而是已经穿过官方模型的真实 CPU 推理。

## Legacy 实现索引

| 文件 | 当前用途 | 状态 |
| --- | --- | --- |
| `pdf_to_png.py` | PDF 多进程拆图 | legacy / 新 `render` replacement 已存在 |
| `highocr3_f2.py` | 图片目录 OCR | legacy |
| `highocr3_f2_pdf.py` / `highocr3_f2_pdf2.py` | PDF OCR 早期实现 | legacy |
| `highocr4_f1_pdf_img.py` | PDF / 图片统一批处理 | legacy / 当前 OCR 主要参考实现 |
| `pdf_creator_with_text_layer5.py` / `6.py` / `7.py` | OCR JSON -> searchable PDF | legacy / 新 execution path 已存在 |
| `pdf_searchable2.5.py` / `pdf_searchable3.py` | searchable PDF 实验实现 | legacy |
| `del_10min_cache.py` | PaddleX cache 维护实验 | legacy / destructive |
| `OCR.yaml` / `OCR2.yaml` | 历史 PaddleX 配置 | legacy configuration |

数字文件名只是历史迭代痕迹，不是稳定 API 或 release 版本。

## 第三方许可证

仓库本身使用 **GNU GPL-3.0**。PDF optional extra 使用 PyMuPDF 与 Pillow；PyMuPDF 当前上游采用 GNU AGPL / commercial 双许可边界，分发组合应用时请自行核对相应义务。

## 下一阶段

重点不再是“证明 OCR 能跑”，而是：

- 一个 process worker 初始化一次 PaddleX pipeline；
- 明确 multiprocessing start method 与 GPU device assignment；
- render / OCR / searchable-PDF 的统一 `run` orchestration；
- 更完整 crash-safe resume；
- GPU self-hosted/manual smoke；
- OCR/Paddle dependency 与模型缓存，降低公共 CI 重复下载成本；
- geometry golden fixtures（中文长文本、旋转、双栏等）。

详细进度见 [`ROADMAP.md`](ROADMAP.md)。
