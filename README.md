# paddle_changeV4

面向**大规模中文 OCR、PDF 批处理与可搜索 PDF 重建**的 PaddleOCR / PaddleX 流水线实验项目。

> 当前仓库正在从“长期迭代的个人脚本集合”重构为可复现、可配置、可测试的公开项目。历史脚本继续保留以保存已经验证过的处理经验；新的安全层、配置系统、manifest 和 CLI 已经落在 `src/paddle_batch_ocr/`。

## 项目要解决什么

这个项目最初服务于几十万页规模的文献数字化处理，重点不是单张图片 OCR，而是把完整工作流跑稳定：

1. 批量扫描 PDF / 图片目录；
2. 将 PDF 页面渲染为图像；
3. 使用 PaddleX / PaddleOCR 执行 OCR；
4. 保存逐页 JSON 结果；
5. 根据 OCR 坐标重建带隐藏文本层的 searchable PDF；
6. 在大量文件、长时间运行、GPU / CPU / 内存 / 磁盘 I/O 同时受压时支持跳过、错误留档和恢复。

现有 legacy 代码来自真实的大批量处理环境，因此包含很多关于吞吐量、缓存、并发、异常页和中文深层路径的经验。但其中不少机器参数曾直接硬编码在源码里，**不能视为通用安全默认值**。

## 当前状态

**Status: legacy pipeline preserved / public refactor active**

现在仓库同时存在两层：

- `src/paddle_batch_ocr/`：新的可安装 package、安全配置、诊断、manifest 和 CLI；
- 根目录历史 `.py`：尚未迁移完成的 OCR / PDF 生产逻辑。

### 已经可用的新 CLI

项目现在可以作为 Python package 安装，而不要求安装 Paddle：

```bash
python -m pip install --no-deps .
paddle-batch-ocr --version
paddle-batch-ocr doctor
```

`--no-deps` 是有意设计：Paddle CPU/GPU/CUDA 环境暂时不强行绑定到项目自身依赖中。

当前可用命令：

```bash
# 无配置也能检查当前机器
paddle-batch-ocr doctor
paddle-batch-ocr doctor --json

# 带项目配置检查输入、磁盘、Paddle 配置等
paddle-batch-ocr doctor --config examples/config.json

# OCR 前只扫描并统计输入，不执行模型
paddle-batch-ocr scan --config examples/config.json

# 查看 SQLite manifest 状态；如果 manifest 不存在，不会创建空数据库
paddle-batch-ocr manifest status --config examples/config.json
paddle-batch-ocr manifest status --config examples/config.json --json

# 默认 dry-run：只告诉你准备清理哪个 temp cache
paddle-batch-ocr cache clean --config examples/config.json

# 只有显式给出 --execute 才真正删除 <cache_root>/temp
paddle-batch-ocr cache clean --config examples/config.json --execute
```

`doctor` 会尽量报告：

- Python / 平台；
- CPU 数量与物理内存；
- Paddle / Paddle GPU / PaddleX / PaddleOCR / PyMuPDF / Pillow 安装版本；
- `nvidia-smi` 可见 GPU、显存和驱动；
- output / log / cache 对应磁盘剩余空间；
- 输入或 Paddle 配置缺失；
- 明显过高的 worker / batch 参数。

### 项目配置

推荐从 [`examples/config.json`](examples/config.json) 开始：

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
    "device": "auto",
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

配置中的相对路径以**配置文件所在目录**为基准解析。若未指定 `manifest_path`，默认使用：

```text
<log_dir>/manifest.sqlite3
```

新的默认原则是保守而不是追求 benchmark：

- worker 默认都是 `1`；
- batch size 默认 `1`；
- `overwrite=false`；
- cache 删除默认 dry-run；
- output / log / cache / manifest 不允许位于输入目录内部；
- output 与 cache 不允许互相嵌套；
- log 与 cache 不允许重叠；
- manifest 不允许放进 cache root；
- cache root 不能是 filesystem root、用户 home 或当前工作目录；
- destructive path 必须先经过真实路径 containment 检查。

JSON 配置完全使用标准库即可读取。YAML 配置是可选支持，需要：

```bash
python -m pip install '.[yaml]'
```

## Manifest 与恢复

新的 manifest 使用 Python 标准库 SQLite，以 `(source_path, stage)` 为键记录任务状态。当前核心能力包括：

- source size / mtime fingerprint；
- `pending / running / success / failed` 状态；
- result path；
- retry count；
- error class / message；
- worker / device；
- started / finished 时间与 duration；
- WAL + busy timeout，允许多个 worker 各自打开连接；
- 已成功任务的结果文件消失时重新运行；
- 源文件 size/mtime 变化时自动把旧 success 失效为 pending。

这比 legacy 的“目标文件存在就跳过”更适合长时间、几十万页级任务。

## OCR JSON / searchable-PDF 兼容层

重构已经把几条历史隐含协议变成了独立模块和测试：

- `paddle_batch_ocr.naming` 保存 v7 的多种 page JSON 文件名匹配优先级；
- `paddle_batch_ocr.ocr_schema` 同时接受历史 `rec_text` 与较新 `rec_texts`；
- `paddle_batch_ocr.layout` 冻结 v7 的两栏排序和 polygon 0/2 点文本矩形行为；
- `paddle_batch_ocr.io_utils.atomic_write_json` 用同目录临时文件、fsync 和原子发布写 OCR JSON，默认不覆盖已有结果。

详细 legacy 行为见 [`docs/LEGACY_BEHAVIOR.md`](docs/LEGACY_BEHAVIOR.md)。

## 尚未迁移的新功能

新的 CLI **目前还不会执行 OCR 或 searchable-PDF 重建**。以下命令仍属于 roadmap，而不是当前稳定入口：

```text
paddle-batch-ocr ocr
paddle-batch-ocr render
paddle-batch-ocr searchable-pdf
paddle-batch-ocr run
```

在新的 OCR engine、PDF adapter 和回归测试覆盖旧行为之前，不会为了“看起来完成”而直接删除历史实现。

## Legacy 实现索引

| 文件 | 当前用途 | 状态 |
| --- | --- | --- |
| `pdf_to_png.py` | PDF 多进程拆图 | legacy |
| `highocr3_f2.py` | 图片目录 OCR | legacy |
| `highocr3_f2_pdf.py` / `highocr3_f2_pdf2.py` | PDF OCR 早期实现 | legacy |
| `highocr4_f1_pdf_img.py` | PDF / 图片统一批处理 | **当前 OCR 主要参考实现** |
| `pdf_creator_with_text_layer5.py` / `6.py` / `7.py` | OCR JSON → 带文本层 PDF | legacy，`7` 为较新版本 |
| `pdf_searchable2.5.py` / `pdf_searchable3.py` | searchable PDF 实验实现 | legacy |
| `del_10min_cache.py` | PaddleX 临时缓存维护实验 | legacy / destructive |
| `OCR.yaml` / `OCR2.yaml` | PaddleX OCR 配置样例 | legacy configuration |

这些文件名中的数字只是历史迭代痕迹，不代表稳定 API 或正式 release。

如果要阅读现有 OCR 生产逻辑，优先看 `highocr4_f1_pdf_img.py`；如果要研究 OCR JSON → searchable PDF 的实现，优先看 `pdf_creator_with_text_layer7.py`。

**不要直接运行 legacy 默认配置。** 多个旧脚本仍包含 `/media/tmzn/...` 的机器路径、高并发参数以及递归缓存处理。

## 安全边界

新实现把安全问题当成程序语义，而不是 README 警告：

- cache cleanup 只允许作用于配置的 `<cache_root>/temp`；
- cache root 本身不会被递归删除；
- `/`、用户 home、当前工作目录等受保护路径会被拒绝；
- containment 使用真实路径关系，不使用字符串前缀，因此 `/data/cache-evil` 不会被误认为 `/data/cache` 的子目录；
- symlink 解析后越界也会被拒绝；
- 删除默认 dry-run，必须显式 `--execute`；
- 输入、输出、日志、cache、manifest 的危险路径重叠会在配置加载时直接报错。

这并不意味着 legacy 脚本已经获得这些保护。直到迁移完成前，根目录旧脚本仍应视为高级用户参考实现。

## 历史运行环境与 CI

旧代码主要在类似以下环境中开发和使用：Ubuntu 22.04、Python 3.9.x、CUDA 11.8、Paddle/PaddleX GPU 推理以及大容量本地磁盘环境。

这只是历史环境，不是当前兼容性承诺。新的公共基线 CI 目前覆盖 Python 3.9 和 3.12 的：

- legacy Python 语法编译；
- 新 package 全量 compile；
- dependency-free 单元测试；
- `pip install --no-deps .`；
- CLI `--version`；
- `doctor --json`。

Paddle、PaddleX、CUDA 的 CPU/GPU 矩阵将在独立 smoke / self-hosted 测试中建立，不让公共 CI 假装拥有 GPU 环境。

## 新代码结构

```text
src/paddle_batch_ocr/
  __init__.py
  cache.py
  cli.py
  config.py
  discovery.py
  doctor.py
  io_utils.py
  layout.py
  manifest.py
  naming.py
  ocr_schema.py
  safety.py
```

后续会继续增加 `ocr.py`、`pdf_render.py`、`searchable_pdf.py`、`progress.py` 等执行层。详细阶段见 [`ROADMAP.md`](ROADMAP.md)。

## 重构原则

1. **先保持行为，再整理结构**：没有 fixtures / tests 前不删已验证的旧逻辑。
2. **安全默认**：递归删除、覆盖、高并发必须显式开启。
3. **配置与代码分离**：机器路径、GPU 数量、数据集路径不再写死在源码中。
4. **可恢复**：几十万页任务用 manifest 识别输入变化、结果丢失和失败状态。
5. **可观测**：吞吐量、错误、跳过、耗时、资源配置要成为结构化数据。
6. **可复现**：分开维护项目依赖与 Paddle/CUDA 环境矩阵。
7. **文献数字化是一等公民**：中文路径、深目录、大 PDF、异常页和海量文件不能在“重构得漂亮”时被牺牲。

## 贡献

当前最有价值的贡献包括 legacy 行为 fixture / golden output、Paddle OCR result schema adapter、PDF 坐标与隐藏文本层回归测试、CPU/GPU 可复现环境报告，以及大任务 resume/manifest。参见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## License

本项目使用 **GNU General Public License v3.0 (GPL-3.0)**，详见 [`LICENSE`](LICENSE)。

## 项目来源与说明

`tmzncty/paddle_changeV4` 保留 `Get-data-all/paddle_change` 的 fork 关系、Git 历史和许可证信息。公开重构不会为了让历史“看起来整齐”而抹掉已有来源记录。
