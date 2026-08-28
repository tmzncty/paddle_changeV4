# paddle_changeV4

面向**大规模中文 OCR、PDF 批处理与可搜索 PDF 重建**的 PaddleOCR / PaddleX 实验性流水线。

> 当前仓库正在从“长期迭代的个人脚本集合”重构为可复现、可配置、可测试的公开项目。现有脚本暂时保留，用于保存已经验证过的处理逻辑；新代码将逐步迁移到统一 CLI、配置系统和模块化实现中。

## 项目要解决什么

这个项目最初服务于几十万页规模的文献数字化处理，重点不是单张图片 OCR，而是把完整工作流跑稳定：

1. 批量扫描 PDF / 图片目录；
2. 将 PDF 页面渲染为图像；
3. 使用 PaddleX / PaddleOCR 执行 OCR；
4. 保存逐页 JSON 结果；
5. 根据 OCR 坐标重建带隐藏文本层的 searchable PDF；
6. 在大量文件、长时间运行、GPU / CPU / 内存 / 磁盘 I/O 同时受压时尽量支持跳过已完成任务、错误留档和恢复。

当前代码来自真实的大批量处理环境，因此包含不少针对吞吐量、缓存、并发和异常文件的经验性处理。但这些经验目前仍有相当一部分以硬编码形式存在，**不能直接视为通用安全默认值**。

## 当前状态

**Status: legacy scripts available / refactor in progress**

仓库目前主要包含若干历史脚本：

| 文件 | 当前用途 | 状态 |
| --- | --- | --- |
| `pdf_to_png.py` | PDF 多进程拆图 | legacy |
| `highocr3_f2.py` | 图片目录 OCR | legacy |
| `highocr3_f2_pdf.py` / `highocr3_f2_pdf2.py` | PDF OCR 的早期实现 | legacy |
| `highocr4_f1_pdf_img.py` | 同时处理 PDF / 图片的较新批处理实现 | **当前主要参考实现** |
| `pdf_creator_with_text_layer5.py` / `6.py` / `7.py` | OCR JSON → 带文本层 PDF | legacy，`7` 为较新版本 |
| `pdf_searchable2.5.py` / `pdf_searchable3.py` | searchable PDF 实验实现 | legacy |
| `del_10min_cache.py` | PaddleX 临时缓存维护实验 | legacy / destructive behavior requires review |
| `OCR.yaml` / `OCR2.yaml` | PaddleX OCR 配置样例 | legacy configuration |

这些文件名中的数字是历史迭代痕迹，不代表稳定 API 或正式 release。后续会把它们收束为模块和版本化发布，而不是继续通过文件名递增版本。

## 重要安全说明

现有 legacy 脚本是为特定机器和目录布局写的，运行前务必阅读源码并修改配置。

尤其需要注意：

- 多个脚本仍包含 `/media/tmzn/...` 形式的**个人绝对路径**；
- 部分脚本默认使用较高的多进程并发；
- OCR / PDF 处理可能产生非常大的临时数据和随机 I/O；
- 部分缓存维护逻辑会递归删除 PaddleX 临时目录；
- 当前没有对所有输入、路径和删除操作提供统一的 dry-run / confirmation 保护；
- 不应在不理解路径配置的情况下以高权限运行这些脚本。

公开重构版的首要目标之一，就是把这些行为改成**显式配置、安全默认值、路径边界检查和可预览操作**。

## 历史测试环境

旧代码主要在类似以下环境中开发和使用：

- Ubuntu 22.04
- Python 3.9.x
- CUDA 11.8 环境
- Paddle / PaddleX GPU 推理
- 大容量本地磁盘与高并发 CPU 环境

这只是历史工作环境，不是当前承诺的兼容性矩阵。Paddle、PaddleX 与 CUDA 的版本匹配请以各自当前官方文档为准。

主要第三方依赖包括：

- Paddle / PaddleX
- PyMuPDF (`fitz`)
- Pillow
- NumPy
- PyYAML
- pypdfium2
- colorama
- natsort
- psutil

重构完成前不会给出一个假装“所有机器都能直接 pip install”的锁定依赖文件，因为 CPU/GPU、CUDA 与 Paddle 版本组合需要单独处理。

## 现阶段如何使用

如果你只是想研究现有实现，建议从：

```text
highocr4_f1_pdf_img.py
```

开始阅读。它包含较完整的：

- PDF 与图片输入扫描；
- PaddleX pipeline 初始化；
- 多阶段并发处理；
- 已完成 JSON 跳过；
- PDF 页面准备；
- OCR 日志和错误文件留档；
- PaddleX 临时目录处理。

如果你想研究 OCR JSON → searchable PDF 的实现，则优先阅读：

```text
pdf_creator_with_text_layer7.py
```

**不要直接运行默认配置。** 先检查脚本顶部的路径、并发数、缓存目录、输入输出目录和删除行为。

## 重构方向

接下来不再继续制造新的 `*_v8_final2.py`，而是把现有经验拆成稳定组件：

```text
src/
  paddle_batch_ocr/
    cli.py
    config.py
    discovery.py
    ocr.py
    pdf_render.py
    searchable_pdf.py
    cache.py
    progress.py
    safety.py

tests/
configs/
docs/
```

计划提供一个统一入口，例如：

```bash
paddle-batch-ocr ocr --config config.yaml
paddle-batch-ocr render input.pdf --output pages/
paddle-batch-ocr searchable-pdf --images pages/ --ocr-json json/ --output book.pdf
```

配置优先级会逐步统一为：

```text
CLI 参数 > 配置文件 > 环境变量 > 安全默认值
```

详细计划见 [`ROADMAP.md`](ROADMAP.md)。

## 重构原则

1. **先保持行为，再整理结构**：旧脚本不会在没有等价测试前被直接删除。
2. **安全默认**：任何递归删除、覆盖、超高并发都不能是隐式默认行为。
3. **配置与代码分离**：不再把机器路径、GPU 数量、数据集路径写死在源码中。
4. **可恢复**：大规模任务必须能够识别已完成工作并从中断处继续。
5. **可观测**：吞吐量、错误、跳过数量、耗时和资源配置必须能够被记录。
6. **可复现**：明确 Python / Paddle / CUDA 兼容矩阵，并逐步建立测试与基准。
7. **保留文献数字化需求**：中文路径、深层目录、大型 PDF、异常页和几十万页级任务仍是一等公民。

## 贡献

目前最有价值的贡献不是继续增加一个脚本副本，而是：

- 把硬编码配置迁移到统一配置模型；
- 为路径映射、页码匹配、OCR JSON 解析等纯逻辑建立测试；
- 将 destructive cache 操作改为受约束的安全实现；
- 提供不同 Paddle / PaddleX / CUDA 组合的可复现环境报告；
- 用最小公开样本构建端到端测试。

参见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## License

本项目使用 **GNU General Public License v3.0 (GPL-3.0)**，详见 [`LICENSE`](LICENSE)。

## 项目来源与说明

`tmzncty/paddle_changeV4` 当前保留了 `Get-data-all/paddle_change` 的 fork 关系与历史。重构将继续保留 Git 历史与许可证信息，不通过复制粘贴的方式抹掉已有来源记录。
