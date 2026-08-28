# Current Repository State

此文档记录重构开始时（2026-08）的仓库状态，用于避免后续整理过程中遗忘 legacy 行为和已知问题。

## Repository shape

当前仓库没有 Python package 目录，主要由根目录脚本组成：

```text
.gitignore
LICENSE
OCR.yaml
OCR2.yaml
README.md
del_10min_cache.py
highocr3_f2.py
highocr3_f2_pdf.py
highocr3_f2_pdf2.py
highocr4_f1_pdf_img.py
pdf_creator_with_text_layer5.py
pdf_creator_with_text_layer6.py
pdf_creator_with_text_layer7.py
pdf_searchable2.5.py
pdf_searchable3.py
pdf_to_png.py
```

不存在：

- `src/`
- `tests/`
- `pyproject.toml`
- dependency lock
- release-oriented changelog
- stable CLI
- public sample fixture

## Known correctness / reproducibility blockers

### 1. Missing `find_json_parent.py`

`pdf_creator_with_text_layer7.py` currently contains:

```python
from find_json_parent import find_unique_json_parent_paths
```

but the referenced module is not present in the repository tree.

因此该脚本不能在一次干净 clone 后按当前状态直接运行。重构时需要先恢复/重写并测试这段 discovery 逻辑，而不是在 README 中继续宣称代码可直接复现。

### 2. README historically referenced a missing script

旧 README 的 v2.0.0 usage section 使用：

```bash
python pdf_creator_with_text_layer7_copy.py ...
```

但仓库中没有该文件。

### 3. Machine-specific absolute paths

多个脚本包含开发机器专属路径，例如：

```text
/media/tmzn/DATA4/...
/media/tmzn/DATA5/...
```

这些路径需要迁移到配置文件 / CLI。

### 4. Aggressive concurrency values are embedded in source

历史脚本中可以看到诸如：

- OCR workers: 16
- PDF preparation workers: 8
- render workers: 24
- searchable PDF worker count: 32
- OCR batch sizes up to 32 / 64

这些数字来自特定硬件环境，不应成为公开项目的默认值。

### 5. Cache cleanup is coupled to normal execution

`highocr3_f2.py` 和后续脚本包含 PaddleX cache 清理逻辑；较新脚本还尝试管理 `PADDLEX_HOME` 与 `~/.paddlex/temp`。

当前主要风险：

- recursive deletion；
- symlink handling；
- cache path 与输入/输出路径没有统一 boundary validation；
- cache cleanup 与正常 OCR 启动耦合。

重构后应把 cache maintenance 变为单独、可预览、可拒绝危险路径的操作。

## Legacy implementation strengths worth preserving

代码虽然结构混乱，但并不是“全部推倒重来”的对象。以下经验值得保留：

### Batch-oriented workflow

现有实现关注的是成百上千 PDF / 大量图片，而不是 demo 级单文件调用。

### Resume by existing outputs

多个阶段会检查已有 JSON / 页面输出并跳过重复工作。这是长时间任务非常重要的行为，后续应升级为 manifest-based resume，而不是删除。

### Error isolation

部分 OCR 路径会把失败图片复制到错误目录，并记录逐文件错误，避免单页错误直接终止整批数据。

### Pipeline initialization per worker

OCR worker 使用 initializer 创建 PaddleX pipeline，避免对每张图片重新初始化模型。这个方向应保留并明确生命周期。

### Mixed PDF / image inputs

`highocr4_f1_pdf_img.py` 已经尝试统一处理 PDF 与图片输入，是未来 unified CLI 的重要来源。

### Searchable PDF reconstruction

仓库里已经积累了多版 OCR coordinate → PDF text layer 的实现，这部分应通过 fixture 和 golden tests 固化，而不是重新凭感觉实现。

## Configuration drift

当前有两个 PaddleX 配置文件：

### `OCR.yaml`

偏保守：

```yaml
Pipeline:
  text_det_model: PP-OCRv4_mobile_det
  text_rec_model: PP-OCRv4_mobile_rec
  text_rec_batch_size: 1
```

### `OCR2.yaml`

偏 GPU / 高吞吐：

```yaml
Pipeline:
  text_det_model: PP-OCRv4_mobile_det
  text_rec_model: PP-OCRv4_mobile_rec
  text_rec_batch_size: 64
  device: "gpu:0"
```

未来配置应区分：

1. PaddleX pipeline config；
2. 本项目的 orchestration config（路径、并发、缓存、resume 等）。

不要把两者混成同一个配置模型。

## Proposed migration map

```text
pdf_to_png.py
  -> pdf_render.py

highocr3_f2.py
highocr3_f2_pdf.py
highocr3_f2_pdf2.py
highocr4_f1_pdf_img.py
  -> discovery.py + ocr.py + pipeline.py + progress.py

pdf_creator_with_text_layer5.py
pdf_creator_with_text_layer6.py
pdf_creator_with_text_layer7.py
pdf_searchable2.5.py
pdf_searchable3.py
  -> searchable_pdf.py + ocr_schema.py

del_10min_cache.py
  -> cache.py + safety.py + explicit CLI command

OCR.yaml / OCR2.yaml
  -> configs/paddlex/*.yaml
```

## First code milestone

第一轮真正的代码重构应尽量小：

1. 建立 package skeleton；
2. 实现不依赖 Paddle 的 config / path safety / discovery helpers；
3. 为这些 helper 写测试；
4. 给 legacy scripts 增加 wrapper 或逐步调用新 helper；
5. 在行为等价确认后再移动脚本。

这样可以避免一次“大重写”把已经处理几十万页时踩过的坑重新踩一遍。
