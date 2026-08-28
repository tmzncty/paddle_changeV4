# Targeted retry examples

这些例子只展示公开 CLI 的安全用法。

## 1. 先看所有 failed rows

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json
```

没有 `--execute`，不会运行任何任务。

## 2. 只看 OCR

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --json
```

## 3. 只看某一错误类

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --error-class PaddleXResultError \
  --json
```

## 4. 执行 eligible 项

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --execute
```

## 5. 允许覆盖已有 intended target

只有 dry-run 已确认该项只是 `blocked`，并且你确实要替换目标时：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --overwrite \
  --execute
```

`--overwrite` 不会绕过 provenance / source / output_root / symlink / pipeline SHA-256 检查。

## 6. 分页处理大量失败

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --limit 100 \
  --offset 0 \
  --json
```

执行前仍建议重新 dry-run，因为 manifest 可能在运行过程中变化。
