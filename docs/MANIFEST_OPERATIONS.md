# Manifest operations

SQLite manifest 是大规模任务的运行状态索引。它不是 OCR JSON 本身，也不是完整事件日志；当前每个 `(source_path, stage)` 只保存**最新状态**和累计 retry count。

本页说明只读查询接口，以及当前数据能回答什么、不能回答什么。

## Fast status

已有命令继续保留：

```bash
paddle-batch-ocr manifest status --config CONFIG
paddle-batch-ocr manifest status --config CONFIG --json
```

它只给全局 status count。

## Aggregate report

```bash
paddle-batch-ocr manifest report --config CONFIG
paddle-batch-ocr manifest report --config CONFIG --json
```

report 在 SQLite 中聚合：

- total job count；
- `pending / running / success / failed` count；
- 每个 stage 的 status matrix；
- 当前 error class count；
- `retry_count` 总和；
- 当前记录的 `duration_s` 总和。

示例 JSON 形状：

```json
{
  "manifest_path": "/work/logs/manifest.sqlite3",
  "exists": true,
  "total": 120000,
  "status": {
    "failed": 23,
    "running": 4,
    "success": 119973
  },
  "stages": {
    "ocr": {
      "failed": 20,
      "running": 4,
      "success": 119000
    },
    "render": {
      "failed": 3,
      "success": 973
    }
  },
  "error_classes": {
    "PaddleXResultError": 12,
    "OSError": 8,
    "PdfRenderError": 3
  },
  "retry_total": 41,
  "duration_total_s": 12345.5
}
```

### Duration semantics

`duration_total_s` 是当前 jobs 表里每条记录现存 `duration_s` 的和。

manifest 当前**不是 append-only attempt/event log**，因此它不能重建每一次历史 retry 的耗时。失败任务再次失败时，`retry_count` 会增加，而 `duration_s` / error fields 保存最新一次记录。

## Filtered jobs

默认 table：

```bash
paddle-batch-ocr manifest jobs --config CONFIG
```

只看失败：

```bash
paddle-batch-ocr manifest jobs \
  --config CONFIG \
  --status failed
```

只看 OCR 的某类错误：

```bash
paddle-batch-ocr manifest jobs \
  --config CONFIG \
  --stage ocr \
  --status failed \
  --error-class PaddleXResultError
```

分页：

```bash
paddle-batch-ocr manifest jobs \
  --config CONFIG \
  --status failed \
  --limit 100 \
  --offset 200 \
  --json
```

底层 query 提供 total matching count；JSON CLI 会逐步暴露该值，便于 agent 精确分页，而不是一直翻到空页才停止。

## CSV export

```bash
paddle-batch-ocr manifest jobs \
  --config CONFIG \
  --status failed \
  --csv > failed.csv
```

CSV 列与 jobs 表的公开查询字段对应：

```text
source_path
stage
source_size
source_mtime_ns
status
result_path
retry_count
error_class
error_message
worker
device
started_at
finished_at
duration_s
```

CSV 使用标准 quoting，因此 error message 含逗号、引号或换行时仍可被标准 CSV parser 读取。

## Read-only guarantee

`manifest report` / `manifest jobs` 不使用会初始化 schema 的 `ManifestStore`，而是单独使用 SQLite URI：

```text
mode=ro
```

并设置：

```text
PRAGMA query_only=ON
```

所以报告连接本身没有写任务状态的权限。

如果配置指向的 manifest 不存在，CLI 返回空 report/jobs，并且不会创建数据库。

symlinked manifest path 会在 resolve 前拒绝。

## Why there is no generic `retry-failed` command yet

当前 `mark_failure()` 会把 `result_path` 清空。这个字段表达的是“已成功发布的结果”，并不保存“这次失败任务原本准备写到哪里”。

因此仅凭一个 failed row，我们能可靠知道：

```text
source
stage
last error
retry count
worker/device
```

但不一定能对所有历史入口可靠恢复：

```text
intended output path
pipeline/options used for that attempt
```

在这些 provenance 进入 manifest 之前，通用 `retry-failed` 命令可能只对某些 project layout 正确。项目选择先提供精确观测，而不是发布一个部分正确的重跑按钮。

后续适合增加：

- intended result path / execution profile provenance；
- targeted retry selection；
- attempt/event history（如果需要完整 retry 时间线）；
- stale-running detection；
- JSON/CSV aggregate export；
- 大规模 manifest 查询索引和 benchmark。
