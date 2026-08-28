# Manifest operations

SQLite manifest 是大规模任务的运行状态索引。它不是 OCR JSON 本身，也不是完整事件日志；当前每个 `(source_path, stage)` 只保存**最新状态**和累计 retry count。

本页说明只读查询接口、retry provenance，以及当前数据能回答什么、不能回答什么。

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
- 当前记录的 `duration_s` 总和；
- 已有 intended-result provenance 的 job 数；
- 已有 execution-profile provenance 的 job 数。

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
  "duration_total_s": 12345.5,
  "provenance": {
    "intended_result_count": 119990,
    "execution_profile_count": 119800
  }
}
```

### Duration semantics

`duration_total_s` 是当前 jobs 表里每条记录现存 `duration_s` 的和。

manifest 当前**不是 append-only attempt/event log**，因此它不能重建每一次历史 retry 的耗时。失败任务再次失败时，`retry_count` 会增加，而 `duration_s` / error fields 保存最新一次记录。

## Retry provenance

新 writer 会在 jobs 表中维护：

```text
intended_result_path
execution_profile_json
```

二者和 `result_path` 的语义不同：

- `result_path`：已经成功发布的结果；失败时仍然清空；
- `intended_result_path`：本次任务准备写到哪里，即使失败也保留；
- `execution_profile_json`：本次请求的、影响执行/结果语义的配置。

这使 failed row 可以开始回答：

```text
原始 source 是什么？
失败发生在哪个 stage？
原本准备写到哪里？
当时请求了什么执行配置？
```

### OCR profile

OCR profile 当前记录：

- pipeline identity；
- device；
- inference engine；
- HPIP flag；
- document-orientation / unwarping / text-line-orientation predict options。

命名 pipeline 记录名称：

```json
{
  "pipeline": {
    "type": "name",
    "value": "OCR"
  }
}
```

本地 pipeline YAML 不只记录路径，还记录文件内容 fingerprint：

```json
{
  "pipeline": {
    "type": "file",
    "path": "/project/OCR.yaml",
    "size": 1234,
    "sha256": "..."
  }
}
```

因此同一路径的 YAML 原地修改后，known-profile success 会被判 stale；单纯文件时间变化但内容未变不会制造假 stale。

### PDF profiles

render profile 当前记录 DPI、PNG、alpha 等公开执行语义。

searchable-PDF profile 除字体、legacy-v7 layout 和 y-offset 外，还保存：

```text
images_dir
ocr_json_dir
expected_page_count
```

这样 failed searchable-PDF row 不需要靠外部记忆猜中间输入目录。

## Backward-compatible migration

writer `ManifestStore` 第一次打开旧 schema 时会原地增加 provenance 列。

迁移使用 `BEGIN IMMEDIATE`，避免多个 spawn worker 同时第一次打开旧 manifest 时竞态执行同一条 `ALTER TABLE`。

历史 success 的 `result_path` 可以作为 intended output 的可靠证据，因此会安全回填：

```text
intended_result_path = result_path
```

历史 execution profile **不会猜**。旧成功记录的 `execution_profile_json` 保持 `NULL`，也不会仅因为项目升级就强制全量重跑。

只有一个记录已经有可信 profile 后，后续 requested profile 变化才参与 stale detection。

只读 `manifest report/jobs` 不会迁移旧库。它会通过 `PRAGMA table_info` 检测 provenance 列是否存在；对完全未迁移的旧数据库，新字段按 `null` / coverage 0 展示。

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

JSON 同时返回：

```json
{
  "count": 100,
  "total_matching": 237,
  "filters": {
    "status": "failed",
    "stage": null,
    "error_class": null,
    "limit": 100,
    "offset": 200
  },
  "jobs": []
}
```

其中 `count` 是当前页实际返回的行数，`total_matching` 是忽略 limit/offset 后符合过滤条件的总数。agent 因此可以精确计算是否还有下一页，而不必一直翻到空结果。

每个 job JSON 现在还会包含：

```text
intended_result_path
execution_profile_json
```

`execution_profile_json` 保留数据库中的 canonical JSON 字符串，方便 CSV、SQLite 和未来 retry engine 共享同一个稳定表示。

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
intended_result_path
execution_profile_json
```

CSV 使用标准 quoting，因此 error message 或 profile JSON 含逗号、引号或换行时仍可被标准 CSV parser 读取。CSV 只输出行数据；分页元信息保留在 JSON/table 模式。

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

aggregate report 的多个 SELECT 在一个显式 read transaction 内执行；paged jobs 的 `COUNT(*)` 和 `LIMIT/OFFSET` 行也来自同一 SQLite snapshot。运行中一边有 worker 写 manifest、一边查询时，单次响应内部仍然自洽。

## Why `retry-failed` is still a separate next step

provenance 现在已经解决了“失败任务原本要写到哪里、请求了什么配置”这个前置问题，但通用 retry engine 仍需要再做一层严格执行合同：

- 只选择带可信 provenance 的 failed rows；
- 对 OCR 本地 pipeline 文件重新校验 SHA-256，避免拿已经变掉的 YAML 冒充历史 profile；
- 对 render/searchable-PDF 验证 stage-specific profile schema；
- 默认不覆盖现有目标；
- 明确 dry-run / execute 边界；
- 将 retry 结果重新写回同一 manifest；
- 对旧 profile / unknown profile 明确列为 ineligible，而不是猜。

因此本阶段先把 retry 所需的事实保存正确，再单独实现可审计的 targeted retry，而不是把 schema migration 和执行器塞进同一个大补丁。

后续适合增加：

- targeted retry selection / dry-run / execute；
- attempt/event history（如果需要完整 retry 时间线）；
- stale-running detection；
- 大规模 manifest 查询索引和 benchmark；
- 更完整的 stage dependency/content fingerprints。
