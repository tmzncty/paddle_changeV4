# Targeted retry

`manifest retry-failed` 用 manifest 中已经验证过的 provenance 精确选择失败任务。它的默认行为是 **dry-run**；没有 `--execute` 时不会运行 OCR、render 或 searchable-PDF，也不会修改任务状态。

## Dry-run first

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json
```

机器可读：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --json
```

计划中的候选分成：

- `eligible`：provenance、source、target 和执行依赖都能重新验证；
- `blocked`：provenance 可用，但当前 target 已存在，需要显式 `--overwrite`；
- `ineligible`：元数据不完整、配置漂移、路径越界、source 变化等，自动重跑被拒绝。

Dry-run 即使包含 blocked / ineligible 项也返回成功状态，因为它是在报告计划，而不是宣称任务执行成功。

## Execute explicitly

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --execute
```

需要覆盖已有目标时必须再显式授权：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --overwrite \
  --execute
```

第一版执行是**顺序执行**。失败重跑通常只包含少量异常项，先保持审计性与状态语义明确；后续如果真实工作负载证明有必要，再在同一合同外增加受控并发。

## Selection

支持：

```text
--stage ocr|render|searchable_pdf
--error-class NAME
--limit N
--offset N
```

例如：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --error-class PaddleXResultError \
  --limit 50 \
  --json
```

## Safety boundary

manifest row 被当作**需要验证的输入**，而不是文件系统写权限的授权凭据。

自动 retry 只允许 intended target 位于当前 config 的 `output_root` 内。目标本身或现有路径组件包含 symlink 时拒绝。source 必须仍是同一个普通文件，size / mtime 必须与失败记录一致。

这意味着旧的 direct `ocr --manifest` 如果曾把结果写到当前 project `output_root` 之外，不会被这个自动 executor 接管。可以继续手工使用原 OCR 命令处理；项目不会为了“自动化更多”而让一条可篡改 SQLite row 获得任意路径覆盖能力。

## OCR reproducibility

OCR 自动重跑要求 profile schema 2，并且要求 pipeline identity 是**本地文件 + SHA-256 provenance**：

```json
{
  "type": "file",
  "path": "/absolute/config.yaml",
  "size": 1234,
  "sha256": "..."
}
```

执行前会重新验证文件存在、不是 symlink、size 一致、SHA-256 一致。

只有 pipeline 名称（例如 `"OCR"`）的记录第一版会判定为 ineligible。原因是 PaddleX 版本或远端 registry 变化后，同一个名字不保证解析到完全相同的模型/配置，不能称为“精确重跑”。

OCR profile 还会验证：

- explicit device（当前允许 `cpu` 或明确 `gpu:N`；`auto` / 裸 `gpu` 不可复现）；
- engine / HPIP 类型；
- 三个当前 General OCR predict/preprocess boolean；
- intended JSON 必须仍符合项目的 deterministic `<stem>_result.json` 映射。

## Render retry

render retry 当前只接受项目已经记录的 profile：

```text
schema=1
kind=pdf_render
format=png
alpha=false
dpi=36..1200
```

它会用失败时记录的 DPI 重新执行事务式 PDF render。

## Searchable-PDF retry

searchable-PDF retry 会重新验证 profile 中记录的：

- `images_dir`；
- `ocr_json_dir`；
- `expected_page_count`；
- `fontname`；
- `y_offset`；
- frozen `legacy-v7` layout。

两个 intermediate 目录必须仍位于 project `output_root` 内且不含 symlink。页序列数量必须一致，每一页必须找到可解析的 OCR JSON。

## Time-of-check / time-of-use protection

Dry-run 之后到真正执行之间，别的 worker 或人工操作可能改变 manifest 或文件系统。

因此 `--execute` 在**每个候选真正动作前**再次检查：

- row 仍然是 `failed`；
- source fingerprint 没变；
- intended target 没变；
- canonical execution profile 没变；
- source 文件没变；
- target 没突然变成 symlink / 已存在冲突。

计划不是长期授权令牌。

## Exit codes

- dry-run：计划成功生成时返回 `0`，即使有 blocked / ineligible；
- execute：只要实际执行项有 failure，返回非零；
- 参数、配置或顶层安全错误：CLI 使用标准错误退出。

`--json --execute` 与 OCR/run 一样隔离 Paddle / oneDNN 的 Python 与 native fd 1 输出，stdout 保持严格 JSON。
