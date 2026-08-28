# Targeted retry

`manifest retry-failed` 用 manifest 中已经验证过的 provenance 精确选择失败任务。默认行为是 **dry-run**；没有 `--execute` 时不会运行 OCR、render 或 searchable-PDF，也不会修改任务状态。

这条能力面向“主任务几十万页里只有少量异常项”的恢复场景。第一版刻意优先可审计、可恢复和可解释，而不是自动无限重试或追求最大并发。

## Quick start

先看计划：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json
```

只看 OCR，并输出机器可读 JSON：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --json
```

只看某一错误类：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --error-class PaddleXResultError \
  --json
```

确认计划后显式执行：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --execute
```

只有 dry-run 已确认目标只是被现有文件挡住，并且你确实要替换它时，再显式增加：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --overwrite \
  --execute
```

`--overwrite` 只授权替换目标，不会绕过 provenance、source、`output_root`、symlink 或 pipeline SHA-256 检查。

## Candidate states

计划中的候选分成三类：

- `eligible`：provenance、source、target 和执行依赖都能重新验证；
- `blocked`：provenance 可信，但 intended target 已存在，需要显式 `--overwrite`；
- `ineligible`：元数据不完整、source/config 漂移、路径越界或 profile 不受支持，自动重跑被拒绝。

Dry-run 即使包含 blocked / ineligible 也返回 `0`，因为它是在成功生成计划，而不是宣称任务已修复。

## Selection and pagination

支持：

```text
--stage ocr|render|searchable_pdf
--error-class NAME
--limit N
--offset N
```

例如分页查看：

```bash
paddle-batch-ocr manifest retry-failed \
  --config project.json \
  --stage ocr \
  --limit 100 \
  --offset 200 \
  --json
```

执行前仍建议基于最新 manifest 重新 dry-run；计划不是长期授权令牌。

## Trust boundary

manifest row 被当作**待验证数据**，不是文件系统写权限的授权凭据。

自动 retry 只允许 intended target 位于当前 config 的 `output_root` 内。目标本身或现有路径组件包含 symlink 时拒绝。source 必须仍是同一个普通文件，size / mtime 必须与失败记录一致。

这意味着旧的 direct `ocr --manifest` 如果曾把结果写到当前 project `output_root` 之外，不会被这个自动 executor 接管。可以继续手工使用原 OCR 命令处理；项目不会为了“自动化更多”而让一条可篡改 SQLite row 获得任意路径覆盖能力。

配置加载也会保留显式 manifest 路径的 symlink 身份，使 SQLite reader / writer 能在 open 前真正拒绝 symlink，而不是先 `resolve()` 后丢失安全信息。

## OCR reproducibility

OCR 自动重跑要求 profile schema 2，并要求 pipeline identity 是**本地文件 + SHA-256 provenance**：

```json
{
  "type": "file",
  "path": "/absolute/config.yaml",
  "size": 1234,
  "sha256": "..."
}
```

执行前重新验证：

- pipeline 文件仍存在且不是 symlink；
- size 一致；
- SHA-256 一致；
- explicit device 可复现；
- engine / HPIP 类型合法；
- 当前三个 General OCR predict/preprocess boolean 完整；
- intended JSON 仍符合 deterministic `<stem>_result.json` 映射。

只有 pipeline 名称（例如 `"OCR"`）的记录第一版会判定为 ineligible。PaddleX 版本或远端 registry 变化后，同一个名字不保证解析到完全相同的模型/配置，因此不能称为“精确重跑”。

`auto` 和裸 `gpu` 也不会被自动 retry 接受为可复现设备；当前自动恢复要求 `cpu` 或明确 `gpu:N`。

## Render retry

render retry 当前接受项目已经记录的 profile：

```text
schema=1
kind=pdf_render
format=png
alpha=false
dpi=36..1200
```

执行时恢复失败记录中的 DPI，并继续使用事务式 PDF renderer。

## Searchable-PDF retry

searchable-PDF retry 会重新验证 profile 中记录的：

- `images_dir`；
- `ocr_json_dir`；
- `expected_page_count`；
- `fontname`；
- `y_offset`；
- frozen `legacy-v7` layout。

两个 intermediate 目录必须仍位于 project `output_root` 内且不含 symlink。页序列数量必须一致，每一页必须找到可解析的 OCR JSON，才允许重建最终 PDF。

## Time-of-check / time-of-use protection

Dry-run 之后到真正执行之间，别的 worker 或人工操作可能改变 manifest 或文件系统。因此 `--execute` 在**每个候选真正动作前**再次检查：

- row 仍然是同一个 `failed` attempt；
- source fingerprint 没变；
- intended target 没变；
- canonical execution profile 没变；
- source 文件没变；
- target 没突然变成 symlink 或新增冲突。

如果计划后状态变化，执行器拒绝该项，而不是把旧计划当成长期授权。

## Execution model

第一版 targeted retry **顺序执行**。失败集合通常远小于主任务集合，而且 targeted retry 的第一目标是可审计、可恢复、可解释。只有真实失败工作负载证明有必要后，才应在同一安全合同外增加受控并发。

明确不做：

- 不自动无限重试；
- 不绕过 `output_root` 写任意路径；
- 不把 named pipeline 当成可复现模型身份；
- 不让 `--overwrite` 绕过安全检查；
- 不把 dry-run 结果永久缓存成执行授权；
- 不在第一版把失败恢复再变成一个多进程调度器。

## Exit codes

- dry-run：计划成功生成时返回 `0`，即使存在 blocked / ineligible；
- execute：只有选中范围内的候选都被成功修复时才返回 `0`；只要仍有实际 failure 或 ineligible / blocked 候选未解决，就返回非零；
- 参数、配置或顶层安全错误：CLI 使用标准错误退出。

这样自动化不会因为“修好了 9 条、还有 1 条无法安全自动重跑”而误判整批恢复完成。

`--json --execute` 与 OCR/run 一样隔离 Paddle / oneDNN 的 Python 与 native fd 1 输出，stdout 保持严格 JSON。

## Real PP-OCRv6 validation

公共 CI 不只测试 planner。真实 CPU gate 会：

1. 用官方 OCR demo 图制造一条 provenance 完整的 `failed` OCR row；
2. 运行 dry-run，确认 manifest 仍是 failed、`retry_count` 不变、目标文件不存在；
3. 用 `--execute --json` 真正启动 PP-OCRv6 small det/rec；
4. 重新解析生成的 OCR JSON；
5. 确认 manifest 变为 `success`，`result_path` / intended target 一致，原 retry counter 与 execution profile 保留。

当前验证日志包含：

```text
targeted_retry_fixture=PASS
targeted_retry_dry_run=PASS
targeted_retry_recognized_lines=4
targeted_retry_real_ocr=PASS
```

验证矩阵与普通 OCR gate 相同：Ubuntu 24.04、Python 3.12、PaddlePaddle CPU 3.2.2、PaddleX 3.7.2、PP-OCRv6 small detection / recognition。
