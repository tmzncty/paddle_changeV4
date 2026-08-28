# Project `run` orchestration

`paddle-batch-ocr run --config CONFIG` 把已经分别验证过的 render、OCR、searchable-PDF execution layer 收成一个确定性项目工作流。

它不是新的 OCR implementation；它负责 stage 排序、artifact layout、manifest 生命周期与跨 stage resume 规则。

## Command

```bash
paddle-batch-ocr run --config examples/config.json

paddle-batch-ocr run \
  --config examples/config.json \
  --dpi 144 \
  --json
```

`--json` 模式与 OCR 命令一样，在第三方 OCR inference 期间保护 stdout，最终只输出项目 summary。

## Input source behavior

`input_sources` 按配置顺序编号：

```text
source-001
source-002
...
```

### PDF source

PDF 文件 source 或 PDF directory 会执行：

```text
render -> OCR -> searchable-PDF
```

目录递归扫描按稳定排序处理。

### Image source

image source 当前只执行 OCR：

```text
image(s) -> OCR JSON
```

不会假装为普通图片 source 自动生成 searchable PDF。

## Artifact layout

PDF source：

```text
<output_root>/
  source-001/
    pdf/
      <relative-document-without-.pdf>/
        pages/
          page_00001.png
          page_00002.png
          ...
        ocr/
          page_00001_result.json
          page_00002_result.json
          ...
        searchable.pdf
```

PDF 目录中的相对层级会保留，因此不同子目录中的同名 PDF 不会挤到一个 artifact directory。

Image source：

```text
<output_root>/
  source-002/
    image/
      ocr/
```

OCR 自身继续保留输入目录相对结构。

## Config fields currently consumed

`run` 当前使用：

- `input_sources`
- `output_root`
- `log_dir`
- `manifest_path`
- `paddle_config`
- `runtime.device`
- `runtime.ocr_workers`
- `resume`
- `overwrite`

`--dpi` 当前是 command-level 参数。

### Important non-claim

虽然配置模型已经有：

```text
pdf_prep_workers
render_workers
batch_size
```

当前 `run` **没有把这些字段全部变成真实并发执行能力**。PDF document orchestration、render、searchable-PDF 仍按确定性顺序执行。当前真正接入 process pool 的是 OCR stage 的 `ocr_workers`。

## Render resume semantics

render stage 以 `(pdf_path, "render")` 记录 manifest。

如果 `pages/` 已存在：

- `resume=true` 且 manifest 不认为 source stale 时，可以验证后 adopt；
- manifest 首次见到但已有完整合法 render，也可以验证后收编；
- 验证会检查页序列和 PDF page count；
- invalid / stale existing pages 在 `overwrite=false` 时明确失败；
- `overwrite=true` 时允许重新事务式 render。

render 仍使用 staging directory，整本完成后才发布最终目录。

## OCR resume semantics

OCR stage 仍是逐图片 / 逐页 manifest record：

```text
(source_page_image, "ocr")
```

`runtime.ocr_workers=1` 使用串行合同；`ocr_workers>1` 使用 spawn CPU worker pool。

多 worker 时 parent 会先完成 resume / stale preflight，只有真正需要 inference 的页才进入 process pool。

## Searchable-PDF resume semantics

searchable stage 以：

```text
(pdf_path, "searchable_pdf")
```

记录 manifest。

已有 `searchable.pdf` 只有在验证 page count 成功、且当前依赖状态允许时才 adopt。

### OCR -> searchable dependency invalidation

仅依靠源 PDF 的 size/mtime **不足以证明**已有 searchable PDF 对应当前 OCR JSON。

因此当前有一条显式规则：

> 只要本轮 OCR 有任何 task 状态为 `success`，已有 searchable PDF 就视为下游 stale。

如果 stale searchable PDF 已存在：

- `overwrite=false`：项目明确失败并要求开启 overwrite；
- `overwrite=true`：重新构建 searchable PDF。

只有 OCR 本轮全部是 `skipped` / adopted 时，searchable stage 才允许按自身 manifest + page-count validation 安全跳过。

这避免出现：

```text
new OCR JSON + old searchable PDF text layer
```

的跨 stage 不一致。

## Failure isolation

`run_project()` 不会因为第一份文档失败就丢掉所有后续 item 的结果。

每个 PDF document / image source 都产生 `ProjectItemResult`，包含：

- source
- kind
- status
- pages_dir（适用时）
- ocr_dir
- searchable_pdf（适用时）
- error

最终 `ProjectRunResult` 汇总 success / failed count。CLI 只要存在 failed item 就返回非零。

## Real project test

PDF smoke suite 已加入 project-level round-trip：

```text
synthetic PDF
  -> real render
  -> deterministic fake OCR
  -> real searchable-PDF build
  -> reopen final PDF
  -> verify hidden Chinese text
```

这个测试不代替真实 PaddleX smoke；它专门验证 orchestration 与真实 PDF stages 的连接方式。

真实 PaddleX 则由 OCR serial + two-worker gates 单独覆盖。

## Current concurrency boundary

项目配置若使用：

```json
{
  "runtime": {
    "device": "cpu",
    "ocr_workers": 4
  }
}
```

PDF render 后的页面 OCR 会进入 spawn worker pool。

以下配置当前会被拒绝用于多 worker：

```text
device=auto
device=gpu
device=gpu:0
```

原因是 GPU worker pool 需要明确 device assignment，而不是把多个模型副本默认放进同一张 GPU。

## Remaining work

- richer stage dependency/content fingerprints；
- targeted rerun of failed items from manifest；
- GPU worker device map；
- GPU manual/self-hosted validation；
- render/document-level parallelism；
- project aggregate statistics；
- large real-corpus end-to-end comparison before legacy archival。
