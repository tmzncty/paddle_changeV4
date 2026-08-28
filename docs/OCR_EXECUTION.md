# OCR execution layer

公开 OCR execution layer 面向 PaddleX 3.x 当前 pipeline API，同时让 Paddle/PaddleX 继续作为可选运行时，而不是 core package 的强制依赖。

## Current upstream baseline

截至 2026-08，本项目按以下上游合同实现：

- PaddleX 3.7.x pipeline API；
- `paddlex.create_pipeline(pipeline="OCR")` 或本地 pipeline YAML；
- `device`, `engine`, `use_hpip`, `hpi_config` 等当前参数；
- prediction 返回 iterable Result；
- OCR Result 提供官方 `.json` mapping；
- 当前识别后 geometry/text 使用 `rec_polys / rec_texts`。

参考：

- PaddleX 3.7 installation docs
- PaddleX pipeline Python API
- PaddleX General OCR tutorial
- PaddleX 3.7 `OCRResult._to_json()` source contract

## Installation policy

PaddlePaddle 不作为普通项目 dependency。原因是 CPU/GPU wheel 需要按硬件和官方索引选择。

先安装合适的 PaddlePaddle runtime，再安装项目 OCR extra：

```bash
python -m pip install '.[ocr]'
```

当前 extra：

```text
paddlex[ocr-core]>=3.7,<3.8
```

使用 `ocr-core` 而不是完整 `paddlex[ocr]`，因为这里只需要 General OCR 的检测/识别核心，不主动引入公式、表格、文档解析等更大依赖面。

## Validated CPU matrix

公共 GitHub Actions 已完成真实模型 smoke：

| Component | Validated value |
| --- | --- |
| OS | Ubuntu 24.04 |
| Python | 3.12.14 |
| PaddlePaddle | 3.2.2 CPU |
| PaddleX | 3.7.2 |
| Detection | PP-OCRv6_small_det |
| Recognition | PP-OCRv6_small_rec |

测试使用 PaddleX 官方 General OCR demo image，并真实下载模型、执行 CPU 推理、保存结果、重新读取结果。

### PaddlePaddle 3.3.0 CPU note

PaddlePaddle 3.3.0 在当前 CPU oneDNN/PIR 路径存在已知上游回归；真实 CI 曾触发同类 `ConvertPirAttribute2RuntimeAttribute` / oneDNN 错误。上游 issue 给出的临时 workaround 是使用 3.2.2，因此公共 CPU smoke 暂时固定 3.2.2。

这不是项目宣称“3.2.2 永远是最佳版本”。上游修复后应重新验证并更新矩阵。

## Serial execution command

```bash
paddle-batch-ocr ocr INPUT \
  --output json/
```

`INPUT` 可以是一张图片，也可以是递归扫描的图片目录。目录模式保留相对目录结构，输出名稳定为：

```text
<stem>_result.json
```

常用参数：

```bash
paddle-batch-ocr ocr images/ --output json/ --device cpu
paddle-batch-ocr ocr images/ --output json/ --device gpu:0
paddle-batch-ocr ocr images/ --output json/ --pipeline ./OCR.yaml
paddle-batch-ocr ocr images/ --output json/ --engine paddle_static
paddle-batch-ocr ocr images/ --output json/ --manifest work/manifest.sqlite3
```

当前执行层故意串行。一个 PaddleX pipeline 惰性初始化，并被所有真正需要 inference 的图片复用；完整 resume 如果只需要 adopt/skip 现有合法 JSON，则不会初始化模型。

## PaddleX Result boundary

### Official `.json` first

真实 PaddleX `OCRResult` 同时具有 Mapping 行为和官方 `.json` 导出接口。运行时 Mapping 还包含 `vis_fonts`、原始图像等只服务于可视化的对象，因此项目必须：

1. 只要 Result 暴露官方 `.json`，优先使用它；
2. 只有普通历史 dict 才直接按 Mapping 读取；
3. 不复制 PaddleX runtime 内部字典当作稳定 JSON schema。

PaddleX 3.7 的 `OCRResult._to_json()` 自己定义了稳定导出字段，因此 `vis_fonts` 等 runtime visualization object 不进入项目输出。

### NumPy values

官方 Result mapping 中 geometry / scores 可以包含 NumPy ndarray / scalar。core package 不为此强依赖 NumPy，而是在边界使用 array-like `.tolist()` / scalar-like `.item()` 转成普通 Python JSON 值；未知对象仍明确报错，不做 `str()` 降级。

## Result geometry compatibility

当前 PaddleOCR 3.x 在 recognition confidence filtering 后，最终识别文本数量可能少于原始 detection 数量。因此 public adapter 优先：

```text
rec_polys + rec_texts
```

历史仓库继续兼容：

```text
dt_polys + rec_texts
dt_polys + rec_text
```

这样不会把被过滤掉的 detection box 与错误文本重新配对。

## Atomic publication

每张图片的 Result 在发布前先：

1. 读取官方 `.json` / historical mapping；
2. 解析 OCR schema；
3. 转成 JSON-safe values；
4. 添加 `_paddle_batch_ocr` provenance；
5. 写同目录临时文件；
6. flush + fsync；
7. atomic replace/publish。

默认不覆盖已有合法 JSON。

## Resume and failure semantics

- valid pre-existing JSON 可在 resume 时 adopted；
- manifest 已认识 source 且 size/mtime 变化时，旧 result 被视为 stale；
- stale existing output 需要显式 overwrite；
- 一张图片失败会记录状态，但不会阻断整批后续图片；
- 任一 task failed 时 CLI 最终返回非零；
- pipeline 初始化失败在一批任务中只尝试一次，避免成千上万图片反复下载/初始化同一个坏环境；
- symlinked input/output/manifest boundary 被拒绝。

## Strict `--json` stdout contract

PaddleX/Paddle 模型下载通常有 Python 日志，但 Paddle/oneDNN 还可能从原生 C/C++ 直接写进 OS file descriptor 1。单纯 `contextlib.redirect_stdout()` 只能改 Python `sys.stdout`，无法保证：

```bash
paddle-batch-ocr ocr ... --json > summary.json
```

得到纯 JSON。

因此 `--json` OCR 模式在**第三方 inference 执行期间**临时把 process fd 1 重定向到 fd 2：

```text
Paddle/PaddleX/oneDNN chatter -> stderr
project summary              -> stdout
```

执行结束后恢复 fd 1，再输出项目自己的唯一 JSON summary。core tests 还会在子进程里直接 `os.write(1, ...)` 模拟 native runtime，确保不是只隔离 Python `print()`。

真实 CPU smoke 进一步要求 `summary.json` 可以直接 `json.loads()`。

## Real CPU smoke gate

当前 CI gate 会真实：

1. 安装 PaddlePaddle 3.2.2 CPU；
2. 安装 `.[ocr]` -> PaddleX 3.7.x OCR core；
3. 下载 PaddleX 官方 General OCR demo image；
4. 使用 `configs/paddlex/ocr-ci-small.yaml`；
5. 下载 PP-OCRv6 small det/rec；
6. 通过项目 CLI 执行 OCR；
7. 验证 summary JSON 为 strict JSON；
8. 打开落盘 `<stem>_result.json`；
9. 重新通过 `parse_ocr_page()`；
10. 要求 recognized lines > 0；
11. 验证 `rec_texts` / polygon field / provenance。

因此该 execution layer 已有真实 PaddleX CPU 证据，不只依赖 fake Result 测试。

## Concurrency next

当前 public OCR engine **不消费 `ocr_workers > 1`**。下一阶段才会加入 process worker pool，并要求：

- 每个 worker initializer 创建恰好一个 pipeline；
- 明确 multiprocessing start method；
- 明确 GPU device assignment；
- 复用完全相同的单任务执行 / schema / manifest / atomic publication contract。

在这之前不会把 legacy 的 16/32/64 高并发默认值搬进新 public API。
