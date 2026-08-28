# OCR execution layer

公开 OCR execution layer 面向 PaddleX 3.7.x 当前 pipeline API，同时让 Paddle/PaddleX 继续作为可选运行时，而不是 core package 的强制依赖。

## Runtime baseline

当前公共 CPU 真实验证矩阵：

| Component | Value |
| --- | --- |
| OS | Ubuntu 24.04 |
| Python | 3.12.14 |
| PaddlePaddle | 3.2.2 CPU |
| PaddleX | 3.7.2 |
| Detection | PP-OCRv6_small_det |
| Recognition | PP-OCRv6_small_rec |

PaddlePaddle 3.3.0 在当前 CPU oneDNN/PIR 路径存在已知上游回归，因此 CI 暂用 3.2.2 workaround。上游修复后应重新验证，不把 3.2.2 当作永久推荐版本。

## Installation policy

PaddlePaddle 不作为普通 project dependency，因为 CPU/GPU wheel 与官方安装源依赖硬件环境。

先安装对应 PaddlePaddle runtime，再安装：

```bash
python -m pip install '.[ocr]'
```

当前 OCR extra：

```text
paddlex[ocr-core]>=3.7,<3.8
```

这里只依赖 General OCR 核心，不主动引入完整表格、公式、文档解析依赖面。

## Result boundary

### Official `.json` first

真实 PaddleX `OCRResult` 同时是 Mapping-like runtime object，又暴露官方 `.json` 导出。runtime mapping 可能包含 `vis_fonts`、原始图像等可视化对象，因此 public adapter：

1. Result 有官方 `.json` 时优先使用；
2. plain historical dict 才直接按 Mapping 读；
3. 不把 runtime internal mapping 复制成稳定 schema。

### Geometry pairing

现代 OCR 优先：

```text
rec_polys + rec_texts
```

历史兼容：

```text
dt_polys + rec_texts
dt_polys + rec_text
```

这样 recognition confidence filtering 后不会拿未过滤 detection box 去错配最终文本。

### NumPy values

geometry / score 可含 ndarray / scalar。core 不强制 NumPy，而是在边界使用 `.tolist()` / `.item()` 转成普通 JSON values；未知对象仍明确失败，不用 `str()` 隐藏 schema drift。

## Atomic publication

每张图片的 Result 发布前会：

1. 读取官方 `.json` / historical mapping；
2. 解析 OCR schema；
3. 转换 JSON-safe values；
4. 加 `_paddle_batch_ocr` provenance；
5. 写同目录临时文件；
6. flush + fsync；
7. atomic publish / replace。

默认 no-overwrite。

## Serial execution

```bash
paddle-batch-ocr ocr INPUT \
  --output json/ \
  --device cpu
```

`workers=1` 时：

- 一个 PaddleX pipeline 惰性初始化；
- 真正需要 inference 的任务复用同一 pipeline；
- 完整 resume 如果全部 adopt/skip 现有合法 JSON，则不加载模型；
- pipeline 初始化失败一批只尝试一次；
- 单任务失败被隔离，batch 最终返回非零。

## CPU process workers

```bash
paddle-batch-ocr ocr images/ \
  --output json/ \
  --device cpu \
  --workers 4 \
  --manifest work/manifest.sqlite3
```

当前并发合同：

- `workers>1` 使用 `multiprocessing` 的 **spawn** start method；
- parent 在 submit 前完成 resume / stale / existing-output preflight；
- 每个参与 process 独立打开一个 SQLite manifest connection；
- 每个 process 惰性创建最多一个 PaddleX pipeline；
- worker 复用同一 `predict_one_to_json()` 与 atomic publication 合同；
- batch summary 恢复成输入发现顺序，不按 future 完成顺序输出；
- 同一目录 `same.png` / `same.jpg` 导致相同 `<stem>_result.json` 时在执行前拒绝；
- `workers>1` 当前要求显式 `device=cpu`。

最后一条是安全边界，不是永久限制。GPU 多 worker 需要明确 device map；在此之前不会让多个 process 无脑复制模型到同一张 GPU。

## Manifest semantics

OCR manifest 以 `(source_path, "ocr")` 为键：

- source size / mtime 变化使旧 success 失效；
- valid existing JSON 可以在首次收编或非 stale resume 中 adopt；
- stale existing JSON 默认拒绝，需明确 overwrite；
- worker 会记录 `pid-<number>` 与 device；
- WAL + busy timeout 支持多个 process 各自持有连接。

## Strict `--json` stdout

Paddle / oneDNN 可能直接从 native C/C++ 写 OS fd 1，绕过 Python `sys.stdout`。因此 `ocr --json` 在第三方 inference 期间临时把 process fd 1 重定向到 stderr，恢复后再输出项目自己的唯一 JSON summary。

```text
Paddle/PaddleX/oneDNN chatter -> stderr
project JSON summary          -> stdout
```

core tests 还会用 `os.write(1, ...)` 模拟 native runtime；真实 CPU smoke 要求 summary 文件可以直接 `json.loads()`。

## Real validation gates

### Serial real OCR

CI 会：

1. 安装 PaddlePaddle 3.2.2 CPU；
2. 安装 `.[ocr]`；
3. 下载 PaddleX 官方 General OCR demo image；
4. 使用 `configs/paddlex/ocr-ci-small.yaml`；
5. 执行真实 PP-OCRv6 small det/rec；
6. 验证 strict JSON summary；
7. 打开落盘 OCR JSON；
8. 再通过 `parse_ocr_page()`；
9. 要求 recognized lines > 0。

### Real two-worker OCR

并行 gate 把官方 demo 复制成 4 个输入，然后真实运行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
paddle-batch-ocr ocr /tmp/ocr-parallel-input \
  --output /tmp/ocr-parallel-json \
  --pipeline configs/paddlex/ocr-ci-small.yaml \
  --device cpu \
  --workers 2 \
  --manifest /tmp/ocr-parallel-manifest.sqlite3 \
  --json
```

验证条件不仅是 `success=4`：

- 4 个 OCR JSON 都存在且能重新过 schema；
- manifest 中的 OCR records 必须出现**两个不同 worker PID**；
- 当前已实际观察到两个独立 PaddleX process 参与执行。

因此 process worker layer 已经有真实 PaddleX 证据，不只是 fake pipeline 生命周期测试。

## CI caching

重型 OCR job 使用：

- setup-python pip cache；
- `actions/cache` 保存 `~/.paddlex/official_models`；
- model cache key 绑定 OCR smoke pipeline YAML。

feature branch 只由 PR 触发完整 gate；main 由 push 触发，避免同一 commit 重复跑两份昂贵 OCR。

## Still not implemented

- GPU worker/device mapping；
- GPU self-hosted/manual smoke；
- automatic retry policy；
- damaged-image / empty-result / model failure 更细状态分类；
- throughput benchmark 与 worker/thread tuning guide。
