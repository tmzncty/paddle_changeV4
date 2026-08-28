# Refactor Roadmap

目标不是把历史脚本重写一遍，而是把已经在真实大规模 OCR 任务中积累的逻辑逐步收束为一个安全、可复现、可维护的工具。

## M0 — Public project baseline

- [x] README 明确项目真实定位与 legacy 状态
- [x] destructive behavior / 高负载风险说明
- [x] 重构路线图与贡献说明
- [x] Python 3.9 / 3.12 baseline CI
- [ ] 打开并维护 issue tracker
- [ ] 补充 repository topics / description

## M1 — Freeze legacy behavior

- [ ] 为每个 legacy 脚本建立完整用途、输入输出与 replacement 表
- [x] 记录主要 OCR legacy 处理阶段与风险默认值
- [x] 记录 searchable-PDF legacy schema、命名和文本层假设
- [x] 冻结历史 page JSON 命名优先级
- [x] 冻结 `rec_text` / `rec_texts` 历史字段
- [x] 冻结 v7 两栏排序与 polygon geometry 行为
- [x] synthetic PDF / OCR JSON fixture
- [x] searchable-PDF round-trip smoke
- [ ] 更丰富 geometry golden fixture（旋转、双栏、中文长文本）

原则：没有行为基线前不删除旧脚本。

## M2 — Configuration and safety

已完成：

- [x] input / output / log / cache / manifest / Paddle config 路径模型
- [x] OCR / PDF / render worker 与 batch 配置字段
- [x] `auto / cpu / gpu / gpu:N` device 字段
- [x] JSON + optional YAML 项目配置
- [x] 保守默认值（worker=1、batch=1、overwrite=false）
- [x] destructive cache cleanup 默认 dry-run
- [x] realpath containment
- [x] filesystem root / home / cwd destructive root 拒绝
- [x] cache temp symlink 拒绝
- [x] input/output/log/cache/manifest 冲突检测
- [x] manifest symlink 在 SQLite open 前拒绝
- [x] PDF final output staging / atomic publication
- [ ] 把所有 legacy `clear_cache()` 调用迁移到新安全层
- [ ] config overwrite/resume 字段接入完整 `run` orchestration

## M3 — Package and unified CLI

已完成：

```bash
paddle-batch-ocr doctor
paddle-batch-ocr scan --config CONFIG
paddle-batch-ocr cache clean --config CONFIG
paddle-batch-ocr manifest status --config CONFIG
paddle-batch-ocr render INPUT.pdf --output DIR
paddle-batch-ocr ocr INPUT --output DIR
paddle-batch-ocr searchable-pdf --images DIR --ocr-json DIR --output FILE.pdf
```

- [x] installable `src/` package
- [x] core package 不强制 Paddle/CUDA
- [x] `pdf` / `yaml` / `ocr` optional extras
- [x] machine-readable JSON for doctor / manifest / render / OCR / searchable-PDF where applicable
- [x] OCR `--json` 隔离 Python 与 native fd 1 runtime chatter
- [ ] `paddle-batch-ocr run --config CONFIG`
- [ ] config 驱动的完整 end-to-end pipeline command

## M4 — OCR engine cleanup

### 已完成 serial execution contract

- [x] 当前 PaddleX 3.7 `create_pipeline` adapter
- [x] pipeline 惰性初始化并在整批中复用
- [x] pipeline 初始化失败整批只尝试一次
- [x] `predict_iter` / `predict` adapter
- [x] 当前 `rec_polys + rec_texts` schema
- [x] 历史 `dt_polys + rec_texts` / `rec_text` 兼容
- [x] PaddleX Result 官方 `.json` 优先于 runtime Mapping
- [x] NumPy array/scalar -> JSON-safe 转换
- [x] atomic OCR JSON publication
- [x] 默认 no-overwrite
- [x] per-image failure isolation + non-zero batch exit
- [x] resume/adopt valid historical JSON without model init
- [x] source fingerprint stale detection with manifest
- [x] symlinked image/input/output/manifest safety
- [x] fd-level stdout isolation for native Paddle/oneDNN chatter in `--json` mode
- [x] real CPU OCR smoke through official demo + PP-OCRv6 small det/rec

### 下一阶段

- [ ] process worker pool
- [ ] 每个 worker 只初始化一次 pipeline
- [ ] 明确 multiprocessing start method
- [ ] GPU worker/device assignment
- [ ] automatic retry policy
- [ ] 对损坏图片 / empty OCR / model failure 建立更细状态
- [ ] GPU self-hosted/manual smoke
- [ ] 吞吐量 benchmark 与 worker/batch tuning guide

## M5 — PDF pipeline cleanup

### Rendering

- [x] 独立 `pdf_render.py`
- [x] 单 PDF open 一次并逐页 render
- [x] 36–1200 DPI
- [x] deterministic `page_00001.png`
- [x] sibling staging directory
- [x] 默认拒绝已有输出
- [x] overwrite backup / replace / rollback
- [x] Python 3.9 / 3.12 real PDF smoke
- [ ] render stage manifest integration
- [ ] colorspace / alpha / format 配置
- [ ] 超大 PDF 分段/恢复策略

### Searchable PDF

- [x] OCR schema adapter
- [x] historical naming compatibility
- [x] legacy two-column/layout heuristic freeze
- [x] complete page sequence requirement
- [x] duplicate / gap / missing JSON hard errors
- [x] sibling temp PDF + atomic publication
- [x] hidden-text extraction round-trip smoke
- [x] Python 3.9 / 3.12 execution validation
- [ ] searchable-PDF stage manifest integration
- [ ] 中文长文本 / 旋转 / 双栏 geometry golden tests
- [ ] legacy mode 与 corrected geometry mode 分界

## M6 — Resume, manifest and observability

SQLite manifest 已建立，以 `(source_path, stage)` 为任务键。

- [x] source path / size / mtime fingerprint
- [x] `pending / running / success / failed`
- [x] result path
- [x] retry count
- [x] error class / message
- [x] worker / device
- [x] timing metrics
- [x] SQLite WAL + busy timeout
- [x] 并发首次登记幂等化
- [x] source change 使旧 success 失效
- [x] success result 丢失后重新判定
- [x] OCR serial execution adoption / stale semantics
- [ ] render / OCR / searchable-PDF 全生命周期统一写入 manifest
- [ ] crash-safe end-to-end resume
- [ ] 从 manifest 定向重跑失败项
- [ ] JSON / CSV aggregate statistics
- [ ] 不扫描全部 source 即获得整体进度

## M7 — Tests and CI

1. [x] Python 3.9 / 3.12 syntax checks
2. [x] dependency-free unit tests
3. [x] package install + CLI smoke
4. [x] Python 3.9 / 3.12 real PDF execution smoke
5. [x] real CPU PaddleX OCR smoke
6. [ ] GPU manual / self-hosted smoke
7. [ ] expanded PDF geometry golden tests

CPU OCR smoke 当前真实验证：

```text
Python 3.12.14
PaddlePaddle CPU 3.2.2
PaddleX 3.7.2
PP-OCRv6_small_det
PP-OCRv6_small_rec
Ubuntu 24.04
```

Smoke 不只检查进程退出码，还会：

- 强制 `--json` stdout 可直接 `json.loads()`；
- 检查 OCR output JSON 实际存在；
- 重新通过项目 `ocr_schema` 解析落盘结果；
- 要求 recognized line count > 0；
- 验证 `_paddle_batch_ocr` provenance。

## M8 — Reproducible environments

- [x] `pyproject.toml` 管理项目
- [x] YAML optional extra
- [x] PDF optional extra
- [x] OCR extra 使用 `paddlex[ocr-core]>=3.7,<3.8`
- [x] PaddlePaddle CPU/GPU runtime 不混进统一 dependency
- [x] Python 3.9 / 3.12 PDF dependency resolution CI
- [x] Paddle/PaddleX CPU 已验证版本矩阵
- [x] 记录 PaddlePaddle 3.3.0 当前 CPU oneDNN/PIR 回归与 3.2.2 workaround
- [ ] CUDA/GPU 已验证矩阵
- [ ] optional Conda/container recipe
- [ ] CI pip/model cache，减少昂贵重复下载

## M9 — Release and legacy archive

在新 CLI 覆盖旧功能且真实数据对比充分后：

- [ ] 将已替代历史脚本移动到 `legacy/`
- [ ] 每个脚本标记 replacement
- [ ] 第一个正式 semantic version
- [ ] changelog / release notes
- [ ] README 最终只保留当前推荐路径

当前 `render`、serial `ocr` 和 `searchable-pdf` 已有可测试的新 execution path，但 legacy 暂不移动；下一门槛是 worker lifecycle、完整 orchestration 与更丰富真实数据/golden coverage。

## Non-goals

短期内不追求：

- 自己重实现 PaddleOCR 模型；
- 通用 GUI；
- 为所有 CUDA/driver 组合兜底；
- 为了“代码漂亮”删除没有替代测试的历史逻辑；
- 用过度抽象牺牲几十万页批处理吞吐量。
