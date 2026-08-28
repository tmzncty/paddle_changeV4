# Refactor Roadmap

这个路线图的目标不是重写一遍所有代码，而是把已经在真实大规模 OCR 任务中积累的逻辑逐步收束为一个安全、可复现、可维护的工具。

## M0 — Public project baseline

- [x] 重写 README，明确项目真实定位与 legacy 状态
- [x] 明确 destructive behavior 与高负载风险
- [x] 建立重构路线图
- [x] 添加最小贡献说明
- [x] 添加 Python 3.9 / 3.12 baseline CI
- [ ] 打开并维护 issue tracker
- [ ] 补充 repository topics / description

**完成标准：** 陌生用户第一次打开仓库时，不会把旧脚本里的个人路径和高并发值误认为通用默认配置。

## M1 — Freeze legacy behavior

目标：在动结构之前，先保存现有脚本到底做什么。

- [ ] 为每个 legacy 脚本建立完整用途、输入输出与 replacement 表
- [x] 记录 `highocr4_f1_pdf_img.py` 的主要处理阶段与风险默认值
- [x] 记录 `pdf_creator_with_text_layer7.py` 的 JSON schema、命名和文本层假设
- [x] 冻结历史 JSON 页文件命名优先级
- [x] 冻结 `rec_text` / `rec_texts` 两种历史字段
- [x] 冻结 v7 两栏排序与 polygon 0/2 点文本框行为
- [x] 建立运行时 synthetic PDF / OCR JSON fixture
- [x] 建立 searchable-PDF round-trip smoke test（页数 + 隐藏文本提取）
- [ ] 建立更丰富的 geometry golden fixture（旋转、列布局、中文长文本等）

当前行为合同见 [`docs/LEGACY_BEHAVIOR.md`](docs/LEGACY_BEHAVIOR.md)。

**原则：** 没有行为基线前，不删除旧脚本。

## M2 — Configuration and safety layer

统一配置模型已经建立，源码中的机器专属策略不再作为新实现的默认值。

当前 package 已支持：

- [x] input sources
- [x] output root
- [x] log directory
- [x] cache root
- [x] manifest path（默认 `<log_dir>/manifest.sqlite3`）
- [x] Paddle config path
- [x] OCR workers
- [x] PDF preparation workers
- [x] render workers
- [x] batch size
- [x] device (`auto`, `cpu`, `gpu`, `gpu:N`)
- [x] temporary image policy 字段
- [x] overwrite policy 字段
- [x] resume policy 字段
- [x] JSON 项目配置
- [x] optional YAML 项目配置

安全要求：

- [x] 默认并发采用保守值（当前均为 1）
- [x] 递归删除限制在明确 cache root 内
- [x] cache cleanup 只作用于 `<cache_root>/temp`
- [x] 删除前做 realpath containment 检查
- [x] 拒绝 filesystem root / home / cwd 作为 destructive root
- [x] cache temp symlink 直接拒绝
- [x] destructive operation 默认 dry-run
- [x] 默认 `overwrite=false`
- [x] 输入、输出、日志、缓存、manifest 做路径冲突检测
- [x] log / manifest 不允许落入 destructive cache boundary
- [x] render / searchable-PDF final output 使用 staging / atomic publication
- [ ] 将 legacy `clear_cache()` 调用全部迁移到新安全层
- [ ] config 的 overwrite / resume 字段接入统一 `run` orchestration

## M3 — Package and unified CLI

当前结构：

```text
src/paddle_batch_ocr/
  __init__.py
  cache.py
  cli.py
  config.py
  discovery.py
  doctor.py
  io_utils.py
  layout.py
  manifest.py
  naming.py
  ocr_schema.py
  pdf_render.py
  safety.py
  searchable_pdf.py
```

已完成：

```bash
paddle-batch-ocr doctor
paddle-batch-ocr doctor --json
paddle-batch-ocr scan --config CONFIG
paddle-batch-ocr cache clean --config CONFIG
paddle-batch-ocr manifest status --config CONFIG
paddle-batch-ocr render INPUT.pdf --output DIR
paddle-batch-ocr searchable-pdf --images DIR --ocr-json DIR --output FILE.pdf
```

其中：

- `cache clean` 默认只做 dry-run，必须显式 `--execute`；
- `manifest status` 在数据库不存在时不会制造空数据库；
- package 可通过 `pip install --no-deps .` 安装，不强制拉取 Paddle/CUDA；
- PDF execution 通过 `pip install '.[pdf]'` 单独安装；
- `render` / `searchable-pdf` 支持机器可读 `--json` 输出。

计划继续提供：

```bash
paddle-batch-ocr ocr --config CONFIG
paddle-batch-ocr run --config CONFIG
```

`doctor` 当前已报告：

- [x] Python 版本
- [x] 平台
- [x] Paddle / Paddle GPU / PaddleX / PaddleOCR package version
- [x] `nvidia-smi` GPU / memory / driver 摘要
- [x] PyMuPDF / Pillow 等关键 package version
- [x] 可用 CPU / 物理内存
- [x] output / log / cache 对应磁盘可用空间
- [x] 输入与 Paddle config 缺失警告
- [x] 明显高 worker / batch 参数警告

## M4 — OCR engine cleanup

已经完成的 engine-independent 基础：

- [x] `rec_text` / `rec_texts` OCR JSON schema adapter
- [x] OCR polygon/text 长度与坐标结构验证
- [x] 原子 JSON 写入 primitive（同目录 temp + fsync + atomic publish）
- [x] 默认禁止覆盖已存在 JSON

仍需接入真正 PaddleX/PaddleOCR 执行层：

- [ ] 每个 worker 只初始化一次 pipeline
- [ ] 明确 multiprocessing start method
- [ ] 对 CPU / GPU 执行模式分别建配置
- [ ] 将 stdout 抑制从全局替换改为更受控实现
- [ ] 统一 PaddleX `predict()` generator / list / single result adapter
- [ ] 将 atomic JSON writer 接入 OCR engine
- [ ] 明确失败重试策略
- [ ] 对损坏图片和空 OCR 结果区分状态
- [ ] CPU tiny-fixture smoke
- [ ] GPU self-hosted/manual smoke

## M5 — PDF pipeline cleanup

### Rendering

- [x] 独立 `pdf_render.py` execution layer
- [x] 单 PDF 只 open 一次并逐页 render
- [x] 统一可配置 DPI（36–1200）
- [x] deterministic `page_00001.png` 命名
- [x] sibling staging directory，整本完成后再发布
- [x] 默认拒绝已有输出
- [x] overwrite 使用 backup / replace / rollback
- [x] Python 3.9 / 3.12 real PDF smoke
- [ ] 将 render stage 接入 manifest
- [ ] 可选 colorspace / alpha / image format 配置
- [ ] 超大 PDF 的分段/恢复策略（不牺牲事务语义）

### Searchable PDF

兼容与执行层：

- [x] 定义稳定 OCR JSON schema adapter
- [x] 支持历史 `rec_text` / `rec_texts` 字段差异
- [x] 抽出历史 page JSON 命名优先级
- [x] 冻结 v7 两栏排序 heuristic
- [x] 冻结 v7 polygon 0/2 点文本矩形行为
- [x] 新 `searchable_pdf.py` 使用这些 adapter
- [x] 页面图要求完整 `page_00001..N` 序列
- [x] duplicate / gap / missing JSON 作为 hard error
- [x] sibling temp PDF + atomic publication
- [x] hidden text extraction round-trip smoke
- [x] Python 3.9 / 3.12 execution validation
- [ ] searchable-PDF stage 接入 manifest
- [ ] 中文长文本、旋转、字号拟合、双栏 geometry golden tests
- [ ] 明确新坐标系策略以及 legacy mode / corrected mode 分界
- [ ] 判断哪些 legacy heuristic 应保留、哪些应作为 bug 修复

## M6 — Resume, manifest and observability

SQLite manifest 核心已经建立，以 `(source_path, stage)` 作为任务键。

已记录字段：

- [x] source path
- [x] source size / mtime
- [x] stage
- [x] started_at / finished_at
- [x] result path
- [x] status (`pending / running / success / failed`)
- [x] retry count
- [x] error class / message
- [x] worker / device information
- [x] timing metrics

已完成基础能力：

- [x] SQLite WAL + busy timeout
- [x] 并发首次登记幂等化
- [x] 源文件变化后使旧 success 失效为 pending
- [x] success 的结果文件丢失后重新判定为需要执行
- [x] failure 记录 retry/error 信息
- [x] stage 独立状态
- [x] CLI status 统计

仍需执行层集成：

- [ ] render / OCR / searchable-PDF worker 全部写入 manifest
- [ ] 真正 crash-safe end-to-end resume
- [ ] 从 manifest 定向重新执行失败项
- [ ] 输出完整统计 JSON / CSV
- [ ] 不扫描几十万源文件即可得到完整整体进度

## M7 — Tests and CI

测试分层：

1. **No-dependency syntax checks** — [x] Python 3.9 / 3.12；
2. **Pure Python unit tests** — [x] discovery / safety / config / cache / CLI / naming / OCR schema / layout / atomic I/O / manifest；
3. **Package install + CLI smoke** — [x] `pip install --no-deps .` + version + doctor；
4. **Real PDF execution smoke** — [x] Python 3.9 / 3.12 synthetic render → OCR JSON → searchable PDF → hidden-text extraction；
5. **CPU OCR smoke test** — [ ]；
6. **GPU manual / self-hosted benchmark** — [ ]；
7. **Expanded PDF golden tests** — [ ] 中文、旋转、双栏、坐标误差。

当前 PDF CI 实际解析并验证：

| Python | PyMuPDF | Pillow |
| --- | --- | --- |
| 3.9.25 | 1.26.5 | 11.3.0 |
| 3.12.14 | 1.28.2 | 12.3.0 |

## M8 — Reproducible environments

不要把 Paddle CPU/GPU 依赖强行塞进一个 requirements 文件。

- [x] `pyproject.toml` 管理项目本身
- [x] YAML 配置作为 optional extra
- [x] PDF execution 作为独立 `pdf` extra
- [x] Python 3.9 / 3.12 PDF dependency resolution 已在 CI 验证
- [x] 记录 PyMuPDF AGPL / commercial upstream license 边界
- [ ] 文档化 Paddle / PaddleX 当前安装步骤
- [ ] 给出 CPU 环境示例
- [ ] 给出 CUDA 环境示例
- [ ] 记录 Paddle / PaddleX / CUDA 已验证版本矩阵
- [ ] 可选 Conda / container recipe

## M9 — Release and legacy archive

在新 CLI 覆盖旧功能且测试通过后：

- [ ] 将已替代历史脚本移动到 `legacy/`
- [ ] 每个脚本标记 replacement
- [ ] 发布第一个真正的 semantic version
- [ ] changelog 从 commit history / release notes 维护
- [ ] README 只保留当前使用方式

目前 `pdf_to_png.py` 和 searchable-PDF 相关 legacy 文件已经有可测试的新 execution replacement，但暂不移动，直到真实数据对比与 geometry golden 覆盖更完整。

## Non-goals

短期内不追求：

- 自己重新实现 PaddleOCR 模型；
- 做通用 GUI；
- 为所有 CUDA / 驱动组合兜底；
- 用过度抽象牺牲几十万页批处理吞吐量；
- 为了“代码漂亮”删除已经验证过但还没有替代测试的处理逻辑。
