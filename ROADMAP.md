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

- [ ] 为每个 legacy 脚本建立用途与输入输出说明
- [ ] 记录 `highocr4_f1_pdf_img.py` 的完整处理阶段
- [ ] 记录 `pdf_creator_with_text_layer7.py` 的 JSON schema 假设
- [ ] 收集多个历史 JSON 命名格式
- [ ] 建立最小公开 fixture：1 个小 PDF、若干图片、对应匿名化 OCR JSON
- [ ] 建立 golden output / smoke test

**原则：** 没有行为基线前，不删除旧脚本。

## M2 — Configuration and safety layer

建立统一配置模型，消灭源码中的机器专属路径。

当前 package 已支持：

- [x] input sources
- [x] output root
- [x] log directory
- [x] cache root
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
- [x] 删除前做 path containment 检查
- [x] 拒绝 filesystem root / home / cwd 等危险目标
- [x] destructive operation 默认 dry-run
- [x] 默认 `overwrite=false`
- [x] 输入、输出、缓存路径做冲突检测
- [ ] 将 legacy `clear_cache()` 全部迁移到新安全层
- [ ] overwrite / resume 字段真正接入 OCR / PDF engine

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
  safety.py
```

已完成：

```bash
paddle-batch-ocr doctor
paddle-batch-ocr doctor --json
paddle-batch-ocr scan --config CONFIG
paddle-batch-ocr cache clean --config CONFIG
```

其中 cache clean 默认只做 dry-run，必须显式 `--execute`。

计划继续提供：

```bash
paddle-batch-ocr render INPUT --output DIR
paddle-batch-ocr ocr --config CONFIG
paddle-batch-ocr searchable-pdf --config CONFIG
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

- [ ] 每个 worker 只初始化一次 pipeline
- [ ] 明确 multiprocessing start method
- [ ] 对 CPU / GPU 执行模式分别建配置
- [ ] 将 stdout 抑制从全局替换改为更受控实现
- [ ] 统一 OCR result adapter
- [ ] 兼容 generator / list / single result
- [ ] 原子写 JSON（临时文件 + rename）
- [ ] 明确失败重试策略
- [ ] 对损坏图片和空 OCR 结果区分状态

## M5 — PDF pipeline cleanup

### Rendering

- [ ] 单页渲染函数独立可测试
- [ ] 统一 DPI / matrix / colorspace 配置
- [ ] 大 PDF 避免不必要的重复 open
- [ ] 临时页图像生命周期可配置
- [ ] 中断后可识别已渲染页

### Searchable PDF

- [ ] 定义稳定 OCR JSON schema adapter
- [ ] 支持历史 `rec_text` / `rec_texts` 等字段差异
- [ ] 页码匹配不依赖脆弱字符串替换
- [ ] 明确坐标系转换
- [ ] 文本层字体、旋转、缩放建立回归测试
- [ ] 支持逐页失败报告而不是整本静默失败

## M6 — Resume, manifest and observability

为几十万页任务引入 manifest / job state，而不是只靠“目标文件是否存在”。

建议每个任务记录：

- source path
- source size / mtime / optional hash
- stage
- started_at / finished_at
- result path
- status
- retry count
- error class
- worker / device information
- timing metrics

目标：

- [ ] crash-safe resume
- [ ] 可重新执行失败项
- [ ] 输入变化后识别旧结果失效
- [ ] 输出统计 JSON / CSV
- [ ] 不扫描几十万文件才能知道整体进度

## M7 — Tests and CI

测试分层：

1. **No-dependency syntax checks** — [x] Python 3.9 / 3.12；
2. **Pure Python unit tests** — [x] discovery / safety / config / cache / CLI 起步；
3. **Package install + CLI smoke** — [x] `pip install --no-deps .` + doctor；
4. **CPU OCR smoke test** — [ ]；
5. **GPU manual / self-hosted benchmark** — [ ]；
6. **Golden PDF tests** — [ ] 页数、文本可搜索性、坐标误差。

## M8 — Reproducible environments

不要把 Paddle CPU/GPU 依赖强行塞进一个 requirements 文件。

- [x] `pyproject.toml` 管理项目本身
- [x] YAML 配置作为 optional extra
- [ ] 文档化 Paddle / PaddleX 当前安装步骤
- [ ] 给出 CPU 环境示例
- [ ] 给出 CUDA 环境示例
- [ ] 记录已验证版本矩阵
- [ ] 可选 Conda / container recipe

## M9 — Release and legacy archive

在新 CLI 覆盖旧功能且测试通过后：

- [ ] 将历史脚本移动到 `legacy/`
- [ ] 每个脚本标记 replacement
- [ ] 发布第一个真正的 semantic version
- [ ] changelog 从 commit history / release notes 维护
- [ ] README 只保留当前使用方式

## Non-goals

短期内不追求：

- 自己重新实现 PaddleOCR 模型；
- 做通用 GUI；
- 为所有 CUDA / 驱动组合兜底；
- 用过度抽象牺牲几十万页批处理吞吐量；
- 为了“代码漂亮”删除已经验证过但还没有替代测试的处理逻辑。
