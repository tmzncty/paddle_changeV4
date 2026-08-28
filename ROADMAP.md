# Refactor Roadmap

这个路线图的目标不是重写一遍所有代码，而是把已经在真实大规模 OCR 任务中积累的逻辑逐步收束为一个安全、可复现、可维护的工具。

## M0 — Public project baseline

- [x] 重写 README，明确项目真实定位与 legacy 状态
- [x] 明确 destructive behavior 与高负载风险
- [x] 建立重构路线图
- [x] 添加最小贡献说明
- [ ] 添加纯语法 CI
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

计划字段：

- input sources
- output root
- log directory
- PaddleX home / cache directory
- OCR config path
- OCR workers
- PDF preparation workers
- render workers
- batch size
- device (`cpu`, `gpu`, future auto)
- temporary image policy
- overwrite policy
- resume policy

安全要求：

- [ ] 默认并发采用保守值或自动探测
- [ ] 递归删除必须限制在明确的 workspace/cache root 内
- [ ] 删除前做 path containment 检查
- [ ] 禁止对 `/`、home、输入根目录等危险目标执行清理
- [ ] destructive operation 提供 dry-run
- [ ] 默认不覆盖已有 OCR JSON / PDF
- [ ] 对输入、输出、缓存路径做冲突检测

## M3 — Package and unified CLI

目标结构：

```text
src/paddle_batch_ocr/
  __init__.py
  cli.py
  config.py
  discovery.py
  models.py
  ocr.py
  pdf_render.py
  searchable_pdf.py
  cache.py
  progress.py
  safety.py
```

计划命令：

```bash
paddle-batch-ocr doctor
paddle-batch-ocr scan --config config.yaml
paddle-batch-ocr render INPUT --output DIR
paddle-batch-ocr ocr --config config.yaml
paddle-batch-ocr searchable-pdf --config config.yaml
paddle-batch-ocr run --config config.yaml
```

`doctor` 应报告：

- Python 版本
- Paddle / PaddleX 版本
- CUDA / GPU 可见性
- 关键依赖
- 可用 CPU / 内存
- 输入输出磁盘可用空间
- cache 目录
- 配置中的潜在危险项

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

1. **No-dependency syntax checks** — 每个 PR 都能跑；
2. **Pure Python unit tests** — 路径、配置、命名、schema adapter、安全检查；
3. **CPU smoke test** — 小样本端到端；
4. **GPU manual / self-hosted benchmark** — 不要求公共 GitHub runner 安装 CUDA；
5. **Golden PDF tests** — 验证页数、文本可搜索性、坐标误差。

## M8 — Reproducible environments

不要把 Paddle CPU/GPU 依赖强行塞进一个 requirements 文件。

计划：

- [ ] `pyproject.toml` 管理项目本身与普通 Python 依赖
- [ ] 文档化 Paddle / PaddleX 安装步骤
- [ ] 给出 CPU 环境示例
- [ ] 给出 CUDA 环境示例
- [ ] 记录已验证版本矩阵
- [ ] 可选 Conda / container recipe

## M9 — Release and legacy archive

在新 CLI 覆盖旧功能且测试通过后：

- [ ] 将历史脚本移动到 `legacy/`
- [ ] 每个脚本标记 replacement
- [ ] 发布第一个真正的 semantic version
- [ ] changelog 从 commit history / release notes 维护，不再堆在 README 顶部
- [ ] README 只保留当前使用方式

## Non-goals

短期内不追求：

- 自己重新实现 PaddleOCR 模型；
- 做通用 GUI；
- 为所有 CUDA / 驱动组合兜底；
- 用过度抽象牺牲几十万页批处理吞吐量；
- 为了“代码漂亮”删除已经验证过但还没有替代测试的处理逻辑。
