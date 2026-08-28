# Contributing

感谢你愿意帮助把这个项目从 production-grown OCR scripts 收束成可维护的公开工具。

当前贡献的第一原则是：**不要为了代码看起来更整齐，就在没有行为基线时删除已经跑过真实大规模数据的逻辑。** 先冻结行为，再有意识地替换。

## Development setup

重构 package 本身不强制依赖 Paddle：

```bash
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
paddle-batch-ocr doctor --json
```

当前公共基线是 Python 3.9 与 3.12。

可选 YAML 项目配置需要：

```bash
python -m pip install -e '.[yaml]'
```

Paddle / PaddleX / CUDA 安装将与项目自身依赖分开维护，因为 CPU/GPU 环境具有不同约束。

## 当前最欢迎的贡献

- 为 legacy 行为建立公开 fixture / golden output；
- 完成 PaddleX/PaddleOCR execution adapter；
- 将 PDF render / searchable PDF 迁移到新 package；
- 报告 Paddle / PaddleX / CUDA 的已验证组合；
- 加强 manifest resume、错误恢复与 observability；
- 修复文档与代码不一致；
- 在不牺牲吞吐量的前提下改进并发和资源控制。

## Safety requirements

涉及删除、覆盖、缓存或路径映射的变化，必须同时测试危险路径，而不只测试 happy path。

新代码不得：

- 把机器专属绝对路径作为默认配置；
- 把高 worker 数、大 batch 作为隐式默认；
- 对任意路径做递归删除；
- 用字符串前缀判断目录 containment；
- 把 logs / manifest 放进 destructive cache boundary；
- 静默覆盖 OCR JSON / PDF；
- 仅仅为了诊断或读取配置就在 package import 时加载 Paddle。

cache maintenance 应经过 `paddle_batch_ocr.safety` 与 `paddle_batch_ocr.cache`。Destructive CLI 行为应保持可预览、显式开启。

## Compatibility rules

修改 legacy OCR/PDF 行为前，请先阅读 [`docs/LEGACY_BEHAVIOR.md`](docs/LEGACY_BEHAVIOR.md)。

当前已经冻结的兼容事实包括：

- page JSON naming precedence；
- `rec_text` / `rec_texts` schema variants；
- polygon/text mismatch rejection；
- v7 的 two-column ordering heuristic；
- v7 使用 polygon 第 0 / 2 点构造文本矩形的行为。

“被测试冻结”不代表该 heuristic 一定正确。如果需要修正，应把行为变化明确成 migration / compatibility choice，而不是悄悄改变旧输出。

## Large-job semantics

目标 workload 不是 demo image。重构应保留或加强：

- deterministic discovery；
- resumability；
- 每文件/每页 failure isolation；
- atomic result publication；
- source-change invalidation；
- SQLite manifest 中的结构化 stage state；
- Unicode / 中文路径；
- 有边界的 CPU/GPU/memory/disk concurrency。

## Legacy code policy

根目录脚本包含大量真实环境中的经验性处理。在替换一个 legacy 脚本前，应说明：

1. 它当前接受什么输入、生成什么输出；
2. 哪些异常情况已被旧代码处理；
3. 哪些历史行为需要兼容；
4. 新实现如何通过 fixture / test 验证；
5. 是否存在性能、磁盘或恢复能力回归。

不要仅因为代码重复、命名混乱或风格不统一就删除逻辑。等 replacement 覆盖并验证后，再把旧脚本移动到 `legacy/` 并留下迁移指引。

## Testing

测试应分层：

1. dependency-free unit tests；
2. CPU Paddle smoke tests with tiny public fixtures；
3. searchable-PDF golden tests；
4. optional/self-hosted GPU benchmarks。

不要让每个 PR 为了测试纯 Python 路径逻辑就下载 CUDA 级环境。

提交行为变化时，请明确说明实际运行了哪些测试，以及 Paddle/CUDA execution 是否真的被验证，而不是只通过了 pure-Python CI。

## Pull request scope

尽量让一个 PR 对应一个可审查的 migration stage。PR 描述应说明：

- 保留了哪些 legacy behavior；
- 哪些行为故意改变；
- 涉及什么安全边界；
- 哪些测试覆盖变化；
- 是否实际运行了 Paddle/CUDA workload。

## License

提交到本仓库的代码将按仓库 GPL-3.0 许可证发布。请不要提交来源不明或许可证不兼容的第三方代码、模型、字体、扫描件或数据集。
