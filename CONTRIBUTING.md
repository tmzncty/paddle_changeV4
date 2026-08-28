# Contributing

感谢你愿意帮助整理这个项目。

当前仓库仍处于从 legacy scripts 向稳定工具迁移的阶段，因此贡献的第一原则是：**不要在没有行为基线的情况下删除已经跑过真实数据的逻辑。**

## 当前最欢迎的贡献

- 将硬编码路径和并发值迁移到配置系统；
- 为路径处理、页码匹配、OCR JSON schema adapter 建立单元测试；
- 为缓存清理和覆盖操作增加安全边界；
- 报告 Paddle / PaddleX / CUDA 的已验证版本组合；
- 提供可公开的小型 OCR fixture；
- 修复文档与代码不一致；
- 将历史脚本中的重复逻辑抽取为模块，同时保持输出兼容。

## Legacy code policy

当前根目录中的脚本包含大量真实环境中的经验性处理。

在替换一个 legacy 脚本之前，请至少说明：

1. 它目前接受什么输入；
2. 它生成什么输出；
3. 哪些异常情况已经被旧代码处理；
4. 新实现如何验证等价行为；
5. 是否存在性能或磁盘空间方面的回归风险。

不要仅因为代码重复、命名混乱或风格不统一就直接删除逻辑。

## Safety requirements

涉及以下行为的 PR 需要特别说明：

- `shutil.rmtree`、递归删除或批量覆盖；
- cache / temp 目录清理；
- 自动创建或修改符号链接；
- 默认并发数提高；
- 默认启用 GPU / 大 batch；
- 修改输入输出目录映射；
- 改变 resume / skip 判定；
- 修改 searchable PDF 的文本坐标计算。

新的 destructive operation 应至少具备：

- 明确目标路径；
- path containment 检查；
- safe default；
- 可测试的拒绝条件；
- 在合理情况下提供 dry-run。

## Commit / PR scope

优先提交小而可验证的变化。例如：

- `refactor: extract page filename parser`
- `test: cover historical OCR json names`
- `safety: constrain cache cleanup to configured root`
- `docs: document CUDA environment`

尽量避免一个 PR 同时进行大规模格式化、重命名、功能重构和行为修改，因为这会让 legacy behavior 很难审查。

## Testing

当前 CI 首先从不安装 Paddle 的语法检查开始。随着模块化重构推进，将逐步增加：

- pure Python unit tests；
- CPU smoke tests；
- golden OCR / PDF fixtures；
- GPU benchmark（通常由 self-hosted 或人工环境运行）。

提交功能变化时，请说明你实际运行了哪些测试，以及使用的 Python / Paddle / PaddleX / CUDA 环境。

## License

提交到本仓库的代码将按仓库的 GPL-3.0 许可证发布。请不要提交来源不明或许可证不兼容的第三方代码、模型、字体、文档扫描件或数据集。
