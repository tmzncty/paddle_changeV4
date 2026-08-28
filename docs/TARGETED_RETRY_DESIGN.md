# Targeted retry design boundary

这一阶段只解决“基于可信 manifest provenance，精确重跑少量 failed rows”。

明确不做：

- 不把 retry 变成自动无限重试策略；
- 不绕过 `output_root` 写任意路径；
- 不把 named pipeline 当成可复现模型身份；
- 不在第一版把 failed retries 再做多进程；
- 不把 dry-run 当执行授权长期缓存；
- 不让 `--overwrite` 绕过 provenance 和路径安全检查。

选择顺序执行是有意的：失败集合通常远小于主任务集合，而且 targeted retry 的第一目标是可审计、可恢复、可解释。吞吐优化只有在真实失败工作负载证明有必要后再做。
