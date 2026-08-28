# Targeted retry trust boundary

`manifest retry-failed` 把 SQLite row 当作**待验证数据**，不是授权凭据。

执行前必须重新确认：source 指纹、stage profile schema、intended target、output_root containment、symlink 状态、本地 pipeline SHA-256，以及执行前 manifest row 仍然是同一个 failed attempt。

如果任一条件不成立，候选只会出现在 dry-run 的 `ineligible` / `blocked` 中，不会通过 `--execute` 自动绕过。
