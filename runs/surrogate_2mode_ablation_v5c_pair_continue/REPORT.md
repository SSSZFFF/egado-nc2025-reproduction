# v5c — 从 v5b 2-flip 断点继续的短 Pass

起点：`runs/v5b_pair_escape/pair_20_13_23_21.npz`（J = 0.8820153891845495）

设置：
- `PasswiseMagnitudeDigitizer`
- `pass_top_n=[30]`
- `pair_escape_top_m=4`
- `max_fdtd_evals=35`
- 使用该 pair 状态重新提取的 `adjoint_pair.npz`（4 solves）

结果：

- `final_J = 0.8820153891845495`
- `total_flips = 0`
- `fdtd_evals = 34`（68 solves）
- `pair_escapes = 0`
- `refresh_errors = 0`

结论：在该 pair 状态上，Top-30 单翻和当前 pair escape 都没有新的正增益，说明它已经是当前候选集内的局部极值。要继续提升需要改变候选集（例如全图 900 格）或引入更高阶协同。
