# v5b 2-flip 耦合逃逸快测

Base: `runs/surrogate_2mode_ablation_v5b_hourglass/final_state.npz`  
Base J = 0.8697724219307283

## 六个 2-flip 组合

Top-4 候选来自 v5b Pass 3 中 `true_dJ` 最接近 0 的拒绝像素：

`(13,25)`, `(20,13)`, `(23,21)`, `(23,25)`

| 组合 | J | dJ |
|---|---:|---:|
| (13,25) + (20,13) | 0.868648 | −0.001124 |
| (13,25) + (23,21) | 0.864966 | −0.004806 |
| (13,25) + (23,25) | 0.865028 | −0.004744 |
| **(20,13) + (23,21)** | **0.882015** | **+0.012243** |
| (20,13) + (23,25) | 0.855688 | −0.014085 |
| (23,21) + (23,25) | 0.867582 | −0.002191 |

全部 6 组均通过两级 DRC（Tier-1 120 nm + Tier-2 248 类 PDK）；没有跳过项。

## 结论

- `(20,13) + (23,21)` 是唯一正增益组合：`J = 0.882015`，相对 v5b 断点 **+0.012243**；
- 这证明“单翻微亏、联合翻转翻正”的 2-flip 逃逸机制真实存在；
- 但与 baseline 最终 `0.886449` 仍差 **0.004434**，所以这套 2-flip 目前只是“站稳 0.88+”，还不能说已经压过 baseline。

## 后续短 Pass 结果

从该 pair 状态继续跑 `PasswiseMagnitudeDigitizer`（1 个 Pass，Top-30，`pair_escape_top_m=4`）：

- `final_J = 0.8820153891845495`
- `total_flips = 0`
- `fdtd_evals = 34`（68 solves）
- `pair_escapes = 0`

Top-30 单翻全部为负；pair escape 也没有新的正增益组合。说明该 pair 状态已经是 **1-flip / 当前 Top-30 范围内的局部极值**，要继续提升需要新的候选集或更长程的多像素协同。

## 总预算

- v5b 原跑：234 solves（113 states ×2 + 2×4 adjoint）
- 6 组 pair 快测：6 states ×2 = 12 solves
- 合计约 **246 solves**，仍远低于 baseline 的 424 solves。

## 文件

- `pair_escape_results.json`
- `pair_*.npz`（6 个 pair 状态）
- `runs/surrogate_2mode_ablation_v5c_pair_continue/summary.json`
- `runs/surrogate_2mode_ablation_v5c_pair_continue/surrogate_history.csv`
