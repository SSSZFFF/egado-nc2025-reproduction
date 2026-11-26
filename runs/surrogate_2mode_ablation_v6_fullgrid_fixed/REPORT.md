# v6 — 2-mode full-grid 900, fixed vector exporter

Run: `F:\codex_project\inverse_design_sun\runs\surrogate_2mode_ablation_v6_fullgrid_fixed`

## 修复内容

`run_surrogate_validation.py` 的 full-grid exporter 已改为：

- 以 `initial.gds` 的原始 `CORE` 矢量为 `base_full_core`
- 只对 `state ^ full_init` 做 cell 级 add/subtract，不再从 `full_fixed_core + 全 cell` 重填
- 全图 Tier-1 使用内存版 `width_check(1200, ignore_angle=85)` / `space_check(...)`
- 全图输出 top cell 名为 `B_SURROGATE_FULL`

1 分钟验收：

- `full_init`（703/900 true）导出后 `geometry_io.supplemental = width=0, space=0`
- FDTD 初始评估 **J = 0.8498173602192169**，与原始 `initial.gds` 的 `0.849817360` 精确一致
- 快速 smoke run 正常完成 2 个状态、无 DRC 错误

## v6 正式结果

| 指标 | v6 full-grid | baseline |
|---|---:|---:|
| 初始 J | 0.849817 | 0.849817 |
| 最终 J | **0.870482** | **0.886449** |
| 增益 | **+0.020664** | +0.036632 |
| 接受翻转 | **6** | 19 |
| FDTD 状态评估 | **134**（268 solves） | 212（424 solves） |
| 伴随刷新 | **2×4 = 8 solves** | — |
| 总 solves | **276** | 424 |
| 每状态命中率 | 4.51% | 8.96% |
| 每 solve 命中率 | 2.24% | 4.48% |
| 墙钟时间 | **2h29m** | 5h15m |
| 达到 baseline 增益比例 | **56.4%** | 100% |

Pass 调度：`48,60,36`；`pair_escapes = 0`；`refresh_errors = 0`；所有导出均通过 248 类 PDK DRC（日志 `common: 248 categories, 0 violations`）。

## 结论

- full-grid exporter 的几何/DRC 问题已经修好，v6 可以正常完成，不再出现之前 `space=8` / `J=0.764760` 的问题；
- 但全图 900 格在 `48,60,36` 三 Pass、134 states / 276 solves 下只到 **0.870482**，仍低于 baseline 0.886449，甚至略低于 v3b；
- 说明“去掉 Canny + 直接 Top-|grad| Pass 扫描”本身不足以保证优于原版；全图自由度增加后，候选排序/Pass 预算/局部搜索策略需要重新设计；
- 暂不应把 v6 直接搬到 4 模。

## 文件

- `summary.json`
- `surrogate_history.csv`
- `pair_history.csv`
- `convergence.png`
- `stdout.log` / `stderr.log`
- `final_state.npz`
