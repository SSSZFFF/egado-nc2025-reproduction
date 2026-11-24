# 2-mode surrogate ablation v2 — result summary

Run: `F:\codex_project\inverse_design_sun\runs\surrogate_2mode_ablation_v2`

## 设置

- 初始态：`DBSRunner(scheme="B")` 的 `initial_state.npz`（30×30，108 个 candidate 格）
- 物理先验：`EGADO-2mode\TO\x3600_y3600_f0239_m0000_0\parameters_100.npz`（beta = 4.30，Projected discreteness = 0.784）
- `--ablation-mask`，`--max-fdtd 150`（状态评估次数；2-mode 每个状态 2 个光源 → 300 forward solves）
- 远程 `py -3.12` 没有 torch，实际 backend = `numpy_ridge_ensemble`

## 结果

| 指标 | 本次 surrogate | B_strict baseline |
|---|---:|---:|
| 初始 J | 0.849817 | 0.849817 |
| 最终 J | **0.860121** | **0.886449** |
| 增益 | +0.010304 | +0.036632 |
| 接受翻转 / 有效翻转 | **14** | **19** |
| FDTD 状态评估 | **150** | 212（424 forward solves） |
| forward solves | **300** | 424 |
| 每 solve 命中率 | 4.67% | 4.48% |
| 每状态命中率 | 9.40% | 8.96% |
| 墙钟时间 | **5,847.95 s ≈ 1h37m** | 18,894.4 s ≈ 5h15m |
| 相对 baseline 的 J 差距 | −0.026328 | — |
| 达到 baseline 增益的比例 | 28.1% | 100% |

baseline 在 **226 forward solves** 时已达到 J = 0.886254，并在 424 solves 收尾到 0.886449。也就是说：本次 surrogate 用 300 solves 只到 0.860121，**没有追上 baseline**；按 per-solve 命中率看只是略高（4.67% vs 4.48%），但最终 J 明显更低。

## 结论

- 两级 DRC 和 Top-K skip/fill 都正常工作：150 次 state evaluation 全部用完，`fdtd_evals=150`，没有因为 DRC 拒绝导致预算缺口。
- 但当前 `parameters_100` 的 analog prior + NumPy Ridge + `top_k=4` 组合在 2 模上落在较差局部；38 轮、14 次 accept 后停在 0.860121。
- 建议下一组优先试：`top_k=2` 或 `top_k=1`（更贪心）、`ucb_kappa` 降到 0.2 左右；若仍不改善，改用全图 900 格或换更强的 Stage-1 特征（`grad_eps` / `e2_field`）。

## 文件

- `surrogate_history.csv`
- `summary.json`
- `final_state.npz`
- `convergence.png`
- `stdout.log` / `stderr.log`
