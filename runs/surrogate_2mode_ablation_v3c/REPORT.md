# 2-mode surrogate v3c — adjoint refresh + EI tie-breaker

Run: `F:\codex_project\inverse_design_sun\runs\surrogate_2mode_ablation_v3c`

## 设置

- 初始态：B_strict `initial_state.npz`（30×30，108 个 candidate 格）
- 物理先验：`adjoint_init_2mode.npz`（`initial.gds`，4 次 FDTD solves）
- 采集：EI（`ei_threshold=1e-4`），`top_k=2`，`--max-fdtd 110`；由于启动器默认 `--max-rounds 50`，实际停在 **101 states（202 forward solves）**
- 伴随刷新：每累计 **5 次接受翻转** 或连续 **4 轮无接受** 刷新一次；共 **7 次刷新 ×4 solves = 28 次附加伴随 solves**
- 刷新后：清空/截断 surrogate 经验池至最近 8 条，清空 `tabu_coords`
- 远程无 torch，backend = `numpy_ridge_ensemble`

## 结果

| 指标 | v3c | v3b | Baseline |
|---|---:|---:|---:|
| 初始 J | 0.849817 | 0.849817 | 0.849817 |
| 最终 J | **0.872816** | 0.871012 | **0.886449** |
| 增益 | **+0.022999** | +0.021194 | +0.036632 |
| 接受翻转 | **16** | 14 | 19 |
| FDTD 状态评估 | **101**（202 solves） | 99（198 solves） | 212（424 solves） |
| 附加伴随 solves | **28**（7 次刷新） | 4（初始一次） | — |
| 每 solve 命中率 | **7.92%**（16/202） | 7.07% | 4.48% |
| 每状态命中率 | **16.00%** | 14.29% | 8.96% |
| 墙钟时间 | **7,039.05 s ≈ 1h57m** | 4,565.61 s ≈ 1h16m | 18,894.4 s ≈ 5h15m |
| 达到 baseline 增益比例 | **62.8%** | 57.9% | 100% |

同预算对比：baseline 在 198 solves 时 J ≈ **0.871467**，在 200 solves 时 ≈ **0.871667**；v3c 在 202 solves 时 **0.872816**，略高于 baseline 的 200-solve 状态，但仍大幅低于 baseline 在 **226 solves** 时的 **0.886254**。

## 结论

- `refresh_physics_fn` 确实接通并执行了 **7 次**，`refresh_errors=0`，`adjoint_refreshes=7`；刷新机制本身正常。
- v3c 比 v3b 高 **+0.001804**（0.872816 vs 0.871012），但多花了 28 次伴随 solves，总 solves 约 230；仍未越过 baseline 226 solves 的 0.886254 台阶。
- 与基线最终 0.886449 的差距为 **0.013633**；刷新后的代理仍主要抓到 +0.0001~+0.002 的小翻转，没有出现 baseline 那种 +0.007~+0.015 的核心大跃迁。
- `--max-fdtd 110` 没有跑满是因为 `run_surrogate_validation.py` 默认 `--max-rounds 50`；若要跑 110 states，应加 `--max-rounds 55`。

## 下一步建议

- 先检查刷新后的 `adjoint_refresh_*.npz` 与当前 state 的 `eps`/`grad_eps` 相关性是否随翻转累积而变差；若刷新后梯度方向仍然弱，考虑把 `eps_fine` 上采样/对齐再做 Ridge，而不是只使用 30×30 block。
- 把 `refresh_after_no_improve` 的触发从 4 轮改成 6~8 轮，避免频繁刷新导致经验池一直只有 8 条、模型无法学习。
- 或将 `top_k` 从 2 调到 3，先用更多 FDTD 样本稳定 Ridge，再在 50 轮内寻找大跃迁。
- 全图 2 模搜索可作为下一步，但应先用当前 v3c 的 0.873 结果确认刷新不会破坏已找到的局部结构。

## 文件

- `adjoint_init_2mode.npz`
- `adjoint_refresh_1..7.npz`（远程 run 目录）
- `surrogate_history.csv`
- `summary.json`
- `final_state.npz`
- `convergence.png`
- `stdout.log` / `stderr.log`
