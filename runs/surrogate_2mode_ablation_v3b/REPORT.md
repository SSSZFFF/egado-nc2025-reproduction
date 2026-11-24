# 2-mode surrogate v3b — true adjoint + EI + tabu

Run: `F:\codex_project\inverse_design_sun\runs\surrogate_2mode_ablation_v3b`

## 设置

- 初始态：B_strict `initial_state.npz`（30×30，108 个 candidate 格）
- 物理先验：`adjoint_init_2mode.npz`，由 `extract_discrete_adjoint.py` 在 `initial.gds` 上用 4 次 FDTD solve（2 模式 × forward/backward）提取
- 模板核对：`common_device_template.lsf` 的 `fom*_mode_exp` 实际 x = **2.1 µm**、y span = **1.9 µm**（不是脚本初版的 2.3 / 1.6）；已作为默认值
- 采集：EI，`top_k=2`，`max_fdtd=100` 状态（2-mode ×2 = **200 forward solves**；实际 99 状态 = 198 solves）
- 远程无 torch，backend = `numpy_ridge_ensemble`

## 验证

- 作者 `initial.gds` 的 top cell 是 `TARGET_120NM_OPTIMIZED`，不是 `B_FULL_DEVICE`；`extract_discrete_adjoint.py --cell auto` 已自动识别。
- `eps_block` 与 `initial_state` 的 candidate 区域相关系数为 **0.990**，方向/行列一致。
- 原始伴随公式与实测 `true_dJ` 呈负相关（约 −0.63）；已修正符号，修正后 `grad_eps` 与实测 delta 的相关性约 **+0.63**。
- Lumerical v261 不能在同一 session 里重复 eval 完整材料模板；改为每个 forward/backward solve 使用独立 session，稳定完成 4 次 solve。

## 结果

| 指标 | v3b | Baseline |
|---|---:|---:|
| 初始 J | 0.849817 | 0.849817 |
| 最终 J | **0.871012** | **0.886449** |
| 增益 | **+0.021194** | +0.036632 |
| 接受翻转 | **14** | **19** |
| FDTD 状态评估 | **99**（198 solves） | 212（424 solves） |
| 附加伴随 solve | 4 | — |
| 每 solve 命中率 | **7.07%** | 4.48% |
| 每状态命中率 | **14.29%** | 8.96% |
| 墙钟时间 | **4,565.61 s ≈ 1h16m** | 18,894.4 s ≈ 5h15m |
| 达到 baseline 增益比例 | **57.9%** | 100% |

同预算对比：baseline 在 **198 forward solves** 时 J ≈ **0.871467**；v3b 在 198 solves 时 J = **0.871012**，基本持平但略低。baseline 在 **226 solves** 时已达到 **0.886254**，v3b 没有追平这个目标。

## 结论

v3b 相比 v2 明显改善：同样用 `numpy_ridge_ensemble`，从 v2 的 0.860121（150 状态/300 solves）提升到 0.871012（99 状态/198 solves），命中率也从 4.67%/solve 升到 7.07%/solve。但相对真正的 B_strict baseline，仍只拿到约 58% 的增益；在同 198 solves 预算下与 baseline 基本持平，没能达到 226 solves 的 0.886254。

## 下一步建议

- 继续 v3b 到 `--max-fdtd 150`，看是否能跨过 baseline 在 226 solves 的台阶。
- 或先调 `ei_threshold`（当前 5e-4 偏保守，很多候选 acq_score 为 0）和 `top_k`（2→1 或 3），观察命中率是否提高。
- 若仍追不上，考虑开启 `refresh_physics_fn`（每 6 个接受翻转刷新一次伴随）或使用 `grad_eps`/`e2_field` 的更细网格特征。

## 文件

- `adjoint_init_2mode.npz`
- `surrogate_history.csv`
- `summary.json`
- `final_state.npz`
- `convergence.png`
- `stdout.log` / `stderr.log`
