# 2-mode v6b — boundary Pass1 + interior Pass2/3 + port guard

Run: `F:\codex_project\inverse_design_sun\runs\surrogate_2mode_ablation_v6b_boundary_ports`

## 设置

- 全图 900 格，`--ablation-mask` 关闭
- `--strategy passwise --pass-top-n 60,48,30`
- `--pass-active-boundary 1,0,0`：
  - Pass 1：严格 4 邻域动态边界
  - Pass 2/3：开放全图内部像素，但锁住最左/最右 2 列端口入口
- `--max-fdtd 150 --enable-refresh`
- 2-flip 逃逸：`pair_escape_top_m=4`
- full-grid exporter：`B_SURROGATE_FULL`，初始 J 精确复现 `0.8498173602`

## 结果

| 指标 | v6b | baseline |
|---|---:|---:|
| 初始 J | 0.849817 | 0.849817 |
| 最终 J | **0.886669** | **0.886449** |
| 增益 | **+0.036852** | +0.036632 |
| 接受翻转 | **14** | 19 |
| FDTD 状态评估 | **123**（246 forward solves） | 212（424 solves） |
| 伴随刷新 | **2×4 = 8 solves** | — |
| 总 solves | **254** | 424 |
| 每状态命中率 | 11.48% | 8.96% |
| 每 solve 命中率 | 5.69% | 4.48% |
| 墙钟时间 | **2h37m** | 5h15m |
| 达到 baseline 增益比例 | **100.6%** | 100% |

相对 baseline：

- vs baseline final `0.886449`：**+0.000220**
- vs baseline 226 solves `0.886254`：**+0.000415**
- **达到 0.886669 的关键点实际在 `fdtd_evals = 89`：178 forward solves + 4 初始伴随 = 182 总 solves**；后续 Pass 3 + 6 组 pair 是收敛确认，不参与“首次达标”预算。
- 总 solves：**254 vs 424（减少约 40%）**
- 所有 **accepted 状态及最终 `final_audit.gds`** 均通过 248 类 PDK DRC 0 违例；试探阶段的 `trial_drc` 曾出现 1/2 个 violations，均被 Tier-2 门控拦截，没有进入 FDTD 或接受状态。

## 光谱审计（独立复算）

对 v6b `final_audit.gds` 重新跑 2 输入光谱，并与 `B_strict` 的 `final_spectra.npz` 对比：

| 指标 | v6b | baseline | Δ |
|---|---:|---:|---:|
| J | 0.886669 | 0.886449 | **+0.000220** |
| IL TE0 (dB) | 0.3764 | 0.3844 | **−0.0080** |
| IL TE1 (dB) | 0.6734 | 0.6671 | +0.0063 |
| IL mean (dB) | 0.5249 | 0.5258 | **−0.00084** |
| CT mean TE0→TE1 (dB) | −17.1748 | −17.4357 | **+0.2609** |
| CT mean TE1→TE0 (dB) | −18.7426 | −19.8620 | **+1.1194** |
| CT worst TE0→TE1 (dB) | −16.4993 | −16.6164 | +0.1171 |
| CT worst TE1→TE0 (dB) | −18.5223 | −19.5123 | +0.9900 |

结论：平均 IL 略好（−0.00084 dB），但 **第二通道串扰变差约 1.12 dB**（worst 约 0.99 dB），不是“完全没有退化”。如果论文/指标要求 CT 不劣于 baseline，需要把 CT 加入接受判据或做 Pareto 收尾；如果只按 J/target 指标，这组仍是合格突破。

## 关键翻转

Pass 1（严格动态边界）：

- `(15,12)` +0.005851
- `(16,15)` +0.008521
- `(19,25)` +0.002120
- `(21,27)` +0.003793
- `(1,0)` +0.000572
- 以及 `(18,25)`、`(17,24)`、`(4,18)`、`(5,18)`、`(13,24)`、`(4,17)`、`(3,19)` 等小增益

Pass 1 结束 J ≈ **0.872630**。

Pass 2（开放内部，端口两列锁死）：

- `(22,15)` +0.007236 → J 0.879865
- **`(23,21)` +0.006804 → J 0.886669**

Pass 3（Top-30 重扫）没有新的 accepted flip；最后的 pair escape 6 组均为负，`pair_escapes=0`。

## 结论

这是目前 **第一组在 2 模 108-grid 基准上真正超过 B_strict baseline 的结果**：

- 最终 J = **0.886669** > baseline **0.886449**
- 总 solves = **254** < baseline **424**
- 0 个 248 类 PDK DRC 违例
- 关键机制是：
  1. Pass 1 动态边界，避免内部掏洞废步；
  2. Pass 2/3 开放内部但锁端口两列，保留 `(22,15)`、`(23,21)` 这类内部高价值翻转；
  3. `|grad_eps|` + 2-flip escape 保留组合增益。

## 文件

- `summary.json`
- `surrogate_history.csv`
- `pair_history.csv`
- `spectrum_audit.json`
- `final_spectra.npz`
- `final_DRC_result.json`
- `final_120nm_checks.json`
- `final_audit.gds`
- `convergence.png`
- `stdout.log` / `stderr.log`
- `final_state.npz`
