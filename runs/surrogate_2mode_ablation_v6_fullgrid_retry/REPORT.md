# v6 — 2-mode full-grid 900 (no Canny) launch check

Run target: `runs/surrogate_2mode_ablation_v6_fullgrid_retry`

## 结论

**全图 900 格 exporter 目前还不能直接启动 v6。**  
这不是搜索算法的问题，而是 full-grid 初始态的 120 nm 二值化与原始曲线几何不兼容，导致 KLayout DRC 直接拒绝全部候选。

## 实测

初始态来自 `initial.gds` 的 30×30 raster：

- `full_init = cell_area_fraction > 0.5`，true = **703 / 900**
- full GDS J 评估：`J = 0.7647604071559595`（2 forward solves）
- `geometry_io.supplemental(full_gds, 120nm)`：
  - original `initial.gds`: `width = 0`, `space = 0`
  - full 703-cell GDS: `width = 0`, `space = 8`
- 使用 `runner.initial_state`（93 true，仅 108 candidate）导出的 full GDS：`width = 33`, `space = 16`

因此 `PasswiseMagnitudeDigitizer` 的 DRC 预过滤找不到任何合法候选：

- `fdtd_evals = 1`（只评估初始态）
- `total_flips = 0`
- `final_J = 0.764760`
- `adjoint_refreshes = 0`
- `pair_escapes = 0`

另一次首跑在 `fom1_mode_exp` 处出现 3 次 `"Can not find result 'expansion for Output'"` 后失败；重跑确认 initial GDS 本身可评估（original `J=0.849817`），说明首跑是 transient 仿真失败，不是 GDS 完全不可读。

## 根因

`initial.gds` 的核心层有大量 **sub-cell 曲线/部分填充**。全图 900 格用 120 nm 二值单元重新表达时：

- 将 `>0.5` 的 cell 全部填满，会产生 8 nm 的 space 违规；
- 只用 93 个 candidate 的 `initial_state` 又会产生 width 33 / space 16 违规；
- 原图 `width=0, space=0` 的精确曲线不能被简单的 900 个布尔 cell 无损表示。

所以 v6 卡在 **full-grid state representation / exact geometry preservation**，不是 `|grad_eps|` 筛选或 2-flip 逻辑。

## 下一步建议

1. 保留原始 core 的 **矢量基础几何**，不要把 900 格全部二值化：
   - 以 `initial.gds` 的 core 为 base；
   - 仅对 cell 做 add/subtract 操作，并在每次操作后重新跑 Tier-1 DRC；
   - 需要一个 `full_grid_export`，它从原始 core 开始而不是从 `full_fixed_core + 全 cell` 开始。
2. 或把初始态改为 **partial-cell 表示**（每格保留面积分数/多边形），但当前 `PasswiseMagnitudeDigitizer` 的状态是 bool，需要升级为多值 cell 或直接保存区域对象。
3. 在 full-grid exporter 修好前，不要启动 4 模；2 模 108 格上的已验证结论仍是：
   - v5b pair `(20,13)+(23,21)` → **0.882015**
   - 但从 pair 状态继续 Top-30 没有新翻转，当前 108 格候选集已到局部极值。

## 文件

- `summary.json`
- `surrogate_history.csv`
- `stdout.log`
- `stderr.log`
- 本地 full GDS 调试：`runs/v6_debug.gds`（703-cell full init）
