# Surrogate 两文件静态核对（2026-11-24）

**状态说明**：本文记录的是用户最初粘贴版本的核对结果。当前仓库中的 `run_surrogate_validation.py` 已按 A1-A5、B1-B3、远程 v261 路径做了修复；A4 的 2 模全图导出和 B1 的 4 模全图 fixed background 已加入。下面保留原始问题记录，便于对照。

对象：

- `physics_surrogate_digitizer.py`
- `run_surrogate_validation.py`

核对方式：只读代码 + 仓库接口 AST/常量核对 + 不启动 FDTD 的几何检查。**未启动任何 Lumerical 仿真。**

`python -m py_compile` 通过；`run_surrogate_validation.py --help` 通过；`PhysicsContext` 用 `2mode_final.npz` 的 `eps` 做了 181→30 分块 smoke test。

## 重点核对结论

| 用户点名检查项 | 结论 |
|---|---|
| `run_2mode` 的 `runner.initial_state` | ✅ 存在；`DBSRunner.__init__` 通过 `geometry_io.inputs()` 设置为 B 的 30×30 初始态 |
| `run_2mode` 的 `runner.export_state_gds` | ❌ **不存在**；`DBSRunner` 只有 `__init__ / init_history / check_drc / run`，内部走 `geometry_io.export_geometry()`；原样运行会在第一个候选处 `AttributeError` |
| `run_4mode` 不带 `--ablation-mask` 时 `_state_to_region` 与 `fixed_bg` 的重叠 | ❌ **确实重叠且会破坏全图搜索**；`fixed_background.gds` 的 32.867741 µm² 全部落在 50×50 = 2500 个 cell 范围内，161 格 ablation 时交集为 0 |

---

## A. 2 模路径的阻断项

### A1. `DBSRunner` 没有 `export_state_gds`

`run_dbs.py` 的 `DBSRunner` 实际方法只有：

```
__init__, init_history, check_drc, run
```

它内部导出临时 GDS 的方式是（`run_dbs.py:355-358`）：

```python
trial_npz = self.scratch_dir / "trial_state.npz"
np.savez(trial_npz, silicon_state=trial_state, candidate_mask=self.mask)
geometry_io.export_geometry(self.root, self.scheme, trial_npz, trial_gds)
```

但 `run_surrogate_validation.py` 在 L116、L125、L131 三次调用 `runner.export_state_gds(...)`，会直接 `AttributeError`。

建议在 launcher 里加 adapter，而不是改 `DBSRunner`：

```python
def export_state_gds(state, gds_path):
    state_npz = scratch_dir / "surrogate_state.npz"
    np.savez(state_npz, silicon_state=state, candidate_mask=runner.mask)
    geometry_io.export_geometry(handoff_dir, runner.scheme, state_npz, Path(gds_path))
```

并且把三处 `runner.export_state_gds` 全部替换为 `export_state_gds`。

### A2. 120 nm 补充 DRC 的返回字段读错

`geometry_io.supplemental()` 返回的是：

```python
{"width": <count>, "space": <count>, ...}
```

而 `check_drc_2mode` 当前写的是：

```python
return int(res.get("total_violations", res.get("violations", 0))) == 0
```

`width` / `space` 不为零时，这个表达式仍然返回 `True`，等于所有试验状态都放过 DRC。

建议改为：

```python
return res["width"] == 0 and res["space"] == 0
```

另外如果要复刻原 B_strict 的“公共 248 类 DRC”口径，应在 supplemental 之后调用 `runner.check_drc(trial_gds)`，而不是只跑 supplemental。

### A3. `eval_2mode` 的 cell name 写错

`geometry_io.write_gds()` 写出的 top cell 是：

```python
f"{scheme}_FULL_DEVICE"  # 即 B_FULL_DEVICE
```

但 `eval_2mode` 调用：

```python
evaluator.evaluate_state(trial_gds, "TOP", key, scratch_dir)
```

`DBSRunner.run()` 自己用的是：

```python
cell_name = f"{self.scheme}_FULL_DEVICE"
```

所以这里应改为 `f"{runner.scheme}_FULL_DEVICE"` 或 `"B_FULL_DEVICE"`，否则 Lumerical 找不到 cell。

### A4. 不带 `--ablation-mask` 的 2 模全图搜索无法使用当前 `geometry_io`

`geometry_io.export_geometry()` 内部调用 `read_states()`，会强制要求：

- 只能改 `candidate_mask`（B = 108 格）
- 非候选格必须与 `initial_state` 完全一致

我实测将全 30×30 状态传给 `export_geometry()`，直接报：

```
ValueError: A non-candidate state changed; this is not the agreed masked DBS
```

因此 `--mode 2mode --ablation-mask` 可以工作（如果 A1-A3 修好），但 `--mode 2mode`（全图 900 格）会在这里失败。

全图搜索需要单独做一个全图几何导出器：用 `inputs/B_curve_DRC120/initial.gds` 的 `CORE` 减去全部 30×30 个 cell box 得到真正的 fixed background，再把当前 `state` 中所有 True 的 cell box 加回去；不要用 B 方案的 `fixed_background.gds`。

### A5. 路径默认值与远程原始项目不一致

`run_2mode` 硬编码：

```python
handoff_dir = repo / "code" / "dbs" / "handoff_2026-10-11"
to_npz = repo / "code" / "two_mode" / "analysis-2mode" / "2mode_final.npz"
```

但远程 `F:\codex_project\inverse_design_sun` 的结构是：

```
DBS_handoff_2026-10-11/
EGADO-4mode/
```

没有 `code/dbs`，`2mode_final.npz` 实际在 `DBS_handoff_2026-10-11/inputs/common/2mode_final.npz`。

建议加 fallback：

```python
handoff_dir = repo / "code" / "dbs" / "handoff_2026-10-11"
if not handoff_dir.exists():
    handoff_dir = repo / "DBS_handoff_2026-10-11"

to_npz_default = repo / "code" / "two_mode" / "analysis-2mode" / "2mode_final.npz"
if not to_npz_default.exists():
    to_npz_default = handoff_dir / "inputs" / "common" / "2mode_final.npz"
```

### A6. 本地 curated repo 少 `pdk-index.local.json` 和规则文件

`DBSRunner.check_drc()` 读：

```python
self.pdk_index_path = root / "config/pdk-index.local.json"
```

本地仓库只有 `pdk-index.template.json`，没有 `pdk-index.local.json` 和 `Wuyue_PDK_Klayout_Common_DRC_rule_V0.9.0.lydrc`。远程 `F:\codex_project\inverse_design_sun\DBS_handoff_2026-10-11\config` 里有这两个文件。若从本地仓库跑，需要先把它们复制进去；若从远程原始项目跑，路径 fallback 即可。

---

## B. 4 模路径的阻断项

### B1. `fixed_bg` 与全图搜索冲突（用户提示的问题属实）

我实际量了远程/本地 `EGADO-4mode/scheme_b_dbs_inputs/fixed_background.gds`：

- fixed region 总面积：`32.867741 µm²`
- 与 161 个 candidate cells 的交集：`0 µm²`（ablation 模式安全）
- 与全部 50×50 = 2500 个 cell boxes 的交集：`32.867741 µm²`（100% 的 fixed 区域都落在全图 cell 范围内）

因此不带 `--ablation-mask` 时：

```python
active_mask = np.ones_like(trial_state, dtype=bool)
```

会把 2500 个 cell 都写进 `candidate_si_boxes`，而 `base_opt.export_gds()` 仍执行：

```python
device = self.fixed_bg | candidate_si_boxes
```

结果 B 方案原有的 fixed 硅图形覆盖在设计区上，无法通过 state 的 0 翻转形成孔。全图 4 模搜索结果不可信。

解决方式二选一：

1. 从 `initial.gds` 减去全部 2500 个 cell box，重建 `fixed_background_full.gds`，并给 `SchemeBDBSOptimizer` 增加 `fixed_bg_full` 或 `export_gds_full`；或
2. 明确禁止 4 模全图搜索，只允许 `--ablation-mask`（161 格）。这更接近当前可验证范围。

### B2. 本地 repo 的 4 模输入缺失

`run_dbs_4mode.py` 固定：

```python
INPUT_DIR = REPO_DIR / "scheme_b_dbs_inputs"
```

当前 curated repo 的 `code/four_mode/` 没有 `scheme_b_dbs_inputs/`（只有脚本）。需要从 `G:\codex_project\inverse_design_sun\EGADO-4mode\scheme_b_dbs_inputs` 或远程同路径复制：

- `initial_state.npz`
- `initial.gds`
- `fixed_background.gds`

否则 `SchemeBDBSOptimizer.__init__` 会 `FileNotFoundError`。

### B3. 远程原始项目路径 fallback

`run_4mode` 硬编码：

```python
four_dir = repo / "code" / "four_mode"
```

远程原始项目是 `repo / "EGADO-4mode"`。建议同样加 fallback：

```python
four_dir = repo / "code" / "four_mode"
if not four_dir.exists():
    four_dir = repo / "EGADO-4mode"
```

### B4. 远程 Lumerical 版本硬编码

远程只装了 v261 的 Lumerical API：

```
D:\Program Files\ANSYS Inc\v261\Lumerical\api\python\lumapi.py  -> True
```

但 `run_dbs.py:29` 固定搜索：

```python
LUMERICAL_API = r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python"
```

`run_dbs_4mode.py:27-29` 也只搜索 v251 的 E/D/C 三个路径。当前远程机器上这些 v251 路径都不存在，`import lumapi` 会失败。需要把 v261 加到两处搜索列表首位，或在启动器里提前 `sys.path.insert(0, r"D:\Program Files\ANSYS Inc\v261\Lumerical\api\python")`。

### B5. 远程 Python 环境

远程测得：

- `py -3.12` 有 `klayout`，但没有 `torch`；`lumapi` 需要手动把 `D:\Program Files\ANSYS Inc\v261\Lumerical\api\python` 加入 `sys.path`（已实测可 import）。
- `C:\Users\22346\.conda\envs\lumopt\python.exe` 有 `lumapi`（加 path）但没有 `klayout`、没有 `torch`。

因此实际跑之前需要确认使用哪个解释器：

- 选项 A：`py -3.12` + 手动 `sys.path` 加 v261 API 路径（有 KLayout、无 torch，自动走 NumPy Ridge fallback）。
- 选项 B：`lumopt` 环境 + 把 base Python 的 `site-packages` 加入 `sys.path`（跨环境 import KLayout 已实测可，但比较脏）。
- 若要 PyTorch CNN 版，需要先安装 torch；否则 `HAS_TORCH=False`，走 Ridge fallback。

远程 `py -3.12` 实测 `HAS_TORCH=False`，因此当前实际运行会走 NumPy Ridge fallback；本地有 torch，`--self-test` 走 PyTorch CNN。

---

## C. 非阻断但建议改

1. `PhysicsContext.has_true_adjoint` 只看 `e2_map` 是否非零；如果只提供 `grad_eps`、没有 `e2_field`，真实伴随梯度不会生效，会退回 analog prior。建议增加 `self.has_grad` 标志，并让 `linear_prior` 在 `grad_map` 为真实梯度的条件下使用它。
2. `SurrogateGuidedDigitizer` 每轮对 mask 内全部候选调用 `check_drc_fn`。2 模当前 callback 每次都写完整 GDS 再跑 supplemental DRC，108 格/轮会很慢；全图 900 格更慢。建议先测一次 `check_drc_fn` 单次耗时，再决定是否缓存 DRC 结果或做增量 DRC。
3. `surrogate_history.csv` 每次 `run()` 用 `"w"` 覆盖，没有 resume；如果跑长实验，建议按 `--out-dir` 分目录或增加 resume。
4. `run_4mode` 当前 `to_npz=None`，`PhysicsContext` 只用 binary initial state 的边缘幅值，不是真正的 Stage-1 `eps`；建议后续找 `EGADO-4mode` 的 `parameters_*.npz` 并显式传 `--to-npz`。
5. `run` 的 `hit_rate` 以 `fdtd_evals - 1` 为分母，和原 DBS 的 19/424 口径不完全一致；对比时建议同时报告绝对翻转数和总 FDTD solve 数。

---

## D. 建议的下一步

先修 A1-A5（2 模 ablation 必需），再跑：

```powershell
python run_surrogate_validation.py --mode 2mode --repo . --ablation-mask --max-fdtd 150
```

在未修 A1-A5 前不要启动，否则第一个候选就会在 `export_state_gds` 处失败，或者 DRC 全部误判为通过。

## E. 当前修复记录（2026-11-24）

- A1：`run_surrogate_validation.py` 增加本地 `export_state_gds()` 适配器，调用 `geometry_io.export_geometry()`。
- A2：`geometry_io.supplemental()` 改为读取 `width` / `space`；若 `pdk-index.local.json` 存在则优先调用 `DBSRunner.check_drc()`。
- A3：2 模 evaluation cell name 改为 `B_FULL_DEVICE`。
- A4：2 模无 mask 模式增加全图 exporter：`initial.gds` 的 CORE 减去全部 30×30 cell box，再按 state 插入。
- A5：增加 curated repo / 远程原始项目两套路径 fallback，并增加 `DBS_交接包_2026-10-11` 兼容。
- B1：4 模无 mask 模式将 `base_opt.fixed_bg` 置空（initial.gds 与 2500 个 cell box 的差集为 0，说明 B 方案 fixed 区域全部位于设计区内部）。
- B3/B4：增加 `EGADO-4mode` fallback；启动前将 v261 API 路径加入 `sys.path`。
- 4 模输入已复制到 `code/four_mode/scheme_b_dbs_inputs/`。

### 实测性能（远程 DESKTOP-IJUDLO1）

- `DBSRunner.check_drc()` 完整 248 类公共 DRC：约 **5.34 s/候选**（KLayout CLI 启动占主要时间）。
- `geometry_io.supplemental()` 120 nm Si 宽/距检查：约 **0.009 s/候选**。

因此 2 模路径采用两级 DRC：`check_drc_2mode` 用 supplemental 快速筛选全部 108 个候选，`verify_drc_2mode` 只对 Top-K 用完整 248 类复核。若每轮 4 个 Top-K，则 DRC 约 `108×0.009 + 4×5.34 ≈ 22 s/轮`，比全部候选都做 248 类（约 9.6 min/轮）低两个数量级。

