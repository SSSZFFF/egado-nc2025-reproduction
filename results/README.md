# 结果目录说明

所有结果均为 Lumerical FDTD 数值仿真结果，不含流片或实验测量。

## two_mode/

两模 MUX 平台（3.6 × 3.6 μm²，30 × 30 个 120 nm 像素）。

| 子目录 | 内容 | 关键指标 |
|---|---|---|
| `strict/` | B_strict：108 个候选格、严格 DRC 门禁下的最终结果 | J = 0.886449；IL 0.384 / 0.667 dB；CT −17.44 / −19.86 dB；DRC 0 违规 |
| `stage2/` | Stage 2：在 B_strict 终态上尝试 119 个相邻格对的联合动作 | 与 strict 终态相同；最大正差 8.1e−5 < 1e−4，无接受动作 |
| `A_D0/` | A 方案（D = 0，238 个候选格）的审计结果 | J = 0.898436；公共 DRC 38 违规（不可制造基线） |
| `figures/` | 两模过程图、DRC 对照、量化分析图 | 仅用于解释，不作为判优依据 |

每个结果目录通常包含：

- `final_spectra.npz`：`J`、`T` (2×2×11)、`f`、`wavelength_nm`、`state`、`mask`
- `final_state.npz`：最终二值状态
- `final.gds`：最终版图
- `history.csv`：DBS 逐候选记录
- `dbs_run.log`：运行日志
- `final_120nm_checks.json`：supplemental 120 nm Si 线宽/间距检查
- `final_DRC/result.json`：公共 248 类 DRC 报告

`T` 的索引含义：`T[input, output_mode, frequency]`，其中两模的目标为 `T[0,0,:]`（上输入→TE0）和 `T[1,1,:]`（下输入→TE1）；串扰为 `T[0,1,:]`、`T[1,0,:]`。

## four_mode/

四模 MUX（6 × 6 μm²，50 × 50 个 120 nm 像素）。

| 文件 | 内容 |
|---|---|
| `final_verification_spectra.npz` | 最终 21 频点复验：`mean_trans`（4 通道）、`overall_J`、`overall_IL`、`CH1`–`CH4`（每个含 TE0–TE3 谱） |
| `final_state.npz` | 最终二值状态 |
| `history.csv` | DBS 3 轮 × 161 候选的逐条记录 |
| `dbs_run.log` | DBS 运行日志 |
| `final.gds` | DBS 最终版图 |
| `final_core_6x6.gds` | 6 × 6 μm² 核心区版图 |
| `final_tapeout_120nm.gds` | 120 nm 投片候选版图（**最终 DRC 报告未归档**） |
| `baseline_4mode_cleaned_iter266.gds` | TO 266 轮清理后的基线版图 |
| `to_convergence_iter266.png` | TO 收敛曲线 |
| `scheme_b_iter266_cleaning_comparison.png` | 曲线骨架与清理版对比 |
| `scheme_b_true_iter266_spectra.png` | 清理版四通道光谱 |

四模最终验证汇总（21 频点）：

- `overall_J = 0.7262847765`
- `overall_IL = 1.403949 dB`
- `mean_trans = [0.82906474, 0.71985236, 0.67438136, 0.68184065]`（CH1–CH4 目标模式平均透射）

## 复算示例

```python
import numpy as np

d = np.load("results/two_mode/strict/final_spectra.npz")
print(float(d["J"]))
print(d["T"].shape)          # (2, 2, 11)
print(d["wavelength_nm"])    # 实际求解器频点对应的波长

d4 = np.load("results/four_mode/final_verification_spectra.npz", allow_pickle=True)
print(float(d4["overall_J"]), float(d4["overall_IL"]))
print(d4["mean_trans"])
```

## 指标口径

- `J` 是优化目标：两模为两个目标通道平均透射的均值；四模为四个目标通道目标模式透射的均值。
- `IL = −10 log10(mean(T_target))`，以 dB 为单位；两模报告逐通道值，四模报告四通道平均。
- 串扰 `CT = 10 log10(mean(T_wrong) / mean(T_target))`，越负越好。
- `T_forward` 以输入功率归一化，是模式透射效率，不是输出模式纯度。
