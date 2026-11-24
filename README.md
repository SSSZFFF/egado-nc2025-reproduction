# EGADO（Nature Communications 2025）复现项目

对 **Sun, A., Xing, S., Deng, X. et al. _Edge-guided inverse design of digital metamaterial-based mode multiplexers for high-capacity multi-dimensional optical interconnect_. Nature Communications 16, 2372 (2025)** 的独立复现与研究档案。

- 论文 DOI：[10.1038/s41467-025-57689-7](https://doi.org/10.1038/s41467-025-57689-7)
- 研究对象：EGADO（Edge-Guided Analog-and-Digital Optimization）算法及数字超材料模分复用器（MUX）
- 本仓库定位：**复现记录 + 可核查数据 + 可复用脚本**，不是作者官方代码仓库

---

## 结论：没有完成论文级完整复现

当前状态是 **部分复现**：两模链路基本打通，四模完成了数值设计但指标未全面达到论文，五模主结果尚未完成。

| 论文对象 | 论文结果 | 本仓库结果 | 判定 |
|---|---:|---:|---|
| **五模 MUX（论文主结果）** | Table 1：IL 1.97 dB，CT −20 dB；仿真 IL < 1.96 dB，CT < −15.81 dB | 仅完成 TO 阶段；`parameters_69.npz` 的 discreteness 约 0.625，未充分二值化；无五模数字版图和 5×5 传输谱 | ❌ 未复现 |
| **四模 MUX** | Table 1：IL 1.24 dB，CT −18 dB（实测 back-to-back 折算） | TO 266 轮 + 受限 DBS 3 轮；最终验证 `J = 0.7263`，平均 IL **1.404 dB**，四通道目标透射 82.9 / 72.0 / 67.4 / 68.2%，最差串扰约 **−14.3 dB**；最终版图已导出，最终 DRC 报告未归档 | ⚠️ 数值接近，串扰未达标 |
| **两模（补充 Fig. S8 小规模验证）** | D = 0 直接转换率 71.22% | TO 连续域 ΣT = **1.966 / 2.0**，插损 0.025 / 0.125 dB；直接转换率 **73.56%** | ✅ 链路与量级复现 |
| **两模数字器件（本仓库严格版）** | 项目自设目标 J ≥ 0.92 | J = **0.8864**，IL 0.384 / 0.667 dB，CT −17.44 / −19.86 dB；公共 248 类 DRC + 120 nm Si 线宽/间距检查 0 违规 | ⚠️ 合规，但 J 未达 0.92 |

> 一句话：**两模能复现方法链路，四模做出了可验证的数值器件，五模和论文的流片实测结果没有复现。**

详细状态与证据见 [docs/00_reproduction_status.md](docs/00_reproduction_status.md)。

---

## 关键结果

### 两模严格版（B_strict）

- 设计区：3.6 × 3.6 μm²，120 nm 数字像素，30 × 30 = 900 格
- TO：249 条收敛记录，最终 β = 708.80，投影后 discreteness = 0.9914，两通道 ModeMatch 之和 1.9660 / 2.0
- 数字 DBS：4 轮 × 108 候选格，19 次接受翻转，424 次 forward solve
- **J = 0.886449**
- 上输入 → TE0：91.53%；下输入 → TE1：85.76%
- 插损：**0.384 dB / 0.667 dB**
- 串扰：**−17.44 dB / −19.86 dB**（最差频点 −16.62 / −19.51 dB）
- DRC：公共 248 类规则 0 违规；supplemental 120 nm Si 线宽/间距 0 违规

数据：[results/two_mode/strict](results/two_mode/strict) · 审计：[results/two_mode/strict_final_audit.json](results/two_mode/strict_final_audit.json)

### 四模（DBS phase 2）

- 设计区：6 × 6 μm²，50 × 50 个 120 nm 像素
- TO：266 轮，最终 β ≈ 38.34，discreteness ≈ 0.927
- 受限 DBS：3 轮 × 161 候选格，最终 `J = 0.721646`（11 频点）
- 最终 21 频点复验：`J = 0.726285`，平均 IL **1.404 dB**
- 四通道目标透射：82.9 / 72.0 / 67.4 / 68.2%
- 最差串扰：约 **−14.3 dB**
- 最终 GDS：[final_tapeout_120nm.gds](results/four_mode/final_tapeout_120nm.gds)

数据：[results/four_mode](results/four_mode)

### 五模

未完成。当前只有 TO 中间结果，参数 `discreteness` 约 0.625，尚未进入可信的数字阶段。仓库中的 `upstream/EGADO-main/` 是作者公开的原始代码快照；作者提供的 `demo.fsp` 未纳入本仓库（体积原因）。

---

## 仓库结构

```
.
├── README.md
├── docs/
│   ├── 00_reproduction_status.md      # ★ 复现状态总表
│   ├── 01_项目详解.md                 # 原始代码逐段分析与问题清单
│   ├── 02_阅读分析与结论校核.md        # 2026-10-11 独立校核
│   ├── 03_改进方案_专家评估说明.md     # 制造性筛选构想与更正
│   ├── 04_B结构修复复算.md            # 几何修复后的光学复算
│   ├── 05_严格版与Stage2_审计.md      # 两模严格版/Stage2 审计
│   ├── 06_两方案DBS_跨电脑操作指南.md
│   ├── 07_接手任务说明.md
│   ├── 08_DBS交接包_先读我.md
│   └── geometry/                      # 连续曲线、局部填格、DRC 对照
├── code/
│   ├── two_mode/analysis-2mode/       # 两模 TO/EG-ATD/DRC/量化分析脚本
│   ├── dbs/handoff_2026-10-11/        # 两方案 DBS 交接包（脚本+输入+证据）
│   ├── four_mode/                     # 四模 TO/DBS/复验脚本
│   └── audit/                         # 只读审计脚本
├── results/
│   ├── two_mode/                      # strict / stage2 / A_D0 最终数据
│   ├── four_mode/                     # 四模最终数据与图
│   └── ...
├── paper/                             # 论文文本抽取与 Table 1（不含 PDF）
└── upstream/                          # 作者 MIT 许可的公开源码快照（仅源码）
```

---

## 环境

已验证环境：

- Windows + PowerShell 7
- Python 3.12
- Ansys Lumerical 2025 R1（`lumapi` + `lumopt`）
- KLayout Python API（`klayout.db`，0.30.x）
- NumPy / SciPy / Matplotlib

复现脚本中的 Lumerical 路径、远程主机别名、盘符等可能需要按本机环境修改。原始记录保留绝对路径作为可追溯证据，运行前请统一改为自己的路径。

---

## 快速开始

```powershell
python -m pip install numpy scipy matplotlib
# 设置 Lumerical API 路径（示例）
$env:LUMERICAL_API = 'C:/Program Files/ANSYS Inc/v251/Lumerical/api/python'

# 两模 TO / 分析
python code/two_mode/analysis-2mode/run_to_2mode.py
python code/two_mode/analysis-2mode/drc_v3.py

# 四模最终光学复验（需要已有 GDS 和求解器）
python code/four_mode/run_final_optical_verification.py
```

详细脚本说明见各源码文件头部与 `docs/` 中的报告。

---

## Physics-residual surrogate 验证（新增）

仓库根目录新增：

- [physics_surrogate_digitizer.py](physics_surrogate_digitizer.py)：Stage 2 + Stage 3 合并的物理残差在线代理模型（支持 PyTorch CNN ensemble；无 torch 时自动退回 NumPy Ridge）。
- [run_surrogate_validation.py](run_surrogate_validation.py)：2 模 / 4 模验证启动器。

首轮 2 模消融：

```powershell
python run_surrogate_validation.py --mode 2mode --repo . --ablation-mask --max-fdtd 150
```

说明：

- 2 模路径使用两级 DRC：`geometry_io.supplemental()` 快速筛选全部候选，`DBSRunner.check_drc()` 只对 Top-K 做 248 类公共 DRC 复核。
- 4 模 `--ablation-mask` 使用现有 161 格候选；不带 mask 时启动器会把 Scheme-B 的 `fixed_bg` 置空，按 50×50 全图重建候选区域。
- 当前 2 模全图模式需要 `initial.gds` 的全图 fixed 区域（启动器已自动构造）；4 模全图模式仍需先确认全图 DRC/版图口径。
- 本目录的 `docs/SURROGATE_STATIC_REVIEW.md` 记录了接口核对、路径 fallback 与已知限制。
- 首组 2 模 ablation 结果：`runs/surrogate_2mode_ablation_v2/REPORT.md`（150 state evals = 300 solves；final J 0.860121；14 flips；详见 `surrogate_history.csv` 和 `summary.json`）。

---

## 复现边界（必须保留）

1. **没有流片、没有实验测量。** 本仓库所有性能都是数值仿真；论文的四/五/六模实测、324 Gb/s 单通道、38.2 Tb/s 总容量不在复现范围内。
2. **五模主结果未完成。** 不能用两模或四模结果替代论文的五模结论。
3. **DRC 只覆盖已执行的规则集。** 两模严格版通过的是 Wuyue SP90 V0.9.0 公共 248 类规则 + 120 nm Si 线宽/间距补充检查，不等于完整 foundry deck 或 Calibre sign-off。
4. **四模最终 DRC 未归档。** `final_tapeout_120nm.gds` 已生成，但最终公共 DRC 报告未在结果目录中保存，不能称为“可投片”。
5. **脚本含历史绝对路径。** 这是研究档案而非即开即用的软件包；运行前需要按 `docs/` 说明调整环境。

---

## 文档与证据入口

- 复现状态总表：[docs/00_reproduction_status.md](docs/00_reproduction_status.md)
- 两模严格版审计：[docs/05_严格版与Stage2_审计.md](docs/05_严格版与Stage2_审计.md)
- 两模数据：[results/two_mode/strict](results/two_mode/strict)
- 四模数据：[results/four_mode](results/four_mode)
- 论文文本与 Table 1：[paper/README.md](paper/README.md)

---

## 上游代码与许可

- 作者公开代码：`upstream/EGADO-2mode/`、`upstream/EGADO-main/`，MIT License，Copyright (c) 2025 Bryvin Sal。
- 本仓库中新增的复现脚本、文档和整理内容同样以 MIT License 发布，详见 [LICENSE](LICENSE)。
- 论文版权归原作者与 Nature Communications 所有；`paper/` 仅包含用于文本检索的开放获取文本抽取和 Table 1，不包含论文 PDF。

## 引用

```bibtex
@article{sun2025edge,
  title   = {Edge-guided inverse design of digital metamaterial-based mode multiplexers for high-capacity multi-dimensional optical interconnect},
  author  = {Sun, A. and Xing, S. and Deng, X. and others},
  journal = {Nature Communications},
  volume  = {16},
  pages   = {2372},
  year    = {2025},
  doi     = {10.1038/s41467-025-57689-7}
}
```
