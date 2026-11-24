# B_strict 与 Stage 2 结果审计交接文档

审计日期：2026-10-13  
工作区：`G:\\codex_project\\inverse_design_sun`  
审计方式：只读读取 JSON、CSV、NPZ、日志和 GDS；没有启动 FDTD，也没有重新运行 DRC。

配套复核脚本：[`audit_strict_final.py`](G:/codex_project/inverse_design_sun/analysis-recheck-20261013/audit_strict_final.py)  
配套机器可读结果：[`strict_final_audit.json`](G:/codex_project/inverse_design_sun/analysis-recheck-20261013/strict_final_audit.json)

## 结论先行

当前最稳妥的工程表述是：

> 在固定的 B 曲线骨架、108 个候选格、20 nm FDTD 网格、11 个实际频率采样点、常数折射率模型，以及 `B_strict` 的已执行门禁条件下，当前找到的合规结果为 `J=0.8864491686`。它通过了已执行的 Wuyue KLayout Common V0.9.0 的 248 类规则，并通过了额外的 Si 线宽/间距 120 nm 检查。

Stage 2 只说明：在这一个终版状态上，对 119 个 4 连通相邻格对各尝试一次“同时翻转两个格子”的动作时，没有动作达到 `dJ >= 1e-4` 的接受阈值。它不能证明 108 格搜索空间的全局最优，也不能证明物理极限。

## 已由文件和数据证明

### 1. 最终状态和光谱

- `results/B_curve_DRC120_strict/final_state.npz` 与 `results/B_curve_stage2_joint/final_state.npz` 的 `silicon_state` 完全相同，差异格数为 0。
- 两份最终光谱的最大逐元素差异约为 `3.11e-15`，可以视为同一光学状态的数值重复保存。
- 严格版最终 `T` 形状为 `(2,2,11)`，实际保存波长为 1530 nm 到 1570 nm 的 11 个求解器频率采样点。
- 由 `final_spectra.npz` 重算的 `J` 与保存值完全一致，绝对误差为 0。

| 指标 | 严格版最终值 |
|---|---:|
| `J` | 0.8864491686 |
| 上输入到输出 TE0 平均透射率 | 0.9152911880 |
| 下输入到输出 TE1 平均透射率 | 0.8576071491 |
| 上输入到输出 TE0 平均插损 `-10log10(mean(T))` | 0.3844 dB |
| 下输入到输出 TE1 平均插损 `-10log10(mean(T))` | 0.6671 dB |
| 两通道插损的算术平均 | 0.5258 dB |
| 上输入到错误 TE1 的平均比值串扰 | -17.4357 dB |
| 下输入到错误 TE0 的平均比值串扰 | -19.8620 dB |
| 频点最差比值串扰 | -16.6164 dB、-19.5123 dB |

这里的串扰是 `10log10(mean(T_wrong)/mean(T_target))`。`T_forward` 是相对输入功率归一化的前向模式透射效率，不能直接称为输出模式纯度。

### 2. Stage 1 严格版执行记录

`results/B_curve_DRC120_strict/history.csv` 有 432 行，正好是 4 轮 × 108 个候选格；记录了 19 次接受翻转。日志的最终值为：

- 初始 `J=0.8498172947`；最终 `J=0.8864491686`；净增约 `0.0366319`。
- 实际累计 forward solves：424。
- DRC 通过并进入光学判优的试探：229 条。
- 公共 PDK DRC 拒绝：44 条。
- supplemental 120 nm 检查拒绝：159 条。
- 第 4 轮接受数为 0，按代码停止。

从初始值和接受行的 `J_trial` 重新构造后，19 次接受增益均为正，最大增益为 `0.013694`，对应日志中的 `(23,21)` 格子。这支持“该次翻转带来约 `+0.0137`”这一数值，但不支持把 CSV 中的 `J_before` 字段当作真实接受前值。

### 3. 最终已执行的几何检查

- 严格版 final GDS SHA256：`9D36B3007C14087D7B4B18A8759347C4156F31BD84A45D3506EF865895100148`。
- 严格版 DRC JSON 记录的 GDS SHA 与当前工作区的 final GDS 一致。
- 公共检查结果：`rule_categories=248`、`violation_count=0`、`status=passed`。
- supplemental 结果：`threshold_nm=120`、`width=0`、`space=0`。
- Stage 2 final 也记录为 248 类、0 违规、120 nm width/space 为 0；其 GDS SHA 为 `8E82018C4317184D3FF92062E329C1B18E6D6E998FBACB9254B7E427A37DE6C7`。

注意：两份 final GDS 的字节 SHA 不同，虽然状态数组相同。两份 DRC 报告分别绑定各自的 GDS，不能用一份报告替代另一份报告。

## 必须改写或撤回的结论

### 1. “全局最优”“物理饱和上限”不能成立

Stage 2 代码枚举了 108 个候选格的 4 连通相邻对，日志确认数量为水平 64 对、垂直 55 对，共 119 对。但每一对只构造了一个候选状态：

- 当前为 `00` 时尝试 `11`；
- 当前为 `11` 时尝试 `00`；
- 当前为 `01` 时尝试 `10`；
- 当前为 `10` 时尝试 `01`。

因此它没有枚举该两格的全部联合状态，也没有单独尝试只翻第一格、只翻第二格，或非相邻两格、三格以上动作。49 个通过门禁并完成 FDTD 的候选中，记录的最大正差约为 `+8.1e-5`，低于 `1e-4` 接受阈值；这只能支持“当前动作集合内无接受动作”。

建议改写为：

> 在固定骨架和指定 119 个 4 连通相邻格对的单一互补动作邻域内，未发现达到 `1e-4` 接受阈值的候选。

Stage 1 本身也是固定掩码上的坐标式局部搜索，受扫描顺序、4 轮上限和 `1e-4` 阈值影响，同样不能给出全局最优证明。

### 2. “完整工业级 120 nm 规则全部通过”不能成立

`final_DRC/result.json` 的 `scope_note` 明确写明：

> V0.9.0 KLayout common rules only; not all V0.9.1 handbook conditions or electrical/optical connectivity.

`final_120nm_checks.json` 的 `scope` 明确写明：

> Supplemental Si width and space only; not a full foundry rule deck.

所以当前只能写成“通过已执行的 248 类公共宏和 supplemental Si 线宽/间距检查”，不能写成完整 foundry deck、Calibre sign-off 或代工厂认证。

### 3. “Foundry Ready / 直接投片”证据不足

目前还缺少或未在本结果目录中证明：

- 代工厂实际 sign-off deck 或 Calibre 结果；
- V0.9.1 手册全部条件、电气连通性和工艺层语义的签核；
- 版图端口连续性、层用途和主版图集成检查；
- 工艺偏置、工艺角、材料色散和制造容差复核；
- 20 nm 与更细网格的一致性复核；
- 最终源模型 `.fsp` 的完整留档与可重放验证。

因此当前状态是“已执行检查下的合规候选”，不是“可直接投片认证”。

### 4. 日志有明确记录 bug

`run_dbs.py` 和 Stage 2 脚本在接受判断后先执行 `current_J = trial_J`，随后才写入：

```python
"J_before": round(current_J, 6)
```

并用更新后的值生成 `ACCEPT` 日志。因此严格版 19 条接受记录的 `J_before` 都等于更新后的 `J_trial`，增益被记录为 0。当前审计通过按接受顺序重建状态，恢复了约 `+0.013694` 的关键增益，但以后运行必须先保存 `J_before = current_J`，再更新当前状态。

建议修复顺序：

```python
j_before = current_J
if trial_J > j_before + self.tolerance:
    current_state = trial_state
    current_J = trial_J
...
"J_before": round(j_before, 9)
"J_trial": round(trial_J, 9)
"J_current": round(current_J, 9)
```

### 5. 断点和配置元数据没有同步

- `B_strict.json` 仍写着 `offline_geometry_inputs_verified_no_FDTD_or_optical_DBS_has_been_run`，与实际结果矛盾。
- 严格版 `latest_checkpoint.json` 的时间早于最终输出，指向 `state_sweep2_step100.npz`，该状态与最终状态相差 1 个格子；它不能作为当前 final 的恢复指针。
- Stage 2 结果目录没有保留 `checkpoints` 目录，不能按脚本设计直接断点恢复。
- DRC JSON 内的 `gds` 路径仍是原机器的 `F:\...` 路径。当前 G 盘文件的 SHA 与报告 SHA 一致，但路径本身不可移植。

这不否定当前 final 文件，但交付前必须重新生成一份包含最终状态、输入 SHA、求解设置、规则 SHA 和实际输出路径的 `run_config.json`，并清理或标注过期 checkpoint。

## 远端需要补跑的项目

下面按优先级排列。它们是为了验证、修复表述和提高可交付性；本机审计没有代替这些工作。

### A. 最低限度的最终结果复核

1. 在保存最终 GDS 的同一台机器上，对严格版和 Stage 2 final 各重新运行一次同版本公共 DRC，使用新的输出目录；核对报告中的 GDS SHA。
2. 对两份 final GDS 各运行 supplemental 120 nm width/space，并保存新的 JSON。
3. 用相同光学模板、材料、端口、源设置、20 nm 网格和 11 个实际频率，对两份 final 各运行两个输入源；保存新的 `final_source1.fsp`、`final_source2.fsp` 和原始 `T_forward` 数据。
4. 对同一 final 至少重复一次双源评估，给出 `J` 波动；若波动接近或超过 `1e-4`，先重新标定接受阈值，不能继续使用当前阈值解释 DBS 收敛。

### B. 如果要继续声称 Stage 2 有效

修复日志后重新运行一个独立输出目录。每个相邻格对至少评估以下未固定状态中的全部候选：

- 只翻第一个格；
- 只翻第二个格；
- 同时翻两个格；
- 如果当前状态不是上述目标，保留当前状态作为基准。

每个候选都必须先通过公共 248 类规则和 supplemental 检查，再做两源光学评估。至少完成一轮 119 对，记录每个候选的真实 `J_before`、`J_trial`、`delta_J`、拒绝原因和求解数。这样才能把结论扩大到“完整的两格局部邻域内未发现增益”，仍然不能称全局最优。

### C. 如果要给出“可投片”结论

必须由实际代工厂 PDK/签核流程补充：完整 deck 或 Calibre sign-off、工艺角/偏置、层语义和连通性、端口与封装接口、器件阵列集成，以及制造容差下的光学复核。在这些结果返回前，报告中使用“候选版图”“已执行规则下通过”即可。

### D. 如果要验证物理性能稳定性

对严格版 final 和比较基线使用完全相同的：

- 20 nm 与 10 nm 设计区横向网格；
- z 网格、源网格和输出监视器网格；
- 常数材料模型和同一套色散材料模型；
- 波长采样、边界条件、仿真时间和自动关断条件。

若 10 nm 结果明显改变 `J` 或串扰排序，应先报告数值收敛问题，不能直接把 20 nm 结果称为物理极限。

## 交付时建议采用的最终措辞

可以使用：

> B_strict final 在当前固定 B 曲线骨架、108 格候选集合和已执行的 Wuyue KLayout Common V0.9.0 规则加 supplemental 120 nm Si 线宽/间距门禁下，得到 `J=0.886449`。其两个目标通道平均透射率为 91.529% 和 85.761%，由平均透射率计算的通道插损为 0.384 dB 和 0.667 dB。Stage 2 对指定的 119 个 4 连通相邻格对各尝试一次互补双格动作，49 个动作通过门禁并完成 FDTD，最大记录正差为 `8.1e-5`，低于 `1e-4` 接受阈值，因此没有改变 final 状态。

不要使用：

- “全局最优物理饱和上限”；
- “完整工业级 120 nm 规则全部通过”；
- “Foundry Ready”；
- “可直接投片”；
- “已证明任何后处理都不可行”。

## 审计产物

- [`strict_final_audit.json`](G:/codex_project/inverse_design_sun/analysis-recheck-20261013/strict_final_audit.json)：脚本输出的完整机器可读审计结果。
- [`audit_strict_final.py`](G:/codex_project/inverse_design_sun/analysis-recheck-20261013/audit_strict_final.py)：只读复核脚本。
- [`results/B_curve_DRC120_strict/final_spectra.npz`](G:/codex_project/inverse_design_sun/results/B_curve_DRC120_strict/final_spectra.npz)：严格版最终光谱。
- [`results/B_curve_DRC120_strict/final_DRC/result.json`](G:/codex_project/inverse_design_sun/results/B_curve_DRC120_strict/final_DRC/result.json)：严格版公共 DRC 结果。
- [`results/B_curve_stage2_joint/history.csv`](G:/codex_project/inverse_design_sun/results/B_curve_stage2_joint/history.csv)：Stage 2 的 119 对动作记录。

