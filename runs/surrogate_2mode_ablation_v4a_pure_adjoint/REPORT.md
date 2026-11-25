# 2-mode surrogate v4a — pure refreshed adjoint control

Run: `F:\codex_project\inverse_design_sun\runs\surrogate_2mode_ablation_v4a_pure_adjoint`

## 设置

- `--pure-adjoint`：关闭 Ridge，直接按刷新后的伴随一阶 prior 排序
- `--top-k 1`，`--max-rounds 120`，`--max-fdtd 110`
- `--enable-refresh`，每 5 次接受翻转或连续 4 轮无接受刷新一次
- 初始先验：`adjoint_init_2mode.npz`（4 solves）
- 远程无 torch，`compact_features=true`，但 pure adjoint 模式不使用特征模型

## 结果

| 指标 | v4a pure adjoint | B_strict baseline |
|---|---:|---:|
| 初始 J | 0.849817 | 0.849817 |
| 最终 J | **0.861105** | **0.886449** |
| 增益 | **+0.011288** | +0.036632 |
| 接受翻转 | **10** | 19 |
| FDTD 状态评估 | **110**（220 solves） | 212（424 solves） |
| 伴随刷新 | **22 次 ×4 = 88 solves** | — |
| 总 solves | **308** | 424 |
| 每 solve 命中率 | 4.55% | 4.48% |
| 墙钟时间 | **10,750.85 s ≈ 2h59m** | 18,894.4 s ≈ 5h15m |
| 达到 baseline 增益比例 | **30.8%** | 100% |

## 关键发现：pure adjoint 在初始态根本排不到 `(23,21)`

- `(23,21)` 在初始态的 `linear_prior = grad_eps * delta_si = -1.0986`，在 108 个 candidate 中排名 **第 99 位**。
- pure adjoint 的 Top-1 从初始态选出 `(15,13)`（prior +1.0889，实测 `-0.012714`）；后续又连续选中 `(12,5)`、`(16,15)` 等，整体没有找到 baseline 的大跃迁 `(23,21)`。
- 全部 110 个评估状态中，`(23,21)` **一次都没有被评估**。

这说明“纯伴随刷新”不能解决 `(23,21)` 的问题：初始伴随梯度对它的符号/排序就是错的；它在 baseline 中的价值是在中间状态才出现的，单靠初始 adjoint 的局部一阶 prior 不会主动探索到它。

## 结论

v4a 纯伴随对照组失败：

- 最终 0.861105，仅比 baseline 初始高 +0.0113；
- 用了 220 FDTD solves + 88 伴随 solves，仍远低于 baseline 0.886449；
- 22 次刷新没有错误，但纯伴随排序始终围绕 `(15,13)`、`(12,5)`、`(16,15)` 等错误像素循环，说明当前 main issue 不是“刷新次数”，而是**一阶伴随在离散大幅翻转上的 ranking 本身不可靠**。
- 因此不建议直接跑 v4b 的 6 维代理作为“修好 pure adjoint”的延伸；6 维特征可能能改善 Ridge，但 `(23,21)` 的初始 prior 排名 99/108，仍是需要显式探索/更长程搜索的问题。

## 下一步建议

- 若继续 v4b，建议给 `(23,21)` 这类“低初始 prior、高潜在增益”的像素加入**显式探索项**（例如按距离/邻域或二阶近似单独打分），而不是纯按一阶 prior 排序。
- 或者把 baseline 已经找到的 `(23,21)` 作为逐像素消融基准，单独测试“在该状态附近刷新伴随后它的排名是否上升”，先验证刷新对关键像素的实际效果。
- 不建议继续用 pure adjoint 的 110 states 预算重复跑；这次 22 次刷新已给出足够负结果。

## 文件

- `adjoint_init_2mode.npz`
- `surrogate_history.csv`
- `summary.json`
- `final_state.npz`
- `convergence.png`
- `stdout.log` / `stderr.log`
