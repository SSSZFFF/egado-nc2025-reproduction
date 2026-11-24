"""Deliver the verified curve -> DRC -> local grid geometry experiment."""

from pathlib import Path
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import klayout.db as db
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle, PathPatch, Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "curve_then_grid_trial"
META = json.loads((OUT / "grid_parameters.json").read_text(encoding="utf-8"))
OPT = json.loads((OUT / "optimization_parameters.json").read_text(encoding="utf-8"))
RULE_SHA = "DB18D3B7CD4FAB7D7CA42B4D1A8572E8F072306EE90F29758F61F3CB799AE797"


def load(path):
    l = db.Layout()
    l.read(str(path))
    r = db.Region(l.top_cell().begin_shapes_rec(l.layer(13, 1)))
    r.merge()
    return l, r


source_layout, source = load(META["source_gds"])
DBU = source_layout.dbu
UPNM = 1 / (DBU * 1000)


def cell_box(row, col):
    return db.Box(*[int(round(v * UPNM)) for v in
                    (-1800 + col * 120, -1800 + row * 120,
                     -1800 + (col + 1) * 120, -1800 + (row + 1) * 120)])


verified = {}
for name, target in (("sp90_90nm", 90), ("target_120nm", 120)):
    parameter = OPT[name]
    report = json.loads((OUT / f"check_{name}_optimized" / "result.json").read_text(encoding="utf-8"))
    assert report["rule_categories"] == 248 and report["violation_count"] == 0
    assert report["source_runset_sha256"] == RULE_SHA
    assert report["gds_sha256"] == parameter["gds_sha256"] == hashlib.sha256(Path(parameter["gds"]).read_bytes()).hexdigest().upper()
    l, r = load(parameter["gds"])
    cells = np.load(OUT / f"{name}_optimized_cells.npz", allow_pickle=False)
    initial = np.load(OUT / f"{name}_cells.npz", allow_pickle=False)["candidate_mask"]
    active, state = cells["candidate_mask"], cells["silicon_state"]
    threshold = int(round(target * UPNM))
    assert r.width_check(threshold, ignore_angle=85).count() == 0
    assert r.space_check(threshold, ignore_angle=85).count() == 0
    whole_mask = db.Region()
    records = []
    for row in range(29, -1, -1):
        for col in range(30):
            if not active[row, col]:
                continue
            b = cell_box(row, col)
            cell = db.Region(b)
            whole_mask.insert(b)
            assert (cell - r).is_empty() if state[row, col] else (cell & r).is_empty()
            records.append({
                "id": len(records) + 1, "row_y_up": row, "column_x_right": col,
                "x_centre_um": (-1800 + (col + .5) * 120) / 1000,
                "y_centre_um": (-1800 + (row + .5) * 120) / 1000,
                "material": "silicon" if state[row, col] else "oxide",
                "was_initial_drc_candidate": bool(initial[row, col]),
            })
    whole_mask.merge()
    outside = ((r ^ source) - whole_mask).area() * DBU ** 2
    assert outside < 1e-5
    for layer in ((13, 2), (251, 16)):
        original_layer = db.Region(source_layout.top_cell().begin_shapes_rec(source_layout.layer(*layer)))
        output_layer = db.Region(l.top_cell().begin_shapes_rec(l.layer(*layer)))
        assert (original_layer ^ output_layer).is_empty()
    port_component = {}
    for component, polygon in enumerate(r.each_merged()):
        part = db.Region(polygon)
        for label, (x, y) in {"IN_TOP": (-2795, 650), "IN_BOTTOM": (-2795, -650), "OUT": (2795, 1350)}.items():
            probe = db.Region(db.Box(int(round((x - 1) * UPNM)), int(round((y - 1) * UPNM)),
                                     int(round((x + 1) * UPNM)), int(round((y + 1) * UPNM))))
            if not (probe & part).is_empty():
                port_component[label] = component
    assert len(port_component) == 3
    with (OUT / f"{name}_final_cells.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    verified[name] = {
        "initial_candidates": int(initial.sum()), "final_cells": int(active.sum()),
        "silicon_cells": int(state[active].sum()), "oxide_cells": int((~state[active]).sum()),
        "retained_initial_candidates": int((active & initial).sum()),
        "restored_initial_candidates": int((initial & ~active).sum()),
        "added_interface_cells": int((active & ~initial).sum()),
        "curve_region_retained_percent_of_design": 100 * (900 - int(active.sum())) / 900,
        "changed_area_percent_of_design_from_curve": parameter["changed_area_percent_of_design"],
        "official_sp90_categories": 248, "official_sp90_violations": 0,
        "supplemental_width_spacing_nm": target, "supplemental_width": 0, "supplemental_space": 0,
        "outside_active_difference_um2": outside,
        "selected_cells_homogeneous": True, "cld_and_dev_unchanged": True,
        "port_component_ids": port_component,
        "all_three_ports_on_same_silicon_component": len(set(port_component.values())) == 1,
        "gds": parameter["gds"], "gds_sha256": report["gds_sha256"],
        "official_report": report["report"], "rule_sha256": report["source_runset_sha256"],
        "optical_simulations": 0,
    }

font = Path(r"C:\Windows\Fonts\msyh.ttc")
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                     "axes.unicode_minus": False, "font.size": 10, "svg.fonttype": "path"})


def draw(ax, region, dbu, alpha=1.0):
    for polygon in region.each_merged():
        rings = [list(polygon.each_point_hull())]
        rings += [list(polygon.each_point_hole(i)) for i in range(polygon.holes())]
        vertices, codes = [], []
        for i, ring in enumerate(rings):
            xy = np.array([(v.x * dbu, v.y * dbu) for v in ring])
            signed = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
            if (i == 0 and signed < 0) or (i > 0 and signed > 0):
                xy = xy[::-1]
            vertices.extend(xy.tolist() + [xy[0].tolist()])
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 1) + [MplPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor="#254f69", edgecolor="none", alpha=alpha))


def grid(ax, mask, fill=None):
    for row, col in np.argwhere(mask):
        x, y = -1.8 + col * .12, -1.8 + row * .12
        face = "none" if fill is None else ("#254f69" if fill[row, col] else "#eef1f4")
        ax.add_patch(Rectangle((x, y), .12, .12, facecolor=face,
                               edgecolor="#d58d20", lw=0.75, zorder=3))


for name in ("sp90_90nm", "target_120nm"):
    v = verified[name]
    l, r = load(v["gds"])
    initial = np.load(OUT / f"{name}_cells.npz", allow_pickle=False)["candidate_mask"]
    final_arrays = np.load(OUT / f"{name}_optimized_cells.npz", allow_pickle=False)
    active, state = final_arrays["candidate_mask"], final_arrays["silicon_state"]
    target = v["supplemental_width_spacing_nm"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.3))
    draw(axes[0, 0], source, DBU)
    draw(axes[0, 1], source, DBU, alpha=.22)
    grid(axes[0, 1], initial)
    draw(axes[1, 0], r, l.dbu)
    grid(axes[1, 0], active)
    grid(axes[1, 1], active, state)
    titles = ("① 原始 TO 提取的连续曲线", f"② 曲线后 DRC｜初始 {v['initial_candidates']} 个候选格",
              f"③ 填格与接缝判优后｜最终 {v['final_cells']} 个方格", "④ 单独显示最终方格｜蓝：硅；浅灰：氧化物")
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.set_xlabel("x（µm）")
        ax.set_ylabel("y（µm）")
        ax.set_xticks([-1.8, -.9, 0, .9, 1.8])
        ax.set_yticks([-1.8, -.9, 0, .9, 1.8])
    fig.suptitle(f"曲线 → DRC → 局部 120 nm 方格｜目标线宽／间距 {target} nm", fontsize=16, y=.98)
    fig.text(.5, .035,
             f"保留曲线版原状的设计区域 {v['curve_region_retained_percent_of_design']:.1f}%  ·  相对曲线的实际材料变化面积 {v['changed_area_percent_of_design_from_curve']:.2f}%\n"
             f"SP90 公共 DRC：248 类，0 违规；额外 {target} nm 线宽／间距检查：0 标记。仅几何判优，未做光学仿真。",
             ha="center", va="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=.08, right=.965, top=.91, bottom=.105, hspace=.30, wspace=.22)
    fig.savefig(OUT / f"{name}_comparison.png", dpi=185, facecolor="white")
    fig.savefig(OUT / f"{name}_comparison.svg", facecolor="white")
    plt.close(fig)

(OUT / "verification.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
md = f"""# 曲线之后 DRC，再局部填为 120 nm 方格

本次从原始 TO 的连续曲线提取结果 original_curve_sigma20.gds 出发，避免混入此前 74 格局部粗化。
这条曲线是在原始 TO 灰度上经过 σ=20 nm 平滑和双三次等值线提取的结果，平滑本身包含小幅几何变化。

## 流程与结果

1. 在连续曲线上检查制造规则，再将报错覆盖范围映射为 120 nm 格子。
2. 初始按每格硅面积占比选择整格硅／氧化物，保留格外的 vector 曲线。
3. 重检曲线与方格接缝，对两态进行几何判优；必要时扩大局部处理区。
4. 通过后恢复可以原样保留的曲线区域，再运行实际 skill 的 248 类公共 DRC。

| 选区标准 | 初始候选 | 最终方格 | 保留曲线版原状区域 | 实际材料变化／设计面积 | 实际 SP90 公共 DRC |
|---|---:|---:|---:|---:|---|
| SP90 90 nm 规则 | {verified['sp90_90nm']['initial_candidates']} | {verified['sp90_90nm']['final_cells']} | {verified['sp90_90nm']['curve_region_retained_percent_of_design']:.2f}% | {verified['sp90_90nm']['changed_area_percent_of_design_from_curve']:.2f}% | 248 类，0 违规 |
| 额外 120 nm 线宽／间距 | {verified['target_120nm']['initial_candidates']} | {verified['target_120nm']['final_cells']} | {verified['target_120nm']['curve_region_retained_percent_of_design']:.2f}% | {verified['target_120nm']['changed_area_percent_of_design_from_curve']:.2f}% | 248 类，0 违规 |

120 nm 目标版还单独通过了 angle exclusion=85° 的 120 nm Si 线宽与间距检查，均为 0 标记。
该额外检查不等同于一套新的完整 120 nm foundry deck，也不代表全部手册规则。

直接按面积多数一次填格仍产生 76／77 项 SP90 报错，主要在方格与曲线接缝。
本次进行了受限的几何两态搜索，并将小缺口、角度和小面积规则也纳入接缝判优，
最终结果由重新运行的实际公共宏确认；没有关闭规则、修改官方 PDK 或删去加工层。

## 保留范围与验证

目标 120 nm 版最终替换 {verified['target_120nm']['final_cells']} / 900 个格子。
初始候选中保留为方格 {verified['target_120nm']['retained_initial_candidates']} 个，
恢复曲线 {verified['target_120nm']['restored_initial_candidates']} 个；另有
{verified['target_120nm']['added_interface_cells']} 个接缝邻域格子加入修复。
这些计数是空间格子数量，不是 DRC 边对标记数。

每个最终格子已检查为单一材料，GDS 回读后再次验证；格外曲线保持原样，
只有 0.1 nm 数据库布尔／序列化量化范围内的面积差。窗口、器件外框逐层检查未变化。
光学端口的 Si 连通分量记录在 verification.json；几何连通不证明模式转换性能。

所用 drawn layer 为 13/1 WGF_COR、13/2 WGF_CLD、251/16 DEV_AREA。
两份优化版输入 SHA 与各自的新 DRC 报告一致，规则 SHA 为 {RULE_SHA}。
文件 *_filled.gds 是初始填格版，*_optimized.gds 是几何判优版。
最终坐标与材料见 *_final_cells.csv；布尔掩码与状态见 *_optimized_cells.npz。

这次材料选择仅使用制造约束和相对原曲线的几何变化面积，未使用光学 FoM，未运行 FDTD。
当前只能确认几何流程可行，插损、串扰和带宽是否保持仍需独立仿真。
"""
(OUT / "曲线后DRC与局部填格说明.md").write_text(md, encoding="utf-8")
print(json.dumps(verified, ensure_ascii=False, indent=2))
