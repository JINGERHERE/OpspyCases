#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：SectionHub.py
@Date    ：2025/8/1 19:17
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from typing import Literal, TypeAlias, Union, Callable, cast
from pathlib import Path

import openseespy.opensees as ops
import opstool as opst

import ops_utilities as opsu
from ops_utilities.pre import ConcHub, ReBarHub, Mander
from ModelUtilities import UNIT, OPSE


"""
# --------------------------------------------------
# ========== < SectionHub > ==========
# --------------------------------------------------
"""


class SectionHub:

    @staticmethod
    def bent_cap(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal[""]] = "",
    ) -> opst.pre.section.FiberSecMesh:
        """
        钢筋混凝土双柱式桥墩 `盖梁` 截面。

        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            info (bool, optional): 是否显示截面信息。默认值为 True。
            save_sec (Union[Path, str, Literal['']], optional): 是否保存截面网格模型。默认值为 ''。

        Returns:
            opst.pre.section.FiberSecMesh: 包含截面网格模型的对象。
        """

        # 获取函数名
        if sys._getframe() is not None:
            sec_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料编号
        cover_tag = manager.next_tag(
            category="uniaxialMaterial", label=f"{sec_name}_cover"
        )
        core_tag = manager.next_tag(
            category="uniaxialMaterial", label=f"{sec_name}_core"
        )
        rebar_tag = manager.next_tag(
            category="uniaxialMaterial", label=f"{sec_name}_rebar"
        )  # 标签标记为绘图所需
        # 定义截面编号
        sec_tag = manager.next_tag(category="section", label=sec_name)  # 定义截面编号

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面控制
        W = 0.55 * UNIT.m
        H = 0.55 * UNIT.m
        cover = 20 * UNIT.mm  # 保护层厚度
        bar_dia = 12 * UNIT.mm  # 钢筋直径
        bar_n = 30  # 纵筋个数

        # 材料控制
        fc = 32.8 * UNIT .mpa # 混凝土抗压强度 /32.8
        Ec = 29.3 * UNIT.gpa # 混凝土弹性模量 /29.3

        fy = 526.2 * UNIT.mpa # 钢筋屈服强度
        Es = 190 * UNIT.gpa # 钢筋弹性模量

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 轮廓线
        sec_outlines = [[0, 0], [W, 0], [W, H], [0, H]]
        cover_outlines = opst.pre.section.offset(sec_outlines, d=cover)  # 保护层线

        # 生成几何形状
        cover_geo = opst.pre.section.create_polygon_patch(
            outline=sec_outlines, holes=[cover_outlines]
        )
        core_geo = opst.pre.section.create_polygon_patch(outline=cover_outlines)

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name="RC Pier Bent Cap Section")
        SEC.add_patch_group({"Cover": cover_geo, "Core": core_geo})
        SEC.set_mesh_size({"Cover": 0.05, "Core": 0.05})
        SEC.set_mesh_color({"Cover": "#ffa756", "Core": "#40a368"})
        SEC.set_ops_mat_tag({"Cover": cover_tag, "Core": core_tag})  # 明确材料编号

        # 底层添加钢筋
        rebars_outer = opst.pre.section.offset(cover_outlines, bar_dia / 2)
        SEC.add_rebar_line(
            points=rebars_outer,
            dia=bar_dia,
            n=bar_n,
            ops_mat_tag=rebar_tag,
            color="red",
        )

        # 生成网格
        SEC.mesh()
        SEC.centring()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info)  # 截面信息

        # 素混凝土
        fc, Ec = 32.8 * UNIT.mpa, 29.3 * UNIT.gpa
        ec, ecu = 0.002, 0.0033
        # 核心混凝土
        Acor = (
            (W - 2 * (bar_dia + cover)) * (H - 2 * (bar_dia + cover)) * (160 * UNIT.mm)
        )  # 核心面积 x 箍筋间距
        As = np.pi * (6 / 2 * UNIT.mm) ** 2  # 箍筋单肢面积
        fcc, ecc, eccu = Mander.rectangular(
            lx=W,
            ly=H,
            coverThick=cover,
            fco=fc,
            fyh=487.8 * UNIT.mpa,  # 箍筋屈服强度(MPa)
            roucc=sec_props["rho_rebar"],
            roux=(2 * As * (W - 2 * (cover + bar_dia)))
            / Acor,  # x方向的体积配箍率, 计算时只计入约束混凝土面积
            rouy=(6 * As * (H - 2 * (cover + bar_dia)))
            / Acor,  # y方向的体积配箍率, 计算时只计入约束混凝土面积
            sl=((W + H) * 2 - 8 * cover) / (bar_n - 1),  # 纵筋间距
            dsl=bar_dia,
            st=160 * UNIT.mm,  # 箍筋间距
            dst=6 * UNIT.mm,  # 箍筋直径
        )
        # 钢筋
        fy, Es = 526.2 * UNIT.mpa, 190 * UNIT.gpa

        # ops 材料参数
        cover_params = dict(fc=-fc, ec=-ec, ecu=-ecu, Ec=Ec)
        core_params = dict(fc=-fcc, ec=-ecc, ecu=-eccu, Ec=Ec)
        rebar_params = dict(fy=fy, Es=Es, b=0.01, R0=18, cR1=0.925, cR2=0.15)

        # 截面轴压力
        sec_props["P"] = 0.1 * (sec_props["A"] * abs(fc))

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料
        ops.uniaxialMaterial("Concrete04 ", cover_tag, *cover_params.values())
        ops.uniaxialMaterial("Concrete04 ", core_tag, *core_params.values())
        ops.uniaxialMaterial("Steel02 ", rebar_tag, *rebar_params.values())

        # 定义截面
        SEC.to_opspy_cmds(
            secTag=sec_tag, GJ=sec_props["J"] * ConcHub.get_G("C25") * UNIT.mpa
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props["strain_stages"] = {
            rebar_tag: (fy / Es, 0.015, 0.055, 0.1),
            core_tag: (-fy / Es, -ecu, 0.75 * -eccu, -eccu),
        }
        # 截面 应变阈值 用于：opst.pre.section.FiberSecMesh.plot_response(thresholds=)
        sec_props["strain_thresholds"] = {
            rebar_tag: (-0.1, 0.1),  # 只能两个值
            cover_tag: (-ecu, 0.0),
            core_tag: (-eccu, 0.0),
        }
        # 截面几何参数
        sec_props["Width"] = W
        sec_props["Height"] = H
        # 储存至管理器
        manager.set_params(
            category="uniaxialMaterial", tag=cover_tag, params=cover_params
        )  # 保护层
        manager.set_params(
            category="uniaxialMaterial", tag=core_tag, params=core_params
        )  # 核心
        manager.set_params(
            category="uniaxialMaterial", tag=rebar_tag, params=rebar_params
        )  # 钢筋
        manager.set_params(category="section", tag=sec_tag, params=sec_props)  # 截面

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化 截面
        if save_sec:
            plt.close("all")
            SEC.view(fill=True, show_legend=True)
            plt.savefig(f"{save_sec}/{sec_name}_mash.png", dpi=300, bbox_inches="tight")
            plt.close("all")

        return SEC

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def pier_col(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal[""]] = "",
    ) -> opst.pre.section.FiberSecMesh:
        """
        钢筋混凝土双柱式桥墩 `墩柱` 截面。

        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            info (bool, optional): 是否显示截面信息。默认值为 True。
            save_sec (Union[Path, str, Literal['']], optional): 是否保存截面网格模型。默认值为 ''。

        Returns:
            opst.pre.section.FiberSecMesh: 包含截面网格模型的对象。
        """

        # 获取函数名
        if sys._getframe() is not None:
            sec_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料编号
        cover_tag = manager.next_tag(
            category="uniaxialMaterial", label=f"{sec_name}_cover"
        )
        core_tag = manager.next_tag(
            category="uniaxialMaterial", label=f"{sec_name}_core"
        )
        rebar_tag = manager.next_tag(
            category="uniaxialMaterial", label=f"{sec_name}_rebar"
        )  # 标签标记为绘图所需
        # 定义截面编号
        sec_tag = manager.next_tag(category="section", label=sec_name)  # 定义截面编号

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面控制
        R = 0.2 * UNIT.m
        cover = 20 * UNIT.mm  # 保护层厚度
        bar_dia = 8 * UNIT.mm  # 钢筋直径
        bar_n = 16  # 纵筋个数

        # 轮廓线
        cover_outline = opst.pre.section.create_circle_points(xo=[0, 0], radius=R - cover, n_sub=30) # 保护层轮廓线
        # 生成网格形状
        cover_geo = opst.pre.section.create_circle_patch(xo=[0, 0], radius=R, holes=[cover_outline], n_sub=30) # 保护层几何形状
        core_geo = opst.pre.section.create_circle_patch(xo=[0, 0], radius=R - cover, n_sub=30) # 核心几何形状

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name="RC Pier Bent Cap Section")
        SEC.add_patch_group({"Cover": cover_geo, "Core": core_geo})
        SEC.set_mesh_size({"Cover": 0.05, "Core": 0.05})
        SEC.set_mesh_color({"Cover": "#ffa756", "Core": "#40a368"})
        SEC.set_ops_mat_tag({"Cover": cover_tag, "Core": core_tag})  # 明确材料编号

        # 添加钢筋
        SEC.add_rebar_circle(
            xo=[0, 0],
            radius=R - cover - bar_dia / 2.0,
            dia=bar_dia,
            n=bar_n + 1, # 计数起点为 0
            ops_mat_tag=rebar_tag,
            color="red",
        )

        # 生成网格
        SEC.mesh()
        SEC.centring()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info)  # 截面信息

        # 素混凝土
        fc, Ec = 25. * UNIT.mpa, 29.3 * UNIT.gpa
        ec, ecu = 0.002, 0.0033
        # 核心混凝土
        fcc, ecc, eccu = Mander.circular(
            hoop="Circular",  # 箍筋类型，Circular圆形箍筋，Spiral螺旋形箍筋
            fco=fc,
            d=R * 2.0,  # 截面直径
            coverThick=cover,  # 保护层厚度
            roucc=sec_props["rho_rebar"],  # 纵筋配筋率, 计算时只计入约束混凝土面积
            s=85 * UNIT.mm,  # 箍筋纵向间距（螺距）
            ds=6 * UNIT.mm,  # 箍筋直径
            fyh=487.8 * UNIT.mpa,  # 箍筋屈服强度(MPa)
        )
        # 钢筋
        fy, Es = 400.2 * UNIT.mpa, 190.0 * UNIT.gpa

        # ops 材料参数
        cover_params = dict(fc=-fc, ec=-ec, ecu=-ecu, Ec=Ec)
        core_params = dict(fc=-fcc, ec=-ecc, ecu=-eccu, Ec=Ec)
        rebar_params = dict(fy=fy, Es=Es, b=0.01, R0=18, cR1=0.925, cR2=0.15)

        # 截面轴压力
        sec_props["P"] = 0.1 * (sec_props["A"] * abs(fc))

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料
        ops.uniaxialMaterial("Concrete04 ", cover_tag, *cover_params.values())
        ops.uniaxialMaterial("Concrete04 ", core_tag, *core_params.values())
        ops.uniaxialMaterial("Steel02 ", rebar_tag, *rebar_params.values())

        # 定义截面
        SEC.to_opspy_cmds(
            secTag=sec_tag, GJ=sec_props["J"] * ConcHub.get_G("C25") * UNIT.mpa
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props["strain_stages"] = {
            rebar_tag: (fy / Es, 0.015, 0.055, 0.1),
            core_tag: (-fy / Es, -ecu, 0.75 * -eccu, -eccu),
        }
        # 截面 应变阈值 用于：opst.pre.section.FiberSecMesh.plot_response(thresholds=)
        sec_props["strain_thresholds"] = {
            rebar_tag: (-0.1, 0.1),  # 只能两个值
            cover_tag: (-ecu, 0.0),
            core_tag: (-eccu, 0.0),
        }
        # 截面几何参数
        sec_props["Radius"] = R
        # 储存至管理器
        manager.set_params(
            category="uniaxialMaterial", tag=cover_tag, params=cover_params
        )  # 保护层
        manager.set_params(
            category="uniaxialMaterial", tag=core_tag, params=core_params
        )  # 核心
        manager.set_params(
            category="uniaxialMaterial", tag=rebar_tag, params=rebar_params
        )  # 钢筋
        manager.set_params(category="section", tag=sec_tag, params=sec_props)  # 截面

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化 截面
        if save_sec:
            plt.close("all")
            SEC.view(fill=True, show_legend=True)
            plt.savefig(f"{save_sec}/{sec_name}_mash.png", dpi=300, bbox_inches="tight")
            plt.close("all")

        return SEC


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":

    # 模型管理器
    from ModelUtilities import MM

    # 数据路径
    out_path = Path("./OutData")
    out_path.mkdir(parents=True, exist_ok=True)

    # Model
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # # 截面
    # SectionHub.bent_cap(manager=MM, save_sec=out_path)
    # SectionHub.pier_col(manager=MM, save_sec=out_path)
    # SectionHub.contact_surf(manager=MM, save_sec=out_path)

    # 批量调用
    callables = opsu.get_callables(SectionHub)
    for name, sec_fun in callables.items():
        sec_fun(manager=MM, save_sec=out_path)

    # 导出模型管理器
    MM.to_excel(Path(out_path) / "ModelManager.xlsx")
