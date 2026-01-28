#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Part_MatSec_MomentCurvature.py
@Date    ：2025/8/1 19:16
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from typing import Literal, TypeAlias, Union, Callable, cast
from pathlib import Path

import openseespy.opensees as ops
import opstool as opst
from sectionproperties.pre.library import steel_sections

import ops_utilities as opsu
from ops_utilities.pre import ConcHub, ReBarHub, Mander
from ModelUtilities import UNIT

"""
# --------------------------------------------------
# ========== < Part_MatSec_MomentCurvature > ==========
# --------------------------------------------------
"""

class SectionHub:
    
    """
    使用静态方法定义截面：
        - sec_I: 工型钢截面
        - sec_rect: 矩形截面
        - sec_polygonal: 多边形截面
        - sec_circle: 圆截面
    """

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def sec_I(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal['']] = ''
        ) -> opst.pre.section.FiberSecMesh:
        
        """
        构建一个工型钢纤维截面网格模型。
        
        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            info (bool, optional): 是否显示截面信息。默认值为 True。
            save_sec (Union[Path, str, Literal['']], optional): 是否保存截面可视化图片。默认值为 ''。
        
        Returns:
            opst.pre.section.FiberSecMesh: 包含截面网格模型的对象。
        """
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            sec_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义编号
        steel_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_steel') # 定义材料编号
        sec_tag = manager.next_tag(category="section", label=sec_name) # 定义截面编号
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 工型钢形状
        geom_half = steel_sections.tapered_flange_channel(
            d=20 * UNIT.cm, b=7 * UNIT.cm,
            t_f=1.15 * UNIT.cm, t_w=0.95 * UNIT.cm,
            r_r=1.15 * UNIT.cm, r_f=0.8 * UNIT.cm,
            alpha=8,
            n_r=16,
            )
        I_geom = geom_half + geom_half.mirror_section(axis="y", mirror_point=(0, 0)) # 镜像叠加
    
        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name="I Section")
        SEC.add_patch_group({"I_sec": I_geom})
        SEC.set_mesh_size({"I_sec": 0.3})
        SEC.set_ops_mat_tag({"I_sec": steel_tag}) # 明确材料编号

        # 生成网格
        SEC.mesh()
        SEC.centring()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info) # 截面信息

        # ops 材料参数
        fy, Es = 400. * UNIT.mpa, 200. * UNIT.gpa
        mat_params = dict(fy=fy, Es=Es, b=0.01, R0=18, cR1=0.925, cR2=0.15)
        
        # 截面轴压力
        sec_props['P'] = 0.1 * (sec_props['A'] * abs(mat_params['fy']))
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料
        ops.uniaxialMaterial("Steel02", steel_tag, *mat_params.values())
        
        # 定义截面
        SEC.to_opspy_cmds(secTag=sec_tag, G=100. * UNIT.gpa)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props['strain_stages'] = {
            steel_tag: [fy / Es, 0.015, 0.055, 0.1],
            # core_tag: [fy / Es, ecu, 0.75 * eccu, eccu],
            }
        # 储存至管理器
        manager.set_params(category="uniaxialMaterial", tag=steel_tag, params=mat_params) # 材料
        manager.set_params(category="section", tag=sec_tag, params=sec_props) # 截面
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化 截面
        if save_sec:
            plt.close('all')
            SEC.view(fill=True, show_legend=True)
            plt.savefig(f'{save_sec}/{sec_name}_mash.png', dpi=300, bbox_inches='tight')
            plt.close('all')
    
        return SEC


    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def sec_rect(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal['']] = ''
        ) -> opst.pre.section.FiberSecMesh:

        """
        构建一个矩形截面的纤维截面网格模型。

        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            info (bool, optional): 是否显示截面信息。默认值为 True。
            save_sec (Union[Path, str, Literal['']], optional): 是否保存截面可视化图片。默认值为 ''。

        Returns:
            opst.pre.section.FiberSecMesh: 包含截面网格模型的对象。
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            sec_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 内部材料标签
        inner_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_inner') # 定义材料编号
        # 截面标签
        sec_tag = manager.next_tag(category="section", label=sec_name) # 定义截面编号
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 形状参数
        W: float = 65. * UNIT.mm
        H: float = 99.157 * UNIT.mm

        # 轮廓线
        sec_outlines = [(0, 0), (W, 0), (W, H), (0, H)]
        # 生成几何形状
        inner_geo = opst.pre.section.create_polygon_patch(sec_outlines)

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name="Simple Rectangular Section")
        SEC.add_patch_group({"inner": inner_geo})
        SEC.set_mesh_size({"inner": 0.005})
        SEC.set_ops_mat_tag({"inner": inner_tag}) # 明确材料编号

        # 生成网格
        SEC.mesh()
        SEC.centring()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info) # 截面信息
        
        # 材料类型（金属）
        steel_type = "HRB500"
        fy, Es = ReBarHub.get_fyk(steel_type) * UNIT.mpa, ReBarHub.get_Es(steel_type) * UNIT.mpa

        # ops 材料参数
        steel_params = dict(fy=fy, Es=Es, b=0.01, R0=18, cR1=0.925, cR2=0.15)

        # 截面轴压力
        sec_props['P'] = 0.1 * (sec_props['A'] * abs(fy))
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料
        ops.uniaxialMaterial("Steel02", inner_tag, *steel_params.values())

        # 定义截面
        SEC.to_opspy_cmds(secTag=sec_tag, G=100. * UNIT.gpa)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 旋转截面
        SEC.rotate(0, remesh=True)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props['strain_stages'] = {
            inner_tag: [fy / Es, 0.015, 0.055, 0.1],
            # core_tag: [fy / Es, ecu, 0.75 * eccu, eccu],
            }
        # 储存至管理器
        manager.set_params(category="uniaxialMaterial", tag=inner_tag, params=steel_params) # 材料
        manager.set_params(category="section", tag=sec_tag, params=sec_props) # 截面

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化 截面
        if save_sec:
            plt.close('all')
            SEC.view(fill=True, show_legend=True)
            plt.savefig(f'{save_sec}/{sec_name}_mash.png', dpi=300, bbox_inches='tight')
            plt.close('all')

        return SEC


    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def sec_polygonal(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal['']] = ''
        ) -> opst.pre.section.FiberSecMesh:

        """
        构建一个兴宁桥盖梁纤维截面网格模型。

        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            info (bool, optional): 是否显示截面信息。默认值为 True。
            view (bool, optional): 是否可视化截面。默认值为 True。

        Returns:
            opst.pre.section.FiberSecMesh: 包含截面网格模型的对象。
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            sec_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料编号
        cover_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_cover')
        core_tag =manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_core')
        rebar_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_rebar')
        # 定义截面编号
        sec_tag = manager.next_tag(category="section", label=sec_name) # 定义截面编号
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 形状参数
        bar_dia: float = 22. * UNIT.mm
        cover: float = 3. * UNIT.cm
        H1: float = 600. * UNIT.mm
        H2: float = 750. * UNIT.mm
        H3: float = 1350. * UNIT.mm
        W1: float = 500.* UNIT.mm
        W2: float = 1100. * UNIT.mm

        # 轮廓线
        sec_outlines = [
            (W1 / 2., 0.),
            (W1 / 2., H1),
            (W2 / 2., H2),
            (W2 / 2., H3),

            (-W2 / 2., H3),
            (-W2 / 2., H2),
            (-W1 / 2., H1),
            (-W1 / 2., 0.),
            ]
        cover_outlines = opst.pre.section.offset(sec_outlines, d=cover) # 保护层线

        # 生成几何形状
        cover_geo = opst.pre.section.create_polygon_patch(outline=sec_outlines, holes=[cover_outlines])
        core_geo = opst.pre.section.create_polygon_patch(outline=cover_outlines)

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name='XingNing Pier Section')
        SEC.add_patch_group({"Cover": cover_geo, "Core": core_geo})
        SEC.set_mesh_size({"Cover": 0.1, "Core": 0.1})
        SEC.set_mesh_color({"Cover": "#ffa756", "Core": "#40a368"})
        SEC.set_ops_mat_tag({"Cover": cover_tag, "Core": core_tag}) # 明确材料编号

        # 底层添加钢筋
        SEC.add_rebar_line(
            points=[
                (-(W1 / 2. - cover - bar_dia / 2.), cover + bar_dia / 2.),
                (W1 / 2. - cover - bar_dia / 2., cover + bar_dia / 2.)
                ],
            dia=bar_dia, n=5,
            ops_mat_tag=rebar_tag, color='red'
            )
        SEC.add_rebar_line(
            points=[
                (-(W1 / 2. - cover - bar_dia / 2.), cover + bar_dia * 2.5),
                (W1 / 2. - cover - bar_dia / 2., cover + bar_dia * 2.5)
                ],
            dia=bar_dia, n=5,
            ops_mat_tag=rebar_tag, color='red'
            )
        # 添加顶层钢筋
        SEC.add_rebar_line(
            points=[
                (-(W2 / 2. - cover - bar_dia / 2.), H3 - cover - bar_dia / 2.),
                (W2 / 2. - cover - bar_dia / 2., H3 - cover - bar_dia / 2.)
                ],
            dia=bar_dia, n=8,
            ops_mat_tag=rebar_tag, color='red'
            )
        SEC.add_rebar_line(
            points=[
                (-(W2 / 2. - cover - bar_dia / 2.), H3 - cover - bar_dia * 2.5),
                (W2 / 2. - cover - bar_dia / 2., H3 - cover - bar_dia * 2.5)
                ],
            dia=bar_dia, n=8,
            ops_mat_tag=rebar_tag, color='red'
            )
        # 添加中层钢筋(右侧)
        SEC.add_rebar_line(
            points=[
                (W2 / 2. - cover - bar_dia / 2., H3 - bar_dia * 10.),
                (W2 / 2. - cover - bar_dia / 2., H2 + bar_dia),
                (W1 / 2. - cover - bar_dia / 2., H1 + bar_dia),
                (W1 / 2. - cover - bar_dia / 2., 0. + bar_dia * 10.)
                ],
            dia=bar_dia, n=8,
            ops_mat_tag=rebar_tag, color='red'
            )
        # 添加中层钢筋(左侧)
        SEC.add_rebar_line(
            points=[
                (-(W2 / 2. - cover - bar_dia / 2.), H3 - bar_dia * 10.),
                (-(W2 / 2. - cover - bar_dia / 2.), H2 + bar_dia),
                (-(W1 / 2. - cover - bar_dia / 2.), H1 + bar_dia),
                (-(W1 / 2. - cover - bar_dia / 2.), 0. + bar_dia * 10.)
                ],
            dia=bar_dia, n=8,
            ops_mat_tag=rebar_tag, color='red'
            )

        # 生成网格
        SEC.mesh()
        SEC.centring()
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info) # 截面信息
        
        # 材料类型
        concrete_type = "C40"
        rebar_type = "HRB400"
        # 素混凝土
        fc, Ec = ConcHub.get_fcuk(concrete_type) * UNIT.mpa, ConcHub.get_Ec(concrete_type) * UNIT.mpa
        ec, ecu = 0.002, 0.0033
        # 核心混凝土
        fcc, ecc, eccu = Mander.rectangular(
            lx=(W1 + W2) / 2., ly=H3, coverThick=cover,
            fco=fc, dsl=bar_dia, roucc=sec_props['rho_rebar'],
            st=25 * UNIT.cm, fyh=500. * UNIT.mpa,
            )
        # 钢筋
        fy, Es = ReBarHub.get_fyk(rebar_type) * UNIT.mpa, ReBarHub.get_Es(rebar_type) * UNIT.mpa
        
        # ops 材料参数
        cover_params = dict(fc=-fc, ec=-ec, ecu=-ecu, Ec=Ec)
        core_params = dict(fc=-fcc, ec=-ecc, ecu=-eccu, Ec=Ec)
        rebar_params = dict(fy=fy, Es=Es, b=0.01, R0=18, cR1=0.925, cR2=0.15)
        
        # 截面轴压力
        sec_props['P'] = 0.1 * (sec_props['A'] * abs(fc))

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料
        ops.uniaxialMaterial("Concrete04 ", cover_tag, *cover_params.values())
        ops.uniaxialMaterial("Concrete04 ", core_tag, *core_params.values())
        ops.uniaxialMaterial("Steel02 ", rebar_tag, *rebar_params.values())

        # 定义截面
        SEC.to_opspy_cmds(
            secTag=sec_tag,
            GJ=sec_props['J'] * ConcHub.get_G(concrete_type) * UNIT.mpa
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props['strain_stages'] = {
            rebar_tag: [fy / Es, 0.015, 0.055, 0.1],
            core_tag: [fy / Es, ecu, 0.75 * eccu, eccu],
            }
        # 储存至管理器
        manager.set_params(category="uniaxialMaterial", tag=cover_tag, params=cover_params) # 保护层
        manager.set_params(category="uniaxialMaterial", tag=core_tag, params=core_params) # 核心
        manager.set_params(category="uniaxialMaterial", tag=rebar_tag, params=rebar_params) # 钢筋
        manager.set_params(category="section", tag=sec_tag, params=sec_props) # 截面

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化 截面
        if save_sec:
            plt.close('all')
            SEC.view(fill=True, show_legend=True)
            plt.savefig(f'{save_sec}/{sec_name}_mash.png', dpi=300, bbox_inches='tight')
            plt.close('all')

        return SEC


    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def sec_circle(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal['']] = ''
        ) -> opst.pre.section.FiberSecMesh:

        """
        构建一个圆截面的纤维截面网格模型，包含保护层和内孔，以及钢筋。

        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            info (bool, optional): 是否显示截面信息。默认值为 True。
            save_sec (Union[Path, str, Literal['']], optional): 是否保存截面可视化图片。默认值为 ''。

        Returns:
            opst.pre.section.FiberSecMesh: 包含截面网格模型的对象。
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            sec_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料编号
        cover_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_cover')
        core_tag =manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_core')
        rebar_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_rebar')
        # 定义截面编号
        sec_tag = manager.next_tag(category="section", label=sec_name) # 定义截面编号

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 形状参数
        bar_dia: float = 28 * UNIT.mm
        cover: float = 3.5 * UNIT.cm
        R: float = 80. * UNIT.cm
        r: float = 20. * UNIT.cm

        # 轮廓线
        cover_outline = opst.pre.section.create_circle_points(xo=[0, 0], radius=R - cover, n_sub=30) # 保护层轮廓线
        inner_outline = opst.pre.section.create_circle_points(xo=[0, 0], radius=r, n_sub=30) # 内孔轮廓线
        # 生成网格形状
        cover_geo = opst.pre.section.create_circle_patch(xo=[0, 0], radius=R, holes=[cover_outline], n_sub=30) # 保护层几何形状
        core_geo = opst.pre.section.create_circle_patch(xo=[0, 0], radius=R - cover, holes=[inner_outline], n_sub=30) # 核心几何形状

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name='RC Circle Section')
        SEC.add_patch_group({"Cover": cover_geo, "Core": core_geo})
        SEC.set_mesh_color({"Cover": "#ffa756", "Core": "#40a368"})
        SEC.set_ops_mat_tag({"Cover": cover_tag, "Core": core_tag}) # 明确材料编号

        # 添加钢筋
        SEC.add_rebar_circle(
            xo=[0, 0], radius=R - cover - bar_dia / 2.,
            dia=bar_dia, n=21,
            ops_mat_tag=rebar_tag, color='red')
        SEC.add_rebar_circle(
            xo=[0, 0], radius=r + cover + bar_dia / 2.,
            dia=bar_dia, n=11,
            ops_mat_tag=rebar_tag, color='red')

        # 生成网格
        SEC.mesh()
        SEC.centring()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info) # 截面信息
        
        # 材料类型
        concrete_type = "C40"
        rebar_type = "HRB400"
        # 素混凝土
        fc, Ec = ConcHub.get_fcuk(concrete_type) * UNIT.mpa, ConcHub.get_Ec(concrete_type) * UNIT.mpa
        ec, ecu = 0.002, 0.0033
        # 核心混凝土
        fcc, ecc, eccu = Mander.circular(
            hoop ='Circular', d=R * 2, coverThick=cover,
            fco=fc, roucc=sec_props['rho_rebar'],
            s=0.1 * UNIT.m, fyh=500. * UNIT.mpa,
            )
        # 钢筋
        fy, Es = ReBarHub.get_fyk(rebar_type) * UNIT.mpa, ReBarHub.get_Es(rebar_type) * UNIT.mpa
        
        # ops 材料参数
        cover_params = dict(fc=-fc, ec=-ec, ecu=-ecu, Ec=Ec)
        core_params = dict(fc=-fcc, ec=-ecc, ecu=-eccu, Ec=Ec)
        rebar_params = dict(fy=fy, Es=Es, b=0.01, R0=18, cR1=0.925, cR2=0.15)
        
        # 截面轴压力
        sec_props['P'] = 0.1 * (sec_props['A'] * abs(fc))

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料
        ops.uniaxialMaterial("Concrete04 ", cover_tag, *cover_params.values())
        ops.uniaxialMaterial("Concrete04 ", core_tag, *core_params.values())
        ops.uniaxialMaterial("Steel02 ", rebar_tag, *rebar_params.values())

        # 定义截面
        SEC.to_opspy_cmds(
            secTag=sec_tag,
            GJ=sec_props['J'] * ConcHub.get_G(concrete_type) * UNIT.mpa
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props['strain_stages'] = {
            rebar_tag: [fy / Es, 0.015, 0.055, 0.1],
            core_tag: [fy / Es, ecu, 0.75 * eccu, eccu],
            }
        # 储存至管理器
        manager.set_params(category="uniaxialMaterial", tag=cover_tag, params=cover_params) # 保护层
        manager.set_params(category="uniaxialMaterial", tag=core_tag, params=core_params) # 核心
        manager.set_params(category="uniaxialMaterial", tag=rebar_tag, params=rebar_params) # 钢筋
        manager.set_params(category="section", tag=sec_tag, params=sec_props) # 截面

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化 截面
        if save_sec:
            plt.close('all')
            SEC.view(fill=True, show_legend=True)
            plt.savefig(f'{save_sec}/{sec_name}_mash.png', dpi=300, bbox_inches='tight')
            plt.close('all')

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
    out_path = Path('./OutData')
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Model
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    
    # # 截面
    # SectionHub.sec_I(manager=MM, save_sec=out_path)
    # SectionHub.sec_rect(manager=MM, save_sec=out_path)
    # SectionHub.sec_polygonal(manager=MM, save_sec=out_path)
    # SectionHub.sec_circle(manager=MM, save_sec=out_path)
    
    # 批量调用
    callables = opsu.get_callables(SectionHub)
    for item in callables:
        item.callable(manager=MM, save_sec=out_path)
    
    # 导出模型管理器
    MM.to_excel(Path(out_path) / 'ModelManager.xlsx')
