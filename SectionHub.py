#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：SectionHub.py
@Date    ：2025/8/1 19:21
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
from ModelUtilities import UNIT


"""
# --------------------------------------------------
# ========== < SectionHub > ==========
# --------------------------------------------------
"""


class SectionHub:

    """
    ops.uniaxialMaterial('MinMax',...
    MinMax材料会有刚度的问题
    """

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def bent_cap(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal['']] = ''
        ) -> opst.pre.section.FiberSecMesh:

        """
        自复位摇摆桥 `盖梁` 截面。

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
        rebar_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_rebar') # 标签标记为绘图所需
        # 定义截面编号
        sec_tag = manager.next_tag(category="section", label=sec_name) # 定义截面编号
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面控制
        W = 500 * UNIT.mm
        H = 400 * UNIT.mm
        cover = 20 * UNIT.mm  # 保护层厚度
        bar_dia = 12 * UNIT.mm  # 钢筋直径
        bar_n = 28  # 外圈纵筋个数

        # 轮廓线
        sec_outlines = [[0, 0], [W, 0], [W, H], [0, H]]
        cover_outlines = opst.pre.section.offset(sec_outlines, d=cover) # 保护层线

        # 生成几何形状
        cover_geo = opst.pre.section.create_polygon_patch(outline=sec_outlines, holes=[cover_outlines])
        core_geo = opst.pre.section.create_polygon_patch(outline=cover_outlines)

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name='Rock Pier Bent Cap Section')
        SEC.add_patch_group({"Cover": cover_geo, "Core": core_geo})
        SEC.set_mesh_size({"Cover": 0.05, "Core": 0.05})
        SEC.set_mesh_color({"Cover": "#ffa756", "Core": "#40a368"})
        SEC.set_ops_mat_tag({"Cover": cover_tag, "Core": core_tag}) # 明确材料编号

        # 底层添加钢筋
        rebars_outer = opst.pre.section.offset(cover_outlines, bar_dia / 2)
        SEC.add_rebar_line(
            points=rebars_outer,
            dia=bar_dia, n=bar_n,
            ops_mat_tag=rebar_tag, color='red'
            )

        # 生成网格
        SEC.mesh()
        SEC.centring()
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info) # 截面信息
        
        # 素混凝土
        fc, Ec = 43.3 * UNIT.mpa, 29.8 * UNIT.gpa
        ec, ecu = 0.002, 0.0033
        # 核心混凝土
        Acor = (W - 2 * (bar_dia + cover)) * (H - 2 * (bar_dia + cover)) * (160 * UNIT.mm) # 核心面积 x 箍筋间距
        As = np.pi * (6 / 2 * UNIT.mm) ** 2 # 箍筋单肢面积
        fcc, ecc, eccu = Mander.rectangular(
            lx=W, ly=H, coverThick=cover, fco=fc, fyh=437.7 * UNIT.mpa,
            dsl=bar_dia, roucc=sec_props['rho_rebar'],
            roux = (2 * As * (W - 2 * (cover + bar_dia))) / Acor, # x方向的体积配箍率, 计算时只计入约束混凝土面积
            rouy = (6 * As * (H - 2 * (cover + bar_dia))) / Acor, # y方向的体积配箍率, 计算时只计入约束混凝土面积
            st=25 * UNIT.cm, dst=6 * UNIT.mm,
            )
        # 钢筋
        fy, Es = 435 * UNIT.mpa, 185 * UNIT.gpa
        
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
            GJ=sec_props['J'] * ConcHub.get_G('C40') * UNIT.mpa
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props['strain_stages'] = {
            rebar_tag: (fy / Es, 0.015, 0.055, 0.1),
            core_tag: (-fy / Es, -ecu, 0.75 * -eccu, -eccu),
            }
        # 截面 应变阈值 用于：opst.pre.section.FiberSecMesh.plot_response(thresholds=)
        sec_props['strain_thresholds'] = {
            rebar_tag: (-0.1, 0.1), # 只能两个值
            cover_tag: (-ecu, 0.),
            core_tag: (-eccu, 0.),
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
    def pier_col(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal['']] = ''
        ) -> opst.pre.section.FiberSecMesh:

        """
        自复位摇摆桥 `墩柱` 截面。

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
        rebar_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}_rebar') # 标签标记为绘图所需
        # 定义截面编号
        sec_tag = manager.next_tag(category="section", label=sec_name) # 定义截面编号
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面控制
        W = 450 * UNIT.mm
        H = 400 * UNIT.mm
        cover = 20 * UNIT.mm  # 保护层厚度
        bar_dia = 12 * UNIT.mm  # 钢筋直径
        bar_n = 28  # 外圈纵筋个数

        # 轮廓线
        sec_outlines = [[0, 0], [W, 0], [W, H], [0, H]]
        cover_outlines = opst.pre.section.offset(sec_outlines, d=cover) # 保护层线

        # 生成几何形状
        cover_geo = opst.pre.section.create_polygon_patch(outline=sec_outlines, holes=[cover_outlines])
        core_geo = opst.pre.section.create_polygon_patch(outline=cover_outlines)

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name='Rock Pier Column Section')
        SEC.add_patch_group({"Cover": cover_geo, "Core": core_geo})
        SEC.set_mesh_size({"Cover": 0.05, "Core": 0.05})
        SEC.set_mesh_color({"Cover": "#ffa756", "Core": "#40a368"})
        SEC.set_ops_mat_tag({"Cover": cover_tag, "Core": core_tag}) # 明确材料编号

        # 底层添加钢筋
        rebars_outer = opst.pre.section.offset(cover_outlines, bar_dia / 2)
        SEC.add_rebar_line(
            points=rebars_outer,
            dia=bar_dia, n=bar_n,
            ops_mat_tag=rebar_tag, color='red'
            )

        # 生成网格
        SEC.mesh()
        SEC.centring()
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info) # 截面信息
        
        # 素混凝土
        fc, Ec = 43.3 * UNIT.mpa, 29.8 * UNIT.gpa
        ec, ecu = 0.002, 0.0033
        # 核心混凝土
        Acor = (W - 2 * (bar_dia + cover)) * (H - 2 * (bar_dia + cover)) * (160 * UNIT.mm) # 核心面积 x 箍筋间距
        As = np.pi * (6 / 2 * UNIT.mm) ** 2 # 箍筋单肢面积
        fcc, ecc, eccu = Mander.rectangular(
            lx=W, ly=H, coverThick=cover, fco=fc, fyh=437.7 * UNIT.mpa,
            dsl=bar_dia, roucc=sec_props['rho_rebar'],
            roux = (2 * As * (W - 2 * (cover + bar_dia))) / Acor, # x方向的体积配箍率, 计算时只计入约束混凝土面积
            rouy = (6 * As * (H - 2 * (cover + bar_dia))) / Acor, # y方向的体积配箍率, 计算时只计入约束混凝土面积
            st=25 * UNIT.cm, dst=6 * UNIT.mm,
            )
        # 钢筋
        fy, Es = 435 * UNIT.mpa, 185 * UNIT.gpa
        
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
            GJ=sec_props['J'] * ConcHub.get_G('C40') * UNIT.mpa
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props['strain_stages'] = {
            rebar_tag: (fy / Es, 0.015, 0.055, 0.1),
            core_tag: (-fy / Es, -ecu, 0.75 * -eccu, -eccu),
            }
        # 截面 应变阈值 用于：opst.pre.section.FiberSecMesh.plot_response(thresholds=)
        sec_props['strain_thresholds'] = {
            rebar_tag: (-0.1, 0.1), # 只能两个值
            cover_tag: (-ecu, 0.),
            core_tag: (-eccu, 0.),
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
    def contact_surf(
        manager: opsu.pre.ModelManager,
        info: bool = True,
        save_sec: Union[Path, str, Literal['']] = ''
        ) -> opst.pre.section.FiberSecMesh:

        """
        节段 `接触面` 截面。

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
        mat_tag = manager.next_tag(category="uniaxialMaterial", label=f'{sec_name}') # 标签标记为绘图所需
        # 截面标签
        sec_tag = manager.next_tag(category="section", label=sec_name) # 定义截面编号
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 形状参数
        W = 450 * UNIT.mm
        H = 400 * UNIT.mm

        # 轮廓线
        sec_outlines = [(0, 0), (W, 0), (W, H), (0, H)]
        # 生成几何形状
        inner_geo = opst.pre.section.create_polygon_patch(sec_outlines)

        # 截面网格
        SEC = opst.pre.section.FiberSecMesh(sec_name="Contact Surface Section")
        SEC.add_patch_group({"inner": inner_geo})
        SEC.set_mesh_size({"inner": 0.03})
        SEC.set_ops_mat_tag({"inner": mat_tag}) # 明确材料编号

        # 生成网格
        SEC.mesh()
        SEC.centring()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取截面信息
        sec_props = SEC.get_sec_props(display_results=info) # 截面信息
        
        # ops 材料参数
        # ENT_params = dict(Ee=30 * UNIT.gpa)
        # ENT_params = dict(Ee=892.9e3)
        ENT_params = dict(Ee=20 * UNIT.gpa)

        # 截面轴压力
        # sec_props['P'] = 0.1 * (sec_props['A'] * abs(fy))
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料
        ops.uniaxialMaterial("ENT", mat_tag, *ENT_params.values())

        # 定义截面
        SEC.to_opspy_cmds(secTag=sec_tag, GJ=57786.97)

        # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # # 旋转截面
        # SEC.rotate(0, remesh=True)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面 组合材料 损伤判断依据： "ops_utilities.post.SecMatStates.get_combined_steps_mat()"
        sec_props['strain_stages'] = {
            mat_tag: (0.1),
            # core_tag: (-fy / Es, -ecu, 0.75 * -eccu, -eccu),
            }
        # 截面 应变阈值 用于：opst.pre.section.FiberSecMesh.plot_response(thresholds=)
        sec_props['strain_thresholds'] = {
            mat_tag: (-0.1, 0.1), # 只能两个值
            # core_tag: (-eccu, 0.),
            }
        # 储存至管理器
        manager.set_params(category="uniaxialMaterial", tag=mat_tag, params=ENT_params) # 材料
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
    # SectionHub.bent_cap(manager=MM, save_sec=out_path)
    # SectionHub.pier_col(manager=MM, save_sec=out_path)
    # SectionHub.contact_surf(manager=MM, save_sec=out_path)
    
    # 批量调用
    callables = opsu.get_callables(SectionHub)
    for name, sec_fun in callables.items():
        sec_fun(manager=MM, save_sec=out_path)
    
    # 导出模型管理器
    MM.to_excel(Path(out_path) / 'ModelManager.xlsx')
