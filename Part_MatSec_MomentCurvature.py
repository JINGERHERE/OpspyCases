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
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from opstool.pre import section
from inspect import currentframe as curt_fra
from typing import Literal, TypeAlias, Union, Callable, cast

from script import UNIT, PVs
from script.pre import MatTools


"""
# --------------------------------------------------
# ========== < Part_MatSec_MomentCurvature > ==========
# --------------------------------------------------
"""

class MPhiSection:
    """
    使用静态方法定义截面：
        Section_Example_01:
            > Main: 常规 ops 建模命令 = {ops材料, ops纤维截面}
            > return: 截面属性

        Section_Example_02:
            > Main: 使用脚本 ops 建模命令 = {MatTools定义材料, ops纤维截面}
            > return: 截面属性

        Section_Example_03:
            > Main: 使用脚本 ops 建模命令 = 圆截面{MatTools定义材料, opstool纤维截面}
            > return: 截面属性

        Section_Example_04:
            > Main: 常规 ops 建模命令 = 矩形截面{MatTools定义材料, opstool纤维截面}
            > return: 截面属性
    """

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def Section_Example_01(
            filepath: str,
            section_tag: int,
            cover_tag: int, core_tag: int,
            bar_tag: int, bar_max_tag: int,
            info: bool
            ) -> PVs.SEC_PROPS:
        """
        - return: 截面属性
            - 截面属性:
                - 截面 面积
                - 截面 配筋率
            - 保护层混凝土 材料标签
            - 保护层混凝土 材料对应参数
            - 核心混凝土 材料标签
            - 材料对应参数
            - 核心混凝土 材料标签
            - 核心混凝土材料对应参数
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            my_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 材料控制
        concrete = cast(PVs.concrete_input, "C30")
        steel = cast(PVs.bar_input, "HRB400")

        # 截面控制
        cover = 0.05 * UNIT.m  # 保护层厚度
        dia = 0.8 * UNIT.m # 圆截面直径
        Area = np.pi * (dia / 2) ** 2 # 圆截面面积

        bar_num = 16 # 纵筋个数
        bar_dia = 16 * UNIT.mm # 纵筋直径
        bar_area = np.pi * (bar_dia / 2.) ** 2

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面信息
        sec_props = PVs.SEC_MASH_PROPS(
            A=Area,
            rho_rebar=(bar_num * bar_area) / Area
            )

        # 材料参数字典
        cover_params = PVs.Concrete04Params()
        core_params = PVs.Concrete04Params()
        bar_params = PVs.Steel02Params()

        cover_params.set_params(
            fc=-25 * UNIT.mpa,
            ec=-0.002,
            ecu=-0.0033,
            Ec=28 * UNIT.gpa,
            )
        core_params.set_params(
            fc=1.2 * -25 * UNIT.mpa,
            ec=-0.007,
            ecu=-0.015,
            Ec=28 * UNIT.gpa,
            )
        bar_params.set_params(
            fy=335 * UNIT.mpa,
            Es=206 * UNIT.gpa,
            b=0.01,
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        cover_params_ls = cover_params.to_tuple(include=('fc', 'ec', 'ecu', 'Ec'))
        core_params_ls = core_params.to_tuple(include=('fc', 'ec', 'ecu', 'Ec'))
        bar_params_ls = bar_params.to_tuple(include=('fy', 'Es', 'b', 'R0', 'cR1', 'cR2'))

        # 定义ops材料
        ops.uniaxialMaterial('Concrete04', cover_tag, *cover_params_ls)
        ops.uniaxialMaterial('Concrete04', core_tag, *core_params_ls)
        ops.uniaxialMaterial('Steel02', bar_tag, *bar_params_ls)
        ops.uniaxialMaterial('MinMax', bar_max_tag, bar_tag, '-max', 0.1)  # 钢筋最大限制应变

        # 定义截面（ops命令）
        ops.section('fiberSec', section_tag, '-GJ', 482166.23821765214)
        ops.patch('circ', cover_tag, 20, 1, 0.0, 0.0, 0.35, 0.4, 0.0, 360.0)
        ops.patch('circ', core_tag, 20, 10, 0.0, 0.0, 0.0, 0.35, 0.0, 360.0)
        ops.layer('circ', bar_max_tag, bar_num, 0.00011304, 0.0, 0.0, 0.35, 0.0, 360.0)

        # 一维扭转材料
        # ops.uniaxialMaterial('Elastic', 702, 482166.23821765214)
        # ops.section('Aggregator', 703, 702, 'T', '-section', 701)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 弯矩曲率分析所需的轴压力
        P = 0.1 * (sec_props.A * abs(cover_params.fc))
        # 输出参数：截面属性，保护层数据，核心数据，钢筋数据
        PROPS = PVs.SEC_PROPS(
            Name=my_name,

            SectionTag=section_tag,
            SecMashProps=sec_props,

            CoverTag=cover_tag,
            CoverProps=cover_params,

            CoreTag=core_tag,
            CoreProps=core_params,

            SteelTag=bar_max_tag,
            SteelProps=bar_params,

            P=P
            )

        return PROPS

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def Section_Example_02(
            filepath: str,
            section_tag: int,
            cover_tag: int, core_tag: int,
            bar_tag: int, bar_max_tag: int,
            info: bool
        ) -> PVs.SEC_PROPS:
        """
        - return: 截面属性
            - 截面属性:
                - 截面 面积
                - 截面 配筋率
            - 保护层混凝土 材料标签
            - 保护层混凝土 材料对应参数
            - 核心混凝土 材料标签
            - 材料对应参数
            - 核心混凝土 材料标签
            - 核心混凝土材料对应参数
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            my_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 材料控制
        concrete = cast(PVs.concrete_input, "C30")
        steel = cast(PVs.bar_input, "HRB400")

        # 截面控制
        cover = 0.045 * UNIT.m  # 保护层厚度
        bar_1 = 20 * UNIT.mm  # 钢筋直径
        bar_2 = 16 * UNIT.mm  # 钢筋直径
        bar_a_1 = np.pi * (bar_1 / 2.) ** 2
        bar_a_2 = np.pi * (bar_2 / 2.) ** 2


        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面信息
        sec_props = PVs.SEC_MASH_PROPS(
            A=0.915 * UNIT.m ** 2,
            rho_rebar=(20 * bar_a_1 + 14 * bar_a_2) / 0.915
        )

        # 从生成的截面中获取截面数据, 用于mander公式
        RectangularSection = PVs.ManderRectangularParams()
        RectangularSection.set_params(
            lx=0.8 * UNIT.m,  # x方向截面的宽度
            ly=1.155 * UNIT.m,  # y方向截面宽度
            coverThick=cover,  # 保护层厚度
            roucc=sec_props.rho_rebar,  # 纵筋配筋率, 计算时只计入约束混凝土面积
            # sl=0.1 * UNIT.m,  # 纵筋间距
            # dsl = 0.032 * UNIT.m,  # 纵筋直径
            # roux = 0.00057,  # x方向的体积配箍率, 计算时只计入约束混凝土面积
            # rouy = 0.00889,  # y方向的体积配箍率, 计算时只计入约束混凝土面积
            # st = 0.3 * UNIT.m,  # 箍筋间距
            # dst = 0.018 * UNIT.m,  # 箍筋直径
            # fyh = 500 * UNIT.mpa,  # 箍筋屈服强度(MPa)
            # fco = 40 * UNIT.mpa,  # 无约束混凝土抗压强度标准值(MPa)
            )

        # 定义材料（ops命令）
        cover_params, core_params, bar_params = MatTools.set_mat_usr(
            RectangularSection,
            "Concrete04", 25. * UNIT.mpa, 28 * UNIT.gpa,
            "Steel02", 335. * UNIT.mpa, 200 * UNIT.gpa,
            info
            )
        MatTools.define(cover_tag, core_tag, bar_tag, bar_max_tag)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义截面（ops命令）
        ops.section('fiberSec', section_tag, '-GJ', 1011538.9888603257)
        # 保护层
        ops.patch('rect', cover_tag, 17, 1, -0.55, 0.4394, 0.55, 0.4844) # 顶部
        ops.patch('quad', cover_tag, 1, 10, -0.55, 0.0344, -0.505, 0.0622, -0.505, 0.4394, -0.55, 0.4394) # 左上
        ops.patch('quad', cover_tag, 1, 10, 0.505, 0.0622, 0.55, 0.0344, 0.55, 0.4394, 0.505, 0.4394) # 右上
        ops.patch('quad', cover_tag, 1, 5, -0.25, -0.1156, -0.205, -0.0878, -0.505, 0.0622, -0.55, 0.0344) # 斜左
        ops.patch('quad', cover_tag, 1, 5, 0.205, -0.0878, 0.25, -0.1156, 0.55, 0.0344, 0.505, 0.0622) # 斜右
        ops.patch('quad', cover_tag, 1, 10, -0.25, -0.6706, -0.205, -0.6706, -0.205, -0.0878, -0.25, -0.1156) # 腹左
        ops.patch('quad', cover_tag, 1, 10, 0.205, -0.6706, 0.25, -0.6706, 0.25, -0.1156, 0.205, -0.0878) # 腹右
        ops.patch('rect', cover_tag, 7, 1, -0.25, -0.7156, 0.25, -0.6706) # 底部
        # 核心
        ops.patch('quad', core_tag, 5, 10, -0.505, 0.0622, -0.205, -0.0878, -0.205, 0.4394, -0.505, 0.4394) # 左上
        ops.patch('quad', core_tag, 5, 10, 0.205, -0.0878, 0.505, 0.0622, 0.505, 0.4394, 0.205, 0.4394) # 右上
        ops.patch('rect', core_tag, 5, 10, -0.205, -0.0878, 0.205, 0.4394) # 中上
        ops.patch('rect', core_tag, 5, 10, -0.205, -0.6706, 0.205, -0.0878) # 中下
        # 纵筋 18
        ops.layer('straight', bar_max_tag, 8, bar_a_1, -0.496, 0.4304, 0.496, 0.4304) # 顶部
        ops.layer('straight', bar_max_tag, 4, bar_a_1, -0.196, -0.6616, 0.196, -0.6616) # 底部
        ops.layer('straight', bar_max_tag, 4, bar_a_1, -0.196, 0.3944, 0.196, 0.3944) # 顶部第二层
        ops.layer('straight', bar_max_tag, 4, bar_a_1, -0.196, -0.6256, 0.196, -0.6256) # 底部第二层
        # 腰筋 10
        ops.fiber(-0.5, 0.3304, bar_a_2, bar_max_tag)  # 左上
        ops.fiber(-0.5, 0.1652, bar_a_2, bar_max_tag)
        ops.fiber(-0.5, 0.0653, bar_a_2, bar_max_tag)
        ops.fiber(-0.2, -0.0828, bar_a_2, bar_max_tag)  # 左下
        ops.fiber(-0.2, -0.1861, bar_a_2, bar_max_tag)
        ops.fiber(-0.2, -0.3722, bar_a_2, bar_max_tag)
        ops.fiber(-0.2, -0.5583, bar_a_2, bar_max_tag)

        ops.fiber(0.5, 0.3304, bar_a_2, bar_max_tag)  # 右上
        ops.fiber(0.5, 0.1652, bar_a_2, bar_max_tag)
        ops.fiber(0.5, 0.0653, bar_a_2, bar_max_tag)
        ops.fiber(0.2, -0.0828, bar_a_2, bar_max_tag)  # 右下
        ops.fiber(0.2, -0.1861, bar_a_2, bar_max_tag)
        ops.fiber(0.2, -0.3722, bar_a_2, bar_max_tag)
        ops.fiber(0.2, -0.5583, bar_a_2, bar_max_tag)

        # 一维扭转材料
        # ops.uniaxialMaterial('Elastic', 802, 1011538.9888603257)
        # ops.section('Aggregator', 803, 802, 'T', '-section', 801)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 弯矩曲率分析所需的轴压力
        P = 0.1 * (sec_props.A * abs(cover_params.fc))

        # 输出参数：截面属性，保护层数据，核心数据，钢筋数据
        PROPS = PVs.SEC_PROPS(
            Name=my_name,

            SectionTag=section_tag,
            SecMashProps=sec_props,

            CoverTag=cover_tag,
            CoverProps=cover_params,

            CoreTag=core_tag,
            CoreProps=core_params,

            SteelTag=bar_max_tag,
            SteelProps=bar_params,

            P=P
            )

        return PROPS

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def Section_Example_03(
            filepath: str,
            section_tag: int,
            cover_tag: int, core_tag: int,
            bar_tag: int, bar_max_tag: int,
            info: bool
        ) -> PVs.SEC_PROPS:
        """
        - return: 截面属性
            - 截面属性:
                - 截面 面积
                - 截面 配筋率
            - 保护层混凝土 材料标签
            - 保护层混凝土 材料对应参数
            - 核心混凝土 材料标签
            - 材料对应参数
            - 核心混凝土 材料标签
            - 核心混凝土材料对应参数
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            my_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 材料控制
        concrete = cast(PVs.concrete_input, "C40")
        steel = cast(PVs.bar_input, "HRB400")

        # 截面控制
        d = 0.8 * UNIT.m
        cover = 35 * UNIT.mm  # 保护层厚度
        dia_bar = 35 * UNIT.mm  # 钢筋直径
        n_bar = 16  # 纵筋个数

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 创建截面网格
        xo_pier = [0, 0]
        outer_radius = d / 2
        outer_points = section.create_circle_points(xo_pier, outer_radius, angles=(0, 360), n_sub=40)
        cover_points = section.create_circle_points(xo_pier, outer_radius - cover, angles=(0, 360), n_sub=40)
        cover_geo = section.create_polygon_patch(outer_points, holes=[cover_points])
        core_geo = section.create_polygon_patch(cover_points)
        # 截面网格配置
        SEC = section.FiberSecMesh(sec_name=f"{my_name}")  # 当前截面名称
        SEC.add_patch_group({"cover": cover_geo, "core": core_geo})
        SEC.set_mesh_size({"cover": 0.1 * UNIT.m, "core": 0.1 * UNIT.m})
        SEC.set_mesh_color({"cover": "#DBB40C", "core": "#88B378"})
        SEC.set_ops_mat_tag({"cover": cover_tag, "core": core_tag})
        SEC.mesh()
        # 按点轮廓线添加钢筋
        rebars_outer = section.create_circle_points(xo_pier, outer_radius - cover - dia_bar / 2, angles=(0, 360), n_sub=n_bar)
        SEC.add_rebar_points(
            points=rebars_outer,
            dia=dia_bar,
            ops_mat_tag=bar_max_tag,
            group_name="rebar",
            color="#580F41"
            )

        # 获取截面信息
        props = SEC.get_frame_props(display_results=info) # 其中: centroid 为根据轮廓线坐标定义的质心
        sec_props = PVs.SEC_MASH_PROPS(**props)
        # 从生成的截面中获取截面数据, 用于mander公式
        CircularSection = PVs.ManderCircularParams()
        CircularSection.set_params(
            hoop="Spiral", # 箍筋类型，Circular圆形箍筋，Spiral螺旋形箍筋
            d=d,  #截面直径
            coverThick=cover,  #保护层厚度
            roucc=sec_props.rho_rebar,  #纵筋配筋率, 计算时只计入约束混凝土面积
            # s=0.1 * UNIT.m,  # 箍筋纵向间距（螺距）
            # ds = 0.014 * UNIT.m,  # 箍筋直径
            # fyh = 400 * UNIT.mpa,  # 箍筋屈服强度(MPa)
            # fco = 40 * UNIT.mpa,  # 无约束混凝土抗压强度标准值(MPa)
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料（ops命令）
        cover_params, core_params, bar_params = MatTools.mat_config(
            CircularSection,
            "Concrete04", concrete,
            "ReinforcingSteel", steel,
            info
            )
        MatTools.define(cover_tag, core_tag, bar_tag, bar_max_tag)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义截面（ops命令）
        SEC.centring()
        SEC.view(fill=True, show_legend=True)
        SEC.to_opspy_cmds(secTag=section_tag, GJ=cover_params.G * sec_props.J)
        plt.savefig(f'{filepath}/{my_name}_mash.png', dpi=300, bbox_inches='tight')

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 弯矩曲率分析所需的轴压力
        P = 0.1 * (sec_props.A * abs(cover_params.fc))

        # 输出参数：截面属性，保护层数据，核心数据，钢筋数据
        PROPS = PVs.SEC_PROPS(
            Name=my_name,

            SectionTag=section_tag,
            SecMashProps=sec_props,

            CoverTag=cover_tag,
            CoverProps=cover_params,

            CoreTag=core_tag,
            CoreProps=core_params,

            SteelTag=bar_max_tag,
            SteelProps=bar_params,

            P=P
            )

        return PROPS

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def Section_Example_04(
            filepath: str,
            section_tag: int,
            cover_tag: int, core_tag: int,
            bar_tag: int, bar_max_tag: int,
            info: bool
        ) -> PVs.SEC_PROPS:
        """
        - return: 截面属性
            - 截面属性:
                - 截面 面积
                - 截面 配筋率
            - 保护层混凝土 材料标签
            - 保护层混凝土 材料对应参数
            - 核心混凝土 材料标签
            - 材料对应参数
            - 核心混凝土 材料标签
            - 核心混凝土材料对应参数
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            my_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section Name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 材料控制
        # 材料控制
        concrete = cast(PVs.concrete_input, "C40")
        steel = cast(PVs.bar_input, "HRB400")

        # 截面控制
        w = 1.6 * UNIT.m
        h = 1.8 * UNIT.m
        t = 0.5 * UNIT.m
        cover = 50 * UNIT.mm  # 保护层厚度
        dia_bar = 35 * UNIT.mm  # 钢筋直径
        n_bar_out = 35  # 外圈纵筋个数
        n_bar_in = 14  # 内圈纵筋个数

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 创建截面网格
        sec_outlines = [[0, 0], [w, 0], [w, h], [0, h]]
        sec_coverlines = section.offset(sec_outlines, cover)
        sec_holelines = section.offset(sec_outlines, t) # 空心部分边缘线
        cover_geo = section.create_polygon_patch(sec_outlines, holes=[sec_coverlines])  # 保护层形状
        core_geo = section.create_polygon_patch(sec_coverlines, holes=[sec_holelines])  # 核心形状
        # 截面网格配置
        SEC = section.FiberSecMesh(sec_name=f"{my_name}")  # 当前截面名称
        SEC.add_patch_group({"cover": cover_geo, "core": core_geo})
        SEC.set_mesh_size({"cover": 0.1 * UNIT.m, "core": 0.1 * UNIT.m})
        SEC.set_mesh_color({"cover": "#dbb40c", "core": "#88b378"})
        SEC.set_ops_mat_tag({"cover": cover_tag, "core": core_tag})
        SEC.mesh()
        # 按线添加钢筋
        rebars_outer = section.offset(sec_coverlines, dia_bar / 2)
        SEC.add_rebar_line(
            points=rebars_outer,
            dia=dia_bar,
            n=n_bar_out,
            ops_mat_tag=bar_max_tag,
            group_name="rebar",
            color="#580F41"
            )
        rebars_inner = section.offset(sec_holelines, -cover - dia_bar / 2)
        SEC.add_rebar_line(
            points=rebars_inner,
            dia=dia_bar,
            n=n_bar_in,
            ops_mat_tag=bar_max_tag,
            group_name="rebar",
            color="#580F41"
            )

        # 获取截面信息
        props = SEC.get_frame_props(display_results=info) # 其中: centroid 为根据轮廓线坐标定义的质心
        sec_props = PVs.SEC_MASH_PROPS(**props)
        # 从生成的截面中获取截面数据, 用于mander公式
        RectangularSection = PVs.ManderRectangularParams()
        RectangularSection.set_params(
            lx=w,
            ly=h ,
            coverThick=cover,
            roucc=sec_props.rho_rebar,
            # sl=0.1 * UNIT.m,  # 纵筋间距
            # dsl = 0.032 * UNIT.m,  # 纵筋直径
            # roux = 0.00057,  # x方向的体积配箍率, 计算时只计入约束混凝土面积
            # rouy = 0.00889,  # y方向的体积配箍率, 计算时只计入约束混凝土面积
            # st = 0.3 * UNIT.m,  # 箍筋间距
            # dst = 0.018 * UNIT.m,  # 箍筋直径
            # fyh = 500 * UNIT.mpa,  # 箍筋屈服强度(MPa)
            # fco = 40 * UNIT.mpa,  # 无约束混凝土抗压强度标准值(MPa)
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料（ops命令）
        cover_params, core_params, bar_params = MatTools.mat_config(
            RectangularSection,
            "Concrete04", concrete,
            "ReinforcingSteel", steel,
            info
            )
        MatTools.define(cover_tag, core_tag, bar_tag, bar_max_tag)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义截面（ops命令）
        SEC.centring()
        SEC.view(fill=True, show_legend=True)
        SEC.to_opspy_cmds(secTag=section_tag, GJ=cover_params.G * sec_props.J)
        plt.savefig(f'{filepath}/{my_name}_mash.png', dpi=300, bbox_inches='tight')

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 弯矩曲率分析所需的轴压力
        P = 0.1 * (sec_props.A * abs(cover_params.fc))

        # 输出参数：截面属性，保护层数据，核心数据，钢筋数据
        PROPS = PVs.SEC_PROPS(
            Name=my_name,

            SectionTag=section_tag,
            SecMashProps=sec_props,

            CoverTag=cover_tag,
            CoverProps=cover_params,

            CoreTag=core_tag,
            CoreProps=core_params,

            SteelTag=bar_max_tag,
            SteelProps=bar_params,

            P=P
            )

        return PROPS

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""

if __name__ == "__main__":
    pass
