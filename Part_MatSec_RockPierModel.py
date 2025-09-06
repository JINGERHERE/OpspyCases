#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Part_MatSec_RockPierModel.py
@Date    ：2025/8/1 19:21
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""



import sys
import rich
from math import pi
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from opstool.pre import section
from collections import namedtuple
from inspect import currentframe as curt_fra

import script
from script import UNIT, PVs
from script.base import random_color
from script.pre import MatTools


"""
# --------------------------------------------------
# ========== < Part_MatSec_RockPierModel > ==========
# --------------------------------------------------
"""


class RockPierModelSection:

    """
    ops.uniaxialMaterial('MinMax',...
    MinMax材料会有刚度的问题
    """

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def bent_cap_sec(
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
            - 保护层混凝土 材料属性:
                - 材料标签
                - 材料对应参数
            - 核心混凝土 材料属性:
                - 材料标签
                - 材料对应参数
            - 钢筋 材料属性:
                - 材料标签
                - 材料对应参数
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            my_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Section name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 材料控制
        fc = 43.3 * UNIT.mpa  # 混凝土抗压强度
        Ec = 29.8 * UNIT.gpa  # 混凝土弹性模量

        fy = 435 * UNIT.mpa  # 钢筋屈服强度
        Es = 185 * UNIT.gpa  # 钢筋弹性模量

        # 截面控制
        W = 500 * UNIT.mm
        H = 400 * UNIT.mm
        # t = 0.6 * UNIT.m
        cover = 20 * UNIT.mm  # 保护层厚度
        dia_bar = 12 * UNIT.mm  # 钢筋直径
        n_bar_out = 28  # 外圈纵筋个数
        n_bar_in = 0  # 内圈纵筋个数

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 创建截面网格
        sec_outlines = [[0, 0], [W, 0], [W, H], [0, H]]
        sec_coverlines = section.offset(sec_outlines, cover)
        # sec_holelines = section.offset(sec_outlines, t)  # 空心部分边缘线
        cover_geo = section.create_polygon_patch(sec_outlines, holes=[sec_coverlines])  # 保护层形状
        core_geo = section.create_polygon_patch(sec_coverlines)  # 核心形状
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
            # ops_mat_tag=bar_max_tag,
            ops_mat_tag=bar_tag,
            group_name="rebar",
            color="#580f41"
        )

        # 获取截面信息
        props = SEC.get_frame_props(display_results=info)  # 其中: centroid 为根据轮廓线坐标定义的质心
        sec_props = PVs.SEC_MASH_PROPS(**props)
        # 从生成的截面中获取截面数据, 用于mander公式
        RectangularSection = PVs.ManderRectangularParams()
        Acor = (W - 2 * (dia_bar + cover)) * (H - 2 * (dia_bar + cover)) * (160 * UNIT.mm) # 核心面积 x 箍筋间距
        As = pi * (6 / 2 * UNIT.mm) ** 2 # 箍筋单肢面积
        RectangularSection.set_params(
            lx=W,
            ly=H,
            coverThick=cover,
            roucc=sec_props.rho_rebar,
            sl=((W + H) * 2 - 8 * cover) / (n_bar_out - 1),  # 纵筋间距
            dsl=dia_bar,  # 纵筋直径
            # roux = (2 * As * (W - 2 * (cover + dia_bar))) / Acor, # x方向的体积配箍率, 计算时只计入约束混凝土面积
            # rouy = (6 * As * (H - 2 * (cover + dia_bar))) / Acor, # y方向的体积配箍率, 计算时只计入约束混凝土面积
            # st=160 * UNIT.mm,  # 箍筋间距
            dst=6 * UNIT.mm,  # 箍筋直径
            fyh=437.7 * UNIT.mpa,  # 箍筋屈服强度(MPa)
            # fco = 40 * UNIT.mpa,  # 无约束混凝土抗压强度标准值(MPa)
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料（ops命令）
        cover_params, core_params, bar_params = MatTools.set_mat_usr(
            RectangularSection,
            "Concrete04", fc, Ec,
            "Steel02", fy, Es,
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
        # 截面材料属性转换为字典
        # cover_dict = cover.to_dict()
        # core_dict = core.to_dict()
        # bar_dict = bar.to_dict()

        # 弯矩曲率分析所需的轴压力
        P = -0.1 * (sec_props.A * abs(cover_params.fc))
        # 输出参数：截面属性，保护层数据，核心数据，钢筋数据
        PROPS = PVs.SEC_PROPS(
            Name=my_name,

            SectionTag=section_tag,
            SecMashProps=sec_props,

            CoverTag=cover_tag,
            CoverProps=cover_params,

            CoreTag=core_tag,
            CoreProps=core_params,

            SteelTag=bar_tag,
            SteelProps=bar_params,

            P=P
        )

        return PROPS

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    @staticmethod
    def pier_sec(
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
            raise RuntimeError("Get Section name Error")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 材料控制
        fc = 43.3 * UNIT.mpa  # 混凝土抗压强度
        Ec = 29.8 * UNIT.gpa  # 混凝土弹性模量

        fy = 435 * UNIT.mpa  # 钢筋屈服强度
        Es = 185 * UNIT.gpa  # 钢筋弹性模量

        # 截面控制
        W = 450 * UNIT.mm
        H = 400 * UNIT.mm
        # t = 0.6 * UNIT.m
        cover = 20 * UNIT.mm  # 保护层厚度
        dia_bar = 12 * UNIT.mm  # 钢筋直径
        n_bar_out = 28  # 外圈纵筋个数
        n_bar_in = 0  # 内圈纵筋个数

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 创建截面网格
        sec_outlines = [[0, 0], [W, 0], [W, H], [0, H]]
        sec_coverlines = section.offset(sec_outlines, cover)
        # sec_holelines = section.offset(sec_outlines, t)  # 空心部分边缘线
        cover_geo = section.create_polygon_patch(sec_outlines, holes=[sec_coverlines])  # 保护层形状
        core_geo = section.create_polygon_patch(sec_coverlines)  # 核心形状
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
            # ops_mat_tag=bar_max_tag,
            ops_mat_tag=bar_tag,
            group_name="rebar",
            color="#580f41"
        )

        # 获取截面信息
        props = SEC.get_frame_props(display_results=info)  # 其中: centroid 为根据轮廓线坐标定义的质心
        sec_props = PVs.SEC_MASH_PROPS(**props)
        # 从生成的截面中获取截面数据, 用于mander公式
        RectangularSection = PVs.ManderRectangularParams()
        Acor = (W - 2 * (dia_bar + cover)) * (H - 2 * (dia_bar + cover)) * (160 * UNIT.mm)  # 核心面积 x 箍筋间距
        As = pi * (6 / 2 * UNIT.mm) ** 2  # 箍筋单肢面积
        RectangularSection.set_params(
            lx=W,
            ly=H,
            coverThick=cover,
            roucc=sec_props.rho_rebar,
            sl=((W + H) * 2 - 8 * cover) / (n_bar_out - 1),  # 纵筋间距
            dsl=dia_bar,  # 纵筋直径
            # roux = (2 * As * (W - 2 * (cover + dia_bar))) / Acor, # x方向的体积配箍率, 计算时只计入约束混凝土面积
            # rouy = (6 * As * (H - 2 * (cover + dia_bar))) / Acor, # y方向的体积配箍率, 计算时只计入约束混凝土面积
            # st=160 * UNIT.mm,  # 箍筋间距
            dst=6 * UNIT.mm,  # 箍筋直径
            fyh=437.7 * UNIT.mpa,  # 箍筋屈服强度(MPa)
            # fco = 40 * UNIT.mpa,  # 无约束混凝土抗压强度标准值(MPa)
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义材料（ops命令）
        cover_params, core_params, bar_params = MatTools.set_mat_usr(
            RectangularSection,
            "Concrete04", fc, Ec,
            "Steel02", fy, Es,
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
        # 截面材料属性转换为字典
        # cover_dict = cover.to_dict()
        # core_dict = core.to_dict()
        # bar_dict = bar.to_dict()

        # 弯矩曲率分析所需的轴压力
        P = -0.1 * (sec_props.A * abs(cover_params.fc))
        # 输出参数：截面属性，保护层数据，核心数据，钢筋数据
        PROPS = PVs.SEC_PROPS(
            Name=my_name,

            SectionTag=section_tag,
            SecMashProps=sec_props,

            CoverTag=cover_tag,
            CoverProps=cover_params,

            CoreTag=core_tag,
            CoreProps=core_params,

            SteelTag=bar_tag,
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
