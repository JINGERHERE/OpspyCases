#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Part_Model_HeighPierModel.py
@Date    ：2025/08/10 12:28:33
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import os
import sys
import time
import warnings
from typing import Literal, TypeAlias, Union, Callable, Any, Optional
import matplotlib.pyplot as plt
import rich
import numpy as np
import pandas as pd
import xarray as xr
import openseespy.opensees as ops
import opstool as opst
import opstool.vis.plotly as opsplt
import opstool.vis.pyvista as opsvis
from inspect import currentframe as curt_fra
from itertools import batched, product, pairwise

from script.pre import NodeTools
from script.base import random_color
from script import UNIT, PVs

from Part_MatSec_HeighPierModel import HighPierModelSection
from script.post import DamageStateTools
from Script_ModelCreate import ModelCreateTools

from typing import NamedTuple
from collections import namedtuple

"""
# --------------------------------------------------
# ========== < Part_Model_HeighPierModel > ==========
# --------------------------------------------------
"""


class HeighPierModelTEST:

    # def __init__(self, rootPath: str = './Data'):
    def __init__(self):

        """实例化：即在当前根目录创建 数据文件夹"""

        # self.rootPath: str = rootPath # 根目录
        # os.makedirs(rootPath, exist_ok=True) # 创建根目录

        # self.modelPath: str # 子目录
        # self.case: str # 工况

        self.model_props: PVs.MODEL_PROPS # 模型属性

        # 结果数据
        self.node_resp: xr.DataArray # 节点响应数据
        self.ele_resp: xr.DataArray # 单元响应数据


    def HeighPier(
        self,
        modelPath: str,
        # Ke: float,
        info: bool
        ) -> PVs.MODEL_PROPS:

        """
        使用 @staticmethod 静态方法，可嵌合在类中，也可独立在外
        双柱式桥墩试验模型：基于兴宁桥 1/2 比例缩尺，上部结构的竖向力通过预应力模拟
        荷载工况适用于： 推覆分析 & 拟静力分析
        参数：
            PierH: 墩高
            PierW: 墩柱中心间距
            L: 盖梁长
            Ke: 拟合刚度
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            my_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Model Name Error")
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        # 实例化建模工具
        CreateTools = ModelCreateTools()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 桥墩尺寸控制
        # L = 2.8 * UNIT.m
        PierH = 60 * UNIT.m
        PierW = 10.2 * UNIT.m

        # 模型收敛刚度拟合 /确定值：0.15 * UNIT.pa 基于全局单位：kN，m
        Ke = 0.15 * UNIT.pa

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁截面 盖梁材料 编号
        pier_section_tags = {
            'section_tag': 100,  # 截面
            'cover_tag': 1,  # 材料-保护层
            'core_tag': 2,  # 材料-核心
            'bar_tag': 3,  # 材料-钢筋
            'bar_max_tag': 4,  # 材料-钢筋最大应变限制
            'info': info
            }
        PierProps = HighPierModelSection.pier_sec(modelPath, **pier_section_tags) # 创建墩柱纤维截面，并获取截面参数

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元积分点
        tag_np_pier = 1
        ops.beamIntegration('Legendre', tag_np_pier, pier_section_tags['section_tag'], 5)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 节点坐标
        pier_1_coord = NodeTools.distribute((0, -PierW / 2, PierH), (0, -PierW / 2, 0), 13, ends=(True, True))
        pier_2_coord = NodeTools.distribute((0, PierW / 2, PierH), (0, PierW / 2, 0), 13, ends=(True, True))

        # 节点编号组
        pier_1_node_start = 1100
        pier_2_node_start = 1200

        # 创建节点
        pier_1_node = CreateTools.node_create(pier_1_node_start, pier_1_coord)
        pier_2_node = CreateTools.node_create(pier_2_node_start, pier_2_coord)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元连接
        pier_1_links = []
        for node_i, node_j in pairwise(pier_1_node):
            pier_1_links.append((node_i, node_j))

        pier_2_links = []
        for node_i, node_j in pairwise(pier_2_node):
            pier_2_links.append((node_i, node_j))

        # 单元编组
        pier_1_ele_start = 2100
        pier_2_ele_start = 2200

        # 创建单元
        pier_1_ele = CreateTools.ele_create(pier_1_ele_start, pier_1_links, tag_np_pier)
        pier_2_ele = CreateTools.ele_create(pier_2_ele_start, pier_2_links, tag_np_pier)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 连接
        ops.uniaxialMaterial('Steel02', 10, 400*UNIT.mpa, 200*UNIT.gpa, 0.01)
        link_Transf = CreateTools.auto_geomTransf(pier_1_node[0], pier_2_node[0])
        ops.element('Truss', 2001, pier_1_node[0], pier_2_node[0], 1.0*UNIT.m*UNIT.m, 10)
        # ops.element('elasticBeamColumn', 2001, *(pier_1_node[0], pier_2_node[0]),
                    # PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    # 4900*UNIT.mm*UNIT.mm, Ke, Ke, Ke, link_Transf)

        # 约束
        ops.fixZ(0., 1, 1, 1, 1, 1, 1)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 质量密度
        rho = 2600 * (UNIT.kg / (UNIT.m ** 3))  # kg/m3

        # 单位节点质量
        pier_1_mass = PierH * PierProps.SecMashProps.A * rho / len(pier_1_node)
        pier_2_mass = PierH * PierProps.SecMashProps.A * rho / len(pier_2_node)
        
        supper_qe = abs(PierProps.P / 9.806) # 上部结构等效质量

        # 节点质量
        ops.mass(pier_1_node[0], supper_qe, supper_qe, supper_qe, 0, 0, 0)
        for i in range(1, len(pier_1_node)):
            ops.mass(pier_1_node[i], pier_1_mass, pier_1_mass, pier_1_mass, 0, 0, 0)
        
        ops.mass(pier_2_node[0], supper_qe, supper_qe, supper_qe, 0, 0, 0)
        for i in range(1, len(pier_2_node)):
            ops.mass(pier_2_node[i], pier_2_mass, pier_2_mass, pier_2_mass, 0, 0, 0)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 关键节点
        # KEY_NODE = namedtuple('KEY_NODE', ['pier_1_top', 'pier_1_base'])
        # KEY_NODE = namedtuple('KEY_NODE', ['pier_1_top', 'pier_2_top', 'pier_1_base', 'pier_2_base'])
        # 关键单元
        # KEY_ELE = namedtuple('KEY_ELE', ['pier_1_top', 'pier_1_base'])
        # KEY_ELE = namedtuple('KEY_ELE', ['pier_1_top', 'pier_2_top', 'pier_1_base', 'pier_2_base'])
        # 定义模型不同部位名称
        LOCATION_INFO = namedtuple('LOCATION_INFO', ['eleTag', 'integ', 'location'])

        # 返回数据
        self.model_props = PVs.MODEL_PROPS(
            Name=my_name,
            SectionMat=PierProps,
            KeyNode={
                'pier_1_top': pier_1_node[0],
                'pier_2_top': pier_2_node[0],
                'pier_1_base': pier_1_node[-1],
                'pier_2_base': pier_2_node[-1]
                },
            KeyEle={
                'pier_1_top': pier_1_ele[0],
                'pier_2_top': pier_2_ele[0],
                'pier_1_base_1': pier_1_ele[-1],
                'pier_2_base_1': pier_2_ele[-1],
                'pier_1_base_2': pier_1_ele[-2],
                'pier_2_base_2': pier_2_ele[-2],
                },
            LocationDamage=(
                # LOCATION_INFO(eleTag=pier_1_ele[-1], integ=1, location=f'col_1_{pier_1_ele[-1]}_1'),
                # LOCATION_INFO(eleTag=pier_1_ele[-1], integ=2, location=f'col_1_{pier_1_ele[-1]}_2'),
                # LOCATION_INFO(eleTag=pier_1_ele[-1], integ=3, location=f'col_1_{pier_1_ele[-1]}_3'),
                # LOCATION_INFO(eleTag=pier_1_ele[-1], integ=4, location=f'col_1_{pier_1_ele[-1]}_4'),
                LOCATION_INFO(eleTag=pier_1_ele[-1], integ=5, location=f'col_1_{pier_1_ele[-1]}_5'),
                
                # LOCATION_INFO(eleTag=pier_1_ele[-2], integ=1, location=f'col_1_{pier_1_ele[-2]}_1'),
                # LOCATION_INFO(eleTag=pier_1_ele[-2], integ=2, location=f'col_1_{pier_1_ele[-2]}_2'),
                # LOCATION_INFO(eleTag=pier_1_ele[-2], integ=3, location=f'col_1_{pier_1_ele[-2]}_3'),
                # LOCATION_INFO(eleTag=pier_1_ele[-2], integ=4, location=f'col_1_{pier_1_ele[-2]}_4'),
                # LOCATION_INFO(eleTag=pier_1_ele[-2], integ=5, location=f'col_1_{pier_1_ele[-2]}_5'),


                # LOCATION_INFO(eleTag=pier_2_ele[-1], integ=1, location=f'col_2_{pier_2_ele[-1]}_1'),
                # LOCATION_INFO(eleTag=pier_2_ele[-1], integ=2, location=f'col_2_{pier_2_ele[-1]}_2'),
                # LOCATION_INFO(eleTag=pier_2_ele[-1], integ=3, location=f'col_2_{pier_2_ele[-1]}_3'),
                # LOCATION_INFO(eleTag=pier_2_ele[-1], integ=4, location=f'col_2_{pier_2_ele[-1]}_4'),
                LOCATION_INFO(eleTag=pier_2_ele[-1], integ=5, location=f'col_2_{pier_2_ele[-1]}_5'),

                # LOCATION_INFO(eleTag=pier_2_ele[-2], integ=1, location=f'col_2_{pier_2_ele[-2]}_1'),
                # LOCATION_INFO(eleTag=pier_2_ele[-2], integ=2, location=f'col_2_{pier_2_ele[-2]}_2'),
                # LOCATION_INFO(eleTag=pier_2_ele[-2], integ=3, location=f'col_2_{pier_2_ele[-2]}_3'),
                # LOCATION_INFO(eleTag=pier_2_ele[-2], integ=4, location=f'col_2_{pier_2_ele[-2]}_4'),
                # LOCATION_INFO(eleTag=pier_2_ele[-2], integ=5, location=f'col_2_{pier_2_ele[-2]}_5'),
                ),
            OtherOptional=None
            )
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 打印完成信息
        color = random_color()
        rich.print(f'[bold {color}] DONE: [/bold {color}] Create {my_name} !')

        return self.model_props

    def determine_damage(self, odb_tag: Union[str, int] ,info: bool):
        # 导入数据
        ODB_ele_sec = opst.post.get_element_responses(odb_tag=odb_tag, ele_type="FiberSection")

        # 逐个判断 关键部位损伤
        df_list = []
        for loc in self.model_props.LocationDamage:
            tool = DamageStateTools(resp_data=ODB_ele_sec, ele_tag=loc.eleTag, integ=loc.integ)
            df = tool.determine_sec(mat_props=self.model_props.SectionMat, dupe=True, info=info)
            df['location'] = loc.location
            df_list.append(df)
        # 合并
        PierModelDS = pd.concat(df_list, ignore_index=False)

        # 整体结构判断
        StructuralDS = DamageStateTools.determine_struc(PierModelDS, info=True)
        
        return StructuralDS
    
    def reasp_fiber_sec(self, odb_tag: Union[str, int], ele_tag: int, integ: int, step: int):
        # 导入数据
        ODB_ele_sec = opst.post.get_element_responses(odb_tag=odb_tag, ele_type="FiberSection")
        # 显示纤维应变云图
        ys = ODB_ele_sec['ys'].sel(eleTags = ele_tag, secPoints = integ)
        zs = ODB_ele_sec['zs'].sel(eleTags = ele_tag, secPoints = integ)
        Strains = ODB_ele_sec['Strains'].sel(eleTags = ele_tag, secPoints = integ).isel(time = step)
        Stresses = ODB_ele_sec['Stresses'].sel(eleTags = ele_tag, secPoints = integ).isel(time = step)
        # 检索0
        break_mask = (Stresses == 0)
        # break_mask = (Strains == 0)
        
        # 绘图
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 4))
        s = ax.scatter(
            ys.where(~break_mask, drop=True), zs.where(~break_mask, drop=True), # drop=True 去掉不满足的数据点
            c=Stresses.where(~break_mask, drop=True),
            s=5,
            cmap="rainbow",
            zorder=2,
        )
        ax.scatter(
            ys.where(break_mask, drop=True), zs.where(break_mask, drop=True),
            c='#BDBDBD',
            s=5,
            label="Broken",
            zorder=1,
        )
        # 坐标轴标签
        ax.set_xlabel('local_y')
        ax.set_ylabel('local_z')
        
        # 数据单位 1:1
        # ax.set_aspect('equal', adjustable='box')
        ax.set_aspect('equal', adjustable='datalim') # 保持 figsize
        
        # colorbar 
        cbar = fig.colorbar(s, ax=ax, pad=0.02)

        # 单独 legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower right',
            bbox_to_anchor=(0.92, 0.01),  # 调整这个值可以移动 legend
            framealpha=0.9)

        return fig
    
    def reasp_disp(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_node_disp_resp = opst.post.get_nodal_responses(odb_tag=odb_tag, resp_type='disp')
        # 墩顶位移数据
        disp_1 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_top'], DOFs='UY') # pier 1 墩顶位移
        disp_2 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_2_top'], DOFs='UY') # pier 2 墩顶位移
        # 比较位移，取大值
        disp = disp_1 if np.max(np.abs(disp_1)) > np.max(np.abs(disp_2)) else disp_2
        
        return disp
    
    def reasp_base_force(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_node_react_resp = opst.post.get_nodal_responses(odb_tag=odb_tag, resp_type='reaction')
        # 墩底响应数据
        react_1 = ODB_node_react_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_base'], DOFs='UY') # pier 1 墩底反力
        react_2 = ODB_node_react_resp.sel(nodeTags=self.model_props.KeyNode['pier_2_base'], DOFs='UY') # pier 2 墩底反力
        # 比较反力，取大值
        react = react_1 if np.max(np.abs(react_1)) > np.max(np.abs(react_2)) else react_2
        
        return react


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    pass
    
