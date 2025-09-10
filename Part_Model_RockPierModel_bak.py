#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Part_Model_RockPierModel.py
@Date    ：2025/8/1 20:18
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""


from collections import namedtuple
from encodings.punycode import T
import os
import sys
import time
import warnings
from typing import Literal, TypeAlias, Union, Callable, Any, Optional, TypedDict
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
from script import UNIT, PVs, ModelCreateTools

from Part_MatSec_RockPierModel import RockPierModelSection

from script.post import DamageStateTools

from typing import NamedTuple


"""
# --------------------------------------------------
# ========== < Part_Model_RockPierModel > ==========
# --------------------------------------------------
"""

class RockPierModelTEST:
    
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

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def RockPier(
        self,
        modelPath: str,
        Ke: float,
        info: bool
        ) -> PVs.MODEL_PROPS:

        """
        使用 @staticmethod 静态方法，可嵌合在类中，也可独立在外
        双柱式自复位桥墩，基于某三跨双柱式桥墩的 1/4 缩尺
        荷载工况适用于： 推覆分析 & 拟静力分析
        参数：
            Ke: 拟合刚度
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取函数名
        if sys._getframe() is not None:
            my_name = sys._getframe().f_code.co_name
        else:
            raise RuntimeError("Get Model Name Error")
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义模型空间
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        # 实例化建模工具
        MCTs = ModelCreateTools()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 桥墩尺寸控制
        L = 1750 * UNIT.mm # 盖梁长
        PierH = 2400 * UNIT.mm # 墩柱高
        Seg = 1200 * UNIT.mm # 节段长
        PierW = 1200 * UNIT.mm # 墩柱中心间距

        ED_l = 100 * UNIT.mm # 耗能钢筋管道长 /无粘结端 100 mm

        # 模型收敛刚度拟合
        Ke = Ke

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁截面 盖梁材料 编号
        bent_cap_section_tags = {
            'section_tag': 100,  # 截面
            'cover_tag': 1,  # 材料-保护层
            'core_tag': 2,  # 材料-核心
            'bar_tag': 3,  # 材料-钢筋
            'bar_max_tag': 4,  # 材料-钢筋最大应变限制
            'info': info

        }
        BentCapProps = RockPierModelSection.bent_cap_sec(modelPath, **bent_cap_section_tags)  # 创建盖梁纤维截面，并获取截面参数

        # 墩柱截面 墩柱材料 编号
        pier_section_tags = {
            'section_tag': 200,  # 截面
            'cover_tag': 5,  # 材料-保护层
            'core_tag': 6,  # 材料-核心
            'bar_tag': 7,  # 材料-钢筋
            'bar_max_tag': 8,  # 材料-钢筋最大应变限制
            'info': info

        }
        PierProps = RockPierModelSection.pier_sec(modelPath, **pier_section_tags)  # 创建墩柱纤维截面，并获取截面参数

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元积分点
        tag_np_beam = 1
        tag_np_pier = 2
        # ops.beamIntegration('Lobatto', tag_np_beam, bent_cap_section_tags['section_tag'], 5)
        # ops.beamIntegration('Lobatto', tag_np_pier, pier_section_tags['section_tag'], 5)
        ops.beamIntegration('Legendre', tag_np_beam, bent_cap_section_tags['section_tag'], 5)
        ops.beamIntegration('Legendre', tag_np_pier, pier_section_tags['section_tag'], 5)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁质心
        lw_bent_cap = BentCapProps.SecMashProps.centroid[0]
        lh_bent_cap = BentCapProps.SecMashProps.centroid[1]
        
        # 墩柱质心
        lw_pier = PierProps.SecMashProps.centroid[0]
        lh_pier = PierProps.SecMashProps.centroid[1]

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        def contact_surface(
            center_node: int,
            edge_node_1: int,
            edge_node_2: int,
            edge_ele_1: int,
            edge_ele_2: int,
            rigid: bool = True
            ):
            """
            # 三节点接触面
            # 横线为Y，竖向为Z
            # X方向不做拓展
            输入：接触面中心节点编号，边缘节点编号，边缘单元编号
            返回：None
            """
            # 获取中心节点坐标
            center_coord = ops.nodeCoord(center_node)
            
            # 节段边缘节点，在Y上拓展接触面节点
            ops.node(edge_node_1, center_coord[0], center_coord[1] - lw_pier, center_coord[2])
            ops.node(edge_node_2, center_coord[0], center_coord[1] + lw_pier, center_coord[2])
            
            # 边缘节点弹性单元坐标转换
            seg_link_Transf = MCTs.auto_geomTransf(edge_node_1, center_node)
            # 边缘节点单元连接
            if rigid:
                ops.rigidLink('beam', center_node, edge_node_1)
                ops.rigidLink('beam', center_node, edge_node_2)
            else:
                ops.element('elasticBeamColumn', edge_ele_1,
                            *(edge_node_1, center_node),
                            PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                            0.1, Ke, Ke, Ke, seg_link_Transf
                            )
                ops.element('elasticBeamColumn', edge_ele_2,
                            *(edge_node_2, center_node),
                            PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                            0.1, Ke, Ke, Ke, seg_link_Transf
                            )
            

        class SegmentReturn(TypedDict):
            # 节段顶部接触面节点号
            edge_node_1: int
            edge_node_top: int
            edge_node_2: int
            # 节段底部接触面节点号
            edge_node_3: int
            edge_node_base: int
            edge_node_4: int
            # 两端主体节点单元
            main_node: list
            main_ele: list

        # 节段函数
        def segment(
            node_start: int,
            ele_start: int,
            start_coord: Union[tuple[float, float, float], list[float]],
            end_coord: Union[tuple[float, float, float], list[float]],
            rigid_top: bool = True,
            rigid_base: bool = True
        ) -> SegmentReturn:
            """
            横线为Y，竖向为Z
            X方向不做拓展
            # 节段建模方向由下到上
                -       +
                3   c   4
                 *——*——*
                    |
                    *
                    |
                    *
                    |
                 *——*——*
                1   c   2
                -       +
            输入：
                节点编号组
                单元编号组
                起始点中心坐标
                结束点中心坐标
            返回：
                接触面三个节点的编号
                两端主体单元号
            """
            
            seg_node_n = 5 # 节段主体节点数（包含两侧）
            
            # 节段中心节点
            seg_center_coord = NodeTools.distribute(
                start_coord, end_coord,
                seg_node_n,
                ends=(True, True)
                ) # 节段中心节点坐标
            seg_center_node = MCTs.node_create(node_start, seg_center_coord) # 创建节点中节点
            
            # 中心节点单元连接
            seg_center_node_links = []
            for node_i, node_j in pairwise(seg_center_node):
                seg_center_node_links.append((node_i, node_j))

            # 创建单元
            seg_center_ele = MCTs.ele_create(ele_start, seg_center_node_links, tag_np_pier)

            "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
            # 节段边缘节点
            contact_surface(
                center_node=seg_center_node[0],
                edge_node_1=seg_center_node[-1] + 1, edge_node_2=seg_center_node[-1] + 2,
                edge_ele_1=seg_center_ele[-1] + 1, edge_ele_2=seg_center_ele[-1] + 2,
                rigid=rigid_base
                )
            contact_surface(
                center_node=seg_center_node[-1],
                edge_node_1=seg_center_node[-1] + 3, edge_node_2=seg_center_node[-1] + 4,
                edge_ele_1=seg_center_ele[-1] + 3, edge_ele_2=seg_center_ele[-1] + 4,
                rigid=rigid_top
                )

            "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
            # 节段返回数据结构
            seg_key: SegmentReturn = {
                'edge_node_1': seg_center_node[-1] + 1,
                'edge_node_base': seg_center_node[0],
                'edge_node_2': seg_center_node[-1] + 2,
                
                'edge_node_3': seg_center_node[-1] + 3,
                'edge_node_top': seg_center_node[-1],
                'edge_node_4': seg_center_node[-1] + 4,
                
                'main_node': seg_center_node,
                'main_ele': seg_center_ele,
                }

            return seg_key

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 墩柱
        seg_1 = segment(
            node_start=1100, ele_start=1100,
            start_coord=(0., -PierW / 2., 0.), end_coord=(0., -PierW / 2., PierH / 2.),
            rigid_top=False, rigid_base=False
            )
        seg_2 = segment(
            node_start=1200, ele_start=1200,
            start_coord=(0., PierW / 2., 0.), end_coord=(0., PierW / 2., PierH / 2.),
            rigid_top=False, rigid_base=False
            )
        seg_3 = segment(
            node_start=1300, ele_start=1300,
            start_coord=(0., -PierW / 2., PierH / 2.), end_coord=(0., -PierW / 2., PierH),
            rigid_top=False, rigid_base=False
            )
        seg_4 = segment(
            node_start=1400, ele_start=1400,
            start_coord=(0., PierW / 2., PierH / 2.), end_coord=(0., PierW / 2., PierH),
            rigid_top=False, rigid_base=False
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁节点
        bent_cap_coord = [
            (0., -L / 2., PierH + lh_bent_cap),
            (0., -PierW / 2., PierH + lh_bent_cap),
            (0., -PierW / 4., PierH + lh_bent_cap),
            
            (0., 0., PierH + lh_bent_cap),
            
            (0., PierW / 4., PierH + lh_bent_cap),
            (0., PierW / 2., PierH + lh_bent_cap),
            (0., L / 2., PierH + lh_bent_cap),
            ]
        bent_cap_node = MCTs.node_create(2000, bent_cap_coord) # 创建盖梁节点
        # 盖梁单元连接
        bent_cap_node_links = []
        for node_i, node_j in pairwise(bent_cap_node):
            bent_cap_node_links.append((node_i, node_j))
        # 盖梁单元
        bent_cap_ele = MCTs.ele_create(2000, bent_cap_node_links, tag_np_beam)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 墩柱 底部 接触面 1
        contact_node_1 = 3100
        contact_ele_1 = 3100
        ops.node(contact_node_1 + 1, 0., -PierW / 2, 0.) # 接触面中心节点
        ops.node(contact_node_1 + 2, 0., -PierW / 2 - lw_pier, 0.) # 接触面中心节点
        ops.node(contact_node_1 + 3, 0., -PierW / 2 + lw_pier, 0.) # 接触面中心节点
        ops.fix(contact_node_1 + 1, 1, 1, 1, 1, 1, 1)
        ops.fix(contact_node_1 + 2, 1, 1, 1, 1, 1, 1)
        ops.fix(contact_node_1 + 3, 1, 1, 1, 1, 1, 1)
        
        # 墩柱 底部 接触面 2
        contact_node_2 = 3200
        contact_ele_2 = 3200
        ops.node(contact_node_2 + 1, 0., PierW / 2., 0.) # 接触面中心节点
        ops.node(contact_node_2 + 2, 0., PierW / 2. - lw_pier, 0.) # 接触面中心节点
        ops.node(contact_node_2 + 3, 0., PierW / 2. + lw_pier, 0.) # 接触面中心节点
        ops.fix(contact_node_2 + 1, 1, 1, 1, 1, 1, 1)
        ops.fix(contact_node_2 + 2, 1, 1, 1, 1, 1, 1)
        ops.fix(contact_node_2 + 3, 1, 1, 1, 1, 1, 1)

        # 墩柱 顶部 接触面 3
        contact_node_3 = 3300
        contact_ele_3 = 3300
        ops.node(contact_node_3 + 1, 0., -PierW / 2., PierH) # 接触面中心节点
        contact_surface(
            center_node=contact_node_3 + 1,
            edge_node_1=contact_node_3 + 2, edge_node_2=contact_node_3 + 3,
            edge_ele_1=contact_ele_3 + 1, edge_ele_2=contact_ele_3 + 2,
            rigid=False
            )
        
        # 墩柱 顶部 接触面 4
        contact_node_4 = 3400
        contact_ele_4 = 3400
        ops.node(contact_node_4 + 1, 0., PierW / 2., PierH) # 接触面中心节点
        contact_surface(
            center_node=contact_node_4 + 1,
            edge_node_1=contact_node_4 + 2, edge_node_2=contact_node_4 + 3,
            edge_ele_1=contact_ele_4 + 1, edge_ele_2=contact_ele_4 + 2,
            rigid=False
            )
        
        # 盖梁接触面连接
        ops.rigidLink('beam', bent_cap_node[1], contact_node_3 + 1)
        ops.rigidLink('beam', bent_cap_node[-2], contact_node_4 + 1)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        Ubig = 1.e6
        Usmall = 1.e-6
        # 接触面材料
        matENT = 10
        ops.uniaxialMaterial('ENT', matENT, 30 * UNIT.gpa) # 无受拉弹性材料
        # 自由度 材料标签
        K_free = 11
        K_fix = 12
        ops.uniaxialMaterial('Elastic', K_free, Usmall)  # 弹性材料 /释放变形
        ops.uniaxialMaterial('Elastic', K_fix, Ubig)  # 弹性材料 /限制变形

        # 材料 对应 自由度
        dir_mats = [matENT, K_fix, K_fix, K_free, K_free, K_free]  # 零长单元局部方向
        dirs = [1, 2, 3, 4, 5, 6]

        # 竖向 零长单元 坐标转换
        vecx = [0, 0, 1]  # 局部x -> 整体坐标
        vecyp = [0, 1, 0]  # 局部y -> 整体坐标
        
        # pier 1 接触面单元
        contact_ele_start_1 = 4100
        ops.element('zeroLength', contact_ele_start_1 + 2, *[contact_node_1 + 2, seg_1['edge_node_1']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 底部
        ops.element('zeroLength', contact_ele_start_1 + 1, *[contact_node_1 + 1, seg_1['edge_node_base']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 底部
        ops.element('zeroLength', contact_ele_start_1 + 3, *[contact_node_1 + 3, seg_1['edge_node_2']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 底部

        ops.element('zeroLength', contact_ele_start_1 + 5, *[seg_1['edge_node_3'], seg_3['edge_node_1']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 中间
        ops.element('zeroLength', contact_ele_start_1 + 4, *[seg_1['edge_node_top'], seg_3['edge_node_base']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 中间
        ops.element('zeroLength', contact_ele_start_1 + 6, *[seg_1['edge_node_4'], seg_3['edge_node_2']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 中间

        ops.element('zeroLength', contact_ele_start_1 + 8, *[seg_3['edge_node_3'], contact_node_3 + 2],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 顶部
        ops.element('zeroLength', contact_ele_start_1 + 7, *[seg_3['edge_node_top'], contact_node_3 + 1],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 顶部
        ops.element('zeroLength', contact_ele_start_1 + 9, *[seg_3['edge_node_4'], contact_node_3 + 3],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 顶部

        # pier 2 接触面单元
        contact_ele_start_2 = 4200
        ops.element('zeroLength', contact_ele_start_2 + 2, *[contact_node_2 + 2, seg_2['edge_node_1']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 底部
        ops.element('zeroLength', contact_ele_start_2 + 1, *[contact_node_2 + 1, seg_2['edge_node_base']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 底部
        ops.element('zeroLength', contact_ele_start_2 + 3, *[contact_node_2 + 3, seg_2['edge_node_2']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 底部

        ops.element('zeroLength', contact_ele_start_2 + 5, *[seg_2['edge_node_3'], seg_4['edge_node_1']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 中间
        ops.element('zeroLength', contact_ele_start_2 + 4, *[seg_2['edge_node_top'], seg_4['edge_node_base']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 中间
        ops.element('zeroLength', contact_ele_start_2 + 6, *[seg_2['edge_node_4'], seg_4['edge_node_2']],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 中间

        ops.element('zeroLength', contact_ele_start_2 + 8, *[seg_4['edge_node_3'], contact_node_4 + 2],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 顶部
        ops.element('zeroLength', contact_ele_start_2 + 7, *[seg_4['edge_node_top'], contact_node_4 + 1],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 顶部
        ops.element('zeroLength', contact_ele_start_2 + 9, *[seg_4['edge_node_4'], contact_node_4 + 3],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # 顶部

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 耗能钢筋材料
        ED_bar = 20 # 耗能钢筋材料编号
        ED_fy = 437.3 * UNIT.mpa
        ED_Es = 197 * UNIT.gpa
        ED_area = np.pi * (6 * UNIT.mm) ** 2 # 耗能钢筋面积
        ops.uniaxialMaterial('Steel02', ED_bar, ED_fy, ED_Es, 0.01, 18, 0.925, 0.15)
        
        # pier 1 耗能钢筋
        ED_node_1 = 5100
        # 粘结端
        ops.node(ED_node_1 + 1, 0., -PierW / 2. - lw_pier / 2, -ED_l)
        ops.node(ED_node_1 + 2, 0., -PierW / 2. + lw_pier / 2, -ED_l)
        ops.fix(ED_node_1 + 1, 1, 1, 1, 1, 1, 1)
        ops.fix(ED_node_1 + 2, 1, 1, 1, 1, 1, 1)
        # 节段端
        ops.node(ED_node_1 + 3, 0., -PierW / 2. - lw_pier / 2, 0.)
        ops.node(ED_node_1 + 4, 0., -PierW / 2. + lw_pier / 2, 0.)
        ops.rigidLink('beam', seg_1['edge_node_base'], ED_node_1 + 3)
        ops.rigidLink('beam', seg_1['edge_node_base'], ED_node_1 + 4)
        # 耗能钢筋
        ED_ele_1 = 5100
        ops.element('Truss', ED_ele_1 + 1, *[ED_node_1 + 1, ED_node_1 + 3], ED_area, ED_bar)
        ops.element('Truss', ED_ele_1 + 2, *[ED_node_1 + 2, ED_node_1 + 4], ED_area, ED_bar)

        # pier 2 耗能钢筋
        ED_node_2 = 5200
        # 粘结端
        ops.node(ED_node_2 + 1, 0., PierW / 2. - lw_pier / 2, -ED_l)
        ops.node(ED_node_2 + 2, 0., PierW / 2. + lw_pier / 2, -ED_l)
        ops.fix(ED_node_2 + 1, 1, 1, 1, 1, 1, 1)
        ops.fix(ED_node_2 + 2, 1, 1, 1, 1, 1, 1)
        # 节段端
        ops.node(ED_node_2 + 3, 0., PierW / 2. - lw_pier / 2, 0.)
        ops.node(ED_node_2 + 4, 0., PierW / 2. + lw_pier / 2, 0.)
        ops.rigidLink('beam', seg_2['edge_node_base'], ED_node_2 + 3)
        ops.rigidLink('beam', seg_2['edge_node_base'], ED_node_2 + 4)
        # 耗能钢筋
        ED_ele_2 = 5200
        ops.element('Truss', ED_ele_2 + 1, *[ED_node_2 + 1, ED_node_2 + 3], ED_area, ED_bar)
        ops.element('Truss', ED_ele_2 + 2, *[ED_node_2 + 2, ED_node_2 + 4], ED_area, ED_bar)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 预应力固定节点
        PT_fix_node = 6000
        # 底部
        ops.node(PT_fix_node + 1, 0., -PierW / 2., -lh_bent_cap * 2.)
        ops.node(PT_fix_node + 2, 0., PierW / 2., -lh_bent_cap * 2.)
        # 底部固定
        ops.fix(PT_fix_node + 1, 1, 1, 1, 1, 1, 1)
        ops.fix(PT_fix_node + 2, 1, 1, 1, 1, 1, 1)

        # 顶部
        ops.node(PT_fix_node + 3, 0., -PierW / 2., PierH + lh_bent_cap * 2.)
        ops.node(PT_fix_node + 4, 0., PierW / 2., PierH + lh_bent_cap * 2.)
        # 连接盖梁
        ops.rigidLink('beam', bent_cap_node[1], PT_fix_node + 3)
        ops.rigidLink('beam', bent_cap_node[-2], PT_fix_node + 4)

        
        # 张拉控制力
        axial_force = 300. * UNIT.kn
        # 钢绞线总面积
        PT_area = 3 * (np.pi * (15.2 * UNIT.mm / 2) ** 2)
        # 张拉控制应力
        sigma = axial_force / PT_area

        # 预应力筋 材料
        PT_mat = 30 # 材料标签
        PT_fy = 1906 * UNIT.mpa
        PT_Es = 200 * UNIT.gpa
        PT_ratio = 0.43 * PT_fy # 控制张拉比例
        ops.uniaxialMaterial('Steel02', PT_mat, PT_fy, PT_Es, 0.01, 18, 0.925, 0.15, 0, 1, 0, 1, sigma)

        # 预应力纤维
        # PT_sec = 300
        # ops.section('fiberSec', 300, '-GJ', 100000000)
        # ops.fiber(0., 0., PT_area, PT_mat)

        # 预应力筋单元
        ops.element('Truss', 100, *(PT_fix_node + 1, PT_fix_node + 3), PT_area, PT_mat)
        ops.element('Truss', 200, *(PT_fix_node + 2, PT_fix_node + 4), PT_area, PT_mat)
        # ops.element('Truss', 100, *(PT_fix_node + 1, PT_fix_node + 3), PT_sec)
        # ops.element('Truss', 200, *(PT_fix_node + 2, PT_fix_node + 4), PT_sec)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 质量密度
        rho = 2600 * (UNIT.kg / (UNIT.m ** 3))  # kg/m3

        # 单位节点质量
        bent_cap_mass = L * BentCapProps.SecMashProps.A * rho / len(bent_cap_node)
        seg_1_mass = Seg * PierProps.SecMashProps.A * rho / len(seg_1['main_node'])
        seg_2_mass = Seg * PierProps.SecMashProps.A * rho / len(seg_2['main_node'])
        seg_3_mass = Seg * PierProps.SecMashProps.A * rho / len(seg_3['main_node'])
        seg_4_mass = Seg * PierProps.SecMashProps.A * rho / len(seg_4['main_node'])

        # 节点质量
        for i in bent_cap_node:
            ops.mass(i, bent_cap_mass, bent_cap_mass, bent_cap_mass, 0, 0, 0)
        for i in seg_1['main_node']:
            ops.mass(i, seg_1_mass, seg_1_mass, seg_1_mass, 0, 0, 0)
        for i in seg_2['main_node']:
            ops.mass(i, seg_2_mass, seg_2_mass, seg_2_mass, 0, 0, 0)
        for i in seg_3['main_node']:
            ops.mass(i, seg_3_mass, seg_3_mass, seg_3_mass, 0, 0, 0)
        for i in seg_4['main_node']:
            ops.mass(i, seg_4_mass, seg_4_mass, seg_4_mass, 0, 0, 0)


        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义模型不同部位名称
        LOCATION_INFO = namedtuple('LOCATION_INFO', ['eleTag', 'integ', 'location'])
        # 返回数据
        self.model_props = PVs.MODEL_PROPS(
            Name=my_name,
            SectionMat={
                'BentCapProps': BentCapProps,
                'PierProps': PierProps,
                },
            KeyNode=bent_cap_node[0],
            KeyEle={
                'Pier_1_PT_bar': 100,
                'Pier_1_PT_bar': 200,
                },
            LocationDamage=[
                LOCATION_INFO(eleTag=seg_1['main_ele'][0], integ=1, location='seg_1_base'),
                LOCATION_INFO(eleTag=seg_1['main_ele'][-1], integ=5, location='seg_1_top'),

                LOCATION_INFO(eleTag=seg_2['main_ele'][0], integ=1, location='seg_2_base'),
                LOCATION_INFO(eleTag=seg_2['main_ele'][-1], integ=5, location='seg_2_top'),
                
                LOCATION_INFO(eleTag=seg_3['main_ele'][0], integ=1, location='seg_3_base'),
                LOCATION_INFO(eleTag=seg_3['main_ele'][-1], integ=5, location='seg_3_top'),

                LOCATION_INFO(eleTag=seg_4['main_ele'][0], integ=1, location='seg_4_base'),
                LOCATION_INFO(eleTag=seg_4['main_ele'][-1], integ=5, location='seg_4_top'),
                ],
            OtherOptional={
                'Ke': 1,
                'rock_params': 1,
            }
        )

        return self.model_props

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def determine_damage(self, odb_tag: Union[str, int] ,info: bool):

        # 导入数据
        ODB_ele_sec = opst.post.get_element_responses(odb_tag=odb_tag, ele_type="FiberSection")

        # 逐个判断
        df_list = []
        for loc in self.model_props.LocationDamage:
            tool = DamageStateTools(resp_data=ODB_ele_sec, ele_tag=loc.eleTag, integ=loc.integ)
            df = tool.determine_sec(mat_props=self.model_props.SectionMat['PierProps'], dupe=True, info=info)
            df['location'] = loc.location
            df_list.append(df)
        # 合并
        PierModelDS = pd.concat(df_list, ignore_index=False)

        # 整体结构判断
        StructuralDS = DamageStateTools.determine_struc(PierModelDS, info=True)
        
        return StructuralDS

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
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
            s=20,
            cmap="rainbow",
            zorder=2,
        )
        ax.scatter(
            ys.where(break_mask, drop=True), zs.where(break_mask, drop=True),
            c='#BDBDBD',
            s=20,
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
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def reasp_disp(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_node_disp_resp = opst.post.get_nodal_responses(odb_tag=odb_tag, resp_type='disp')
        # 控制节点位移
        disp = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode, DOFs='UY')
        
        return disp
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def reasp_PT_force(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_truss_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type='Truss')
        
        axialForce_1 = ODB_truss_resp['axialForce'].sel(eleTags=100)
        axialForce_2 = ODB_truss_resp['axialForce'].sel(eleTags=200)
        
        disp = np.array(self.reasp_disp(odb_tag))
        
        plt.close('all')
        plt.figure(figsize=(6, 4))
        plt.plot(disp, axialForce_1, label='Pier 1 PT bar', zorder=2)
        plt.plot(disp, axialForce_2, label='Pier 2 PT bar', zorder=2)
        plt.xlabel('Displacement (m)')
        plt.ylabel('PT bar Axial Force (kN)')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.grid(linestyle='--', linewidth=0.5, zorder=1)
        
        return plt

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    
    test_path = './OutTest'
    os.makedirs(test_path, exist_ok=True)
    
    test_model = RockPierModelTEST()
    test_model.RockPier(modelPath=test_path, Ke=1.5, info=True)
    
    opsplt.set_plot_props(point_size=5, line_width=3)
    fig = opsplt.plot_model(
        show_node_numbering=False,
        show_ele_numbering=False,
        show_local_axes=True
        )
    fig.show(renderer="browser")
