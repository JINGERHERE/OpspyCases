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
from script.base import rich_showwarning

import warnings
warnings.showwarning = rich_showwarning

# from Part_Model_CreateBRB import create_brb_lite

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
    def _set_BRB(
        self,
        yield_disp: float,
        yield_force: float,
        node_i: int, node_j: int,
        core_ratio: float, core_area: float,
        link_E: float = 1.e6,
        groupTag: int = 1,
        ):
        
        """
        yield_disp: 桥墩屈服位移
        yield_force: 桥墩屈服力
        alpha: 桥墩刚度 占 整体刚度的比例
        miu: 桥墩屈服位移 / BRB屈服位移
        """
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 夹角计算函数
        def calculate_angle_with_horizontal(point1, point2):
            """
            计算两点连线与水平线的夹角
            point1, point2: 两个点的坐标列表，如 [x, y, z]
            返回：角度值（度）
            """
            p1, p2 = np.array(point1), np.array(point2)
            # 计算水平距离和高度差
            horizontal_dist = np.linalg.norm(p2[:2] - p1[:2])  # 只取xy平面
            dz = p2[2] - p1[2]
            
            # return np.degrees(np.arctan2(dz, horizontal_dist)) # 计算并返回角度（度）
            return np.arctan2(dz, horizontal_dist) # 计算并返回角度（弧度）

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # BRB 材料数据
        Q235_fy = 235 * UNIT.mpa # Q235屈服强度
        Q235_Es = 206 * UNIT.gpa # Q235弹性模量
        Q235_eps_y = Q235_fy / Q235_Es # Q235屈服应变
        
        # 定义 BRB 材料
        BRBmat = groupTag + 1
        ops.uniaxialMaterial('Steel02', BRBmat, Q235_fy, Q235_Es, 0.01) # BRB材料

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取两端连接节点坐标
        i_coord = np.array(ops.nodeCoord(node_i))
        j_coord = np.array(ops.nodeCoord(node_j))
        
        # 计算距离
        L = np.linalg.norm(j_coord - i_coord) # 固定端总长度
        # 计算直线参数
        center_point = (i_coord + j_coord) / 2 # 直线中心坐标
        dir_vector = (j_coord - i_coord) / L # 直线方向向量
        
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 相关参数
        H = 2400 * UNIT.mm # 柱高
        nb = 2 # 沿 柱高方向 BRB个数
        # 桥墩刚度
        Kc = yield_force / yield_disp
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 夹角
        theta = abs(
            calculate_angle_with_horizontal(i_coord, j_coord)
            )
        
        # BRB 核心段 长度
        core_len = core_ratio * L
        # BRB 屈服位移
        hy = core_len * np.sin(theta) # BRB 核心长度 在竖直方向投影
        delta_yb = Q235_eps_y * core_len * H / (hy * np.cos(theta))  # BRB 屈服位移
        
        # 连接段 在水平线上的投影
        Lbx = (L - core_len) / 2 * np.cos(theta)
        
        # BRB 强度贡献
        V_yb = 2 * nb * Q235_fy * Lbx * core_area * np.sin(theta)
        
        # BRB 刚度 贡献
        Kb = V_yb / delta_yb

        # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # print(f'# 桥墩位移 / BRB位移：{yield_disp / delta_yb}')
        # print(f'# 桥墩刚度 / BRB刚度：{Kc / Kb}')

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 缩放后的端点坐标
        core_i_coord = center_point - (dir_vector * core_len / 2)
        core_j_coord = center_point + (dir_vector * core_len / 2)
        
        # 节点标签
        core_node_tag_i = groupTag + 1 # 核心 I 端 节点标签
        core_node_tag_j = groupTag + 2 # 核心 J 端 节点标签

        # 节点
        ops.node(core_node_tag_i, *core_i_coord) # 核心 I 端
        ops.node(core_node_tag_j, *core_j_coord) # 核心 J 端

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元 标签
        link_ele_tag_i = groupTag + 1 # I 端 连接段 单元标签
        link_ele_tag_j = groupTag + 2 # J 端 连接段 单元标签
        core_ele_tag = groupTag + 3 # 核心 单元 标签

        # 单元
        ops.element('elasticBeamColumn', link_ele_tag_i,
                    *(node_i, core_node_tag_i),
                    200 * UNIT.gpa, 100 * UNIT.gpa,
                    0.1, link_E, link_E, link_E, self.MCTs.geomTransf_other())

        ops.element('Truss', core_ele_tag, *[core_node_tag_i, core_node_tag_j], core_area, BRBmat)

        ops.element('elasticBeamColumn', link_ele_tag_j,
                    *(core_node_tag_j, node_j),
                    200 * UNIT.gpa, 100 * UNIT.gpa,
                    0.1, link_E, link_E, link_E, self.MCTs.geomTransf_other())

        # 返回：BRB 单元号、BRB 屈服应变、BRB 屈服位移、BRB 屈服力、BRB 刚度
        return core_ele_tag, Q235_eps_y, delta_yb, V_yb, Kb

    def _set_pier(
        self,
        modelPath: str,
        Ke: float,
        info: bool
        ):

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 桥墩尺寸控制
        L = 1950. * UNIT.mm # 盖梁长
        PierH = 2400. * UNIT.mm # 墩柱高
        Seg = 1200. * UNIT.mm # 节段长
        PierW = 1200. * UNIT.mm # 墩柱中心间距

        # 耗能钢筋管道长
        ED_l = 100. * UNIT.mm # 无粘结段 100 mm

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
        # 接触面材料
        matENT = 10
        # ops.uniaxialMaterial('ENT', matENT, 30 * UNIT.gpa) # 无受拉弹性材料
        # ops.uniaxialMaterial('ENT', matENT, 892.9e3) # 无受拉弹性材料
        ops.uniaxialMaterial('ENT', matENT, 20 * UNIT.gpa) # 无受拉弹性材料

        # 接触面纤维截面
        sec_ENT = 300
        ops.section('fiberSec', sec_ENT, '-GJ', 57786.97)
        ops.patch('rect', matENT, 20, 20, *[-lw_pier, -lh_pier], *[lw_pier, lh_pier])

        # 大刚度 小刚度
        Ubig = 1.e6
        Usmall = 1.e-6
        # 自由度 材料标签
        K_free = 11
        K_fix = 12
        K_ke = 13
        ops.uniaxialMaterial('Elastic', K_free, Usmall)  # 弹性材料 /释放变形
        ops.uniaxialMaterial('Elastic', K_fix, Ubig)  # 弹性材料 /限制变形
        ops.uniaxialMaterial('Elastic', K_ke, 1.5e3)
    
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 耗能钢筋材料
        ED_bar = 20 # 耗能钢筋材料编号
        ED_fy = 437.3 * UNIT.mpa
        ED_Es = 201 * UNIT.gpa
        ED_area = np.pi * (6 * UNIT.mm) ** 2 # 耗能钢筋面积
        ops.uniaxialMaterial('Steel02', ED_bar, ED_fy, ED_Es, 0.01, 18, 0.925, 0.15)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 1 柱 - 上节段 节点坐标
        pier_1_top_seg_coord = [
            # (0., -PierW / 2, PierH - Seg * 0 - Seg * (0/4)-0.1), # 顶部节点
            (0., -PierW / 2, PierH - Seg * 0 - Seg * (0/4)), # 顶部节点
            (0., -PierW / 2, PierH - Seg * 0 - Seg * (0/4) - lw_pier), # BRB 柱中 节点
            (0., -PierW / 2, PierH - Seg * 0 - Seg * (1/4)),
            (0., -PierW / 2, PierH - Seg * 0 - Seg * (2/4)),
            (0., -PierW / 2, PierH - Seg * 0 - Seg * (3/4)),
            (0., -PierW / 2, PierH - Seg * 0 - Seg * (4/4)), # 底部节点
            # (0., -PierW / 2, PierH - Seg * 0 - Seg * (4/4)+0.1), # 底部节点
            ]
        # 1 柱 - 下节段 节点坐标
        pier_1_base_seg_coord = [
            # (0., -PierW / 2, PierH - Seg * 1 - Seg * (0/4)-0.1), # 顶部节点
            (0., -PierW / 2, PierH - Seg * 1 - Seg * (0/4)), # 顶部节点
            (0., -PierW / 2, PierH - Seg * 1 - Seg * (1/4)),
            (0., -PierW / 2, PierH - Seg * 1 - Seg * (2/4)),
            (0., -PierW / 2, PierH - Seg * 1 - Seg * (3/4)),
            (0., -PierW / 2, PierH - Seg * 1 - Seg * (4/4) + lw_pier), # BRB 柱中 节点
            (0., -PierW / 2, PierH - Seg * 1 - Seg * (4/4)), # 底部节点
            # (0., -PierW / 2, PierH - Seg * 1 - Seg * (4/4)+0.1), # 底部节点
            ]
        # 2 柱 - 上节段 节点坐标
        pier_2_top_seg_coord = [
            # (0., PierW / 2, PierH - Seg * 0 - Seg * (0/4)-0.1), # 顶部节点
            (0., PierW / 2, PierH - Seg * 0 - Seg * (0/4)), # 顶部节点
            (0., PierW / 2, PierH - Seg * 0 - Seg * (1/4)),
            (0., PierW / 2, PierH - Seg * 0 - Seg * (2/4)),
            (0., PierW / 2, PierH - Seg * 0 - Seg * (3/4)),
            (0., PierW / 2, PierH - Seg * 0 - Seg * (4/4) + lw_pier), # BRB 柱中 节点
            (0., PierW / 2, PierH - Seg * 0 - Seg * (4/4)), # 底部节点
            # (0., PierW / 2, PierH - Seg * 0 - Seg * (4/4)+0.1), # 底部节点
            ]
        # 2 柱 - 下节段 节点坐标
        pier_2_base_seg_coord = [
            # (0., PierW / 2, PierH - Seg * 1 - Seg * (0/4)-0.1), # 顶部节点
            (0., PierW / 2, PierH - Seg * 1 - Seg * (0/4)), # 顶部节点
            (0., PierW / 2, PierH - Seg * 1 - Seg * (0/4) - lw_pier), # BRB 柱中 节点
            (0., PierW / 2, PierH - Seg * 1 - Seg * (1/4)),
            (0., PierW / 2, PierH - Seg * 1 - Seg * (2/4)),
            (0., PierW / 2, PierH - Seg * 1 - Seg * (3/4)),
            (0., PierW / 2, PierH - Seg * 1 - Seg * (4/4)), # 底部节点
            # (0., PierW / 2, PierH - Seg * 1 - Seg * (4/4)+0.1), # 底部节点
            ]

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 柱 节点
        pier_1_top_seg_node = self.MCTs.node_create(startTag=1100, coords=pier_1_top_seg_coord)
        pier_1_base_seg_node = self.MCTs.node_create(startTag=1200, coords=pier_1_base_seg_coord)
        pier_2_top_seg_node = self.MCTs.node_create(startTag=2100, coords=pier_2_top_seg_coord)
        pier_2_base_seg_node = self.MCTs.node_create(startTag=2200, coords=pier_2_base_seg_coord)

        # 1 柱 - 上节段 单元连接
        pier_1_top_seg_node_links = []
        for node_i, node_j in pairwise(pier_1_top_seg_node):
            pier_1_top_seg_node_links.append((node_i, node_j))
        # 1 柱 - 上节段 单元
        pier_1_top_seg_ele = self.MCTs.ele_create(1100, pier_1_top_seg_node_links, tag_np_pier)

        # 1 柱 - 下节段 单元连接
        pier_1_base_seg_node_links = []
        for node_i, node_j in pairwise(pier_1_base_seg_node):
            pier_1_base_seg_node_links.append((node_i, node_j))
        # 1 柱 - 下节段 单元
        pier_1_base_seg_ele = self.MCTs.ele_create(1200, pier_1_base_seg_node_links, tag_np_pier)

        # 2 柱 - 上节段 单元连接
        pier_2_top_seg_node_links = []
        for node_i, node_j in pairwise(pier_2_top_seg_node):
            pier_2_top_seg_node_links.append((node_i, node_j))
        # 2 柱 - 上节段 单元
        pier_2_top_seg_ele = self.MCTs.ele_create(2100, pier_2_top_seg_node_links, tag_np_pier)

        # 2 柱 - 下节段 单元连接
        pier_2_base_seg_node_links = []
        for node_i, node_j in pairwise(pier_2_base_seg_node):
            pier_2_base_seg_node_links.append((node_i, node_j))
        # 2 柱 - 下节段 单元
        pier_2_base_seg_ele = self.MCTs.ele_create(2200, pier_2_base_seg_node_links, tag_np_pier)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        '''BRB 柱边缘 节点'''
        
        # BRB 1柱 边缘节点
        ops.node(5101, *(0., -PierW / 2 + lw_pier, PierH - Seg * 0 - Seg * (0/4) - lw_pier)) # 1 柱 上节段 顶部
        ops.node(5102, *(0., -PierW / 2 + lw_pier, PierH - Seg * 1 - Seg * (4/4) + lw_pier)) # 1 柱 下节段 底部
        
        # BRB 2柱 边缘节点
        ops.node(5201, *(0., PierW / 2 - lw_pier, PierH - Seg * 0 - Seg * (4/4) + lw_pier)) # 2 柱 上节段 底部
        ops.node(5202, *(0., PierW / 2 - lw_pier, PierH - Seg * 1 - Seg * (0/4) - lw_pier)) # 2 柱 下节段 顶部
        
        # 检索距离最近的节点
        c1_st = self.MCTs.node_closest(nodeTag=5101, indexList=pier_1_top_seg_node, dim=3)
        c1_sb = self.MCTs.node_closest(nodeTag=5102, indexList=pier_1_base_seg_node, dim=3)
        
        c2_sb = self.MCTs.node_closest(nodeTag=5201, indexList=pier_2_top_seg_node, dim=3)
        c2_st = self.MCTs.node_closest(nodeTag=5202, indexList=pier_2_base_seg_node, dim=3)
        
        # 全自由度 耦合
        # 1 柱
        equal_BRB_dof = [1, 2, 3, 4, 5, 6]
        # ops.equalDOF(c1_st, 5101, *equal_BRB_dof)
        # ops.equalDOF(c1_sb, 5102, *equal_BRB_dof)
        ops.rigidLink('beam', c1_st, 5101)
        ops.rigidLink('beam', c1_sb, 5102)
        # 2 柱
        # ops.equalDOF(c2_sb, 5201, *equal_BRB_dof)
        # ops.equalDOF(c2_st, 5202, *equal_BRB_dof)
        ops.rigidLink('beam', c2_sb, 5201)
        ops.rigidLink('beam', c2_st, 5202)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁节点坐标
        bent_cap_coord = [
            (0., -L / 2., PierH + lh_bent_cap),
            (0., -PierW / 2., PierH + lh_bent_cap),
            (0., -PierW / 4., PierH + lh_bent_cap),
            
            (0., 0., PierH + lh_bent_cap),
            
            (0., PierW / 4., PierH + lh_bent_cap),
            (0., PierW / 2., PierH + lh_bent_cap),
            (0., L / 2., PierH + lh_bent_cap),
            ]
        bent_cap_node = self.MCTs.node_create(3000, bent_cap_coord) # 创建盖梁节点
        
        # 盖梁单元连接
        bent_cap_node_links = []
        for node_i, node_j in pairwise(bent_cap_node):
            bent_cap_node_links.append((node_i, node_j))
        # 盖梁单元
        bent_cap_ele = self.MCTs.ele_create(3000, bent_cap_node_links, tag_np_beam)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 接触面节点
        ops.node(1001, 0., -PierW / 2, PierH) # 1 柱 顶
        # ops.rigidLink('beam', 1001, bent_cap_node[1])
        ops.node(1002, 0., -PierW / 2, 0.) # 1 柱 底
        ops.fix(1002, 1, 1, 1, 1, 1, 1)
        
        ops.node (2001, 0., PierW / 2, PierH) # 2 柱 顶
        # ops.rigidLink('beam', 2001, bent_cap_node[-2])
        ops.node(2002, 0., PierW / 2, 0.) # 2 柱 底
        ops.fix(2002, 1, 1, 1, 1, 1, 1)

        # 梁柱连接 - 单元
        ops.element('elasticBeamColumn', 9001, *(1001, bent_cap_node[1]),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ubig, Ubig, Ubig, self.MCTs.geomTransf_ver())
        ops.element('elasticBeamColumn', 9002, *(2001, bent_cap_node[-2]),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ubig, Ubig, Ubig, self.MCTs.geomTransf_ver())

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 竖向 零长单元 坐标转换
        vecx = [0, 0, -1]  # 局部x -> 整体坐标
        vecyp = [0, -1, 0]  # 局部y -> 整体坐标
        
        '''零长度截面单元 /截面旋转会导致两端节点产生竖向位移'''
        # 接触面：盖梁 -- 1 柱 上节段
        ops.element('zeroLengthSection', 11001,
                    *[1001, pier_1_top_seg_node[0]],
                    sec_ENT, '-orient', *vecx, *vecyp)
        # 接触面：1 柱 上节段 -- 1 柱 下节段
        ops.element('zeroLengthSection', 11002,
                    *[pier_1_top_seg_node[-1], pier_1_base_seg_node[0]],
                    sec_ENT, '-orient', *vecx, *vecyp)
        # 接触面：1 柱 下节段 -- 墩底
        ops.element('zeroLengthSection', 11003,
                    *[pier_1_base_seg_node[-1], 1002],
                    sec_ENT, '-orient', *vecx, *vecyp)
        
        # 接触面：盖梁 -- 2 柱 上节段
        ops.element('zeroLengthSection', 12001,
                    *[2001, pier_2_top_seg_node[0]],
                    sec_ENT, '-orient', *vecx, *vecyp)
        # 接触面：2 柱 上节段 -- 2 柱 下节段
        ops.element('zeroLengthSection', 12002,
                    *[pier_2_top_seg_node[-1], pier_2_base_seg_node[0]],
                    sec_ENT, '-orient', *vecx, *vecyp)
        # 接触面：2 柱 下节段 -- 墩底
        ops.element('zeroLengthSection', 12003,
                    *[pier_2_base_seg_node[-1], 2002],
                    sec_ENT, '-orient', *vecx, *vecyp)
        
        # 接触面 平移约束 equalDOF(主，从，[自由度])
        # equal_dof = [1, 2, 3, 4, 5, 6]
        equal_dof = [1, 2]
        ops.equalDOF(pier_1_top_seg_node[0], 1001, *equal_dof) # 盖梁 -- 1 柱 上节段
        ops.equalDOF(pier_1_base_seg_node[0], pier_1_top_seg_node[-1], *equal_dof) # 1 柱 上节段 -- 1 柱 下节段
        ops.equalDOF(1002, pier_1_base_seg_node[-1], *equal_dof) # 1 柱 下节段 -- 墩底
        
        ops.equalDOF(pier_2_top_seg_node[0], 2001, *equal_dof) # 盖梁 -- 2 柱 上节段
        ops.equalDOF(pier_2_base_seg_node[0], pier_2_top_seg_node[-1], *equal_dof) # 2 柱 上节段 -- 2 柱 下节段
        ops.equalDOF(2002, pier_2_base_seg_node[-1], *equal_dof) # 2 柱 下节段 -- 墩底
        
        # # 材料 对应 自由度
        # dir_mats = [K_free, K_ke, K_ke, K_free, K_free, K_free]  # 零长单元局部方向
        # dirs = [1, 2, 3, 4, 5, 6]
        # '''零长度单元 /可考虑接触面滑移'''
        # # 接触面：盖梁 -- 1 柱 上节段
        # ops.element('zeroLength', 21001,
        #             *[1001, pier_1_top_seg_node[0]],
        #             '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        # # 接触面：1 柱 上节段 -- 1 柱 下节段
        # ops.element('zeroLength', 21002,
        #             *[pier_1_top_seg_node[-1], pier_1_base_seg_node[0]],
        #             '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        # # 接触面：1 柱 下节段 -- 墩底
        # ops.element('zeroLength', 21003,
        #             *[pier_1_base_seg_node[-1], 1002],
        #             '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        
        # # 接触面：盖梁 -- 2 柱 上节段
        # ops.element('zeroLength', 22001,
        #             *[2001, pier_2_top_seg_node[0]],
        #             '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        # # 接触面：2 柱 上节段 -- 2 柱 下节段
        # ops.element('zeroLength', 22002,
        #             *[pier_2_top_seg_node[-1], pier_2_base_seg_node[0]],
        #             '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        # # 接触面：2 柱 下节段 -- 墩底
        # ops.element('zeroLength', 22003,
        #             *[pier_2_base_seg_node[-1], 2002],
        #             '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 1 柱 耗能钢筋 节段节点
        ops.node(4101, 0., -PierW / 2. - lw_pier / 2, 0.) # 节段部分
        ops.node(4102, 0., -PierW / 2. + lw_pier / 2, 0.) # 节段部分
        # ops.rigidLink('beam', pier_1_base_seg[-1], 4101)
        # ops.rigidLink('beam', pier_1_base_seg[-1], 4102)
        ops.element('elasticBeamColumn', 9101, *(pier_1_base_seg_node[-1], 4101),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ubig, Ubig, Ubig, self.MCTs.geomTransf_other())
        ops.element('elasticBeamColumn', 9102, *(pier_1_base_seg_node[-1], 4102),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ubig, Ubig, Ubig, self.MCTs.geomTransf_other())
        
        # 1 柱 耗能钢筋 粘结节点
        ops.node(4103, 0., -PierW / 2. - lw_pier / 2, -ED_l) # 粘结部分
        ops.node(4104, 0., -PierW / 2. + lw_pier / 2, -ED_l) # 粘结部分
        ops.fix(4103, 1, 1, 1, 1, 1, 1)
        ops.fix(4104, 1, 1, 1, 1, 1, 1)
        # 1 柱 耗能钢筋 单元
        ops.element('Truss', 41001, *[4101, 4103], ED_area, ED_bar)
        ops.element('Truss', 41002, *[4102, 4104], ED_area, ED_bar)
        
        # 1 柱 耗能钢筋 节段底部 竖向限位 /限制耗能钢筋受压
        ops.node(4105, 0., -PierW / 2. - lw_pier / 2, 0.)
        ops.node(4106, 0., -PierW / 2. + lw_pier / 2, 0.)
        ops.fix(4105, 1, 1, 1, 1, 1, 1)
        ops.fix(4106, 1, 1, 1, 1, 1, 1)
        
        
        # 2 柱 耗能钢筋 节段节点
        ops.node(4201, 0., PierW / 2. - lw_pier / 2, 0.) # 节段部分
        ops.node(4202, 0., PierW / 2. + lw_pier / 2, 0.) # 节段部分
        # ops.rigidLink('beam', pier_2_base_seg[-1], 4201)
        # ops.rigidLink('beam', pier_2_base_seg[-1], 4202)
        ops.element('elasticBeamColumn', 9201, *(pier_2_base_seg_node[-1], 4201),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ubig, Ubig, Ubig, self.MCTs.geomTransf_other())
        ops.element('elasticBeamColumn', 9202, *(pier_2_base_seg_node[-1], 4202),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ubig, Ubig, Ubig, self.MCTs.geomTransf_other())
        
        # 2 柱 耗能钢筋 粘结节点
        ops.node(4203, 0., PierW / 2. - lw_pier / 2, -ED_l) # 粘结部分
        ops.node(4204, 0., PierW / 2. + lw_pier / 2, -ED_l) # 粘结部分
        ops.fix(4203, 1, 1, 1, 1, 1, 1)
        ops.fix(4204, 1, 1, 1, 1, 1, 1)
        # 2 柱 耗能钢筋 单元
        ops.element('Truss', 42001, *[4201, 4203], ED_area, ED_bar)
        ops.element('Truss', 42002, *[4202, 4204], ED_area, ED_bar)

        # 2 柱 耗能钢筋 节段底部 竖向限位 /限制耗能钢筋受压
        ops.node(4205, 0., PierW / 2. - lw_pier / 2, 0.)
        ops.node(4206, 0., PierW / 2. + lw_pier / 2, 0.)
        ops.fix(4205, 1, 1, 1, 1, 1, 1)
        ops.fix(4206, 1, 1, 1, 1, 1, 1)


        '''竖向仅受压材料，水平方向自由度限制，其余方向释放刚度'''
        # 材料 对应 自由度
        dir_mats = [matENT, K_ke, K_ke, K_free, K_free, K_free]  # 零长单元局部方向
        dirs = [1, 2, 3, 4, 5, 6]
        # 1 柱 耗能钢筋 接触面 1
        ops.element('zeroLength', 41101,
                    *[4101, 4105],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        ops.element('zeroLength', 41102,
                    *[4102, 4106],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)

        # 2 柱 耗能钢筋 接触面 1
        ops.element('zeroLength', 42201,
                    *[4201, 4205],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)
        ops.element('zeroLength', 42202,
                    *[4202, 4206],
                    '-mat', *dir_mats, '-dir', *dirs, '-orient', *vecx, *vecyp)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁顶固定节点
        ops.node(3101, 0., -PierW / 2., PierH + lh_bent_cap * 2)
        ops.node(3201, 0., PierW / 2., PierH + lh_bent_cap * 2)
        ops.rigidLink('beam', bent_cap_node[1], 3101)
        ops.rigidLink('beam', bent_cap_node[-2], 3201)
        
        # 底座 固定节点
        ops.node(3102, 0., -PierW / 2., -lh_bent_cap * 2)
        ops.node(3202, 0., PierW / 2., -lh_bent_cap * 2)
        ops.fix(3102, 1, 1, 1, 1, 1, 1)
        ops.fix(3202, 1, 1, 1, 1, 1, 1)

        # 张拉控制力
        axial_force = 300.9 * UNIT.kn
        # 钢绞线总面积
        # PT_area = 3 * (np.pi * (15.2 * UNIT.mm / 2) ** 2)
        PT_area = 3 * 140 * UNIT.mm**2

        # 张拉控制应力
        sigma = axial_force / PT_area

        # 预应力筋 材料
        PT_mat = 30 # 材料标签
        PT_fy = 1860 * UNIT.mpa
        PT_Es = 195 * UNIT.gpa
        PT_ratio = 0.40 * PT_fy # 控制张拉比例
        # ops.uniaxialMaterial('Steel02', PT_mat, PT_fy, PT_Es, 0.01, 18, 0.925, 0.15, 0, 1, 0, 1, sigma)
        ops.uniaxialMaterial('Steel02', PT_mat, PT_fy, PT_Es, 0.01, 18, 0.925, 0.15, 0, 1, 0, 1, PT_ratio)

        # 预应力筋单元
        ops.element('Truss', 100, *(3101, 3102), PT_area, PT_mat)
        ops.element('Truss', 200, *(3201, 3202), PT_area, PT_mat)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 质量密度
        rho = 2600 * (UNIT.kg / (UNIT.m ** 3))  # kg/m3

        # 单位节点质量
        bent_cap_mass = L * BentCapProps.SecMashProps.A * rho / len(bent_cap_node)
        pier_1_top_seg_mass = Seg * PierProps.SecMashProps.A * rho / len(pier_1_top_seg_node)
        pier_1_base_seg_mass = Seg * PierProps.SecMashProps.A * rho / len(pier_1_base_seg_node)
        pier_2_top_seg_mass = Seg * PierProps.SecMashProps.A * rho / len(pier_2_top_seg_node)
        pier_2_base_seg_mass = Seg * PierProps.SecMashProps.A * rho / len(pier_2_base_seg_node)

        # 节点质量
        for i in bent_cap_node:
            ops.mass(i, bent_cap_mass, bent_cap_mass, bent_cap_mass, 0, 0, 0)
        for i in pier_1_top_seg_node:
            ops.mass(i, pier_1_top_seg_mass, pier_1_top_seg_mass, pier_1_top_seg_mass, 0, 0, 0)
        for i in pier_1_base_seg_node:
            ops.mass(i, pier_1_base_seg_mass, pier_1_base_seg_mass, pier_1_base_seg_mass, 0, 0, 0)
        for i in pier_2_top_seg_node:
            ops.mass(i, pier_2_top_seg_mass, pier_2_top_seg_mass, pier_2_top_seg_mass, 0, 0, 0)
        for i in pier_2_base_seg_node:
            ops.mass(i, pier_2_base_seg_mass, pier_2_base_seg_mass, pier_2_base_seg_mass, 0, 0, 0)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 定义模型不同部位名称
        LOCATION_INFO = namedtuple('LOCATION_INFO', ['eleTag', 'integ', 'location'])
        # 主函数 要返回的数据
        SectionMat={
                'BentCapProps': BentCapProps,
                'PierProps': PierProps,
                }
        KeyNode={
                # 位移控制节点
                'ctrl_node': bent_cap_node[0],
                # 接触面节点监控
                'pier_1_seg_1_1': 1001, # 与盖梁链接的节点
                'pier_1_seg_1_2': pier_1_top_seg_node[0], # 1 柱 上节段 顶 节点
                
                'pier_1_seg_2_1': pier_1_top_seg_node[-1], # 1 柱 上节段 底 节点
                'pier_1_seg_2_2': pier_1_base_seg_node[0], # 1 柱 下节段 顶 节点
                
                'pier_1_seg_3_1': pier_1_base_seg_node[-1], # 1 柱 下节段 底 节点
                'pier_1_seg_3_2': 1002, # 底部固定节点
                # 上部 BRB 端节点
                'top_BRB_i': 5101,
                'top_BRB_j': 5201,
                # 下部 BRB 端节点
                'base_BRB_i': 5102,
                'base_BRB_j': 5202,
                }
        KeyEle={
                # 预应力
                'Pier_1_PT_bar': 100,
                'Pier_2_PT_bar': 200,
                # 耗能钢筋
                'Pier_1_ED_bar_1': 41001,
                'Pier_1_ED_bar_2': 41002,
                'Pier_2_ED_bar_1': 42001,
                'Pier_2_ED_bar_2': 42002,
                # ENT 纤维截面
                'ENT_sec': 11001,
                }
        LocationDamage=[
                LOCATION_INFO(eleTag=pier_1_top_seg_ele[0], integ=1, location='pier_1_top_seg_top'),
                LOCATION_INFO(eleTag=pier_1_top_seg_ele[-1], integ=5, location='pier_1_top_seg_base'),

                LOCATION_INFO(eleTag=pier_1_base_seg_ele[0], integ=1, location='pier_1_base_seg_top'),
                LOCATION_INFO(eleTag=pier_1_base_seg_ele[-1], integ=5, location='pier_1_base_seg_base'),
                
                LOCATION_INFO(eleTag=pier_2_top_seg_ele[0], integ=1, location='seg_3_base'),
                LOCATION_INFO(eleTag=pier_2_top_seg_ele[-1], integ=5, location='seg_3_top'),

                LOCATION_INFO(eleTag=pier_2_base_seg_ele[0], integ=1, location='seg_4_base'),
                LOCATION_INFO(eleTag=pier_2_base_seg_ele[-1], integ=5, location='seg_4_top'),
                ]
        OtherOptional={
                'ED_bar_yield_strain': ED_fy / ED_Es,
                }
        
        return SectionMat, KeyNode, KeyEle, LocationDamage, OtherOptional


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
        self.MCTs = ModelCreateTools()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        (
            section_mat,
            key_node,
            key_ele,
            location_damage,
            other_optional
            ) = self._set_pier(
                modelPath=modelPath,
                Ke=Ke,
                info=info
                )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化输出模型
        opsplt.set_plot_props(point_size=5, line_width=3)
        fig = opsplt.plot_model(
            show_node_numbering=True,
            show_ele_numbering=False,
            show_local_axes=False
            )
        fig.write_html(f"{modelPath}/{my_name}.html", full_html=False, include_plotlyjs="cdn")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 返回数据
        self.model_props = PVs.MODEL_PROPS(
            Name=my_name,
            SectionMat=section_mat,
            KeyNode=key_node,
            KeyEle=key_ele,
            LocationDamage=location_damage,
            OtherOptional=other_optional
        )

        return self.model_props


    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def RockPierBRB(
        self,
        modelPath: str,
        
        Ke: float,
        yield_disp: float,
        yield_force: float,
        core_ratio: float,
        core_area: float,

        info: bool = True
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
        self.MCTs = ModelCreateTools()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 设置桥墩
        (
            section_mat,
            key_node,
            key_ele,
            location_damage,
            other_optional
            ) = self._set_pier(
                modelPath=modelPath,
                Ke=Ke,
                info=info
                )
            
        # 设置 BRB
        (
            top_brb_tag,
            top_brb_yield_disp,
            top_brb_yield_strain,
            top_brb_yield_force, 
            top_brb_yield_stiff
            ) = self._set_BRB(
                yield_disp=yield_disp,
                yield_force=yield_force,
                node_i=key_node['top_BRB_i'],
                node_j=key_node['top_BRB_j'],
                core_ratio=core_ratio, core_area=core_area,
                link_E=1.e6,
                groupTag=81000
                )
        (
            base_brb_tag,
            base_brb_yield_disp,
            base_brb_yield_strain,
            base_brb_yield_force,
            base_brb_yield_stiff
            ) = self._set_BRB(
                yield_disp=yield_disp,
                yield_force=yield_force,
                node_i=key_node['base_BRB_i'],
                node_j=key_node['base_BRB_j'],
                core_ratio=core_ratio, core_area=core_area,
                link_E=1.e6,
                groupTag=82000
                )
            
        # BRB 单元好 添加到 单元合集
        key_ele['top_brb'] = top_brb_tag
        key_ele['base_brb'] = base_brb_tag
        
        # BRB 设计指标 添加到 其他合集
        other_optional['top_BRB_indicator_alpha'] = top_brb_yield_stiff / (yield_force / yield_disp)
        other_optional['top_BRB_indicator_miu'] = yield_disp / top_brb_yield_disp
        other_optional['base_BRB_indicator_alpha'] = base_brb_yield_stiff / (yield_force / yield_disp)
        other_optional['base_BRB_indicator_miu'] = yield_disp / base_brb_yield_disp
        
        # BRB 其他参数 添加到 其他合集
        other_optional['top_brb_yield_disp'] = top_brb_yield_disp
        other_optional['top_brb_yield_strain'] = top_brb_yield_strain
        other_optional['top_brb_yield_force'] = top_brb_yield_force
        other_optional['top_brb_yield_stiff'] = top_brb_yield_stiff
        
        other_optional['base_brb_yield_disp'] = base_brb_yield_disp
        other_optional['base_brb_yield_strain'] = base_brb_yield_strain
        other_optional['base_brb_yield_force'] = base_brb_yield_force
        other_optional['base_brb_yield_stiff'] = base_brb_yield_stiff


        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化输出模型
        opsplt.set_plot_props(point_size=5, line_width=3)
        fig = opsplt.plot_model(
            show_node_numbering=True,
            show_ele_numbering=False,
            show_local_axes=False
            )
        fig.write_html(f"{modelPath}/{my_name}.html", full_html=False, include_plotlyjs="cdn")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 返回数据
        self.model_props = PVs.MODEL_PROPS(
            Name=my_name,
            SectionMat=section_mat,
            KeyNode=key_node,
            KeyEle=key_ele,
            LocationDamage=location_damage,
            OtherOptional=other_optional
        )

        return self.model_props


    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def determine_damage(self, odb_tag: Union[str, int] ,info: bool):

        # 导入数据
        ODB_ele_sec = opst.post.get_element_responses(odb_tag=odb_tag, ele_type="FiberSection", print_info=False)

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
        ODB_ele_sec = opst.post.get_element_responses(odb_tag=odb_tag, ele_type="FiberSection", print_info=False)
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
    def reasp_top_disp(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_node_disp_resp = opst.post.get_nodal_responses(odb_tag=odb_tag, resp_type='disp', print_info=False)
        # 控制节点位移
        disp = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['ctrl_node'], DOFs='UY')
        
        return disp
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def show_fiber_sec(self, ele_tag: int):
        plt.close('all')
        opst.pre.section.vis_fiber_sec_real(
            ele_tag=ele_tag, show_matTag=True,
            # highlight_matTag=SecProps.SteelTag, highlight_color="r",
            )
        
        return plt
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def reasp_seg_node_disp(self, odb_tag: Union[str, int]):
        """零长截面单元 - 两节点的竖向位移"""
        # 导入数据
        ODB_node_disp_resp = opst.post.get_nodal_responses(odb_tag=odb_tag, resp_type='disp', print_info=False)
        
        # 墩顶位移数据
        disp = np.array(self.reasp_top_disp(odb_tag))
        
        # 接触面监控节点
        pier_1_seg_1_1 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_seg_1_1'], DOFs='UZ')
        pier_1_seg_1_2 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_seg_1_2'], DOFs='UZ')
        
        pier_1_seg_2_1 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_seg_2_1'], DOFs='UZ')
        pier_1_seg_2_2 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_seg_2_2'], DOFs='UZ')
        
        pier_1_seg_3_1 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_seg_3_1'], DOFs='UZ')
        pier_1_seg_3_2 = ODB_node_disp_resp.sel(nodeTags=self.model_props.KeyNode['pier_1_seg_3_2'], DOFs='UZ')
        
        # 绘图
        plt.close('all')
        plt.figure(figsize=(6, 4))
        
        plt.plot(disp, np.array(pier_1_seg_1_1), label='Seg 1 Node 1', zorder=2)
        plt.plot(disp, np.array(pier_1_seg_1_2), label='Seg 1 Node 2', zorder=2)
        plt.plot(disp, np.array(pier_1_seg_2_1), label='Seg 2 Node 1', zorder=2)
        plt.plot(disp, np.array(pier_1_seg_2_2), label='Seg 2 Node 2', zorder=2)
        plt.plot(disp, np.array(pier_1_seg_3_1), label='Seg 3 Node 1', zorder=2)
        plt.plot(disp, np.array(pier_1_seg_3_2), label='Seg 3 Node 2', zorder=2)
        
        plt.xlabel('Displacement (m)')
        plt.ylabel('Seg UZ Disp (m)')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.grid(linestyle='--', linewidth=0.5, zorder=1)
        
        return plt
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def reasp_PT_force(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_truss_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type='Truss', print_info=False)
        # print(ODB_truss_resp)
        
        # 墩顶位移数据
        disp = np.array(self.reasp_top_disp(odb_tag))
        # 预应力轴力数据
        PT_1_axialForce = ODB_truss_resp['axialForce'].sel(eleTags=self.model_props.KeyEle['Pier_1_PT_bar'])
        PT_2_axialForce = ODB_truss_resp['axialForce'].sel(eleTags=self.model_props.KeyEle['Pier_2_PT_bar'])
        
        # 绘图
        plt.close('all')
        plt.figure(figsize=(6, 4))
        plt.plot(disp, PT_1_axialForce, label='Pier 1 PT bar', zorder=2)
        plt.plot(disp, PT_2_axialForce, label='Pier 2 PT bar', zorder=2)
        plt.xlabel('Displacement (m)')
        plt.ylabel('PT bar Axial Force (kN)')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.grid(linestyle='--', linewidth=0.5, zorder=1)
        
        return plt

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def reasp_ED_stress_strain(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_truss_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type='Truss', print_info=False)
        # print(ODB_truss_resp)
        
        # 墩顶位移数据
        disp = np.array(self.reasp_top_disp(odb_tag))
        # 耗能钢筋轴力数据
        Pier_1_ED_1_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_1_ED_bar_1'])
        Pier_1_ED_1_stress = ODB_truss_resp['Stress'].sel(eleTags=self.model_props.KeyEle['Pier_1_ED_bar_1'])
        Pier_1_ED_2_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_1_ED_bar_2'])
        Pier_1_ED_2_stress = ODB_truss_resp['Stress'].sel(eleTags=self.model_props.KeyEle['Pier_1_ED_bar_2'])
        
        Pier_2_ED_1_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_2_ED_bar_1'])
        Pier_2_ED_1_stress = ODB_truss_resp['Stress'].sel(eleTags=self.model_props.KeyEle['Pier_2_ED_bar_1'])
        Pier_2_ED_2_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_2_ED_bar_2'])
        Pier_2_ED_2_stress = ODB_truss_resp['Stress'].sel(eleTags=self.model_props.KeyEle['Pier_2_ED_bar_2'])
        
        # 绘图
        plt.close('all')
        plt.figure(figsize=(6, 4))
        
        plt.plot(Pier_1_ED_1_strain, Pier_1_ED_1_stress, label='Pier 1 ED bar 1', zorder=2)
        plt.plot(Pier_1_ED_2_strain, Pier_1_ED_2_stress, label='Pier 1 ED bar 2', zorder=2)
        
        plt.xlabel('Strain')
        plt.ylabel('Stress')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.grid(linestyle='--', linewidth=0.5, zorder=1)
        
        return plt
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def yield_ED(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_truss_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type='Truss', print_info=False)
        # print(ODB_truss_resp)
        
        # 墩顶位移数据
        disp = np.array(self.reasp_top_disp(odb_tag))
        # 耗能钢筋轴力数据 /转成 dataframe 方便筛选应变
        ED_1_strain_1 = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_1_ED_bar_1']).to_dataframe().reset_index()
        ED_1_strain_2 = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_1_ED_bar_2']).to_dataframe().reset_index()
        ED_2_strain_1 = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_2_ED_bar_1']).to_dataframe().reset_index()
        ED_2_strain_2 = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['Pier_2_ED_bar_2']).to_dataframe().reset_index()

        # 检索阈值
        yield_strain = self.model_props.OtherOptional['ED_bar_yield_strain']
        # print(f'# yield strain: {yield_strain}')
        
        # 检索
        ED_1_yield_1 = ED_1_strain_1[
            ED_1_strain_1["Strain"].abs() >= yield_strain
            ]
        ED_1_yield_2 = ED_1_strain_2[
            ED_1_strain_2["Strain"].abs() >= yield_strain
            ]
        ED_2_yield_1 = ED_2_strain_1[
            ED_2_strain_1["Strain"].abs() >= yield_strain
            ]
        ED_2_yield_2 = ED_2_strain_2[
            ED_2_strain_2["Strain"].abs() >= yield_strain
            ]
        
        # 最先屈服点
        min_yield_step = min(
            ED_1_yield_1.index.min(),
            ED_1_yield_2.index.min(),
            ED_2_yield_1.index.min(),
            ED_2_yield_2.index.min()
            )

        # 判断：是否有屈服
        if pd.isna(min_yield_step):
            warnings.warn(f"所有耗能钢筋均未屈服，返回做大步长：{len(disp)}", UserWarning)
            min_yield_step = len(disp)
            # 返回屈服step
            return min_yield_step
        
        else:
            # print(f'# yield step: {min_yield_step}')
            # 返回屈服step
            return min_yield_step

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def reasp_BRB_stress_strain(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_truss_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type='Truss', print_info=False)
        # print(ODB_truss_resp)
        
        # 墩顶位移数据
        disp = np.array(self.reasp_top_disp(odb_tag))
        # 预应力轴力数据
        top_BRB_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['top_brb'])
        top_BRB_stress = ODB_truss_resp['Stress'].sel(eleTags=self.model_props.KeyEle['top_brb'])
        base_BRB_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['base_brb'])
        base_BRB_stress = ODB_truss_resp['Stress'].sel(eleTags=self.model_props.KeyEle['base_brb'])
        
        # 绘图
        plt.close('all')
        plt.figure(figsize=(6, 4))
        # plt.plot(disp, PT_1_axialForce, label='Pier 1 ED bar', zorder=2)
        # plt.plot(disp, PT_2_axialForce, label='Pier 2 ED bar', zorder=2)
        
        plt.plot(top_BRB_strain, top_BRB_stress, label='Top BRB', zorder=2)
        plt.plot(base_BRB_strain, base_BRB_stress, label='Base BRB', zorder=2)
        
        plt.xlabel('Strain')
        plt.ylabel('Stress')
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.grid(linestyle='--', linewidth=0.5, zorder=1)
        
        return plt

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def yield_BRB(self, odb_tag: Union[str, int], min_step: bool = False):
        # 导入数据
        ODB_truss_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type='Truss', print_info=False)
        # print(ODB_truss_resp)
        
        # 墩顶位移数据
        disp = np.array(self.reasp_top_disp(odb_tag))
        # 耗能钢筋轴力数据 /转成 dataframe 方便筛选应变
        top_BRB_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['top_brb']).to_dataframe().reset_index()
        base_BRB_strain = ODB_truss_resp['Strain'].sel(eleTags=self.model_props.KeyEle['base_brb']).to_dataframe().reset_index()

        # 检索阈值
        top_BRB_yield_strain = self.model_props.OtherOptional['top_brb_yield_strain']
        base_BRB_yield_strain = self.model_props.OtherOptional['base_brb_yield_strain']
        # print(f'# yield strain: {yield_strain}')
        
        # 检索
        top_BRB_yield = top_BRB_strain[
            top_BRB_strain["Strain"].abs() >= top_BRB_yield_strain
            ]
        base_BRB_yield = base_BRB_strain[
            base_BRB_strain["Strain"].abs() >= base_BRB_yield_strain
            ]

        # 判断：是否有屈服
        if pd.isna(top_BRB_yield).all().all() and pd.isna(base_BRB_yield).all().all():
            # 警告
            warnings.warn(f"所有BRB均未屈服，返回做大步长：{len(disp)}", UserWarning)
            # 处理数据
            brb_yield_step = len(disp)
            # 返回屈服step
            return [brb_yield_step]
        
        else:
            
            if min_step:
                    # 最先屈服点
                    min_yield_step = min(
                        top_BRB_yield.index.min(),
                        base_BRB_yield.index.min(),
                        )
                    # print(f'# BRB yield step: {min_yield_step}')
                    # 返回 最小屈服step
                    return min_yield_step
            else:
                # 返回 两个屈服step
                return [top_BRB_yield.index.min(), base_BRB_yield.index.min()]


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
    
    # opsplt.set_plot_props(point_size=5, line_width=3)
    # fig = opsplt.plot_model(
    #     show_node_numbering=False,
    #     show_ele_numbering=False,
    #     show_local_axes=True
    #     )
    # fig.show(renderer="browser")
    # fig.write_html("images/model_plotly0.html", full_html=False, include_plotlyjs="cdn")
    

