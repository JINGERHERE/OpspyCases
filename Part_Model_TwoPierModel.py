#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Part_Model_TwoPierModel.py
@Date    ：2025/8/1 19:24
@IDE     ：PyCharm
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
from script import UNIT, PVs, ModelCreateTools

from Part_MatSec_TwoPierModel import TwoPierModelSection
from script.post import DamageStateTools

from typing import NamedTuple
from collections import namedtuple


"""
# --------------------------------------------------
# ========== < Part_Model_TwoPierModel > ==========
# --------------------------------------------------
"""


class TwoPierModelTEST:

    def __init__(self):
        
        # self.Ubig = 20000. * UNIT.gpa # 约束刚度
        # self.Usmall = 2.e-8 * UNIT.gpa # 释放刚度
        self.Ubig = 1.e8 # 约束刚度
        self.Usmall = 1.e-8 * UNIT.gpa # 释放刚度
        self.MCTs: ModelCreateTools # 模型创建工具

        self.model_props: PVs.MODEL_PROPS # 模型的输出属性

        # 结果数据
        self.node_resp: xr.DataArray # 节点响应数据
        self.ele_resp: xr.DataArray # 单元响应数据
        
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def _set_BRB_lite(
        self,
        node_i: int, node_j: int,
        core_ratio: float, core_area: float,
        link_E: float = 1.e6,
        groupTag: int = 1,
        ):
        
        """
        node_i: i 节点
        node_j: j 节点
        core_ratio: 核心段长度比
        core_area: 核心段面积
        link_E: 连接段刚度
        groupTag: 组标签
        """
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # BRB 材料数据
        Q235_fy = 235 * UNIT.mpa # Q235屈服强度
        Q235_Es = 200 * UNIT.gpa # Q235弹性模量
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
        # BRB 核心段 长度
        core_len = core_ratio * L
        
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
        # ops.rigidLink('beam', node_i, core_node_tag_i)

        ops.element('Truss', core_ele_tag, *[core_node_tag_i, core_node_tag_j], core_area, BRBmat)

        ops.element('elasticBeamColumn', link_ele_tag_j,
                    *(core_node_tag_j, node_j),
                    200 * UNIT.gpa, 100 * UNIT.gpa,
                    0.1, link_E, link_E, link_E, self.MCTs.geomTransf_other())
        # ops.rigidLink('beam', node_j, core_node_tag_j)
        
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
        
        # 相关参数
        nb = 1 # 沿 柱高方向 BRB个数
        
        # 夹角
        theta = abs(
            calculate_angle_with_horizontal(i_coord, j_coord)
            )
        
        # BRB 屈服位移
        hy = core_len * np.sin(theta) # BRB 核心长度 在竖直方向投影
        delta_yb = (Q235_eps_y * core_len) / np.cos(theta)  # BRB 相对屈服位移
        
        # BRB 绝对屈服位移
        delta_yb = delta_yb * nb
        
        # 连接段 在水平线上的投影
        Lbx = (L - core_len) / 2 * np.cos(theta)
        
        # BRB 强度贡献
        V_yb = 2 * nb * Q235_fy * Lbx * core_area * np.sin(theta)
        
        # BRB 刚度 贡献
        Kb = V_yb / delta_yb

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 返回：BRB 单元号、BRB 屈服应变、BRB 屈服位移、BRB 屈服力、BRB 刚度
        # return core_ele_tag, Q235_eps_y, delta_yb, V_yb, Kb
        return core_ele_tag
        
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def _set_BRB_buckling(
        self,
        node_i: int, node_j: int,
        groupTag: int,
        core_ratio: float,
        gap: Union[int, float], gapK: float,
        bucklingK: float,
        ):
        
        """
        node_i: 既有的 i 节点号
        node_j: 既有的 j 节点号
        core_ratio: 核心长度比
        groupTag: 节点单元编号组
        gap: 隙距
        gapE: 隙摩擦刚度
        bulckingE: 屈曲刚度
        """
        """
            node_i bucklingGap buckling        buckling  bucklingGap node_j
              * ———— * | * ———— * | * ———— ———— * | * ———— * | * ———— *
                    zeroLength zeroLength     zeroLength zeroLength
        """
        
        if core_ratio >=1:
            ValueError("Core ratio should be less than 1.")
        elif core_ratio <= 0:
            ValueError("Core ratio should be greater than 0.")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 节点坐标
        i_coord = np.array(ops.nodeCoord(node_i))
        j_coord = np.array(ops.nodeCoord(node_j))

        # 计算距离
        L = np.linalg.norm(j_coord - i_coord) # 固定端总长度
        core_len = L * core_ratio # 核心段长度

        # 计算直线参数
        center_point = (i_coord + j_coord) / 2 # 直线中心坐标
        dir_vector = (j_coord - i_coord) / L # 直线方向向量

        # 缩放后的端点坐标
        core_i_coord = center_point - (dir_vector * core_len / 2)
        core_j_coord = center_point + (dir_vector * core_len / 2)

        i_link_coord = (i_coord + core_i_coord) / 2
        j_link_coord = (j_coord + core_j_coord) / 2

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # I 端 节点标签
        I_bucklingGap_i = groupTag + 1
        I_bucklingGap_j = groupTag + 2
        I_buckling_i = groupTag + 3
        I_buckling_j = groupTag + 4

        # J 端 节点标签
        J_buckling_i = groupTag + 5
        J_buckling_j = groupTag + 6
        J_bucklingGap_i = groupTag + 7
        J_bucklingGap_j = groupTag + 8

        # 节点
        # I 端：buckling + gap
        ops.node(I_bucklingGap_i, *i_link_coord)
        ops.node(I_bucklingGap_j, *i_link_coord)
        # # I 端：buckling
        ops.node(I_buckling_i, *core_i_coord)
        ops.node(I_buckling_j, *core_i_coord)
        # # J 端：buckling
        ops.node(J_buckling_i, *core_j_coord)
        ops.node(J_buckling_j, *core_j_coord)
        # # J 端：buckling + gap
        ops.node(J_bucklingGap_i, *j_link_coord)
        ops.node(J_bucklingGap_j, *j_link_coord)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # BRB Material
        Q235 = groupTag + 1
        Q235_fy = 309.57 * UNIT.mpa
        Q235_fu = 476.9 * UNIT.mpa
        Q235_Es = 176.7 * UNIT.gpa
        Q235_Esh = 0.01 * Q235_Es
        Q235_esh = 1.5 * (Q235_fy / Q235_Es)
        Q235_eult  = 0.1
        # ops.uniaxialMaterial('Steel02', Q235, Q235_fy, Q235_Es, 0.01, 18, 0.925, 0.15)
        
        GABuck_lsr = 6.0
        GABuck_beta = 1.0
        GABuck_r = 0.0
        GABuck_gamma = 0.5
        ops.uniaxialMaterial(
            'ReinforcingSteel', Q235, Q235_fy, Q235_fu, Q235_Es, Q235_Esh, Q235_esh, Q235_eult,
            '-GABuck', GABuck_lsr, GABuck_beta, GABuck_r, GABuck_gamma)

        # 屈曲材料
        bucklingMat = groupTag + 2
        ops.uniaxialMaterial('Elastic', bucklingMat, bucklingK)
        
        # 约束材料
        fixMat = groupTag + 3
        freeMat = groupTag + 4
        ops.uniaxialMaterial('Elastic', fixMat, self.Ubig)
        ops.uniaxialMaterial('Elastic', freeMat, self.Usmall)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        " # ===== < 方法 1: HookGap - Elastic > ===== #"
        # # 间隙材料
        # gapMat = groupTag + 10 + 1
        # ops.uniaxialMaterial("HookGap", gapMat, Q235_Es, -gap, gap) # tag     E    gap_p   gap_n
        
        # # 摩擦材料
        # slipMat = groupTag + 10 + 2
        # ops.uniaxialMaterial('Elastic', slipMat, gapK)
        # # 并联材料：gap + slip
        # GapModel = groupTag + 10 + 3
        # ops.uniaxialMaterial('Parallel', GapModel, gapMat, slipMat)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        " # ===== < 方法 2: Elastic - ElasticPPGap > ===== #"
        # # 摩擦刚度
        # slipMat = groupTag + 10 + 1
        # ops.uniaxialMaterial('Elastic', slipMat, gapK)
        # # 间隙位移限制刚度
        # gapLimitMat_1 = groupTag + 10 + 2
        # gapLimitMat_2 = groupTag + 10 + 3
        # ops.uniaxialMaterial("ElasticPPGap", gapLimitMat_1, Q235_Es, Q235_fy, gap)
        # ops.uniaxialMaterial("ElasticPPGap", gapLimitMat_2, Q235_Es, -Q235_fy, -gap)
        # # 并联材料：gap + slip
        # GapModel = groupTag + 10 + 4
        # ops.uniaxialMaterial('Parallel', GapModel, slipMat, gapLimitMat_1, gapLimitMat_2)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        " # ===== < 方法 3: ElasticPPGap > ===== #"
        # 摩擦刚度
        slipMat_1 = groupTag + 10 + 1
        slipMat_2 = groupTag + 10 + 2
        ops.uniaxialMaterial("ElasticPPGap", slipMat_1, gapK, self.Ubig, gap/200.)
        ops.uniaxialMaterial("ElasticPPGap", slipMat_2, bucklingK, -self.Ubig, -gap/200.)
        # 间隙位移限制刚度
        gapLimitMat_1 = groupTag + 10 + 3
        gapLimitMat_2 = groupTag + 10 + 4
        ops.uniaxialMaterial("ElasticPPGap", gapLimitMat_1, Q235_Es, self.Ubig, gap)
        ops.uniaxialMaterial("ElasticPPGap", gapLimitMat_2, bucklingK, -self.Ubig, -gap)
        # 并联材料：gap + slip
        GapModel = groupTag + 10 + 5
        ops.uniaxialMaterial('Parallel', GapModel, slipMat_1, slipMat_2, gapLimitMat_1, gapLimitMat_2)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # BRB section
        secBRB = groupTag + 1
        ops.section('fiberSec', secBRB, '-GJ', 3.153628770231478)
        ops.patch('rect ', Q235, 5, 10, -10./2. * UNIT.mm, -101.67/2. * UNIT.mm, 10./2. * UNIT.mm, 101.67/2. * UNIT.mm)
        
        secLink = groupTag + 2
        ops.section('fiberSec', secLink, '-GJ', 6.515607347333899)
        ops.patch('rect ', Q235, 5, 10, -10./2 * UNIT.mm, -200./2 * UNIT.mm, 10./2 * UNIT.mm, 200./2 * UNIT.mm)
        
        # BRB integration
        npBRB = groupTag + 1
        ops.beamIntegration('Legendre', npBRB, secBRB, 5)
        npLink = groupTag + 2
        ops.beamIntegration('Legendre', npLink, secLink, 5)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # I 端 单元号
        I_link_1 = groupTag + 1
        I_bucklingGap = groupTag + 2
        I_link_2 = groupTag + 3
        I_buckling = groupTag + 4

        # 核心单元号
        coreEle = groupTag + 5

        # J 端 单元号
        J_buckling = groupTag + 6
        J_link_1 = groupTag + 7
        J_bucklingGap = groupTag + 8
        J_link_2 = groupTag + 9

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 零长单元方向材料分配
        dirs = [1, 2, 3, 4, 5, 6]
        bucklingGap_mats = [GapModel, fixMat, fixMat, fixMat, fixMat, fixMat]
        buckling_mats = [fixMat, fixMat, fixMat, fixMat, bucklingMat, bucklingMat]
        '这个压缩方向的屈曲是真的忘记了，真没招了，就先这样吧'
        
        # 零长单元 坐标转换
        vecx = dir_vector.tolist()  # 局部x -> 整体方向
        vecyp = self.MCTs.vecTransf_other()  # 局部y -> 整体方向
        transfTag = self.MCTs.auto_geomTransf(node_i, node_j)  # 单元坐标转换标签

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 连接段面接
        area_link = 10. * UNIT.mm * 200. * UNIT.mm
        E_link = 200. * UNIT.gpa # 连接段弹性模量
        G_link = 80. * UNIT.gpa # 连接段剪切模量
        # 单元
        # I 端
        ops.element('elasticBeamColumn', I_link_1, *(node_i, I_bucklingGap_i), E_link, G_link, area_link, self.Ubig, self.Ubig, self.Ubig, transfTag)
        ops.element('zeroLength', I_bucklingGap, *(I_bucklingGap_i, I_bucklingGap_j), '-mat', *bucklingGap_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling + gap
        ops.element('elasticBeamColumn', I_link_2, *(I_bucklingGap_j, I_buckling_i), E_link, G_link, area_link, self.Ubig, self.Ubig, self.Ubig, transfTag)
        
        # # CORE
        ops.element('zeroLength', I_buckling, *(I_buckling_i, I_buckling_j), '-mat', *buckling_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling
        ops.element('dispBeamColumn', coreEle, *(I_buckling_j, J_buckling_i), transfTag, npBRB)
        ops.element('zeroLength', J_buckling, *(J_buckling_i, J_buckling_j), '-mat', *buckling_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling
        
        # # J 端
        ops.element('elasticBeamColumn', J_link_1, *(J_buckling_j, J_bucklingGap_i), E_link, G_link, area_link, self.Ubig, self.Ubig, self.Ubig, transfTag)
        ops.element('zeroLength', J_bucklingGap, *(J_bucklingGap_i, J_bucklingGap_j), '-mat', *bucklingGap_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling + gap
        ops.element('elasticBeamColumn', J_link_2, *(J_bucklingGap_j, node_j), E_link, G_link, area_link, self.Ubig, self.Ubig, self.Ubig, transfTag)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        return coreEle
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def _set_pier(
        self,
        modelPath: str,
        Ke: float,
        info: bool
        ):
        
        """
        该模型考虑了墩底滑移，对试验滞回曲线进行拟合
            modelPath: 模型路径
            Ke: 模型收敛刚度拟合
            info: 是否打印信息
        """

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 桥墩尺寸控制
        L = 2.8 * UNIT.m
        PierH = 1.6 * UNIT.m
        PierW = 2 * UNIT.m

        # 模型收敛刚度拟合 /确定值：0.15 * UNIT.pa 基于全局单位：kN，m
        Ke = 0.15 * UNIT.pa

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
        BentCapProps = TwoPierModelSection.bent_cap_sec(modelPath, **bent_cap_section_tags)  # 创建盖梁纤维截面，并获取截面参数

        # 墩柱截面 墩柱材料 编号
        pier_section_tags = {
            'section_tag': 200,  # 截面
            'cover_tag': 5,  # 材料-保护层
            'core_tag': 6,  # 材料-核心
            'bar_tag': 7,  # 材料-钢筋
            'bar_max_tag': 8,  # 材料-钢筋最大应变限制
            'info': info
            }
        PierProps = TwoPierModelSection.pier_sec(modelPath, **pier_section_tags)  # 创建墩柱纤维截面，并获取截面参数

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元积分点
        tag_np_beam = 1
        tag_np_pier = 2
        # ops.beamIntegration('Lobatto', tag_np_beam, bent_cap_section_tags['section_tag'], 5) # 基于力
        # ops.beamIntegration('Lobatto', tag_np_pier, pier_section_tags['section_tag'], 5)
        ops.beamIntegration('Legendre', tag_np_beam, bent_cap_section_tags['section_tag'], 5) # 基于位移
        ops.beamIntegration('Legendre', tag_np_pier, pier_section_tags['section_tag'], 5)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁质心
        lw = BentCapProps.SecMashProps.centroid[0]
        lh = BentCapProps.SecMashProps.centroid[1]
        # 节点坐标
        bent_cap_coord = [
            (0., -L / 2, PierH + lh),

            (0., -PierW / 2, PierH + lh),

            (0., -PierW / 2 * (2/3), PierH + lh),
            (0., -PierW / 2 * (1/3), PierH + lh),
            (0., 0., PierH + lh),
            (0., PierW / 2 * (1/3), PierH + lh),
            (0., PierW / 2 * (2/3), PierH + lh),
            
            (0., PierW / 2 -0.2, PierH + lh), # BRB 连接点

            (0., PierW / 2, PierH + lh),

            (0., L / 2, PierH + lh),
            ]
        pier_1_coord = NodeTools.distribute(
            (0., -PierW / 2, PierH),
            (0., -PierW / 2, 0.),
            7, (1, 1)
            )
        pier_2_coord = NodeTools.distribute(
            (0., PierW / 2, PierH),
            (0., PierW / 2, 0.),
            7, (1, 1)
            )

        # 节点编号组
        bent_cap_node_start = 1000
        pier_1_node_start = 2100
        pier_2_node_start = 2200

        # 创建节点
        bent_cap_node = self.MCTs.node_create(bent_cap_node_start, bent_cap_coord)
        pier_1_node = self.MCTs.node_create(pier_1_node_start, pier_1_coord)
        pier_2_node = self.MCTs.node_create(pier_2_node_start, pier_2_coord)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元连接
        bent_cap_links = []
        for node_i, node_j in pairwise(bent_cap_node):
            bent_cap_links.append((node_i, node_j))

        pier_1_links = []
        for node_i, node_j in pairwise(pier_1_node):
            pier_1_links.append((node_i, node_j))

        pier_2_links = []
        for node_i, node_j in pairwise(pier_2_node):
            pier_2_links.append((node_i, node_j))

        # 单元编组
        bent_cap_ele_start = 1000
        pier_1_ele_start = 2100
        pier_2_ele_start = 2200

        # 创建单元
        bent_cap_ele = self.MCTs.ele_create(bent_cap_ele_start, bent_cap_links, tag_np_beam)
        pier_1_ele = self.MCTs.ele_create(pier_1_ele_start, pier_1_links, tag_np_pier)
        pier_2_ele = self.MCTs.ele_create(pier_2_ele_start, pier_2_links, tag_np_pier)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 梁柱连接
        link_1_Transf = self.MCTs.auto_geomTransf(bent_cap_node[1], pier_1_node[0])
        link_2_Transf = self.MCTs.auto_geomTransf(bent_cap_node[-2], pier_2_node[0])

        ops.element('elasticBeamColumn', 12, *(bent_cap_node[1], pier_1_node[0]),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ke, Ke, Ke, link_1_Transf)
        ops.element('elasticBeamColumn', 13, *(bent_cap_node[-2], pier_2_node[0]),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, Ke, Ke, Ke, link_2_Transf)

        # ops.rigidLink('beam', bent_cap_node[1], pier_1_node[0])
        # ops.rigidLink('beam', bent_cap_node[-2], pier_2_node[0])

        # 约束
        ops.fixZ(0., 1, 1, 1, 1, 1, 1)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # BRB连接节点
        ops.node(3001, 0., PierW / 2 - 0.2, PierH) # 顶部
        closest_node = self.MCTs.node_closest(nodeTag=3001, indexList=bent_cap_node, dim=2)
        # ops.equalDOF(*(closest_node, 3001), *(1, 2, 3, 4, 5, 6)) # 
        # ops.equalDOF(*(3001, closest_node), *(1, 2, 3, 4, 5, 6)) # 
        # ops.rigidLink('beam', closest_node, 3001)
        ops.element('elasticBeamColumn', 111220, *(closest_node, 3001),
                    PierProps.CoverProps.Ec, PierProps.CoverProps.G,
                    0.1, 1.e6, 1.e6, 1.e6, link_2_Transf)
                    # 0.1, self.Ubig, self.Ubig, self.Ubig, link_2_Transf)
        
        ops.node(3002, 0., -PierW / 2 + 0.2, 0.) # 底部
        ops.fix(3002, 1, 1, 1, 1, 1, 1)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 张拉控制力
        axial_force = 377. * UNIT.kn
        # 钢绞线总面积
        PT_area = 2 * (np.pi * (15.2 * UNIT.mm / 2) ** 2)
        # 张拉控制应力
        sigma = axial_force / PT_area

        # 预应力筋 材料
        PT_mat = 10 # 材料标签
        PT_fy = 1860 * UNIT.mpa
        PT_Es = 206 * UNIT.gpa
        PT_ratio = 0.43 * PT_fy # 控制张拉比例
        ops.uniaxialMaterial('Steel02', PT_mat, PT_fy, PT_Es, 0.01, 18, 0.925, 0.15, 0, 1, 0, 1, sigma)

        # 预应力纤维
        # PT_sec = 300
        # ops.section('fiberSec', 300, '-GJ', 100000000)
        # ops.fiber(0., 0., PT_area, PT_mat)

        # 预应力筋单元
        ops.element('Truss', 100, *(bent_cap_node[1], pier_1_node[-1]), PT_area, PT_mat)
        ops.element('Truss', 200, *(bent_cap_node[-2], pier_2_node[-1]), PT_area, PT_mat)
        # ops.element('Truss', 100, *(bent_cap_node[1], pier_1_node[-1]), PT_sec)
        # ops.element('Truss', 200, *(bent_cap_node[-2], pier_2_node[-1]), PT_sec)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 质量密度
        rho = 2600 * (UNIT.kg / (UNIT.m ** 3))  # kg/m3

        # 单位节点质量
        bent_cap_mass = L * BentCapProps.SecMashProps.A * rho / len(bent_cap_node)
        pier_1_mass = PierH * PierProps.SecMashProps.A * rho / len(pier_1_node)
        pier_2_mass = PierH * PierProps.SecMashProps.A * rho / len(pier_2_node)

        # 节点质量
        for i in bent_cap_node:
            ops.mass(i, bent_cap_mass, bent_cap_mass, bent_cap_mass, 0, 0, 0)

        for i in pier_1_node:
            ops.mass(i, pier_1_mass, pier_1_mass, pier_1_mass, 0, 0, 0)

        for i in pier_2_node:
            ops.mass(i, pier_2_mass, pier_2_mass, pier_2_mass, 0, 0, 0)

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
                # BRB 连接节点
                'BRB_i': 3001,
                'BRB_j': 3002,
                }
        KeyEle={
                'pier_1_top': pier_1_ele[0],
                'pier_2_top': pier_2_ele[0],
                'pier_1_base': pier_1_ele[-1],
                'pier_2_base': pier_2_ele[-1],
                }
        LocationDamage=[
                LOCATION_INFO(eleTag=pier_1_ele[0], integ=1, location='col_1_top'),
                LOCATION_INFO(eleTag=pier_2_ele[0], integ=1, location='col_2_top'),
                LOCATION_INFO(eleTag=pier_1_ele[-1], integ=5, location='col_1_base'),
                LOCATION_INFO(eleTag=pier_2_ele[-1], integ=5, location='col_2_base'),
                ]
        OtherOptional=None
        
        return SectionMat, KeyNode, KeyEle, LocationDamage, OtherOptional

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def RCPier(
        self,
        modelPath: str,
        Ke: float = 1.,
        info: bool = True
        ) -> PVs.MODEL_PROPS:

        """
        创建 桥墩 模型
            modelPath: 模型路径
            Ke: 拟合刚度
            info: 是否输出信息
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
        
        # 打印模型到 json
        ops.printModel("-JSON", "-file", f"{modelPath}/{my_name}.json")
        
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
    def RCPierBRB(
        self,
        modelPath: str,
        
        core_ratio: float,
        gap: Union[int, float],
        gapK: float,
        bucklingK: float,
        
        Ke: float = 1.,
        info: bool = True
        ) -> PVs.MODEL_PROPS:

        """
        创建 桥墩 + BRB 模型
            modelPath: 模型路径
            Ke: 拟合刚度
            core_ratio: BRB 核心段长度比
            core_area: BRB 核心面积
            info: 是否输出信息
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
        # top_brb_tag = self._set_BRB_lite(
        #         node_i=key_node['BRB_i'],
        #         node_j=key_node['BRB_j'],
        #         core_ratio=core_ratio, core_area=core_area,
        #         link_E=1.e6,
        #         groupTag=80000
        #         )
        top_brb_tag = self._set_BRB_buckling(
                node_i=key_node['BRB_i'],
                node_j=key_node['BRB_j'],
                groupTag=4000,
                core_ratio=core_ratio,
                gap=gap, gapK=gapK,
                bucklingK=bucklingK,
                )
            
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # BRB 单元号 添加到 单元合集
        key_ele['BRB'] = top_brb_tag
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 可视化输出模型
        opsplt.set_plot_props(point_size=5, line_width=3)
        fig = opsplt.plot_model(
            show_node_numbering=True,
            show_ele_numbering=False,
            show_local_axes=False
            )
        fig.write_html(f"{modelPath}/{my_name}.html", full_html=False, include_plotlyjs="cdn")
        
        # 打印模型到 json
        ops.printModel("-JSON", "-file", f"{modelPath}/{my_name}.json")

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
        ODB_ele_sec = opst.post.get_element_responses(odb_tag=odb_tag, ele_type="FiberSection")

        # 逐个判断 关键部位损伤
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
        ax.set_xlabel('local y')
        ax.set_ylabel('local z')
        
        # 保证数据单位 1:1
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
    def reasp_BRB(self, odb_tag: Union[str, int]):
        # 导入数据
        ODB_BRB_resp = opst.post.get_element_responses(odb_tag=odb_tag, ele_type='Frame', print_info=False)
        # print(ODB_BRB_resp)
        
        # 墩顶位移数据
        disp = np.array(self.reasp_top_disp(odb_tag))
        
        # BRB 轴力 - 变形 数据
        BRB_axialForce = ODB_BRB_resp['basicForces'].sel(eleTags=self.model_props.KeyEle['BRB'], basicDofs='N') # 轴力
        BRB_axialDefo = ODB_BRB_resp['basicDeformations'].sel(eleTags=self.model_props.KeyEle['BRB'], basicDofs='N') # 轴向变形
        
        plt.close('all')
        # BRB
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

        # BRB 轴力 - 变形
        ax1.plot(BRB_axialDefo, BRB_axialForce, zorder=2)
        ax1.set_xlabel('Axial Deformation (m)')
        ax1.set_ylabel('Axial Force (kN)')
        ax1.set_title('BRB Axial Deformation - Axial Force')
        ax1.grid(linestyle='--', linewidth=0.5, zorder=1)

        # BRB 位移 - 轴力
        ax2.plot(disp, BRB_axialDefo, zorder=2)
        ax2.set_xlabel('Displacement (m)')
        ax2.set_ylabel('Axial Deformation (m)')
        ax2.set_title('BRB Displacement - Axial Deformation')
        ax2.grid(linestyle='--', linewidth=0.5, zorder=1)

        plt.tight_layout()

        # # Base BRB
        # fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 8))

        # # Base BRB 变形图
        # ax3.plot(disp, base_BRB_axialDefo, zorder=2)
        # ax3.set_xlabel('Displacement (m)')
        # ax3.set_ylabel('Axial Deformation (m)')
        # ax3.set_title('Base BRB Axial Deformation')
        # ax3.grid(linestyle='--', linewidth=0.5, zorder=1)

        # # Base BRB 轴力图
        # ax4.plot(disp, base_BRB_axialForce, zorder=2)
        # ax4.set_xlabel('Displacement (m)')
        # ax4.set_ylabel('Axial Force (kN)')
        # ax4.set_title('Base BRB Axial Force')
        # ax4.grid(linestyle='--', linewidth=0.5, zorder=1)

        # plt.tight_layout()

        # # 创建DataFrame存储所有数据
        # df_BRB = pd.DataFrame({
        #     'disp': disp,
        #     'top_BRB_strain': np.array(top_BRB_strain),
        #     'top_BRB_stress': np.array(top_BRB_stress),
        #     'top_BRB_axialForce': np.array(top_BRB_axialForce),
        #     'top_BRB_axialDefo': np.array(top_BRB_axialDefo),
        #     'base_BRB_strain': np.array(base_BRB_strain),
        #     'base_BRB_stress': np.array(base_BRB_stress),
        #     'base_BRB_axialForce': np.array(base_BRB_axialForce),
        #     'base_BRB_axialDefo': np.array(base_BRB_axialDefo)
        # })
    
        # return fig1, fig2, df_BRB
        
        return fig1

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    
    # 模型参数路径
    params_path = './OutModel'
    os.makedirs(params_path, exist_ok=True)
    # 实例化模型
    model_params = TwoPierModelTEST()
    
    # 桥墩 模型
    model_params.RCPier(
        modelPath=params_path,
        )
    # 桥墩 + BRB 模型
    model_params.RCPierBRB(
        modelPath=params_path,
        core_ratio=0.43821,
        gap=0.02, gapK=18.e6 * UNIT.pa,
        bucklingK=176.7 * UNIT.gpa,
        )
