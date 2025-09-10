#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional, TypeAlias, Union

import openseespy.opensees as ops
import opstool as opst
import opstool.vis.plotly as opsplt
import opstool.vis.pyvista as opsvis

from script import UNIT
from script import ModelCreateTools
from script import AnalysisTools

"# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
# BRB材料
# ops.uniaxialMaterial('Steel02', 1)

# BRB截面

"# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
def create_brb_lite(
    fix_i: int, fix_j: int,
    core_ratio: float, core_area: float,
    brb_mat_tag: int,
    link_E: float = 1.e6,
    groupTag: int = 1,
    ):

    if core_ratio >=1:
        ValueError("Core ratio should be less than 1.")

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 节点坐标
    i_coord = np.array(ops.nodeCoord(fix_i))
    j_coord = np.array(ops.nodeCoord(fix_j))
    
    # 计算距离
    L = np.linalg.norm(j_coord - i_coord) # 固定端总长度
    core_len = L * core_ratio # 核心段长度
    
    # 计算直线参数
    center_point = (i_coord + j_coord) / 2 # 直线中心坐标
    dir_vector = (j_coord - i_coord) / L # 直线方向向量
    
    # 缩放后的端点坐标
    core_i_coord = center_point - (dir_vector * core_len / 2)
    core_j_coord = center_point + (dir_vector * core_len / 2)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
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
                *(fix_i, core_node_tag_i),
                200 * UNIT.gpa, 100 * UNIT.gpa,
                0.1, link_E, link_E, link_E, MCTs.geomTransf_other())

    ops.element('Truss', core_ele_tag, *[core_node_tag_i, core_node_tag_j], core_area, brb_mat_tag)

    ops.element('elasticBeamColumn', link_ele_tag_j,
                *(core_node_tag_j, fix_j),
                200 * UNIT.gpa, 100 * UNIT.gpa,
                0.1, link_E, link_E, link_E, MCTs.geomTransf_other())




def create_brb(
    fix_i: int, fix_j: int,
    core_ratio: float,
    core_sec_np: int, link_sec_np: int,
    groupTag: int,
    gap: float, gapK: float,
    bulckingK: float,
    ):
    """
    fix_i: 既有的 i 节点号
    fix_j: 既有的 j 节点号
    core_ratio: 核心长度比
    core_sec_np: 核心截面号（积分编号）
    groupTag: 节点单元编号组
    gap: 隙距
    gapE: 隙摩擦刚度
    bulckingE: 屈曲刚度
    """
    """
        fix_i bucklingGap buckling        buckling  bucklingGap fix_j
          * ———— * | * ———— * | * ———— ———— * | * ———— * | * ———— *
                zeroLength zeroLength     zeroLength zeroLength
    """

    if core_ratio >=1:
        ValueError("Core ratio should be less than 1.")

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 节点坐标
    i_coord = np.array(ops.nodeCoord(fix_i))
    j_coord = np.array(ops.nodeCoord(fix_j))
    
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
    Ubig = 20000. * UNIT.gpa
    Usmall = 2.e-8 * UNIT.gpa

    # 约束材料
    fixMat = groupTag + 1
    freeMat = groupTag + 2
    ops.uniaxialMaterial('Elastic', fixMat, Ubig)
    ops.uniaxialMaterial('Elastic', freeMat, Usmall)
    
    # 间隙材料
    gapMat = groupTag + 3
    ops.uniaxialMaterial("HookGap", gapMat, Ubig, -gap, gap) # tag     E    gap_p   gap_n
    # 摩擦材料
    slipMat = groupTag + 4
    ops.uniaxialMaterial('Elastic', slipMat, gapK)
    # 并联材料：gap + slip
    GapModel = groupTag + 5
    ops.uniaxialMaterial('Parallel', GapModel, gapMat, slipMat)
    # 屈曲材料
    bucklingMat = groupTag + 6
    ops.uniaxialMaterial('Elastic', bucklingMat, bulckingK)
    
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
    bucklingGap_mats = [GapModel, fixMat, fixMat, fixMat, bucklingMat, bucklingMat]
    buckling_mats = [fixMat, fixMat, fixMat, fixMat, bucklingMat, bucklingMat]
    # 零长单元 坐标转换
    vecx = dir_vector.tolist()  # 局部x -> 整体方向
    vecyp = MCTs.auto_geomTransf(fix_i, fix_j, vector=True)  # 局部y -> 整体方向
    veczp = MCTs.auto_geomTransf(fix_i, fix_j, vector=True)  # 局部z -> 整体方向
    transfTag = MCTs.auto_geomTransf(fix_i, fix_j)  # 单元坐标转换标签
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 单元
    # I 端
    ops.element('dispBeamColumn', I_link_1, *(fix_i, I_bucklingGap_i), transfTag, link_sec_np)
    ops.element('zeroLength', I_bucklingGap, *(I_bucklingGap_i, I_bucklingGap_j), '-mat', *bucklingGap_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling + gap
    ops.element('dispBeamColumn', I_link_2, *(I_bucklingGap_j, I_buckling_i), transfTag, link_sec_np)
    # # CORE
    ops.element('zeroLength', I_buckling, *(I_buckling_i, I_buckling_j), '-mat', *buckling_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling
    ops.element('dispBeamColumn', coreEle, *(I_buckling_j, J_buckling_i), transfTag, core_sec_np)
    ops.element('zeroLength', J_buckling, *(J_buckling_i, J_buckling_j), '-mat', *buckling_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling
    # # J 端
    ops.element('dispBeamColumn', J_link_1, *(J_buckling_j, J_bucklingGap_i), transfTag, link_sec_np)
    ops.element('zeroLength', J_bucklingGap, *(J_bucklingGap_i, J_bucklingGap_j), '-mat', *bucklingGap_mats, '-dir', *dirs, '-orient', *vecx, *vecyp) # buckling + gap
    ops.element('dispBeamColumn', J_link_2, *(J_bucklingGap_j, fix_j), transfTag, link_sec_np)



if __name__ == '__main__':
    
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # 建模工具
    MCTs = ModelCreateTools()
    # 分析工具
    ATs = AnalysisTools()
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # BRB材料
    Q235 = 10
    Q235_fy = 309.57 * UNIT.mpa
    Q235_Es = 176.7 * UNIT.gpa
    ops.uniaxialMaterial('Steel02', Q235, Q235_fy, Q235_Es, 0.01, 18, 0.925, 0.15)

    # 截面
    secCol = 100
    ops.section('fiberSec', secCol, '-GJ', 1000000)
    ops.patch('circ', Q235, 20, 10, 0.0, 0.0, 0., 0.1, 0.0, 360.0) # 面积很小，没什么力
    
    secBRB = 200
    ops.section('fiberSec', secBRB, '-GJ', 1000000)
    ops.patch('circ', Q235, 20, 10, 0.0, 0.0, 0., 2., 0.0, 360.0) # 面积为10
    secLink = 201
    ops.section('fiberSec', secLink, '-GJ', 1000000)
    ops.patch('circ', Q235, 20, 10, 0.0, 0.0, 0., 10., 0.0, 360.0) # 面积为10

    # 单元积分点
    npCol = 1
    ops.beamIntegration('Legendre', npCol, secCol, 5)
    npBRB = 2
    ops.beamIntegration('Legendre', npBRB, secBRB, 5)
    npLink = 3
    ops.beamIntegration('Legendre', npLink, secLink, 5)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    ops.node(1, 0., 0., 0.)
    ops.node(2, 0., 0., 5.)
    MCTs.ele_create(0, [(1, 2)], npCol)
    
    # 创建测试节点
    ops.node(1001, 0., 5., 0.)
    ops.fixZ(0., 1, 1, 1, 1, 1, 1)

    # BRB部分
    create_brb(
        fix_i=1001,
        fix_j=2,
        core_ratio=0.3,
        core_sec_np=npBRB,
        link_sec_np=npLink,

        groupTag=9000,
        gap=0.01,
        gapK=2.e-4 * UNIT.gpa,
        bulckingK=2000. * UNIT.gpa,
        )

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    opsplt.set_plot_props(point_size=5, line_width=3)
    fig = opsplt.plot_model(
        show_node_numbering=False,
        show_ele_numbering=False,
        show_local_axes=True,
        local_axes_scale=0.5
        )
    fig.show(renderer="browser")




    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 测试数据路径
    data_path = './TEST'
    os.makedirs(data_path, exist_ok=True)
    
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)
    
    # 响应数据库
    opst.post.set_odb_path(data_path)
    ODB = opst.post.CreateODB(odb_tag=1)
    
    # 重力荷载工况
    grav_pattern = 100
    ops.pattern("Plain", grav_pattern, ts)
    g = 9.80665 * (UNIT.m / UNIT.sec ** 2)
    opst.pre.create_gravity_load(direction='Z', factor=-g)  # 从整体模型的节点质量获取重力荷载
    # 重力分析
    ATs.GRAVITY(filepath=data_path, RESP_ODB=ODB)
    
    # 控制节点
    ctrl_node = 2
    # 单位控制荷载
    F = 1.
    # 控制荷载工况
    static_pattern = 200
    ops.pattern("Plain", static_pattern, ts)
    # ops.load(ctrl_node, 0.0, 0.0, F, 0.0, 0.0, 0.0)  # 节点荷载
    ops.load(ctrl_node, 0.0, F, 0.0, 0.0, 0.0, 0.0)  # 节点荷载

    disp_path = np.array([
            0.,
            # 0.01, -0.01,
            # 0.02, -0.02,
            # 0.03, -0.03,
            # 0.04, -0.04,
            # 0.05, -0.05,
            # 0.06, -0.06,
            # 0.07, -0.07,
            # 0.08, -0.08,
            # 0.09, -0.09,
            # 0.10, -0.10,
            0.10, -0.10,
            0.20, -0.20,
            0.40, -0.40,
            0.60, -0.60,
            0.
        ]) * UNIT.m

    disp, load = ATs.STATIC(
        filepath=data_path,
        pattern=static_pattern,
        ctrl_node=ctrl_node,
        protocol=disp_path,
        incr=0.01,
        direction=2,
        RESP_ODB=ODB
        )

    # 保存数据库
    ODB.save_response(zlib=True)

    # 荷载-位移 曲线
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'Displacement-Load Curve')
    plt.plot(disp, load, alpha=1, linewidth=0.8, label='FEM', zorder=3) # FEM
    plt.xlabel('Displacement (m)')
    plt.ylabel('Load (kN)')
    plt.xlim(-np.max(np.abs(disp)) * 1.2, np.max(np.abs(disp) * 1.2))
    plt.ylim(-np.max(np.abs(load)) * 1.2, np.max(np.abs(load)) * 1.2)
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    plt.show()

    fig = opsplt.plot_nodal_responses_animation(
        odb_tag=1,
        # slides=True,
        resp_type="disp",
        resp_dof=["UX", "UY", "UZ"],
        show_outline=True,
        defo_scale=1.0
    )
    # fig.show(renderer="browser")




