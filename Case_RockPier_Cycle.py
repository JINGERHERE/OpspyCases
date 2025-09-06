#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Case_RockPier_Cycle.py
@Date    ：2025/8/1 19:20
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""


import os
import time

import numpy as np
import pandas as pd
import opstool as opst
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import rich

import opstool.vis.plotly as opsplt
import opstool.vis.pyvista as opsvis

from script import UNIT
from script import AnalysisTools as ATs
from script.base import random_color

from Part_Model_RockPierModel import RockPierModelTEST


import multiprocessing as mulp
from joblib import Parallel, delayed
from rich.progress import Progress, BarColumn, TimeElapsedColumn


"""
# --------------------------------------------------
# ========== < Case_RockPier_Cycle > ==========
# --------------------------------------------------
"""


"# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
# 试验原始数据
test_data_path = './.RAW_DATA'
data_file = '/SCB.xlsx'
# data_file = '/SCB_EDB.xlsx'
# 导入
test_data = pd.read_excel(
    f'{test_data_path + data_file}',
    # header=0,
)
# 清洗数据 转换为数值
test_data['m'] = pd.to_numeric(test_data['m'], errors='coerce')
test_data['kN']  = pd.to_numeric(test_data['kN'],  errors='coerce')

"# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
def ANALYSIS_CASE(CASE: float):

    """
    工况函数：
        输入：截面， 对应的分析工况
        执行：输入截面 对应方向的 弯矩曲率分析
        返回：-
    """

    root_path = './OutData'
    case_path = os.path.join(root_path, f'{CASE:.3e}')
    os.makedirs(case_path, exist_ok=True)

    # 实例化模型
    Model = RockPierModelTEST()

    # 模型参数
    params = {
        'Ke': CASE,
        'info': False
    }

    # 创建模型
    ModelProps = Model.RockPier(case_path, **params)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 响应数据库
    opst.post.set_odb_path(case_path)
    ODB = opst.post.CreateODB(
        odb_tag=f"{CASE:.3e}",
        elastic_frame_sec_points=9,
        node_tags=None,
        frame_tags=None,
        fiber_ele_tags="ALL"
        )

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 重力荷载工况
    grav_pattern = 100
    ops.pattern("Plain", grav_pattern, ts)
    g = 9.80665 * (UNIT.m / UNIT.sec ** 2)
    opst.pre.create_gravity_load(direction='Z', factor=-g)  # 从整体模型的节点质量获取重力荷载
    # 重力分析
    ATs.GRAVITY(filepath=case_path, RESP_ODB=ODB)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 控制节点
    ctrl_node = ModelProps.KeyNode
    # 单位控制荷载
    F = 1.
    # 控制荷载工况
    static_pattern = 200
    ops.pattern("Plain", static_pattern, ts)
    ops.load(ctrl_node, 0.0, F, 0.0, 0.0, 0.0, 0.0)  # 节点荷载

    # 位移路径
    disp_path = 0.12 * UNIT.m
    disp_path = np.array([
            0.,
            0.01, -0.01,
            0.02, -0.02,
            0.03, -0.03,
            0.04, -0.04,
            0.05, -0.05,
            0.06, -0.06,
            0.08, -0.08,
            0.10, -0.10,
            0.12, -0.12,
            0.
        ]) * UNIT.m

    # 静力分析
    disp, load = ATs.STATIC(
        filepath=case_path,
        pattern=static_pattern,
        ctrl_node=ctrl_node,
        protocol=disp_path,
        incr=0.005,
        direction=2,
        RESP_ODB=ODB
        )

    # 荷载-位移 曲线
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'{ModelProps.Name} Displacement-Load Curve')
    plt.plot(test_data['m']*UNIT.m, test_data['kN']*UNIT.kn, linewidth=0.8, label='TEST', zorder=2) # 实验数据
    plt.plot(disp, load, alpha=1, linewidth=0.8, label='FEM', zorder=3) # FEM
    plt.xlabel('Displacement (m)')
    plt.ylabel('Load (kN)')
    plt.xlim(-np.max(np.abs(disp)) * 1.2, np.max(np.abs(disp) * 1.2))
    plt.ylim(-np.max(np.abs(load)) * 1.2, np.max(np.abs(load)) * 1.2)
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    # plt.show()
    plt.savefig(f'{case_path}/disp_load.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 保存数据库
    ODB.save_response(zlib=True)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 损伤判断
    StrucDS = Model.determine_damage(odb_tag=f'{CASE:.3e}', info=params['info'])
    StrucDS.to_excel(f'{case_path}/{ModelProps.Name}_damage.xlsx', index=False)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 截面状态
    fig = Model.reasp_fiber_sec(odb_tag=f'{CASE:.3e}', ele_tag=1101, integ=1, step=-1)
    fig.savefig(f'{case_path}/fiber_sec_state.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 预应力
    fig = Model.reasp_PT_force(odb_tag=f'{CASE:.3e}')
    fig.savefig(f'{case_path}/PT_bar_Axial_Force.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {ModelProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')

    # opsvis.set_plot_props(point_size=5, line_width=3)
    # fig = opsvis.plot_nodal_responses(
    # # fig = opsvis.plot_nodal_responses_animation(
    #     odb_tag=f"{CASE:.3e}",
    #     # slides=True,
    #     resp_type="disp",
    #     resp_dof=["UX", "UY", "UZ"],
    #     cpos = 'yz'
    # )
    # fig.show()

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    # 截面列表
    CASE_LIST = [
        0.050 * UNIT.pa,
        0.050e2 * UNIT.pa,
    ]

    # 是否启用并行计算
    if len(CASE_LIST) >= 2:
        PARALLEL: bool = True
    else:
        PARALLEL: bool = False

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    START = time.time() # 计时

    if PARALLEL:
        '''并行计算：joblib模块'''
        # CPU核心数
        n_cpu = mulp.cpu_count()
        rich.print(f'# 计算机核心数：{n_cpu}')

        # 并行计算
        Parallel(n_jobs=-1)(
            delayed(ANALYSIS_CASE)(case) for case in CASE_LIST
        )

    else:
        '''正常计算：for循环'''
        for case in CASE_LIST:
            ANALYSIS_CASE(case)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印完成信息
    COLOR = random_color()
    rich.print(f'[bold {COLOR}] :tada: DONE: All Model Analyze Successfully ! :tada: [/bold {COLOR}]')
    rich.print(f'[bold {COLOR}] # ===== ===== ===== ===== << END >> ===== ===== ===== ===== # [/bold {COLOR}]\n')

    rich.print(f'总用时：{time.time() - START} s')
    

