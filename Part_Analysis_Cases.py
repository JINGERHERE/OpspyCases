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

"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
def CASE_MODEL(
    ROOT_PATH: str, # ./OutData
    
    Ke: float = 1.,
    # DISP: float = 0.02,
    # FORCE: float = 50.,
    # CORE_RATIO: float = 0.2,
    # CORE_AREA: float = 320 * (UNIT.mm**2),
    
    CYCLE_MODE: bool = False,
    ):

    """
    工况函数：
        输入：截面， 对应的分析工况
        执行：输入截面 对应方向的 弯矩曲率分析
        返回：基本性能参数
    """
    
    # 静力分析 模式判断
    if CYCLE_MODE:
        # 工况文件夹命名
        case_file_name = f'CYCLE' 
        # case_file_name = f'CYCLE_Ke={Ke:.3e}' 
        
        # 位移路径
        disp_path = np.array([
                0.,
                # 0.001, -0.001,
                # 0.001, -0.001,
                # 0.001, -0.001,
                # 0.003, -0.003,
                # 0.003, -0.003,
                # 0.003, -0.003,
                # 0.006, -0.006,
                # 0.006, -0.006,
                # 0.006, -0.006,
                # 0.009, -0.009,
                # 0.009, -0.009,
                # 0.009, -0.009,
                # 0.012, -0.012,
                # 0.012, -0.012,
                # 0.012, -0.012,
                # 0.015, -0.015,
                # 0.015, -0.015,
                # 0.015, -0.015,
                # 0.0225, -0.0225,
                # 0.0225, -0.0225,
                # 0.0225, -0.0225,
                # 0.0300, -0.0300,
                # 0.0300, -0.0300,
                # 0.0300, -0.0300,
                # 0.0375, -0.0375,
                # 0.0375, -0.0375,
                # 0.0375, -0.0375,
                # 0.0450, -0.0450,
                # 0.0450, -0.0450,
                # 0.0450, -0.0450,
                # 0.0525, -0.0525,
                # 0.0525, -0.0525,
                # 0.0525, -0.0525,
                # 0.0600, -0.0600,
                # 0.0600, -0.0600,
                # 0.0600, -0.0600,
                # 0.0675, -0.0675,
                # 0.0675, -0.0675,
                # 0.0675, -0.0675,
                # 0.0750, -0.0750,
                # 0.0750, -0.0750,
                # 0.0750, -0.0750,
                # 0.0825, -0.0825,
                # 0.0825, -0.0825,
                # 0.0825, -0.0825,
                # 0.0900, -0.0900,
                # 0.0900, -0.0900,
                # 0.0900, -0.0900,
                # 0.0975, -0.0975,
                # 0.0975, -0.0975,
                # 0.0975, -0.0975,
                # 0.1050, -0.1050,
                # 0.1050, -0.1050,
                # 0.1050, -0.1050,
                # 0.1125, -0.1125,
                # 0.1125, -0.1125,
                # 0.1125, -0.1125,
                # 0.1200, -0.1200,
                # 0.1200, -0.1200,
                0.1200, -0.1200,
                0.
            ]) * UNIT.m
        
    else:
        # 工况文件夹命名
        case_file_name = f'PUSH'
        # case_file_name = f'PUSH_Ke={Ke:.3e}'
        
        # 位移路径
        disp_path = 0.12 * UNIT.m
    
    # 创建工况路径
    case_path = os.path.join(ROOT_PATH, case_file_name)
    os.makedirs(case_path, exist_ok=True)
    # 设置数据库文件夹
    opst.post.set_odb_path(case_path)

    # 实例化模型
    Model = RockPierModelTEST()

    # 模型参数
    params = {
        'Ke': Ke,
        
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
    ODB = opst.post.CreateODB(
        odb_tag=case_file_name,
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
    ctrl_node = ModelProps.KeyNode['ctrl_node']
    # 单位控制荷载
    F = 1.
    # 控制荷载工况
    static_pattern = 200
    ops.pattern("Plain", static_pattern, ts)
    ops.load(ctrl_node, 0.0, F, 0.0, 0.0, 0.0, 0.0)  # 节点荷载

    # 静力分析
    disp, load = ATs.STATIC(
        filepath=case_path,
        pattern=static_pattern,
        ctrl_node=ctrl_node,
        protocol=disp_path,
        incr=0.001,
        direction=2,
        RESP_ODB=ODB
        )

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 保存数据库
    ODB.save_response(zlib=True)
    
    # 导出数据 荷载-位移
    out_disp_load = pd.DataFrame({'x': disp, 'y': load})
    out_disp_load.to_excel(f'{case_path}/Disp_Load.xlsx', index=False)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 损伤判断
    StrucDS = Model.determine_damage(odb_tag=case_file_name, info=params['info'])
    StrucDS.to_excel(f'{case_path}/{ModelProps.Name}_damage.xlsx', index=False)
    # disp_yield = disp[int(StrucDS.index.min()) - 10]
    # load_yield = load[int(StrucDS.index.min()) - 10]
    # yield_df = pd.DataFrame({'yield_disp (m)': disp_yield, 'yield_force (kN)': load_yield}, index=[0])
    # yield_df.to_csv(f'{case_path}/Yield.txt', sep='\t', index=False, encoding='utf-8')

    # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # # 截面状态
    # fig = Model.reasp_fiber_sec(odb_tag=case_file_name, ele_tag=ModelProps.KeyEle['ENT_sec'], integ=1, step=-1)
    # fig.savefig(f'{case_path}/fiber_sec_state.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 接触面纤维截面
    fig = Model.show_fiber_sec(ele_tag=ModelProps.KeyEle['Pier_1_ENT_sec'])
    fig.savefig(f'{case_path}/ENT_sec.png', dpi=300, bbox_inches='tight')
    
    # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # # 接触面节点竖向位移
    # fig = Model.reasp_seg_node_disp_UZ(odb_tag=case_file_name)
    # fig.savefig(f'{case_path}/pier_1_seg_disp_test.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 预应力
    fig = Model.reasp_PT_force(odb_tag=case_file_name)
    fig.savefig(f'{case_path}/PT_bar_Axial_Force.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 耗能钢筋
    fig_1, fig_2, df_ED = Model.reasp_ED_stress_strain(odb_tag=case_file_name)
    fig_1.savefig(f'{case_path}/Pier_1_ED_bar_Stress_Strain.png', dpi=300, bbox_inches='tight')
    fig_2.savefig(f'{case_path}/Pier_2_ED_bar_Stress_Strain.png', dpi=300, bbox_inches='tight')
    df_ED.to_excel(f'{case_path}/ED_bar_data.xlsx')
    
    # 耗能钢筋屈服判断
    yield_step = Model.yield_ED(odb_tag=case_file_name)
    disp_yield = disp[yield_step - 10]
    load_yield = load[yield_step - 10]
    yield_df = pd.DataFrame({'yield_disp (m)': disp_yield, 'yield_force (kN)': load_yield}, index=[0])
    yield_df.to_csv(f'{case_path}/Yield.txt', sep='\t', index=False, encoding='utf-8')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 摇摆点
    fig_1, fig_2, rock_start = Model.rock_start(odb_tag=case_file_name)
    fig_1.savefig(f'{case_path}/Pier_1_Rock_Start.png', dpi=300, bbox_inches='tight')
    fig_2.savefig(f'{case_path}/Pier_2_Rock_Start.png', dpi=300, bbox_inches='tight')
    disp_rock = disp[rock_start - 10]
    load_rock = load[rock_start - 10]
    rock_df = pd.DataFrame({'rock_disp (m)': disp_rock, 'rock_force (kN)': load_rock}, index=[0])
    rock_df.to_csv(f'{case_path}/Rock.txt', sep='\t', index=False, encoding='utf-8')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 荷载-位移 曲线
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'{ModelProps.Name} Displacement-Load Curve')
    
    plt.plot(
        test_data['m']*UNIT.m, test_data['kN']*UNIT.kn,
        linewidth=0.4,
        color='gray', linestyle='--',
        label='TEST', zorder=2) # 实验数据
    
    plt.plot(disp, load, alpha=1, linewidth=0.8, label='FEM', zorder=3) # FEM
    
    plt.scatter(disp_yield, load_yield,
                color = 'red', alpha=1,
                marker='o', s=15,
                label='Yield', zorder=3) # 屈服点
    plt.scatter(disp_rock, load_rock,
                color = 'yellow', alpha=1,
                marker='^', s=15,
                label='Rock Start', zorder=3) # 摇摆点
    
    plt.xlabel('Displacement (m)')
    plt.ylabel('Load (kN)')
    plt.xlim(-np.max(np.abs(disp)) * 1.2, np.max(np.abs(disp) * 1.2))
    plt.ylim(-np.max(np.abs(load)) * 1.2, np.max(np.abs(load)) * 1.2)
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    # plt.show()
    plt.savefig(f'{case_path}/Disp_Load.png', dpi=300, bbox_inches='tight')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {ModelProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')
    
    return disp_yield, load_yield

"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
def CASE_MODEL_BRB(
    ROOT_PATH: str, # ./OutData
    
    Ke: float = 1.,
    DISP: float = 0.02,
    FORCE: float = 50.,
    CORE_RATIO: float = 0.2,
    CORE_AREA: float = 320 * (UNIT.mm**2),
    
    CYCLE_MODE: bool = False,
    ):

    """
    工况函数：
        输入：截面， 对应的分析工况
        执行：输入截面 对应方向的 弯矩曲率分析
        返回：基本性能参数
    """
    
    # 静力分析 模式判断
    if CYCLE_MODE:
        # 工况文件夹命名
        case_file_name = f'CYCLE_BRB_a={CORE_AREA:.3e}_c={CORE_RATIO:.2f}'
        # case_file_name = f'CYCLE_BRB_a={CORE_AREA:.3e}_c={CORE_RATIO}_Ke={Ke:.3e}' 
        
        # 位移路径
        disp_path = np.array([
                0.,
                0.001, -0.001,
                0.001, -0.001,
                0.001, -0.001,
                0.003, -0.003,
                0.003, -0.003,
                0.003, -0.003,
                0.006, -0.006,
                0.006, -0.006,
                0.006, -0.006,
                0.009, -0.009,
                0.009, -0.009,
                0.009, -0.009,
                0.012, -0.012,
                0.012, -0.012,
                0.012, -0.012,
                0.015, -0.015,
                0.015, -0.015,
                0.015, -0.015,
                0.0225, -0.0225,
                0.0225, -0.0225,
                0.0225, -0.0225,
                0.0300, -0.0300,
                0.0300, -0.0300,
                0.0300, -0.0300,
                0.0375, -0.0375,
                0.0375, -0.0375,
                0.0375, -0.0375,
                0.0450, -0.0450,
                0.0450, -0.0450,
                0.0450, -0.0450,
                0.0525, -0.0525,
                0.0525, -0.0525,
                0.0525, -0.0525,
                0.0600, -0.0600,
                0.0600, -0.0600,
                0.0600, -0.0600,
                0.0675, -0.0675,
                0.0675, -0.0675,
                0.0675, -0.0675,
                0.0750, -0.0750,
                0.0750, -0.0750,
                0.0750, -0.0750,
                0.0825, -0.0825,
                0.0825, -0.0825,
                0.0825, -0.0825,
                0.0900, -0.0900,
                0.0900, -0.0900,
                0.0900, -0.0900,
                0.0975, -0.0975,
                0.0975, -0.0975,
                0.0975, -0.0975,
                0.1050, -0.1050,
                0.1050, -0.1050,
                0.1050, -0.1050,
                0.1125, -0.1125,
                0.1125, -0.1125,
                0.1125, -0.1125,
                0.1200, -0.1200,
                0.1200, -0.1200,
                0.1200, -0.1200,
                0.
            ]) * UNIT.m
        
    else:
        # 工况文件夹命名
        case_file_name = f'PUSH_BRB_a={CORE_AREA:.3e}_c={CORE_RATIO:.2f}'
        # case_file_name = f'PUSH_BRB_a={CORE_AREA:.3e}_c={CORE_RATIO}_Ke={Ke:.3e}'
        
        # 位移路径
        disp_path = 0.12 * UNIT.m
    
    # 创建工况文件夹
    case_path = os.path.join(ROOT_PATH, case_file_name)
    os.makedirs(case_path, exist_ok=True)
    # 设置数据库文件夹
    opst.post.set_odb_path(case_path)

    # 实例化模型
    Model = RockPierModelTEST()

    # 模型参数
    params = {
        'Ke': Ke,
        
        'yield_disp': DISP,
        'yield_force': FORCE,
        'core_ratio': CORE_RATIO,
        'core_area': CORE_AREA,
        
        'info': False
    }

    # 创建模型
    # ModelProps = Model.RockPier(case_path, **params)
    ModelProps = Model.RockPierBRB(case_path, **params)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 响应数据库
    ODB = opst.post.CreateODB(
        odb_tag=case_file_name,
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
    ctrl_node = ModelProps.KeyNode['ctrl_node']
    # 单位控制荷载
    F = 1.
    # 控制荷载工况
    static_pattern = 200
    ops.pattern("Plain", static_pattern, ts)
    ops.load(ctrl_node, 0.0, F, 0.0, 0.0, 0.0, 0.0)  # 节点荷载

    # 静力分析
    disp, load = ATs.STATIC(
        filepath=case_path,
        pattern=static_pattern,
        ctrl_node=ctrl_node,
        protocol=disp_path,
        incr=0.001,
        direction=2,
        RESP_ODB=ODB
        )

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 保存数据库
    ODB.save_response(zlib=True)
    
    # 导出数据 荷载-位移
    out_disp_load = pd.DataFrame({'x': disp, 'y': load})
    out_disp_load.to_excel(f'{case_path}/Disp_Load.xlsx', index=False)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 损伤判断
    StrucDS = Model.determine_damage(odb_tag=case_file_name, info=params['info'])
    StrucDS.to_excel(f'{case_path}/{ModelProps.Name}_damage.xlsx', index=False)
    # disp_yield = disp[int(StrucDS.index.min()) - 10]
    # load_yield = load[int(StrucDS.index.min()) - 10]
    # yield_df = pd.DataFrame({'yield_disp (m)': disp_yield, 'yield_force (kN)': load_yield}, index=[0])
    # yield_df.to_csv(f'{case_path}/Yield.txt', sep='\t', index=False, encoding='utf-8')

    # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # # 截面状态
    # fig = Model.reasp_fiber_sec(odb_tag=case_file_name, ele_tag=ModelProps.KeyEle['ENT_sec'], integ=1, step=-1)
    # fig.savefig(f'{case_path}/fiber_sec_state.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 接触面纤维截面
    fig = Model.show_fiber_sec(ele_tag=ModelProps.KeyEle['Pier_1_ENT_sec'])
    fig.savefig(f'{case_path}/ENT_sec.png', dpi=300, bbox_inches='tight')
    
    # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # # 接触面节点竖向位移
    # fig = Model.reasp_seg_node_disp_UZ(odb_tag=case_file_name)
    # fig.savefig(f'{case_path}/pier_1_seg_disp_test.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 预应力
    fig = Model.reasp_PT_force(odb_tag=case_file_name)
    fig.savefig(f'{case_path}/PT_bar_Axial_Force.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 耗能钢筋
    fig_1, fig_2, df_ED = Model.reasp_ED_stress_strain(odb_tag=case_file_name)
    fig_1.savefig(f'{case_path}/Pier_1_ED_bar_Stress_Strain.png', dpi=300, bbox_inches='tight')
    fig_2.savefig(f'{case_path}/Pier_2_ED_bar_Stress_Strain.png', dpi=300, bbox_inches='tight')
    df_ED.to_excel(f'{case_path}/ED_bar_data.xlsx')
    
    # 耗能钢筋屈服判断
    yield_step = Model.yield_ED(odb_tag=case_file_name)
    disp_yield = disp[yield_step - 10]
    load_yield = load[yield_step - 10]
    yield_df = pd.DataFrame({'yield_disp (m)': disp_yield, 'yield_force (kN)': load_yield}, index=[0])
    yield_df.to_csv(f'{case_path}/Yield.txt', sep='\t', index=False, encoding='utf-8')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # BRB
    fig_1, fig_2, BRB_data = Model.reasp_BRB(odb_tag=case_file_name)
    fig_1.savefig(f'{case_path}/top_BRB.png', dpi=300, bbox_inches='tight')
    fig_2.savefig(f'{case_path}/base_BRB.png', dpi=300, bbox_inches='tight')
    BRB_data.to_excel(f'{case_path}/BRB_data.xlsx')
    
    # BRB屈服判断
    yield_step_brb = Model.yield_BRB(odb_tag=case_file_name, min_step=False)
    # 顶部 BRB
    top_disp_brb_yield = disp[yield_step_brb[0] - 10]
    top_load_brb_yield = load[yield_step_brb[0] - 10]
    # top_stiff_brb_yield = top_load_brb_yield / top_disp_brb_yield
    # 底部 BRB
    base_disp_brb_yield = disp[yield_step_brb[1] - 10]
    base_load_brb_yield = load[yield_step_brb[1] - 10]
    # base_stiff_brb_yield = base_load_brb_yield / base_disp_brb_yield
    
    # BRB 整体刚度贡献
    if top_disp_brb_yield <= base_disp_brb_yield:
        stiffness_contribution = top_load_brb_yield / top_disp_brb_yield
    else:
        stiffness_contribution = base_load_brb_yield / base_disp_brb_yield
    
    brb_yield_df = pd.DataFrame({
        '屈服位移 (m)': [top_disp_brb_yield, base_disp_brb_yield],
        '屈服时整体强度 (kN)': [top_load_brb_yield, base_load_brb_yield],
        })
    brb_yield_df.to_csv(f'{case_path}/BRB_Yield.txt', sep='\t', index=False, encoding='utf-8')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 摇摆点
    fig_1, fig_2, rock_start = Model.rock_start(odb_tag=case_file_name)
    fig_1.savefig(f'{case_path}/Pier_1_Rock_Start.png', dpi=300, bbox_inches='tight')
    fig_2.savefig(f'{case_path}/Pier_2_Rock_Start.png', dpi=300, bbox_inches='tight')
    disp_rock = disp[rock_start - 10]
    load_rock = load[rock_start - 10]
    rock_df = pd.DataFrame({'rock_disp (m)': disp_rock, 'rock_force (kN)': load_rock}, index=[0])
    rock_df.to_csv(f'{case_path}/Rock.txt', sep='\t', index=False, encoding='utf-8')
    
    # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # # BRB对比 设计 - 模拟
    # df_compare = pd.DataFrame({
    #     'DESIGN': [
    #         DISP,
    #         FORCE,
            
    #         # ModelProps.OtherOptional[
    #         #     'top_brb_yield_force'
    #         #     ] + ModelProps.OtherOptional[
    #         #         'base_brb_yield_force'
    #         #         ],
    #         ModelProps.OtherOptional[
    #             'top_brb_yield_stiff'
    #             ] + ModelProps.OtherOptional[
    #                 'base_brb_yield_stiff'
    #                 ],
    #         ModelProps.OtherOptional['BRB_indicator_alpha'],
            
    #         ModelProps.OtherOptional['top_brb_yield_disp'],
    #         ModelProps.OtherOptional['base_brb_yield_disp'],
    #         ModelProps.OtherOptional['top_BRB_indicator_miu'],
    #         ModelProps.OtherOptional['base_BRB_indicator_miu']
    #         ],
    #     'FEM': [
    #         disp_yield,
    #         load_yield,
            
    #         # top_load_brb_yield + base_load_brb_yield,
    #         stiffness_contribution,
    #         stiffness_contribution / (FORCE / DISP),
            
    #         top_disp_brb_yield,
    #         base_disp_brb_yield,
    #         disp_yield / top_disp_brb_yield,
    #         disp_yield / base_disp_brb_yield,
    #         ],
    #     }, index=[
    #         '桥墩屈服位移',
    #         '桥墩屈服荷载',
            
    #         # '整体_BRB_强度贡献',
    #         '整体_BRB_刚度贡献',
    #         '整体_BRB_刚度比',
            
    #         '顶部_BRB_屈服位移',
    #         '底部_BRB_屈服位移',
    #         '顶部_BRB_位移比',
    #         '底部_BRB_位移比',
    #         ]
    #     )
    # # 计算误差
    # df_compare['ERROR(%)'] = (df_compare['DESIGN'].abs() / df_compare['FEM'].abs() - 1) * 100
    # df_compare['ERROR_REMARK'] = [
    #     '整体变化幅度',
    #     '整体变化幅度',
        
    #     # '仅考虑所有BRB屈服前',
    #     '仅考虑所有BRB屈服前',
    #     '仅考虑所有BRB屈服前',
        
    #     '',
    #     '',
    #     '',
    #     '',
    #     ]
    # # 导出数据
    # df_compare.to_excel(f'{case_path}/模拟验证设计.xlsx', index=True)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 荷载-位移 曲线
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'{ModelProps.Name} Displacement-Load Curve')
    
    plt.plot(
        test_data['m']*UNIT.m, test_data['kN']*UNIT.kn,
        linewidth=0.4,
        color='gray', linestyle='--',
        label='TEST', zorder=2) # 实验数据
    
    plt.plot(disp, load, alpha=1, linewidth=0.8, label='FEM', zorder=3) # FEM
    
    plt.scatter(disp_yield, load_yield,
                color = 'red', alpha=1,
                marker='o', s=15,
                label='Yield', zorder=3) # 屈服点
    plt.scatter(disp_rock, load_rock,
                color = 'yellow', alpha=1,
                marker='^', s=15,
                label='Rock Start', zorder=3) # 摇摆点
    plt.scatter(top_disp_brb_yield, top_load_brb_yield,
                # color = 'yellow', alpha=1,
                marker='*', s=15,
                label='Top BRB Yield', zorder=3) # BRB屈服点
    plt.scatter(base_disp_brb_yield, base_load_brb_yield,
                # color = 'yellow', alpha=1,
                marker='*', s=15,
                label='Base BRB Yield', zorder=3) # BRB屈服点
    
    plt.xlabel('Displacement (m)')
    plt.ylabel('Load (kN)')
    plt.xlim(-np.max(np.abs(disp)) * 1.2, np.max(np.abs(disp) * 1.2))
    plt.ylim(-np.max(np.abs(load)) * 1.2, np.max(np.abs(load)) * 1.2)
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    # plt.show()
    plt.savefig(f'{case_path}/Disp_Load.png', dpi=300, bbox_inches='tight')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {ModelProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')
    
    # return disp_yield, load_yield



"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    
    # 根目录
    root_file = f'./OutData'
    
    # 工况列表
    CASE_LIST = [
        {'ROOT_PATH': root_file, 'Ke':  1., 'CYCLE_MODE': False,},
        {'ROOT_PATH': root_file, 'Ke':  1., 'CYCLE_MODE': True,},
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
            delayed(CASE_MODEL)(**case) for case in CASE_LIST
        )

    else:
        '''正常计算：for循环'''
        for case in CASE_LIST:
            CASE_MODEL(**case)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印完成信息
    COLOR = random_color()
    rich.print(f'[bold {COLOR}] :tada: DONE: All Model Analyze Successfully ! :tada: [/bold {COLOR}]')
    rich.print(f'[bold {COLOR}] # ===== ===== ===== ===== << END >> ===== ===== ===== ===== # [/bold {COLOR}]\n')

    rich.print(f'总用时：{time.time() - START} s')
    

