#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    Part_Analysis_Cases.py
@Date    ：2025/8/1 19:02
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

"""
# --------------------------------------------------
# ========== < Part_Analysis_Cases > ==========
# --------------------------------------------------
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

import script
from script import UNIT
from script import AnalysisTools as ATs
from script.base import random_color

from Part_Model_TwoPierModel import TwoPierModelTEST

import multiprocessing as mulp
from joblib import Parallel, delayed
from rich.progress import Progress, BarColumn, TimeElapsedColumn


"# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
# 试验原始数据
test_data_path = './.RAW_DATA'
data_file = '/20230821WJRC.xlsx'
data_file_BRB = '/20230826WJBRB.xlsx'
# 导入
test_data = pd.read_excel(f'{test_data_path + data_file}', header=0)
test_data_BRB = pd.read_excel(f'{test_data_path + data_file_BRB}', header=0)
# 清洗数据 转换为数值
test_data['mm'] = pd.to_numeric(test_data['mm'], errors='coerce') * UNIT.mm
test_data['N']  = pd.to_numeric(test_data['N'],  errors='coerce') * UNIT.n
test_data_BRB['mm'] = pd.to_numeric(test_data_BRB['mm'], errors='coerce') * UNIT.mm
test_data_BRB['N']  = pd.to_numeric(test_data_BRB['N'],  errors='coerce') * UNIT.n

"# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
def CASE_MODEL(
    ROOT_PATH: str, # ./OutData
    
    Ke: float = 1.,
    
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
        disp_step_1 = np.arange(0.002, 0.012 + 0.002, 0.002) # 第一阶段 控制位移幅值
        disp_step_2 = np.arange(0.016, 0.064 + 0.004, 0.004) # 第二节段 控制唯一幅值
        disp_step = np.repeat(
            np.concatenate((disp_step_1, disp_step_2)), # 合并
            3 # 重复三次
            )
        disp_pairs = np.stack((disp_step, -disp_step), axis=1).flatten() # 控制位移成对，整理
        disp_path = np.concatenate(([0.], disp_pairs, [0.])) * UNIT.m # 添加首尾
    else:
        # 工况文件夹命名
        case_file_name = f'PUSH'
        # case_file_name = f'PUSH_Ke={Ke:.3e}'
        
        # 位移路径
        disp_path = 0.064 * UNIT.m
    
    # 创建工况路径
    case_path = os.path.join(ROOT_PATH, case_file_name)
    os.makedirs(case_path, exist_ok=True)
    # 设置数据库文件夹
    opst.post.set_odb_path(case_path)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"    # 实例化模型
    Model = TwoPierModelTEST(modelPath=case_path)

    # 模型参数
    params = {
        'Ke': Ke,
        'info': False
    }

    # 创建模型
    ModelProps = Model.RCPier(**params)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 响应数据库
    opst.post.set_odb_path(case_path)
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

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 损伤判断
    StrucDS = Model.determine_damage(odb_tag=case_file_name, info=params['info'])
    StrucDS.to_excel(f'{case_path}/{ModelProps.Name}_damage.xlsx', index=False)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 关键部位的界面损伤状态
    Model.plot_pier_sec_state(odb_tag=case_file_name)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 荷载-位移 曲线
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'{ModelProps.Name} Displacement-Load Curve')
    
    plt.plot(test_data['mm'], test_data['N'], linewidth=0.8, label='TEST', zorder=2) # 实验数据
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
    # 导出 曲线
    out_data = pd.DataFrame({
        'Disp(m)': disp,
        'Load(m)': load
        })
    out_data.to_excel(f'{case_path}/{ModelProps.Name}_disp_load.xlsx', index=True)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {ModelProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')

"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
def CASE_MODEL_BRB(
    ROOT_PATH: str, # ./OutData
    
    CORE_RATIO: float = 0.43821,
    GAP: float = 0.02,
    GAP_K: float = 18.e6 * UNIT.pa,
    BUCKLING_K: float = 176.7 * UNIT.gpa,

    Ke: float = 1.,
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
        case_file_name = f'CYCLE_BRB' 
        # case_file_name = f'CYCLE_Ke={Ke:.3e}' 
        
        # 位移路径
        disp_step_1 = np.arange(0.002, 0.012 + 0.002, 0.002) # 第一阶段 控制位移幅值
        disp_step_2 = np.arange(0.016, 0.052 + 0.004, 0.004) # 第二节段 控制唯一幅值
        disp_step = np.repeat(
            np.concatenate((disp_step_1, disp_step_2)), # 合并
            3 # 重复三次
            )
        disp_pairs = np.stack((disp_step, -disp_step), axis=1).flatten() # 控制位移成对，整理
        disp_path = np.concatenate(([0.], disp_pairs, [0.056, -0.056, 0.056, -0.056, 0.])) * UNIT.m # 添加首尾
        
        # 破坏后的位移幅值
        disp_path_break = np.array([
                0.,
                0.056, -0.056,
                0.060, -0.060,
                0.060, -0.060,
                0.060, -0.060,
                0.064, -0.064,
                0.064, -0.064,
                0.064, -0.064,
                0.
            ]) * UNIT.m
        
    else:
        # 工况文件夹命名
        case_file_name = f'PUSH_BRB'
        # case_file_name = f'PUSH_Ke={Ke:.3e}'
        
        # 位移路径
        '''issues? 只有正向时会报错'''
        disp_path = np.array([0., 0.056, -0.056, 0.]) * UNIT.m # 仅正向时，不收敛
        disp_path_break = np.array([0., 0.064, -0.064, 0.]) * UNIT.m # 仅正向时，不收敛
    
    # 创建工况路径
    case_path = os.path.join(ROOT_PATH, case_file_name)
    os.makedirs(case_path, exist_ok=True)
    # 设置数据库文件夹
    opst.post.set_odb_path(case_path)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 实例化模型
    Model = TwoPierModelTEST(modelPath=case_path)

    # 模型参数
    params = {
        'Ke': Ke,
        
        'core_ratio': CORE_RATIO,
        'gap': GAP,
        'gapK': GAP_K,
        'bucklingK': BUCKLING_K,

        'info': False
    }

    # 创建模型
    ModelProps = Model.RCPierBRB(**params)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 响应数据库
    opst.post.set_odb_path(case_path)
    ODB = opst.post.CreateODB(
        odb_tag=case_file_name,
        elastic_frame_sec_points=9,
        node_tags=None,
        frame_tags=None,
        fiber_ele_tags="ALL",
        model_update = True, # 分析过程更新模型，中途删除单元
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
    # 固定分析
    ops.loadConst('-time', 0.0)
    # 删除BRB单元
    ops.remove('ele', ModelProps.KeyEle['BRB']) # 删除单元
    
    # 控制荷载工况
    remove_pattern = 300
    ops.pattern("Plain", remove_pattern, ts)
    ops.load(ctrl_node, 0.0, F, 0.0, 0.0, 0.0, 0.0)  # 节点荷载

    # 执行后续分析
    disp_break, load_break = ATs.STATIC(
        filepath=case_path,
        pattern=remove_pattern,
        ctrl_node=ctrl_node,
        protocol=disp_path_break,
        incr=0.001,
        direction=2,
        RESP_ODB=ODB
        )

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 保存数据库
    ODB.save_response(zlib=True)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 损伤判断
    StrucDS = Model.determine_damage(odb_tag=case_file_name, info=params['info'])
    StrucDS.to_excel(f'{case_path}/{ModelProps.Name}_damage.xlsx', index=False)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 关键部位的界面损伤状态
    Model.plot_pier_sec_state(odb_tag=case_file_name)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # BRB 响应
    fig = Model.resp_BRB(odb_tag=case_file_name)
    fig.savefig(f'{case_path}/BRB_reasp.png', dpi=300, bbox_inches='tight')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 荷载-位移 曲线
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'{ModelProps.Name} Displacement-Load Curve')

    plt.plot(test_data_BRB['mm'], test_data_BRB['N'], linewidth=0.8, label='TEST', zorder=2) # 实验数据
    plt.plot(disp, load, alpha=1, linewidth=0.8, color='#FF9800', label='FEM', zorder=3) # FEM
    plt.plot(disp_break, load_break + load[-1], alpha=1, linewidth=0.8, color='#E91E63', label='FEM-break', zorder=3) # FEM
    
    plt.xlabel('Displacement (m)')
    plt.ylabel('Load (kN)')
    plt.xlim(-np.max(np.abs(disp)) * 1.2, np.max(np.abs(disp) * 1.2))
    plt.ylim(-np.max(np.abs(load)) * 1.2, np.max(np.abs(load)) * 1.2)
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    # plt.show()
    plt.savefig(f'{case_path}/disp_load.png', dpi=300, bbox_inches='tight')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 导出 曲线
    out_data = pd.DataFrame({
        'Disp(m)': np.concatenate([disp, disp_break]),
        'Load(m)': np.concatenate([load, load_break + load[-1]])
        })
    out_data.to_excel(f'{case_path}/{ModelProps.Name}_disp_load.xlsx', index=True)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # fig = opst.vis.plotly.plot_nodal_responses(odb_tag=case_file_name, defo_scale=2.0, slides=True)
    # fig.write_html(f'{case_path}/Model_Deformation.html')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {ModelProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')
    

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
        {'ROOT_PATH': root_file, 'CYCLE_MODE': False,},
        {'ROOT_PATH': root_file, 'CYCLE_MODE': True,},
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