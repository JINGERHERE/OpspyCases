#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Case_HeighPierModel.py
@Date    ：2025/08/10 12:43:29
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from encodings.punycode import T
import os
import time
from turtle import Turtle

import numpy as np
import pandas as pd
import opstool as opst
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import rich
from pathlib import Path
import opstool.vis.plotly as opsplt
import opstool.vis.pyvista as opsvis
from typing import Union, List, Tuple, Dict, Any, Optional

import script
from script import UNIT
from script.base import random_color
from script.pre import WaveReader
from Part_Model_HeighPierModel import HeighPierModelTEST
from Script_ModelAnalyze import AnalysisTools as ATs
from script.post import IDA

import multiprocessing as mulp
from joblib import Parallel, delayed
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from collections import namedtuple


"""
# --------------------------------------------------
# ========== < Case_HeighPierModel > ==========
# --------------------------------------------------
"""

def ANALYSIS_CASE_STATIC(path: Union[str, Path], CASE: str) -> pd.DataFrame:

    """
    工况函数：
        输入：截面， 对应的分析工况
        执行：输入截面 对应方向的 弯矩曲率分析
        返回：-
    """
    
    # rich.print(f'[bold green]# ===== ===== ===== ===== << {CASE:.3e} >> ===== ===== ===== ===== #[/bold green]')
    
    root_path = Path(path)
    
    case_path = os.path.join(root_path, CASE)
    os.makedirs(case_path, exist_ok=True)
    
    # 实例化模型
    Model = HeighPierModelTEST()

    # 创建模型
    ModelProps = Model.HeighPier(case_path, info=False)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 响应数据库
    opst.post.set_odb_path(case_path)
    ODB = opst.post.CreateODB(
        odb_tag=CASE,
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
    ctrl_node = ModelProps.KeyNode['pier_1_top']
    # 单位控制荷载
    F = 1.
    # 控制荷载工况
    static_pattern = 200
    ops.pattern("Plain", static_pattern, ts)
    ops.load(ctrl_node, 0.0, F, 0.0, 0.0, 0.0, 0.0)  # 节点荷载

    # 位移路径
    disp_path = 2.4 * UNIT.m
    # disp_path = np.array([
    #         0.,
    #         # 0.01, -0.01,
    #         # 0.02, -0.02,
    #         # 0.03, -0.03,
    #         # 0.04, -0.04,
    #         # 0.05, -0.05,
    #         # 0.06, -0.06,
    #         # 0.07, -0.07,
    #         # 0.08, -0.08,
    #         0.09, -0.09,
    #         # 0.10, -0.10,
    #         # 0.11, -0.11,
    #         # 0.12, -0.12,
    #         # 0.14, -0.14,
    #         # 0.16, -0.16,
    #         0.18, -0.18,
    #         # 0.20, -0.20,
    #         # 0.22, -0.22,
    #         # 0.24, -0.24,
    #         0.24, -0.24,
    #         0.
    #     ]) * 10 * UNIT.m

    # 静力分析
    disp, load = ATs.STATIC(
        filepath=case_path,
        pattern=static_pattern,
        ctrl_node=ctrl_node,
        protocol=disp_path,
        incr=0.1,
        direction=2,
        RESP_ODB=ODB
        )

    # # 荷载-位移 曲线
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'{ModelProps.Name} Displacement-Load Curve')
    # plt.plot(test_data['mm']*UNIT.mm, test_data['N']*UNIT.n, label='TEST', zorder=2) # 实验数据
    plt.plot(disp, load, alpha=0.8, label='FEM', zorder=3) # FEM
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
    # # 损伤判断
    PierDS = Model.determine_damage(odb_tag=CASE, info=True)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 截面状态
    fig_1 = Model.reasp_fiber_sec(odb_tag=CASE, ele_tag=ModelProps.KeyEle['pier_1_base_1'], integ=5, step=-1)
    fig_2 = Model.reasp_fiber_sec(odb_tag=CASE, ele_tag=ModelProps.KeyEle['pier_2_base_1'], integ=5, step=-1)

    fig_1.savefig(f'{case_path}/col_1_fiber_sec_state.png', dpi=300, bbox_inches='tight')
    fig_2.savefig(f'{case_path}/col_2_fiber_sec_state.png', dpi=300, bbox_inches='tight')
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 墩顶位移数据
    disp = Model.reasp_disp(odb_tag=CASE)
    # 墩底响应数据
    react = Model.reasp_base_force(odb_tag=CASE)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 各损伤阶段对应的 [位移， 水平荷载]
    disp_df = disp.to_pandas().reset_index() # 转换为df，重置索引列
    react_df = react.to_pandas().reset_index()

    pointDS = pd.DataFrame()
    pos = PierDS.index # 损伤的行索 .astype(int)转换整数
    pointDS['disp'] = disp_df.iloc[pos][['disp']].copy() # 损伤对应的位移
    pointDS['reaction'] = react_df.iloc[pos][['reaction']].copy().abs() # 损伤对应的反力
    pointDS['DS'] = PierDS['DS']
    print(f'# pointDS:\n{pointDS}')
    
    pointDS.to_excel(f'{case_path}/analysis_result.xlsx', index=False) # 导出为excel

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {ModelProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # # 要求保存的数据库完整
    # opsvis.set_plot_props(point_size=5, line_width=3)
    # fig = opsvis.plot_nodal_responses(
    #     odb_tag=CASE,
    #     # slides=True,
    #     resp_type="disp",
    #     resp_dof=["UX", "UY", "UZ"],
    #     show_outline=True,
    #     cpos="yz",
    #     defo_scale=5.0
    #     )
    # fig.show()
    return pointDS


def ANALYSIS_CASE_DYNAMIC(path: Union[str, Path], CASE: Path, tar_pga: float) -> dict[str, Any]:
    
    root_path = Path(path)
    # CASE为文件路径，CASE.name为该路径下的文件名
    case_name = f'{Path(CASE.name).stem}_{tar_pga}'
    
    # 创建模型文件夹
    case_path = os.path.join(root_path, case_name)
    os.makedirs(case_path, exist_ok=True)

    # 实例化模型
    Model = HeighPierModelTEST()

    # 创建模型
    ModelProps = Model.HeighPier(case_path, info=False)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 响应数据库
    opst.post.set_odb_path(case_path)
    ODB = opst.post.CreateODB(
        odb_tag=str(case_name),
        elastic_frame_sec_points=9,
        node_tags=[
            ModelProps.KeyNode['pier_1_top'],
            ModelProps.KeyNode['pier_2_top'],
            ModelProps.KeyNode['pier_1_base'],
            ModelProps.KeyNode['pier_2_base'],
            ],
        frame_tags=[
            ModelProps.KeyEle['pier_1_base_1'],
            ModelProps.KeyEle['pier_2_base_1'],
            ModelProps.KeyEle['pier_1_base_2'],
            ModelProps.KeyEle['pier_2_base_2'],
            ],
        fiber_ele_tags=[
            ModelProps.KeyEle['pier_1_base_1'],
            ModelProps.KeyEle['pier_2_base_1'],
            ModelProps.KeyEle['pier_1_base_2'],
            ModelProps.KeyEle['pier_2_base_2'],
            ],
        )

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)
    # 重力荷载工况
    grav_pattern = 100
    ops.pattern("Plain", grav_pattern, ts)
    g = 9.80665 * (UNIT.m / UNIT.sec ** 2)
    opst.pre.create_gravity_load(direction='Z', factor=-g)  # 从整体模型的节点质量获取重力荷载
    # 重力分析
    ATs.GRAVITY(filepath=case_path, RESP_ODB=ODB)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 读取地震波
    wave_reader = WaveReader()
    dt, npts, accel = wave_reader.read_record(in_file=CASE, plot=False, zeros=False)
    
    # 调幅
    tar_ratio = tar_pga / np.max(np.abs(accel))
    tar_accel = tar_ratio * accel
    
    # 保存调幅后数据
    wave_output = Path(f'{case_path}/{case_name}.DAT')
    np.savetxt(wave_output, tar_accel, newline='\n')
    
    # 加速度时间序列
    accel_ts = 2
    ops.timeSeries('Path', accel_ts, '-filePath', str(wave_output), '-dt', dt, '-factor', g)
    
    # 荷载工况编号
    dynamic_pattern = 300
    ATs.DYNAMIC(pattern=dynamic_pattern, dt=dt, npts=npts, wave_ts=accel_ts, RESP_ODB=ODB)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 保存数据库
    ODB.save_response(zlib=True)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 损伤判断
    StrucDS = Model.determine_damage(odb_tag=CASE, info=params['info'])
    StrucDS.to_excel(f'{case_path}/{ModelProps.Name}_damage.xlsx', index=False)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 墩顶位移数据
    disp = Model.reasp_disp(odb_tag=str(case_name))
    # 墩底响应数据
    react = Model.reasp_base_force(odb_tag=str(case_name))
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    times = disp['time']

    # 绘图
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.title(f'{ModelProps.Name} Top Displacement - {str(case_name)}')
    plt.plot(times, np.array(disp), zorder=2)
    plt.xlim(0., times.max())
    plt.ylim(-np.max(np.abs(disp)) * 1.2, np.max(np.abs(disp)) * 1.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    # plt.show()
    plt.savefig(f'{case_path}/top_disp.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {ModelProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')

    result = {
        'wave': str(Path(CASE.name).stem),
        'pga': tar_pga,
        'base_react': np.array(np.max(np.abs(react)))
    }

    result_df = pd.DataFrame([result])
    result_df.to_excel(f'{case_path}/analysis_result.xlsx', index=False) # 导出为excel

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # # 要求保存的数据库完整
    # opsvis.set_plot_props(point_size=5, line_width=3)
    # fig = opsvis.plot_nodal_responses_animation(
    #     odb_tag=str(case_name),
    #     # slides=True,
    #     resp_type="disp",
    #     resp_dof=["UX", "UY", "UZ"],
    #     show_outline=True,
    #     cpos="yz",
    #     )
    # fig.show()
    return result # 打包为字典返回


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""

if __name__ == "__main__":
    
    PATH = './ModelData'
    # PATH = Path('./ModelData')
    
    start_time = time.time()

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # Pushover 分析
    key_point = ANALYSIS_CASE_STATIC(PATH, 'Pushover')
    key_point.to_excel(f'{PATH}/PushoverDS.xlsx', index=False)   # 不写行索引
    
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 实例化地震波读取工具
    wave_reader = WaveReader()
    # 地震波文件夹
    wave_path = Path('./GMfiles/BestArtificial/TimeHistories')
    wave_names = [fp for fp in wave_path.glob('*.AT2')] # 匹配文件名 /不包含路径

    # 时程分析
    # ANALYSIS_CASE_DYNAMIC(path=PATH, CASE=wave_names[0], tar_pga=5.5)
    
    # 一键起爆
    # PGAs = [1.0, 2.0]
    PGAs = [0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.7, 1.8, 1.9, 2.0,
            ]
    all_wave_react_ls = []
    for target_pga in PGAs:
        results = Parallel(n_jobs=-1)(
            delayed(ANALYSIS_CASE_DYNAMIC)(PATH, case, target_pga) for case in wave_names
        )
        # 返回的字典数据转换为DataFrame
        wave_react = pd.DataFrame(list(results)) if results else pd.DataFrame(columns=["wave","pga","base_react"])
        wave_react.to_excel(f'{PATH}/wave_react_{target_pga}g.xlsx', index=False) # 每一个pga保存一次
        # 汇总每一次
        all_wave_react_ls.append(wave_react)
    # 汇总后保存为DataFrame
    all_wave_react = pd.concat(all_wave_react_ls, ignore_index=True)
    all_wave_react.to_excel(f'{PATH}/all_wave_react.xlsx', index=False) # 导出为excel
    print(f'# 数据汇总：\n{all_wave_react}')

    rich.print(f'# 总用时：{time.time()-start_time: 3f} s')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    ida = IDA()
    # 回归分析 [ln(PGA), ln(Sd/Sc)]
    # pgas = np.array(all_wave_react['pga'])
    pgas = pd.to_numeric(all_wave_react['pga'], errors='coerce').to_numpy(dtype=np.float64)

    Sd_series = all_wave_react['base_react']
    Sc_series = key_point.loc[key_point['DS'] == 'DS2', 'reaction']

    Sd = pd.to_numeric(Sd_series, errors='coerce').to_numpy(dtype=np.float64)
    Sc = pd.to_numeric(Sc_series, errors='coerce').to_numpy(dtype=np.float64)

    ln_IM = np.log(pgas)
    ln_SdSc = np.log(np.array(Sd) / np.array(Sc))
    
    # 回归曲线
    p, Sr, R2, y_predi = ida.quadratic_fit(np.array(ln_IM), np.array(ln_SdSc))
    # 回归曲线绘图
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.scatter(ln_IM, ln_SdSc, s=5, color='black',zorder=2)
    plt.plot(ln_IM, y_predi, 'r', zorder=3)
    plt.xlabel('ln(IM)')
    plt.ylabel('ln(Sd/Sc)')
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    # plt.show()
    plt.savefig(f'{PATH}/quadratic_fit.png', dpi=300, bbox_inches='tight')
    
    # 易损性曲线
    # 概率函数
    Pf = ida.probability()
    # 超越概率绘图
    plt.close('all')
    plt.figure(figsize=(6, 4))
    plt.plot(pgas, Pf, '-o')
    plt.xlabel('PGA(g)')
    plt.ylabel('Probability')
    plt.grid(linestyle='--', linewidth=0.5, zorder=1)
    # plt.show()
    plt.savefig(f'{PATH}/probability.png', dpi=300, bbox_inches='tight')
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"

    # opsplt.set_plot_props(point_size=5, line_width=3)
    # fig = opsplt.plot_model(show_nodal_loads=True)
    # fig.write_html('model.html')
