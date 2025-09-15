#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Part_Run_BRB.py
@Date    ：2025/09/10 20:05:24
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import os
import time

import numpy as np
import pandas as pd
import xarray as xr
import rich

import multiprocessing as mulp
from joblib import Parallel, delayed

from script import UNIT
from script.base import random_color
import Part_Analysis_Cases as PACs

"""
# --------------------------------------------------
# ========== < Part_Run_BRB > ==========
# --------------------------------------------------
"""

# 根目录
root_file = f'./OutData'

# Pushover 获取性能参数
yield_disp, yield_load = PACs.CASE_MODEL(
    ROOT_PATH = root_file,
    Ke = 1.,
    CYCLE_MODE = False,
    )

# 工况列表
CASE_LIST = [

    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.06,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.07,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.08,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.09,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.10,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.15,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.20,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.25,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.30,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.40,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.50,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.60,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.70,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.80,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.90,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.93,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    
    ]

'''长度比 0.35 误差最小 /基于面积300'''

# 工况列表
CASE_LIST = [
    
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 50 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP': yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 150 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 200 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 250 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 300 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 350 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 400 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 450 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 500 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 550 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 600 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 650 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 700 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 750 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 800 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 850 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 900 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 950 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    # {
    #     'ROOT_PATH': root_file,
    #     'DISP':yield_disp, 'FORCE': yield_load,
    #     'CORE_RATIO': 0.35,'CORE_AREA': 1000 * (UNIT.mm**2),
    #     'Ke':  1., 'CYCLE_MODE': True,},
    ]

'''面积 300 合适 /基于长度比0.35'''

# 工况列表
CASE_LIST = [
    
    {
        'ROOT_PATH': root_file,
        'DISP': yield_disp, 'FORCE': yield_load,
        'CORE_RATIO': 0.35,'CORE_AREA': 300 * (UNIT.mm**2),
        'Ke':  1., 'CYCLE_MODE': True,},

    ]

# 是否启用并行计算
if len(CASE_LIST) >= 2:
    PARALLEL: bool = True
else:
    PARALLEL: bool = False

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":

    START = time.time() # 计时
    
    if PARALLEL:
        '''并行计算：joblib模块'''
        # CPU核心数
        n_cpu = mulp.cpu_count()
        rich.print(f'# 计算机核心数：{n_cpu}')

        # 并行计算
        Parallel(n_jobs=-1)(
            delayed(PACs.CASE_MODEL_BRB)(**case) for case in CASE_LIST
        )

    else:
        '''正常计算：for循环'''
        for case in CASE_LIST:
            PACs.CASE_MODEL_BRB(**case)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印完成信息
    COLOR = random_color()
    rich.print(f'[bold {COLOR}] :tada: DONE: All Model Analyze Successfully ! :tada: [/bold {COLOR}]')
    rich.print(f'[bold {COLOR}] # ===== ===== ===== ===== << END >> ===== ===== ===== ===== # [/bold {COLOR}]\n')

    rich.print(f'总用时：{time.time() - START} s')
