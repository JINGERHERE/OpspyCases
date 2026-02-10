#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：CaseHub.py
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

import ops_utilities as opsu
from ops_utilities.pre import AutoTransf as ATf
from SectionHub import SectionHub
from ModelUtilities import UNIT, MM, OPSE
import AnalysisLibraries as ANL

import multiprocessing as mulp
from joblib import Parallel, delayed


"""
# --------------------------------------------------
# ========== < CaseHub > ==========
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
class ANALYSIS_CASE:
    @classmethod
    def push(cls):
        ...



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
