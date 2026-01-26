#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：RUN_Case.py
@Date    ：2025/7/11 21:08
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import os
import time
import numpy as np
from typing import Callable, Any
import opstool as opst
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import rich

import script
from script import UNIT, PVs
from script.base import random_color
from Part_Model_MomentCurvature import SectionModel
from SectionHub import SectionHub

import multiprocessing as mulp
from joblib import Parallel, delayed
from rich.progress import Progress, BarColumn, TimeElapsedColumn


"""
# --------------------------------------------------
# ========== < RUN_Case > ==========
# --------------------------------------------------
"""

def ANALYSIS_CASE(case: dict):

    # 模型空间
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # 截面材料标签汇总
    secTag_props = {
        'section_tag': 100,  # 截面
        'cover_tag': 1,  # 材料-保护层
        'core_tag': 2,  # 材料-核心
        'bar_tag': 3,  # 材料-钢筋
        'bar_max_tag': 4,  # 材料-钢筋最大应变限制
        'info': False
    }
    
    root_path = './OutData'
    
    # 截面
    section_props = case['sec_func'](root_path, **secTag_props)
    
    # 模型
    section_model = SectionModel(
        filepath=root_path,
        sec_props=section_props,
        ctrl_dir=case['dir']
        )
    # 分析
    section_model.run_analysis(
        targets_phi=np.array(0.2), incr_phi=1.e-4,
        ds_info=False, ds_out=False
        )
    # 后处理
    section_model.get_Phi_M(plot_out=True, to_excel=True)
    section_model.equivalent_bilinear(plot_out=True, data_out=True, info=False)
    section_model.plot_strain_state(map_style='coolwarm', step='BREAK', plot_out=True)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {section_props.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')


"""
# -------------------------------------------------- 
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":

    # 截面列表
    CASE_LIST = [
        {'sec_func': SectionHub.Section_Example_01, 'dir': "y"},
        {'sec_func': SectionHub.Section_Example_02, 'dir': "y"},
        {'sec_func': SectionHub.Section_Example_02, 'dir': "z"},
        {'sec_func': SectionHub.Section_Example_03, 'dir': "y"},
        {'sec_func': SectionHub.Section_Example_04, 'dir': "y"},
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
    rich.print(f'[bold {COLOR}] :tada: DONE: All Section Analyze Successfully ! :tada: [/bold {COLOR}]')
    rich.print(f'[bold {COLOR}] # ===== ===== ===== ===== << END >> ===== ===== ===== ===== # [/bold {COLOR}]\n')

    rich.print(f'总用时：{time.time() - START: .3f} s')
