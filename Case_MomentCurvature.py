#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Case_MomentCurvature.py
@Date    ：2025/7/11 21:08
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import os
import time
from typing import Callable, Any
import opstool as opst
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import rich

import script
from script import UNIT, PVs
from script.base import random_color
from Part_Model_MomentCurvature import SectionModel
from Part_MatSec_MomentCurvature import MPhiSection

import multiprocessing as mulp
from joblib import Parallel, delayed
from rich.progress import Progress, BarColumn, TimeElapsedColumn


"""
# --------------------------------------------------
# ========== < Case_MomentCurvature > ==========
# --------------------------------------------------
"""

def ANALYSIS_CASE(sec: Callable[..., PVs.SEC_PROPS], direction: str):

    """
    工况函数：
        输入：截面， 对应的分析工况 // 需要输入构造截面的函数，且该函数的输出类型为 PVs.SecProps
        执行：输入截面 对应方向的 弯矩曲率分析
        返回：-
    """

    # 实例化模型
    section_model = SectionModel('./OutData')

    # 截面材料标签汇总
    secTag_props = {
        'section_tag': 100,  # 截面
        'cover_tag': 1,  # 材料-保护层
        'core_tag': 2,  # 材料-核心
        'bar_tag': 3,  # 材料-钢筋
        'bar_max_tag': 4,  # 材料-钢筋最大应变限制
        'info': False
    }
    
    # 创建截面
    SecProps = section_model.create_section(func=sec, axis=direction, **secTag_props)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 弯矩曲率分析
    MC = opst.anlys.MomentCurvature(sec_tag=SecProps.SectionTag, axial_force=-SecProps.P)
    MC.analyze(
        axis=direction,
        max_phi=0.2 / UNIT.m, incr_phi=1e-4,
        limit_peak_ratio=0.8,
        smart_analyze=True, debug=False
        )

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 显示纤维截面 MomentCurvature默认单元号为1
    opst.pre.section.vis_fiber_sec_real(
        ele_tag=1, show_matTag=False,
        highlight_matTag=SecProps.SteelTag, highlight_color="r",
        )
    # plt.savefig(f'{model.subdir}/{SecProps['Name']}_real_fiber.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{section_model.secPath}/{SecProps.Name}_real_fiber.png', dpi=300, bbox_inches='tight')

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 后处理：截面损伤
    section_model.determine_damage(MC, bilinear=True, info=True)  # 损伤判断，开启等效双折线计算
    # 后处理：绘图
    section_model.section_rainbow(step='DS5', style='all')  # 绘制 对应步骤 纤维截面云图：应力，应变，材料

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印当前截面完成信息
    color = random_color()
    rich.print(f'[bold {color}] :tada: DONE: {SecProps.Name} Analyze Successfully ! :tada: [/bold {color}]')
    rich.print(f'[bold {color}] Prepare the next >>>>> [/bold {color}]\n')


"""
# -------------------------------------------------- 
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":

    # 截面列表
    CASE_LIST = [
        (MPhiSection.Section_Example_01, "y"),
        (MPhiSection.Section_Example_02, "y"),
        (MPhiSection.Section_Example_02, "z"),
        (MPhiSection.Section_Example_03, "y"),
        (MPhiSection.Section_Example_04, "y"),
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
            delayed(ANALYSIS_CASE)(sec_case, dir_case) for sec_case, dir_case in CASE_LIST
        )

    else:
        '''正常计算：for循环'''
        for sec_case, dir_case in CASE_LIST:
            ANALYSIS_CASE(sec_case, dir_case)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 打印完成信息
    COLOR = random_color()
    rich.print(f'[bold {COLOR}] :tada: DONE: All Section Analyze Successfully ! :tada: [/bold {COLOR}]')
    rich.print(f'[bold {COLOR}] # ===== ===== ===== ===== << END >> ===== ===== ===== ===== # [/bold {COLOR}]\n')

    rich.print(f'总用时：{time.time() - START: .3f} s')
