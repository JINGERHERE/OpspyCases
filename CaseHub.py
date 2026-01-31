#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：ModelCaseHub.py
@Date    ：2025/7/11 21:08
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Literal
from collections import namedtuple

from ModelHub import SectionModel
from ModelUtilities import MM, PostProcess

from joblib import Parallel, delayed

import matplotlib.pyplot as plt


"""
# --------------------------------------------------
# ========== < ModelCaseHub > ==========
# --------------------------------------------------
"""


def ANALYSIS_CASE(name: str, deg: float, dof: str, file_path: Path) -> None:
    
    """
    分析截面模型并进行后处理。
    
    Args:
        name (str): 截面名称。
        deg (float): 截面创建旋转角度。
        dof (str): 分析方向（'y' 或 'z'）。
        file_path (Path): 数据存储路径。
    
    Returns:
        None: 不返回任何值。
    """
    
    # 创建数据路径
    data_path = file_path / f'{name}_{deg:.1f}_{dof}'
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 曲率控制
    tars=0.5
    incr_phi=1.e-4
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 创建截面模型
    sec_model = SectionModel(manager=MM, sec_name=name, data_path=data_path)
    sec_model.model(deg=deg, info=False)
    # 执行分析
    phi, moment = sec_model.analysis(targets=np.array(tars), max_step=incr_phi, dof=dof)
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 后处理
    post = PostProcess(manager=MM, name=name, data_path=data_path, print_info=False)
    
    # 筛选数据
    sel_phi = phi[: post.ds_stages[-1] + 1]
    sel_moment = moment[: post.ds_stages[-1] + 1]
    yield_step = min(post.ds_stages) # 屈服点索引
    end_step = max(post.ds_stages)
    
    # 保存曲线数据
    df = pd.DataFrame({'Curvature': sel_phi, 'Moment': sel_moment})
    df.to_excel(data_path / 'Moment_Curvature.xlsx', index=False)
    
    # 等效双线性
    post.plot_equivalent_bilinear(
        line_x=sel_phi, line_y=sel_moment,
        point_idx=yield_step, info=False
        )
    # 纤维响应
    post.plot_fiber_resp()
    # 截面响应
    post.plot_sec_resp(SEC=sec_model.SEC, step=yield_step)
    # 截面响应动图 /gif
    post.plot_sec_resp_ani(SEC=sec_model.SEC, max_steps=end_step, speed=5)


"""
# -------------------------------------------------- 
# ========== < TEST > ==========
# --------------------------------------------------
"""


if __name__ == "__main__":

    # 数据路径
    root_path = Path('./OutData')
    root_path.mkdir(parents=True, exist_ok=True)

    # 所有截面名称及分析方向
    CASE = namedtuple('CASE', ['name', 'deg', 'dof'])
    section_cases = [
        # CASE(name='sec_I', deg=0.0, dof='y'),
        # CASE(name='sec_rect', deg=0.0, dof='y'),
        # CASE(name='sec_polygonal', deg=0.0, dof='y'),
        # CASE(name='sec_polygonal', deg=0.0, dof='z'),
        CASE(name='sec_circle', deg=0.0, dof='y'),
        # CASE(name='sec_circle', deg=0.0, dof='z'),
        ]
    
    # 串行计算
    for case in section_cases:
        ANALYSIS_CASE(case.name, case.deg, case.dof, root_path)

    # # 并行计算
    # Parallel(n_jobs=-1)(
    #     delayed(ANALYSIS_CASE)(case.name, case.deg, case.dof, root_path) for case in section_cases
    #     )
