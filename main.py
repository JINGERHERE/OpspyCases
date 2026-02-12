#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：main.py
@Date    ：2026/01/29 19:34:13
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from pathlib import Path
from typing import Literal
from collections import namedtuple


"""
# --------------------------------------------------
# ========== < main > ==========
# --------------------------------------------------
"""


from CaseHub import ANALYSIS_CASE
import multiprocessing as mulp
from joblib import Parallel, delayed


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""

if __name__ == "__main__":

    # 数据路径
    root_path = Path("./OutData")
    root_path.mkdir(parents=True, exist_ok=True)

    # 所有截面名称及分析方向
    CASE = namedtuple("CASE", ["name", "deg", "dof"])
    section_cases = [
        CASE(name="sec_I", deg=0.0, dof="y"),
        CASE(name="sec_rect", deg=0.0, dof="y"),
        CASE(name="sec_rect", deg=30.0, dof="y"),
        CASE(name="sec_polygonal", deg=0.0, dof="y"),
        CASE(name="sec_polygonal", deg=0.0, dof="z"),
        CASE(name="sec_circle", deg=0.0, dof="y"),
        CASE(name="sec_circle", deg=0.0, dof="z"),
    ]

    # # 串行计算
    # for case in section_cases:
    #     ANALYSIS_CASE(case.name, case.deg, case.dof, root_path)
    #     break

    # 并行计算
    print(f"# 计算机核心数：{mulp.cpu_count()}")
    Parallel(n_jobs=-1)(
        delayed(ANALYSIS_CASE)(case.name, case.deg, case.dof, root_path)
        for case in section_cases
    )
