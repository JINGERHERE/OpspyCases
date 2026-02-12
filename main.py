#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：main.py
@Date    ：2026/02/11 13:33:34
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

import ops_utilities as opsu
from ModelUtilities import UNIT, MM, PlotyHub
from CaseHub import CaseHub
from ModelUtilities import PostProcess
import AnalysisLibraries as ANL

import multiprocessing as mulp
from joblib import Parallel, delayed

"""
# --------------------------------------------------
# ========== < main > ==========
# --------------------------------------------------
"""

# 关闭图形显示
matplotlib.use("Agg")

# 终端根目录
root_path = Path().cwd()

# 试验原始数据
test_file = "SCB.xlsx"
# test_file = 'SCB_EDB.xlsx'
# 导入
test_data = pd.read_excel(root_path / ".RAW_DATA" / test_file)
# 清洗数据 转换为数值
disp_test = pd.to_numeric(test_data["m"], errors="coerce") * UNIT.m
force_test = pd.to_numeric(test_data["kN"], errors="coerce") * UNIT.kn


class AnalysisCase:

    # 数据路径
    data_path = root_path / "OutData"
    data_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def static(cls, cycle: bool = False, fit: float = 0):

        # 模型实例
        if cycle:
            CH = CaseHub(MM, cls.data_path / "cycle", fit)
            disp, force = CH.cycle()  # 分析
        else:
            CH = CaseHub(MM, cls.data_path / "push", fit)
            disp, force = CH.push()  # 分析

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 后处理实例
        PP = PostProcess(MM, CH.case_name, CH.data_path, print_info=False)
        # 耗能钢筋
        PP.plot_ED_resp(pier=1)
        PP.plot_ED_resp(pier=2)
        # 预应力
        PP.plot_PT_resp()
        # 柱底接触面
        PP.plot_sec_resp(pier=1, SEC=CH.model.SEC_cont, step=PP.surf_sep_step - 1)
        PP.plot_sec_resp(pier=2, SEC=CH.model.SEC_cont, step=PP.surf_sep_step - 1)
        PP.plot_sec_resp_ani(
            pier=1, SEC=CH.model.SEC_cont, max_steps=len(disp) - 1, speed=5
        ) # 减去一步初始状态

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 绘制 Pushover 图
        plt.close("all")
        fig = plt.figure(dpi=100)
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, facecolor="none")

        # 数据 / 关键点数据要减去 10步重力分析 和 1步静力分析初始状态
        ax.plot(disp, force, label="FEM", zorder=11)
        ax.scatter(
            disp[PP.surf_sep_step - 11],
            force[PP.surf_sep_step - 11],
            c="blue",
            label="Sepa.",
            zorder=12,
        )
        ax.scatter(
            disp[PP.ED_yield_step - 11],
            force[PP.ED_yield_step - 11],
            c="red",
            label="Yield",
            zorder=12,
        )
        ax.plot(disp_test, force_test, label="TEST", zorder=10)
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Force (kN)")
        ax.autoscale()  # 自动调整坐标轴范围
        # 图例
        leg = ax.legend(
            loc="lower right",
            bbox_to_anchor=(0.97, 0.05),
            labelcolor=(0, 0, 0, 1),
            # frameon=True, fancybox=True, shadow=False,
        )

        # 调整尺寸
        adj_fig = PlotyHub.adjust_single(fig, ax, leg)
        # 保存图片
        adj_fig.savefig(CH.data_path / f"figure_disp_force.png", dpi=320)


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""


if __name__ == "__main__":

    AnalysisCase.static(cycle=False, fit=1.5e3)
    # AnalysisCase.static(cycle=True, fit=1.5e3)
