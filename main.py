#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：main.py
@Date    ：2026/02/13 12:01:42
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

class AnalysisCase:

    # 关闭图形显示
    matplotlib.use("Agg")

    # 终端根目录
    root_path = Path().cwd()

    # 数据路径
    data_path = root_path / "OutData"
    data_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def static(cls, BRB: bool = False, cycle: bool = False, fit: float = 0):

        # 模型实例
        if cycle:
            CH = CaseHub(MM, cls.data_path / "cycle", BRB=BRB, fit=fit)
            disp, force = CH.cycle()  # 分析

        else:
            CH = CaseHub(MM, cls.data_path / "push", BRB=BRB, fit=fit)
            disp, force = CH.push()  # 分析

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 试验原始数据
        test_file = "20230826WJBRB.xlsx" if BRB else "20230821WJRC.xlsx"

        # 导入
        test_data = pd.read_excel(cls.root_path / ".RAW_DATA" / test_file)
        # 清洗数据 转换为数值
        disp_test = pd.to_numeric(test_data["mm"], errors="coerce") * UNIT.mm
        force_test = pd.to_numeric(test_data["N"], errors="coerce") * UNIT.n

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 后处理实例
        PP = PostProcess(MM, CH.case_name, CH.data_path, print_info=False)

        # 截面状态
        PP.plot_sec_resp(pier=1, SEC=CH.model.SEC_pier, step=PP.limit_step)
        PP.plot_sec_resp(pier=2, SEC=CH.model.SEC_pier, step=PP.limit_step)

        # BRB
        # PP.plot_BRB_resp()

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 绘制 Pushover 图
        plt.close("all")
        fig = plt.figure(dpi=100)
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, facecolor="none")

        # 数据 / 关键点数据要减去 10步重力分析 和 1步静力分析初始状态
        ax.plot(disp, force, label="FEM", zorder=11)
        ax.scatter(
            disp[PP.yield_step - 11],
            force[PP.yield_step - 11],
            c="blue",
            label="Yield",
            zorder=12,
        )
        ax.scatter(
            disp[PP.limit_step - 11],
            force[PP.limit_step - 11],
            c="red",
            label="Limit",
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

    # AnalysisCase.static(cycle=False, BRB=False)
    # AnalysisCase.static(cycle=False, BRB=True)
    # AnalysisCase.static(cycle=True, BRB=False)
    AnalysisCase.static(cycle=True, BRB=True)

    # Parallel(n_jobs=-1)(
    #     delayed(AnalysisCase.static)(fit=i, cycle=True, BRB=True)
    #     # for i in [2.1e3, 3.2e3, 4.2e3, 5.2e3, 6.2e3, 7.2e3, 2.1e4]
    # )
