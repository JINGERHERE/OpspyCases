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

import numpy as np
import pandas as pd
import opstool as opst
import openseespy.opensees as ops
import matplotlib.pyplot as plt
import rich
from pathlib import Path

import ops_utilities as opsu
from ModelHub import RockPierModel
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
test_data_path = "./.RAW_DATA"
data_file = "/SCB.xlsx"
# data_file = '/SCB_EDB.xlsx'
# 导入
test_data = pd.read_excel(
    f"{test_data_path + data_file}",
    # header=0,
)
# 清洗数据 转换为数值
test_data["m"] = pd.to_numeric(test_data["m"], errors="coerce")
test_data["kN"] = pd.to_numeric(test_data["kN"], errors="coerce")

"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
class CaseHub:

    def __init__(
        self, manager: opsu.pre.ModelManager, data_path: Path, fit: float = 0
    ) -> None:
        """
        模型工况实例

        Args:
            manager (opsu.pre.ModelManager): 模型管理器
            data_path (Path): 数据路径
            fit (float, optional): 模型拟合参数. Defaults to 0.

        Returns:
            None
        """

        # 创建数据路径
        self.case_name = f"model_fit_{fit:.2e}" if fit else "model"
        self.data_path = data_path / self.case_name
        self.data_path.mkdir(parents=True, exist_ok=True)

        # 数据库
        self.MM = manager
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 实例化模型
        self.model = RockPierModel(self.MM, self.data_path)
        self.model.model(info=False)
        # if BRB:
        #     self.model.BRB(info=False)

        # 输出模型
        self.MM.to_excel(self.data_path / "ModelManager.xlsx")
        # 输出模型
        ops.printModel("-JSON", "-file", str(self.data_path / "thisModel.json"))
        # 可视化模型
        fig = opst.vis.plotly.plot_model(show_local_axes=True)
        fig.write_html(self.data_path / "thisModel.html")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 实例化模型
        self.model = RockPierModel(self.MM, self.data_path)
        self.model.model(fit=fit, info=False)

        # 时间序列
        self.ts = OPSE.timeSeries("Linear")

        # 数据库
        opst.post.set_odb_path(str(self.data_path))  # 数据库输出路径
        opst.post.set_odb_format(odb_format="zarr")  # 数据库输出格式
        # 创建数据库
        self.ODB = opst.post.CreateODB(
            odb_tag=self.case_name,
            elastic_frame_sec_points=9,
            fiber_ele_tags="ALL",
            zlib=True
        )

    def gravity(self, save_odb: bool = True) -> None:
        """
        重力分析方法

        Args:
            save_odb (bool, optional): 是否保存数据库. Defaults to True.

        Returns:
            None
        """

        # 时间荷载重置
        ops.loadConst("-time", 0.0)

        # 参数
        g = 9.80665 * (UNIT.m / UNIT.sec**2)

        # 重力荷载模式
        OPSE.pattern("Plain", self.ts)
        opst.pre.create_gravity_load(direction="Z", factor=-g)

        # 实例化
        case_grav = ANL.GravityAnalysis(self.ODB)
        # 执行分析
        case_grav.analyze(10)

        # 保存数据库
        if save_odb:
            self.ODB.save_response()

    def _static(self, targets, incr):
        """
        < 内部方法 > 静力分析方法

        Args:
            targets (float): 位移目标路径
            incr (float): 每一步增量

        Returns:
            Tuple[np.ndarray, np.ndarray]: 位移路径, 力路径
                - disp: 位移路径
                - force: 力路径 (控制力 * 荷载乘子)
        """

        # 时间荷载重置
        ops.loadConst("-time", 0.0)

        # 参数
        ctrl = self.MM.get_tag("node", label="disp_ctrl")[0]
        force = 1.0

        # 静力荷载模式
        pattern = OPSE.pattern("Plain", self.ts)
        ops.load(ctrl, *(0.0, force, 0.0, 0.0, 0.0, 0.0))  # 节点荷载

        # 实例化
        case_static = ANL.StaticAnalysis(pattern, self.ODB)
        # 执行分析
        disp, froce_lbd = case_static.analyze(
            ctrl_node=ctrl, dof=2, targets=targets, max_step=incr
        )

        return disp, force * froce_lbd

    def push(self):
        """
        Pushover 分析方法

        Returns:
            Tuple[np.ndarray, np.ndarray]: 位移路径, 力路径
                - disp: 位移路径
                - force: 力路径
        """

        # 重力分析
        self.gravity(save_odb=False)

        # 位移路径
        incr = 0.001 * UNIT.m
        disp_path = 0.12 * UNIT.m

        # 静力分析
        disp, force = self._static(targets=disp_path, incr=incr)

        # 保存数据库
        self.ODB.save_response()

        return disp, force

    def cycle(self):
        """
        Cycle 分析方法

        Returns:
            Tuple[np.ndarray, np.ndarray]: 位移路径, 力路径
                - disp: 位移路径
                - force: 力路径
        """

        # 重力分析
        self.gravity(save_odb=False)

        # 位移路径
        incr = 0.002 * UNIT.m
        # 全程
        disp_1 = np.arange(0.003, 0.015 + 0.003, 0.003)  # 第一阶段 控制位移幅值
        disp_2 = np.arange(0.016, 0.12 + 0.0075, 0.0075)  # 第二节段 控制唯一幅值
        disp_step = np.repeat(np.concatenate((disp_1, disp_2)), 3)  # 合并后 重复三次
        disp_pairs = np.stack((disp_step, -disp_step), axis=1).flatten()  # 正负成对
        disp_path = np.concatenate(([0.0], disp_pairs, [0.0])) * UNIT.m  # 添加首尾
        # 最大圈
        # disp_path = np.array([0.0, 0.12, -0.12, 0.0]) * UNIT.m

        # 静力分析
        disp, force = self._static(targets=disp_path, incr=incr)

        # 保存数据库
        self.ODB.save_response()

        return disp, force


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":

    def case(fit: float):
        # 根目录
        data_path = Path().cwd() / "OutData"
        data_path.mkdir(parents=True, exist_ok=True)

        ch = CaseHub(MM, data_path, fit=fit)
        ch.gravity()
        disp, force = ch.push()
        # disp, force = ch.cycle()

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"

    Parallel(n_jobs=-1)(delayed(case)(i) for i in [0.0, 1.4e3, 1.5e3, 1.6e3])
