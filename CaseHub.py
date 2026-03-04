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
from typing import Optional, Union, Tuple, List, Literal
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


def gen_path(
    peaks: Union[List, Tuple, np.ndarray],
    repeat: int = 1,
    mode: Literal["full", "half", "push"] = "full",
    zero_start: bool = True,
    zero_end: bool = True,
) -> np.ndarray:
    """
    生成路径

    Args:
        peaks (Union[List, Tuple]): 峰值列表
        repeat (int, optional): 重复次数. Defaults to 1.
        mode (Literal["full", "half", "push"], optional): 模式. Defaults to "full".
        zero_start (bool, optional): 是否在开头添加零. Defaults to True.
        zero_end (bool, optional): 是否在结尾添加零. Defaults to True.

    Returns:
        np.ndarray: 路径数组
    """

    path = []
    for p in peaks:
        # 定义模式
        patterns = {"full": [p, -p], "half": [p, 0.0], "push": [p]}
        # 获取模式，若输入错误则抛出异常
        unit = patterns.get(mode)
        if unit is None:
            raise ValueError(
                f"Invalid mode: {mode}. Choose from 'full', 'half', 'push'."
            )

        # 将单元重复 repeat 次并存入路径
        path.extend(unit * repeat)

    res = np.array(path, dtype=float)

    # 零点处理
    if zero_start:
        res = np.insert(res, 0, 0.0)
    if zero_end and (len(res) == 0 or res[-1] != 0.0):
        res = np.append(res, 0.0)

    return res


class CaseHub:

    def __init__(
        self,
        manager: opsu.pre.ModelManager,
        data_path: Path,
        BRB: bool = False,
        fit: float = 0,
    ) -> None:
        """
        模型工况实例

        Args:
            manager (opsu.pre.ModelManager): 模型管理器
            data_path (Path): 数据路径
            BRB (bool, optional): 是否包含 BRB. Defaults to False.
            fit (float, optional): 模型拟合参数. Defaults to 0.

        Returns:
            None
        """

        # 创建数据路径
        self.brb = BRB
        if BRB:
            self.case_name = f"model_BRB_fit_{fit:.2e}" if fit else "model_BRB"
        else:
            self.case_name = f"model_fit_{fit:.2e}" if fit else "model"
        self.data_path = data_path / self.case_name
        self.data_path.mkdir(parents=True, exist_ok=True)

        # 数据库
        self.MM = manager
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 实例化模型
        self.model = RockPierModel(self.MM, self.data_path)
        self.model.model(info=False, fit=fit)
        if BRB:
            self.model.BRB(info=False)

        # 输出模型
        self.MM.to_excel(self.data_path / "ModelManager.xlsx")
        # 输出模型
        ops.printModel("-JSON", "-file", str(self.data_path / "thisModel.json"))
        # 可视化模型
        fig = opst.vis.plotly.plot_model(show_local_axes=False)
        fig.write_html(self.data_path / "thisModel.html")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 时间序列
        self.ts = OPSE.timeSeries("Linear")

        # 数据库
        opst.post.set_odb_path(str(self.data_path))  # 数据库输出路径
        opst.post.set_odb_format(odb_format="zarr")  # 数据库输出格式
        # 创建数据库
        self.ODB = opst.post.CreateODB(
            odb_tag=self.case_name,
            model_update=True,
            elastic_frame_sec_points=9,
            fiber_ele_tags="ALL",
            zlib=True,
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
            Tuple(np.ndarray, np.ndarray): 位移路径, 力路径
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

        # 位移路径步长
        incr = 0.002 * UNIT.m

        # 循环重复次数
        repeat = 3

        if self.brb:
            # 第一段
            disp_path_1 = (
                gen_path(
                    peaks=(
                        0.0030,
                        0.0060,
                        0.0090,
                        0.0120,
                        0.0150,
                        0.0225,
                        0.0300,
                        0.0375,
                    ),
                    repeat=repeat,
                )
                * UNIT.m
            )
            disp_1, force_1 = self._static(targets=disp_path_1, incr=incr)

            # 第二段
            # ops.remove("ele", 53)
            ops.remove("ele", self.MM.get_tag("element", label="brb_top")[0])  # 53
            # ops.remove("ele", self.MM.get_tag("element", label="brb_base")[0])  # 56
            disp_path_2 = (
                gen_path(
                    peaks=(
                        0.0450,
                        0.0525,
                        0.0600,
                        0.0675,
                        0.0750,
                        0.0825,
                        0.0900,
                        # 0.0975,
                        # 0.1050,
                        # 0.1125,
                        # 0.1200,
                    ),
                    repeat=repeat,
                )
                * UNIT.m
            )
            disp_2, force_2 = self._static(targets=disp_path_2, incr=incr)

            # 合并数据
            # disp = np.concatenate((disp_1, disp_2))
            # force = np.concatenate((force_1, force_2 + force_1[-1]))

            # 第三段
            # ops.remove("ele", 56)
            # ops.remove("ele", self.MM.get_tag("element", label="brb_top")[0])  # 53
            # ops.remove("ele", self.MM.get_tag("element", label="brb_base")[0])  # 56
            disp_path_3 = (
                gen_path(
                    peaks=(
                        0.0975,
                        0.1050,
                        0.1125,
                        0.1200,
                    ),
                    repeat=repeat,
                )
                * UNIT.m
            )
            disp_3, force_3 = self._static(targets=disp_path_3, incr=incr)

            # 合并数据
            disp = np.concatenate((disp_1, disp_2, disp_3))
            force = np.concatenate((force_1, force_2 + force_1[-1], force_3))

        else:
            # 全程
            disp_path = (
                gen_path(
                    peaks=np.concatenate(
                        (
                            np.arange(0.003, 0.015 + 0.003, 0.003),
                            np.arange(0.015, 0.12 + 0.0075, 0.0075),
                        )
                    ),
                    repeat=repeat,
                )
                * UNIT.m
            )

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
