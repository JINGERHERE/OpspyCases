#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：ModelHub.py
@Date    ：2025/8/1 19:24
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""


from collections import namedtuple
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import xarray as xr
from typing import Literal, TypeAlias, Union, Callable, Any, Optional, TypedDict

import openseespy.opensees as ops
import opstool as opst
import opstool.vis.plotly as opsplt
import opstool.vis.pyvista as opsvis

import ops_utilities as opsu
from ops_utilities.pre import AutoTransf as ATf
from SectionHub import SectionHub
from ModelUtilities import UNIT, MM, OPSE
import AnalysisLibraries as ANL

import warnings

warnings.showwarning = opsu.rich_showwarning

"""
# --------------------------------------------------
# ========== < ModelHub > ==========
# --------------------------------------------------
"""


class TwoPierModel:

    def __init__(
        self,
        manager: opsu.pre.ModelManager,
        data_path: Union[Path, str, Literal[""]] = "",
    ) -> None:
        """
        双柱式桥墩模型实例

        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            data_path (Union[Path, str, Literal['']], optional): 保存路径。 `默认值：` '当前路径'.

        Returns:
            None: 不返回任何值。

        Raises:
            ValueError: 截面序号超出范围。

        """

        # 管理器
        self.MM = manager.wipe()  # 清空管理器

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面的返回数据对象
        self.SEC_beam: opst.pre.section.FiberSecMesh  # 盖梁
        self.SEC_pier: opst.pre.section.FiberSecMesh  # 墩柱
        self.SEC_cont: opst.pre.section.FiberSecMesh  # 接触面

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 保存路径
        self.data_path = Path("./")
        if data_path:
            # 数据路径
            self.data_path = Path(data_path)

        # 创建数据路径
        self.data_path.mkdir(parents=True, exist_ok=True)

    def model(self, Kfit: float = 0.0, info: bool = True) -> None:
        """
        创建 < 双柱式桥墩 > 模型

        Args:
            Kfit (float, optional): 模型收敛刚度拟合。默认值为 0.。
            info (bool, optional): 是否显示模型信息。默认值为 True。

        Returns:
            None: 不返回任何值。
        """

        # 桥墩尺寸控制
        L = 2.8 * UNIT.m  # 盖梁长
        PierH = 1.6 * UNIT.m  # 墩柱高
        PierW = 2 * UNIT.m  # 墩柱中心间距

        # 混凝土性质
        Ec = opsu.pre.ConcHub.get_Ec("C25") * UNIT.mpa
        Gc = opsu.pre.ConcHub.get_G("C25") * UNIT.mpa
        rho = 2600 * (UNIT.kg / (UNIT.m**3))  # 密度：kg/m3

        # 预应力
        PT_fy, PT_Es, PT_area = (
            1860 * UNIT.mpa,
            206 * UNIT.gpa,
            2 * (np.pi * (15.2 * UNIT.mm / 2) ** 2),
            # 2 * (140 * UNIT.mm ** 2)
        )
        PT_f = (310.0 * UNIT.kn) / PT_area  # 张拉控制力

        # 模型收敛刚度拟合
        Kfit = 8.0e-5
        Ubig = 1.0e6
        Usmall = 1.0e-6

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        OPSE.wipe()
        OPSE.model("basic", "-ndm", 3, "-ndf", 6)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 纤维截面
        self.SEC_beam = SectionHub.bent_cap(
            manager=self.MM, info=info, save_sec=self.data_path
        )
        self.SEC_pier = SectionHub.pier_col(
            manager=self.MM, info=info, save_sec=self.data_path
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面添加到模型管理器
        OPSE.add_manager(self.MM)

        # 盖梁质心
        lw_bc = (
            self.MM.get_param(category="section", label="bent_cap", key="Width") / 2.0
        )
        lh_bc = (
            self.MM.get_param(category="section", label="bent_cap", key="Height") / 2.0
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面编号
        sec_tag_beam = self.MM.get_tag("section", label="bent_cap")[0]
        sec_tag_pier = self.MM.get_tag("section", label="pier_col")[0]

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面单元积分
        integ = {
            "bent_cap": OPSE.beamIntegration("Legendre", sec_tag_beam, 5),
            "pier_col": OPSE.beamIntegration("Legendre", sec_tag_pier, 5),
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁节点
        bent_cap_node = {
            "start": OPSE.node(0.0, -L / 2.0, PierH + lh_bc),
            "pier1": OPSE.node(0.0, -PierW / 2.0, PierH + lh_bc),
            "n1": OPSE.node(0.0, -PierW / 2.0 * (2 / 3), PierH + lh_bc),
            "n2": OPSE.node(0.0, -PierW / 2.0 * (1 / 3), PierH + lh_bc),
            "center": OPSE.node(0.0, 0.0, PierH + lh_bc),
            "n3": OPSE.node(0.0, PierW / 2.0 * (1 / 3), PierH + lh_bc),
            "n4": OPSE.node(0.0, PierW / 2.0 * (2 / 3), PierH + lh_bc),
            "brb": OPSE.node(0.0, PierW / 2.0 - 0.2, PierH + lh_bc),  # BRB 连接点
            "pier2": OPSE.node(0.0, PierW / 2.0, PierH + lh_bc),
            "end": OPSE.node(0.0, L / 2.0, PierH + lh_bc),
        }
        # 墩柱节点
        pier_node = {
            "pier_1": {
                "top": OPSE.node(0.0, -PierW / 2.0, PierH * (9 / 9)),
                "n1": OPSE.node(0.0, -PierW / 2.0, PierH * (8 / 9)),
                "n2": OPSE.node(0.0, -PierW / 2.0, PierH * (7 / 9)),
                "n3": OPSE.node(0.0, -PierW / 2.0, PierH * (6 / 9)),
                "n4": OPSE.node(0.0, -PierW / 2.0, PierH * (5 / 9)),
                "n5": OPSE.node(0.0, -PierW / 2.0, PierH * (4 / 9)),
                "n6": OPSE.node(0.0, -PierW / 2.0, PierH * (3 / 9)),
                "n7": OPSE.node(0.0, -PierW / 2.0, PierH * (2 / 9)),
                "n8": OPSE.node(0.0, -PierW / 2.0, PierH * (1 / 9)),
                "base": OPSE.node(0.0, -PierW / 2.0, PierH * (0 / 9)),
            },
            "pier_2": {
                "top": OPSE.node(0.0, PierW / 2.0, PierH * (9 / 9)),
                "n1": OPSE.node(0.0, PierW / 2.0, PierH * (8 / 9)),
                "n2": OPSE.node(0.0, PierW / 2.0, PierH * (7 / 9)),
                "n3": OPSE.node(0.0, PierW / 2.0, PierH * (6 / 9)),
                "n4": OPSE.node(0.0, PierW / 2.0, PierH * (5 / 9)),
                "n5": OPSE.node(0.0, PierW / 2.0, PierH * (4 / 9)),
                "n6": OPSE.node(0.0, PierW / 2.0, PierH * (3 / 9)),
                "n7": OPSE.node(0.0, PierW / 2.0, PierH * (2 / 9)),
                "n8": OPSE.node(0.0, PierW / 2.0, PierH * (1 / 9)),
                "base": OPSE.node(0.0, PierW / 2.0, PierH * (0 / 9)),
            },
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 辅助节点
        aid_node = {
            "brb_top": OPSE.node(0.0, PierW / 2.0 - 0.2, PierH),
            "brb_base": OPSE.node(0.0, -PierW / 2.0 + 0.2, 0.0),
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 节点约束
        OPSE.fix(pier_node["pier_1"]["base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(pier_node["pier_2"]["base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(aid_node["brb_base"], *(1, 1, 1, 1, 1, 1))

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 预应力材料
        PT_mat = OPSE.uniaxialMaterial(
            "Steel02", *(PT_fy, PT_Es, 0.01), *(18, 0.925, 0.15), *(0, 1, 0, 1, PT_f)
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元坐标转换
        beam_transf = ATf.ndm3(bent_cap_node["start"], bent_cap_node["end"])
        pier_1_transf = ATf.ndm3(
            pier_node["pier_1"]["top"], pier_node["pier_1"]["base"]
        )
        pier_2_transf = ATf.ndm3(
            pier_node["pier_2"]["top"], pier_node["pier_2"]["base"]
        )

        # 节点 tag 列表
        beam_node_tags = list(bent_cap_node.values())
        pier_1_node_tags = list(pier_node["pier_1"].values())
        pier_2_node_tags = list(pier_node["pier_2"].values())

        # 盖梁单元
        bent_cap_ele = {
            f"e{i + 1}": OPSE.element(
                "dispBeamColumn",
                *(beam_node_tags[i], beam_node_tags[i + 1]),
                *(beam_transf, integ["bent_cap"]),
            )
            for i in range(len(beam_node_tags) - 1)
        }

        # 墩柱单元
        pier_ele = {
            "pier_1": {
                f"e{i + 1}": OPSE.element(
                    "dispBeamColumn",
                    *(pier_1_node_tags[i], pier_1_node_tags[i + 1]),
                    *(pier_1_transf, integ["pier_col"]),
                )
                for i in range(len(pier_1_node_tags) - 1)
            },
            "pier_2": {
                f"e{i + 1}": OPSE.element(
                    "dispBeamColumn",
                    *(pier_2_node_tags[i], pier_2_node_tags[i + 1]),
                    *(pier_2_transf, integ["pier_col"]),
                )
                for i in range(len(pier_2_node_tags) - 1)
            },
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 预应力单元
        PT_ele = {
            "pier_1_PT": OPSE.element(
                "Truss",
                *(bent_cap_node["pier1"], pier_node["pier_1"]["base"]),
                PT_area,
                PT_mat,
            ),
            "pier_2_PT": OPSE.element(
                "Truss",
                *(bent_cap_node["pier2"], pier_node["pier_2"]["base"]),
                PT_area,
                PT_mat,
            ),
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 辅助单元 / 接触面相关
        aid_ele = {
            "beam_pier_1": OPSE.element(
                "elasticBeamColumn",
                *(bent_cap_node["pier1"], pier_node["pier_1"]["top"]),
                *(0.1, Ec, Gc),
                *(Kfit, Kfit, Kfit),
                pier_1_transf,
            ),
            "beam_pier_2": OPSE.element(
                "elasticBeamColumn",
                *(bent_cap_node["pier2"], pier_node["pier_2"]["top"]),
                *(0.1, Ec, Gc),
                *(Kfit, Kfit, Kfit),
                pier_2_transf,
            ),
            "beam_brb": OPSE.element(
                "elasticBeamColumn",
                *(bent_cap_node["brb"], aid_node["brb_top"]),
                *(0.1, Ec, Gc),
                *(Ubig, Ubig, Ubig),
                pier_1_transf,
            ),
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 刷新 将缓存同步至管理器
        OPSE.refresh()

        # 配置材料 预应力
        self.MM.tag_config(
            "uniaxialMaterial",
            tag=PT_mat,
            label="PT",
            params={"fy": PT_fy, "Es": PT_Es, "force": PT_f, "yield": PT_fy / PT_Es},
        )

        # 配置节点 - 位移控制点
        self.MM.tag_config(
            "node", tag=bent_cap_node["start"], label="disp_ctrl"
        )  # 位移控制节点

        # 配置单元 - 预应力
        self.MM.tag_config("element", tag=PT_ele["pier_1_PT"], label="PT_1")
        self.MM.tag_config("element", tag=PT_ele["pier_2_PT"], label="PT_2")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 输出数据库
        self.MM.to_excel(self.data_path / "ModelManager.xlsx")
        # 输出模型
        ops.printModel("-JSON", "-file", str(self.data_path / "thisModel.json"))
        # 可视化模型
        fig = opst.vis.plotly.plot_model()
        fig.write_html(self.data_path / "thisModel.html")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 节点收集
        nodes_beam = list(bent_cap_node.values())
        nodes_pier = list(pier_node["pier_1"].values()) + list(
            pier_node["pier_2"].values()
        )

        # 计算节点质量
        mass_beam = (
            L * self.MM.get_param("section", "bent_cap", "A") * rho / len(nodes_beam)
        )  # 盖梁质量
        mass_pier = (
            (2 * PierH)
            * (self.MM.get_param("section", "pier_col", "A") * rho)
            / len(nodes_pier)
        )  # 墩柱质量

        # 盖梁节点质量
        for i in bent_cap_node.values():
            OPSE.mass(i, *(mass_beam, mass_beam, mass_beam), *(0.0, 0.0, 0.0))
        # 墩柱节点质量
        for i in nodes_pier:
            OPSE.mass(i, *(mass_pier, mass_pier, mass_pier), *(0.0, 0.0, 0.0))


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":

    # 试验原始数据
    test_data_path = './.RAW_DATA'
    data_file = '/20230821WJRC.xlsx'
    data_file_BRB = '/20230826WJBRB.xlsx'
    # 导入
    test_data = pd.read_excel(f'{test_data_path + data_file}', header=0)
    # 清洗数据 转换为数值
    test_data['mm'] = pd.to_numeric(test_data['mm'], errors='coerce') * UNIT.mm
    test_data['N']  = pd.to_numeric(test_data['N'],  errors='coerce') * UNIT.n

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"

    g = 9.80665 * (UNIT.m / UNIT.sec**2)

    # 模型参数路径
    data_path = Path().cwd() / "OutData"
    data_path.mkdir(parents=True, exist_ok=True)

    # 实例化模型
    model = TwoPierModel(MM, data_path)
    model.model(info=False)

    ODB = opst.post.CreateODB(
        odb_tag=1,
        elastic_frame_sec_points=9,
        node_tags=None,
        frame_tags=None,
        fiber_ele_tags="ALL",
    )
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 时间序列
    ts = 1
    ops.timeSeries("Linear", ts)
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    # 重力荷载工况
    grav_pattern = 100
    ops.pattern("Plain", grav_pattern, ts)
    a = opst.pre.create_gravity_load(
        direction="Z", factor=-g
    )  # 从整体模型的节点质量获取重力荷载
    # print(*a.values(), sep="\n")

    ga = ANL.GravityAnalysis(ODB)
    ga.analyze()
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    ops.loadConst("-time", 0.0)

    ctrl = 1
    disp_path = np.array([0.0, 0.064, -0.064, 0.0]) * UNIT.m
    # disp_path = 0.064 * UNIT.m

    disp_pattern = 101
    ops.pattern("Plain", disp_pattern, ts)
    ops.load(ctrl, *(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))  # 节点荷载

    d = ANL.StaticAnalysis(disp_pattern, ODB)
    x, y = d.analyze(ctrl_node=ctrl, dof=2, targets=disp_path, max_step=0.001)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    plt.close("all")
    plt.plot(test_data["mm"], test_data['N'])
    plt.plot(x, y)
    plt.xlim(-0.07, 0.07)
    plt.ylim(-400, 400)
    plt.grid(True)
    plt.show()
