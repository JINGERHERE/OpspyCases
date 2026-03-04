#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：ModelHub.py
@Date    ：2025/8/1 20:18
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

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


class RockPierModel:

    def __init__(
        self,
        manager: opsu.pre.ModelManager,
        data_path: Union[Path, str, Literal[""]] = "",
    ) -> None:
        """
        自复位桥墩模型实例

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

        self.SEC_BRB_core: opst.pre.section.FiberSecMesh  # BRB核心
        self.SEC_BRB_link: opst.pre.section.FiberSecMesh  # BRB连接

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 保存路径
        self.data_path = Path("./")
        if data_path:
            # 数据路径
            self.data_path = Path(data_path)

        # 创建数据路径
        self.data_path.mkdir(parents=True, exist_ok=True)

    def model(self, fit: float = 0.0, info: bool = True) -> None:
        """
        创建 < 自复位桥墩 > 模型

        Args:
            fit (float, optional): 模型收敛刚度拟合。默认值为 0.。
            info (bool, optional): 是否显示模型信息。默认值为 True。

        Returns:
            None: 不返回任何值。
        """

        # 桥墩尺寸控制
        L = 1950.0 * UNIT.mm  # 盖梁长
        PierH = 2400.0 * UNIT.mm  # 墩柱高
        Seg = 1200.0 * UNIT.mm  # 节段长
        PierW = 1200.0 * UNIT.mm  # 墩柱中心间距

        # 混凝土性质
        Ec = opsu.pre.ConcHub.get_Ec("C40") * UNIT.mpa
        Gc = opsu.pre.ConcHub.get_G("C40") * UNIT.mpa
        rho = 2600 * (UNIT.kg / (UNIT.m**3))  # 密度：kg/m3

        # 耗能钢筋
        ED_fy, ED_Es, ED_area = (
            637.3 * UNIT.mpa,
            201 * UNIT.gpa,
            np.pi * (6 * UNIT.mm) ** 2,
        )
        # 耗能钢筋管道长
        ED_l = 100.0 * UNIT.mm  # 无粘结段 100 mm

        # 预应力
        PT_fy, PT_Es, PT_area = (
            1860 * UNIT.mpa,
            195 * UNIT.gpa,
            3 * 140 * UNIT.mm**2,
            # 3 * (np.pi * (15.2 * UNIT.mm / 2) ** 2)
        )
        PT_f = 0.4 * PT_fy  # 张拉控制力

        # 模型收敛刚度拟合
        # Kfit = 4.0e3
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
        self.SEC_cont = SectionHub.contact_surf(
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

        # 墩柱质心
        lw_p = (
            self.MM.get_param(category="section", label="pier_col", key="Width") / 2.0
        )
        lh_p = (
            self.MM.get_param(category="section", label="pier_col", key="Height") / 2.0
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面编号
        sec_tag_beam = self.MM.get_tag("section", label="bent_cap")[0]
        sec_tag_pier = self.MM.get_tag("section", label="pier_col")[0]
        sec_tag_surf = self.MM.get_tag("section", label="contact_surf")[0]

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
            "pier1": OPSE.node(0.0, -PierW / 2.0, PierH + lh_bc),  # 1 柱 对应节点
            "n1": OPSE.node(0.0, -PierW / 4.0, PierH + lh_bc),
            "center": OPSE.node(0.0, 0.0, PierH + lh_bc),
            "n2": OPSE.node(0.0, PierW / 4.0, PierH + lh_bc),
            "pier2": OPSE.node(0.0, PierW / 2.0, PierH + lh_bc),  # 2 柱 对应节点
            "end": OPSE.node(0.0, L / 2.0, PierH + lh_bc),
        }
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 1 柱 - 节点
        self.pier_1_node = {
            # 1 柱 - 上节段 节点
            "top": {
                # "start": OPSE.node(
                #     0.0, -PierW / 2, PierH - Seg * 0 - Seg * (0 / 4) - 0.1
                # ),  # 顶部节点
                "start": OPSE.node(
                    0.0, -PierW / 2, PierH - Seg * 0 - Seg * (0 / 4)
                ),  # 顶部节点
                "brb": OPSE.node(
                    0.0, -PierW / 2, PierH - Seg * 0 - Seg * (0 / 4) - lw_p
                ),  # BRB 柱中 节点
                "n1": OPSE.node(0.0, -PierW / 2, PierH - Seg * 0 - Seg * (1 / 4)),
                "n2": OPSE.node(0.0, -PierW / 2, PierH - Seg * 0 - Seg * (2 / 4)),
                "n3": OPSE.node(0.0, -PierW / 2, PierH - Seg * 0 - Seg * (3 / 4)),
                "end": OPSE.node(
                    0.0, -PierW / 2, PierH - Seg * 0 - Seg * (4 / 4)
                ),  # 底部节点
                # "end": OPSE.node(
                #     0.0, -PierW / 2, PierH - Seg * 0 - Seg * (4 / 4) + 0.1
                # ),  # 底部节点
            },
            # 1 柱 - 下节段 节点
            "base": {
                # "start": OPSE.node(
                #     0.0, -PierW / 2, PierH - Seg * 1 - Seg * (0 / 4) - 0.1
                # ),  # 顶部节点
                "start": OPSE.node(
                    0.0, -PierW / 2, PierH - Seg * 1 - Seg * (0 / 4)
                ),  # 顶部节点
                "n1": OPSE.node(0.0, -PierW / 2, PierH - Seg * 1 - Seg * (1 / 4)),
                "n2": OPSE.node(0.0, -PierW / 2, PierH - Seg * 1 - Seg * (2 / 4)),
                "n3": OPSE.node(0.0, -PierW / 2, PierH - Seg * 1 - Seg * (3 / 4)),
                "brb": OPSE.node(
                    0.0, -PierW / 2, PierH - Seg * 1 - Seg * (4 / 4) + lw_p
                ),  # BRB 柱中 节点
                "end": OPSE.node(
                    0.0, -PierW / 2, PierH - Seg * 1 - Seg * (4 / 4)
                ),  # 底部节点
                # "end": OPSE.node(
                #     0.0, -PierW / 2, PierH - Seg * 1 - Seg * (4 / 4) + 0.1
                # ),  # 底部节点
            },
        }
        # 2 柱 - 节点
        self.pier_2_node = {
            # 2 柱 - 上节段 节点
            "top": {
                # "start": OPSE.node(
                #     0.0, PierW / 2, PierH - Seg * 0 - Seg * (0 / 4) - 0.1
                # ),  # 顶部节点
                "start": OPSE.node(
                    0.0, PierW / 2, PierH - Seg * 0 - Seg * (0 / 4)
                ),  # 顶部节点
                "n1": OPSE.node(0.0, PierW / 2, PierH - Seg * 0 - Seg * (1 / 4)),
                "n2": OPSE.node(0.0, PierW / 2, PierH - Seg * 0 - Seg * (2 / 4)),
                "n3": OPSE.node(0.0, PierW / 2, PierH - Seg * 0 - Seg * (3 / 4)),
                "brb": OPSE.node(
                    0.0, PierW / 2, PierH - Seg * 0 - Seg * (4 / 4) + lw_p
                ),  # BRB 柱中 节点
                "end": OPSE.node(
                    0.0, PierW / 2, PierH - Seg * 0 - Seg * (4 / 4)
                ),  # 底部节点
                # "end": OPSE.node(
                #     0.0, PierW / 2, PierH - Seg * 0 - Seg * (4 / 4) + 0.1
                # ),  # 底部节点
            },
            # 2 柱 - 下节段 节点
            "base": {
                # "start": OPSE.node(
                #     0.0, PierW / 2, PierH - Seg * 1 - Seg * (0 / 4) - 0.1
                # ),  # 顶部节点
                "start": OPSE.node(
                    0.0, PierW / 2, PierH - Seg * 1 - Seg * (0 / 4)
                ),  # 顶部节点
                "brb": OPSE.node(
                    0.0, PierW / 2, PierH - Seg * 1 - Seg * (0 / 4) - lw_p
                ),  # BRB 柱中 节点
                "n1": OPSE.node(0.0, PierW / 2, PierH - Seg * 1 - Seg * (1 / 4)),
                "n2": OPSE.node(0.0, PierW / 2, PierH - Seg * 1 - Seg * (2 / 4)),
                "n3": OPSE.node(0.0, PierW / 2, PierH - Seg * 1 - Seg * (3 / 4)),
                "end": OPSE.node(
                    0.0, PierW / 2, PierH - Seg * 1 - Seg * (4 / 4)
                ),  # 底部节点
                # "end": OPSE.node(
                #     0.0, PierW / 2, PierH - Seg * 1 - Seg * (4 / 4) + 0.1
                # ),  # 底部节点
            },
        }
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 耗能钢筋节点
        ED_node = {
            "pier_1": {
                # 耗能钢筋 柱中节点
                "ED_1_top": OPSE.node(0.0, -PierW / 2.0 - lw_p / 2, 0.0),
                # 耗能钢筋 1 底部固定
                "ED_1_base": OPSE.node(0.0, -PierW / 2.0 - lw_p / 2, -ED_l),
                # 耗能钢筋 柱中节点
                "ED_2_top": OPSE.node(0.0, -PierW / 2.0 + lw_p / 2, 0.0),
                # 耗能钢筋 2 底部固定
                "ED_2_base": OPSE.node(0.0, -PierW / 2.0 + lw_p / 2, -ED_l),
            },
            "pier_2": {
                # 耗能钢筋 柱中节点
                "ED_1_top": OPSE.node(0.0, PierW / 2.0 - lw_p / 2, 0.0),
                # 耗能钢筋 1 底部固定
                "ED_1_base": OPSE.node(0.0, PierW / 2.0 - lw_p / 2, -ED_l),
                # 耗能钢筋 柱中节点
                "ED_2_top": OPSE.node(0.0, PierW / 2.0 + lw_p / 2, 0.0),
                # 耗能钢筋 2 底部固定
                "ED_2_base": OPSE.node(0.0, PierW / 2.0 + lw_p / 2, -ED_l),
            },
        }
        # 预应力节点
        PT_node = {
            "pier_1": {
                "top": OPSE.node(0.0, -PierW / 2.0, PierH + lh_bc * 2),
                "base": OPSE.node(0.0, -PierW / 2.0, -lh_bc * 2),
            },
            "pier_2": {
                "top": OPSE.node(0.0, PierW / 2.0, PierH + lh_bc * 2),
                "base": OPSE.node(0.0, PierW / 2.0, -lh_bc * 2),
            },
        }
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 辅助节点
        self.aid_node = {
            "pier_1": {
                "surf_top": OPSE.node(0.0, -PierW / 2, PierH),  # 1 柱顶 接触面节点
                "surf_base": OPSE.node(0.0, -PierW / 2, 0.0),  # 1 柱底 接触面节点
                # 耗能钢筋 压缩限位
                "ED_1_lim": OPSE.node(0.0, -PierW / 2.0 - lw_p / 2, 0.0),
                "ED_2_lim": OPSE.node(0.0, -PierW / 2.0 + lw_p / 2, 0.0),
                # 柱边缘 BRB节点
                "edge_top": OPSE.node(
                    0.0, -PierW / 2 + lw_p, PierH - Seg * 0 - Seg * (0 / 4) - lw_p
                ),  # BRB 上节段 柱边缘 节点
                "edge_base": OPSE.node(
                    0.0, -PierW / 2 + lw_p, PierH - Seg * 1 - Seg * (4 / 4) + lw_p
                ),  # BRB 下节段 柱边缘 节点
            },
            "pier_2": {
                "surf_top": OPSE.node(0.0, PierW / 2, PierH),  # 2 柱顶 接触面节点
                "surf_base": OPSE.node(0.0, PierW / 2, 0.0),  # 2 柱底 接触面节点
                # 耗能钢筋 压缩限位
                "ED_1_lim": OPSE.node(0.0, PierW / 2.0 - lw_p / 2, 0.0),
                "ED_2_lim": OPSE.node(0.0, PierW / 2.0 + lw_p / 2, 0.0),
                # 柱边缘 BRB节点
                "edge_top": OPSE.node(
                    0.0, PierW / 2 - lw_p, PierH - Seg * 0 - Seg * (4 / 4) + lw_p
                ),  # BRB 上节段 柱边缘 节点
                "edge_base": OPSE.node(
                    0.0, PierW / 2 - lw_p, PierH - Seg * 1 - Seg * (0 / 4) - lw_p
                ),  # BRB 下节段 柱边缘 节点
            },
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 柱 1 节点约束
        OPSE.fix(self.aid_node["pier_1"]["surf_base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(self.aid_node["pier_1"]["ED_1_lim"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(self.aid_node["pier_1"]["ED_2_lim"], *(1, 1, 1, 1, 1, 1))
        # 柱 2 节点约束
        OPSE.fix(self.aid_node["pier_2"]["surf_base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(self.aid_node["pier_2"]["ED_1_lim"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(self.aid_node["pier_2"]["ED_2_lim"], *(1, 1, 1, 1, 1, 1))

        # 耗能钢筋
        OPSE.fix(ED_node["pier_1"]["ED_1_base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(ED_node["pier_1"]["ED_2_base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(ED_node["pier_2"]["ED_1_base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(ED_node["pier_2"]["ED_2_base"], *(1, 1, 1, 1, 1, 1))
        # 预应力 节点约束
        OPSE.fix(PT_node["pier_1"]["base"], *(1, 1, 1, 1, 1, 1))
        OPSE.fix(PT_node["pier_2"]["base"], *(1, 1, 1, 1, 1, 1))

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 耗能钢筋材料
        ED_mat = OPSE.uniaxialMaterial(
            "Steel02", *(ED_fy, ED_Es, 0.01), *(18, 0.925, 0.15)
        )

        # 预应力材料
        PT_mat = OPSE.uniaxialMaterial(
            "Steel02", *(PT_fy, PT_Es, 0.01), *(18, 0.925, 0.15), *(0, 1, 0, 1, PT_f)
        )

        # 辅助材料
        aid_mat = {
            "fix": OPSE.uniaxialMaterial("Elastic", Ubig),
            "free": OPSE.uniaxialMaterial("Elastic", Usmall),
            # 用于 耗能钢筋 压缩限位
            "ENT": OPSE.uniaxialMaterial("ENT", 20 * UNIT.gpa),
        }
        # 用于刚度拟合
        if fit:
            # if Kfit:
            fit_mat = OPSE.uniaxialMaterial("Elastic", fit)
            # fit_mat = OPSE.uniaxialMaterial("Elastic", Kfit)
        else:
            fit_mat = aid_mat["fix"]

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 单元坐标转换
        transf_beam = ATf.ndm3(bent_cap_node["start"], bent_cap_node["end"])
        transf_pier_1 = ATf.ndm3(
            self.pier_1_node["top"]["start"], self.pier_1_node["base"]["end"]
        )
        transf_pier_2 = ATf.ndm3(
            self.pier_2_node["top"]["start"], self.pier_2_node["base"]["end"]
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 盖梁单元
        bent_cap_ele = {
            "e1": OPSE.element(
                "dispBeamColumn",
                *(bent_cap_node["start"], bent_cap_node["pier1"]),
                *(transf_beam, integ["bent_cap"]),
            ),
            "e2": OPSE.element(
                "dispBeamColumn",
                *(bent_cap_node["pier1"], bent_cap_node["n1"]),
                *(transf_beam, integ["bent_cap"]),
            ),
            "e2": OPSE.element(
                "dispBeamColumn",
                *(bent_cap_node["n1"], bent_cap_node["center"]),
                *(transf_beam, integ["bent_cap"]),
            ),
            "e3": OPSE.element(
                "dispBeamColumn",
                *(bent_cap_node["center"], bent_cap_node["n2"]),
                *(transf_beam, integ["bent_cap"]),
            ),
            "e4": OPSE.element(
                "dispBeamColumn",
                *(bent_cap_node["n2"], bent_cap_node["pier2"]),
                *(transf_beam, integ["bent_cap"]),
            ),
            "e5": OPSE.element(
                "dispBeamColumn",
                *(bent_cap_node["pier2"], bent_cap_node["end"]),
                *(transf_beam, integ["bent_cap"]),
            ),
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 1 柱 单元
        pier_1_ele = {
            "top": {
                "e1": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["top"]["start"], self.pier_1_node["top"]["brb"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e2": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["top"]["brb"], self.pier_1_node["top"]["n1"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e3": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["top"]["n1"], self.pier_1_node["top"]["n2"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e4": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["top"]["n2"], self.pier_1_node["top"]["n3"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e5": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["top"]["n3"], self.pier_1_node["top"]["end"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
            },
            "base": {
                "e1": OPSE.element(
                    "dispBeamColumn",
                    *(
                        self.pier_1_node["base"]["start"],
                        self.pier_1_node["base"]["n1"],
                    ),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e2": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["base"]["n1"], self.pier_1_node["base"]["n2"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e3": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["base"]["n2"], self.pier_1_node["base"]["n3"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e4": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["base"]["n3"], self.pier_1_node["base"]["brb"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
                "e5": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_1_node["base"]["brb"], self.pier_1_node["base"]["end"]),
                    *(transf_pier_1, integ["pier_col"]),
                ),
            },
        }
        # 2 柱 单元
        pier_2_ele = {
            "top": {
                "e1": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["top"]["start"], self.pier_2_node["top"]["n1"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e2": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["top"]["n1"], self.pier_2_node["top"]["n2"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e3": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["top"]["n2"], self.pier_2_node["top"]["n3"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e4": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["top"]["n3"], self.pier_2_node["top"]["brb"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e5": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["top"]["brb"], self.pier_2_node["top"]["end"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
            },
            "base": {
                "e1": OPSE.element(
                    "dispBeamColumn",
                    *(
                        self.pier_2_node["base"]["start"],
                        self.pier_2_node["base"]["brb"],
                    ),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e2": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["base"]["brb"], self.pier_2_node["base"]["n1"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e3": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["base"]["n1"], self.pier_2_node["base"]["n2"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e4": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["base"]["n2"], self.pier_2_node["base"]["n3"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
                "e5": OPSE.element(
                    "dispBeamColumn",
                    *(self.pier_2_node["base"]["n3"], self.pier_2_node["base"]["end"]),
                    *(transf_pier_2, integ["pier_col"]),
                ),
            },
        }
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 耗能钢筋单元
        ED_ele = {
            "pier_1": {
                "ED1": OPSE.element(
                    "Truss",
                    *(ED_node["pier_1"]["ED_1_top"], ED_node["pier_1"]["ED_1_base"]),
                    *(ED_area, ED_mat),
                ),
                "ED2": OPSE.element(
                    "Truss",
                    *(ED_node["pier_1"]["ED_2_top"], ED_node["pier_1"]["ED_2_base"]),
                    *(ED_area, ED_mat),
                ),
            },
            "pier_2": {
                "ED1": OPSE.element(
                    "Truss",
                    *(ED_node["pier_2"]["ED_1_top"], ED_node["pier_2"]["ED_1_base"]),
                    *(ED_area, ED_mat),
                ),
                "ED2": OPSE.element(
                    "Truss",
                    *(ED_node["pier_2"]["ED_2_top"], ED_node["pier_2"]["ED_2_base"]),
                    *(ED_area, ED_mat),
                ),
            },
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 预应力单元
        PT_ele = {
            "pier_1_PT": OPSE.element(
                "Truss",
                *(PT_node["pier_1"]["top"], PT_node["pier_1"]["base"]),
                PT_area,
                PT_mat,
            ),
            "pier_2_PT": OPSE.element(
                "Truss",
                *(PT_node["pier_2"]["top"], PT_node["pier_2"]["base"]),
                PT_area,
                PT_mat,
            ),
        }
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 接触面 < 零长单元 > 节点连接顺序
        link_surf = {
            "pier_1": {
                "top": (
                    self.aid_node["pier_1"]["surf_top"],
                    self.pier_1_node["top"]["start"],
                ),
                "mid": (
                    self.pier_1_node["top"]["end"],
                    self.pier_1_node["base"]["start"],
                ),
                "base": (
                    self.pier_1_node["base"]["end"],
                    self.aid_node["pier_1"]["surf_base"],
                ),
            },
            "pier_2": {
                "top": (
                    self.aid_node["pier_2"]["surf_top"],
                    self.pier_2_node["top"]["start"],
                ),
                "mid": (
                    self.pier_2_node["top"]["end"],
                    self.pier_2_node["base"]["start"],
                ),
                "base": (
                    self.pier_2_node["base"]["end"],
                    self.aid_node["pier_2"]["surf_base"],
                ),
            },
        }  # 作为 equalDOF 时 需要 *reversed() 将节点反序

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 竖向 零长单元 坐标转换
        vecx = (0, 0, -1)  # 局部 x -> 整体坐标
        vecyp = (0, 1, 0)  # 局部 y -> 整体坐标

        # 材料 对应 自由度
        dir_mats = (
            *(aid_mat["ENT"], fit_mat, fit_mat),
            *(aid_mat["free"], aid_mat["free"], aid_mat["free"]),
        )  # 零长单元局部方向
        dirs = (1, 2, 3, 4, 5, 6)

        # 辅助单元 / 接触面相关
        aid_ele = {
            "pier_1": {
                "beam_top": OPSE.element(
                    "elasticBeamColumn",
                    *(PT_node["pier_1"]["top"], bent_cap_node["pier1"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(PT_node["pier_1"]["top"], bent_cap_node["pier1"]),
                ),  # 盖梁 - 预应力顶
                "beam_col": OPSE.element(
                    "elasticBeamColumn",
                    *(bent_cap_node["pier1"], self.aid_node["pier_1"]["surf_top"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(bent_cap_node["pier1"], self.aid_node["pier_1"]["surf_top"]),
                ),  # 盖梁 - 柱顶
                "pier_top": OPSE.element(
                    "zeroLengthSection",
                    *link_surf["pier_1"]["top"],
                    sec_tag_surf,
                    *("-orient", *vecx, *vecyp),
                ),  # 柱顶 接触面
                "pier_mid": OPSE.element(
                    "zeroLengthSection",
                    *link_surf["pier_1"]["mid"],
                    sec_tag_surf,
                    *("-orient", *vecx, *vecyp),
                ),  # 柱中 接触面
                "pier_base": OPSE.element(
                    "zeroLengthSection",
                    *link_surf["pier_1"]["base"],
                    sec_tag_surf,
                    *("-orient", *vecx, *vecyp),
                ),  # 柱底 接触面
                "pier_base_edge_1": OPSE.element(
                    "elasticBeamColumn",
                    *(self.pier_1_node["base"]["end"], ED_node["pier_1"]["ED_1_top"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(
                        self.pier_1_node["base"]["end"], ED_node["pier_1"]["ED_1_top"]
                    ),
                ),  # 柱底 边缘钢臂 1
                "pier_base_edge_2": OPSE.element(
                    "elasticBeamColumn",
                    *(self.pier_1_node["base"]["end"], ED_node["pier_1"]["ED_2_top"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(
                        self.pier_1_node["base"]["end"], ED_node["pier_1"]["ED_2_top"]
                    ),
                ),  # 柱底 边缘钢臂 2
                "ED_1_lim": OPSE.element(
                    "zeroLength",
                    *(ED_node["pier_1"]["ED_1_top"], self.aid_node["pier_1"]["ED_1_lim"]),
                    *("-mat", *dir_mats, "-dir", *dirs),
                    *("-orient", *vecx, *vecyp),
                ),  # 耗能钢筋 1 压缩限位
                "ED_2_lim": OPSE.element(
                    "zeroLength",
                    *(ED_node["pier_1"]["ED_2_top"], self.aid_node["pier_1"]["ED_2_lim"]),
                    *("-mat", *dir_mats, "-dir", *dirs),
                    *("-orient", *vecx, *vecyp),
                ),  # 耗能钢筋 2 压缩限位
            },
            "pier_2": {
                "beam_top": OPSE.element(
                    "elasticBeamColumn",
                    *(PT_node["pier_2"]["top"], bent_cap_node["pier2"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(PT_node["pier_2"]["top"], bent_cap_node["pier2"]),
                ),  # 盖梁 - 预应力顶
                "beam_col": OPSE.element(
                    "elasticBeamColumn",
                    *(bent_cap_node["pier2"], self.aid_node["pier_2"]["surf_top"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(bent_cap_node["pier1"], self.aid_node["pier_1"]["surf_top"]),
                ),  # 盖梁 - 柱顶
                "pier_top": OPSE.element(
                    "zeroLengthSection",
                    *link_surf["pier_2"]["top"],
                    sec_tag_surf,
                    *("-orient", *vecx, *vecyp),
                ),  # 柱顶 接触面
                "pier_mid": OPSE.element(
                    "zeroLengthSection",
                    *link_surf["pier_2"]["mid"],
                    sec_tag_surf,
                    *("-orient", *vecx, *vecyp),
                ),  # 柱中 接触面
                "pier_base": OPSE.element(
                    "zeroLengthSection",
                    *link_surf["pier_2"]["base"],
                    sec_tag_surf,
                    *("-orient", *vecx, *vecyp),
                ),  # 柱底 接触面
                "pier_base_edge_1": OPSE.element(
                    "elasticBeamColumn",
                    *(self.pier_2_node["base"]["end"], ED_node["pier_2"]["ED_1_top"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(
                        self.pier_2_node["base"]["end"], ED_node["pier_2"]["ED_1_top"]
                    ),
                ),  # 柱底 边缘钢臂 1
                "pier_base_edge_2": OPSE.element(
                    "elasticBeamColumn",
                    *(self.pier_2_node["base"]["end"], ED_node["pier_2"]["ED_2_top"]),
                    *(0.1, Ec, Gc),
                    *(Ubig, Ubig, Ubig),
                    ATf.ndm3(
                        self.pier_2_node["base"]["end"], ED_node["pier_2"]["ED_2_top"]
                    ),
                ),  # 柱底 边缘钢臂 2
                "ED_1_lim": OPSE.element(
                    "zeroLength",
                    *(ED_node["pier_2"]["ED_1_top"], self.aid_node["pier_2"]["ED_1_lim"]),
                    *("-mat", *dir_mats, "-dir", *dirs),
                    *("-orient", *vecx, *vecyp),
                ),  # 耗能钢筋 1 压缩限位
                "ED_2_lim": OPSE.element(
                    "zeroLength",
                    *(ED_node["pier_2"]["ED_2_top"], self.aid_node["pier_2"]["ED_2_lim"]),
                    *("-mat", *dir_mats, "-dir", *dirs),
                    *("-orient", *vecx, *vecyp),
                ),  # 耗能钢筋 2 压缩限位
            },
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 自由度等效
        brb_dof_equal = (1, 2, 3, 4, 5, 6)  # BRB 全自由度约束
        # 柱 1 BRB 自由度等效
        OPSE.equalDOF(
            *(self.pier_1_node["top"]["brb"], self.aid_node["pier_1"]["edge_top"]),
            *brb_dof_equal,
        )
        OPSE.equalDOF(
            *(self.pier_1_node["base"]["brb"], self.aid_node["pier_1"]["edge_base"]),
            *brb_dof_equal,
        )
        # 柱 2 BRB 自由度等效
        OPSE.equalDOF(
            *(self.pier_2_node["top"]["brb"], self.aid_node["pier_2"]["edge_top"]),
            *brb_dof_equal,
        )
        OPSE.equalDOF(
            *(self.pier_2_node["base"]["brb"], self.aid_node["pier_2"]["edge_base"]),
            *brb_dof_equal,
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 自由度等效
        surf_dof_equal = (1, 2)  # 固定 水平方向 位移
        # 柱 1 接触面 自由度等效
        OPSE.equalDOF(*reversed(link_surf["pier_1"]["top"]), *surf_dof_equal)
        OPSE.equalDOF(*reversed(link_surf["pier_1"]["mid"]), *surf_dof_equal)
        # 柱 2 接触面 自由度等效
        OPSE.equalDOF(*reversed(link_surf["pier_2"]["top"]), *surf_dof_equal)
        OPSE.equalDOF(*reversed(link_surf["pier_2"]["mid"]), *surf_dof_equal)
        # 柱底滑移控制
        if not fit:
            # if not Kfit:
            OPSE.equalDOF(*reversed(link_surf["pier_1"]["base"]), *surf_dof_equal)
            OPSE.equalDOF(*reversed(link_surf["pier_2"]["base"]), *surf_dof_equal)
        # 自由度继承以底部节点为主 ----- ----- -----

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 刷新 将缓存同步至管理器
        OPSE.refresh()

        # 配置材料 耗能钢筋
        self.MM.tag_config(
            "uniaxialMaterial",
            tag=ED_mat,
            label="ED",
            params={"fy": ED_fy, "Es": ED_Es, "area": ED_area, "yield": ED_fy / ED_Es},
        )
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

        # 配置单元 - 底部接触面
        self.MM.tag_config(
            "element", tag=aid_ele["pier_1"]["pier_base"], label="pier_1_surf"
        )
        self.MM.tag_config(
            "element", tag=aid_ele["pier_2"]["pier_base"], label="pier_2_surf"
        )

        # 配置单元 - 耗能钢筋
        self.MM.tag_config("element", tag=ED_ele["pier_1"]["ED1"], label="pier_1_ED_1")
        self.MM.tag_config("element", tag=ED_ele["pier_1"]["ED2"], label="pier_1_ED_2")
        self.MM.tag_config("element", tag=ED_ele["pier_2"]["ED1"], label="pier_2_ED_1")
        self.MM.tag_config("element", tag=ED_ele["pier_2"]["ED2"], label="pier_2_ED_2")

        # 配置单元 - 预应力
        self.MM.tag_config("element", tag=PT_ele["pier_1_PT"], label="PT_1")
        self.MM.tag_config("element", tag=PT_ele["pier_2_PT"], label="PT_2")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 节点收集
        nodes_beam = list(bent_cap_node.values())
        nodes_pier = (
            list(self.pier_1_node["top"].values())
            + list(self.pier_1_node["base"].values())
            + list(self.pier_2_node["top"].values())
            + list(self.pier_2_node["base"].values())
        )

        # 计算节点质量
        mass_beam = (
            L * self.MM.get_param("section", "bent_cap", "A") * rho / len(nodes_beam)
        )  # 盖梁质量
        mass_pier = (
            (2 * PierH)
            * self.MM.get_param("section", "pier_col", "A")
            * rho
            / len(nodes_pier)
        )  # 墩柱质量

        # 盖梁节点质量
        for i in bent_cap_node.values():
            OPSE.mass(i, *(mass_beam, mass_beam, mass_beam), *(0.0, 0.0, 0.0))
        # 墩柱节点质量
        for i in nodes_pier:
            OPSE.mass(i, *(mass_pier, mass_pier, mass_pier), *(0.0, 0.0, 0.0))

    def BRB(self, info: bool = False) -> None:
        """
        配置 BRB 模型
        """

        # 创建截面
        self.SEC_BRB_core = SectionHub.brb_core(
            manager=self.MM, info=info, save_sec=self.data_path
        )
        self.SEC_BRB_link = SectionHub.brb_link(
            manager=self.MM, info=info, save_sec=self.data_path
        )

        # 刷新 将缓存同步至管理器
        OPSE.refresh()
        # 截面编号
        sec_tag_brb_core = self.MM.get_tag("section", label="brb_core")[0]
        sec_tag_brb_link = self.MM.get_tag("section", label="brb_link")[0]

        # 截面单元积分
        brb_core_integ = OPSE.beamIntegration("Legendre", sec_tag_brb_core, 5)
        brb_link_integ = OPSE.beamIntegration("Legendre", sec_tag_brb_link, 5)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        def _creator(node_i: int, node_j: int) -> int:

            # 检查输入
            # print(f'创建 BRB 模型：{node_i} -> {node_j}')

            # 获取两端连接节点坐标
            node_i_coord = np.array(ops.nodeCoord(node_i))  # 不存在会报错
            node_j_coord = np.array(ops.nodeCoord(node_j))

            # 计算距离
            L = np.linalg.norm(node_j_coord - node_i_coord)  # 固定端总长度

            # 计算直线参数
            center_point = (node_i_coord + node_j_coord) / 2  # 直线中心坐标
            dir_vector = (node_j_coord - node_i_coord) / L  # 直线方向向量

            "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
            # 单元坐标转换
            transf_brb = ATf.ndm3(node_i, node_j)

            "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
            # BRB 核心段 长度
            core_len = 360 * UNIT.mm

            # 缩放后的端点坐标
            core_i_coord = center_point - (dir_vector * core_len / 2)
            core_j_coord = center_point + (dir_vector * core_len / 2)

            # 节点
            core_i_tag = OPSE.node(*core_i_coord)  # 核心 I 端
            core_j_tag = OPSE.node(*core_j_coord)  # 核心 J 端

            # 连接段单元
            # link_i_tag = OPSE.element(
            #     "dispBeamColumn",
            #     *(node_i, core_i_tag),
            #     *(transf_brb, brb_link_integ),
            # )
            # link_j_tag = OPSE.element(
            #     "dispBeamColumn",
            #     *(core_j_tag, node_j),
            #     *(transf_brb, brb_link_integ),
            # )

            Ubig = 1.e6
            # link_i_tag = OPSE.element(
            #     "elasticBeamColumn",
            #     *(node_i, core_i_tag),
            #     *(1.536e-3, 200 * UNIT.gpa, 100 * UNIT.gpa),
            #     *(Ubig, Ubig, Ubig),
            #     transf_brb,
            # )
            # link_j_tag = OPSE.element(
            #     "elasticBeamColumn",
            #     *(core_j_tag, node_j),
            #     *(1.536e-3, 200 * UNIT.gpa, 100 * UNIT.gpa),
            #     *(Ubig, Ubig, Ubig),
            #     transf_brb,
            # )

            link_i_tag = OPSE.element(
                "elasticBeamColumn",
                *(node_i, core_i_tag),
                *(sec_tag_brb_link, transf_brb)
            )
            link_j_tag = OPSE.element(
                "elasticBeamColumn",
                *(core_j_tag, node_j),
                *(sec_tag_brb_link, transf_brb)
            )

            # 核心单元
            core_tag = OPSE.element(
                "dispBeamColumn",
                *(core_i_tag, core_j_tag),
                *(transf_brb, brb_link_integ),
            )
            # core_tag = OPSE.element(
            #     "TrussSection",
            #     *(core_i_tag, core_j_tag),
            #     sec_tag_brb_core,
            # )

            return core_tag

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 节点定位
        core_tag_top = _creator(
            node_i=self.aid_node["pier_1"]["edge_top"],
            node_j=self.aid_node["pier_2"]["edge_top"],
        )
        # print(f'顶部: {core_tag_top}')
        core_tag_base = _creator(
            node_i=self.aid_node["pier_1"]["edge_base"],
            node_j=self.aid_node["pier_2"]["edge_base"],
        )
        # print(f'底部: {core_tag_base}')

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 刷新 将缓存同步至管理器
        OPSE.refresh()
        # 核心单元
        self.MM.tag_config("element", tag=core_tag_top, label="brb_top")
        self.MM.tag_config("element", tag=core_tag_base, label="brb_base")

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""

if __name__ == "__main__":
    g = 9.80665 * (UNIT.m / UNIT.sec**2)

    # 模型参数路径
    data_path = Path().cwd() / "OutData"
    data_path.mkdir(parents=True, exist_ok=True)

    # 实例化模型
    model = RockPierModel(MM, data_path)
    model.model(info=False)
    # model.model(1.5e3, False)
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"

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
    # disp_path = np.array([0.0, 0.12, -0.12, 0.0]) * UNIT.m
    disp_path = 0.12 * UNIT.m

    disp_pattern = 101
    ops.pattern("Plain", disp_pattern, ts)
    ops.load(ctrl, *(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))  # 节点荷载

    d = ANL.StaticAnalysis(disp_pattern, ODB)
    x, y = d.analyze(ctrl_node=ctrl, dof=2, targets=disp_path, max_step=0.001)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    plt.close("all")
    plt.plot(x, y)
    plt.xlim(-0.12, 0.12)
    plt.ylim(-200, 200)
    plt.grid(True)
    plt.show()

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    fig = opst.vis.plotly.plot_model(
        # show_node_numbering=True,
        show_local_axes=True
    )
    # fig.show()
    fig.write_html(data_path / "thisModel.html")
