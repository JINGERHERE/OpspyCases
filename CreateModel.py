#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：CreateModel.py
@Date    ：2025/8/1 19:22
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""


from scipy.integrate import trapezoid
from pathlib import Path
import os
import sys
import time
import warnings
from typing import Literal, TypeAlias, Union, Callable, Any, Optional, Tuple
import matplotlib.pyplot as plt
# import matplotlib
import rich
import numpy as np
import pandas as pd
import xarray as xr
import openseespy.opensees as ops
import opstool as opst
import opstool.vis.plotly as opsplt
import opstool.vis.pyvista as opsvis
# from inspect import currentframe as curt_fra
from itertools import batched, product, pairwise
import pickle

import imageio.v2 as imageio

# from script.pre import NodeTools
# from script.base import random_color, rich_showwarning
# from script import UNIT, PVs



# from Part_MatSec_MomentCurvature import MPhiSection
# from script.post import DamageStateTools
# from Script_ModelCreate import ModelCreateTools

import ops_utilities as opsu
import AnalysisLibraries as ALs
from SectionHub import SectionHub

from ops_utilities import rich_showwarning
import warnings
warnings.showwarning = rich_showwarning

"""
# --------------------------------------------------
# ========== < CreateModel > ==========
# --------------------------------------------------
"""

class SectionModel:
    
    def __init__(
        self,
        manager: opsu.pre.ModelManager, num: int,
        save: Union[Path, str, Literal['']] = ''
        ) -> None:
        
        """
        截面模型实例类
            - 
        
        Args:
            manager (opsu.pre.ModelManager): 模型管理器对象。
            num (int): 'SectionHub' 中截面顺序对应序号。
            save (Union[Path, str, Literal['']], optional): 保存路径。 `默认值：` '当前路径'.
        
        Returns:
            None: 不返回任何值。
        
        Raises:
            ValueError: 截面序号超出范围。
        
        """
        
        # 管理器
        self.MM = manager
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 编号
        self.node_load = manager.next_tag(category='node', label='load')
        self.node_fix = manager.next_tag(category='node', label='fix')
        self.ele_sec = manager.next_tag(category='element', label='sec')
        
        self.ts = manager.next_tag(category='timeSeries', label='ts')
        self.axial_force = manager.next_tag(category='pattern')
        self.ctrl_force = manager.next_tag(category='pattern', label='ctrl_force')
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取可调用截面对象
        self.sections = opsu.get_callables(SectionHub)
        if num > len(self.sections):
            raise ValueError(f"在 '{[n.name for n in self.sections]}' 中没有找到更多截面，num <= {len(self.sections)}。")
        # 当前截面函数
        self.current_sec = self.sections[num]
        # 当前截面名称
        self.current_sec_name = self.current_sec.name
        # 当前截面的返回对象
        self.SEC: opst.pre.section.FiberSecMesh
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 保存路径
        self.data_path = Path('./')
        if save:
            # 数据路径
            self.data_path = Path(save)
            self.data_path.mkdir(parents=True, exist_ok=True)
            # opstool 数据路径
            opst.post.set_odb_path(str(out_path))
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def model(self, deg: float = 0.):
        
        """
        创建截面模型。
        
        Args:
            deg (float, optional): 截面模型旋转角度。 `默认值：` 0.
                - 单位：度。
        
        Returns:
            str: 当前截面名称。
        """
        
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
        # 截面模型 - node
        ops.node(self.node_load, *(0., 0., 0.))
        ops.node(self.node_fix, *(0., 0., 0.))
        
        # 截面模型 - 约束
        ops.fix(self.node_fix, *(1, 1, 1, 1, 1, 1))
        ops.fix(self.node_load, *(0, 1, 1, 1, 0, 0))
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面模型 - section
        self.SEC = self.current_sec.callable(manager=self.MM, save_sec=self.data_path) # 创建截面模型
        sec_tag = self.MM.get_tag(category='section', label=self.current_sec_name)[0]
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面模型 - element
        ops.element(
            'zeroLengthSection', self.ele_sec, *(self.node_fix, self.node_load), sec_tag,
            '-orient', *(1, 0, 0), *(0, np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg)))
            ) # 局部x指向整体X，局部y指向整体Y（可调整旋转角）
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        if self.data_path:
            self.MM.to_excel(Path(self.data_path) / f'ModelManager_{self.current_sec_name}.xlsx')
        
        return self.current_sec_name

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def analysis(
        self,
        targets: Union[list, tuple, np.ndarray], max_step: float,
        dof: Literal['y', 'z'] = 'y', P: Optional[float] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        执行截面模型分析。
            - 若 `SectionHub` 定义截面时未在管理器中明确 `P`，则默认 `P = 1.`。
        
        Args:
            targets (Union[list, tuple, np.ndarray]): 曲率控制路径。
            max_step (float): 分析的最大步长。
            dof (Literal['y', 'z'], optional): 分析方向。 `默认值：` 'y'。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 弯矩-曲率 数据。
                - 弯矩数据。
                - 曲率数据。
        """
        
        # 配置数据库
        opst.post.set_odb_format(odb_format="nc") # 数据库输出格式
        try:
            ODB = opst.post.CreateODB(odb_tag=self.current_sec_name, fiber_ele_tags="ALL")
        except :
            raise ModuleNotFoundError("opstool 数据库创建失败！请先定义截面模型。")
        
        # 获取轴向荷载
        if not P:
            try:
                P = self.MM.get_param(category='section', label=self.current_sec_name, key='P')
            except KeyError:
                P = 1.
                warnings.warn(f"截面 '{self.current_sec_name}' 未定义参数轴向力 'P' 。使用默认值 P = {P}")
        
        # 识别荷载方向
        dof_map = {'y': 5, 'z': 6}
        loads = [0.0] * 6
        loads[dof_map[dof] - 1] = 1.0 # 自动识别方向
        
        if dof not in dof_map.keys():
            raise ValueError(f"Only supported axis = 'y' or 'z'! But got '{dof}'.")
        
        # 时间序列 - 线性
        ops.timeSeries("Linear", self.ts)
        
        # ---------- ----------
        # 轴向荷载
        # ---------- ----------
        # 模式 - 轴向荷载
        ops.loadConst('-time', 0.0)
        ops.pattern("Plain", self.axial_force, self.ts)
        ops.load(self.node_load, *(-P, 0., 0., 0., 0., 0.))
        # 分析 - 轴向荷载
        analysis_axial = ALs.GravityAnalysis(ODB=ODB)
        analysis_axial.analyze(Nsteps=10)
        
        # ---------- ----------
        # 控制荷载
        # ---------- ----------
        # 模式 - 控制荷载
        ops.loadConst('-time', 0.0)
        ops.pattern("Plain", self.ctrl_force, self.ts)
        ops.load(self.node_load, *loads)
        # 分析 - 控制荷载
        analysis_ctrl = ALs.StaticAnalysis(pattern_tag=self.ctrl_force, ODB=ODB)
        moment, phi = analysis_ctrl.analyze(
            ctrl_node=self.node_load, dof=dof_map[dof],
            targets=targets, max_step=max_step,
            )

        # ---------- ----------
        # 保存数据库
        # ---------- ----------
        ODB.save_response()
        
        return moment, phi


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""


if __name__ == "__main__":
    
    # 模型管理器
    from ModelUtilities import MM
    from SectionHub import SectionHub
    
    # 数据路径
    out_path = Path('./OutData')
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 截面模型
    sec_model = SectionModel(manager=MM, num=0, save=out_path)
    sec_model.model()
    x, y = sec_model.analysis(
        targets=[0., 0.2],
        max_step=0.0005,
        dof='z',
        )
    
    # # 查看数据
    # plt.close('all')
    # plt.plot(x, y)
    # plt.xlabel('Target')
    # plt.ylabel('Phi')
    # plt.title(f'{sec_model.current_sec_name}')
    # plt.grid()
    # plt.show()
    


