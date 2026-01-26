#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：Part_Model_MomentCurvature.py
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
from inspect import currentframe as curt_fra
from itertools import batched, product, pairwise
import pickle

import imageio.v2 as imageio

from script.pre import NodeTools
from script.base import random_color, rich_showwarning
from script import UNIT, PVs

import warnings
warnings.showwarning = rich_showwarning

# from Part_MatSec_MomentCurvature import MPhiSection
from script.post import DamageStateTools
# from Script_ModelCreate import ModelCreateTools


"""
# --------------------------------------------------
# ========== < Part_Model_MomentCurvature > ==========
# --------------------------------------------------
"""

class SectionModel:

    def __init__(
        self,
        filepath: Union['str', Path],
        sec_props: PVs.SEC_PROPS,
        ctrl_dir: str = 'Y',
        deg: float = 0.,
        sec_object: Optional[opst.pre.section.FiberSecMesh] = None
        ):
        """
        # SectionModel 实例对象参数说明：
            实例化前需要定义模型空间与截面
            '''
            ops.wipe()
            ops.model('basic', '-ndm', 3, '-ndf', 6)
            '''
            实例化时需明确以下参数
        
        :param filepath: 数据文件根目录，用于导出文件
        :type filepath: Union['str', Path]
        
        :param sec_props: 截面属性对象
        :type sec_props: PVs.SEC_PROPS
        
        :param ctrl_dir: 默认值为 Y。
            弯矩控制方向。弯矩作用于整体坐标系（右手定则）。
        :type ctrl_dir: str
        
        :param deg: 默认值为 0. ，即局部 y 与整体 Y 重合。
            输入正值即截面绕 X 逆时针旋转（右手定则）。
        :type deg: float
        
        :param sec_object: 默认值为 None。
            需要导入opstool.pre.section.FiberSecMesh的网格截面对象，若为 None 则会尝试读取数据根目录下的 pickle 文件（由 FiberSecMesh 储存）。
            由此来决定 plot_strain_state() 方法是否启用网格模式。
        :type sec_object: Optional[opst.pre.section.FiberSecMesh]
        """

        self.rootPath: Union['str', Path] = filepath # 根目录
        self.filepath: Union['str', Path] = os.path.join(self.rootPath, f"{sec_props.Name}_{ctrl_dir}")
        os.makedirs(self.filepath, exist_ok=True) # 创建根目录

        self.sec_props: PVs.SEC_PROPS = sec_props # 截面属性

        self.M: np.ndarray
        self.Phi: np.ndarray
        
        self.sec_resp_data: Union[xr.DataArray, xr.Dataset] # 截面纤维响应数据
        self.sec_damage: pd.DataFrame # 截面损伤数据

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        if sec_object is None:
            try:
                with open(f"{self.rootPath}/{self.sec_props.Name}_section_code.pkl", "rb") as f:
                    self.SEC = pickle.load(f)
                
                self.SEC.view(fill=True, show_legend=True)
                # plt.show()
                plt.savefig(f'{self.filepath}/{self.sec_props.Name}_mash.png', dpi=300, bbox_inches='tight')
                
                self.opst_mode = True
            
            except FileNotFoundError:
                warnings.warn(f"{self.sec_props.Name} 未识别到opstool生成的截面网格", UserWarning)
                self.opst_mode = False
        
        else:
            self.SEC = sec_object
            
            self.opst_mode = True

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 节点单元索引
        self.node_tag_1 = 1 # fix
        self.node_tag_2 = 2 # load
        self.ele_tag = 1
        self.ts_axial_load = 1
        self.ts_ctrl_load = 2
        self.pattern_axial_load = 1
        self.pattern_ctrl_load = 2

        # 方向索引
        ctrl_dof_map = {'y': 5, 'z': 6}
        ctrl_load_map = {
            'y': (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            'z': (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            }
        
        try:
            self.ctrl_dof = ctrl_dof_map[f'{ctrl_dir.lower()}']
            self.ctrl_load_dir = ctrl_load_map[f'{ctrl_dir.lower()}']
        except KeyError:
            raise RuntimeError("Only supported axis = y or z!")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 创建截面 - node
        ops.node(self.node_tag_1, 0., 0., 0.)
        ops.node(self.node_tag_2, 0., 0., 0.)
        
        # 创建截面 - element
        ops.element(
            'zeroLengthSection', self.ele_tag, *(self.node_tag_1, self.node_tag_2), self.sec_props.SectionTag,
            '-orient', *(1, 0, 0), *(0, np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg)))
            ) # 局部x指向整体X，局部y指向整体Y（可调整旋转角）
        
        # 约束
        ops.fix(self.node_tag_1, *(1, 1, 1, 1, 1, 1))
        ops.fix(self.node_tag_2, *(0, 1, 1, 1, 0, 0))
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        opst.pre.section.vis_fiber_sec_real(
            ele_tag=self.ele_tag,
            highlight_matTag=self.sec_props.SteelTag, highlight_color="r"
            )
        # plt.show()
        plt.savefig(f'{self.filepath}/{self.sec_props.Name}_fiber_sec_real.png', dpi=1280, bbox_inches='tight')
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 截面轴压力
        if self.sec_props.P > 0.:
            warnings.warn("Axial force must be less than 0.", UserWarning)
            axial_load = -self.sec_props.P
        else:
            axial_load = self.sec_props.P
        
        ops.timeSeries("Linear", self.ts_axial_load)
        ops.pattern("Plain", self.pattern_axial_load, self.ts_axial_load)
        ops.load(self.node_tag_2, *(axial_load, 0.0, 0.0, 0.0, 0.0, 0.0))
        
        ops.wipeAnalysis()
        ops.system("BandGeneral")
        ops.constraints("Plain")
        ops.numberer("Plain")
        ops.test("NormDispIncr", 1.0e-10, 10, 3)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1 / 10)
        ops.analysis("Static")
        ops.analyze(10)
        ops.loadConst("-time", 0.0)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 控制荷载
        ops.timeSeries("Linear", self.ts_ctrl_load)
        ops.pattern("Plain", self.pattern_ctrl_load, self.ts_ctrl_load)
        ops.load(self.node_tag_2, *self.ctrl_load_dir)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 打印完成信息
        color = random_color()
        rich.print(f'[bold {color}] DONE: [/bold {color}] Created {self.sec_props.Name} !')

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def run_analysis(
        self,
        targets_phi: Union[list, tuple, np.ndarray],
        incr_phi: Optional[float],
        ds_info: bool = True,
        ds_out: bool = False
        ):
        
        """
        # run_analysis 方法的说明
        
        :param targets_phi: 加载的曲率路径。
            可支持：单向路径 或 往复循环路径。
        :type targets_phi: Union[list, tuple, np.ndarray]
        
        :param incr_phi: 曲率路径的步长
        :type incr_phi: Optional[float]
        
        :param ds_info: 默认值为 True。
            执行完分析后会进行初步后处理，是否在终端打印截面损伤信息。
        :type ds_info: bool
        
        :param ds_out: 默认值为 False
            执行完分析后会进行初步后处理，是否输出截面损伤信息至excel。
        :type ds_out: bool
        """

        ops.wipeAnalysis()
        ops.system('BandGeneral')
        ops.constraints('Transformation')
        ops.numberer('RCM')
        # ops.system("BandGeneral")
        # ops.constraints("Plain")
        # ops.numberer("Plain")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        opst.post.set_odb_path(self.filepath) # bug
        self.ODB = opst.post.CreateODB(odb_tag=self.sec_props.SectionTag, fiber_ele_tags="ALL")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        analysis = opst.anlys.SmartAnalyze(
            "Static",
            # testType="EnergyIncr",
            # testTol=1e-12,
            # minStep=1e-12,
            # tryAddTestTimes=True,
            # tryAlterAlgoTypes=True,
            # tryLooseTestTol=False,
            # debugMode=False,
            )
        segs = analysis.static_split(targets=targets_phi, maxStep=incr_phi)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        force_lambda: list = [0.0]
        node_disp: list = [0.0]
        for seg in segs:
            ok = analysis.StaticAnalyze(node=self.node_tag_2, dof=self.ctrl_dof, seg=seg)
            if ok < 0:
                raise RuntimeError("Analysis failed")
            # Fetch response
            self.ODB.fetch_response_step()
            force_lambda.append(ops.getLoadFactor(self.pattern_ctrl_load))
            node_disp.append(ops.nodeDisp(self.node_tag_2, self.ctrl_dof))

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        self.ODB.save_response(zlib=True)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 初步后处理
        self.M = np.array(force_lambda)
        self.Phi = np.array(node_disp)
        self.sec_damage = self._determine_damage(info=ds_info)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        if ds_out:
            self.sec_damage.to_excel(f'{self.filepath}/{self.sec_props.Name}_damage_info.xlsx', index=True)

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def get_Phi_M(self, plot_out: bool = False, to_excel: bool = False):
        
        """### 返回的第一项为 Phi，第二项为 Moment"""
        
        if plot_out:
            plt.close('all')
            plt.figure(figsize=(6, 4))
            plt.title(f'{self.sec_props.Name} P-M-Φ Curve')
            plt.plot(self.Phi, self.M, linewidth=1.0, label='P-M-Φ Curve', zorder=2)
            # plt.xlim(-0.4, 0.4)
            # plt.ylim(-500 * UNIT.kn, 500 * UNIT.kn)
            plt.xlabel('Φ (rad)')
            plt.ylabel('Moment (kN·m)')
            plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
            plt.grid(linestyle='--', linewidth=0.5, zorder=1)
            # plt.show()
            plt.savefig(f'{self.filepath}/{self.sec_props.Name}_M_Phi.png', dpi=1280, bbox_inches='tight')
        
        if to_excel:
            Phi_M = pd.DataFrame({
                'Phi': self.Phi,
                'M': self.M
                })
            Phi_M.to_excel(f'{self.filepath}/{self.sec_props.Name}_Phi_M.xlsx', index=True)
        
        return self.Phi, self.M
    
    def get_M(self, to_excel: bool = False):
        
        if to_excel:
            Phi_M = pd.DataFrame({'M': self.M})
            Phi_M.to_excel(f'{self.filepath}/{self.sec_props.Name}_M.xlsx', index=True)
            
        return self.M
    
    def get_Phi(self, to_excel: bool = False):
        
        if to_excel:
            Phi_Phi = pd.DataFrame({'Phi': self.Phi})
            Phi_Phi.to_excel(f'{self.filepath}/{self.sec_props.Name}_Phi.xlsx', index=True)
            
        return self.Phi
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def _determine_damage(self, info: bool) -> pd.DataFrame:
        """
        # 内部方法 _determine_damage 的 说明
        
        :param info: 是否在终端打印截面损伤信息。
        :type info: bool
        
        :return: 返回截面损伤状态的 DataFram
        :rtype: DataFrame
        
        """
        
        # 读取数据
        self.sec_resp_data = opst.post.get_element_responses(odb_tag=self.sec_props.SectionTag, ele_type="FiberSection")
        # print(f'截面分析数据库：\n{self.sec_resp_data}')
        
        # 创建实例：损伤判断工具
        determine_tools = DamageStateTools(
            resp_data=self.sec_resp_data,
            ele_tag=self.ele_tag, integ=1
            )
        # 判断截面损伤状态
        section_damage = determine_tools.det_sec(
            mat_props=self.sec_props,
            dupe=True,
            MC_DOF=self.M,
            info=info
            )
        
        return section_damage
    
    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def _energy_based_equivalent_bilinear_calculator(
        self,
        point_x: Union[float, np.ndarray], point_y: Union[float, np.ndarray],
        line_x: np.ndarray, line_y: np.ndarray,
        info: bool = True
        ) -> Tuple[np.float64, np.float64]:
        
        """
        # 内部方法 _energy_based_equivalent_bilinear_calculator 的 说明
            基于能量法的双线性等效计算器
        
        :param point_x: 实际屈服点的 X 值（曲率）
        :type point_x: Union[float, np.ndarray]
        
        :param point_y: 实际屈服点的 Y 值（弯矩）
        :type point_y: Union[float, np.ndarray]
        
        :param line_x: 曲线的所有 X 坐标（曲率）
        :type line_x: np.ndarray
        
        :param line_y: 曲线的所有 Y 坐标（弯矩）
        :type line_y: np.ndarray
        
        :param info: 是否打印结果
        :type info: bool
        
        :return: 等效屈服 [曲率，弯矩]
        :rtype: Tuple[float64, float64]
        """
        
        x_ult = line_x[-1]
        Q = trapezoid(line_y, line_x)
        # print(f'# Q: {Q}')

        # 上升段斜率
        k = point_y / point_x
        # 计算等效点
        eq_x = (k * x_ult - np.sqrt((k * x_ult) ** 2 - 2 * k * Q)) / k # 二次求根公式
        eq_y = k * eq_x
        # 检查数据
        if info:
            rich.print(f'# Equivalent_x: {eq_x}')
            rich.print(f'# Equivalent_y: {eq_y}')
        
        return eq_x, eq_y

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def equivalent_bilinear(
        self,
        plot_out: bool = True,
        data_out: bool = True,
        info: bool = True
        ):
        
        """
        # equivalent_bilinear 的 说明
            采用基于能量法的等效计算方法计算等效屈服曲率和弯矩
        
        :param plot_out: 默认值为 True。
            是否输出包含等效双折线的弯矩曲率图像。
            若为 False 则打印图像至窗口。
        :type plot_out: bool
        
        :param info: 是否打印结果
        :type info: bool
        
        :param data_out: 默认值为 True。
            是否输出包含等效值的损伤关键点 Excel。
        :type data_out: bool
        """
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 曲线极限点对应的 step
        if self.sec_damage['DS'].isin(['DS5']).any():
            # 有完全损伤，则以完全损伤的点为结尾
            end_step = self.sec_damage.index.max()
        else:
            # 没到完全损伤，则保留 当前数据库 全程
            end_step = len(self.M)
            warnings.warn(f"{self.sec_props.Name} 截面没全坏哦~", UserWarning)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 弯矩-曲率 线数据切片
        x_line = self.Phi[: end_step + 1]
        y_line = self.M[: end_step + 1]
        
        # 弯矩-曲率 线上损伤关键点
        key_point_phi = self.Phi[self.sec_damage.index.values]
        key_point_moment = self.M[self.sec_damage.index.values]
        
        # 计算等效屈服点
        eq_point_phi, eq_point_moment = self._energy_based_equivalent_bilinear_calculator(
            point_x=key_point_phi[0], point_y=key_point_moment[0],
            line_x=x_line, line_y=y_line,
            info=info
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 损伤关键点数据 -> DataFrame
        key_data = pd.DataFrame({
            'type': self.sec_damage['DS'],
            'phi': key_point_phi,
            'M': key_point_moment,
            })
        # 添加等效屈服数据
        key_data.loc[0] = ['Eq_y', eq_point_phi, eq_point_moment]
        key_data.loc[-1] = ['Eq_u', key_point_phi[-1], eq_point_moment]
        # print(f'所有关键点数据：{key_data}')
        
        if data_out:
            key_data.to_excel(f'{self.filepath}/{self.sec_props.Name}_DS_key_point.xlsx', index=True)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        x_line_eq = [0., eq_point_phi, key_point_phi[-1]]
        y_line_eq = [0., eq_point_moment, eq_point_moment]
        
        # 绘图
        plt.close('all')
        plt.figure(figsize=(6, 4))
        plt.title(f"{self.sec_props.Name} - Bilinear Equivalent Curve")
        
        plt.plot(x_line, y_line, color='#03A9F4', label='Initial M-Φ', zorder=2) # 原曲线
        plt.plot(x_line_eq, y_line_eq, color='#E91E63', label='Equivalent M-Φ', marker='o', zorder=3) # 等效双折线
        plt.scatter(key_point_phi, key_point_moment, color='#FF9800', label='DS key point', marker='o', zorder=4) # 所有损伤关键点
        plt.scatter(key_point_phi[0], key_point_moment[0], color='#FF9800', marker='o', zorder=4) # 实际屈服关键点

        # 实际屈服点 箭头标注
        plt.annotate(
            text='Initial yield',  # 标签文本
            color='k',  # 标签颜色

            xy=(key_point_phi[0], key_point_moment[0]),  # 要标记的数据坐标
            xycoords='data',  # 使用被注释对象（参数为xy）的坐标系统

            xytext=(50, -50),  # 标签偏移量
            textcoords='offset points',  # 相对被标注点的偏移量，单位为点
            # fontsize=18,  # 字体大小

            arrowprops=dict(
                arrowstyle='->',  # 箭头样式
                # linewidth=2,  # 箭线宽度
                # mutation_scale=20,  # 箭头大小
                color='k',  # 箭头颜色
                connectionstyle='arc3,rad=-0.2',  # 带弧度的箭头
            ),
            zorder=4
        )
        # 等效屈服点 箭头标注
        plt.annotate(
            text='Equivalent yield',  # 标签文本
            color='k',  # 标签颜色

            xy=(eq_point_phi, eq_point_moment),  # 要标记的数据坐标
            xycoords='data',  # 使用被注释对象（参数为xy）的坐标系统

            xytext=(50, -50),  # 标签偏移量
            textcoords='offset points',  # 相对被标注点的偏移量，单位为点
            # fontsize=18,  # 字体大小

            arrowprops=dict(
                arrowstyle='->',  # 箭头样式
                # linewidth=2,  # 箭线宽度
                # mutation_scale=20,  # 箭头大小
                color='k',  # 箭头颜色
                connectionstyle='arc3,rad=-0.2',  # 带弧度的箭头
            ),
            zorder=4
        )

        plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
        plt.xlabel('Φ (rad)')
        plt.ylabel('Moment (kN·m)')
        plt.xlim(0, max(x_line) * 1.1)
        plt.ylim(0, eq_point_moment * 1.2)
        plt.grid(linestyle='--', linewidth=0.5, zorder=1)
        
        if plot_out:
            plt.savefig(f'{self.filepath}/{self.sec_props.Name}_M_phi_bilinear.png', dpi=1280, bbox_inches='tight')
        else:
            plt.show()

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def _plot_fiber(
        self,
        point_y, point_z,
        resp_data: xr.DataArray,
        resp_mat: xr.DataArray,
        cover_tag: int, core_tag: int, bar_tag: int,
        cover_eps_thr: float, core_eps_thr: float, bar_eps_thr: float,
        map_style: str = 'coolwarm',
        ):
        
        """
        # 内部方法 _plot_fiber 的 说明
            仅支持钢筋混凝土截面（2混凝土 + 1金属材料）纤维截面的应变响应状态
        
        :param point_y: 纤维截面 纤维 的 y 坐标
        :param point_z: 纤维截面 纤维 的 z 坐标
        
        :param resp_data: 读取 ODB 内的 应变数据
        :type resp_data: xr.DataArray
        
        :param resp_mat: 读取 ODB 内的 材料标签
        :type resp_mat: xr.DataArray
        
        :param cover_tag: 保护层混凝土材料标识
        :type cover_tag: int
        
        :param core_tag: 核心混凝土材料标识
        :type core_tag: int
        
        :param bar_tag: 钢筋材料标识
        :type bar_tag: int
        
        :param cover_eps_thr: 保护层混凝土破坏应变阈值（绝对值）
        :type cover_eps_thr: float
        
        :param core_eps_thr: 核心混凝土破坏应变阈值（绝对值）
        :type core_eps_thr: float
        
        :param bar_eps_thr: 钢筋破坏应变阈值（绝对值）
        :type bar_eps_thr: float
        
        :param map_style: 默认值为：coolwarm。绘图主题。
        :type map_style: str
        
        :return: ax, char
        :rtype: ax, char
        """
        
        fig, ax = plt.subplots(figsize=(6, 4))
        # 应力为零的索引
        cover_cond = (resp_mat == cover_tag) & (-cover_eps_thr <= resp_data) & (resp_data <= 0.)
        core_cond = (resp_mat == core_tag) & (-core_eps_thr <= resp_data) & (resp_data <= 0.)
        bar_cond = (resp_mat == bar_tag) & (-bar_eps_thr <= resp_data) & (resp_data <= bar_eps_thr)
        break_cond = ~cover_cond & ~core_cond & ~bar_cond
        
        
        s_cover = ax.scatter(
            point_y.where(cover_cond, drop=True), point_z.where(cover_cond, drop=True), # drop=True 去掉不满足的数据点
            c=resp_data.where(cover_cond, drop=True), s=5,
            cmap=map_style, zorder=2,
        )
        s_core = ax.scatter(
            point_y.where(core_cond, drop=True), point_z.where(core_cond, drop=True), # drop=True 去掉不满足的数据点
            c=resp_data.where(core_cond, drop=True), s=5,
            cmap=map_style, zorder=2,
        )
        s_bar = ax.scatter(
            point_y.where(bar_cond, drop=True), point_z.where(bar_cond, drop=True), # drop=True 去掉不满足的数据点
            c=resp_data.where(bar_cond, drop=True), s=10,
            cmap=map_style, zorder=2,
        )
        s_break = ax.scatter(
            point_y.where(break_cond, drop=True), point_z.where(break_cond, drop=True), # drop=True 去掉不满足的数据点
            c='#BDBDBD', s=5,
            label="Broken", zorder=1,
        )
        # colorbar 
        cbar = fig.colorbar(s_core, ax=ax, pad=0.02)
        
        # 数据单位 1:1
        ax.set_aspect('equal', adjustable='datalim') # 保持 figsize
        # 单独 legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower right',
            bbox_to_anchor=(0.92, 0.01),  # 调整这个值可以移动 legend
            framealpha=0.9
            )
        
        return ax, cbar

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def plot_strain_state(
        self,
        map_style: str = 'coolwarm',
        step: Optional[Union[str, int]] = 'DS5',
        plot_out: bool = True,
        # animation: bool = False
        ):
        
        """
        # plot_strain_state 的 说明
            绘制纤维截面的应变状态，根据当前实例实例化时是否存在 sec_object 参数识别是否启用网格模式
            #### 非网格模式时仅支持钢筋混凝土截面的纤维应变状态
        
        :param map_style: 默认值为 coolwarm。
            纤维截面应变状态绘图主题
        :type map_style: str
        
        :param step: 默认值为 DS5.
            绘制纤维截面的哪一状态。支持 int 和 str。
        :type step: Optional[Union[str, int]]
        
        :param plot_out: 默认值为 True.
            True 为输出到本地，False 为输出到窗口。
        :type plot_out: bool
        """
        
        step_map = {
            'ds2': 'DS2',
            'ds3': 'DS3',
            'ds4': 'DS4',
            'ds5': 'DS5',
            'yield': 'DS2',
            'broken': 'DS5',
            'break': 'DS5',
        }
        
        if isinstance(step, str):
            try:
                get_value = step_map[step.lower()]
            except KeyError:
                warnings.warn(f"Expected one of {list(step_map.keys())} or an integer string (int).", UserWarning)
                warnings.warn(f"Section break strain state will be shown.", UserWarning)
                step_index = int(self.sec_damage[self.sec_damage['DS'] == 'DS5'].index.item())
            else:
                step_index = int(self.sec_damage[self.sec_damage['DS'] == get_value].index.item())

        elif isinstance(step, int):
            step_index = step

        else:
            raise ValueError(
                f"Invalid step string: {step}. Expected one of {list(step_map.keys())} or an integer string."
                )
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 材料标签与值索引
        cover_tag = self.sec_props.CoverTag
        core_tag = self.sec_props.CoreTag
        bar_tag = self.sec_props.SteelTag
        
        cover_eps_thr = abs(self.sec_props.CoverProps.eps_ultra)
        core_eps_thr = abs(self.sec_props.CoreProps.eps_ultra)
        bar_eps_thr = abs(self.sec_props.SteelProps.eps_ultra)
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        mat = self.sec_resp_data["matTags"].sel(eleTags=self.ele_tag, secPoints=1)
        stress = self.sec_resp_data["Stresses"].sel(eleTags=self.ele_tag, secPoints=1).isel(time=step_index)
        strain = self.sec_resp_data["Strains"].sel(eleTags=self.ele_tag, secPoints=1).isel(time=step_index)
        ys = self.sec_resp_data["ys"].sel(eleTags=self.ele_tag, secPoints=1)
        zs = self.sec_resp_data["zs"].sel(eleTags=self.ele_tag, secPoints=1)
        points = np.stack((ys.values, zs.values), axis=-1) # 截面纤维点的坐标
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        cond = (mat == cover_tag) | (mat == core_tag) | (mat == bar_tag) # concrete fibers only
        # overall min strain across all time steps
        vmin = self.sec_resp_data["Strains"].sel(eleTags=self.ele_tag, secPoints=1, fiberPoints=cond).min().values
        # overall max strain across all time steps
        vmax = self.sec_resp_data["Strains"].sel(eleTags=self.ele_tag, secPoints=1, fiberPoints=cond).max().values
        # overall strain across all time steps
        abmax = max(abs(vmin), abs(vmax))
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        if self.opst_mode:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax, cbar = self.SEC.plot_response(
                points=points,  # (num_fiber_points, 2)
                response=strain.values,  # (num_fiber_points,)
                mat_tag=(cover_tag, core_tag, bar_tag),
                thresholds={
                    cover_tag: (-cover_eps_thr, 0.),
                    core_tag: (-core_eps_thr, 0.),
                    bar_tag: (-bar_eps_thr, bar_eps_thr)
                    },
                cmap=map_style,
                ax=ax,
            )
            
        else:
            '''不够健壮，只支持钢筋混凝土截面'''
            plt.close('all')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax, cbar = self._plot_fiber(
                point_y=ys, point_z=zs,
                resp_data=strain, resp_mat=mat,
                cover_tag=cover_tag, core_tag=core_tag, bar_tag=bar_tag,
                cover_eps_thr=cover_eps_thr,core_eps_thr=core_eps_thr,bar_eps_thr=bar_eps_thr,
                map_style='coolwarm'
            )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        fig.tight_layout(rect=(0, 0, 1, 1))
        cbar.set_label("Strain", fontsize=12)
        ax.set_title(f"Strain Distribution\n(Element {self.ele_tag}, Section: {self.sec_props.Name})", fontsize=14)
        ax.set_xlabel("Y", fontsize=12)
        ax.set_ylabel("Z", fontsize=12)
        # cbar.mappable.set_clim(float(vmin), float(vmax))
        cbar.mappable.set_clim(float(-abmax), float(abmax))
        
        if plot_out:
            plt.savefig(f'{self.filepath}/{self.sec_props.Name}_Strain_Distribution.png', dpi=1280, bbox_inches='tight')
        else:
            plt.show()
        
        
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        animation = False
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        if animation:
            
            warnings.warn("动画起码5分钟，等着吧", UserWarning)

            "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
            with imageio.get_writer(f"{self.filepath}/fiber-section-strain.gif", mode="I", fps=6) as writer:
                # for t in range(len(self.sec_resp_data["time"])):
                for t in range(self.sec_damage.index.max()):

                    strain_ani = self.sec_resp_data["Strains"].sel(eleTags=1, secPoints=1).isel(time=t)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax, cbar = self.SEC.plot_response(
                        points=points,  # (num_fiber_points, 2)
                        response=strain_ani.values,  # (num_fiber_points,)
                        mat_tag=(cover_tag,core_tag,bar_tag),
                        thresholds={
                            cover_tag: (-cover_eps_thr, 0.),
                            core_tag: (-core_eps_thr, 0.),
                            bar_tag: (-bar_eps_thr, bar_eps_thr)
                            },
                        cmap=map_style,
                        ax=ax,
                    )
                    fig.tight_layout(rect=(0, 0, 1, 1))
                    cbar.set_label("Strain", fontsize=12)
                    ax.set_title(f"Strain Distribution\n(Element {self.ele_tag}, Section: {self.sec_props.Name})", fontsize=14)
                    ax.set_xlabel("Y", fontsize=12)
                    ax.set_ylabel("Z", fontsize=12)
                    # cbar.mappable.set_clim(float(vmin), float(vmax))
                    cbar.mappable.set_clim(float(-abmax), float(abmax))

                    # Convert Matplotlib figure to image and append to gif
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    image = image.reshape((*fig.canvas.get_width_height()[::-1], 4))
                    writer.append_data(image)

                    plt.close(fig)
                
                    print(f'step: {t}')
                    # warnings.warn(f"动画起码5分钟，等着吧, step: {t}", UserWarning)


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    
    from SectionHub import SectionHub
    
    secTag_props = {
        'section_tag': 1,  # 截面
        'cover_tag': 1,  # 材料-保护层
        'core_tag': 2,  # 材料-核心
        'bar_tag': 3,  # 材料-钢筋
        'bar_max_tag': 4,  # 材料-钢筋最大应变限制
        'info': False
    }
    
    CASE_LIST = [
        {'sec_func': SectionHub.Section_Example_01, 'dir': "y"},
        {'sec_func': SectionHub.Section_Example_02, 'dir': "y"},
        {'sec_func': SectionHub.Section_Example_02, 'dir': "z"},
        {'sec_func': SectionHub.Section_Example_03, 'dir': "y"},
        {'sec_func': SectionHub.Section_Example_04, 'dir': "y"},
        ]
    
    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    root_path = './Data'
    os.makedirs(root_path, exist_ok=True) # 创建根目录
    
    case = CASE_LIST[4]
    
    # 模型空间
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    # 截面
    section_props = case['sec_func'](root_path, **secTag_props)
    # 模型
    model_test = SectionModel(
        filepath=root_path,
        sec_props=section_props,
        ctrl_dir=case['dir']
        )
    
    model_test.run_analysis(targets_phi=np.array(0.1/UNIT.m), incr_phi=1.e-4)
    model_test.get_Phi_M(plot_out=True, to_excel=True)
    model_test.equivalent_bilinear(plot_out=True, data_out=True)
    model_test.plot_strain_state(map_style='coolwarm', step='yield', plot_out=True)


