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


from cProfile import label
import os
import sys
import time
import warnings
from typing import Literal, TypeAlias, Union, Callable, Any, Optional
import matplotlib.pyplot as plt
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

from script.pre import NodeTools
from script.base import random_color
from script import UNIT, PVs

# from Part_MatSec_MomentCurvature import MPhiSection
from script.post import DamageStateTools
# from Script_ModelCreate import ModelCreateTools


"""
# --------------------------------------------------
# ========== < Part_Model_MomentCurvature > ==========
# --------------------------------------------------
"""

class SectionModel:

    def __init__(self, rootPath: str = './Data'):

        """实例化：即在当前根目录创建 数据文件夹"""

        self.rootPath: str = rootPath # 根目录
        os.makedirs(rootPath, exist_ok=True) # 创建根目录

        self.secPath: str # 子目录

        # self.name = None # 截面名
        self.direction: str # 截面弯矩方向
        self.sec_props: PVs.SEC_PROPS # 截面属性

        self.fiber_data: xr.DataArray # 截面纤维响应数据
        self.sec_damage: pd.DataFrame # 截面损伤数据

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def create_section(
            self,
            func: Callable[..., PVs.SEC_PROPS],
            axis: str,
            **kwargs
    ) -> PVs.SEC_PROPS:

        # 创建子文件夹
        self.secPath = os.path.join(self.rootPath, f"{func.__name__}_{axis}")
        os.makedirs(self.secPath, exist_ok=True)

        # 模型空间
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        # 创建截面
        self.sec_props = func(self.secPath, **kwargs)

        # 打印完成信息
        color = random_color()
        rich.print(f'[bold {color}] DONE: [/bold {color}] Create {self.sec_props.Name} !')

        return self.sec_props

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def determine_damage(self, MC, bilinear: bool = False, info: bool = True):
        """
        MC: opstool 截面分析返回的分析数据
        bilinear: 是否计算双线性等效曲线
        """

        # M_phi 完整曲线
        MC.plot_M_phi()
        plt.savefig(f'{self.secPath}/{self.sec_props.Name}_M_phi.png', dpi=300, bbox_inches='tight')

        # 所有纤维 < 应变-应力 > 曲线
        MC.plot_fiber_responses()
        plt.savefig(f'{self.secPath}/{self.sec_props.Name}_fiber_responses.png', dpi=300, bbox_inches='tight')

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取数据
        phi, M = MC.get_M_phi() # 弯矩，曲率
        self.fiber_data = MC.get_fiber_data() # 纤维响应数据
        
        # 导出 弯矩-曲率 数据
        M_Phi = pd.DataFrame({
            'M': M,
            'phi': phi
            })
        M_Phi.to_excel(f'{self.secPath}/{self.sec_props.Name}_M_phi.xlsx', index=True)
        
        # 创建实例：损伤判断工具
        determine_tools = DamageStateTools(resp_data=self.fiber_data)
        # 返回截面损伤列表
        self.sec_damage = determine_tools.determine_sec(mat_props=self.sec_props, dupe=True, MC_DOF=np.array(M), info=info)
        self.sec_damage.to_excel(f'{self.secPath}/{self.sec_props.Name}_damage_state.xlsx', index=True) # 导出数据

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        if bilinear:

            # 曲线极限点对应的 step
            if self.sec_damage['DS'].isin(['DS5']).any():
                # 有完全损伤，则以完全损伤的点为结尾
                end_step = self.sec_damage.index.max()
            else:
                # 没到完全损伤，则保留 当前数据库 全程
                end_step = len(M)


            # 弯矩-曲率 线上损伤关键点
            phi_point_ls = phi[self.sec_damage.index.values]
            moment_point_ls = M[self.sec_damage.index.values]

            # 汇总关键点数据
            key_point_data = pd.DataFrame({
                'type': self.sec_damage['DS'],
                'phi': phi_point_ls,
                'M': moment_point_ls,
                })

            # 弯矩-曲率 线数据切片
            x_line = phi[: end_step + 1]
            y_line = M[: end_step + 1]

            # 判断截面延性
            if self.sec_damage['DS'].isin(['DS2']).any():
                # 等效双折线
                eq_bi_linear = DamageStateTools.energy_based_Eq(
                    phi_point_ls[0],
                    moment_point_ls[0],
                    x_line,
                    y_line,
                    )
            else:
                # warnings.warn("纵筋未屈服，请优化截面配筋！", UserWarning)
                eq_bi_linear = np.zeros((3, 2), dtype=float) # 延性不足等效曲线为零
            
            # 新增关键点数据
            key_point_data.loc[0] = ['Eq', eq_bi_linear[1,0], eq_bi_linear[1,1]]
            
            # 导出关键点数据
            key_point_data.to_excel(f'{self.secPath}/{self.sec_props.Name}_key_point_data.xlsx', index=True)

            "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
            # 绘图
            plt.close('all')
            plt.figure(figsize=(6, 4))
            plt.title(f"{self.sec_props.Name} - Bilinear Equivalent Curve")
            
            plt.plot(x_line, y_line, color='#03A9F4', label='Initial M-Φ', zorder=2) # 原曲线
            plt.plot(eq_bi_linear[:,0], eq_bi_linear[:,1], color='#E91E63', label='Equivalent M-Φ', marker='o', zorder=3) # 等效双折线
            # plt.scatter(phi_point_ls, moment_point_ls, color='#FF9800', label='DS key point', marker='o', zorder=4) # 所有损伤关键点
            plt.scatter(phi_point_ls[0], moment_point_ls[0], color='#FF9800', marker='o', zorder=4) # 实际屈服关键点
            
            # 实际屈服点 箭头标注
            plt.annotate(
                text='Initial yield',  # 标签文本
                color='k',  # 标签颜色

                xy=(phi_point_ls[0], moment_point_ls[0]),  # 要标记的数据坐标
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

                xy=(eq_bi_linear[1, 0], eq_bi_linear[1, 1]),  # 要标记的数据坐标
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
            plt.xlim(0, eq_bi_linear[-1, 0] * 1.1)
            plt.ylim(0, eq_bi_linear[-1, 1] * 1.2)
            plt.grid(linestyle='--', linewidth=0.5, zorder=1)

            # plt.show()
            plt.savefig(f'{self.secPath}/{self.sec_props.Name}_M_phi_bilinear.png', dpi=1280, bbox_inches='tight')

    "===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="
    def section_rainbow(self, step: Optional[Union[str, int]], style = None):
        """
        step: 要绘制哪一步的云图
        style: 哪一种云图 应变/strain 应力/stress 材料/mat
        """

        step_map = {
            'ds2': 'DS2',
            'ds3': 'DS3',
            'ds4': 'DS4',
            'ds5': 'DS5',
            'yield': 'DS2',
            'broken': 'DS5',
            'break': 'DS5'
        }

        if isinstance(step, str):
            if step.lower() in step_map:
                times = self.sec_damage[
                    self.sec_damage['DS'] == step_map.get(step.lower())
                    ].index.values
            else:
                raise ValueError(
                    f"Invalid step string: {step}. Expected one of {list(step_map.keys())} or an integer string."
                    )

        elif isinstance(step, int):
            times = step
        else:
            raise ValueError(
                f"Invalid step string: {step}. Expected one of {list(step_map.keys())} or an integer string."
                )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 切片数据
        fiber_data_last = self.fiber_data.isel(Steps=times)

        # 筛选 坐标数据
        ys = fiber_data_last.sel(Properties="yloc")
        zs = fiber_data_last.sel(Properties="zloc")

        # 云图类型
        matTag = fiber_data_last.sel(Properties="mat")
        Strains = fiber_data_last.sel(Properties="strain")
        Stresses = fiber_data_last.sel(Properties="stress")

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 初始化
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 4))

        # 应力为零的索引
        break_mask = (Stresses == 0.)

        # 散点图 应力不为零的点
        s = ax.scatter(
            ys.where(~break_mask, drop=True), zs.where(~break_mask, drop=True), # drop=True 去掉不满足的数据点
            c=Stresses.where(~break_mask, drop=True),
            s=5,
            cmap="rainbow",
            zorder=2,
        )
        # 散点图 应力为零的点
        ax.scatter(
            ys.where(break_mask, drop=True), zs.where(break_mask, drop=True),
            c='#BDBDBD',
            s=5,
            label="Broken",
            zorder=1,
        )
        # 坐标轴标签
        ax.set_xlabel('local_y')
        ax.set_ylabel('local_z')

        # 数据单位 1:1
        # ax.set_aspect('equal', adjustable='box')
        ax.set_aspect('equal', adjustable='datalim') # 保持 figsize

        # colorbar 
        cbar = fig.colorbar(s, ax=ax, pad=0.02)

        # 单独 legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower right',
            bbox_to_anchor=(0.92, 0.01),  # 调整这个值可以移动 legend
            framealpha=0.9)

        # fig.show()
        fig.savefig(f'{self.secPath}/{self.sec_props.Name}_stress_rainbow.png', dpi=300, bbox_inches='tight')

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    pass
