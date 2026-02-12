#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：ModelUtilities.py
@Date    ：2026/01/26 19:40:43
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.colorbar import Colorbar
import gif
from typing import Union, Optional, Literal, List, Tuple, Dict

from pathlib import Path

import opstool as opst
import ops_utilities as opsu
from ops_utilities.post import AdjustPlot

"""
# --------------------------------------------------
# ========== < ModelUtilities > ==========
# --------------------------------------------------
"""

# 全局单位系统
UNIT = opst.pre.UnitSystem(length="m", force="kn", time="sec")

# 全局模型管理器
MM = opsu.pre.ModelManager(include_start=True)

"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="


# 绘图方法
class PlotyHub:

    # 全局长度单位转换为：英寸
    UNIT = opst.pre.UnitSystem(length="inch")

    # ----- ----- ----- ----- -----
    # / 内部方法 / < 坐标轴 > 调整方法
    # ----- ----- ----- ----- -----
    @classmethod
    def _adjust_ax(
        cls,
        ax: Axes,
        title: str = "",
        title_size: int = 18,
        title_x: float = 0.5,
        title_y: float = -0.25,
        lower_label: str = "",
        left_label: str = "",
        label_size: int = 20,
        label_pad: float = 7.0,
        lower_ticks: bool = True,
        left_ticks: bool = True,
        ticks_size: int = 18,
        ticks_pad: float = 12.0,
    ) -> Axes:
        """
        调整坐标轴内的标题、背景、轴线刻度、标签、数字等。

        Args:
            ax (Axes): 要调整的 Axes 对象。
            title (str): 坐标轴标题。`默认值：`""。
            title_size (int): 坐标轴标题字体大小。`默认值：`18。
            title_x (float): 坐标轴标题水平位置。`默认值：`0.5。
            title_y (float): 坐标轴标题垂直位置。`默认值：`-0.25。

            lower_label (str): 下 x轴 标签。`默认值：`""。
            left_label (str): 左 y轴 标签。`默认值：`""。
            label_size (int): 坐标轴标签字体大小。`默认值：`20。
            label_pad (float): 坐标轴标签与轴线的距离。`默认值：`7.0。

            lower_ticks (bool): 是否显示下 x轴 刻度。`默认值：`True。
            left_ticks (bool): 是否显示左 y轴 刻度。`默认值：`True。
            ticks_size (int): 坐标轴刻度字体大小。`默认值：`18。
            ticks_pad (float): 坐标轴刻度与轴线的距离。`默认值：`12.0。

        Returns:
            Axes: 返回调整后的 Axes 对象。

        """

        # 坐标系 标题
        if not title:
            title = ax.get_title()
        ax.set_title(title, fontsize=title_size, color="black", x=title_x, y=title_y)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 背景调整
        ax = AdjustPlot.adjust_bg(ax)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 坐标轴调整
        AdjustPlot.adjust_x(
            ax, lower_label, label_size, label_pad, lower_ticks, ticks_size, ticks_pad
        )
        AdjustPlot.adjust_y(
            ax, left_label, label_size, label_pad, left_ticks, ticks_size, ticks_pad
        )

        # "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # # 共享 下x轴
        # y_right = ax.twinx()
        # cls._adjust_y(y_right, right_label, label_size, label_pad, right_ticks, ticks_size, ticks_pad)
        # # 共享 左y轴
        # x_upper = ax.twiny()
        # cls._adjust_x(x_upper, upper_label, label_size, label_pad, upper_ticks, ticks_size, ticks_pad)
        """返回的共享轴需要有数据，否则 ticks 范围只有 [0, 1]"""

        return ax

    # ----- ----- ----- ----- -----
    # < 单坐标轴 > 绘图参数调整方法
    # ----- ----- ----- ----- -----
    @classmethod
    def adjust_single(
        cls,
        fig: Figure,
        ax: Axes,
        leg: Optional[Legend] = None,
        cbar: Optional[Colorbar] = None,
    ) -> Figure:
        """
        单坐标轴绘图参数调整方法。

        Args:
            fig (Figure): 输入的 matplotlib.figure.Figure 对象。
            ax (Axes): 输入的 matplotlib.axes.Axes 对象。
            leg (Legend, optional): 输入的 `leg = ax.legend()` 对象。

        Returns:
            Figure: 返回调整后的 matplotlib.figure.Figure 对象。

        """

        # 画布参数
        fig_params = {
            # 创建画布（将厘米转换为英寸：1厘米=0.3937英寸）
            "fig_width": 27 * cls.UNIT.cm,  # 单位厘米
            "fig_height": 18 * cls.UNIT.cm,
            # 子图间距
            "w_space": 5.4 * cls.UNIT.cm,
            "h_space": 6.3 * cls.UNIT.cm,
            # 边缘距离
            "left": 5.0 * cls.UNIT.cm,
            "right": 1.5 * cls.UNIT.cm,
            "top": 1.5 * cls.UNIT.cm,
            "bottom": 2.5 * cls.UNIT.cm,
        }
        # 画布调整
        fig_adj = AdjustPlot.adjust_fig(fig, **fig_params)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 坐标轴 参数
        ax_params = {
            "title_size": 20,
            "title_x": 0.5,
            "title_y": -0.25,
            "label_size": 22,
            "label_pad": 7.0,
            "ticks_size": 18,
            "ticks_pad": 12.0,
        }
        # 坐标轴参数调整
        cls._adjust_ax(ax, **ax_params)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 图例参数
        leg_params = {
            "font_size": 22,
            "pad": 1,
            "rounding_size": 0.8,
        }
        # 图例
        if leg is not None:
            AdjustPlot.adjust_leg(leg, **leg_params)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 颜色条参数
        cbar_params = {
            "label": "",
            "label_size": 20,
            "label_pad": 7.0,
            "ticks": True,
            "ticks_size": 18,
            "ticks_pad": 12.0,
        }
        # 颜色条
        if cbar is not None:
            AdjustPlot.adjust_cbar_y(cbar, **cbar_params)

        return fig_adj

    # ----- ----- ----- ----- -----
    # < 双坐标轴 > 绘图参数调整方法
    # ----- ----- ----- ----- -----
    @classmethod
    def adjust_dual(
        cls,
        fig: Figure,
        ax1: Axes,
        ax2: Axes,
        leg1: Optional[Legend] = None,
        leg2: Optional[Legend] = None,
    ) -> Figure:
        """
        双坐标轴绘图参数调整方法。

        Args:
            fig (Figure): 输入的 matplotlib.figure.Figure 对象。
            ax1 (Axes): 输入的 matplotlib.axes.Axes 对象。
            ax2 (Axes): 输入的 matplotlib.axes.Axes 对象。
            leg1 (Legend, optional): 输入的 `leg1 = ax1.legend()` 对象。
            leg2 (Legend, optional): 输入的 `leg2 = ax2.legend()` 对象。

        Returns:
            Figure: 返回调整后的 matplotlib.figure.Figure 对象。

        """
        fig_params = {
            # 创建画布（将厘米转换为英寸：1厘米=0.3937英寸）
            "fig_width": 2 * 27 * cls.UNIT.cm,  # 单位厘米
            "fig_height": 20 * cls.UNIT.cm,
            # 子图间距
            "w_space": 16 * cls.UNIT.cm,
            "h_space": 6.3 * cls.UNIT.cm,
            # 边缘距离
            "left": 6.5 * cls.UNIT.cm,
            "right": 1.5 * cls.UNIT.cm,
            "top": 2.8 * cls.UNIT.cm,
            "bottom": 5.5 * cls.UNIT.cm,
        }

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 画布参数调整
        fig_adj = AdjustPlot.adjust_fig(fig, **fig_params)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 坐标轴 参数
        ax_params = {
            "title_size": 38,
            "title_x": 0.5,
            "title_y": -0.4,
            "label_size": 36,
            "label_pad": 9.0,
            "ticks_size": 25,
            "ticks_pad": 12.0,
        }
        # 坐标轴1 调整
        cls._adjust_ax(ax1, **ax_params)
        # 坐标轴2 调整
        cls._adjust_ax(ax2, **ax_params)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 图例参数
        leg_params = {
            "font_size": 30,
            "pad": 1.8,
            "rounding_size": 0.8,
        }
        # 图例调整
        if leg1 is not None:
            AdjustPlot.adjust_leg(leg1, **leg_params)
        if leg2 is not None:
            AdjustPlot.adjust_leg(leg2, **leg_params)

        return fig_adj


"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="


# 数据后处理方法
class PostProcess:

    def __init__(
        self,
        manager: opsu.pre.ModelManager,
        name: str,
        data_path: Union[Path, str, Literal[""]] = "",
        print_info: bool = True,
    ) -> None:
        """
        截面模型后处理类
        Args:
            manager (opsu.pre.ModelManager): 模型管理器实例。
            name (str): 数据库中索引名称。
            data_path (Union[Path, str, Literal['']], optional): 数据保存路径。`默认值`为当前路径。
            print_info (bool, optional): 是否打印信息。默认值为 True。

        Returns:
            None: 无返回值。
        """

        # 数据路径
        self.data_path = Path("./")
        if data_path:
            self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        # 模型管理器
        self.MM = manager
        self.name = name
        self.ele_tag = manager.get_tag(category="element", label=name)[0]  # 单元号
        self.strain_stages = manager.get_param(
            category="section", label=name, key="strain_stages"
        )  # 应变阶段

        # 应变状态实例
        self.SMS = opsu.post.SecMatStates(
            odb_tag=name, ele_tag=self.ele_tag, integ=1, print_info=print_info
        )
        # 损伤状态
        sec_damage = self.SMS.get_combined_steps_mat(
            mat_config=self.strain_stages, data_type="Strains", warn=print_info
        )

        # self.ds_stages = list(filter(None, sec_damage))  # 过滤 None 和 0 值
        self.ds_stages: List[int] = list(
            x for x in sec_damage if x is not None and x != 1
        )  # 过滤 None 和 0 值

    def plot_fiber_resp(self):
        """
        纤维响应图
            - 保存所有纤维应力应变响应
            - 材料标签需要在 `SectionHub` 中定义截面时，在 `ModelManager` 设置 `label`。

        Args:
            None: 无输入值。

        Returns:
            None: 无返回值。
        """

        plt.close("all")
        # 创建新的 Figure 对象
        fig = plt.figure(dpi=100)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        try:
            points = "all"

            fig.suptitle(
                f"{self.name} - fiber response",
                fontsize=22,
                color="black",
                x=0.5,
                y=0.98,
                alpha=1.0,
            )
            # 获取材料编号
            mat_tag = self.MM.get_tag(
                category="uniaxialMaterial", label=f"{self.name}_steel"
            )[
                0
            ]  # 用户定义的 label
            # 获取应力应变数据
            strains = self.SMS.get_data_mat(
                mat_tag=mat_tag, data_type="Strains", points=points
            ).iloc[: self.ds_stages[-1] + 1, :]
            stresses = self.SMS.get_data_mat(
                mat_tag=mat_tag, data_type="Stresses", points=points
            ).iloc[: self.ds_stages[-1] + 1, :]

            # 坐标轴
            ax = plt.subplot2grid(
                (1, 1), (0, 0), rowspan=1, colspan=1, facecolor="none"
            )
            # 数据
            ax.plot(strains, stresses, linewidth=1.5, label="Rebar")
            ax.set_xlabel("Strain")
            ax.set_ylabel("Stress (kPa)")
            # 调整尺寸
            adj_fig = PlotyHub.adjust_single(fig, ax)
            # 保存图片
            adj_fig.savefig(self.data_path / f"{self.name}_fiber_resp.png", dpi=320)

        except:
            points = "all"

            fig.suptitle(
                f"{self.name} - fiber response",
                fontsize=38,
                color="black",
                x=0.5,
                y=0.98,
                alpha=1.0,
            )
            # 获取材料编号
            mat_tag_rebar = self.MM.get_tag(
                category="uniaxialMaterial", label=f"{self.name}_rebar"
            )[
                0
            ]  # 用户定义的 label
            mat_tag_core = self.MM.get_tag(
                category="uniaxialMaterial", label=f"{self.name}_core"
            )[
                0
            ]  # 用户定义的 label
            # 获取应力应变数据
            strains_rebar = self.SMS.get_data_mat(
                mat_tag=mat_tag_rebar, data_type="Strains", points=points
            ).iloc[: self.ds_stages[-1] + 1, :]
            stresses_rebar = self.SMS.get_data_mat(
                mat_tag=mat_tag_rebar, data_type="Stresses", points=points
            ).iloc[: self.ds_stages[-1] + 1, :]
            strains_core = self.SMS.get_data_mat(
                mat_tag=mat_tag_core, data_type="Strains", points=points
            ).iloc[: self.ds_stages[-1] + 1, :]
            stresses_core = self.SMS.get_data_mat(
                mat_tag=mat_tag_core, data_type="Stresses", points=points
            ).iloc[: self.ds_stages[-1] + 1, :]

            # 坐标轴 1
            ax1 = plt.subplot2grid(
                (1, 2), (0, 0), rowspan=1, colspan=1, facecolor="none"
            )
            ax1.plot(strains_rebar, stresses_rebar, linewidth=1.5)
            ax1.set_title("(a) Rebar")
            ax1.set_xlabel("Strain")
            ax1.set_ylabel("Stress (kPa)")

            # 坐标轴 2
            ax2 = plt.subplot2grid(
                (1, 2), (0, 1), rowspan=1, colspan=1, facecolor="none"
            )
            ax2.plot(strains_core, stresses_core, linewidth=1.5)
            ax2.set_title("(b) Core")
            ax2.set_xlabel("Strain")
            ax2.set_ylabel("Stress (kPa)")

            # 调整尺寸
            adj_fig = PlotyHub.adjust_dual(fig, ax1, ax2)
            # 保存图片
            adj_fig.savefig(self.data_path / f"{self.name}_fiber_resp.png", dpi=320)

    def plot_equivalent_bilinear(
        self,
        line_x: Union[list, np.ndarray],
        line_y: Union[list, np.ndarray],
        point_idx: int,
        info: bool = True,
    ) -> None:
        """
        绘制等效双线性曲线
            - 会保存有数据至本地

        Args:
            line_x (Union[list, np.ndarray]): 输入的 x 坐标数据。
            line_y (Union[list, np.ndarray]): 输入的 y 坐标数据。
            point_idx (int): 数据点索引（屈服点）。
            info (bool, optional): 是否打印计算信息。默认值为 True。

        Returns:
            None: 无返回值，会保存有数据至本地。
        """

        # 原始数据
        original_x = np.array(line_x)
        original_y = np.array(line_y)
        # 数据点
        x_key = original_x[point_idx]
        y_key = original_y[point_idx]
        x_end = original_x[-1]

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 计算等效点
        eq_x, eq_y = opsu.post.equivalent_bilinear(
            line_x=original_x, line_y=original_y, point_idx=point_idx, info=info
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        plt.close("all")
        # 创建新的 Figure 对象
        fig = plt.figure(dpi=100)
        fig.suptitle(
            f" {self.name} M - Φ Equivalent Bilinear Curve",
            fontsize=22,
            color="black",
            x=0.5,
            y=0.98,
            alpha=1.0,
        )
        # 坐标轴
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, facecolor="none")

        # 曲线数据
        ax.plot(
            original_x,
            original_y,
            label="Original",
            color="#03A9F4",
            linewidth=2.5,
            zorder=10,
        )
        # 等效双线性数据
        ax.plot(
            [0.0, eq_x, x_end],
            [0, eq_y, eq_y],
            label="Equivalent",
            color="#E91E63",
            linewidth=2.5,
            marker="o",
            markersize=8,
            zorder=15,
        )
        ax.scatter(x_key, y_key, label="KeyPoint", color="#FF9800", s=60, zorder=20)

        # 坐标系参数
        ax.set_xlim(0, max(original_x) * 1.1)
        ax.set_ylim(0, max(original_y) * 1.1)
        ax.set_xlabel("Φ")
        ax.set_ylabel("Monent (kN·m)")

        # 图例
        leg = ax.legend(
            loc="lower right",
            bbox_to_anchor=(0.97, 0.05),
            labelcolor=(0, 0, 0, 1),
            # frameon=True, fancybox=True, shadow=False,
        )

        # 调整尺寸
        adj_fig = PlotyHub.adjust_single(fig, ax, leg)

        # 箭头样式
        arrowprops = dict(
            arrowstyle="->",  # 使用箭头 'fancy', '->'
            linewidth=2,  # 线宽
            mutation_scale=20,  # 箭头大小
            color="k",  # 箭头颜色
            connectionstyle="arc3,rad=-0.2",  # 带弧度的箭头
        )
        # 添加箭头
        ax.annotate(
            text="Equivalent yield",
            color="k",
            xytext=(eq_x * 2.0, eq_y * 0.6),
            textcoords="data",
            xy=(eq_x, eq_y),
            xycoords="data",
            fontsize=20,
            arrowprops=arrowprops,
            zorder=25,
        )
        ax.annotate(
            text="Initial yield",
            color="k",
            xytext=(x_key * 2.0, y_key * 0.5),
            textcoords="data",
            xy=(x_key, y_key),
            xycoords="data",
            fontsize=20,
            arrowprops=arrowprops,
            zorder=25,
        )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 保存图片
        adj_fig.savefig(self.data_path / f"{self.name}_MPhi.png", dpi=320)

        # 保存数据
        with open(self.data_path / f"{self.name}_MPhi.txt", "w") as f:
            f.write(f"Yield X: {x_key}\n")
            f.write(f"Yield Y: {y_key}\n")
            f.write(f"Equivalent X: {eq_x}\n")
            f.write(f"Equivalent Y: {eq_y}\n")

    def _sec_resp(self, SEC: opst.pre.section.FiberSecMesh, step: int) -> Figure:
        """
        绘制截面响应图。

        Args:
            SEC (opst.pre.section.FiberSecMesh): 截面网格对象。
            step (int): 分析步长。

        Returns:
            Figure: 截面响应图。
        """

        # 获取阈值
        strain_thresholds = self.MM.get_param(
            category="section", label=self.name, key="strain_thresholds"
        )  # 应材料变阈值

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        plt.close("all")
        # 画布
        fig = plt.figure(dpi=100)
        fig.suptitle(
            f"{self.name} - section response - step {step}",
            fontsize=22,
            color="black",
            x=0.5,
            y=0.98,
            alpha=1.0,
        )
        # 坐标轴
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, facecolor="none")

        # 绘制截面
        ax_sec, cbar_sec = self.SMS.plot_sec(
            SEC=SEC,
            data_type="Strains",
            step=step,
            thresholds=strain_thresholds,
            ax=ax,
            fontsize=20,
        )
        # 调整尺寸
        adj_fig = PlotyHub.adjust_single(fig=fig, ax=ax_sec, cbar=cbar_sec)

        return adj_fig

    def plot_sec_resp(self, SEC: opst.pre.section.FiberSecMesh, step: int) -> None:
        """
        绘制截面响应图。

        Args:
            SEC (opst.pre.section.FiberSecMesh): 截面网格对象。
            step (int): 分析步长。

        Returns:
            None: 不返回任何值。
        """

        fig = self._sec_resp(SEC, step)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 保存画布
        fig.savefig(self.data_path / f"{self.name}_resp.png", dpi=320)

    def plot_sec_resp_ani(
        self, SEC: opst.pre.section.FiberSecMesh, max_steps: int, speed: int = 5
    ) -> None:
        """
        绘制截面响应动画。

        Args:
            SEC (opst.pre.section.FiberSecMesh): 截面网格对象。
            max_steps (int): 最大分析步长。
            speed (int, optional): 动画速度，默认为5。

        Returns:
            None: 不返回任何值。
        """

        @gif.frame
        def animation_func(SEC, step):
            fig = self._sec_resp(SEC, step)
            return fig

        # 创建绘图容器
        frames = []
        for step in list(range(0, max_steps, speed)) + [max_steps]:
            # 叠加绘图
            frame = animation_func(SEC, step)  # 每一张新图都是起始点到当前点的数据范围
            frames.append(frame)

        # 保存GIF
        gif.save(frames, str(self.data_path / f"{self.name}_resp.gif"))


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""

if __name__ == "__main__":

    print(f"全局单位：\n{UNIT}\n")
    print(f"全局模型管理器：\n{MM}\n")

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"

    plt.close("all")
    # 画布
    fig = plt.figure(dpi=100)
    fig.suptitle("Example Plot", fontsize=22, color="black", x=0.5, y=0.98, alpha=1.0)
    # 坐标轴
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, facecolor="none")
    ax.plot([1, 2, 3, 4, 5], [6, -2, 3, -4, 5], label="Line 1")
    ax.set_xlim(0, 6)
    ax.set_ylim(-7, 7)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    # 图例
    leg = ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        labelcolor=(0, 0, 0, 1),
        frameon=True,
        fancybox=True,
        shadow=False,
    )
    # 调整
    adj_fig = PlotyHub.adjust_single(fig, ax, leg)

    plt.show()
    # adj_fig.savefig('./ExamplePlot_single.png', dpi=320)

    "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
    plt.close("all")
    # 画布
    fig = plt.figure(dpi=100)
    # fig.suptitle('Example Plot', fontsize=22, color='black', x=0.5, y=0.98, alpha=1.0)
    # 坐标轴
    ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1, facecolor="none")
    ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1, facecolor="none")

    ax1.set_title("(a) Subplot 1")
    ax1.plot([1, 2, 3, 4, 5], [6, -2, 3, -4, 5], label="Line 1")
    ax1.set_xlim(0, 6)
    ax1.set_ylim(-7, 7)
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    leg1 = ax1.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        labelcolor=(0, 0, 0, 1),
        frameon=True,
        fancybox=True,
        shadow=False,
    )

    ax2.set_title("(b) Subplot 2")
    ax2.plot([1, 2, 3, 4, 5], [-6, 2, -3, 4, -5], label="Line 2")
    ax2.set_xlim(0, 6)
    ax2.set_ylim(-7, 7)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    leg2 = ax2.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        labelcolor=(0, 0, 0, 1),
        frameon=True,
        fancybox=True,
        shadow=False,
    )

    # 调整
    adj_fig = PlotyHub.adjust_dual(fig, ax1, ax2, leg1, leg2)

    # plt.show()
    # adj_fig.savefig('./ExamplePlot_dual.png', dpi=320)
