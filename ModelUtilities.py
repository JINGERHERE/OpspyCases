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

import gif
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.colorbar import Colorbar
from typing import Union, Optional, Literal, List, Tuple, Dict

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

# 全局 OpenSeesEasy 实例
OPSE = opsu.pre.OpenSeesEasy()

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
            cbar (Colorbar, optional): 输入的 `cbar = fig.colorbar()` 对象。

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
        cbar1: Optional[Colorbar] = None,
        cbar2: Optional[Colorbar] = None,
    ) -> Figure:
        """
        双坐标轴绘图参数调整方法。

        Args:
            fig (Figure): 输入的 matplotlib.figure.Figure 对象。
            ax1 (Axes): 输入的 matplotlib.axes.Axes 对象。
            ax2 (Axes): 输入的 matplotlib.axes.Axes 对象。
            leg1 (Legend, optional): 输入的 `leg1 = ax1.legend()` 对象。
            leg2 (Legend, optional): 输入的 `leg2 = ax2.legend()` 对象。
            cbar1 (Colorbar, optional): 输入的 `cbar1 = fig.colorbar()` 对象。
            cbar2 (Colorbar, optional): 输入的 `cbar2 = fig.colorbar()` 对象。

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
        # 颜色条1 调整
        if cbar1 is not None:
            AdjustPlot.adjust_cbar_y(cbar1, **cbar_params)
        # 颜色条2 调整
        if cbar2 is not None:
            AdjustPlot.adjust_cbar_y(cbar2, **cbar_params)

        return fig_adj


"===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ====="


# 数据后处理方法
class PostProcess:

    def __init__(
        self,
        manager: opsu.pre.ModelManager,
        odb_name: str,
        data_path: Union[Path, str, Literal[""]] = "",
        print_info: bool = True,
    ) -> None:
        """
        截面模型后处理类
        Args:
            manager (opsu.pre.ModelManager): 模型管理器实例。
            odb_name (str): 数据库的索引名称。
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
        self.odb_name = odb_name

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 塑性铰单元号
        pier_1_top = manager.get_tag(category="element", label="pier_1_top")  # 单元号
        pier_1_base = manager.get_tag(category="element", label="pier_1_base")  # 单元号
        pier_2_top = manager.get_tag(category="element", label="pier_2_top")  # 单元号
        pier_2_base = manager.get_tag(category="element", label="pier_2_base")  # 单元号

        # 应变状态实例
        self.pier_1_top_SMS = opsu.post.SecMatStates(
            odb_tag=odb_name,
            ele_tag=pier_1_top[0],
            integ=1,
            print_info=print_info,
            lazy_load=True,
        )
        self.pier_1_base_SMS = opsu.post.SecMatStates(
            odb_tag=odb_name,
            ele_tag=pier_1_base[0],
            integ=5,
            print_info=print_info,
            lazy_load=True,
        )
        self.pier_2_top_SMS = opsu.post.SecMatStates(
            odb_tag=odb_name,
            ele_tag=pier_2_top[0],
            integ=1,
            print_info=print_info,
            lazy_load=True,
        )
        self.pier_2_base_SMS = opsu.post.SecMatStates(
            odb_tag=odb_name,
            ele_tag=pier_2_base[0],
            integ=5,
            print_info=print_info,
            lazy_load=True,
        )

        # 截面材料应变状态
        sec_strains = manager.get_param(
            category="section", label="pier_col", key="strain_stages"
        )

        # 截面损伤状态
        pier_1_top_damage = self.pier_1_top_SMS.get_combined_steps_mat(
            mat_config=sec_strains, data_type="Strains", warn=print_info
        )
        pier_1_base_damage = self.pier_1_base_SMS.get_combined_steps_mat(
            mat_config=sec_strains, data_type="Strains", warn=print_info
        )
        pier_2_top_damage = self.pier_2_top_SMS.get_combined_steps_mat(
            mat_config=sec_strains, data_type="Strains", warn=print_info
        )
        pier_2_base_damage = self.pier_2_base_SMS.get_combined_steps_mat(
            mat_config=sec_strains, data_type="Strains", warn=print_info
        )

        # 损伤汇总
        all_damage = (
            [i for i in pier_1_top_damage if i is not None and i != 1]
            + [i for i in pier_1_base_damage if i is not None and i != 1]
            + [i for i in pier_2_top_damage if i is not None and i != 1]
            + [i for i in pier_2_base_damage if i is not None and i != 1]
        )
        # 屈服 - 极限
        self.yield_step = min(all_damage)
        self.limit_step = max(all_damage)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # self.brb_SMS = opsu.post.SecMatStates(
        #     odb_tag=odb_name,
        #     ele_tag=self.MM.get_tag(category="element", label="brb")[0],
        #     integ=3,
        #     print_info=print_info,
        #     lazy_load=True,
        # )

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        ctrl_node = self.MM.get_tag(category="node", label="disp_ctrl")  # 节点号
        # 节点状态实例
        self.ctrl_NS = opsu.post.NodalStates(
            odb_tag=odb_name, resp_type="disp", print_info=print_info
        )
        # 节点位移数据
        self.d_ctrl = self.ctrl_NS.get_data(node_tag=ctrl_node[0], dof="UY")

    # def plot_BRB_resp(self) -> None:
    #     """
    #     绘制预应力响应图。

    #     Returns:
    #         None: 不返回任何值。
    #     """

    #     # 力数据
    #     x = self.brb_SMS.get_data(data_type="secDefo", dofs="P")
    #     y = self.brb_SMS.get_data(data_type="secForce", dofs="P")

    #     # 绘图
    #     plt.close("all")
    #     fig = plt.figure(dpi=100)
    #     ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, facecolor="none")
    #     # ax.plot(self.d_ctrl, f_PT_1, label="PT_1", zorder=11)
    #     ax.plot(x, y, label="brb", zorder=10)
    #     ax.set_xlabel("Displacement (m)")
    #     ax.set_ylabel("Force (kN)")
    #     ax.autoscale()  # 自动调整坐标轴范围
    #     # 图例
    #     leg = ax.legend(
    #         loc="lower right",
    #         bbox_to_anchor=(0.97, 0.05),
    #         labelcolor=(0, 0, 0, 1),
    #         # frameon=True, fancybox=True, shadow=False,
    #     )

    #     # 调整尺寸
    #     adj_fig = PlotyHub.adjust_single(fig, ax, leg)
    #     # 保存图片
    #     adj_fig.savefig(self.data_path / f"figure_BRB_resp.png", dpi=320)

    def _sec_resp(
        self, pier: Literal[1, 2], SEC: opst.pre.section.FiberSecMesh, step: int
    ) -> Figure:
        """
        绘制截面响应图。

        Args:
            pier (Literal[1, 2]):  墩柱编号。
            SEC (opst.pre.section.FiberSecMesh): 截面网格对象。
            step (int): 分析步长。

        Returns:
            Figure: 截面响应图。
        """
        if pier == 1:
            pier_top, pier_base = self.pier_1_top_SMS, self.pier_1_base_SMS
        elif pier == 2:
            pier_top, pier_base = self.pier_2_top_SMS, self.pier_2_base_SMS

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 获取阈值
        strain_thresholds = self.MM.get_param(
            category="section", label="pier_col", key="strain_thresholds"
        )  # 应材料变阈值

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        plt.close("all")
        # 画布
        fig = plt.figure(dpi=100)
        fig.suptitle(
            f"pier {pier} - pier_col response - step {step}",
            fontsize=38,
            color="black",
            x=0.5,
            y=0.98,
            alpha=1.0,
        )
        # 坐标轴
        ax_top = plt.subplot2grid(
            (1, 2), (0, 0), rowspan=1, colspan=1, facecolor="none"
        )
        ax_top.set_title("(a) Top")
        ax_base = plt.subplot2grid(
            (1, 2), (0, 1), rowspan=1, colspan=1, facecolor="none"
        )
        ax_base.set_title("(b) Base")

        # 绘制截面
        ax_sec_top, cbar_sec_top = pier_top.plot_sec(
            SEC=SEC,
            data_type="Strains",
            step=step,
            thresholds=strain_thresholds,
            ax=ax_top,
            fontsize=20,
        )
        ax_sec_base, cbar_sec_base = pier_base.plot_sec(
            SEC=SEC,
            data_type="Strains",
            step=step,
            thresholds=strain_thresholds,
            ax=ax_base,
            fontsize=20,
        )

        # 调整尺寸
        adj_fig = PlotyHub.adjust_dual(
            fig=fig,
            ax1=ax_sec_top,
            ax2=ax_sec_base,
            cbar1=cbar_sec_top,
            cbar2=cbar_sec_base,
        )
        
        # plt.tight_layout()  # 自动调整子图参数，防止标签重叠 / 每次参数不固定

        return adj_fig

    def plot_sec_resp(
        self, pier: Literal[1, 2], SEC: opst.pre.section.FiberSecMesh, step: int
    ) -> None:
        """
        绘制截面响应图。

        Args:
            pier (Literal[1, 2]):  墩柱编号。
            SEC (opst.pre.section.FiberSecMesh): 截面网格对象。
            step (int): 分析步长。

        Returns:
            None: 不返回任何值。
        """

        fig = self._sec_resp(pier, SEC, step)

        "# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----"
        # 保存画布
        fig.savefig(self.data_path / f"pier{pier}_resp_step{step}.png", dpi=320)

    def plot_sec_resp_ani(
        self,
        pier: Literal[1, 2],
        SEC: opst.pre.section.FiberSecMesh,
        max_steps: int,
        speed: int = 5,
    ) -> None:
        """
        绘制截面响应动画。

        Args:
            pier (Literal[1, 2]):  墩柱编号。
            SEC (opst.pre.section.FiberSecMesh): 截面网格对象。
            max_steps (int): 最大分析步长。
            speed (int, optional): 动画速度，默认为5。

        Returns:
            None: 不返回任何值。
        """

        @gif.frame
        def animation_func(SEC, step):
            fig = self._sec_resp(pier, SEC, step)
            return fig

        # 创建绘图容器
        frames = []
        for step in list(range(0, max_steps, speed)) + [max_steps]:
            # 叠加绘图
            frame = animation_func(SEC, step)  # 每一张新图都是起始点到当前点的数据范围
            frames.append(frame)

        # 保存GIF
        gif.save(frames, str(self.data_path / f"pier{pier}_resp.gif"))


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
