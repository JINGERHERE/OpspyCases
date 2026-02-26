#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_adjust_plot.py
@Date    ：2026/01/28 15:01:47
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.colorbar import Colorbar

"""
# --------------------------------------------------
# ========== < _adjust_plot > ==========
# --------------------------------------------------
"""

class AdjustPlot:
    @classmethod
    def adjust_fig(
        cls,
        fig: Figure,
        fig_width: float = 10.63,
        fig_height: float = 7.09,
        w_space: float = 2.13,
        h_space: float = 2.48,
        left: float = 1.38,
        right: float = 0.59,
        top: float = 0.59,
        bottom: float = 0.98,
    ) -> Figure:
        """
        调整 matplotlib.figure.Figure 对象的大小、子图间距、边缘距离等。

        Args:
            fig (Figure): 要调整的 Figure 对象。
            fig_width (float): 画布宽度。`默认值：`10.63，单位：英寸。
            fig_height (float): 画布高度。`默认值：`7.09，单位：英寸。
            w_space (float): 子图水平间距。`默认值：`2.13，单位：英寸。
            h_space (float): 子图垂直间距。`默认值：`2.48，单位：英寸。
            left (float): 子图左边缘距离。`默认值：`1.38，单位：英寸。
            right (float): 子图右边缘距离。`默认值：`0.59，单位：英寸。
            top (float): 子图上边缘距离。`默认值：`0.59，单位：英寸。
            bottom (float): 子图下边缘距离。`默认值：`0.98，单位：英寸。

        Returns:
            Figure: 返回调整后的 Figure 对象。

        """

        ...

    @classmethod
    def adjust_bg(cls, ax: Axes) -> Axes:
        """
        调整 matplotlib.axes.Axes 对象的背景颜色、透明度等。

        Args:
            ax (Axes): 要调整的 Axes 对象。

        Returns:
            Axes: 返回调整后的 Axes 对象。

        """

        ...

    @classmethod
    def adjust_x(
        cls,
        ax: Axes,
        label: str = "",
        label_size: int = 20,
        label_pad: float = 7.0,
        ticks: bool = True,
        ticks_size: int = 18,
        ticks_pad: float = 12.0,
    ) -> Axes:
        """
        调整 matplotlib.axes.Axes 对象的下 x轴 轴线刻度、标签、字体等。

        Args:
            ax (Axes): 要调整的 Axes 对象。

            label (str): 下 x轴 标签。`默认值：`''。
            label_size (int): x轴 标签字体大小。`默认值：`20，单位：磅。
            label_pad (float): x轴 标签与轴线的间距。`默认值：`7.，单位：磅。

            ticks (bool): 是否显示 x轴 刻度。`默认值：`True。
            ticks_size (int): x轴 刻度字体大小。`默认值：`18，单位：磅。
            ticks_pad (float): x轴 刻度与数字的间距。`默认值：`12.，单位：磅。

        Returns:
            Axes: 返回调整后的 Axes 对象。

        """

        ...

    @classmethod
    def adjust_y(
        cls,
        ax: Axes,
        label: str = "",
        label_size: int = 20,
        label_pad: float = 7.0,
        ticks: bool = True,
        ticks_size: int = 18,
        ticks_pad: float = 12.0,
    ) -> Axes:
        """
        调整 matplotlib.axes.Axes 对象的左 y轴 轴线刻度、标签、字体等。

        Args:
            ax (Axes): 要调整的 Axes 对象。

            label (str): 左 y轴 标签。`默认值：`''。
            label_size (int): y轴 标签字体大小。`默认值：`20，单位：磅。
            label_pad (float): y轴 标签与轴线的间距。`默认值：`7.，单位：磅。

            ticks (bool): 是否显示 y轴 刻度。`默认值：`True。
            ticks_size (int): y轴 刻度字体大小。`默认值：`18，单位：磅。
            ticks_pad (float): y轴 刻度与数字的间距。`默认值：`12.，单位：磅。

        Returns:
            Axes: 返回调整后的 Axes 对象。

        """

        ...

    @classmethod
    def adjust_leg(
        cls, leg: Legend, font_size: int = 18, pad: int = 1, rounding_size: int = 2
    ) -> Legend:
        """
        调整图例位置、字体大小、透明度等。

        Args:
            leg (Legend): 要调整的 Legend 对象。
            loc (str, optional): 图例位置。默认值为 'upper right'。
            font_size (int, optional): 字体大小。默认值为 18。
            pad (int, optional): 图例边框与内容的内边距。默认值为 1。
            rounding_size (int, optional): 图例边框圆角大小。默认值为 2。

        Returns:
            Legend: 返回调整后的 Legend 对象。

        """

        ...

    @classmethod
    def adjust_cbar_y(
        cls,
        cbar: Colorbar,
        label: str = "",
        label_size: int = 20,
        label_pad: float = 7.0,
        ticks: bool = True,
        ticks_size: int = 18,
        ticks_pad: float = 12.0,
    ) -> Colorbar:
        """
        调整 `竖向` 颜色条（Colorbar）轴线刻度、标签、字体等。

        Args:
            cbar (Colorbar): 要调整的 Colorbar 对象。

            label (str): 颜色条标签。`默认值：`''。
            label_size (int): 颜色条标签字体大小。`默认值：`20，单位：磅。
            label_pad (float): 颜色条标签与轴线的间距。`默认值：`7.，单位：磅。

            ticks (bool): 是否显示颜色条刻度。`默认值：`True。
            ticks_size (int): 颜色条刻度字体大小。`默认值：`18，单位：磅。
            ticks_pad (float): 颜色条刻度与数字的间距。`默认值：`12.，单位：磅。

        Returns:
            Colorbar: 返回调整后的 Colorbar 对象。

        """

        ...

    @classmethod
    def adjust_cbar_x(
        cls,
        cbar: Colorbar,
        label: str = "",
        label_size: int = 20,
        label_pad: float = 7.0,
        ticks: bool = True,
        ticks_size: int = 18,
        ticks_pad: float = 12.0,
    ) -> Colorbar:
        """
        调整 `横向` 颜色条（Colorbar）轴线刻度、标签、字体等。

        Args:
            cbar (Colorbar): 要调整的 Colorbar 对象。

            label (str): 颜色条标签。`默认值：`''。
            label_size (int): 颜色条标签字体大小。`默认值：`20，单位：磅。
            label_pad (float): 颜色条标签与轴线的间距。`默认值：`7.，单位：磅。

            ticks (bool): 是否显示颜色条刻度。`默认值：`True。
            ticks_size (int): 颜色条刻度字体大小。`默认值：`18，单位：磅。
            ticks_pad (float): 颜色条刻度与数字的间距。`默认值：`12.，单位：磅。

        Returns:
            Colorbar: 返回调整后的 Colorbar 对象。

        """

        ...
