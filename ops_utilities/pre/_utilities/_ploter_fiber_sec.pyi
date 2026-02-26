#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_ploter_fiber_sec.py
@Date    ：2026/02/05 21:14:14
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

"""
@Copy Right: opstool
@Code Author: Yan Yexiang
@Repository: https://github.com/yexiang92/opstool/blob/master/opstool/pre/section/_plot_fiber_sec_by_cmds.py
"""

import copy
import inspect
import numpy as np
from functools import wraps
from collections import defaultdict
from typing import Optional, Union, List, Tuple, Dict, Callable

import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..._color_map import ColorHub, random_color


"""
# --------------------------------------------------
# ========== < _fiber_sec_plot > ==========
# --------------------------------------------------
"""


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# < 网格点 收集实例 >
# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
class MeshPointCollector:

    @staticmethod
    def line(
        y1: float, z1: float, y2: float, z2: float, num: int
    ) -> List[Tuple[float, float]]:
        """
        直线网格

        Args:
            y1 (float): 直线起始点 x 坐标
            z1 (float): 直线起始点 y 坐标
            y2 (float): 直线结束点 x 坐标
            z2 (float): 直线结束点 y 坐标
            num (int): 直线网格数

        Returns:
            list: 直线网格节点坐标列表
        """

        length = np.sqrt((y2 - y1) ** 2 + (z2 - z1) ** 2)

        cos_alpha = np.abs(y2 - y1) / length
        sin_alpha = np.abs(z2 - z1) / length

        delta_l = length / num
        delta_y = np.sign(y2 - y1) * delta_l * cos_alpha
        delta_z = np.sign(z2 - z1) * delta_l * sin_alpha

        nodes = []
        for i in range(num + 1):
            nodes.append((y1 + i * delta_y, z1 + i * delta_z))

        return nodes

    @staticmethod
    def arc(
        ang0: float, ang1: float, r: float, num_c: int, yc: float, zc: float
    ) -> List[Tuple[float, float]]:
        """
        弧形网格

        Args:
            ang0 (float): 弧形起始角度（度）
            ang1 (float): 弧形结束角度（度）
            r (float): 弧形半径
            num_c (int): 弧形网格数
            yc (float): 弧形中心 x 坐标
            zc (float): 弧形中心 y 坐标

        Returns:
            list: 弧形网格节点坐标列表

        """

        ang0 = np.deg2rad(ang0)
        ang1 = np.deg2rad(ang1)
        delta_ang = (ang1 - ang0) / num_c

        nodes = []
        for i in range(num_c + 1):
            nodes.append(
                (
                    r * np.cos(ang0 + i * delta_ang) + yc,
                    r * np.sin(ang0 + i * delta_ang) + zc,
                )
            )

        return nodes


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
# < 纤维截面绘制器 实例 >
# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
class FiberSecPloter:

    def __init__(self):
        """
        纤维截面绘制器
        """

        self.patches: dict = defaultdict(lambda: [])
        self.line2Ds: dict = defaultdict(lambda: [])
        self.sec_tag: int = 1

    def clear(self):
        """
        清除所有纤维截面数据
        """

        self.patches.clear()
        self.line2Ds.clear()
        self.sec_tag = 1

    def set_tag(self, tag: int):
        """
        设置当前截面标签
        """

        self.sec_tag = tag

    def add_patch_rect(
        self, *args, color: Union[str, tuple, list], alpha: float
    ) -> None:
        """
        Add a patch fiber.
            - ops.patch("rect", ...)

        Args:
            *args: Same to OpenSees.
            color(Union[str, tuple, list]): Color of the fiber.
            alpha(float): Transparency of the fiber.

        Returns:
            None

        Examples:
            >>> add_patch_rect(1, 1, 0.0, 0.0, 1.0, 1.0)
        """

        numy, numz = args[1], args[2]
        yi, zi, yj, zj = args[3:]
        rect = mpathes.Rectangle(
            (yi, zi), yj - yi, zj - zi, color=color, alpha=alpha, zorder=10
        )
        self.patches[self.sec_tag].append(rect)

        delta_y = np.abs(yj - yi) / numy
        delta_z = np.abs(zj - zi) / numz
        yg = np.arange(yi, yj + 0.1 * delta_y, delta_y)
        zg = np.arange(zi, zj + 0.1 * delta_z, delta_z)

        for _y in yg:
            self.line2Ds[self.sec_tag].append([(_y, zi), (_y, zj)])
        for _z in zg:
            self.line2Ds[self.sec_tag].append([(yi, _z), (yj, _z)])

    def add_patch_quad(
        self, *args, color: Union[str, tuple, list], alpha: float
    ) -> None:
        """
        Add a patch fiber.
            - ops.patch("quad", ...)

        Args:
            *args: Same to OpenSees.
            color(Union[str, tuple, list]): Color of the fiber.
            alpha(float): Transparency of the fiber.

        Returns:
            None

        Examples:
            >>> add_patch_quad(1, 1, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0)
        """

        numij, numjk = args[1], args[2]
        yi, zi, yj, zj, yk, zk, yl, zl = args[3:]
        yz = np.array([[yi, zi], [yj, zj], [yk, zk], [yl, zl]])

        yzij = MeshPointCollector.line(yi, zi, yj, zj, numij)
        yzjk = MeshPointCollector.line(yj, zj, yk, zk, numjk)
        yzkl = MeshPointCollector.line(yk, zk, yl, zl, numij)
        yzli = MeshPointCollector.line(yl, zl, yi, zi, numjk)

        polygon = mpathes.Polygon(yz, color=color, alpha=alpha, zorder=10)
        self.patches[self.sec_tag].append(polygon)

        for i in range(len(yzij)):
            yz1 = yzij[i]
            yz2 = yzkl[::-1][i]
            self.line2Ds[self.sec_tag].append([(yz1[0], yz1[1]), (yz2[0], yz2[1])])
        for i in range(len(yzjk)):
            yz1 = yzjk[i]
            yz2 = yzli[::-1][i]
            self.line2Ds[self.sec_tag].append([(yz1[0], yz1[1]), (yz2[0], yz2[1])])

    def add_patch_circ(
        self, *args, color: Union[str, tuple, list], alpha: float
    ) -> None:
        """
        Add a patch fiber.
            - ops.patch("circ", ...)

        Args:
            *args: Same to OpenSees.
            color(Union[str, tuple, list]): Color of the fiber.
            alpha(float): Transparency of the fiber.

        Returns:
            None

        Examples:
            >>> add_patch_circ(1, 1, 0.0, 0.0, 1.0, 2.0)
        """

        num_circ, num_rad = args[1], args[2]
        yc, zc = args[3], args[4]
        r_in, r_ex = args[5], args[6]

        if len(args) > 7:
            ang_s, ang_e = args[7], args[8]
        else:
            ang_s, ang_e = 0.0, 360

        node_ex = MeshPointCollector.arc(ang_s, ang_e, r_ex, num_circ, yc, zc)
        node_in = MeshPointCollector.arc(ang_s, ang_e, r_in, num_circ, yc, zc)

        wedge = mpathes.Wedge(
            (yc, zc),
            r_ex,
            ang_s,
            ang_e,
            width=r_ex - r_in,
            color=color,
            alpha=alpha,
            zorder=10,
        )
        self.patches[self.sec_tag].append(wedge)

        for i in range(num_circ + 1):
            yz_ex = node_ex[i]
            yz_in = node_in[i]
            self.line2Ds[self.sec_tag].append(
                [(yz_in[0], yz_in[1]), (yz_ex[0], yz_ex[1])]
            )

        for i in range(num_rad + 1):
            delta_r = (r_ex - r_in) / num_rad
            arc = mpathes.Arc(
                (yc, zc),
                2 * (r_in + i * delta_r),
                2 * (r_in + i * delta_r),
                theta1=ang_s,
                theta2=ang_e,
                lw=0.5,
                color="k",
                alpha=alpha,
                zorder=10,
            )
            self.patches[self.sec_tag].append(arc)

    def add_fiber_points(
        self, *args, color: Union[str, tuple, list], alpha: float
    ) -> None:
        """
        Add a point fiber.
            - ops.fiber(...)

        Args:
            *args: Same to OpenSees.
            color(Union[str, tuple, list]): Color of the fiber.
            alpha(float): Transparency of the fiber.

        Returns:
            None

        Examples:
            >>> add_fiber_points(0.0, 0.0, 1.0)
        """

        # 坐标，面积
        y, z, area = args[0], args[1], args[2]
        r = np.sqrt(area / np.pi)

        circle = mpathes.Circle((y, z), r, color=color, alpha=alpha, zorder=12)

        self.patches[self.sec_tag].append(circle)

    def add_layer_straight(
        self, *args, color: Union[str, tuple, list], alpha: float
    ) -> None:
        """
        Add a layer of straight fibers.

        Args:
            *args: Same to OpenSees.
            color(Union[str, tuple, list]): Color of the fibers.
            alpha(float): Transparency of the fibers.

        Returns:
            None

        Examples:
            >>> add_layer_straight(1, 10, 1.0, 0.0, 0.0, 1.0)
        """

        num, area = args[1], args[2]
        r = np.sqrt(area / np.pi)
        yi, zi, yj, zj = args[3:]

        node = MeshPointCollector.line(yi, zi, yj, zj, num - 1)
        for i in range(num):
            circle = mpathes.Circle(node[i], r, color=color, alpha=alpha, zorder=12)
            self.patches[self.sec_tag].append(circle)

    def add_layer_circ(
        self, *args, color: Union[str, tuple, list], alpha: float
    ) -> None:
        """
        Add a layer of circular fibers.
            - ops.patch("circ", ...)

        Args:
            *args: Same to OpenSees.
            color(Union[str, tuple, list]): Color of the fibers.
            alpha(float): Transparency of the fibers.

        Returns:
            None

        Examples:
            >>> add_layer_circ(1, 10, 1.0, 0.0, 0.0, 1.0)
        """

        num, area = args[1], args[2]
        yc, zc, r = args[3], args[4], args[5]

        if len(args) > 6:
            ang_s, ang_e = args[6], args[7]
        else:
            ang_s, ang_e = 0.0, 360.0 - 360 / num

        node = MeshPointCollector.arc(ang_s, ang_e, r, num - 1, yc, zc)
        for i in range(num):
            circle = mpathes.Circle(
                node[i], np.sqrt(area / np.pi), color=color, alpha=alpha, zorder=12
            )
            self.patches[self.sec_tag].append(circle)

    def fetch(self, func: Callable) -> Callable:
        """
        < 装饰器 > 获取纤维点数据。
            - 被装饰函数可显式声明 `color` 和 `opacity` 参数。
                - color(Union[str, tuple, list]): 纤维显示颜色
                - opacity(float): 显示透明度。

        Args:
            func(Callable): The function to be fetched.

        Returns:
            Callable: The fetched function.
        """

        # 获取函数签名
        sig = inspect.signature(func)
        params = list(sig.parameters.values())  # 获取函数参数
        is_method = len(params) > 0 and params[0].name in (
            "self",
            "cls",
        )  # 判断是否为类实例中的方法

        @wraps(func)
        def wrapper(*args, **kwargs):

            # 参数列表
            ls_args = list(args)

            # 移除第一个参数：self 或 cls
            if is_method:
                ls_args.pop(0)

            # 获取颜色
            provide_color = kwargs.pop("color", random_color())  # 保底为 随机颜色
            # 获取透明度
            provide_opacity = kwargs.pop("opacity", 1.0)  # 保底为 1.0

            if func.__name__ == "fiber":
                self.add_fiber_points(
                    *ls_args, color=provide_color, alpha=provide_opacity
                )

            else:
                # 获取第一个参数并删除：生成截面几何形状的类型
                fiber_geom = ls_args.pop(0)

                # 函数映射
                fetch_map = {
                    "patch": {
                        "quad": self.add_patch_quad,
                        "rect": self.add_patch_rect,
                        "circ": self.add_patch_circ,
                    },
                    "layer": {
                        "straight": self.add_layer_straight,
                        "circ": self.add_layer_circ,
                    },
                }
                # 调用对应的函数
                fetch_map[func.__name__][fiber_geom](
                    *ls_args, color=provide_color, alpha=provide_opacity
                )

            return func(*args)

        return wrapper

    def plot_sec(
        self,
        sec_tag: int,
        title: str = "My Section",
        label_size: int = 15,
        tick_size: int = 12,
        title_size: int = 18,
    ) -> Tuple[Figure, Axes]:
        """
        绘制 纤维截面 图。

        Args:
            sec_tag (int): 截面标签。
            title (str, optional): 图标题。默认值为 "My Section"。
            label_size (int, optional): 坐标轴标签字体大小。默认值为 15。
            tick_size (int, optional): 坐标轴刻度字体大小。默认值为 12。
            title_size (int, optional): 图标题字体大小。默认值为 18。

        Returns:
            Tuple(Figure, Axes):
                - Figure: 截面 的画布对象。
                - Axes: 截面 的轴对象。
        """

        plt.close("all")

        # 画布创建
        fig_sec, ax_sec = plt.subplots()

        # ----------------------------------------
        # 补丁块
        for pat in self.patches[sec_tag]:
            ax_sec.add_patch(copy.copy(pat))

        # 补丁区域网格线
        lc = LineCollection(
            self.line2Ds[sec_tag], linewidths=0.5, colors="k", zorder=11
        )
        ax_sec.add_collection(lc)

        # ----------------------------------------
        # 画布坐标系设置
        ax_sec.set_title(title, fontsize=title_size)
        ax_sec.set_xlabel("y", fontsize=label_size)
        ax_sec.set_ylabel("z", fontsize=label_size)
        ax_sec.tick_params(
            axis="both", which="major", width=1.2, length=5, labelsize=tick_size
        )
        ax_sec.grid(True, "major", linestyle="--", linewidth=1.0, zorder=1)

        # 边框线宽设置
        for loc in ["bottom", "top", "left", "right"]:
            ax_sec.spines[loc].set_linewidth(1.0)

        # ----------------------------------------
        # 保底调整设置
        ax_sec.autoscale()  # 自动调整坐标轴范围
        ax_sec.axis("equal")  # 使 x 轴和 y 轴的刻度比例相等
        plt.tight_layout()  # 自动调整子图参数，防止标签重叠

        return fig_sec, ax_sec


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""


if __name__ == "__main__":
    pass
