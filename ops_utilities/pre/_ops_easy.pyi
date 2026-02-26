#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_ops_easy.py
@Date    ：2026/02/01 20:09:33
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from typing import Self, Union, Optional, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from ._model_manager import ModelManager

"""
# --------------------------------------------------
# ========== < _ops_easy > ==========
# --------------------------------------------------
"""

# 分配器相关 默认值
tag_start = 1
time_lim = 30  # 秒
tag_1st = {"idx": 0}  # tag 位于 idx=0 的位置
tag_2nd = {"idx": 1}  # tag 位于 idx=1 的位置

class OpenSeesEasy:
    def __init__(self, manager: Optional[ModelManager] = None):
        """
        调用 OpenSees 命令，尝试可用的 tag 值。
            - 实例化时，可传入 ModelManager 实例，用于管理已存在的 tag 列表。
            - `OpenSees` 所有报错详细信息已被捕获至 `.OpenSeesEasy.log` 文件内。终端返回的错误信息不包含错误原因。

        tag 相关说明：
            - 对于需要定义唯一 `tag` 值的 OpenSees 命令，可忽略输入 `tag` 值，依次输入其余参数。
            - 若需要从特定 `tag` 开始尝试，需显式声明。

        lim 相关说明：
            - 对于需要唯一 `tag` 值的 OpenSees 命令，需在 `lim` 秒内成功获取可用 `tag` 值。
            - 若在 `lim` 秒内未成功获取可用 `tag` 值，则抛出异常。
            - 若未指定 `lim` 值，则使用默认值 30 秒。
            - 指定 `lim` 值时，需显式声明。

        log 相关说明：
            - `log` 参数为是否打印：需要 `tag` 参数的 `OpenSees` 命令运行日志。
            - 若未指定 `log` 值，则使用默认值 `True`。
                - `log = False` 可大幅提高运行效率。
            - 指定 `log` 值时，需显式声明。

        Args:
            manager (ModelManager): 模型管理器实例`ope_utilities.pre.ModelManager`，用于管理已存在的 tag 列表。

        Returns:
            int: 成功定义的 tag 值。
        """

        ...

    def add_manager(self, manager: ModelManager) -> Self:
        """
        添加 ModelManager 实例。

        Args:
            manager (ModelManager): 要添加的 ModelManager 实例。

        Returns:
            Self: 当前实例，用于链式调用。
        """

        ...

    def get_manager(self) -> ModelManager:
        """
        获取当前实例的 ModelManager 实例。

        Returns:
            ModelManager: 当前实例的 ModelManager 实例。
        """

        ...

    def refresh(self) -> Self:
        """
        刷新当前实例的 ModelManager 实例。
            - 将缓存中的数据同步到 管理器 中。

        Returns:
            Self: 当前实例，用于链式调用。
        """

        ...

    def _sync_model(self) -> Self:
        """
        将 OpenSees域 和 管理器 中已存在的 tag 列表，同步到 装饰器 中。

        Args:
            None: 无参数。

        Returns:
            Self: 当前实例，用于链式调用。
        """

        ...

    def _to_cache(self):
        """
        将新获取的 (category, tag) 元组，暂存到缓冲区。
        """

        ...

    def _sync_to_manager(self):
        """
        将 装饰器 中已分配的 tag 列表同步到 管理器 中。
        """

        ...

    def wipe(self) -> Self:
        """
        Wipe all.
            - 清空所有 `OpenSees` 相关数据。
            - 清空 `ModelManager` 中的数据。
            - 清空当前 `类实例缓存` 中的数据 (self._buffer)。

        Returns:
            Self: 当前实例本身。可链式调用。
        """

        ...

    def model(self, *args) -> None:
        """
        调用 OpenSees 命令，定义模型。

        Args:
            *args: 传递给 OpenSees 指定的参数

        Returns:
            None

        Examples:
            >>> model('basic', '-ndm', 1, '-ndf', 1)
            >>> model('basic', '-ndm', 2, '-ndf', 3)
            >>> model('basic', '-ndm', 3, '-ndf', 6)
        """

        ...

    def node(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `node command
        <https://openseespydoc.readthedocs.io/en/latest/src/node.html>`_
        """

        ...

    def element(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `element command
        <https://openseespydoc.readthedocs.io/en/latest/src/element.html>`_
        """

        ...

    def beamIntegration(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `beamIntegration command
        <https://openseespydoc.readthedocs.io/en/latest/src/beamIntegration.html>`_
        """

        ...

    def block2D(self, *args) -> None:
        """
        ``args`` see `block2D command
        <https://openseespydoc.readthedocs.io/en/latest/src/block2D.html>`_
        """

        ...

    def block3D(self, *args) -> None:
        """
        ``args`` see `block3D command
        <https://openseespydoc.readthedocs.io/en/latest/src/block3D.html>`_
        """

        ...

    def mass(self, *args) -> None:
        """
        ``args`` see `mass command
        <https://openseespydoc.readthedocs.io/en/latest/src/mass.html>`_
        """

        ...

    def region(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `region command
        <https://openseespydoc.readthedocs.io/en/latest/src/region.html>`_
        """

        ...

    def rayleigh(self, *args) -> None:
        """
        ``args`` see `rayleigh command
        <https://openseespydoc.readthedocs.io/en/latest/src/reyleigh.html>`_
        """

        ...

    def geomTransf(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `geomTransf command
        <https://openseespydoc.readthedocs.io/en/latest/src/geomTransf.html>`_
        """

        ...

    def uniaxialMaterial(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `uniaxialMaterial command
        <https://openseespydoc.readthedocs.io/en/latest/src/uniaxialMaterial.html>`_
        """

        ...

    def nDMaterial(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `nDMaterial command
        <https://openseespydoc.readthedocs.io/en/latest/src/ndMaterial.html>`_
        """

        ...

    def section(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `section command
        <https://openseespydoc.readthedocs.io/en/latest/src/section.html#section>`_
        """

        ...

    def fiber(
        self, *args, color: Union[str, tuple, list] = "red", opacity: float = 1.0
    ) -> None:
        """
        ``args`` see `fiber command
        <https://openseespydoc.readthedocs.io/en/latest/src/fiber.html>`_
        """

        ...

    def patch(
        self, *args, color: Union[str, tuple, list] = "#BFBFFF", opacity: float = 1.0
    ) -> None:
        """
        ``args`` see `patch command
        <https://openseespydoc.readthedocs.io/en/latest/src/patch.html>`_
        """

        ...

    def layer(
        self, *args, color: Union[str, tuple, list] = "red", opacity: float = 1.0
    ) -> None:
        """
        ``args`` see `layer command
        <https://openseespydoc.readthedocs.io/en/latest/src/layer.html>`_
        """

        ...

    def plot_sec(self, sec_tag: int = 1) -> Tuple[Figure, Axes]:
        """
        绘制 纤维截面 图。

        Args:
            sec_tag (int, optional): 截面标签。默认值为 1。

        Returns:
            Tuple(Figure, Axes):
                - Figure: 截面 的画布对象。
                - Axes: 截面 的轴对象。
        """

        ...

    def frictionModel(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `frictionModel command
        <https://openseespydoc.readthedocs.io/en/latest/src/frictionModel.html>`_
        """

        ...

    def fix(self, *args) -> None:
        """
        ``args`` see `fix command
        <https://openseespydoc.readthedocs.io/en/latest/src/fix.html>`_
        """

        ...

    def fixX(self, *args) -> None:
        """
        ``args`` see `fixX command
        <https://openseespydoc.readthedocs.io/en/latest/src/fixX.html>`_
        """

        ...

    def fixY(self, *args) -> None:
        """
        ``args`` see `fixY command
        <https://openseespydoc.readthedocs.io/en/latest/src/fixY.html>`_
        """

        ...

    def fixZ(self, *args) -> None:
        """
        ``args`` see `fixZ command
        <https://openseespydoc.readthedocs.io/en/latest/src/fixZ.html>`_
        """

        ...

    def rigidLink(self, *args) -> None:
        """
        ``args`` see `rigidLink command
        <https://openseespydoc.readthedocs.io/en/latest/src/rigidLink.html>`_
        """

        ...

    def rigidDiaphragm(self, *args) -> None:
        """
        ``args`` see `rigidDiaphragm command
        <https://openseespydoc.readthedocs.io/en/latest/src/rigidDiaphragm.html>`_
        """

        ...

    def equalDOF(self, *args) -> None:
        """
        ``args`` see `equalDOF command
        <https://openseespydoc.readthedocs.io/en/latest/src/equalDOF.html>`_
        """

        ...

    def equalDOF_Mixed(self, *args) -> None:
        """
        ``args`` see `equalDOF_Mixed command
        <https://openseespydoc.readthedocs.io/en/latest/src/equalDOF_Mixed.html>`_
        """

        ...

    def pressureConstraint(self, *args) -> None:
        """
        ``args`` see `pressureConstraint command
        <https://openseespydoc.readthedocs.io/en/latest/src/pressureConstraint.html>`_
        """

        ...

    def timeSeries(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `timeSeries command
        <https://openseespydoc.readthedocs.io/en/latest/src/timeSeries.html>`_
        """

        ...

    def pattern(
        self,
        *args,
        tag: int = tag_start,
        lim: Union[int, float] = time_lim,
        log: bool = False,
    ) -> int:
        """
        ``args`` see `pattern command
        <https://openseespydoc.readthedocs.io/en/latest/src/pattern.html>`_
        """

        ...
