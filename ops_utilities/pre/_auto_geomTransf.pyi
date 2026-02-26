#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_auto_geomTransf.py
@Date    ：2026/01/12 12:18:48
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
from typing import Literal, Union, Tuple, List

"""
# --------------------------------------------------
# ========== < _auto_geomTransf > ==========
# --------------------------------------------------
"""

def get_angle(
    p1: Union[List, Tuple, np.ndarray],
    p2: Union[List, Tuple, np.ndarray],
    dof: int,
    ndim: int = 3,
    deg: bool = False,
) -> float:
    """
    计算 `两点连线` 与 `指定维度正交轴线（正向）` 的夹角
        - `dof = 2` 代表第二个维度的正交轴线（ y 正轴）

    Args:
        p1 (Union[List, Tuple, np.ndarray]): 第一个点的坐标，长度为 `ndim` 或小于 `ndim`。
        p2 (Union[List, Tuple, np.ndarray]): 第二个点的坐标，长度为 `ndim` 或小于 `ndim`。
        dof (int): 指定的维度，必须在 `[1, ndim]` 范围内。
        ndim (int, optional): 空间维度，默认值为 `3`。
        deg (bool, optional): 是否返回角度（度），默认值为 `False`（返回弧度）。

    Raises:
        ValueError: 如果 `dof` 不在 `[1, ndim]` 范围内。
        ValueError: 如果 `p1` 或 `p2` 的维度大于 `ndim`。

    Returns:
        float: 计算得到的夹角（弧度或度）。
    """

    ...

# 指定局部坐标转换类型
transfType = Literal["Linear", "PDelta", "Corotational"]

class AutoTransf:
    """
    自动局部坐标转换。
        - ndm2(): `2D` 局部坐标。
        - ndm3(): `3D` 局部坐标。
            - 所有竖向单元的局部 z 轴指向整体 X 轴。
            - 所有非竖向单元的局部 z 轴指向整体 Z 轴。
    """

    @classmethod
    def ndm2(cls, transfType: transfType = "Linear") -> int:
        """
        局部坐标自动转换 `2D`。

        Args:
            transfType (transfType, optional): `默认值：'Linear'` 局部坐标转换类型。

        Returns:
            int: 局部坐标转换编号。

        Examples:
            >>> ops.wipe()
            >>> ops.model('basic', '-ndm', 2, '-ndf', 3)
            >>> ops.node(1, 0., 0.)
            >>> ops.node(2, 0., 1.)
            >>> AutoTransf.ndm2()
            >>> AutoTransf.ndm2('PDelta')
            >>> transf_tags = ops.getCrdTransfTags()
            >>> print(f'All transf tags: {transf_tags}')
            >>> All transf tags: [1, 2]
        """

        ...

    @classmethod
    def ndm3(
        cls,
        node_i: int,
        node_j: int,
        deg: Union[int, float] = 0.0,
        transfType: transfType = "PDelta",
    ) -> int:
        """
        局部坐标自动转换 `3D`。
            - 默认：所有竖向单元的局部 z 轴指向整体 X 轴。
            - 默认：所有非竖向单元的局部 z 轴指向整体 Z 轴。

        Args:
            node_i (int): 单元起始节点编号。
            node_j (int): 单元终止节点编号。
            deg (Union[int, float], optional): `默认值：0.` 。
                - 角度值 (degree)。
                - 单元局部 z 轴的旋转角度。
                - 对于平行于任意整体坐标轴的单元，旋转方向遵循（绕局部 x 轴的）右手定则。
            transfType (transfType, optional): `默认值：'PDelta'` 局部坐标转换类型。

        Returns:
            int: 局部坐标转换编号。

        Examples:
            >>> ops.wipe()
            >>> ops.model('basic', '-ndm', 3, '-ndf', 6)
            >>> ops.node(1, 0., 0., 0.)
            >>> ops.node(2, 0., 1., 1.)
            >>> AutoTransf.ndm3(*(1, 2), 20)
            >>> AutoTransf.ndm3(*(2, 1), 20)
            >>> transf_tags = ops.getCrdTransfTags()
            >>> print(f'All transf tags: {transf_tags}')
            >>> All transf tags: [1, 2]
        """

        ...
