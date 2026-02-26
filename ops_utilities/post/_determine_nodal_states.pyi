#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_determine_nodal_states.py
@Date    ：2026/01/16 18:41:16
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
import pandas as pd
from typing import Literal, Union

"""
# --------------------------------------------------
# ========== < _determine_nodal_states > ==========
# --------------------------------------------------
"""

class NodalStates:
    """
    基于 节点状态 的判断方法
        - get_data(): 获取节点 响应 数据。
        - get_steps(): 获取对应状态的分析步
    """

    def __init__(
        self,
        odb_tag: Union[str, int],
        resp_type: Literal[
            "disp",
            "vel",
            "accel",
            "reaction",
            "reactionIncInertia",
            "rayleighForces",
            "pressure",
        ],
        lazy_load: bool = False,
        print_info: bool = True,
    ) -> None:
        """
        基于 节点状态 的判断方法

        Args:
            odb_tag (int): opstool 数据库标签。
            ele_tag (int): 单元编号。
            integ (int): 单元积分点。
            lazy_load (bool): 是否按需加载数据。默认值为 False。
            print_info (bool): 是否打印加载信息。默认值为 True。

        Returns:
            None

        Examples:
            >>> NS = NodalStates(
            >>>     odb_tag=CASE, resp_type='disp'
            >>>     )
        """

        ...

    def get_data(
        self, node_tag: int, dof: Literal["UX", "UY", "UZ", "RX", "RY", "RZ"]
    ) -> pd.DataFrame:
        """
        获取 节点 对应状态 的数据。

        Args:
            node_tag (int): 节点标签。
            dof (Literal['UX', 'UY', 'UZ', 'RX', 'RY', 'RZ']): 自由度。

        Returns:
            pd.DataFrame: 对应状态 的数据。

        Examples:
            >>> NS = NodalStates(odb_tag=CASE, resp_type='disp')
            >>> NS.get_data(node_tag=1, dof='UX')
        """

        ...

    def get_steps(
        self,
        node_tag: int,
        dof: Literal["UX", "UY", "UZ", "RX", "RY", "RZ"],
        stages: Union[int, float, list, tuple, np.ndarray],
        warn: bool = True,
    ):
        """
        根据给定 '数据' 的 node_tag 和 dof 划分阶段，返回每个阶段中最小的 step 值。
            - stages 要求维度为 1， 且单调递增，非递增则自动排序。
            - 如果某个阶段没有数据，返回 None。

        Args:
            node_tag (int): 节点标签。
            dof (Literal['UX', 'UY', 'UZ', 'RX', 'RY', 'RZ']): 自由度。
            stages (Union[int, float, list, tuple, np.ndarray]): 分割点。可以是单个数值，也可以是序列。

        Returns:
            List[Optional[int]]: 每个阶段中最小的 step 值列表，无法到达的阶段为 None。

        Examples:
            >>> NS = NodalStates(odb_tag=CASE, resp_type='disp')
            >>> NS.get_steps(node_tag=1, dof='UX', stages=0.002)
            >>> [1, 100]
            >>> NS.get_steps(node_tag=1, dof='UX', stages=[-0.012, -0.0033, -0.2])
            >>> # [start, stage1, stage2, ..., end]
            >>> [1, 120, 600, None]
        """

        ...
