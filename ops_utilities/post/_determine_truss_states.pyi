#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_determine_tress_states.py
@Date    ：2026/01/16 14:40:33
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
import pandas as pd
from typing import Literal, Union, Optional, List

"""
# --------------------------------------------------
# ========== < _determine_tress_states > ==========
# --------------------------------------------------
"""

class TrussStates:
    """
    基于 Truss单元状态 的判断方法
        - Truss单元 定义时非纤维截面单元
        - get_steps(): 获取对应状态的分析步
    """

    def __init__(
        self,
        odb_tag: Union[int, str],
        ele_tag: int,
        lazy_load: bool = False,
        print_info: bool = True,
    ):
        """
        基于 Truss单元状态 的判断方法

        Args:
            odb_tag (Union[int, str]): opstool 数据库标签。
            ele_tag (int): 单元编号。
            integ (int): 单元积分点。
            lazy_load (bool): 是否按需加载数据。默认值为 False。
            print_info (bool): 是否打印加载信息。默认值为 True。

        Returns:
            None

        Examples:
            >>> TS = TrussStates(odb_tag=odb_tag, ele_tag=ele_tag, integ=1)
        """

        ...

    def get_data(
        self,
        data_type: Union[
            Literal["all"],
            Literal["Strain", "Stress", "axialDefo", "axialForce"],
            List[Literal["Strain", "Stress", "axialDefo", "axialForce"]],
        ] = "all",
    ) -> pd.DataFrame:
        """
        获取 Truss单元 对应状态 的数据。

        Args:
            data_type (Union[
                Literal['all'],
                Literal['Strain', 'Stress', 'axialDefo', 'axialForce'],
                List[Literal['Strain', 'Stress', 'axialDefo', 'axialForce']]
                ]): 数据类型。

        Returns:
            pd.DataFrame: 对应状态 的数据。

        Examples:
            >>> TS = TrussStates(odb_tag, ele_tag, integ)
            >>> TS.get_resp(data_type='Strain')
        """

        ...

    def get_steps(
        self,
        data_type: Literal["Strain", "Stress", "axialDefo", "axialForce"],
        stages: Union[int, float, list, tuple, np.ndarray],
        warn: bool = True,
    ) -> List[Optional[int]]:
        """
        根据给定 '数据' 的 Strain/Stress/axialDefo/axialForce 划分阶段，返回每个阶段中最小的 step 值。
            - stages 要求维度为 1， 且单调递增，非递增则自动排序。
            - 如果某个阶段没有数据，返回 None。

        Args:
            data_type (Literal['secDefo', 'secForce']): 数据类型。
            stages (Union[int, float, list, tuple, np.ndarray]): 分割点。可以是单个数值，也可以是序列。

        Returns:
            List[Optional[int]]: 每个阶段中最小的 step 值列表，无法到达的阶段为 None。

        Examples:
            >>> TS = TrussStates(odb_tag, ele_tag, integ)
            >>> TS.get_steps(data_type='axialDefo', stages=0.002)
            >>> [1, 100]
            >>> TS.get_steps(data_type='axialForce', stages=[-0.012, -0.0033, -0.2])
            >>> # [start, stage1, stage2, ..., end]
            >>> [1, 120, 600, None]
        """

        ...
