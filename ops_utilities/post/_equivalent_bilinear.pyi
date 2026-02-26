#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_equivalent_bilinear.py
@Date    ：2026/01/30 20:22:30
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
from typing import Union, Tuple

"""
# --------------------------------------------------
# ========== < _equivalent_bilinear > ==========
# --------------------------------------------------
"""

def equivalent_bilinear(
    line_x: Union[list, np.ndarray],
    line_y: Union[list, np.ndarray],
    point_idx: int,
    info: bool = True,
) -> Tuple[float, float]:
    """
    计算等效双线性点

    Args:
        line_x (Union[list, np.ndarray]): 输入的 x 坐标数据。
        line_y (Union[list, np.ndarray]): 输入的 y 坐标数据。
        point_idx (int): 数据点索引（屈服点）。
        info (bool, optional): 是否打印计算信息。默认值为 True。

    Returns:
        Tuple[float, float]: 等效双线性点的坐标 (eq_x, eq_y)。
            - eq_x: 等效 x 值。
            - eq_y: 等效 y 值。
    """

    ...
