#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_get_callables.py
@Date    ：2026/01/27 20:14:17
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from typing import Union, Literal, Dict, Callable

"""
# --------------------------------------------------
# ========== < _get_callables > ==========
# --------------------------------------------------
"""

def get_callables(
    obj: Union[type, object],
    kind: Literal["function", "method", "property", "data", "all"] = "all",
) -> Dict[str, Callable]:
    """
    返回类对象中所有可调用的函数。
        - 不包含魔法方法（以 '__' 开头）。
        - 对于实例对象则获取真实的类对象。

    Args:
        obj (Union[type, object]): 类对象。
        kind (Literal['function', 'method', 'property', 'data', 'all'], optional):
        可调用对象的类型。默认值为 'all'。

    Raises:
        TypeError: obj 不是类对象或实例对象。
        ValueError: kind 不是允许的值。

    Returns:
        List(Dict[str, Callable]): 包含函数名和函数对象的列表。
    """

    ...
