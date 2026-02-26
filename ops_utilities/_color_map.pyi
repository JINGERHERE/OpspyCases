#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_Color_Map.py
@Date    ：2025/7/14 21:34
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
from rich.console import Console

"""
# --------------------------------------------------
# ========== < script_unit > ==========
# --------------------------------------------------
"""

# 指定颜色库
class ColorHub:
    """print 颜色库"""

    @staticmethod
    def red(text: str) -> str:
        """红色"""
        ...

    @staticmethod
    def yellow(text: str) -> str:
        """黄色"""
        ...

    @staticmethod
    def bold(text: str) -> str: ...

# 随机颜色库
def random_color():
    """随机颜色库"""
    ...

def rich_showwarning(message, category, filename, lineno, file=None, line=None):
    # message: Warning 实例；category: Warning 类
    ...
