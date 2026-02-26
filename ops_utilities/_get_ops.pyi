#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_get_ops.py
@Date    ：2026/02/09 19:41:20
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from types import ModuleType

"""
# --------------------------------------------------
# ========== < _get_ops > ==========
# --------------------------------------------------
"""

class OPS:
    """
    动态获取 openseespy.opensees module.
    """

    @classmethod
    def get_methods(cls) -> set[str]:
        """获取 openseespy.opensees module 中的所有方法."""

        ...

    @classmethod
    def get_model(cls) -> ModuleType:
        """获取 openseespy.opensees module."""

        ...
