#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：ModelUtilities.py
@Date    ：2026/01/26 19:40:43
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import opstool as opst
import ops_utilities as opsu

"""
# --------------------------------------------------
# ========== < ModelUtilities > ==========
# --------------------------------------------------
"""

# 全局单位系统
UNIT = opst.pre.UnitSystem(
    length='m', force='kn', time='sec'
    )

# 全局模型管理器
MM = opsu.pre.ModelManager(include_start=False)

"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""

if __name__ == "__main__":
    
    print(f'全局单位：\n{UNIT}\n')
    print(f'全局模型管理器：\n{MM}\n')
