"""
Opspy Utilities
===============

This package contains utility functions and classes for creating OpenSeespy models and  data post-processing.

Modules
-------
    - pre :
        Pre-processing utilities.
    - post :
        Post-processing utilities.

Example
-------
    >>> import ops_utilities as opsu
"""

# --------------------------------------------------
# ========== < 全局方法 > ==========
# --------------------------------------------------
from . import pre
from . import post

__all__ = ["pre", "post"]

from ._color_map import rich_showwarning, random_color

__all__ += ["rich_showwarning", "random_color"]

from ._get_callables import get_callables

__all__ += ["get_callables"]

# --------------------------------------------------
# ========== < 主文件运行 > ==========
# --------------------------------------------------

if __name__ == "__main__":
    pass
