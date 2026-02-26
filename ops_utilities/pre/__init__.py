"""
Pre-processing utilities
===============

This package contains utility functions and classes for creating OpenSeespy models.

Modules
-------
    - ReBarHub, ConcHub :
        Rebar and concrete material hub.
    - Mander :
        Mander calculator.
    - AutoTransf :
        Auto geometry transformation for creating OpenSeespy elements.
    - ModelManager :
        Manager for creating OpenSeespy models.
    - OpenSeesEasy :
        Easy-to-use interface for creating OpenSeespy models.
"""

# --------------------------------------------------
# ========== < 全局方法 > ==========
# --------------------------------------------------

__all__ = []

from ._material_hub import ReBarHub, ConcHub

__all__ += ["ReBarHub", "ConcHub"]

from ._mander_calculater import Mander

__all__ += ["Mander"]

from ._auto_geomTransf import AutoTransf, get_angle

__all__ += ["AutoTransf", "get_angle"]

from ._model_manager import ModelManager

__all__ += ["ModelManager"]

from ._ops_easy import OpenSeesEasy

__all__ += ["OpenSeesEasy"]

from ._utilities import HistoryCounter, IntervalCounter

__all__ += ["HistoryCounter", "IntervalCounter"]


# --------------------------------------------------
# ========== < 主文件运行 > ==========
# --------------------------------------------------

if __name__ == "__main__":
    pass
