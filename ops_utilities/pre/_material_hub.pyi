#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_material_hub.py
@Date    ：2026/01/12 01:27:48
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

from typing import Literal, Union

"""
# --------------------------------------------------
# ========== < _material_hub > ==========
# --------------------------------------------------
"""

# 指定输入的类型
conc_types = Literal[
    "C15",
    "C20",
    "C25",
    "C30",
    "C35",
    "C40",
    "C45",
    "C50",
    "C55",
    "C60",
    "C65",
    "C70",
    "C75",
    "C80",
]
rebar_types = Literal[
    "HPB300", "HRB335", "HRB400", "HRBF400", "RRB400", "HRB500", "HRBF500"
]

def _get_val(type: str) -> float:
    """
    获取指定类型字符串中的数字部分。
        - 混凝土类型：[C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]
        - 钢筋类型：[HPB300, HRB335, HRB400, HRBF400, RRB400, HRB500, HRBF500]

    Args:
        type (str): 输入指定类型的字符串。

    Returns:
        float: 字符串中的数字部分。

    Raises:
        ValueError: 输入的字符串不是指定类型。

    Examples:
        >>> result = _get_val("C40")
        >>> print(result)
        >>> 40.0

    """

    ...

class ReBarHub:
    """
    钢筋参数库
        - :get_fyk: 钢筋 屈服强度 `标准值` (MPa)
        - :get_fstk: 钢筋 极限强度 `标准值` (MPa)
        - :get_fy: 钢筋 屈服强度 `设计值` (MPa)
        - :get_Es: 钢筋弹性模量 (MPa)
    """

    @classmethod
    def get_fyk(cls, type: Union[str, rebar_types]) -> float:
        """
        钢筋 屈服强度 `标准值` (MPa)
            - rebar_types: [HPB300, HRB335, HRB400, HRBF400, RRB400, HRB500, HRBF500]

        Args:
            type (Union[str, rebar_types]): 钢筋类型

        Returns:
            float: 钢筋 屈服强度 标准值 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ReBarHub.get_fyk("HRB400")
            >>> print(result)
            >>> 400.0
        """

        ...

    @classmethod
    def get_fstk(cls, type: Union[str, rebar_types]) -> float:
        """
        钢筋 极限强度 `标准值` (MPa)
            - rebar_types: [HPB300, HRB335, HRB400, HRBF400, RRB400, HRB500, HRBF500]

        Args:
            type (Union[str, rebar_types]): 钢筋类型

        Returns:
            float: 钢筋 极限强度 标准值 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ReBarHub.get_fstk("HRB400")
            >>> print(result)
            >>> 540.0
        """

        ...

    @classmethod
    def get_fy(cls, type: Union[str, rebar_types]) -> float:
        """
        钢筋 屈服强度 `设计值` (MPa)
            - rebar_types: [HPB300, HRB335, HRB400, HRBF400, RRB400, HRB500, HRBF500]

        Args:
            type (Union[str, rebar_types]): 钢筋类型

        Returns:
            float: 钢筋 屈服强度 设计值 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ReBarHub.get_fy("HRB400")
            >>> print(result)
            >>> 360.0
        """

        ...

    @classmethod
    def get_Es(cls, type: Union[str, rebar_types]) -> float:
        """
        钢筋弹性模量 (MPa)
            - rebar_types: [HPB300, HRB335, HRB400, HRBF400, RRB400, HRB500, HRBF500]

        Args:
            type (Union[str, rebar_types]): 钢筋类型

        Returns:
            float: 钢筋 弹性模量 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ReBarHub.get_Es("HRB400")
            >>> print(result)
            >>> 2.e5
        """

        ...

class ConcHub:
    """
    混凝土参数库
        - :get_Ec: 混凝土 弹性模量 (MPa)
        - :get_G: 混凝土 剪切模量 (MPa)
        - :get_fcuk: 立方体混凝土 抗压强度 `标准值` (MPa)
        - :get_fck: 棱柱混凝土 抗压强度 `标准值` (MPa)
        - :get_fc: 棱柱混凝土 抗压强度 `设计值` (MPa)
        - :get_ftk: 棱柱混凝土 轴心抗拉强度 `标准值` (MPa)
        - :get_ft: 棱柱混凝土 轴心抗拉强度 `设计值` (MPa)
    """

    @classmethod
    def _alpha_c1(cls, fcuk: float) -> float:
        """
        棱柱体混凝土抗压强度与立方体混凝土抗压强度之比: aplha_c1

        Args:
            fcuk (float): 立方体混凝土抗压强度 标准值 (MPa)

        Returns:
            float: 棱柱体混凝土抗压强度与立方体混凝土抗压强度之比: aplha_c1

        Examples:
            >>> result = ConcHub._alpha_c1(50.0)
            >>> print(result)
            >>> 0.76
        """

        ...

    @classmethod
    def _alpha_c2(cls, fcuk: float) -> float:
        """
        高强混凝土脆性折破坏减系数：alpha_c2

        Args:
            fcuk (float): 立方体混凝土抗压强度 标准值 (MPa)

        Returns:
            float: 高强混凝土脆性折破坏减系数：alpha_c2

        Examples:
            >>> result = ConcHub._alpha_c2(40.0)
            >>> print(result)
            >>> 1.00
        """

        ...

    @classmethod
    def get_Ec(cls, type: Union[str, conc_types]) -> float:
        """
        混凝土的弹性模量 (MPa)
            - type: [C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]

        Args:
            type (Union[str, conc_types]): 混凝土类型

        Returns:
            float: 混凝土的弹性模量 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ConcHub.get_Ec("C40")
            >>> print(result)
            >>> 3.26e4
        """

        ...

    @classmethod
    def get_G(cls, type: Union[str, conc_types]) -> float:
        """
        混凝土的剪切模量 (MPa)
            - type: [C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]

        Args:
            type (Union[str, conc_types]): 混凝土类型

        Returns:
            float: 混凝土的剪切模量 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ConcHub.get_G("C40")
            >>> print(result)
            >>> 1.397e4
        """

        ...

    @classmethod
    def get_fcuk(cls, type: Union[str, conc_types]) -> float:
        """
        立方体混凝土 抗压强度 `标准值` (MPa)
            - type: [C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]

        Args:
            type (Union[str, conc_types]): 混凝土类型

        Returns:
            float: 立方体混凝土 抗压强度 标准值 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ConcHub.get_fcuk("C40")
            >>> print(result)
            >>> 40.0
        """

        ...

    @classmethod
    def get_fck(cls, type: Union[str, conc_types]) -> float:
        """
        棱柱混凝土 轴心抗压强度 `标准值` (MPa)
            - type: [C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]

        Args:
            type (Union[str, conc_types]): 混凝土类型

        Returns:
            float: 棱柱混凝土 轴心抗压强度 标准值 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ConcHub.get_fck("C40")
            >>> print(result)
            >>> 26.75
        """

        ...

    @classmethod
    def get_fc(cls, type: Union[str, conc_types]) -> float:
        """
        棱柱混凝土 轴心抗压强度 `设计值` (MPa)
            - type: [C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]

        Args:
            type (Union[str, conc_types]): 混凝土类型

        Returns:
            float: 棱柱混凝土 轴心抗压强度 设计值 (MPa)

        Examples:
            >>> result = ConcHub.get_fc("C40")
            >>> print(result)
            >>> 19.10
        """

        ...

    @classmethod
    def get_ftk(cls, type: Union[str, conc_types]) -> float:
        """
        棱柱混凝土 轴心抗拉强度 `标准值` (MPa)
            - type: [C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]

        Args:
            type (Union[str, conc_types]): 混凝土类型

        Returns:
            float: 棱柱混凝土 轴心抗拉强度 标准值 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ConcHub.get_ftk("C40")
            >>> print(result)
            >>> 2.39
        """

        ...

    @classmethod
    def get_ft(cls, type: Union[str, conc_types]) -> float:
        """
        棱柱混凝土 轴心抗拉强度 `设计值` (MPa)
            - type: [C15, C20, C25, C30, C35, C40, C45, C50, C55, C60, C65, C70, C75, C80]

        Args:
            type (Union[str, conc_types]): 混凝土类型

        Returns:
            float: 棱柱混凝土 轴心抗拉强度 设计值 (MPa)

        Raises:
            ValueError: 输入的 {type} 不是指定类型。

        Examples:
            >>> result = ConcHub.get_ft("C40")
            >>> print(result)
            >>> 1.707
        """

        ...
