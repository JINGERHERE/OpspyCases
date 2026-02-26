#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_determine_material_states.py
@Date    ：2025/7/16 17:47
@IDE     ：PyCharm
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from typing import Literal, Union, Optional, List, Dict, Tuple
import opstool as opst

"""
# --------------------------------------------------
# ========== < _determine_material_states > ==========
# --------------------------------------------------
"""

class SecMatStates:
    """
    基于 纤维截面 状态判断方法
        - get_data(): 获取截面 secDefo, secForce 数据。
        - get_steps(): 基于给定数据类型 获取对应状态的分析步。
        - get_data_mat(): 获取截面材料 Strains, Stresses 数据。
        - get_steps_mat(): 基于给定数据类型 获取对应状态的分析步。
        - get_combined_steps(): 基于多种材料 获取对应状态的分析步。
    """

    def __init__(
        self,
        odb_tag: Union[int, str],
        ele_tag: int,
        integ: int,
        lazy_load: bool = False,
        print_info: bool = True,
    ) -> None:
        """
        基于 纤维截面 材料 应变/应力 状态判断方法

        Args:
            odb_tag (Union[int, str]): opstool 数据库标签。
            ele_tag (int): 单元编号。
            integ (int): 单元积分点。
            lazy_load (bool): 是否按需加载数据。默认值为 False。
            print_info (bool): 是否打印加载信息。默认值为 True。

        Returns:
            None

        Examples:
            >>> SMS = SecMatStates(
            >>>     resp_data=opst.post.get_element_responses(odb_tag=tag, ele_type="FiberSection"),
            >>>    ele_tag=1, integ=1
            >>>     )
        """

        ...

    def get_data(
        self,
        data_type: Union[
            Literal["all"],
            Literal["secDefo", "secForce"],
            List[Literal["secDefo", "secForce"]],
        ],
        dofs: Literal["P", "Mz", "My", "T"],
    ) -> pd.DataFrame:
        """
        获取 截面 对应状态 的数据。

        Args:
            data_type (Union[
                Literal['all'],
                Literal['secDefo', 'secForce'],
                List[Literal['secDefo', 'secForce']]
                ]): 数据类型。
            dofs (Literal['P', 'Mz', 'My', 'T']): 自由度。

        Returns:
            pd.DataFrame: 对应状态 的数据。

        Examples:
            >>> SS = SecStates(odb_tag=CASE, ele_tag=1, dofs='P')
            >>> SS.get_resp(data_type='secDefo')
        """

        ...

    def get_steps(
        self,
        data_type: Literal["secDefo", "secForce"],
        dofs: Literal["P", "Mz", "My", "T"],
        stages: Union[int, float, list, tuple, np.ndarray],
        warn: bool = True,
    ) -> List[Optional[int]]:
        """
        根据给定 '数据' 的 secDefo/secForce 划分阶段，返回每个阶段中最小的 step 值。
            - 判断时不包含首行的 0 数据。
            - stages 要求维度为 1， 且单调递增，非递增则自动排序。
            - 如果某个阶段没有数据，返回 None。

        Args:
            data_type (Literal['secDefo', 'secForce']): 数据类型。
            stages (Union[int, float, list, tuple, np.ndarray]): 分割点。可以是单个数值，也可以是序列。

        Returns:
            List[Optional[int]]: 每个阶段中最小的 step 值列表，无法到达的阶段为 None。

        Examples:
            >>> SS = SecStates(resp_data, 1, 'P')
            >>> SS.get_steps(data_type='secDefo', stages=0.002)
            >>> [1, 100]
            >>> SS.get_steps(data_type='secForce', stages=[-0.012, -0.0033, -0.2])
            >>> # [start, stage1, stage2, ..., end]
            >>> [1, 120, 600, None]
        """

        ...

    def get_data_mat(
        self,
        mat_tag: int,
        data_type: Literal["Strains", "Stresses"],
        points: Union[Literal["all"], int, List[int], Tuple[int, ...]] = "all",
        warn: bool = True,
    ) -> pd.DataFrame:
        """
        获取截面纤维点的 '应变 / 应力' 数据

        Args:
            mat_tag (int): 材料标签。
            data_type (Literal['Strains', 'Stresses']): 数据类型。
            points (Union[Literal['all'], int, List[int], Tuple[int, ...]]): 指定的纤维点 ID。
            warn (bool, optional): 是否打印警告信息。Default: True.

        Returns:
            pd.DataFrame: 截面目标纤维点的 '应变 / 应力' 数据。

        Raises:
            ValueError: 输入的纤维点 ID 无匹配项。

        Examples:
            >>> SMS = SecMatStates(resp_data, 1, 5)
            >>> strain = SMS.get_data_mat(mat_tag=1, data_type='Strains', points=(100, 120))
            >>> print(strain)
        """

        ...

    def get_steps_mat(
        self,
        mat_tag: int,
        data_type: Literal["Strains", "Stresses"],
        stages: Union[int, float, list, tuple, np.ndarray],
        warn: bool = True,
    ) -> List[Optional[int]]:
        """
        根据给定 '数据' 的 Strains/Stresses 划分阶段，返回每个阶段中最小的 step 值。
            - 判断时不包含首行的 0 数据。
            - stages 要求维度为 1， 且单调递增，非递增则自动排序。
            - 如果某个阶段没有数据，返回 None。

        Args:
            mat_tag (int): 材料编号。
            data_type (Literal['Strains', 'Stresses']): 数据类型。
            stages (Union[int, float, list, tuple, np.ndarray]): 分割点。可以是单个数值，也可以是序列。
            warn (bool): 是否打印警告信息。Default: True.

        Returns:
            List[Optional[int]]: 每个阶段中最小的 step 值列表，无法到达的阶段为 None。

        Examples:
            >>> SMS = SecMatStates(resp_data, 1, 5)
            >>> SMS.get_steps(mat_tag=1, data_type='Strains', stages=0.002)
            >>> [1, 100]
            >>> SMS.get_steps(mat_tag=1, data_type='Strains', stages=[-0.012, -0.0033, -0.2])
            >>> # [start, stage1, stage2, ..., end]
            >>> [1, 120, 600, None]
        """

        ...

    def get_combined_steps_mat(
        self,
        mat_config: Dict[int, Union[int, float, list, tuple, np.ndarray]],
        data_type: Literal["Strains", "Stresses"],
        warn: bool = True,
    ) -> List[Optional[int]]:
        """
        根据给定的多个材料 Strains/Stresses 阶段配置，返回所有材料种每个阶段的最小 step 值。
            - 判断时不包含首行的 0 数据。
            - 所有材料的 stages 长度必须一致。
            - 其余要求同 get_steps() 方法。

        Args:
            mat_config (Dict[int, Union[int, float, list, tuple, np.ndarray]]): 材料配置字典。
            data_type (Literal['Strains', 'Stresses']): 数据类型。

        Returns:
            List[Optional[int]]: 所有材料种每个阶段的最小 step 值列表，无法到达的阶段为 None。

        Raises:
            ValueError: 若所有材料的 stages 长度不一致时，抛出 ValueError 异常。

        Examples:
            >>> SMS = SecMatStates(resp_data, 1, 5)
            >>> SMS.get_combined_steps(
            >>>     mat_config={
            >>>         1: (0.002, 0.1, 0.5),
            >>>         2: (-0.012, -0.0033, -0.2)
            >>>         },
            >>>     data_type='Strains'
            >>>     )
            >>> # [start, stage1, stage2, ..., end]
            >>> [1, 100, 500, None]
        """

        ...

    def plot_sec(
        self,
        SEC: opst.pre.section.FiberSecMesh,
        data_type: Literal["Strains", "Stresses"],
        step: int,
        thresholds: Dict[int, Union[list, tuple]],
        cmap: str = "coolwarm",
        ax: Optional[Axes] = None,
        fontsize: int = 12,
    ) -> Tuple[Axes, Colorbar]:
        """
        绘制截面材料状态图。

        Args:
            SEC (opst.pre.section.FiberSecMesh): 纤维截面对象。
            data_type (Literal['Strains', 'Stresses']): 数据类型。
            step (int): 显示哪一分析步。
            thresholds (Dict[int, Union[list, tuple]]): 材料状态阈值字典。
                - 字典内 - 键: 材料标签。
                - 字典内 - 值: 材料状态阈值。
                - 阈值状态只能两个值，且必须单调递增。
                - `{mat_tag: (负向 threshold, 正向 threshold)}`
            cmap (str, optional): 颜色映射。默认值为 'coolwarm'。
            ax (Optional[Axes], optional): 坐标轴对象。默认值为 None。
            fontsize (int, optional): 标签字体大小。默认值为 12。

        Returns:
            Tuple(Axes, Colorbar): 坐标轴对象和颜色条对象。
                - Axes: 坐标轴对象。
                - Colorbar: 颜色条对象。
        """

        ...
