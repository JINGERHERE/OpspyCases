#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_determine_frame_states.py
@Date    ：2026/01/19 14:27:50
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
# ========== < _determine_frame_states > ==========
# --------------------------------------------------
"""

class FrameStates:
    def __init__(
        self,
        odb_tag: Union[int, str],
        ele_tag: int,
        lazy_load: bool = False,
        print_info: bool = True,
    ):
        """
        基于 单元状态 的判断方法

        Args:
            odb_tag (Union[int, str]): opstool 数据库标签。
            ele_tag (int): 单元编号。
            integ (int): 单元积分点。
            lazy_load (bool): 是否按需加载数据。默认值为 False。
            print_info (bool): 是否打印加载信息。默认值为 True。

        Returns:
            None

        Examples:
            >>> FS = FrameStates(
            >>>     odb_tag=CASE, resp_type='disp'
            >>>     )
        """

        ...

    def get_data_local(
        self,
        dofs: Union[
            Literal["all"],
            Literal[
                "FX1",
                "FX2",
                "FY1",
                "FY2",
                "FZ1",
                "FZ2",
                "MX1",
                "MX2",
                "MY1",
                "MY2",
                "MZ1",
                "MZ2",
            ],
            List[
                Literal[
                    "FX1",
                    "FX2",
                    "FY1",
                    "FY2",
                    "FZ1",
                    "FZ2",
                    "MX1",
                    "MX2",
                    "MY1",
                    "MY2",
                    "MZ1",
                    "MZ2",
                ]
            ],
        ] = "all",
    ) -> pd.DataFrame:
        """
        获取单元 local 响应 数据。
            localForces  (time, eleTags, localDofs)

        Args:
            dofs (类型): 要获取的 localDofs。

        Returns:
            pd.DataFrame: 返回对应数据表。

        Examples:
            >>> result = self.get_data_local("all")
            >>> print(result)
        """

        ...

    def get_steps_local(
        self,
        dofs: Literal[
            "FX1",
            "FX2",
            "FY1",
            "FY2",
            "FZ1",
            "FZ2",
            "MX1",
            "MX2",
            "MY1",
            "MY2",
            "MZ1",
            "MZ2",
        ],
        stages: Union[int, float, list, tuple, np.ndarray],
        warn: bool = True,
    ) -> List[Optional[int]]:
        """
        根据给定的状态值，返回对应分析步索引列表。

        Args:
            dofs (Literal['FX1','FX2','FY1','FY2','FZ1','FZ2','MX1','MX2','MY1' 'MY2','MZ1','MZ2']): 要获取的 localDofs。
            stages (Union[int, float, list, tuple, np.ndarray]): 要获取的状态值。
            warn (bool, optional): 是否打印警告信息。默认值为 True。

        Returns:
            List[Optional[int]]: 返回对应分析步索引列表。

        Examples:
            >>> result = self.get_steps_local("FX1", stages=[-0.012, -0.0033, -0.2])
            >>> print(result)
            >>> [1, 100, 500, None]
        """

        ...

    def get_data_basic(
        self,
        data_type: Literal["basicForces", "basicDeformations", "plasticDeformation"],
        dofs: Union[
            Literal["all"],
            Literal["MY1", "MY2", "MZ1", "MZ2", "N", "T"],
            List[Literal["MY1", "MY2", "MZ1", "MZ2", "N", "T"]],
        ] = "all",
    ) -> pd.DataFrame:
        """
        获取单元 basic 响应 数据。
            - basicForces          (time, eleTags, basicDofs)
            - basicDeformations    (time, eleTags, basicDofs)
            - plasticDeformation   (time, eleTags, basicDofs)

        Args:
            data_type (Literal['basicForces', 'basicDeformations', 'plasticDeformation']): 要获取的数据类型。
            dofs (Union[Literal['all'], Literal['MY1', 'MY2', 'MZ1', 'MZ2', 'N', 'T'], List[Literal['MY1', 'MY2', 'MZ1', 'MZ2', 'N', 'T']]]): 要获取的 basicDofs。

        Returns:
            pd.DataFrame: 返回对应数据表。

        Examples:
            >>> result = self.get_data_basic("basicForces", "all")
            >>> print(result)
        """

        ...

    def get_steps_basic(
        self,
        data_type: Literal["basicForces", "basicDeformations", "plasticDeformation"],
        dofs: Literal["MY1", "MY2", "MZ1", "MZ2", "N", "T"],
        stages: Union[int, float, list, tuple, np.ndarray],
        warn: bool = True,
    ) -> List[Optional[int]]:
        """
        根据给定的状态值，返回对应分析步索引列表。

        Args:
            data_type (Literal['basicForces', 'basicDeformations', 'plasticDeformation']): 要获取的数据类型。
            dofs (Literal['MY1', 'MY2', 'MZ1', 'MZ2', 'N', 'T']): 要获取的 basicDofs。
            stages (Union[int, float, list, tuple, np.ndarray]): 要获取的状态值。
            warn (bool, optional): 是否打印警告信息。默认值为 True。

        Returns:
            List[Optional[int]]: 返回对应分析步索引列表。

        Examples:
            >>> result = self.get_steps_basic(
            >>>     data_type="basicDeformations", dofs="MY1",
            >>>     stages=[-0.012, -0.0033, -0.2]
            >>> )
            >>> print(result)
            >>> [1, 100, 500, None]
        """

        ...

    def get_data_sec(
        self,
        data_type: Literal["sectionForces", "sectionDeformations"],
        sec_points: int,
        dofs: Union[
            Literal["all"],
            Literal["MY", "MZ", "N", "T", "VY", "VZ"],
            List[Literal["MY", "MZ", "N", "T", "VY", "VZ"]],
        ] = "all",
    ):
        """
        获取单元 section 响应 数据。
            - sectionForces        (time, eleTags, secPoints, secDofs)
            - sectionDeformations  (time, eleTags, secPoints, secDofs)

        Args:
            data_type (Literal['sectionForces', 'sectionDeformations']): 要获取的数据类型。
            sec_points (int): 要获取的单元积分点索引。
            dofs (Union[Literal['all'], Literal['MY', 'MZ', 'N', 'T', 'VY', 'VZ'], List[Literal['MY', 'MZ', 'N', 'T', 'VY', 'VZ']]]): 要获取的 secDofs。

        Returns:
            pd.DataFrame: 返回对应数据表。

        Examples:
            >>> result = self.get_data_sec(
            >>>     data_type="sectionForces", sec_points=1,
            >>>     dofs="all"
            >>> )
            >>> print(result)
        """

        ...

    def get_steps_sec(
        self,
        data_type: Literal["sectionForces", "sectionDeformations"],
        sec_points: int,
        dofs: Literal["MY", "MZ", "N", "T", "VY", "VZ"],
        stages: Union[int, float, list, tuple, np.ndarray],
        warn: bool = True,
    ) -> List[Optional[int]]:
        """
        根据给定的状态值，返回对应分析步索引列表。

        Args:
            data_type (Literal['sectionForces', 'sectionDeformations']): 要获取的数据类型。
            sec_points (int): 要获取的单元积分点索引。
            dofs (Literal['MY', 'MZ', 'N', 'T', 'VY', 'VZ']): 要获取的 secDofs。
            stages (Union[int, float, list, tuple, np.ndarray]): 要获取的状态值。
            warn (bool, optional): 是否打印警告信息。默认值为 True。

        Returns:
            List[Optional[int]]: 返回对应分析步索引列表。

        Examples:
            >>> result = self.get_steps_sec(
            >>>     data_type="sectionDeformations", sec_points=1,
            >>>     dofs="MY", stages=[-0.012, -0.0033, -0.2]
            >>> )
            >>> print(result)
            >>> [1, 100, 500, None]
        """

        ...

    def get_data_locs(
        self,
        point: int,
        locs: Union[
            Literal["all"],
            Literal["X", "Y", "Z", "alpha"],
            List[Literal["X", "Y", "Z", "alpha"]],
        ] = "all",
    ) -> pd.DataFrame:
        """
        获取单元 section 位置 数据。
            - sectionLocs  (time, eleTags, secPoints, locs)

        Args:
            point (int): 要获取的 secPoints。
            locs (Union[Literal['all'], Literal['X', 'Y', 'Z', 'alpha'], List[Literal['X', 'Y', 'Z', 'alpha']]]): 要获取的 locs。

        Returns:
            pd.DataFrame: 返回对应数据表。

        Examples:
            >>> result = self.get_data_locs(
            >>>     point=1, locs="all"
            >>> )
            >>> print(result)
        """

        ...
