#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：_model_manager.py
@Date    ：2026/01/10 20:28:15
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import pandas as pd
from typing import (
    Self,
    Any,
    TypedDict,
    Optional,
    Literal,
    Unpack,
    Union,
    List,
    Tuple,
    Sequence,
)

"""
# --------------------------------------------------
# ========== < _model_manager > ==========
# --------------------------------------------------
"""

# 类型定义保持不变
class StartsDict(TypedDict, total=False):
    node: int
    element: int
    beamIntegration: int
    section: int
    uniaxialMaterial: int
    nDMaterial: int
    geomTransf: int
    timeSeries: int
    pattern: int
    region: int
    stiffnessDegradation: int
    strengthDegradation: int
    strengthControl: int
    frictionModel: int
    unloadingRule: int
    groundMotion: int
    parameter: int
    performanceFunction: int

Category = Literal[
    "node",
    "element",
    "beamIntegration",
    "section",
    "uniaxialMaterial",
    "nDMaterial",
    "geomTransf",
    "timeSeries",
    "pattern",
    "region",
    "stiffnessDegradation",
    "strengthDegradation",
    "strengthControl",
    "frictionModel",
    "unloadingRule",
    "groundMotion",
    "parameter",
    "performanceFunction",
]

class ModelManager:
    """
    ModelManager：用于管理 OpenSees 等环境中不同类别实体的整数标签（tag）。
        该类通过 Pandas DataFrame 存储数据，支持自动/手动编号、支持标签及分组、添加参数。
    """

    def __init__(
        self, include_start: bool = True, **starts: Unpack[StartsDict]
    ) -> None:
        """
        初始化 ModelManager 并设置各类别初始起始编号。

        Args:
            node (int): 节点起始编号。
            element (int): 元素起始编号。
            beamIntegration (int): 梁积分起始编号。
            section (int): 截面起始编号。
            uniaxialMaterial (int): 单轴 材料起始编号。
            nDMaterial (int): 多轴 材料起始编号。
            geomTransf (int): 几何变换起始编号。
            timeSeries (int): 时间序列起始编号。
            pattern (int): 模式起始编号。

            region (int): 区域起始编号。

            stiffnessDegradation (int): 刚度退化起始编号。
            strengthDegradation (int): 强度退化起始编号。
            strengthControl (int): 强度控制起始编号。

            frictionModel (int): 摩擦模型起始编号。
            unloadingRule (int): 卸载规则起始编号。

            groundMotion (int): 地面运动起始编号。

            parameter (int): 参数起始编号。
            performanceFunction (int): 性能函数起始编号。

        Returns:
            None
        """

        ...

    def _check_category(self, category: str) -> None:
        """
        检查输入的类别是否有效。

        Warnings:
            - 类别不在初始化时提供的类别中，会发出警告并使用默认起始值 1。
            - 类别不在指定的类别中，会发出错误警告。
        """

        ...

    def _init_counters(self) -> None:
        """
        初始化标签计数器。
        """

        ...

    def wipe(self) -> Self:
        """
        清空管理器中的所有条目，自动编号恢复至初始状态。

        Returns:
            Self: 实例本身，用于链式调用。
        """

        ...

    def _add_entry(self, category: Category, tag: int) -> None:
        """
        基础插入方法：向数据库追加一行。

        Args:
            category (Category): 类别。
            tag (int): 编号。
        """

        ...

    def _get_entry_index_by_tag(self, category: Category, tag: int) -> int:
        """
        获取指定 (category, tag) 在数据库中的行索引。

        Args:
            category (Category): 类别。
            tag (int): 编号。

        Returns:
            int: 对应的 DataFrame 索引。
        """

        ...

    def _get_entry_indexes_by_tags(
        self, category: Category, tags: Union[List[int], Tuple[int, ...]]
    ) -> List[int]:
        """
        获取指定 (category, tags) 在数据库中的行索引。

        Args:
            category (Category): 类别。
            tags (List[int]): 编号列表。

        Returns:
            List[int]: 对应的 DataFrame 索引列表。

        """

        ...

    def _get_entry_index_by_label(self, category: Category, label: str) -> int:
        """
        获取指定 (category, label) 在数据库中的行索引。

        Args:
            category (Category): 类别。
            label (str): 标签。

        Returns:
            int: 对应的 DataFrame 索引。
        """

        ...

    def _get_entry_indexs_by_group(self, category: Category, group: str) -> List[int]:
        """
        获取指定 (category, group) 在数据库中的行索引。

        Args:
            category (Category): 类别。
            group (str): 分组。

        Returns:
            List[int]: 对应的 DataFrame 索引列表。
        """

        ...

    def _check_label_unique(self, category: Category, label: str) -> None:
        """
        检查 label 是否在指定 category 中唯一。

        Args:
            category (Category): 类别。
            label (str): 待检查的标签。

        Raises:
            ValueError: 如果同一类别下已存在该 label。
        """

        ...

    def merge_data(self, df: pd.DataFrame, ignore_index: bool = True) -> Self:
        """
        合并新的 DataFrame 到当前库中。
            - 直接合并，不检查任何重复项。

        Args:
            df (pd.DataFrame): 包含新数据的 DataFrame。
            ignore_index (bool, optional): 是否忽略索引。默认 True。

        Returns:
            Self: ModelManager 实例。
        """

        ...

    def next_tag(
        self, category: Category, label: str = "", group: str = "", params: dict = {}
    ) -> int:
        """
        自动分配该类别下的下一个可用整数 tag。

        Args:
            category (Category): 类别。
            label (str, optional): 标签描述（需唯一）。
            group (str, optional): 分组名。

        Returns:
            int: 自动分配的编号。
        """

        ...

    def manual_tag(
        self,
        category: Category,
        tag: int,
        label: str = "",
        group: str = "",
        params: dict = {},
    ) -> int:
        """
        手动插入一个指定的编号。

        Args:
            category (Category): 类别。
            tag (int): 手动指定的编号。
            label (str, optional): 标签描述。
            group (str, optional): 分组名。

        Returns:
            int: 手动指定的编号。
        """

        ...

    def tag_config(
        self,
        category: Category,
        tag: int,
        label: str = "",
        group: str = "",
        params: dict = {},
    ):
        """
        配置标签、分组、参数。

        Args:
            category (Category): 类别。
            tag (int): 编号。
            label (str, optional): 标签描述。
            group (str, optional): 分组名。
            params (dict, optional): 参数字典。

        Returns:
            None
        """

        ...

    def set_label(
        self, category: Category, tag: int, label: str, warn: bool = True
    ) -> None:
        """
        设置标签。
            - 同一类别下标签唯一。
        Args:
            category (Category): 类别。
            tag (int): 编号。
            label (str): 字符串标签。
            warn (bool): 是否打印警告信息。

        Returns:
            None
        """

        ...

    def set_group(
        self,
        category: Category,
        tags: Union[int, List[int], Tuple[int, ...]],
        group: str,
        warn: bool = True,
    ) -> None:
        """
        设置分组。
            - 同一类别下标签分组可重复。
        Args:
            category (Category): 类别。
            tags (Union[int, List[int], Tuple[int, ...]]): 编号。
            group (str): 分组名称。

        Returns:
            None
        """

        ...

    def set_params(
        self, category: Category, tag: int, params: dict, warn: bool = True
    ) -> None:
        """
        设置标签的参数。
            - 需要字典格式。 `params: Dict[str, float]`。

        Args:
            category (Category): 类别。
            tag (int): 编号。
            params (Dict[str, float]): 要设置的参数。

        Returns:
            None
        """

        ...

    def rename_label(
        self, category: Category, label: str, new_label: str, warn: bool = True
    ) -> int:
        """
        为已存在的 tag 重命名 label (标签)。

        Args:
            category (Category): 类别。
            label (str): 目标标签。
            new_label (str): 新的标签。

        Returns:
            int: 成功设置的 tag 编号。
        """

        ...

    def rename_group(
        self,
        category: Category,
        group: Union[str, List[str], Tuple[str, ...]],
        new_group: str,
        warn: bool = True,
    ) -> List[int]:
        """
        给一个或多个 group 重命名为 new_group 。

        Args:
            category (Category): 类别。
            group (Union[str, List[str], Tuple[str, ...]]): 目标组名。
            new_group (str): 新的组名。

        Returns:
            List[int]: 成功设置的 tag 编号列表。
        """

        ...

    def get_tag(
        self, category: Category, label: str = "", group: str = ""
    ) -> List[int]:
        """
        查询符合条件的 Tag。

        Args:
            category (Category): 类别。
            label (str, optional): 若提供，返回该类别下唯一匹配的 编号：List[int]。
            group (str, optional): 若提供且无 label，返回该组所有 tag。

        Returns:
            List[int]: 匹配的 Tag 列表。

        Raises:
            ValueError: 同一类别下无 label 或 group 的匹配项。
        """

        ...

    def get_label(self, category: Category, tag: int) -> str:
        """
        查询指定 tag 的 标签。

        Args:
            category (Category): 类别。
            tag (int): 待查询标签的编号。

        Returns:
            str: 标签描述。

        Raises:
            ValueError: 标签不存在。
        """

        ...

    def get_group(self, category: Category, tag: int) -> str:
        """
        查询指定 tag 的分组名。

        Args:
            category (Category): 实体类别。
            tag (int): 待查询的标签编号。

        Returns:
            str: 分组名。

        Raises:
            ValueError: 标签不存在。
        """

        ...

    def get_param(self, category: Category, label: str, key: str) -> Any:
        """
        获取单个参数值。

        Args:
            category (Category): 类别。
            label (str): 标签。
            key (str): 参数名。

        Returns:
            Any: 参数内容。

        Raises:
            ValueError: 标签不存在。
            KeyError: 键不存在。
        """

        ...

    def get_params(
        self, category: Category, label: str, keys: Sequence[str]
    ) -> List[Any]:
        """
        按 key 输入顺序获取多个参数值。

        Args:
            category (Category): 类别。
            label (str): 标签。
            keys (Sequence[str]): 参数名列表。

        Returns:
            List[Any]: 按 key 输入顺序的参数值列表。

        Raises:
            ValueError: 标签不存在。
        """

        ...

    def remove_ctg(self, category: Literal[Literal["all"], Category] = "all") -> None:
        """
        清空对应类别的所有数据。

        Args:
            category (Literal[Literal['all'], Category]): 类别。
                - 'all': 清空所有数据。
                - 其他类别: 清空对应类别的所有数据。

        Returns:
            None: 不返回值。
        """

        ...

    def remove_tag(
        self,
        category: Category,
        tags: Union[None, Literal["all"], int, List[int], Tuple[int, ...]] = None,
    ) -> None:
        """
        根据 tag 删除指定条目。

        Args:
            category (Category): 类别。
            tags (Union[None, Literal['all'], int, List[int], Tuple[int, ...]]): 待删除的编号。
                - 'all': 删除该类别所有数据。
                - 其他: 删除指定编号的条目。

        Returns:
            None: 不返回值。
        """

        ...

    def remove_label(
        self, category: Category, label: Union[None, Literal["all"], str] = None
    ) -> None:
        """
        根据 label 删除指定条目。
            - 标签删除伴随删除该行的所有参数。
            - 修改标签请使用 `set_label` 方法。

        Args:
            category (Category): 类别。
            label (Union[None, Literal['all'], str]): 待删除的标签。

        Returns:
            None: 不返回值。
        """

        ...

    def remove_group(
        self, category: Category, group: Union[None, Literal["all"], str] = None
    ) -> None:
        """
        根据 group 删除指定条目。
            - 分组删除伴随删除该分组下的所有标签记录。
            - 修改分组请使用 `set_group` 方法。

        Args:
            category (Category): 类别。
            group (str): 待删除的分组名。

        Returns:
            None: 不返回值。
        """

        ...

    def __getitem__(self, key): ...
    def __getattr__(self, name): ...
    def __repr__(self) -> str: ...
    def _repr_html_(self) -> Optional[str]: ...
    def __len__(self): ...
    def __iter__(self): ...
