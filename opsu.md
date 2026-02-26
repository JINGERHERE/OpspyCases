# Openseespy 建模及后处理实用工具集

`ops_utilities` 包含 `openseepy-3.7` 建模和后处理的实用工具实例。

### 环境依赖需求
```
python: >=3.12, <=3.13
openseespy: >=3.7
opstool: >= 1.0.26
```

### 工具集文件结构
```
ops_utilities/
   ├── pre/
   │   ├── ReBarHub
   │   ├── ConcHub
   │   ├── Mander
   │   ├── AutoTransf
   │   ├── ModelManager
   │   └── OpenSeesEasy
   ├── post/
   │   ├── NodalStates
   │   ├── TrussStates
   │   ├── SecMatStates
   │   └── FrameStates
```

### 工具集实例简介
实例的具体应用方法参考工具集内的`*.pyi`文件。

   - **pre/**
     - **ReBarHub**: 钢筋材料基础参数库。
     - **ConcHub**: 混凝土材料基础参数库。
     - **Mander**: Mander本构参数计算器。
     - **AutoTransf**: 自动化转换单元局部坐标（自动生成坐标转换所需的 tag）。
     - **ModelManager**: ops 模型 tag 管理工具，辅助管理创建模型过程中的 tag。
     - **OpenSeesEasy**: 为 ops 所有（不确定有没有漏）需要指定 tag 的命令自动生成 tag，建模过程无需考虑（定义） tag，每一条命令将直接返回当前 tag 值。该实例内部为纤维截面提供了可视化方法，可显示通过当前实例定义的纤维截面。

   - **post/**
   所有实例基于 opstool 保存的数据库实现。所有实例支持输入节点或单元响应状态的阈值（支持多阶段阈值），并返回对应的分析步。
     - **NodalStates**: 指定获取 ops 模型中节点的状态（位移、速度、加速度、等）。
     - **TrussStates**: 指定获取 ops 模型中桁架单元的状态（应变、应力、轴向变形、轴向力）。
     - **SecMatStates**: 指定获取 ops 模型中截面材料的状态（截面变形，截面力、具体材料纤维的应力和应变）。
     - **FrameStates**: 指定获取 ops 模型中框架单元的状态（local、basic、section、sectionLocs）。

### 参考
- [Openseespy官方文档](https://openseespydoc.readthedocs.io)
- opstool: [文档](https://opstool.readthedocs.io/en/latest/index.html)，[仓库](https://github.com/yexiang92/opstool)
- [MCAnalisis仓库](https://github.com/Penghui-Zhang/MCAnalysis)
