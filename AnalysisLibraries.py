#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
@File    ：AnalysisLibraries.py
@Date    ：2026/01/17 18:17:47
@IDE     ：Visual Studio Code
@Author  ：Wang Jing
@Email   ：wangjing_zjnb@outlook.com
# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Iterable, Literal, Optional, Union, Any, List, Dict, Tuple
import warnings

import openseespy.opensees as ops
import opstool as opst


"""
# --------------------------------------------------
# ========== < AnalysisLibraries > ==========
# --------------------------------------------------
"""

"# ===== ===== ===== ===== #"
"# < Gravity Analysis > #"
"# ===== ===== ===== ===== #"
class GravityAnalysis:
    
    def __init__(self, ODB: opst.post.CreateODB) -> None:
        
        """
        Gravity analysis class
        
        Args:
            ODB: CreateODB object: 已创建的 opstool 数据库对象。
        """
        
        self.ODB = ODB
    
    def analyze(self, Nsteps: int = 10) -> None:
        
        """
        Gravity analysis.
        
        Args:
            Nsteps (int, optional): 分析步数。 Defaults to 10.
        
        Returns:
            None
        """
        
        # 重力分析系统配置
        ops.system('BandGeneral')  # 求解器类型，BandGeneral适用于带状矩阵，如梁柱结构
        ops.constraints('Transformation')  # 约束处理方法，Transformation，适用于大多数问题
        ops.numberer('RCM')  # 节点编号方法，RCM (Reverse Cuthill-McKee)算法，可以减少带宽
        ops.test('NormDispIncr', 1.0e-12, 15, 3)  # 收敛测试:位移增量范数,容差1.0e-12,最大迭代数15
        ops.algorithm('Newton')  # 解算法，Newton-Raphson法，适用于大多数非线性问题
        ops.integrator('LoadControl', 1 / Nsteps)  # Nsteps与步长的乘积一定要为1，代表施加一倍荷载，乘积为2代表施加两倍荷载
        ops.analysis('Static')  # 分析模式：Static，Transient。（静态，动态）

        # 执行重力分析
        ok = 0
        for i in range(Nsteps):
            ok = ops.analyze(1)
            if ok < 0:
                raise RuntimeError(f"Gravity Analyze Failure in Step {i}! ")
            
            self.ODB.fetch_response_step()


"# ===== ===== ===== ===== #"
"# < Static Analysis > #"
"# ===== ===== ===== ===== #"
class StaticAnalysis:
    
    def __init__(self, pattern_tag: int, ODB: opst.post.CreateODB) -> None:
        
        """
        Static analysis class
            - For Pushover or Cycle.

        Args:
            pattern_tag (int): 已经定义的荷载模式标签。
            ODB: CreateODB object: 已创建的 opstool 数据库对象。

        Returns:
            None
        """
        
        self.pattern_tag = pattern_tag
        self.ODB = ODB

    def analyze(
        self,
        ctrl_node: int, dof: Union[int, Literal[1, 2, 3, 4, 5, 6]],
        targets: Union[list, tuple, np.ndarray], max_step: float,
        ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Static analysis function
            - For Pushover or Cycle.

        Args:
            ctrl_node (int): 被控制位移的 节点标签。
            dof (Union[int, Literal[1, 2, 3, 4, 5, 6]]): 节点 被控制位移的方向。
            max_step (float): 分析时的最大步长。
            targets (Union[list, tuple, np.ndarray]): 位移路径。

        Returns:
            (np.ndarray, np.ndarray): A tuple containing：
                - node_disp (np.ndarray): Displacement path.
                - force_lambda (np.ndarray): Force path.
        """

        # 分析配置
        ops.wipeAnalysis() # 清空已有的分析配置
        ops.system('BandGeneral')
        ops.constraints('Transformation')
        ops.numberer('RCM')

        # 智能分析参数
        analysis = opst.anlys.SmartAnalyze(
            analysis_type="Static",
            testType='EnergyIncr', # 收敛判断类型
            testTol=1e-10,  # 收敛容差
            minStep=1e-6,  # 最小收敛步长
            tryAddTestTimes=True, # 不收敛时尝试增加最大迭代次数
            # 默认：最后一次不收敛范数 normTol<1000，再尝试50次 testIterTimesMore=[50]
            tryAlterAlgoTypes=True,  # 尝试不同test方法
            # 默认：algoTypes=[40, 10, 20, 30, 50, 60, 70, 90]
            tryLooseTestTol=True,  # 尝试宽松的收敛条件
            # 默认：looseTestTolTo = 1000 * testTol
            debugMode=False,  # False for progress bar, True for debug info
            )

        # 分析路径分割
        segs = analysis.static_split(targets=targets, maxStep=max_step)

        # fetch the status of the current pattern before analysis
        force_lambda: Union[list, float] = [ops.getLoadFactor(self.pattern_tag)]
        node_disp: Union[list, float] = [ops.nodeDisp(ctrl_node, dof)]

        # 执行分析
        for seg in segs:
            ok = analysis.StaticAnalyze(node=ctrl_node, dof=dof, seg=seg)
            if ok < 0:
                analysis.close() # 关闭分析
                raise RuntimeError("Analysis failed")

            # Fetch response
            self.ODB.fetch_response_step()
            force_lambda.append(ops.getLoadFactor(self.pattern_tag))
            node_disp.append(ops.nodeDisp(ctrl_node, dof))

        analysis.close() # 关闭分析

        return np.array(node_disp), np.array(force_lambda)


"# ===== ===== ===== ===== #"
"# < Transient Analysis > #"
"# ===== ===== ===== ===== #"
class TransientAnalysis:
    
    def __init__(self, ODB: opst.post.CreateODB) -> None:
        
        self.ODB = ODB

    def analyze(self, dt: float, npts: int):
        
        """
        Transient analysis function
        
        Args:
            dt (float): 地震波每一步的时间间隔
            npts (int): 这个地震波总共有多少步
        
        Returns:
            None
        """
        
        # 重置分析
        ops.wipeAnalysis()
        ops.integrator('Newmark', 0.5, 0.25) # Dynamic analysis requires external settings

        # 智能分析参数
        analysis = opst.anlys.SmartAnalyze(
            analysis_type="Transient",
            testType='NormDispIncr', # 收敛判断类型
            testTol=1e-10,  # 收敛容差
            minStep=1e-6,  # 最小收敛步长
            tryAddTestTimes=True, # 不收敛时尝试增加最大迭代次数
            # 默认：最后一次不收敛范数 normTol<1000，再尝试50次 testIterTimesMore=[50]
            tryAlterAlgoTypes=True,  # 尝试不同test方法
            # 默认：algoTypes=[40, 10, 20, 30, 50, 60, 70, 90]
            tryLooseTestTol=True,  # 尝试宽松的收敛条件
            # 默认：looseTestTolTo = 1000 * testTol
            debugMode=False,  # False for progress bar, True for debug info
            )
        segs = analysis.transient_split(npts)  # Tells the program the total number of steps, which is necessary for outputting a progress bar

        # 执行分析
        for _ in segs:
            ok = analysis.TransientAnalyze(dt)
            if ok < 0:
                analysis.close() # 关闭分析
                raise RuntimeError("Analysis failed")
        
        # 关闭分析
        analysis.close()


"""
# --------------------------------------------------
# ========== < TEST > ==========
# --------------------------------------------------
"""
if __name__ == "__main__":
    pass
