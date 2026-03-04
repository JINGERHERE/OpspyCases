# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.18.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import inspect
    from pathlib import Path
    from itertools import count
    from typing import NamedTuple, List, Type, Union, Literal, Optional
    from collections import namedtuple
    from dataclasses import dataclass, fields, asdict
    return (np,)


@app.cell
def _(np):
    def generate_path(peaks, repeat=1, mode='full', add_zero_start=True, add_zero_end=True):
        path = []
    
        for p in peaks:
            # 使用字典定义模式对应的“基本单元”
            patterns = {
                'full': [p, -p],
                'half': [p, 0.0],
                'push': [p]
            }
        
            # 获取模式，若输入错误则抛出异常
            unit = patterns.get(mode)
            if unit is None:
                raise ValueError(f"Invalid mode: {mode}. Choose from 'full', 'half', 'push'.")
        
            # 将单元重复 repeat 次并存入路径
            path.extend(unit * repeat)

        res = np.array(path, dtype=float)

        # 零点处理
        if add_zero_start:
            res = np.insert(res, 0, 0.0)
        if add_zero_end and (len(res) == 0 or res[-1] != 0.0):
            res = np.append(res, 0.0)
        
        return res
    return (generate_path,)


@app.cell
def _(generate_path, np):
    disp_path = np.array([1, 2, 3, 4, 5])
    tt = generate_path(
        disp_path, repeat=2,
        mode='half',
        add_zero_start=False, add_zero_end=False
    )
    tuple(tt)
    return


@app.cell
def _(np):
    np.arange(0.003, 0.015 + 0.003, 0.003)
    return


if __name__ == "__main__":
    app.run()
