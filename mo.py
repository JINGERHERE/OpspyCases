# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.18.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import inspect
    from typing import NamedTuple, List, Type, Union, Literal, Optional
    from collections import namedtuple
    from dataclasses import dataclass, fields, asdict
    return (np,)


@app.cell
def _():
    ON = False
    props = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print([n*2 if ON else "_" for n in props])
    return


@app.cell
def _():
    D1 = {'a': 1, 'b': 2}
    D2 = {'c': 1, 'd': 2}
    D3 = {'e': 1, 'f': 2}
    t = dict(**D1, **D2, **D3)

    for k, v in t.items():
        print(f'key: {k}, values: {v}')
    return


@app.cell
def _():
    DD = {
        1: (1, 2, 3),
        2: (1, 2, 3),
        3: (1, 2, 3),
        4: (1, 2, 3),
        }
    print(tuple(DD.keys()))
    return


@app.cell
def _():
    from scipy.integrate import trapezoid

    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    y = [0, 2, 23, 44, 56, 124]
    print(trapezoid(y, x))
    return


@app.cell
def _():
    def pp(a, b):
        print(f'a: {a}, b: {b}')

    section_case = [
        ('sec_I', 'y'),
        ('sec_rect', 'y'),
        ('sec_polygonal', 'y'),
        ('sec_polygonal', 'z'),
        ('sec_circle', 'y'),
        ('sec_circle', 'z'),
        ]

    for i in section_case:
        print(i[0])
    return


@app.cell
def _():
    N = 4750
    speed = 50

    r = list(range(0, N, speed)) + [N]

    for i in r:
        print(i)
    return


@app.cell
def _(np):
    import warnings
    def get_angle(p1, p2, dof, ndim: int = 3, deg: bool = False):

        """
        计算 `两点连线` 于 `指定维度正交轴线(正向)` 的夹角
            - `dof = 2` 代表第二个维度的正交轴线
        """

        if len(p1) > ndim:
            raise ValueError(f"ndim of point length must less than {ndim}, but got p1: '{len(p1)}'")
        if len(p2) > ndim:
            raise ValueError(f"ndim of point length must less than {ndim}, but got p2: '{len(p2)}'")

        if len(p1) < ndim and len(p2) == ndim: # 其中一个点满足维度要求
            warnings.warn(f"Input p1 is lower-dimensional; zero-padding applied.", UserWarning)
        if len(p1) == ndim and len(p2) < ndim:
            warnings.warn(f"Input p2 is lower-dimensional; zero-padding applied.", UserWarning)


        # 填充维度
        p1 = np.pad(np.array(p1), (0, ndim - np.array(p1).size))
        p2 = np.pad(np.array(p2), (0, ndim - np.array(p2).size))

        # 两点连线距离
        distance = np.linalg.norm(
            # np.delete(p2, dof - 1) - np.delete(p1, dof - 1)
            p2 - p1
            )

        # 指定方法的正交距离
        orthogonal = p2[dof-1] - p1[dof-1]

        # 计算夹角
        angle = np.arccos(orthogonal / distance)

        if deg:
            return np.degrees(angle) # 计算并返回角度（度）
        else:
            return angle # 计算并返回角度（弧度）

    get_angle(
        (0, 0, 0),
        (100, 1, 1),
        dof=2,
        ndim=3,
        deg=True
        )
    return


if __name__ == "__main__":
    app.run()
