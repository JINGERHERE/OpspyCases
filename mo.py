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
    return List, Literal, Union, dataclass, inspect, namedtuple


@app.cell
def _():
    # 定义类
    class Test:
        def __init__(self):
            self.a = 1
            self.b = 2

        def __name__(self):
            return 'test_test'

        def test0(self):
            print(f'这是实例方法')

        @staticmethod
        def test1():
            print(f'这是静态方法')

        @classmethod
        def test2(cls):
            print(f'这是类方法')

        @property
        def ps(self):
            """要触发绑定才会返回1"""
            print("这是属性方法")
            return 1

    # 实例化类
    test = Test()
    return Test, test


@app.cell
def _(dataclass):
    @dataclass
    class Data1:
        a=1
        b=2
        c=3

    @dataclass
    class Data2:
        a: int = 1
        b: int = 2
        c: int = 3

    data = Data1()
    return (Data1,)


@app.cell
def _(List, Literal, Union, inspect, namedtuple):
    CallableInfo = namedtuple('CallableInfo', ['name', 'kind', 'callable'])
    def get_callables(
        obj: Union[type, object],
        kind: Literal['function', 'method', 'property', 'data', 'all'] = 'all',
        ) -> List[CallableInfo]:

        """
        返回类对象中所有可调用的函数。
            - 不包含魔法方法（以 '__' 开头）。

        Args:
            obj (Union[type, object]): 类对象。
            kind (Literal['function', 'method', 'property', 'data', 'all'], optional): 可调用对象的类型。默认值为 'all'。

        Raises:
            TypeError: 如果 obj 不是类对象。
            ValueError: 如果 kind 不是允许的值。

        Returns:
            List[CallableInfo]: 包含函数名和函数对象的列表。

        """

        # 限制输入
        kind_allowed = {'function', 'method', 'property', 'data', 'all'}
        if kind not in kind_allowed:
            raise ValueError(f"kind must be one of {kind_allowed}, got {kind!r}")

        if obj.__class__ is not type:
            raise TypeError(f"obj must be a '<class 'type'>' or '<class 'object'>', got {type(obj)}")

        # 获取实际的对象
        if isinstance(obj, type):
            obj_class = obj # 类对象
        else:
            obj_class = obj.__class__ # 获取类本身

        # 获取可调用对象
        is_property = lambda c: isinstance(c, property)
        get_func = inspect.getmembers(obj_class, predicate=inspect.isfunction) # 函数
        get_method = inspect.getmembers(obj_class, predicate=inspect.ismethod) # 方法
        get_prop = inspect.getmembers(obj_class, predicate=is_property) # 属性

        # 汇总所有可调用对象
        cls_callables = dict(
            function = [
                CallableInfo(name=name, kind='function', callable=getattr(obj_class, name))
                    for name, _ in get_func
                        if not name.startswith('__')
                ],
            method = [
                CallableInfo(name=name, kind='method', callable=getattr(obj_class, name))
                    for name, _ in get_method
                        if not name.startswith('__')
                ],
            property = [
                CallableInfo(name=name, kind='property', callable=getattr(obj_class, name))
                    for name, _ in get_prop
                        if not name.startswith('__')
                ],
            data = [
                CallableInfo(name=name, kind='data', callable=getattr(obj_class, name))
                    for name, val in inspect.getmembers(obj_class)
                        if not name.startswith('__') and not callable(val)
                ],
            )

        # 返回指定类型的可调用对象
        if kind == 'all':
            return cls_callables['function'] \
                    + cls_callables['method'] \
                    + cls_callables['property'] \
                    + cls_callables['data']
        else:
            return cls_callables[kind]
    return (get_callables,)


@app.cell
def _(get_callables, test):
    f1 = get_callables(obj=test)[2]
    f1.callable
    return


@app.cell
def _(Test, get_callables):
    f2 = get_callables(obj=Test)[2]
    f2.callable()
    return


@app.cell
def _(Data1, get_callables):
    get_callables(obj=Data1, kind='data')[1]
    return


@app.cell
def _(Data1, get_callables):
    get_callables(obj=Data1)
    return


@app.function
def tttt():
    ...


@app.cell
def _():
    tttt.__class__
    return


@app.cell
def _(Test, test):
    print(type(Test))
    print(type(test))
    print(type(tttt))
    return


@app.cell
def _():
    sec_props = {
        "A": 1.,
        "E": 2.,
        "J": 3.,
        "I": 0.1
        }
    return (sec_props,)


@app.cell
def _(sec_props):
    sec_props['C'] = 0.01
    print(sec_props)
    return


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


if __name__ == "__main__":
    app.run()
