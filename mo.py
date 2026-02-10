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
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import inspect
    from typing import NamedTuple, List, Type, Union, Literal, Optional
    from collections import namedtuple
    from dataclasses import dataclass, fields, asdict
    return


@app.cell
def _():
    def tt(*args):
        print(args)

    t  =(1, 2, 3, 4, 5, 6, 7)
    # *rt, = reversed(t)
    tt(*reversed(t))
    return


if __name__ == "__main__":
    app.run()
