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
    return np, pd


@app.cell
def _(np, pd):
    disp = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    force = np.array([1, 1, 1, 1, 2, 3, 4, 5])

    data = pd.DataFrame({
        "Disp": disp,
        "Force": force
    })
    data
    return


if __name__ == "__main__":
    app.run()
