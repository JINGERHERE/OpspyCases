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
    return (Path,)


@app.cell
def _(Path):
    fit: float = 1.5e3
    data_path = Path().cwd() / f"model_fit_{fit:.2e}"
    data_path.mkdir(parents=True, exist_ok=True)
    print(f'{fit: .2e}')
    return


app._unparsable_cell(
    r"""
    ttt = ["strain", "s"]
    if ttt isin
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
