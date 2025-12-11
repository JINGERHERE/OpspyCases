# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.18.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return (np,)


@app.cell
def _(np):

    deg = 90.
    print(np.cos(np.deg2rad(deg)))
    print(np.sin(np.deg2rad(deg)))
    return


@app.cell
def _():
    step_map = {'k': 1}

    kk = 'j'

    try:
        gv = step_map[kk]
    except KeyError:
        print(f'没有这个 key，将采用保底')
        print(f'保底取值：{step_map['k']}')
    else:
        print(f'key正确，取值为：{step_map[kk]}')
    return


if __name__ == "__main__":
    app.run()
