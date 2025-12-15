import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import os
    import numpy as np
    return np, os


@app.cell
def _(np):
    disp_step_1 = np.arange(0.002, 0.012 + 0.002, 0.002) # 第一阶段 控制位移幅值
    disp_step_2 = np.arange(0.016, 0.064 + 0.004, 0.004) # 第二节段 控制唯一幅值
    disp_step = np.repeat(
        np.concatenate((disp_step_1, disp_step_2)), # 合并
        3 # 重复三次
        )
    disp_pairs = np.stack((disp_step, -disp_step), axis=1).flatten() # 控制位移成对，整理
    disp_path = np.concatenate(([0.], disp_pairs, [0.])) # 添加首位


    print(disp_path)
    return


@app.cell
def _(os):
    def remove_files(root_dir):
        """
        删除指定目录及其子目录中所有.nc后缀的文件
        :param root_dir: 根目录路径
        """
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.zarr'):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"已删除文件: {file_path}")
                    except OSError as e:
                        print(f"删除文件 {file_path} 时出错: {e}")

    remove_files('./')
    return


if __name__ == "__main__":
    app.run()
