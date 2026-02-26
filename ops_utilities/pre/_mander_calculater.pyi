# -*- coding:utf-8 -*-
# @Time         : 2020/11/30 18:04
# @Author       : Penghui Zhang
# @Email        : penghui@tongji.edu.cn
# @File         : Mander.py
# @Software     : PyCharm

# @Last editor  : Wang Jing
# @Last time    : 2025/7/5 19:12


import math
from decimal import Decimal, getcontext
from typing import Literal, Union, TypedDict, Unpack


class Mander:
    """
    Mander 本构计算器：核心混凝土 `强度` `应变`
        - circular: 圆形截面
        - rectangular: 矩形截面
    """

    eco = 0.0033  # 无约束混凝土最大应力时的应变
    esu = 0.09  # 箍筋拉断应变

    @classmethod
    def _william_warnke(cls, sigma1, sigma2, sigma3) -> Decimal:
        """
        William-Warnke混凝土五参数模型确定的破坏面。

        Args:
            sigma1 : 第一主应力。
            sigma2 : 第二主应力。
            sigma3 : 第三主应力。

        Returns:
            Decimal: 破坏面函数值。

        """

        getcontext().prec = 30
        sigma1, sigma2, sigma3 = Decimal(sigma1), Decimal(sigma2), Decimal(sigma3)
        sigmaa = (sigma1 + sigma2 + sigma3) / Decimal(3)
        taoa = (
            (sigma1 - sigma2) ** Decimal(2)
            + (sigma2 - sigma3) ** Decimal(2)
            + (sigma3 - sigma1) ** Decimal(2)
        ) ** Decimal(0.5) / Decimal(15) ** Decimal(0.5)

        # 若根据实验数据对破坏面进行标定，则按如下公式取值
        # alphat, alphac, kexi, rou1, rou2=Decimal(0.15),Decimal(1.8),Decimal(3.67),Decimal(1.5),Decimal(1.94)
        # a2 = Decimal(9)*(Decimal(1.2)**Decimal(0.5)*kexi*(alphat-alphac)-Decimal(1.2)**Decimal(0.5)*alphat*alphac+rou1*(Decimal(2)*alphac+alphat))/((Decimal(2)*alphac+alphat)*(Decimal(3)*kexi-Decimal(2)*alphac)*(Decimal(3)*kexi+alphat))
        # a1 = (Decimal(2)*alphac-alphat)*a2/Decimal(3)+Decimal(1.2)**Decimal(0.5)*(alphat-alphac)/(Decimal(2)*alphac+alphat)
        # a0 = Decimal(2)*alphac*a1/Decimal(3)-Decimal(4)*alphac**Decimal(2)*a2/Decimal(9)+(Decimal(2)/Decimal(15))**Decimal(0.5)*alphac
        # kexi0 = (-a1-(a1**2-Decimal(4)*a0*a2)**Decimal(0.5))/(Decimal(2)*a2)
        # b2 = Decimal(9)*(rou2*(kexi0+Decimal(1)/Decimal(3))-(Decimal(2)/Decimal(15))**Decimal(0.5)*(kexi0+kexi))/((kexi+kexi0)*(Decimal(3)*kexi-Decimal(1))*(Decimal(3)*kexi0+Decimal(1)))
        # b1 = (kexi+Decimal(1)/Decimal(3))*b2+(Decimal(1.2)**Decimal(0.5)-Decimal(3)*rou2)/(Decimal(3)*kexi-Decimal(1))
        # b0 = -kexi0*b1-kexi0**Decimal(2)*b2
        # r1 = a0+a1*sigmaa+a2*sigmaa**Decimal(2)
        # r2 = b0+b1*sigmaa+b2*sigmaa**Decimal(2)

        # 参考Schickert-Winkler的实验数据得到的结果
        r1 = (
            Decimal(0.053627)
            - Decimal(0.512079) * sigmaa
            - Decimal(0.038226) * sigmaa ** Decimal(2)
        )
        r2 = (
            Decimal(0.095248)
            - Decimal(0.891175) * sigmaa
            - Decimal(0.244420) * sigmaa ** Decimal(2)
        )
        r21 = r2 ** Decimal(2) - r1 ** Decimal(2)

        cosxita = (Decimal(2) * sigma1 - sigma2 - sigma3) / (
            Decimal(2) ** Decimal(0.5)
            * (
                (sigma1 - sigma2) ** Decimal(2)
                + (sigma2 - sigma3) ** Decimal(2)
                + (sigma3 - sigma1) ** Decimal(2)
            )
            ** Decimal(0.5)
        )
        rxita = (
            Decimal(2) * r2 * r21 * cosxita
            + r2
            * (Decimal(2) * r1 - r2)
            * (
                Decimal(4) * r21 * cosxita ** Decimal(2)
                + Decimal(5) * r1 ** Decimal(2)
                - Decimal(4) * r1 * r2
            )
            ** Decimal(0.5)
        ) / (
            Decimal(4) * r21 * cosxita ** Decimal(2)
            + (r2 - Decimal(2) * r1) ** Decimal(2)
        )

        # 破坏面函数值
        return taoa / rxita - 1

    @classmethod
    def _confinedStrengthRatio(
        cls,
        confiningStrengthRatio1: Union[float, int],
        confiningStrengthRatio2: Union[float, int],
    ) -> float:
        """
        根据两个方向的约束应力比计算核心混凝土的强度提高系数。

        Args:
            confiningStrengthRatio1 (float, int): x方向的约束应力比。
            confiningStrengthRatio2 (float, int): y方向的约束应力比。

        Returns:
            float: 核心混凝土强度提高系数。
        """

        sigma1 = -min(confiningStrengthRatio1, confiningStrengthRatio2)
        sigma2 = -max(confiningStrengthRatio1, confiningStrengthRatio2)
        sigma3Min = -4
        sigma3Max = -1

        while True:
            sigma3Mid = (sigma3Min + sigma3Max) / 2
            fun_min = cls._william_warnke(sigma1, sigma2, sigma3Min)
            fun_max = cls._william_warnke(sigma1, sigma2, sigma3Max)
            fun_mid = cls._william_warnke(sigma1, sigma2, sigma3Mid)
            if abs(fun_mid) < 0.001:  # 当误差小于设定范围时，输出值
                return -sigma3Mid
                break
            elif fun_min * fun_mid < 0:
                sigma3Max = sigma3Mid
            elif fun_max * fun_mid < 0:
                sigma3Min = sigma3Mid

    # ---------------------------------------------------------
    # 计算圆形截面
    # ---------------------------------------------------------
    class ParamsCircular(TypedDict, total=False):
        coverThick: float  # 保护层厚度
        roucc: float  # 纵筋配筋率，计算时只计入约束混凝土面积
        s: float  # 箍筋纵向间距（螺距）
        ds: float  # 箍筋直径
        fyh: float  # 箍筋屈服强度 (MPa)

    @classmethod
    def circular(
        cls,
        hoop: Literal["Circular", "Spiral"],
        d: Union[float, int],
        fco: Union[float, int],
        **params: Unpack[ParamsCircular],
    ) -> tuple[float, float, float]:
        """
        计算圆截面混凝土柱的Mander模型参数。

        Args:
            hoop (Literal['Circular', 'Spiral']): 箍筋类型，'Circular'圆形箍筋，'Spiral'螺旋形箍筋
            fco (float, int): 无约束混凝土抗压强度标准值 (识别单位)
            d (float, int): 截面直径

            coverThick (float, int): `默认值：0.08 m` 保护层厚度
            roucc (float, int): `默认值：0.03` 纵筋配筋率，计算时只计入约束混凝土面积
            s (float, int): `默认值：0.1 m` 箍筋纵向间距（螺距）
            ds (float, int): `默认值：0.014 m` 箍筋直径
            fyh (float, int): `默认值：400 MPa` 箍筋屈服强度 (识别单位)

        Returns:
            (float, float, float): A tuple containing：
                - fcc (float): 核心混凝土抗压强度(识别单位)
                - ecc (float): 核心混凝土抗压强度对应的应变
                - ecu (float): 核心混凝土极限应变

        Examples:
            >>> fcc, ecc, ecu = Mander.circular(hoop="Circular", fco=40, d=1.0)
            >>> print(f'fcc: {fcc:.4f}')
            >>> 48.7033
            >>> print(f'ecc: {ecc:.4f}')
            >>> 0.0069
            >>> print(f'ecu: {ecu:.4f}')
            >>> 0.0117
        """

        # 默认值
        defs = {
            "coverThick": 0.08,
            "roucc": 0.03,
            "s": 0.1,
            "ds": 0.014,
            "fyh": 400,
        }

        # 更新默认值
        defs.update(params)

        # 计算有效距离
        de = d - 2 * defs["coverThick"] - defs["ds"]

        # 判断箍筋类型
        if hoop.lower() == "circular":
            ke = (1 - 0.5 * defs["s"] / de) ** 2 / (1 - defs["roucc"])
        elif hoop.lower() == "spiral":
            ke = (1 - 0.5 * defs["s"] / de) / (1 - defs["roucc"])
        else:
            raise ValueError("hoop mast be 'Circular' or 'Spiral'")

        # 计算核心混凝土的抗压强度和应变
        rous = math.pi * defs["ds"] ** 2 / (de * defs["s"])
        fle = 0.5 * ke * rous * defs["fyh"]
        fcc = fco * (-1.254 + 2.254 * math.sqrt(1 + 7.94 * fle / fco) - 2 * fle / fco)
        ecc = cls.eco * (1 + 5 * (fcc / fco - 1))
        ecu = 0.004 + 1.4 * rous * defs["fyh"] * cls.esu / fcc

        # 返回核心混凝土的抗压强度、对应应变、极限应变
        return float(fcc), float(ecc), float(ecu)

    # ---------------------------------------------------------
    # 计算矩形截面
    # ---------------------------------------------------------
    class ParamsRectangular(TypedDict, total=False):
        coverThick: Union[float, int]  # 保护层厚度
        roucc: Union[float, int]  # 纵筋配筋率，计算时只计入约束混凝土面积
        sl: Union[float, int]  # 纵筋间距
        dsl: Union[float, int]  # 纵筋直径
        roux: Union[float, int]  # x方向的体积配箍率，计算时只计入约束混凝土面积
        rouy: Union[float, int]  # y方向的体积配箍率，计算时只计入约束混凝土面积
        st: Union[float, int]  # 箍筋间距
        dst: Union[float, int]  # 箍筋直径
        fyh: Union[float, int]  # 箍筋屈服强度 (MPa)

    @classmethod
    def rectangular(
        cls,
        lx: Union[float, int],
        ly: Union[float, int],
        fco: Union[float, int],
        **params: Unpack[ParamsRectangular],
    ) -> tuple[float, float, float]:
        """
        计算矩形截面混凝土柱的Mander模型参数。

        Args:
            fco (float, int): 无约束混凝土抗压强度标准值 (识别单位)
            lx (float, int): x方向截面的宽度
            ly (float, int): y方向截面宽度

            coverThick (float, int): `默认值：0.08 m` 保护层厚度
            roucc (float, int): `默认值：0.03` 纵筋配筋率，计算时只计入约束混凝土面积
            sl (float, int): `默认值：0.1 m` 纵筋间距
            dsl (float, int): `默认值：0.032 m` 纵筋直径
            roux (float, int): `默认值：0.00057` x方向的体积配箍率，计算时只计入约束混凝土面积
            rouy (float, int): `默认值：0.00889` y方向的体积配箍率，计算时只计入约束混凝土面积
            st (float, int): `默认值：0.3 m` 箍筋间距
            dst (float, int): `默认值：0.018 m` 箍筋直径
            fyh (float, int): `默认值：500 MPa` 箍筋屈服强度 (识别单位)

        Returns:
            (float, float, float): A tuple containing：
                - fcc (float): 核心混凝土抗压强度 (识别单位)
                - ecc (float): 核心混凝土抗压强度对应的应变
                - ecu (float): 核心混凝土极限应变

        Examples:
            >>> fcc, ecc, ecu = Mander.rectangular(fco=40, lx=4.0, ly=6.5)
            >>> print(f'fcc: {fcc:.4f}')
            >>> 48.9062
            >>> print(f'ecc: {ecc:.4f}')
            >>> 0.0070
            >>> print(f'ecu: {ecu:.4f}')
            >>> 0.0162
        """

        # 默认值
        defs = {
            "coverThick": 0.08,
            "roucc": 0.03,
            "sl": 0.1,
            "dsl": 0.032,
            "roux": 0.00057,
            "rouy": 0.00889,
            "st": 0.3,
            "dst": 0.018,
            "fyh": 500,
        }

        # 更新默认值
        defs.update(params)

        # 计算有效距离
        lxe = lx - 2 * defs["coverThick"] - defs["dst"]
        lye = ly - 2 * defs["coverThick"] - defs["dst"]

        nsl = (
            2 * (lxe + lye - 2 * defs["dsl"]) / defs["sl"]
        )  # 绕截面一圈需要布置的纵筋数
        ke = (
            (1 - nsl * (defs["sl"] ** 2) / (6 * lxe * lye))
            * (1 - 0.5 * defs["st"] / lxe)
            * (1 - 0.5 * defs["st"] / lye)
            / (1 - defs["roucc"])
        )
        flxe = ke * defs["roux"] * defs["fyh"]
        flye = ke * defs["rouy"] * defs["fyh"]
        fcc = fco * cls._confinedStrengthRatio(flxe / fco, flye / fco)
        ecc = cls.eco * (1 + 5 * (fcc / fco - 1))
        ecu = 0.004 + 1.4 * (defs["roux"] + defs["rouy"]) * defs["fyh"] * cls.esu / fcc

        # 返回核心混凝土的抗压强度、对应应变、极限应变
        return float(fcc), float(ecc), float(ecu)


if __name__ == "__main__":

    print(f"\nTest circular section")
    fcc, ecc, ecu = Mander.circular(hoop="Circular", fco=40, d=1.0)
    print(f"fcc: {fcc:.4f}")
    print(f"ecc: {ecc:.4f}")
    print(f"ecu: {ecu:.4f}")

    print(f"\nTest rectangular section")
    fcc, ecc, ecu = Mander.rectangular(fco=40, lx=4.0, ly=6.5)
    print(f"fcc: {fcc:.4f}")
    print(f"ecc: {ecc:.4f}")
    print(f"ecu: {ecu:.4f}")
