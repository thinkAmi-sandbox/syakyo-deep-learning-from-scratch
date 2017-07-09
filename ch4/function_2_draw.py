import numpy as np
import matplotlib.pylab as plt
from numerical_diff import numerical_diff


def function_2(x):
    """引数の二乗和を求める

    NumPy配列の各要素を2乗して、その和を求める

    :param x: NumPy配列
    :return:
    """
    return x[0] ** 2 + x[1] ** 2


def function_temp1(x0):
    """変数がx0だけの関数"""
    return x0 * x0 + 4.0 ** 2.0


def function_temp2(x1):
    """変数がx1だけの関数"""
    return 3.0 ** 2.0 + x1 * x1


if __name__ == '__main__':
    # x0=3, x1=4のときの、x0に対する偏微分を求める
    # x1の数値を固定
    result_x0 = numerical_diff(function_temp1, 3.0)
    print(f'x0に対する偏微分 -> {result_x0}')
    # => x0に対する偏微分 -> 6.00000000000378

    # x0=3, x1=4のときの、x1に対する偏微分を求める
    # x0の数値を固定
    result_x1 = numerical_diff(function_temp2, 4.0)
    print(f'x1に対する偏微分 -> {result_x1}')
    # => x1に対する偏微分 -> 7.999999999999119
