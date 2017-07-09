import numpy as np
import matplotlib.pylab as plt
from numerical_diff import numerical_diff


def function_1(x):
    """y=0.01^2 + 0.1x という関数"""
    return 0.01 * x ** 2 + 0.1 * x


def tangent_line(f, x):
    # 数値微分
    d = numerical_diff(f, x)

    y = f(x) - d * x

    # 接線を求める関数
    return lambda t: d * t + y


def draw_graph_function_1_with_tangent():
    """function_1のグラフとその接線を描く"""

    # 0から20まで、0.1刻みのx配列
    x = np.arange(0.0, 20.0, 0.1)

    y = function_1(x)

    plt.xlabel('x')
    plt.ylabel('f(x)')

    # function_1のグラフ
    plt.plot(x, y)

    # function_1の接線(tangent_line)
    # x=5の時の接線
    tf1 = tangent_line(function_1, 5)
    y2 = tf1(x)
    plt.plot(x, y2)

    # x=10のときの接線
    tf2 = tangent_line(function_1, 10)
    y3 = tf2(x)
    plt.plot(x, y3)

    # グラフの表示
    plt.show()


if __name__ == '__main__':
    # xに対するf(x)の変化量(微分)を求める
    # x=5のときの、function_1の微分
    x_5 = numerical_diff(function_1, 5)
    print(f'x=5のときのfunction_1の微分 -> {x_5}')
    # => x=5のときのfunction_1の微分 -> 0.1999999999990898
    # 解析的な解：x=5のとき、0.2 (厳密に一致しないが、誤差は非常に小さい)

    # x=10のときの、function_1伸び分
    x_10 = numerical_diff(function_1, 10)
    print(f'x=10のときのfunction_1の微分 -> {x_10}')
    # => x=10のときのfunction_1の微分 -> 0.2999999999986347
    # 解析的な解：x=10のとき、0.3

    # function_1のグラフを表示
    draw_graph_function_1_with_tangent()
