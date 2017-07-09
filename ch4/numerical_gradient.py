import numpy as np
import matplotlib.pylab as plt
from function_2_draw import function_2
from mpl_toolkits.mplot3d import Axes3D  # これは使われていない


def _numerical_gradient(f, x):
    """xの各要素に対して、数値微分を求める

    :param f: 関数
    :param x: NumPy配列

    :return:
    """
    # 0.0001
    h = 1e-4

    # xと同じ形状の配列で、その要素がすべて0の配列
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]

        # f(x+h)の計算
        x[i] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[i] = tmp_val - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)

        # 値を元に戻す
        x[i] = tmp_val

    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)

        for idx, x in enumerate(x):
            grad[idx] = _numerical_gradient(f, x)

        return grad


def numerical_gradient_all(f, x):
    h = 1e-4    # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        tmp_val = x[i]
        x[i] = float(tmp_val) + h
        fxh1 = f(x)

        x[i] = float(tmp_val) - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)

        x[i] = tmp_val  # 元に戻す
        it.iternext()

    return grad





if __name__ == '__main__':
    result_3_4 = numerical_gradient(function_2, np.array([3.0, 4.0]))
    print(f'点(3,4)での勾配 -> {result_3_4}')
    # => 点(3,4)での勾配 -> [ 6.  8.]

    result_0_2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
    print(f'点(0,2)での勾配 -> {result_0_2}')
    # => 点(0,2)での勾配 -> [ 0.  4.]

    result_3_0 = numerical_gradient(function_2, np.array([3.0, 0.0]))
    print(f'点(3,0)での勾配 -> {result_3_0}')
    # => 点(3,0)での勾配 -> [ 6.  0.]

    # 勾配図を書きたいが、うまくいかない
    # -2から2.5までを0.25区切りで
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)

    x, y = np.meshgrid(x0, x1)

    x = x.flatten()
    y = y.flatten()

    grad = numerical_gradient(function_2, np.array([x, y]))

    plt.figure()
    plt.quiver(x, y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
