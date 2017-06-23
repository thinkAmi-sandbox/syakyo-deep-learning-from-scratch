import numpy as np
import matplotlib.pylab as plt


def sigmoid_function(x):
    """

    :param x: NumPy配列
    :return:
    """
    # NumPyのブロードキャスト機能により、
    # スカラ値とNumPy配列での演算が行われると、
    # スカラ値とNumPy配列の各要素同士で演算が行われる
    # => enp.expもNumPy配列を生成するため、NumPy配列の各要素の間で計算される
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    sample = np.array([-1.0, 1.0, 2.0])
    print(sigmoid_function(sample))
    # => [ 0.26894142  0.73105858  0.88079708]

    # -5.0から5.0までの範囲を0.1刻みでNumpy配列を生成する([-5.0, -4.9, ... , 4.9])
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid_function(x)

    # x, y配列をプロットする
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
    plt.show()
