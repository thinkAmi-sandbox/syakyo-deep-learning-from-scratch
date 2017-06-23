import numpy as np
import matplotlib.pylab as plt


def relu_function(x):
    # np.maximum()は、入力された値から大きい方の値を選んで出力する関数
    return np.maximum(0, x)


if __name__ == '__main__':
    sample = np.array([-1.0, 1.0, 2.0])
    print(relu_function(sample))
    # => [ 0.  1.  2.]

    # -5.0から5.0までの範囲を0.1刻みでNumpy配列を生成する([-5.0, -4.9, ... , 4.9])
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu_function(x)

    # x, y配列をプロットする
    plt.plot(x, y)
    plt.ylim(-1, 5)  # y軸の範囲を指定
    plt.show()
