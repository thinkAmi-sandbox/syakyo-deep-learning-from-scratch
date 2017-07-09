import numpy as np
import matplotlib.pylab as plt
from numerical_gradient import numerical_gradient
from function_2_draw import function_2


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """勾配降下法で計算する

    :param f: 最適化したい関数
    :param init_x: 初期値
    :param lr: 学習率(learning rate)
    :param step_num: 勾配法による繰り返しの数

    :return:
    """
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def draw_gradient_descent_history(x_history):
    """勾配降下法による更新のプロセスを描画する

    ただし、図の等高線を表す破線は表示していない
    gradient_method.pyより移植

    :param x_history:
    """
    plt.plot( [-5, 5], [0,0], '--b')
    plt.plot( [0,0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()


if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    result, x_history = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
    print(f'勾配降下法による結果 -> {result}')
    # => 勾配降下法による結果 -> [ -6.11110793e-10   8.14814391e-10]

    result_too_big, _ = gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
    print(f'勾配降下法による結果で、学習率が大きすぎる場合(lr=10 -> {result_too_big}')
    # => 勾配降下法による結果で、学習率が大きすぎる場合(lr=10 -> [ -2.58983747e+13  -1.29524862e+12]

    result_too_small, _ = gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
    print(f'勾配降下法による結果で、学習率が小さすぎる場合(lr=1e-10 -> {result_too_small}')
    # => 勾配降下法による結果で、学習率が小さすぎる場合(lr=1e-10 -> [  2.34235971e+12  -3.96091057e+12]

    # グラフの描画
    # グラフを描画すると、それ以降のコンソール出力がなされないため、一番最後に実行
    draw_gradient_descent_history(x_history)
