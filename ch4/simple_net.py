import sys, os
sys.path.append(os.pardir)
import numpy as np
from ch3.softmax_function import softmax
from cross_entropy_error_with_batch import cross_entropy_error
from numerical_gradient import numerical_gradient_all


class SimpleNet:
    def __init__(self):
        # ガウス分布(正規分布)で初期化
        self.w = np.random.randn(2, 3)

    def predict(self, x):
        """予測するためのメソッド

        :param x:
        :return:
        """
        return np.dot(x, self.w)

    def loss(self, x, t):
        """損失関数の値を求めるメソッド

        :param x: 入力データ
        :param t: 正解ラベル
        :return:
        """
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    net = SimpleNet()

    print(f'重みパラメータ -> {net.w}')
    # => 重みパラメータ -> [[-0.30475805 -0.9719402   0.44859077]
    #                     [ 0.18549092 -0.88873329 -0.97088813]]

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(f'予測した結果 -> {p}')
    # => 予測した結果 -> [-0.015913   -1.38302408 -0.60464486]

    print(f'最大値のインデックス -> {np.argmax(p)}')
    # => 最大値のインデックス -> 0

    t = np.array([0, 0, 1])
    result = net.loss(x, t)
    print(f'損失関数の値 -> {result}')
    # => 損失関数の値 -> 3.1468807988518575

    # 勾配を数値微分を使って求める
    # net.w(重みパラメータ)を引数にとり、損失関数を計算する関数
    f = lambda w: net.loss(x, t)
    dw = numerical_gradient_all(f, net.w)
    print(f'dw -> {dw}')
    # 2x3の2次元配列
    # => dw -> [[-0.20545498 -0.34654804  0.55200302]
    #           [-0.30818247 -0.51982206  0.82800453]]
    # ・w11に関する勾配は、おおよそ -0.20
    #   -> w11をhだけ増やすと、損失関数の値は0.20hだけ減少する
    # ・w23に関する勾配は、おおよそ 0.82
    #   -> w23をhだけ増やすと、損失関数の値は0.82h増加する
    # -> 損失関数を減らすという観点からは、(符号を逆転するので)
    #    w11はプラス方向へ、w23はマイナス方向へ更新するのが良いと分かる
    # -> 更新の度合いも、w23の方がw11よりも絶対値が大きいので、
    #    w23のほうが大きく貢献するのが分かる
