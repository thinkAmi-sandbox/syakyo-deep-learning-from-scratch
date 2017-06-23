import numpy as np
from sigmoid_function import sigmoid_function


def print_dot(x, w):
    print(f'x.shape -> {x.shape}')
    print(f'w.shape -> {w.shape}')

    y = np.dot(x, w)
    print(f'dot -> ${y}')


def sum_1st_layer():
    # 入力信号(x)、重み(w)、バイアス(b)は適当な値を設定している
    x = np.array([1.0, 0.5])  # 要素数2の配列
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])   # 2x3の配列
    b1 = np.array([0.1, 0.2, 0.3])

    print(f'x.shape -> {x.shape}')
    print(f'w1.shape -> {w1.shape}')
    print(f'b1.shape -> {b1.shape}')

    # 1層目の重み付き和(a1) = 1層の全入力(x)と1層の全重み(w1)のドット積 + 1層のバイアス
    a1 = np.dot(x, w1) + b1
    print(f'a1 -> {a1}')

    return a1


def sum_2nd_layer(z1):
    # 2層の全重み: 2x3の配列
    w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    # 2層のバイアス 2要素
    b2 = np.array([0.1, 0.2])

    print(f'z1.shape -> {z1.shape}')
    print(f'w2.shape -> {w2.shape}')
    print(f'b2.shape -> {b2.shape}')

    # 2層目の重み付き和
    a2 = np.dot(z1, w2) + b2
    print(f'a2 -> {a2}')
    return a2


def identity_function(x):
    """活性化関数"""
    # 入力したものと同じ値を返すので、恒等関数という
    return x


def sum_3rd_layer(z2):
    # 3層の全重み：2x2の配列
    w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    # 3層のバイアス：2要素
    b3 = np.array([0.1, 0.2])

    print(f'z2.shape -> {z2.shape}')
    print(f'w3.shape -> {w3.shape}')
    print(f'b3.shape -> {b3.shape}')

    # 3層目の重み付き和
    a3 = np.dot(z2, w3) + b3
    print(f'a3 -> {a3}')

    return a3


def all_in_one():
    def init_network():
        network = {}
        # ニューラルネットワークの慣例として、
        # ・重み付け: W1のような大文字
        # ・それ以外(バイアスや中韓結果): 小文字
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network['b1'] = np.array([0.1, 0.2, 0.3])
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network['b2'] = np.array([0.1, 0.2])
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network['b3'] = np.array([0.1, 0.2])

        return network

    def forward(network, x):
        """入力信号が出力へと変換されるプロセスがまとめて実装されている

        forward: 入力から出力方向への伝達処理を表す
        """
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid_function(a1)

        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid_function(a2)

        a3 = np.dot(z2, W3) + b3
        y = identity_function(a3)

        return y

    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)

    return y


if __name__ == '__main__':
    # 0層目から1層目
    x = np.array([1, 2])
    w = np.array([[1, 3, 5], [2, 4, 6]])
    print_dot(x, w)
    # x.shape -> (2,)
    # w.shape -> (2, 3)
    # dot -> $[ 5 11 17]

    print('-' * 20)
    a1 = sum_1st_layer()
    # x.shape -> (2,)
    # w1.shape -> (2, 3)
    # b1.shape -> (3,)
    # a1 -> [ 0.3  0.7  1.1]

    # 活性化関数によるプロセス：シグモイド関数を使う
    z1 = sigmoid_function(a1)
    print(f'1st layer sigmoid -> {z1}')
    # 1st layer sigmoid -> [ 0.57444252  0.66818777  0.75026011]

    # 1層目から2層目
    print('-' * 20)
    a2 = sum_2nd_layer(z1)
    # z1.shape -> (3,)
    # w2.shape -> (3, 2)
    # b2.shape -> (2,)
    # a2 -> [ 0.51615984  1.21402696]

    z2 = sigmoid_function(a2)
    print(f'2nd layer sigmoid -> {z2}')
    # 2nd layer sigmoid -> [ 0.62624937  0.7710107 ]

    # 2層目から出力層
    print('-' * 20)
    a3 = sum_3rd_layer(z2)
    # z2.shape -> (2,)
    # w3.shape -> (2, 2)
    # b3.shape -> (2,)
    # a3 -> [ 0.31682708  0.69627909]

    y = identity_function(a3)
    print(f'identity_function -> {y}')
    # identity_function -> [ 0.31682708  0.69627909]

    # ニューラルネットワークのフォワード方向の実装をまとめて行ったもの
    y_all = all_in_one()
    print(f'y_all_in_one -> {y_all}')
    # y_all_in_one -> [ 0.31682708  0.69627909]
