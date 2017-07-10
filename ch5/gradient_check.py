import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net_back_using_propagation import TwoLayerNetBackPropagation


def main():
    """勾配確認を行う

    数値微分と誤差逆伝播法の結果を比較することで、
    誤差逆伝播法の実装に誤りがないことを確認する
    """
    (x_train, t_train), _ = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNetBackPropagation(
        input_size=784, hidden_size=50, output_size=10
    )

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    # 各重みの絶対誤差の平均を求める
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(f'{key}:{diff}')


if __name__ == '__main__':
    main()
    # 各値ともe-10〜e-13なので、誤差は0に近い値になっている
    # そのため、誤差逆伝播法の実装に誤りがないと考えられる
    # W1:2.1713868350846638e-13
    # b1:7.465760865424387e-13
    # W2:8.007084573288456e-13
    # b2:1.192379542325206e-10
