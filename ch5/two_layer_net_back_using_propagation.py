import sys, os
sys.path.append(os.pardir)
import numpy as np
from layers import Affine, Relu, SoftmaxWithLoss
from ch4.functions import numerical_gradient
from collections import OrderedDict


class TwoLayerNetBackPropagation:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みとバイアスの初期化
        self.params = {}
        # 1層目の重み
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        # 1層目のバイアス
        self.params['b1'] = np.zeros(hidden_size)
        # 2層目の重み
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 2層目のバイアス
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        # 逆伝播ではlayerを正しい順序でひっくり返して使えるようにするため、OrderDictを用いる
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """認識(推論、フォワード処理)を行う

        :param x: 画像データ
        :return:
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数の値を求める

        認識(predict())結果と正解ラベルを元に、交差エントロピー誤差を求める

        :param x: 入力データ(画像データ)
        :param t: 教師データ(正解ラベル)
        :return:
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        """認識精度を求める

        :param x: 入力データ(画像データ)
        :param t: 教師データ(正解ラベル)
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """数値微分を使って、重みパラメータに対する勾配を求める

        :param x: 入力データ(画像データ)
        :param t: 教師データ(正解ラベル)
        :return:
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """誤差逆伝播法を使って、重みパラメータに対する勾配を求める

        :param x: 入力データ(画像データ)
        :param t: 教師データ(正解ラベル)
        :return:
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db

        return grads
