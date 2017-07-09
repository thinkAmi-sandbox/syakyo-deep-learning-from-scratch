import sys, os
sys.path.append(os.pardir)
import numpy as np
# from ch3.sigmoid_function import sigmoid_function
# from ch3.softmax_function import softmax
# from numerical_gradient import numerical_gradient_all
from cross_entropy_error_with_batch import cross_entropy_error

from functions import *


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """

        :param input_size: 入力層のニューロンの数
        :param hidden_size: 隠れ層のニューロンの数
        :param output_size: 出力層のニューロンの数
        :param weight_init_std:
        """
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

    def predict(self, x):
        """認識(推論、フォワード処理)を行う

        :param x: 画像データ
        :return:
        """
        w1 = self.params['W1']
        w2 = self.params['W2']

        b1 = self.params['b1']
        b2 = self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, w2) + b2

        y = softmax(a2)

        return y

    def loss(self, x, t):
        """損失関数の値を求める

        認識(predict())結果と正解ラベルを元に、交差エントロピー誤差を求める

        :param x: 入力データ(画像データ)
        :param t: 教師データ(正解ラベル)
        :return:
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """認識精度を求める

        :param x: 入力データ(画像データ)
        :param t: 教師データ(正解ラベル)
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        """重みパラメータに対する勾配を求める

        :param x: 入力データ(画像データ)
        :param t: 教師データ(正解ラベル)
        :return:
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        # 1層目の重みの勾配
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        # 1層目のバイアスの勾配
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        # 2層目の重みの勾配
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        # 2層目のバイアスの勾配
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads
