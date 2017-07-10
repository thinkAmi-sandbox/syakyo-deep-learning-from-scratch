import sys, os
sys.path.append(os.pardir)
import numpy as np
from ch4.functions import cross_entropy_error, softmax


class Relu:
    def __init__(self):
        # True/FalseからなるNumPy配列
        # 順伝播の入力であるxの要素で0以下の場所をTrue、それ以外をFalseとして保持する
        # -> 逆伝播の時に上流から伝わってきた値を0にするために利用
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        # 逆伝播の計算で使うため、インスタンス変数outに保存しておく
        self.out = out
        return out

    def backward(self, dout):
        # 順伝播時の出力(self.out)を使って、逆伝播時の出力を計算する
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None   # softmaxの出力
        self.t = None   # 教師データ (one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        # 他のレイヤーと引数を合わせるため、使っていないdoutも定義する
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx


if __name__ == '__main__':
    # maskの例
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)
    # [[ 1.  -0.5]
    #  [-2.   3. ]]

    mask = (x <= 0)
    print(mask)
    # [[False  True]
    #  [ True False]]

    # バッチ版Affineレイヤでのバイアス計算例
    # 順伝播でのバイアスの加算は、それぞれのデータに対して加算が行われる
    x_dot_w = np.array([[0, 0, 0], [10, 10, 10]])
    print(x_dot_w)
    # [[ 0  0  0]
    #  [10 10 10]]

    b = np.array([1, 2, 3])
    print(x_dot_w + b)
    # [[ 1  2  3]
    #  [11 12 13]]

    # 逆伝播の場合は、それぞれの逆伝播の値がバイアスの要素に集約される必要がある
    d_y = np.array([[1, 2, 3], [4, 5, 6]])
    print(d_y)
    # [[1 2 3]
    # [4 5 6]]

    # 2個のデータとした場合、
    # バイアスの逆伝播は、2個のデータ(要素)に対しての微分を、データ(要素)ごとに合算して求める
    # => 0番目の軸(データを単位とした軸)に対して(axis=0)の総和を求める
    db = np.sum(d_y, axis=0)
    print(db)
    # [5 7 9]

