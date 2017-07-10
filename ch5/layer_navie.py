class MulLayer:
    """乗算レイヤ"""

    def __init__(self):
        """順伝播時の入力値を保持するために属性を用意"""
        self.x = None
        self.y = None

    def forward(self, x, y):
        """順伝播を出力

        :param x:
        :param y:
        :return:
        """
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        """逆伝播を計算する

        xとyをひっくり返して、戻す

        :param dout: 順伝播の際の出力変数に対する微分
        :return:
        """
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    """加算レイヤ"""

    def __init__(self):
        pass

    def forward(self, x, y):
        """2つの引数を取り、それらを加算して出力する

        :param x:
        :param y:
        :return:
        """
        out = x + y
        return out

    def backward(self, dout):
        """上流から伝わってきた微分(dout)をそのまま下流に流す

        :param dout: 上流より伝わってきた微分
        :return:
        """
        dx = dout * 1
        dy = dout * 1
        return dx, dy
