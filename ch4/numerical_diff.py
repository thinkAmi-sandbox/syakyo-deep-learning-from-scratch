import numpy as np


def numerical_diff(f, x, is_print=False):
    """数値微分(数値勾配)を求める

    :param f: 関数f
    :param x: 関数fへの引数x

    :return: 数値微分(数値勾配)
    """

    # h(小さな変化量)は丸め誤差を発生させない程度にする
    # 0.0001程度の値を用いると、良い結果が得られる
    h = 1e-4

    # 1e-10のような非常に小さな数の場合、正しく数値が表現できない
    if is_print:
        print(np.float32(1e-50))
        # => 0.0

    # 数値微分の誤差を減らす工夫として、中心差分(3点近似)の公式を使う
    # 中心差分： xを中心として、その前後の差分を計算すること
    return (f(x + h) - f(x - h)) / (2 * h)
