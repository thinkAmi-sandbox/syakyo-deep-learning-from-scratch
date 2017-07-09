import numpy as np


def mean_squared_error(y, t, is_print=False):
    """2乗和誤差を計算する

    :param np.ndarray y: ニューラルネットワークの出力(ソフトマックス関数の出力)
    :param np.ndarray t: 教師データ
    :param bool is_print: 2乗和誤差の結果を出力する場合、True
    """
    result = 0.5 * np.sum((y - t) ** 2)
    if is_print:
        print(f'損失関数(2乗和誤差)の結果(型{type(result)}) -> {result}')

    return result


if __name__ == '__main__':
    # index=2 を正解とする教師データ
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # 例1 index=2の確率が最も高い(ソフトマックス関数の出力 = 0.6)場合
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    mean_squared_error(np.array(y1), np.array(t), is_print=True)
    # => 損失関数(2乗和誤差)の結果(型<class 'numpy.float64'>) -> 0.09750000000000003

    # 例2 index=7の確率が最も高い場合
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    mean_squared_error(np.array(y2), np.array(t), is_print=True)
    # => 損失関数(2乗和誤差)の結果(型<class 'numpy.float64'>) -> 0.5975
