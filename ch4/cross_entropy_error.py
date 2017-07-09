import numpy as np


def cross_entropy_error(y, t, is_print=False):
    """交差エントロピー誤差を計算する

    :param np.ndarray y: ニューラルネットワークの出力(ソフトマックス関数の出力)
    :param np.ndarray t: 教師データ
    :param bool is_print: 2乗和誤差の結果を出力する場合、True
    """
    # ・deltaを使う理由
    #   np.log(0)のような計算が発生した場合に、np.log(0)は-inf(マイナス無限大)になる
    #   その結果、それ以上計算を進められなくなってしまう
    #   それを回避するため、微小な値を追加して、マイナス無限大を発生させないようにする
    # ・pythonの浮動小数点リテラル
    #   https://docs.python.jp/3/reference/lexical_analysis.html#floating-point-literals
    #   整数部と指数は、常に10を基数として解釈される
    #   1e-7の場合、10のマイナス7乗(0.0000001)となる
    delta = 1e-7

    result = -np.sum(t * np.log(y + delta))
    if is_print:
        print(f'損失関数(交差エントロピー誤差)の結果(型{type(result)}) -> {result}')

    return result


if __name__ == '__main__':
    # index=2 を正解とする教師データ
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # 例1 index=2の確率が最も高い(ソフトマックス関数の出力 = 0.6)場合
    y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    cross_entropy_error(np.array(y1), np.array(t), is_print=True)
    # => 損失関数(交差エントロピー誤差)の結果(型<class 'numpy.float64'>) -> 0.510825457099338

    # 例2 index=7の確率が最も高い場合
    y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    cross_entropy_error(np.array(y2), np.array(t), is_print=True)
    # => 損失関数(交差エントロピー誤差)の結果(型<class 'numpy.float64'>) -> 2.302584092994546


