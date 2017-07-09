# あまり意味がわからなかったので、コミットする必要もないかもしれない

import numpy as np


def cross_entropy_error(y, t, is_one_hot=False, is_print=False):
    """バッチ対応版：交差エントロピー誤差の取得

    :param y: ニューラルネットワークの出力
    :param t: 教師データ
    :param is_one_hot: one-hot表現の場合True、ラベルとして与えられた時False
    :param is_print:
    :return:
    """

    # yの次元数が1 = データ１つあたりの交差エントロピー誤差を求める場合
    # ndimは配列の次元数を表す
    if y.ndim == 1:
        # データの形状を整形
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    if is_one_hot:
        t = t.argmax(axis=1)

        # バッチの枚数で正規化し、1枚あたりの平均交差エントロピー誤差を求める
        return -np.sum(t * np.log(y)) / batch_size

    # 教師データがラベルで与えられた時
    # ・np.arrange(batch_size)は、0〜batch_size - 1までの配列を生成
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
