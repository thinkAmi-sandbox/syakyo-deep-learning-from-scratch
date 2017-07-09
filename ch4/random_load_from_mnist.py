import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist


def random_load_from_mnist(batch_size=10, is_print=False):
    # one-hot表現として、データを読み込む
    # (正解となるラベルだけが1で、他は0)
    # ・訓練データ：60,000個
    # ・入力データ：784列 (28x28サイズ)
    # ・教師データ：10列
    (x_train, t_train), _ = load_mnist(normalize=True, one_hot_label=True)

    if is_print:
        print(f'(訓練データ, 入力データ) -> {x_train.shape}')
        print(f'(訓練データ, 教師データ) -> {t_train.shape}')

    train_size = x_train.shape[0]

    # 訓練データの中からランダムにbatch_size分だけ抽出する
    # np.random.choice()で、0~60000の数字の中からランダムで10個の数字を選ぶ
    batch_mask = np.random.choice(train_size, batch_size)
    if is_print:
        print(f'ミニバッチとして選び出すindex -> {batch_mask}')

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    return x_batch, t_batch


if __name__ == '__main__':
    x, t = random_load_from_mnist(is_print=True)
    # => (訓練データ, 入力データ) -> (60000, 784)
    # => (訓練データ, 教師データ) -> (60000, 10)
    # => ミニバッチとして選び出すindex -> [ 8040 47295 57466 58863 41766 38563 40438  6264  6963 59651]

    print(f'x_batch -> {x}')
    # x_patch -> [[ 0.  0.  0. ...,  0.  0.  0.]
    #             [ 0.  0.  0. ...,  0.  0.  0.]
    # [ 0.  0.  0. ...,  0.  0.  0.]
    # ...,
    # [ 0.  0.  0. ...,  0.  0.  0.]
    # [ 0.  0.  0. ...,  0.  0.  0.]
    # [ 0.  0.  0. ...,  0.  0.  0.]]

    print(f't_batch -> {t}')
    # t_patch -> [[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
    #             [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
    # [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
    # [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
    # [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
    # [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
    # [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
    # [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
    # [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
    # [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

