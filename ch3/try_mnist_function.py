import os, sys
# 親ディレクトリのファイルをimportするための設定
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


def try_load_mnist():
    # 最初の呼び出しは数分待つ
    # 正規化なし、入力画像は1次元配列、正解となるラベルだけ出す
    result = load_mnist(normalize=False, flatten=True, one_hot_label=False)

    # load_mnist関数の戻り値は、(訓練画像, 訓練ラベル), (テスト画像, テストラベル) となる
    (x_train, t_train), (x_test, t_test) = result

    # それぞれのデータの形状を出力
    print(f'x_train.shape -> {x_train.shape}')
    print(f't_train.shape -> {t_train.shape}')
    print(f'x_test.shape -> {x_test.shape}')
    print(f't_test.shape -> {t_test.shape}')


if __name__ == '__main__':
    try_load_mnist()
    # x_train.shape -> (60000, 784)
    # t_train.shape -> (60000,)
    # x_test.shape -> (10000, 784)
    # t_test.shape -> (10000,)
