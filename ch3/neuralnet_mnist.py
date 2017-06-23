import os
import sys

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
from sigmoid_function import sigmoid_function
from softmax_function import softmax


def get_data():
    # normalize=Trueのため、関数の内部では画像の各ピクセルの値を255で除算し、
    # データの値が0.0~1.0の範囲に収まるように変換した

    _, (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x, is_print=False):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 各層の重みの確認
    if is_print:
        print(f'x.shape -> {x.shape}')
        print(f'x.shape[0] -> {x.shape[0]}')
        print(f'W1.shape -> {W1.shape}')
        print(f'W2.shape -> {W2.shape}')
        print(f'W3.shape -> {W3.shape}')

    # 一つ目の隠れ層
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)

    # 二つ目の隠れ層
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)

    # 出力層
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    # 各ラベル(0~9)の確率がNumPy配列として返される
    return y


def run(x, t, network):
    accuracy_cnt = 0

    # xに格納された画像データを1枚ずつfor文で取り出す
    for i in range(len(x)):

        # predict()関数により分類する
        y = predict(network, x[i])

        # 確率リスト(y)から、最も大きな値のIndex(何番目の要素が一番確率が高いか)を取り出し、予測結果とする
        # NumPyのargmax()関数は、引数xに与えられた配列で最大の値を持つ要素のIndexを取得する
        # (最も確率の高い要素のIndexを取得)
        p = np.argmax(y)

        # ニューラルネットワークが予測した答え(p)と、正解ラベル(t[i])を比較して
        # 正解した割合を認識精度(accuracy)とする
        if p == t[i]:
            accuracy_cnt += 1

    print(f'Accuracy: {float(accuracy_cnt) / len(x)}')
    # => Accuracy: 0.9352


def run_batch(x, t, network):
    batch_size = 100
    accuracy_cnt = 0

    # rangeの第三引数にstepを与えることで、batch_size飛ばしのリストを作成する
    # list(range(0, 10, 3)) => [0, 3, 6, 9]
    for i in range(0, len(x), batch_size):
        # リストを元に、入力データからバッチを抜き出す
        # 入力データのi番目から i + batch_n番目のデータまで取り出す
        # => この例では、先頭から100枚ずつ、バッチとして取り出す
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)

        # axis=1で、100x10の配列の中で一次元目の要素ごとに、一次元目を軸として、最大値のIndexを取得
        # => 0次元目は最初の次元に対応
        p = np.argmax(y_batch, axis=1)

        # バッチ単位で分類した結果と、実際の答えを比較
        # NumPy配列同士で比較演算処理によって、True/Falseからなるブーリアン配列を作成し、
        # Trueの個数を算出する
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print(f'Accuracy: {float(accuracy_cnt) / len(x)}')


if __name__ == '__main__':
    # ニューラルネットワークによる推論処理 => どれだけ正しく分類できるかを評価する
    # MNISTデータセットを取得し、ネットワークを生成
    x, t = get_data()
    network = init_network()
    # 1枚ずつ実行
    run(x, t, network)
    # => Accuracy: 0.9352

    # 各層の重みの確認
    predict(network, x, is_print=True)
    # x.shape -> (10000, 784)
    # x.shape[0] -> 10000
    # W1.shape -> (784, 50)
    # W2.shape -> (50, 100)
    # W3.shape -> (100, 10)

    # 100枚のバッチで実行
    run_batch(x, t, network)
    # => Accuracy: 0.9352

    # argmax()の引数axis=1を確認
    x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
    y = np.argmax(x, axis=1)
    print(f'argmax(axis=1) -> {y}')
    # => argmax(axis=1) -> [1 2 1 0]

    # NumPy配列同士の比較演算子により、ブーリアン配列ができるかを確認
    y = np.array([1, 2, 1, 0])
    t = np.array([1, 2, 0, 0])
    print(f'y == t -> {y == t}')
    # 配列間で同じIndexで要素の値が一致している時はTrue、そうでないときはFalse
    # => y == t -> [ True  True False  True]

    # ブーリアン配列のTrueの個数を確認
    result = np.sum(y == t)
    print(f'sum(y == t) -> {result}')
    # => sum(y == t) -> 3



