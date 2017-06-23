import numpy as np


# type hintを使いたかったので、わざわざis_printを入れた
def softmax_with_overflow(a: np.ndarray, is_print: bool = False) -> np.ndarray:
    # aは入力信号

    # ソフトマックス関数の分子の計算
    # NumPyのexp関数は指数関数
    exp_a = np.exp(a)

    if is_print:
        print(f'numerator -> {exp_a}')

    # ソフトマックス関数の分母の計算
    # 指数関数の和
    sum_exp_a = np.sum(exp_a)

    if is_print:
        print(f'denominator -> {sum_exp_a}')

    # ソフトマックス関数の結果
    y = exp_a / sum_exp_a

    if is_print:
        print(f'softmax result -> {y}')

    return y


def softmax(a: np.ndarray, is_print: bool = False) -> np.ndarray:
    # aは入力信号

    # 入力信号のうち、最大の値を取得する
    c = np.max(a)

    # 確認
    if is_print:
        print(f'a - c -> {a - c}')

    # ソフトマックス関数の分子の計算
    # NumPyのexp関数は指数関数
    # オーバーフロー対策のため入力信号から最大の値を減算する
    exp_a = np.exp(a - c)
    if is_print:
        print(f'numerator -> {exp_a}')

    # ソフトマックス関数の分母の計算
    # 指数関数の和
    sum_exp_a = np.sum(exp_a)
    if is_print:
        print(f'denominator -> {sum_exp_a}')

    # ソフトマックス関数の結果
    y = exp_a / sum_exp_a
    if is_print:
        print(f'softmax result -> {y}')

    return y


if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = softmax_with_overflow(a, is_print=True)
    # numerator -> [  1.34985881  18.17414537  54.59815003]
    # denominator -> 74.1221542101633
    # softmax result -> [ 0.01821127  0.24519181  0.73659691]

    # 大きな値の計算：オーバーフローあり
    a = np.array([1010, 1000, 990])
    y = softmax_with_overflow(a, is_print=True)
    # 正しく計算されない
    # softmax_function.py:10: RuntimeWarning: overflow encountered in exp
    # exp_a = np.exp(a)
    # numerator -> [ inf  inf  inf]
    # denominator -> inf
    # softmax_function.py:23: RuntimeWarning: invalid value encountered in true_divide
    # y = exp_a / sum_exp_a
    # softmax result -> [ nan  nan  nan]
    # ここで、`nan`は「not a number：不定」という意味になる

    # 大きな値の計算：オーバーフロー無し
    a = np.array([1010, 1000, 990])
    y = softmax(a)
    # a - c -> [  0 -10 -20]
    # numerator -> [  1.00000000e+00   4.53999298e-05   2.06115362e-09]
    # denominator -> 1.0000454019909162
    # softmax result -> [  9.99954600e-01   4.53978686e-05   2.06106005e-09]

    # ニューラルネットワークの出力の確認
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(f'neural_network - y -> {y}')
    # neural_network - y -> [ 0.01821127  0.24519181  0.73659691]

    result = np.sum(y)
    print(f'neural_network - result -> {result}')
    # neural_network - result -> 1.0
