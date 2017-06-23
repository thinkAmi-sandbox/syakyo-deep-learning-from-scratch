import numpy as np

# pyvenv + matplotlibを使うと、

# RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly
# if Python is not installed as a framework. See the Python documentation for more information on installing Python
# as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends.
# If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'.
# See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.

# というエラーが出る。そのため、以下を参考に、`~/.matplotlib/matplotlibrc`ファイルを作成し、backendにTkAggを指定する
# http://qiita.com/katryo/items/918667f28301fdec89ba

import matplotlib.pylab as plt


def simple_step_function(x):
    """ステップ関数

    ・入力が0を超えたら1を出力
    ・それ以外は0を出力
    """
    if x > 0:
        return 1
    else:
        return 0


def try_numpy_array(list_arg):
    x = np.array(list_arg)
    print(f'x -> {x}')

    # numpy配列xに対して不等号の演算を行うと、
    # 配列の各要素に対して不等号の演算が行われる
    # => その結果として、boolean配列yが生成される
    #    0より大きい要素はTrue, 0以下の要素はFalse
    y = x > 0
    print(f'y -> {y}')

    # ステップ関数は0か1のint型を出力する関数
    # => numpy配列yの要素の形をbooleanからintへ、astype()メソッドを使って変換する
    #    astype()メソッドは、引数に変換後の型を指定する(今回はint型)
    #    Pythonでは、booleanをintへ変換すると、Trueは1に、Falseは0に変換される
    z = y.astype(np.int)
    print(f'z -> {z}')


def step_function(x):
    """NumPy配列を引数ニトリ、配列の各要素に対してステップ関数を実行し、結果を配列として返す"""
    return np.array(x > 0, dtype=np.int)


if __name__ == '__main__':
    try_numpy_array([-1.0, 1.0, 2.0])
    # x -> [-1.  1.  2.]
    # y -> [False  True  True]
    # z -> [0 1 1]

    # -5.0から5.0までの範囲を0.1刻みでNumpy配列を生成する([-5.0, -4.9, ... , 4.9])
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)

    # x, y配列をプロットする
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
    # ステップ関数は、0を境にして、出力が0から1へと切り替わる
    plt.show()
