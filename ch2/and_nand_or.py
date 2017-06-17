import numpy as np


def and_without_bias(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1


def and_(x1, x2):
    x = np.array([x1, x2])      # 入力
    w = np.array([0.5, 0.5])    # 重み
    b = -0.7                    # バイアス
    # NumPy配列の乗算の場合、要素数が同じならば、その要素同士が乗算される
    # numpy.sum()メソッドは、各要素の総和が計算される
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def nand_(x1, x2):
    x = np.array([x1, x2])
    # NANDは重みとバイアスだけがANDと違う
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp < 0:
        return 0
    else:
        return 1


def or_(x1, x2):
    x = np.array([x1, x2])
    # ORも、重みとバイアスだけがANDと違う
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp < 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    print('--- AND without bias ---')
    print(f'0, 0 => {and_without_bias(0, 0)}')
    print(f'1, 0 => {and_without_bias(1, 0)}')
    print(f'0, 1 => {and_without_bias(0, 1)}')
    print(f'1, 1 => {and_without_bias(1, 1)}')

    print('--- AND ---')
    print(f'0, 0 => {and_(0, 0)}')
    print(f'1, 0 => {and_(1, 0)}')
    print(f'0, 1 => {and_(0, 1)}')
    print(f'1, 1 => {and_(1, 1)}')

    print('--- NAND ---')
    print(f'0, 0 => {nand_(0, 0)}')
    print(f'1, 0 => {nand_(1, 0)}')
    print(f'0, 1 => {nand_(0, 1)}')
    print(f'1, 1 => {nand_(1, 1)}')

    print('--- OR ---')
    print(f'0, 0 => {or_(0, 0)}')
    print(f'1, 0 => {or_(1, 0)}')
    print(f'0, 1 => {or_(0, 1)}')
    print(f'1, 1 => {or_(1, 1)}')
