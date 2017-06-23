import numpy as np
import traceback
import sys


def print_dimensional_array(x):
    print(x)

    print(f'NumPy配列の次元数(np.ndim()) -> {np.ndim(x)}')

    # 多次元配列で統一した表記とするため、x.shapeはタプルが返ってくる
    print(f'NumPy配列の要素数(x.shape) -> {x.shape}')
    print(x.shape[0])


def print_dot(a, b):
    print(f'x.shape -> {a.shape}')
    print(f'y.shape -> {b.shape}')

    c = np.dot(a, b)
    print(f'dot -> {c}')


if __name__ == '__main__':
    print('---1次元配列---')
    a = np.array([1, 2, 3, 4])
    print_dimensional_array(a)
    # => [1 2 3 4]
    # => NumPy配列の次元数(np.ndim()) -> 1
    # => NumPy配列の要素数(x.shape) -> (4,)
    # => 4

    print('---2次元配列---')
    b = np.array([[1, 2], [3, 4], [5, 6]])
    print_dimensional_array(b)
    # => [[1 2]
    #     [3 4]
    #     [5 6]]
    # => NumPy配列の次元数(np.ndim()) -> 2
    # => NumPy配列の要素数(x.shape) -> (3, 2)
    # => 3

    print('---行列の積：2x2 * 2x2---')
    c = np.array([[1, 2], [3, 4]])
    d = np.array([[5, 6], [7, 8]])
    print_dot(c, d)
    # x.shape -> (2, 2)
    # y.shape -> (2, 2)
    # dot -> [[19 22]
    #         [43 50]]

    print('---行列の積：2x3 * 3x2---')
    e = np.array([[1, 2, 3], [4, 5, 6]])
    f = np.array([[1, 2], [3, 4], [5, 6]])
    print_dot(e, f)
    # x.shape -> (2, 3)
    # y.shape -> (3, 2)
    # dot -> [[22 28]
    #         [49 64]]

    print('---行列の積：2x3 * 2x2---')
    g = np.array([[1, 2, 3], [4, 5, 6]])
    h = np.array([[1, 2], [3, 4]])
    try:
        print_dot(g, h)
    except ValueError:
        _, msg, tb = sys.exc_info()
        traceback.print_tb(tb)
        print(msg)
        # x.shape -> (2, 3)
        # y.shape -> (2, 2)
        # File "multi_dimensional_array.py", line 57, in <module>
        # print_dot(g, h)
        # File "multi_dimensional_array.py", line 20, in print_dot
        # z = np.dot(x, y)
        # shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)

    print('---行列の積：3x4 * 4x2---')
    i = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    j = np.array([[1,2],[3,4],[5,6],[7,8]])
    print_dot(i, j)
    # x.shape -> (3, 4)
    # y.shape -> (4, 2)
    # dot -> [[ 50  60]
    #         [114 140]
    #         [178 220]]

    print('---行列の積：3x2 * 2---')
    k = np.array([[1, 2], [3, 4], [5, 6]])
    l = np.array([7, 8])
    print_dot(k, l)
    # x.shape -> (3, 2)
    # y.shape -> (2,)
    # dot -> [23 53 83]
