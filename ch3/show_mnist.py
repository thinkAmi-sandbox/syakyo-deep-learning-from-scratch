# 使うには pip install pillow が必要
# PILは開発停滞でPython2.7までの対応にとどまるため

import os, sys
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    # NumPyとして格納された画像データを、PIL用のデータオブジェクトに変換する
    # np.uint8は、8bitの符号なし整数で、0,...,255の値を表す
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def try_image_show():
    # flatten=Trueで読み込んだ画像は、NumPy配列として1次元で格納されている
    (x_train, t_train), _ = load_mnist(normalize=False, flatten=True, one_hot_label=False)

    img = x_train[0]
    label = t_train[0]
    print(f'label -> {label}')
    # => label -> 5

    print(f'img.shape before -> {img.shape}')
    # => img.shape before -> (784,)

    # NumPy配列の1次元では画像を表示できないので、元の形状である28x28サイズに再変形する必要がある
    # NumPy配列の形状の変形はreshape()メソッドによって行う
    # reshape()メソッドの引数に、希望する形状を指定する
    img = img.reshape(28, 28)
    print(f'img.shape after  -> {img.shape}')
    # => img.shape after  -> (28, 28)
    img_show(img)


if __name__ == '__main__':
    try_image_show()
