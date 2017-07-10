import sys, os
sys.path.append(os.pardir)
import datetime
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net_back_using_propagation import TwoLayerNetBackPropagation


def record_by_epoch():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # ハイパーパラメータ
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 2
    learning_rate = 0.1

    network = TwoLayerNetBackPropagation(input_size=784, hidden_size=50, output_size=10)

    print(f'開始時間 -> {datetime.datetime.now()}')

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 1エポックあたりの繰り返し数
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        # ミニバッチの取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 誤差逆伝播法による、勾配の計算
        grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 学習経過の記録
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # 途中経過時間
        if (i + 1) % 1000 == 0:
            print(f'{i + 1}回終了 -> {datetime.datetime.now()}')

        if i % iter_per_epoch == 0:
            print(f'{i}, {iter_per_epoch}')
            train_acc = network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)

            test_acc = network.accuracy(x_test, t_test)
            test_acc_list.append(test_acc)

            print(f'train acc, test acc | {str(train_acc)}, {str(test_acc)}')

    print(f'終了時間 -> {datetime.datetime.now()}')


if __name__ == '__main__':
    record_by_epoch()
