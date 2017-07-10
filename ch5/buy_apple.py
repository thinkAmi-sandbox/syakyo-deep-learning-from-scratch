from layer_navie import MulLayer


def main():
    apple_unit_price = 100
    apple_quantity = 2
    tax = 1.1

    # layer
    apple_mul_layer = MulLayer()
    tax_mul_layer = MulLayer()

    # forward
    apple_price = apple_mul_layer.forward(apple_unit_price, apple_quantity)
    print(f'順伝播：apple price -> {apple_price}')

    total = tax_mul_layer.forward(apple_price, tax)
    print(f'順伝播：total -> {total}')

    # backward
    d_total = 1
    d_apple_price, d_tax = tax_mul_layer.backward(d_total)
    print(f'逆伝播：d apple price -> {d_apple_price}')
    print(f'逆伝播：d tax -> {d_tax}')

    d_apple_unit_price, d_apple_quantity = apple_mul_layer.backward(d_apple_price)
    print(f'逆伝播：d apple unit price -> {d_apple_unit_price}')
    print(f'逆伝播：d apple quantity -> {d_apple_quantity}')


if __name__ == '__main__':
    main()
    # 順伝播：apple price -> 200
    # 順伝播：total -> 220.00000000000003
    # 逆伝播：d apple price -> 1.1
    # 逆伝播：d tax -> 200
    # 逆伝播：d apple unit price -> 2.2
    # 逆伝播：d apple quantity -> 110.00000000000001
