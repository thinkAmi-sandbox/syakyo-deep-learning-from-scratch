from layer_navie import MulLayer, AddLayer


def main():
    apple_unit_price = 100
    apple_quantity = 2
    orange_unit_price = 150
    orange_quantity = 3
    tax = 1.1

    # layer
    apple_mul_layer = MulLayer()
    orange_mul_layer = MulLayer()
    apple_orange_add_layer = AddLayer()
    tax_mul_layer = MulLayer()

    # forward
    apple_price = apple_mul_layer.forward(apple_unit_price, apple_quantity)
    print(f'順伝播：apple price -> {apple_price}')

    orange_price = orange_mul_layer.forward(orange_unit_price, orange_quantity)
    print(f'順伝播：orange price -> {orange_price}')

    all_price = apple_orange_add_layer.forward(apple_price, orange_price)
    print(f'順伝播：all price -> {all_price}')

    total = tax_mul_layer.forward(all_price, tax)
    print(f'順伝播：total -> {total}')

    # backward
    d_total = 1
    d_all_price, d_tax = tax_mul_layer.backward(d_total)
    print(f'逆伝播：d all price -> {d_all_price}')
    print(f'逆伝播：d tax -> {d_tax}')

    d_apple_price, d_orange_price = apple_orange_add_layer.backward(d_all_price)
    print(f'逆伝播：d apple price -> {d_apple_price}')
    print(f'逆伝播：d orange price -> {d_orange_price}')

    d_orange_unit_price, d_orange_quantity = orange_mul_layer.backward(d_orange_price)
    print(f'逆伝播：d orange unit price -> {d_orange_unit_price}')
    print(f'逆伝播：d orange quantity -> {d_orange_quantity}')

    d_apple_unit_price, d_apple_quantity = apple_mul_layer.backward(d_apple_price)
    print(f'逆伝播：d apple unit price -> {d_apple_unit_price}')
    print(f'逆伝播：d apple quantity -> {d_apple_quantity}')


if __name__ == '__main__':
    main()
    # 順伝播：apple price -> 200
    # 順伝播：orange price -> 450
    # 順伝播：all price -> 650
    # 順伝播：total -> 715.0000000000001
    # 逆伝播：d all price -> 1.1
    # 逆伝播：d tax -> 650
    # 逆伝播：d apple price -> 1.1
    # 逆伝播：d orange price -> 1.1
    # 逆伝播：d orange unit price -> 3.3000000000000003
    # 逆伝播：d orange quantity -> 165.0
    # 逆伝播：d apple unit price -> 2.2
    # 逆伝播：d apple quantity -> 110.00000000000001
