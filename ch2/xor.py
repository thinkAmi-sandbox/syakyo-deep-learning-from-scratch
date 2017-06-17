from and_nand_or import and_, nand_, or_


def xor_(x1, x2):
    s1 = nand_(x1, x2)
    s2 = or_(x1, x2)
    y = and_(s1, s2)
    return y


if __name__ == '__main__':
    print('--- XOR ---')
    print(f'0, 0 => {xor_(0, 0)}')
    print(f'1, 0 => {xor_(1, 0)}')
    print(f'0, 1 => {xor_(0, 1)}')
    print(f'1, 1 => {xor_(1, 1)}')