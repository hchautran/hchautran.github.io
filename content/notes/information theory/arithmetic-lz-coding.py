"""HW3 Problem 7: binary arithmetic coding and LZ78, with round-trip checks.
The decoder receives the source length n; headers are excluded from measured lengths.
Run: python3 hw3-coding.py
"""
from fractions import Fraction
from itertools import product
import math
import random

FULL = 1 << 32
HALF, QUARTER = FULL // 2, FULL // 4


def model(theta):
    theta = Fraction(str(theta))
    if not 0 <= theta <= 1 or theta.denominator > 65536:
        raise ValueError('Use a Bernoulli probability with denominator at most 65536')
    return theta.denominator - theta.numerator, theta.denominator


def arithmetic_encode(bits, theta):
    zero, total = model(theta)
    if zero in (0, total):
        expected = int(zero == 0)
        if any(bit != expected for bit in bits):
            raise ValueError('Sequence has zero probability under the model')
        return ''
    low, high, pending, output = 0, FULL - 1, 0, []
    def emit(bit):
        nonlocal pending
        output.append(str(bit))
        output.extend(str(1 - bit) for _ in range(pending))
        pending = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError('Binary symbols required')
        size = high - low + 1
        split = low + size * zero // total
        if bit == 0:
            high = split - 1
        else:
            low = split
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low *= 2
            high = 2 * high + 1
    pending += 1
    emit(int(low >= QUARTER))
    return ''.join(output)


def arithmetic_decode(code, n, theta):
    zero, total = model(theta)
    if zero in (0, total):
        return [int(zero == 0)] * n
    stream = iter(map(int, code))
    def read():
        return next(stream, 0)
    low, high, value = 0, FULL - 1, 0
    for _ in range(32):
        value = 2 * value + read()
    result = []
    for _ in range(n):
        size = high - low + 1
        split = low + size * zero // total
        bit = int(value >= split)
        result.append(bit)
        if bit == 0:
            high = split - 1
        else:
            low = split
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                low -= QUARTER
                high -= QUARTER
                value -= QUARTER
            else:
                break
            low *= 2
            high = 2 * high + 1
            value = 2 * value + read()
    return result


def index_bits(index, width):
    return format(index, f'0{width}b') if width else ''


def lz78_encode(bits):
    dictionary, next_index, prefix, output = {}, 1, 0, []
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError('Binary symbols required')
        key = (prefix, bit)
        if key in dictionary:
            prefix = dictionary[key]
        else:
            width = (next_index - 1).bit_length()
            output.append(index_bits(prefix, width) + str(bit))
            dictionary[key] = next_index
            next_index += 1
            prefix = 0
    if prefix:
        output.append(index_bits(prefix, (next_index - 1).bit_length()))
    return ''.join(output)


def lz78_decode(code, n):
    dictionary, cursor, result, size = [''], 0, [], 0
    while size < n:
        width = (len(dictionary) - 1).bit_length()
        index = int(code[cursor:cursor + width], 2) if width else 0
        cursor += width
        prefix = dictionary[index]
        if size + len(prefix) == n:
            result.append(prefix)
            size = n
            break
        phrase = prefix + code[cursor]
        cursor += 1
        dictionary.append(phrase)
        result.append(phrase)
        size += len(phrase)
    if cursor != len(code):
        raise ValueError('Unexpected trailing bits')
    return list(map(int, ''.join(result)))


def verify():
    for n in range(9):
        for values in product((0, 1), repeat=n):
            bits = list(values)
            assert lz78_decode(lz78_encode(bits), n) == bits
            for theta in (0.1, 0.5, 0.9):
                assert arithmetic_decode(arithmetic_encode(bits, theta), n, theta) == bits
    for theta in (0, 1):
        bits = [theta] * 100
        assert arithmetic_decode(arithmetic_encode(bits, theta), len(bits), theta) == bits


def experiment():
    rng = random.Random(563)
    print('n, arithmetic_mean_bits, lz78_mean_bits, arithmetic_rate, lz78_rate')
    for n in (100, 1000, 100000):
        lengths = []
        for _ in range(100):
            bits = [int(rng.random() < 0.1) for _ in range(n)]
            arithmetic, lz = arithmetic_encode(bits, 0.1), lz78_encode(bits)
            assert arithmetic_decode(arithmetic, n, 0.1) == bits
            assert lz78_decode(lz, n) == bits
            lengths.append((len(arithmetic), len(lz)))
        a = sum(item[0] for item in lengths) / 100
        l = sum(item[1] for item in lengths) / 100
        print(f'{n}, {a:.2f}, {l:.2f}, {a/n:.6f}, {l/n:.6f}')
    print('H(0.1) =', -0.1 * math.log2(0.1) - 0.9 * math.log2(0.9))


if __name__ == '__main__':
    verify()
    experiment()
