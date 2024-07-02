from __future__ import annotations

from copy import copy
from typing import Tuple


def int_to_binary(integer, length: int, lsb_first: bool) -> Tuple[int, ...]:
    if lsb_first:
        return tuple([int(x) for x in list(bin(integer)[2:].rjust(length, "0"))][::-1])
    else:
        return tuple([int(x) for x in list(bin(integer)[2:].rjust(length, "0"))])


def binary_to_int(binary: Tuple[int, ...], lsb_first: bool) -> int:
    if lsb_first:
        return sum([2 ** i for i in range(len(binary)) if binary[i]])
    else:
        return sum([2 ** i for i in range(len(binary)) if binary[::-1][i]])


def hamming_weight(x: int) -> int:
    c = 0
    if x < 0:
        raise ValueError("Negative integers are not allowed.")
    while x != 0:
        c += x & 1
        x >>= 1
    return c


if __name__ == "__main__":
    import numpy as np
    from itertools import product

    trial = 10_000
    for idx_trial, lsb_first in product(range(trial), [True, False]):
        rand_int = np.random.randint(0, 1023)
        rand_length = np.min([int(np.log2(rand_int + 1)), np.random.randint(1, 11)])
        b = int_to_binary(rand_int, rand_length, lsb_first)
        if idx_trial == 10:
            print(f"{'LSB first' if lsb_first else 'MSB first'}, {b}, {rand_int}, {bin(rand_int)}")
        assert binary_to_int(b, lsb_first) == rand_int, f"{b} -> {binary_to_int(b, lsb_first)} != {rand_int}"
        assert hamming_weight(rand_int) == sum(b)
