import numpy as np
from itertools import product

from ofex.utils.binary import int_to_binary, hamming_weight, binary_to_int

TRIAL = 10_000
for idx_trial, _lsb_first in product(range(TRIAL), [True, False]):
    rand_int = np.random.randint(0, 1023)
    rand_length = np.min([int(np.log2(rand_int + 1)), np.random.randint(1, 11)])
    b = int_to_binary(rand_int, rand_length, _lsb_first)
    if idx_trial == 10:
        print(f"{'LSB first' if _lsb_first else 'MSB first'}, {b}, {rand_int}, {bin(rand_int)}")
    assert binary_to_int(b, _lsb_first) == rand_int, f"{b} -> {binary_to_int(b, _lsb_first)} != {rand_int}"
    assert hamming_weight(rand_int) == sum(b)
