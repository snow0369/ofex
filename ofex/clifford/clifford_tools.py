from collections import Counter
from itertools import product
from typing import Tuple, Optional, List

import galois
import numpy as np
from galois import FieldArray
from openfermion import QubitOperator

from ofex.operators.types import SinglePauli

gf = galois.GF(2)


def single_pauli_to_tableau(pauli: SinglePauli, num_qubits: int) -> FieldArray:
    v = np.zeros(2 * num_qubits, dtype=int)
    for idx, p in pauli:
        if p == "I":
            continue
        elif p == "X":
            v[idx] = 1
        elif p == "Y":
            v[idx] = 1
            v[idx + num_qubits] = 1
        elif p == "Z":
            v[idx + num_qubits] = 1
        else:
            raise ValueError
    return gf(v)


def pauli_to_tableau(pauli_list: QubitOperator, num_qubits: int) \
        -> Tuple[FieldArray, np.ndarray]:
    """
    Returns:
        G: [G_x/G_z] integer 2D array, 2N by r (N = #of qubits, r = number of terms)
        coeff: (vector len = r) Corresponding coefficients
    """
    coeff_type = type(list(pauli_list.terms.values())[0])
    pauli_list = list(pauli_list.terms.items())

    r = len(pauli_list)

    g_mat = np.zeros((2 * num_qubits, r), dtype=int)
    coeff_arr = np.zeros(r, dtype=coeff_type)
    for i, (p, coeff) in enumerate(pauli_list):
        v = single_pauli_to_tableau(p, num_qubits)
        g_mat[:, i] = v
        coeff_arr[i] = coeff
    return gf(g_mat), coeff_arr


def tableau_to_pauli(mat: np.ndarray,
                     ph: Optional[np.ndarray] = None,
                     coeff: Optional[np.ndarray] = None) -> List[QubitOperator]:
    num_qubits, num_op = mat.shape
    if num_qubits % 2 != 0:
        raise ValueError
    num_qubits //= 2
    if ph is not None:
        if ph.shape[0] != num_op:
            raise ValueError
    else:
        ph = gf(np.zeros(num_op, dtype=int))
    if coeff is not None:
        if coeff.shape[0] != num_op:
            raise ValueError
        coeff = [-c if ph[i] else c for i, c in enumerate(coeff)]
    else:
        coeff = [-1.0 if p else 1.0 for p in ph]

    ret = list()
    for op_idx in range(num_op):
        op_vec, c = mat[:, op_idx], coeff[op_idx]
        op = "".join(["Y" if x and z else "Z" if z else "X" if x else "I"
                      for x, z in zip(op_vec[:num_qubits], op_vec[num_qubits:])])
        op = tuple([(q_idx, p) for q_idx, p in enumerate(op) if p != "I"])
        ret.append(QubitOperator(op, coeff[op_idx]))
    return ret


def dot_tableau(a: FieldArray, b: FieldArray) -> int:
    """

    Args:
        a: vector
        b: vector

    Returns:
        dot: 0: commute, 1: anti-commute
    """
    if not a.shape == b.shape:
        raise ValueError
    num_qubits = a.shape[0] // 2
    ax, az = a[:num_qubits], a[num_qubits:]
    bx, bz = b[:num_qubits], b[num_qubits:]
    return int(np.dot(ax, bz) + np.dot(az, bx))


def print_tableau(mat: FieldArray, ph: Optional[FieldArray]):
    n_row, n_pauli = mat.shape
    n_qubits = n_row // 2
    for i in range(n_row):
        print(mat[i])
        if i == n_qubits - 1:
            print("==" * (n_pauli + 1))
    if ph is not None:
        print("==" * (n_pauli + 1))
        print(ph)


def print_tableau_side_by_side(mat1: FieldArray, ph1: Optional[FieldArray],
                               mat2: FieldArray, ph2: Optional[FieldArray]):
    n_row, n_pauli1 = mat1.shape
    n_row2, n_pauli2 = mat2.shape
    n_qubits = n_row // 2
    if n_row != n_row2:
        raise ValueError
    for i in range(n_row):
        print(mat1[i], end="\t")
        print(mat2[i])
        if i == n_qubits - 1:
            print("=" * (2 * n_pauli1 + 1) + "\t" + "=" * (2 * n_pauli2 + 1))
    print("=" * (2 * n_pauli1 + 1) + "\t" + "=" * (2 * n_pauli2 + 1))
    if ph1 is not None:
        print(ph1, end="\t")
    else:
        print("  " * (n_pauli1 + 1), end="\t")
    if ph2 is not None:
        print(ph2)


def xor_mat(s_t_pair: List[Tuple[int, int]],
            num_entries: int):
    """
    w = X v
    w[j] = v[i] + v[j] if (i, j) in s_t_pair
           v[j] else
    """
    new_s_t_pair = list()
    for pair, num_occ in Counter(s_t_pair):
        if num_occ % 2 == 1:
            new_s_t_pair.append(pair)
    s_t_pair = new_s_t_pair
    for (i, (si, ti)), (j, (sj, tj)) in product(enumerate(s_t_pair), repeat=2):
        if i == j:
            continue
        if si == tj or ti == sj:
            raise ValueError("Non-Commuting XOR")

    ret_mat = gf(np.eye(num_entries, dtype=int))
    for s, t in s_t_pair:
        ret_mat[s, t] = 1
    return ret_mat


def is_zero_gf(a: FieldArray):
    return np.allclose(np.array(a), 0)


def is_equal_gf(a: FieldArray, b: FieldArray):
    return np.allclose(np.array(a + b), 0)
