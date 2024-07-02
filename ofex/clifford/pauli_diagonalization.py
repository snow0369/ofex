from typing import Union, List, Tuple

import numpy as np
from galois import FieldArray
from openfermion import QubitOperator

from ofex.clifford.clifford_tools import pauli_to_tableau, gf, is_zero_gf, print_tableau_side_by_side, is_equal_gf, \
    print_tableau
from ofex.clifford.simulation import clifford_apply
from ofex.clifford.standard_operators import hadamard, clifford_op_str, cx, cz, s_gate


def diagonalizing_clifford(pauli_list: Union[List[QubitOperator], QubitOperator],
                           num_qubits: int,
                           debug=False) \
        -> Tuple[FieldArray, np.ndarray, List[str]]:
    """
        Returns:
            mat : Pauli tableau of transformed operators
            coeff : coefficients of transformed operators
            clifford_history : Operation history
    """

    clifford_list: List[str] = list()

    # INIT
    a_mat, a_coeff = pauli_to_tableau(pauli_list, num_qubits)
    a_mat = gf(a_mat)
    num_qubits, num_paulis = a_mat.shape
    assert num_qubits % 2 == 0
    num_qubits = num_qubits // 2

    # Commute Check
    xmat = a_mat[:num_qubits, :]
    zmat = a_mat[num_qubits:, :]
    if not is_zero_gf(xmat.T @ zmat + zmat.T @ xmat):
        raise ValueError("Non-commuting set!")

    if debug:
        print("INIT")
        print(a_mat)

    # 1. FIRST GAUSSIAN
    b_mat = a_mat.T.row_reduce().T
    tmp_mat = gf(np.zeros(b_mat.shape, dtype=b_mat.dtype))
    num_ind_paulis = 0
    for i in range(b_mat.shape[1]):
        if not is_zero_gf(b_mat[:, i]):
            tmp_mat[:, i] = b_mat[:, i]
            num_ind_paulis += 1
    b_mat = gf(tmp_mat[:, :num_ind_paulis])
    b_ph = gf([0 for _ in range(num_ind_paulis)])

    if debug:
        print("GAUSS")
        print_tableau_side_by_side(a_mat, None, b_mat, b_ph)

    # 2. MAX X RANK
    c_mat, c_ph = gf(b_mat), gf(b_ph)
    for i in range(num_qubits):
        c_mat_x = c_mat[:num_qubits, :]
        r = np.linalg.matrix_rank(c_mat_x.T)
        c_mat_x_h = hadamard(c_mat, c_ph, i)[0][:num_qubits, :]
        r_hi = np.linalg.matrix_rank(c_mat_x_h.T)
        if r < r_hi:
            c_mat, c_ph = hadamard(c_mat, c_ph, i)
            clifford_list.append(clifford_op_str("H", i))

    if debug:
        print("\nMAX_X_RANK")
        print(f"rank_X : {np.linalg.matrix_rank(b_mat[:num_qubits, :].T)} ->"
              f"{np.linalg.matrix_rank(c_mat[:num_qubits, :].T)}")
        print_tableau_side_by_side(b_mat, b_ph, c_mat, c_ph)
        print(clifford_list)

    # 3-1. ZEROING OUT UPPER TRIANGULAR MATRIX
    d_mat, d_ph = gf(c_mat), gf(c_ph)
    for i in range(min(d_mat.shape[0] // 2, d_mat.shape[1])):
        if d_mat[i, i] == 0:
            for j in range(i + 1, d_mat.shape[1]):
                if d_mat[i, j] == 1:
                    d_mat[:, [i, j]] = d_mat[:, [j, i]]
                    break
            else:
                for j in range(i + 1, num_qubits):
                    if d_mat[j, i] == 1:
                        d_mat[[i, j], :] = d_mat[[j, i], :]
                        d_mat[[i + num_qubits, j + num_qubits], :] = d_mat[[j + num_qubits, i + num_qubits], :]
                        clifford_list.append(clifford_op_str("QSW", i, j))
                        break
    for i in range(min(d_mat.shape[0] // 2, d_mat.shape[1])):
        if d_mat[i, i] == 0:
            continue
        for j in range(num_qubits):
            if i == j:
                continue
            if d_mat[j, i] == 1:
                d_mat, d_ph = cx(d_mat, d_ph, i, j)
                clifford_list.append(clifford_op_str("CX", i, j))

    if debug:
        print("\nZERO X")
        print_tableau_side_by_side(c_mat, c_ph, d_mat, d_ph)
        print(clifford_list)

    # 3-2. Diag X
    d1_mat, d1_ph = gf(d_mat), gf(d_ph)
    # Check diagonal
    for i in range(min(d1_mat.shape[0] // 2, d1_mat.shape[1])):
        if d1_mat[i, i] == 0:
            if is_zero_gf(d1_mat[i, :]):
                for j in range(num_qubits):
                    if d1_mat[j, i] == 1:
                        d1_mat[[i, j], :] = d1_mat[[j, i], :]
                        d1_mat[[i + num_qubits, j + num_qubits], :] = d1_mat[[j + num_qubits, i + num_qubits], :]
                        clifford_list.append(clifford_op_str("QSW", i, j))
            for j in range(i + 1, min(d1_mat.shape[0] // 2, d1_mat.shape[1])):
                if d1_mat[i, j]:
                    d1_mat[:, [i, j]] = d1_mat[:, [j, i]]
                    d_ph[[i, j]] = d_ph[[j, i]]
                    break
        for j in range(min(d1_mat.shape[0] // 2, d1_mat.shape[1])):
            if i == j:
                continue
            if d1_mat[i, j]:
                d1_mat[:, j] = d1_mat[:, i] + d1_mat[:, j]
                d1_ph[j] = d1_ph[j] + d1_ph[i]
    if d1_mat.shape[0] // 2 > d1_mat.shape[1]:
        for i in range(d1_mat.shape[1], d1_mat.shape[0] // 2):
            if not is_zero_gf(d1_mat[i, :]):
                for j in range(d1_mat.shape[1]):
                    if d1_mat[i, j] == 1:
                        d1_mat, d1_ph = cx(d1_mat, d1_ph, j, i)
                        clifford_list.append(clifford_op_str("CX", j, i))

    if debug:
        print("\nDIAG X")
        print_tableau_side_by_side(d_mat, d_ph, d1_mat, d1_ph)
        print(clifford_list)
    d_mat, d_ph = d1_mat, d1_ph

    # 4. ZEROING OUT the Z BLOCK
    e_mat, e_ph = gf(d_mat), gf(d_ph)
    for i in range(min(e_mat.shape[0] // 2, e_mat.shape[1])):
        for j in range(i + 1, num_qubits):
            if i == j:
                continue
            if e_mat[j + num_qubits, i] == 1:
                e_mat, e_ph = cz(e_mat, e_ph, i, j)
                clifford_list.append(clifford_op_str("CZ", i, j))
        if e_mat[i + num_qubits, i] == 1:
            e_mat, e_ph = s_gate(e_mat, e_ph, i)
            clifford_list.append(clifford_op_str("S", i))

    if debug:
        print("\nZERO Z")
        print_tableau_side_by_side(d_mat, d_ph, e_mat, e_ph)
        print(clifford_list)

    # 6. TURNING PAULI X TO Z
    f_mat, f_ph = gf(e_mat), gf(e_ph)
    for i in range(num_qubits):
        if np.any(f_mat[i, :]) and is_zero_gf(f_mat[i + num_qubits, :]):
            f_mat, f_ph = hadamard(f_mat, f_ph, i)
            clifford_list.append(clifford_op_str("H", i))
        elif np.any(f_mat[i, :]) and is_equal_gf(f_mat[i, :], f_mat[i + num_qubits, :]):
            f_mat, f_ph = s_gate(f_mat, f_ph, i)
            clifford_list.append(clifford_op_str("S", i))
            f_mat, f_ph = hadamard(f_mat, f_ph, i)
            clifford_list.append(clifford_op_str("H", i))

    if debug:
        print("\nFINAL")
        print_tableau(f_mat, f_ph)
        print(clifford_list)

    # clifford_mat = clifford_compiler(clifford_list, num_qubits)
    # Final check
    a_mat, a_ph = clifford_apply(a_mat, None, clifford_list)
    if not is_zero_gf(a_mat[:num_qubits, :]):
        raise ValueError(pauli_list)
    a_coeff = a_coeff * np.array([-1.0 if p else 1.0 for p in a_ph])
    return a_mat, a_coeff, clifford_list
