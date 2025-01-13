from typing import Union, List, Tuple, Optional

import galois
import numpy as np
from galois import FieldArray
from openfermion import QubitOperator

from ofex.clifford.clifford_tools import pauli_to_tableau, str_tableau_side_by_side, str_tableau
from ofex.clifford.simulation import clifford_apply
from ofex.clifford.standard_operators import hadamard, clifford_op_str, cx, cz, s_gate
from ofex.utils.binary_matrix import gf_is_zero, gf_is_equal

__all__ = ["diagonalizing_clifford_tableau", "diagonalizing_clifford"]
gf = galois.GF(2)

def diagonalizing_clifford_tableau(mat: FieldArray,
                                   coeff_list: Optional[np.ndarray]=None,
                                   debug=False) \
        -> Tuple[FieldArray, np.ndarray, List[str]]:
    """
    Diagonalizes a set of mutually commuting Pauli operators using Clifford gates.

    This function computes the Clifford operations needed to transform a set of commuting
    Pauli operators into a diagonal form, represented by a Pauli tableau. It ensures
    The operations ensure commutation among the input operators and apply transformations
    step by step to achieve diagonalization. The resulting tableau contains Z-type operators
    where the upper portion is set to zero.

    Parameters:
        mat (FieldArray): A binary matrix (Pauli tableau) representing the Pauli operators. 
        coeff_list (Optional[np.ndarray], optional): An array of coefficients corresponding 
            to the operators in the Pauli tableau. If None, a default array of ones is used. 
        debug (bool, optional): If set to True, additional debug information will be 
            printed for each step, showing intermediate transformations. Defaults to False.
    
    Returns:
        Tuple[FieldArray, np.ndarray, List[str]]:
            - mat (FieldArray): A Pauli tableau of diagonalized operators.
                These are Z-type operators, and the upper half is zero.
            - coeff (np.ndarray): The coefficients of the transformed Pauli operators.
            - clifford_history (List[str]): A log of the Clifford operations (as strings)
                applied during the diagonalization process.

    Raises:
        ValueError: If the input Pauli operators do not commute.
    """
    a_mat = gf(mat)
    clifford_list: List[str] = list()
    if coeff_list is None:
        coeff_list = np.ones(a_mat.shape[1])

    num_qubits, num_paulis = a_mat.shape
    assert num_qubits % 2 == 0
    num_qubits = num_qubits // 2

    # Commute Check
    xmat = a_mat[:num_qubits, :]
    zmat = a_mat[num_qubits:, :]
    if not gf_is_zero(xmat.T @ zmat + zmat.T @ xmat):
        raise ValueError("Non-commuting set!")

    if debug:
        print("INIT")
        print(a_mat)

    # 1. FIRST GAUSSIAN
    b_mat = a_mat.T.row_reduce().T
    tmp_mat = gf.Zeros(b_mat.shape)
    num_ind_paulis = 0
    for i in range(b_mat.shape[1]):
        if not gf_is_zero(b_mat[:, i]):
            tmp_mat[:, i] = b_mat[:, i]
            num_ind_paulis += 1
    b_mat = gf(tmp_mat[:, :num_ind_paulis])
    b_ph = gf([0 for _ in range(num_ind_paulis)])

    if debug:
        print("GAUSS")
        print(str_tableau_side_by_side(a_mat, None, b_mat, b_ph))

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
        print(str_tableau_side_by_side(b_mat, b_ph, c_mat, c_ph))
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
        print(str_tableau_side_by_side(c_mat, c_ph, d_mat, d_ph))
        print(clifford_list)

    # 3-2. Diag X
    d1_mat, d1_ph = gf(d_mat), gf(d_ph)
    # Check diagonal
    for i in range(min(d1_mat.shape[0] // 2, d1_mat.shape[1])):
        if d1_mat[i, i] == 0:
            if gf_is_zero(d1_mat[i, :]):
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
            if not gf_is_zero(d1_mat[i, :]):
                for j in range(d1_mat.shape[1]):
                    if d1_mat[i, j] == 1:
                        d1_mat, d1_ph = cx(d1_mat, d1_ph, j, i)
                        clifford_list.append(clifford_op_str("CX", j, i))

    if debug:
        print("\nDIAG X")
        print(str_tableau_side_by_side(d_mat, d_ph, d1_mat, d1_ph))
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
        print(str_tableau_side_by_side(d_mat, d_ph, e_mat, e_ph))
        print(clifford_list)

    # 6. TURNING PAULI X TO Z
    f_mat, f_ph = gf(e_mat), gf(e_ph)
    for i in range(num_qubits):
        if np.any(f_mat[i, :]) and gf_is_zero(f_mat[i + num_qubits, :]):
            f_mat, f_ph = hadamard(f_mat, f_ph, i)
            clifford_list.append(clifford_op_str("H", i))
        elif np.any(f_mat[i, :]) and gf_is_equal(f_mat[i, :], f_mat[i + num_qubits, :]):
            f_mat, f_ph = s_gate(f_mat, f_ph, i)
            clifford_list.append(clifford_op_str("S", i))
            f_mat, f_ph = hadamard(f_mat, f_ph, i)
            clifford_list.append(clifford_op_str("H", i))

    if debug:
        print("\nFINAL")
        print(str_tableau(f_mat, f_ph))
        print(clifford_list)

    # clifford_mat = clifford_compiler(clifford_list, num_qubits)
    # Final check
    a_mat, a_ph = clifford_apply(a_mat, None, clifford_list)
    if not gf_is_zero(a_mat[:num_qubits, :]):
        raise ValueError("Non-commuting set!")
    coeff_list = coeff_list * np.array([-1.0 if p else 1.0 for p in a_ph])
    return a_mat, coeff_list, clifford_list


def diagonalizing_clifford(pauli_list: Union[List[QubitOperator], QubitOperator],
                           num_qubits: int,
                           debug=False) \
        -> Tuple[FieldArray, np.ndarray, List[str]]:
    """
    Diagonalizes a set of mutually commuting Pauli operators using Clifford gates.
    
    This function computes the Clifford operations needed to transform a set of commuting 
    Pauli operators into a diagonal form, represented by a Pauli tableau. It ensures 
    commutation among the input operators, generates the tableau, and applies gate 
    transformations step by step to achieve diagonalization.
    
    Parameters:
        pauli_list (Union[List[QubitOperator], QubitOperator]): A single or list of 
            mutually commuting Pauli operators.
        num_qubits (int): The number of qubits involved in the operators.
        debug (bool, optional): If True, outputs debug information for intermediate 
            transformation steps. Defaults to False.
    
    Returns:
        Tuple[FieldArray, np.ndarray, List[str]]:
            - mat (FieldArray): A Pauli tableau of diagonalized operators.
                These are Z-type operators, and the upper half is zero.
            - coeff (np.ndarray): The coefficients of the transformed Pauli operators.
            - clifford_history (List[str]): A log of the Clifford operations (as strings) 
                applied during the diagonalization process.
    
    Raises:
        ValueError: If the input Pauli operators do not commute.
    """

    # INIT
    a_mat, a_coeff = pauli_to_tableau(pauli_list, num_qubits)
    try:
        a_mat, a_coeff, clifford_list = diagonalizing_clifford_tableau(a_mat, a_coeff, debug)
    except ValueError as e:
        raise ValueError(pauli_list) from e
    return a_mat, a_coeff, clifford_list
