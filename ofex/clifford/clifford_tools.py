from collections import Counter
from itertools import product
from typing import Tuple, Optional, List, Union

import galois
import numpy as np
from galois import FieldArray
from openfermion import QubitOperator

from ofex.exceptions import OfexTypeError
from ofex.operators.symbolic_operator_tools import single_term
from ofex.operators.types import SinglePauli

__all__ = [
    "pauli_to_tableau", "tableau_to_pauli", "dot_tableau", "str_tableau", "str_tableau_side_by_side",
    "is_zero_gf", "is_equal_gf"
]

gf = galois.GF(2)


def single_pauli_to_tableau(pauli: SinglePauli,
                            num_qubits: int)\
        -> FieldArray:
    """
    Converts a single Pauli operator into its tableau representation.

    Args:
        pauli: A SinglePauli object representing a single-term Pauli operator 
               (e.g., [("X", 0), ("Y", 1)]).
        num_qubits: The total number of qubits in the system.

    Returns:
        FieldArray: A vector of size 2N representing the tableau form of the Pauli operator,
                    where N is the number of qubits. Specifically:
                    - Elements [0:N] correspond to the X components,
                    - Elements [N:2N] correspond to the Z components.
    """
    
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


def pauli_to_tableau(pauli_list: Union[List[QubitOperator], QubitOperator],
                     num_qubits: int) \
        -> Tuple[FieldArray, np.ndarray]:
    """
    Converts a Pauli operator or list of Pauli operators into tableau form.

    Args:
        pauli_list: A QubitOperator or a list of QubitOperators to be converted. 
                    Each QubitOperator consists of Pauli terms and their coefficients.
        num_qubits: The total number of qubits being represented.

    Returns:
        Tuple[FieldArray, np.ndarray]:

        - G: A 2N x r integer matrix in tableau form, where
            - N is the number of qubits,
            - r is the number of terms in the Pauli list,
            - The first N rows represent G_x (X components of Pauli terms),
            - The next N rows represent G_z (Z components of Pauli terms).

        - coeff: A 1D array of length r corresponding to the coefficients of the Pauli terms.
    """
    if isinstance(pauli_list, QubitOperator):
        pauli_list = list(pauli_list.terms.items())
    elif isinstance(pauli_list, List):
        pauli_list = [single_term(p) for p in pauli_list]
    else:
        raise OfexTypeError(pauli_list)
    coeff_type = type(pauli_list[0][1])

    r = len(pauli_list)

    g_mat = np.zeros((2 * num_qubits, r), dtype=int)
    coeff_arr = np.zeros(r, dtype=coeff_type)
    for i, (p, c) in enumerate(pauli_list):
        v = single_pauli_to_tableau(p, num_qubits)
        g_mat[:, i] = v
        coeff_arr[i] = c
    return gf(g_mat), coeff_arr


def tableau_to_pauli(mat: FieldArray,
                     coeffs: Optional[np.ndarray] = None,
                     ph: Optional[FieldArray] = None)\
        -> List[QubitOperator]:
    """
    Converts a tableau representation of Pauli operators back into QubitOperator objects.

    Args:
        mat: A 2N x r FieldArray matrix representing the tableau form of the Pauli operators,
             where N is the number of qubits and r is the number of operators.
             The first N rows represent G_x (X components of the Pauli terms),
             and the next N rows represent G_z (Z components of the Pauli terms).
        coeffs: A 1D NumPy array of coefficients of length r for the operators. If not specified,
               default coefficients of +/-1.0 will be assigned depending on the phase (ph) array.
        ph: A 1D FieldArray of length r encoding the phase of each Pauli operator (with values 0 or 1).
            If not provided, all values are assumed to be 0.

    Returns:
        A list of QubitOperator objects representing the corresponding Pauli operators described
        by the tableau, each with its associated coefficient.
    """
    
    num_qubits, num_op = mat.shape
    if num_qubits % 2 != 0:
        raise ValueError
    num_qubits //= 2
    if ph is not None:
        if ph.shape[0] != num_op:
            raise ValueError
    else:
        ph = gf(np.zeros(num_op, dtype=int))
    if coeffs is not None:
        if coeffs.shape[0] != num_op:
            raise ValueError
        coeffs = [-c if ph[i] else c for i, c in enumerate(coeffs)]
    else:
        coeffs = [-1.0 if p else 1.0 for p in ph]

    ret = list()
    for op_idx in range(num_op):
        op_vec, c = mat[:, op_idx], coeffs[op_idx]
        op = "".join(["Y" if x and z else "Z" if z else "X" if x else "I"
                      for x, z in zip(op_vec[:num_qubits], op_vec[num_qubits:])])
        op = tuple([(q_idx, p) for q_idx, p in enumerate(op) if p != "I"])
        ret.append(QubitOperator(op, coeffs[op_idx]))
    return ret


def dot_tableau(a: FieldArray, b: FieldArray) -> int:
    """
    Compute the relationship between two vectors `a` and `b` as described in
    Section VIII of Arxiv:1701.08213. This function determines whether the
    vectors commute or anti-commute based on their dot product, following
    the conventions provided in the referenced document.

    Args:
        a (FieldArray): The first input vector.
        b (FieldArray): The second input vector.

    Returns:
        int: Returns 0 if the vectors commute, or 1 if the vectors anti-commute.
    """
    if not a.shape == b.shape:
        raise ValueError
    num_qubits = a.shape[0] // 2
    ax, az = a[:num_qubits], a[num_qubits:]
    bx, bz = b[:num_qubits], b[num_qubits:]
    return int(np.dot(ax, bz) + np.dot(az, bx))


def str_tableau(mat: FieldArray, ph: Optional[FieldArray])\
        -> str:
    """
    Generates a string representation of the tableau form of Pauli operators.

    Args:
        mat: A 2N x r FieldArray matrix representing the tableau of Pauli operators, 
             where N is the number of qubits and r is the number of operators.
             The first N rows correspond to the X components (G_x),
             and the next N rows to the Z components (G_z) of the tableau.
        ph: An optional FieldArray of length r representing the phase of each operator. 
            If provided, it is included in the returned string after the tableau rows;
            otherwise, no phase information will be included.

    Returns:
        str: A formatted string representing the tableau row by row, with a separator 
        line after the X component rows and before the phase information (if present). 
        The separation helps visually distinguish between components and phase.
    """
    n_row, n_pauli = mat.shape
    n_qubits = n_row // 2
    result = []
    for i in range(n_row):
        result.append(" ".join(map(str, mat[i].tolist())))
        if i == n_qubits - 1:
            result.append("=" * (2*n_pauli-1))
    if ph is not None:
        result.append("=" * (2*n_pauli-1))
        result.append(" ".join(map(str, ph.tolist())))
    return "\n".join(result)


def str_tableau_side_by_side(mat1: FieldArray, ph1: Optional[FieldArray],
                             mat2: FieldArray, ph2: Optional[FieldArray])\
        -> str:
    """
    Generates a side-by-side string representation of two tableau matrices and their phases.

    Args:
        mat1 (FieldArray): The first 2N x r tableau matrix where N is the number of qubits 
                           and r is the number of operators.
        ph1 (Optional[FieldArray]): The optional phase information for the first tableau matrix.
        mat2 (FieldArray): The second 2N x r tableau matrix where N is the number of qubits 
                           and r is the number of operators.
        ph2 (Optional[FieldArray]): The optional phase information for the second tableau matrix.

    Returns:
        str: A string representation of the two matrices displayed side-by-side.
    """
    n_row, n_pauli1 = mat1.shape
    n_row2, n_pauli2 = mat2.shape
    n_qubits = n_row // 2
    if n_row != n_row2:
        raise ValueError
    result = []
    for i in range(n_row):
        result.append(f"{' '.join(map(str, mat1[i].tolist()))}  {' '.join(map(str, mat2[i].tolist()))}")
        if i == n_qubits - 1:
            result.append("=" * (2 * n_pauli1 - 1) + "  " + "=" * (2 * n_pauli2 - 1))
    result.append("=" * (2 * n_pauli1 - 1) + "  " + "=" * (2 * n_pauli2 - 1))
    if ph1 is not None:
        result.append(f"{' '.join(map(str, ph1.tolist())) if ph1 is not None else ''}  "
                      f"{' '.join(map(str, ph2.tolist())) if ph2 is not None else ''}")
    return "\n".join(result)


def xor_mat(s_t_pair: List[Tuple[int, int]],
            num_entries: int):
    """
    Constructs a matrix `X` over GF(2) such that for an input vector `v`, the 
    resulting vector `w` is defined as:
    - w[j] = v[i] + v[j], if (i, j) is in `s_t_pair`
    - w[j] = v[j],        otherwise.

    Args:
        s_t_pair (List[Tuple[int, int]]): A list of pairs of indices `(i, j)` specifying 
                                          where the XOR operations should occur.
        num_entries (int): The size of the resulting square matrix `X`.

    Returns:
        FieldArray: A square matrix `X` of size `num_entries x num_entries` over GF(2), 
                    encoding the XOR transformations as specified by `s_t_pair`.

    Raises:
        ValueError: If the XOR transformations described by `s_t_pair` involve 
                    non-commuting operations.
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
    """
    Checks if all elements of a given FieldArray are zero.

    Args:
        a (FieldArray): An input FieldArray to check.

    Returns:
        bool: True if all elements in the FieldArray are zero, otherwise False.
    """
    return np.allclose(np.array(a), 0)


def is_equal_gf(a: FieldArray, b: FieldArray):
    """
    Checks if two FieldArray objects are equal in GF(2).

    Args:
        a (FieldArray): The first input FieldArray.
        b (FieldArray): The second input FieldArray.

    Returns:
        bool: True if the two FieldArrays are equal, otherwise False.
    """
    if not isinstance(a, FieldArray) or not isinstance(b, FieldArray):
        raise TypeError("Both inputs must be of type FieldArray")
    return np.allclose(np.array(a + b), 0)
