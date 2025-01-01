from itertools import product
from typing import Union, List

import galois
import numpy as np
from galois import FieldArray
from openfermion import QubitOperator

from ofex.classical_algorithms.clique import find_max_clique
from ofex.clifford import pauli_to_tableau, dot_tableau
from ofex.clifford.clifford_tools import locality
from ofex.operators.symbolic_operator_tools import single_term
from ofex.utils.binary_matrix import gf_concatenate, gf_eye

gf = galois.GF(2)

def find_pauli_symmetry(pauli_list: Union[List[QubitOperator], QubitOperator],
                        num_qubits: int) -> FieldArray:
    if isinstance(pauli_list, QubitOperator) and () in pauli_list.terms:
        c = pauli_list.terms[()]
        del pauli_list.terms[()]
        if abs(c) > 1e-5:
            raise Warning('Constant term is ignored.')
    elif isinstance(pauli_list, list):
        new_pauli_list = []
        for pauli in pauli_list:
            op, coeff = single_term(pauli)
            if op == ():
                if abs(coeff) > 1e-5:
                    raise Warning('Constant term is ignored.')
            else:
                new_pauli_list.append(pauli)
        pauli_list = new_pauli_list
    else:
        raise TypeError

    # Setup the E matrix
    g_mat, coeff_list = pauli_to_tableau(pauli_list, num_qubits)
    gx, gz = g_mat[:num_qubits, :], g_mat[num_qubits:, :]
    e_mat = gf_concatenate((gz.T, gx.T), axis=1)

    # Null space finding
    symm = e_mat.null_space()

    # Find the maximally compatible symmetries.
    comm = gf.Ones(symm.shape) + gf_eye(symm.shape[0]) + dot_tableau(symm, symm)
    idxs_max_symm = list(find_max_clique(comm))
    symm = symm[idxs_max_symm]

    # Minimize the locality of symmetry operators
    symm = minimize_locality(symm)
    return symm

def minimize_locality(mat: FieldArray):
    """

    Args:
        mat:

    Returns:

    """
    n_qubits, n_operators = mat.shape
    loc = locality(mat)
    loc_set = np.sum(loc, axis=0)
    loc_tot = np.sum(loc)
    while True:
        current_loc = loc_tot
        for i, j in product(range(n_operators), repeat=2):
            if i == j:
                continue
            mul = (mat[:, i] + mat[:, j]) % 2
            loc_mul = np.sum(locality(mul))
            loc_red_i = loc_set[i] - loc_mul
            loc_red_j = loc_set[j] - loc_mul
            if loc_red_i > loc_red_j and loc_red_i > 0:
                mat[:, i] += mat[:, j]  # add_col(j, i)
                loc_set[i] = loc_mul
                current_loc -= loc_red_i
            elif loc_red_j > 0:
                mat[:, j] += mat[:, i]  # add_col(i, j)
                loc_set[j] = loc_mul
                current_loc -= loc_red_j
        if current_loc == loc_tot:
            break
        loc_tot = current_loc
    return mat