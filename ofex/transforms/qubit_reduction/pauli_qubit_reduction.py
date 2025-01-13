from copy import deepcopy
from itertools import product
from typing import Union, List, Tuple, Dict

import galois
import numpy as np
from galois import FieldArray
from openfermion import QubitOperator
from pyscf.symm import symm_ops

from ofex.classical_algorithms.clique import find_max_clique
from ofex.clifford import pauli_to_tableau, dot_tableau, diagonalizing_clifford_tableau, tableau_to_pauli, \
    clifford_apply_pauli
from ofex.clifford.clifford_tools import locality
from ofex.clifford.standard_operators import hadamard
from ofex.operators.symbolic_operator_tools import single_term
from ofex.utils.binary_matrix import gf_concatenate, gf_eye, gf_is_zero

gf = galois.GF(2)

def find_pauli_symmetry(pauli_list: Union[List[QubitOperator], QubitOperator],
                        num_qubits: int)\
        -> Tuple[FieldArray, List[str], List[int], QubitOperator]:
    # Check types and remove constant term.
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

    # Set up the E matrix
    # E = [E_x | E_z], E_x = G_z^T, E_z = G_x^T
    g_mat, coeff_list = pauli_to_tableau(pauli_list, num_qubits)
    gx, gz = g_mat[:num_qubits, :], g_mat[num_qubits:, :]
    e_mat = gf_concatenate((gz.T, gx.T), axis=1)

    # Null space finding
    symm = e_mat.null_space()

    # Find the maximally compatible symmetries.
    # Since dot_tableau returns anticommutation, logical NOT is done by using 'ones'.
    # As we are not interested in self-commutation, which is trivial, we add 'eye'.
    n_op = symm.shape[1]
    comm = gf.Ones((n_op, n_op)) + gf_eye(n_op) + dot_tableau(symm, symm)
    # Find maximal mutually commuting symmetries (clique finding in the commutation graph).
    idxs_max_symm = list(find_max_clique(comm))
    symm = symm[:, idxs_max_symm]

    # Perform gaussian elimination to generate z-only symmetry operators.
    symm, _, symm_clifford_list = diagonalizing_clifford_tableau(symm)

    # Minimize the locality of symmetry operators by iteratively evaluating pairwise combinations 
    # and modifying the matrix such that the sum of operator localities is reduced.
    symm = minimize_locality(symm)

    # Find non-trivial qubits
    non_trivial_qubits = gf(np.sum(locality(symm), axis=1))

    # Convert symmetry operator to X-type operators
    tmp_phase = gf.Zeros(symm.shape[1])
    for q in non_trivial_qubits:
        hadamard(symm, tmp_phase, q, symm_clifford_list)

    assert gf_is_zero(symm[num_qubits:, :])

    # CHC† commutes with σ_z(q(i)) with the following qubits
    symm_qubits = select_symm_qubits(symm)
    symm_unitary = construct_symmetry_unitary(symm, symm_qubits)

    return symm, symm_clifford_list, symm_qubits, symm_unitary

def minimize_locality(mat: FieldArray):
    mat = deepcopy(mat)
    n_qubits, n_operators = mat.shape
    n_qubits //= 2
    loc = locality(mat)
    loc_set = np.sum(loc, axis=0)  # locality for each operator
    loc_tot = np.sum(loc)
    while True:
        current_loc = loc_tot
        for i, j in product(range(n_operators), repeat=2):
            if i == j:
                continue
            mul = (mat[:, i] + mat[:, j]) % 2
            loc_mul = np.sum(locality(mul))
            # Reduction of locality if multiplied operator replaces the i(j)th operator.
            loc_red_i = loc_set[i] - loc_mul
            loc_red_j = loc_set[j] - loc_mul

            if loc_red_i > loc_red_j and loc_red_i > 0:  # If reduction of ith operator is larger
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

def select_symm_qubits(mat: FieldArray) -> List[int]:
    # Select qubit q_j for each symmetry operator τ_j such that
    # [σ_z(q_j), τ_i] = 0 for i≠j
    # {σ_z(q_j), τ_j} = 0

    n_qubits, n_operators = mat.shape
    n_qubits = n_qubits // 2

    if not gf_is_zero(mat[n_qubits:, :]):
        raise ValueError("Not a X-only matrix.")
    # mat = np.column_stack(sorted([
    #     mat[:, i] for i in range(n_operators)
    # ], key = lambda x : np.sum(locality(x))))
    # For non-trivial qubits
    it_over = tuple([
        tuple(np.argwhere(mat[:, i]).squeeze())
        for i in range(n_operators)
    ])
    # Minimize the number of overlapped qubits
    opt_idxs = None
    min_overlap = n_operators
    for idxs in product(*it_over):
        overlap = n_operators - len(set(idxs))
        if overlap == 0:
            opt_idxs = idxs
            break
        elif overlap < min_overlap:
            opt_idxs = idxs
            min_overlap = overlap
    else:
        raise ValueError("Can not find p_i satisfying the criteria")
    return list(opt_idxs)

def construct_symmetry_unitary(symm: FieldArray,
                               symm_qubits: List[int])\
        -> QubitOperator:
    # p_i_set = select_p_i(mat)
    n_qubits, n_operators = symm.shape
    n_qubits = n_qubits // 2
    ret_pauli: List[QubitOperator] = list()
    for i, p_i in enumerate(symm_qubits):
        z_vec = gf.Zeros(2 * n_qubits)
        z_vec[p_i + n_qubits] = 1
        ps = QubitOperator.accumulate(
            tableau_to_pauli(
                gf(np.column_stack((symm[:, i], z_vec))),
                coeffs=np.ones(2) * np.sqrt(0.5)
            )
        )
        ret_pauli.append(ps)
    symm_unitary = QubitOperator.identity()
    for p in ret_pauli:
        symm_unitary = p * symm_unitary
    return symm_unitary

def qubit_reduction_operator(operator: Union[QubitOperator, List[QubitOperator]],
                             num_qubits: int,
                             symm_clifford_list: List[str],
                             symm_qubits: List[int],
                             symm_unitary: QubitOperator)\
    -> Dict[Tuple[Tuple[int, int], ...] : Union[QubitOperator, List[QubitOperator]]]:
    operator = clifford_apply_pauli(operator, num_qubits, symm_clifford_list)
    is_list = isinstance(operator, list)
    if is_list:
        operator = [symm_unitary * op * symm_unitary for op in operator]

    else:
        operator = symm_unitary * operator * symm_unitary

def edit_operator_for_symmetry(operator: QubitOperator,
                               qubit_idx: List[int])\
    -> Dict[Tuple[int, int] : Union[QubitOperator]]:
    # Similar but generalized version of
    # openfermion.transform.opconversion.remove_symmetry.edit_hamiltonian_for_spin().

    decompose_keys = [tuple([(qubit, parity) for qubit, parity in zip(qubit_idx, parity_list)])
                      for parity_list in product([-1, 1], repeat=len(qubit_idx))]
    non_trivial_qubits = [[qubit for qubit, parity in key if parity == -1] for key in decompose_keys]

    new_terms = {k: dict() for k in decompose_keys}
    for dec_key, parity_qubits in zip(decompose_keys, non_trivial_qubits):
        for term, coeff in operator.terms.items():
            count_parity = sum([(qubit, "Z") in term for qubit in parity_qubits])
            new_coeff = coeff * (-1) ** count_parity
            


def block_diagonalize_operator_by_symmetry(operator: Union[QubitOperator, List[QubitOperator]],
                                           num_qubits: int,
                                           symm: FieldArray)\
    -> Union[QubitOperator, List[QubitOperator]]:
    pass
