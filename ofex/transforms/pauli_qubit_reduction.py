from copy import deepcopy
from itertools import product
from typing import Union, List, Tuple, Dict

import galois
import numpy as np
from galois import FieldArray
from openfermion import QubitOperator, commutator
from openfermion.transforms.opconversions.remove_symmetry_qubits import remove_indices

from ofex.classical_algorithms.clique import find_max_clique
from ofex.clifford import pauli_to_tableau, dot_tableau, diagonalizing_clifford_tableau, tableau_to_pauli, \
    clifford_apply_pauli, clifford_simulation, clifford_apply
from ofex.clifford.clifford_tools import locality
from ofex.clifford.standard_operators import hadamard
from ofex.linalg.sparse_tools import apply_operator
from ofex.operators.qubit_operator_tools import single_pauli_commute_chk
from ofex.operators.symbolic_operator_tools import single_term
from ofex.state.state_tools import to_sparse_dict, norm, compress_sparse, normalize
from ofex.state.types import State, SparseStateDict
from ofex.transforms.fermion_qubit import remove_indices_state
from ofex.utils.binary_matrix import gf_concatenate, gf_eye, gf_is_zero, gf_is_equal

__all__ = ["find_pauli_symmetry", "qubit_reduction_operator", "qubit_reduction_state"]

gf = galois.GF(2)


def find_pauli_symmetry(pauli_list: Union[List[QubitOperator], QubitOperator],
                        num_qubits: int,
                        debug: bool = False,) \
        -> Tuple[FieldArray, List[str], List[int], QubitOperator]:
    r"""
    Identifies Pauli symmetry operators within a given (list of) QubitOperators.
    Pauli symmetry operators(:math:`\hat{S}_k`) commute with all input pauli operators(:math:`\hat{H}`):

    .. math::

        \hat{H}= \sum_j \alpha_j \hat{P}_j

    .. math::

        [\hat{S}_k, \hat{P}_j] = 0 \quad \forall k, j \quad\mathrm{thus, }\quad [\hat{S}_k, \hat{H}] = 0 \quad\forall k.

    Symmetry operators are selected with maximal mutual commutativity:

    .. math::

        \mathcal{S}_M=arg\,max_{\mathcal{S}}|\mathcal{S}|\quad \mathrm{s.t. }\quad
        \mathcal{S}\subseteq \{\hat{S}_k: \forall k\}, \quad [\hat{S}_k,
        \hat{S}_l] = 0 \quad \forall \hat{S}_k, \hat{S}_l \in \mathcal{S}.


    Symmetry operators are diagonalized as X-type by applying the following Clifford transformation:

    .. math::

        \hat{S}^{(X)}_k = \hat{C} \hat{S}_k \hat{C}^{\dagger} \quad \forall \hat{S}_k \in \mathcal{S}_M.

    For each symmetry operator :math:`\hat{S}^{(X)}_k`, a qubit :math:`q(k)` is selected such that:

    .. math::
        \{\hat{S}^{(X)}_k, \hat{Z}_{q(k)}\} = 0 \quad [\hat{S}^{(X)}_k, \hat{Z}_{q(l)}] = 0 \quad \forall k \neq l.

    Then the following unitary operator is constructed:

    .. math::
        \hat{U} = \prod_{k=1}^M \frac{1}{\sqrt{2}}(\hat{S}^{(X)}_k + \hat{Z}_{q(k)}).

    Which block diagonalizes the input Pauli operators:

    .. math::
        \hat{H}_{\mathrm{sym}} = \hat{U} \hat{C} \hat{H} \hat{C}^{\dagger} \hat{U}^{\dagger}

    .. math::
        [\hat{H}_{\mathrm{sym}}, \hat{Z}_{q(k)}] = 0 \quad \forall k.

    Refer to `arXiv:1701.08213 <https://arxiv.org/abs/1701.08213>`_ for more details.

    Args:
        pauli_list (Union[List[QubitOperator], QubitOperator]): Input list of QubitOperators or 
            a single QubitOperator from which the symmetry is to be found.
        num_qubits (int): The number of qubits in the system.
        debug (bool): If True, validate the symmetry operators found.

    Returns:
        Tuple[FieldArray, List[str], List[int], QubitOperator]: A tuple containing:
            - The final symmetry operators in tableau form (FieldArray).
            - The Clifford transformations :math:`\hat{C}` as a list of string representations.
            - The indices of qubits :math:`q(k)` involved in symmetry operations.
            - A unitary operator :math:`\hat{U}` block diagonalizing the input Pauli operators.
    """
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
    symm = e_mat.null_space().T
    if debug:
        assert gf_is_zero(e_mat @ symm)

    # Find the maximally compatible symmetries.
    # Since dot_tableau returns anticommutation, logical NOT is done by using 'ones'.
    # As we are not interested in self-commutation, which is trivial, we add 'eye'.
    n_op = symm.shape[1]
    comm = gf.Ones((n_op, n_op)) + gf_eye(n_op) + dot_tableau(symm, symm)
    # Find maximal mutually commuting symmetries (clique finding in the commutation graph).
    idxs_max_symm = list(find_max_clique(comm))
    symm = symm[:, idxs_max_symm]
    if debug:
        n_max_op = len(idxs_max_symm)
        symm_p_list = tableau_to_pauli(symm)
        for i, j in product(range(n_max_op), repeat=2):
            assert dot_tableau(symm[:, i], symm[:, j]) == 0
            assert single_pauli_commute_chk(symm_p_list[i], symm_p_list[j])

        if isinstance(pauli_list, list):
            pham = QubitOperator.accumulate(pauli_list)
        else:
            pham = pauli_list
        for i in range(n_max_op):
            comm = commutator(pham, symm_p_list[i])
            comm.compress()
            assert comm == QubitOperator.zero()

    # Perform gaussian elimination to generate z-only symmetry operators.
    if debug:
        symm_after, _, symm_clifford_list = diagonalizing_clifford_tableau(symm)
        symm_after_comp, _ = clifford_apply(symm, ph=None, clifford_hist=symm_clifford_list, inverse=False)
        assert gf_is_equal(symm_after, symm_after_comp)
        symm_comp, _ = clifford_apply(symm_after, ph=None, clifford_hist=symm_clifford_list, inverse=True)
        assert gf_is_equal(symm, symm_comp)

        symm = symm_after
    else:
        symm, _, symm_clifford_list = diagonalizing_clifford_tableau(symm)
    # Minimize the locality of symmetry operators by iteratively evaluating pairwise combinations 
    # and modifying the matrix such that the sum of operator localities is reduced.
    symm = minimize_locality(symm)

    # Find non-trivial qubits
    non_trivial_qubits = np.argwhere(np.sum(locality(symm), axis=1)).squeeze()

    # Convert symmetry operator to X-type operators
    tmp_phase = gf.Zeros(symm.shape[1])
    for q in non_trivial_qubits:
        symm, tmp_phase = hadamard(symm, tmp_phase, int(q), symm_clifford_list)

    assert gf_is_zero(symm[num_qubits:, :])

    if debug:
        pham_clifford = clifford_apply_pauli(pham, num_qubits, symm_clifford_list, inverse=False)
        symm_p_list = tableau_to_pauli(symm)
        for i in range(symm.shape[1]):
            comm = commutator(pham_clifford, symm_p_list[i])
            comm.compress()
            assert comm == QubitOperator.zero()

    # CHC† commutes with σ_z(q(i)) with the following qubits
    symm_qubits = select_symm_qubits(symm)
    symm_unitary = construct_symmetry_unitary(symm, symm_qubits)

    return symm, symm_clifford_list, symm_qubits, symm_unitary


def minimize_locality(mat: FieldArray):
    r"""
    Minimizes the locality of input symmetry operators by iteratively combining pairs of operators
    to reduce the overall locality cost.

    Locality is defined as the number of qubits involved in a symmetry operator. This function 
    evaluates pairwise combinations of the operators and reduces their locality while ensuring 
    the commutation relations are preserved.

    Args:
        mat (FieldArray): A 2N × M matrix (tableau representation) describing M symmetry operators
            over N qubits.

    Returns:
        FieldArray: A modified 2N × M matrix where the symmetry operators have been optimized 
        to minimize their locality.
    """
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
            mul = mat[:, i] + mat[:, j]
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
    r"""
    Selects a qubit :math:`q_k` for each symmetry operator :math:`\hat{S}_k` such that 
    the conditions for symmetry are satisfied.

    Specifically, the selected qubits satisfy the following conditions for the symmetry 
    operator matrix:

    .. math::
        \{\sigma_z(q_k), \hat{S}_k \} = 0  \quad \mathrm{and} \quad [\sigma_z(q_k), \hat{S}_j] = 0, \, j \neq k.

    Args:
        mat (FieldArray): A 2N × M matrix (tableau form) where N represents the number of 
            qubits and M is the number of symmetry operators.

    Returns:
        List[int]: A list containing indices of the selected qubits for each symmetry operator.
    """
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
        tuple(np.argwhere(mat[:, i]).squeeze(axis=1))
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
                               symm_qubits: List[int]) \
        -> QubitOperator:
    r"""
    Constructs a unitary operator :math:`\hat{U}` that uses the identified symmetry operators 
    to block-diagonalize the input Pauli operators.

    The unitary operator is of the form:

    .. math::
        \hat{U} = \prod_{k=1}^M \frac{1}{\sqrt{2}} (\hat{S}^{(X)}_k + \hat{Z}_{q(k)}),

    where :math:`\hat{S}^{(X)}_k` are the X-type transformed symmetry operators, and 
    :math:`q(k)` are the qubits corresponding to the symmetries.

    Args:
        symm (FieldArray): A 2N × M tableau matrix representing the symmetry operators.
        symm_qubits (List[int]): A list of qubit indices corresponding to the symmetries.

    Returns:
        QubitOperator: The unitary operator :math:`\hat{U}` constructed from the symmetry operators.
    """
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
                             symm_unitary: QubitOperator) \
        -> Dict[Tuple[Tuple[int, int], ...], Union[QubitOperator, List[QubitOperator]]]:
    r"""
    Reduces a Pauli operator or a list of Pauli operators based on identified symmetry properties.

    The reduction involves applying Clifford transformations, symmetry unitaries, and qubit 
    removal on the input operators. The reduced operators are decomposed into symmetry sectors.

    Args:
        operator (Union[QubitOperator, List[QubitOperator]]): The input Pauli operator(s) to be 
            reduced using symmetry.
        num_qubits (int): The total number of qubits in the system.
        symm_clifford_list (List[str]): A list of Clifford transformations corresponding to 
            symmetry diagonalization.
        symm_qubits (List[int]): The indices of qubits involved in symmetry.
        symm_unitary (QubitOperator): The unitary operator :math:`\hat{U}` constructed from the symmetry operators.

    Returns:
        Dict[Tuple[Tuple[int, int], ...]: Union[QubitOperator, List[QubitOperator]]]: 
            A dictionary mapping symmetry sectors (defined as tuples of qubit states and parities) 
            to the reduced operator(s).
    """
    operator = clifford_apply_pauli(operator, num_qubits, symm_clifford_list, inverse=False)
    is_list = isinstance(operator, list)
    if is_list:
        operator = [symm_unitary * op * symm_unitary for op in operator]
        operator = [edit_operator_for_symmetry(op, symm_qubits) for op in operator]
        operator = [{k: remove_indices(op[k], symm_qubits) for k in op.keys()} for op in operator]
        keys = list(operator[0].keys())
        ret_dict = {k: list() for k in keys}
        for k in keys:
            for op in operator:
                ret_dict[k].append(op[k])
        return ret_dict
    else:
        operator = symm_unitary * operator * symm_unitary
        operator = edit_operator_for_symmetry(operator, symm_qubits)
        operator = {k: remove_indices(operator[k], symm_qubits) for k in operator.keys()}
        return operator


def qubit_reduction_state(state: State,
                          symm_clifford_list: List[str],
                          symm_qubits: List[int],
                          symm_unitary: QubitOperator) \
        -> Tuple[Dict[Tuple[Tuple[int, int], ...], SparseStateDict], Dict[Tuple[Tuple[int, int], ...], float]]:
    r"""
    Reduces a quantum state by decomposing it with respect to symmetry sectors.

    Args:
        state (State): The quantum state to reduce.
        symm_clifford_list (List[str]): A list of Clifford transformations corresponding to
            symmetry diagonalization.
        symm_qubits (List[int]): The indices of qubits involved in symmetry.
        symm_unitary (QubitOperator): The unitary operator :math:`\hat{U}` constructed from the symmetry operators.

    Returns:
        Tuple[Dict[Tuple[Tuple[int, int], ...]: SparseStateDict], Dict[Tuple[Tuple[int, int], ...]: float]]:
            A tuple containing:
            - The reduced states in sparse dictionary format.
            - The normalization factors for each reduced state.
    """
    state = clifford_simulation(state, symm_clifford_list)
    state = apply_operator(symm_unitary, state)
    state = to_sparse_dict(state)

    decompose_keys = [tuple([(qubit, parity) for qubit, parity in zip(symm_qubits, parity_list)])
                      for parity_list in product([-1, 1], repeat=len(symm_qubits))]
    decompose_keys_binary = {tuple([(1 - parity) // 2 for qubit, parity in k]): k for k in decompose_keys}
    dec_states = {k: dict() for k in decompose_keys}

    for bin_vec, coeff in state.items():
        k = decompose_keys_binary[bin_vec[symm_qubits]]
        dec_states[k][bin_vec] = coeff

    dec_state = {k: compress_sparse(remove_indices_state(st, symm_qubits, check_symmetry_conserved=False))
                 for k, st in dec_states.items()}
    norms = {k: norm(st) for k, st in dec_state.items()}
    dec_state = {k: normalize(st) for k, st in dec_state.items()}
    return dec_state, norms


def edit_operator_for_symmetry(operator: QubitOperator,
                               symm_qubits: List[int]) \
        -> Dict[Tuple[Tuple[int, int], ...], Union[QubitOperator]]:
    """
    Modifies an operator by decomposing it according to symmetry properties.

    Args:
        operator (QubitOperator): The input operator to edit.
        symm_qubits (List[int]): Indices of symmetry qubits.

    Returns:
        Dict[Tuple[Tuple[int, int], ...]: Union[QubitOperator]]: A dictionary mapping symmetry sectors to decomposed operators.
    """
    # Similar but generalized version of
    # openfermion.transform.opconversion.remove_symmetry.edit_hamiltonian_for_spin().
    operator.compress()
    for q in symm_qubits:
        for term, coeff in operator.terms.items():
            if ((q, "Y") in term) or ((q, "X") in term):
                raise ValueError(f"Operator contains Y or X terms at symmetry qubits: {term} {coeff}.")

    decompose_keys = [tuple([(qubit, parity) for qubit, parity in zip(symm_qubits, parity_list)])
                      for parity_list in product([-1, 1], repeat=len(symm_qubits))]
    non_trivial_qubits = [[qubit for qubit, parity in key if parity == -1] for key in decompose_keys]

    new_terms = {k: dict() for k in decompose_keys}
    new_hamiltonians = {k: QubitOperator() for k in decompose_keys}

    for dec_key, parity_qubits in zip(decompose_keys, non_trivial_qubits):
        for term, coeff in operator.terms.items():
            count_parity = sum([(qubit, "Z") in term for qubit in parity_qubits])
            new_coeff = coeff * (-1) ** count_parity
            new_term = tuple((q, op) for (q, op) in term if q not in symm_qubits)
            if new_term in new_terms[dec_key]:
                new_terms[dec_key][new_term] += new_coeff
            else:
                new_terms[dec_key][new_term] = new_coeff
        new_hamiltonians[dec_key].terms = new_terms[dec_key]
        new_hamiltonians[dec_key].compress()

    return new_hamiltonians
