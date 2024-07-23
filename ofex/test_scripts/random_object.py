from typing import Tuple

import numpy as np
from openfermion import QubitOperator

from ofex.clifford.clifford_tools import tableau_to_pauli
from ofex.state.binary_fock import int_to_fock
from ofex.state.state_tools import to_dense, to_scipy_sparse
from ofex.state.types import SparseStateDict, ScipySparse
from ofex.utils.binary import int_to_binary


def random_complex():
    return 2 * np.random.rand() - 1 + (2 * np.random.rand() - 1) * 1j


def random_state_spdict(n_qubits, density=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if density is not None:
        density = min(2 ** n_qubits, density)
        non_zero_idx = np.random.choice(2 ** n_qubits, size=density, replace=False)
    else:
        non_zero_idx = np.arange(2 ** n_qubits)
    state = dict()
    for idx in non_zero_idx:
        state[int_to_fock(idx, n_qubits)] = random_complex()
    norm = sum([abs(x) ** 2 for x in state.values()]) ** (1 / 2)
    for fock, value in state.items():
        state[fock] = value / norm
    return state


def random_state_nparray(n_qubits, density=None, seed=None):
    return to_dense(random_state_spdict(n_qubits, density, seed))


def random_state_sparse(n_qubits, density=None, seed=None):
    return to_scipy_sparse(random_state_spdict(n_qubits, density, seed))


def random_state_all(n_qubits, density=None, seed=None) -> Tuple[SparseStateDict, np.ndarray, ScipySparse]:
    state_dict = random_state_spdict(n_qubits, density, seed)
    return state_dict, to_dense(state_dict), to_scipy_sparse(state_dict)


def random_qubit_operator(n_qubits, num_terms, seed=None, hermitian=True):
    if seed is not None:
        np.random.seed(seed)
    num_terms = min(4 ** n_qubits, num_terms)
    idx_list = np.random.choice(4 ** n_qubits, size=num_terms, replace=False)
    pauli_tableau = np.array([list(int_to_binary(idx, 2 * n_qubits, lsb_first=True)) for idx in idx_list], dtype=int).T
    coeff = np.random.rand(num_terms) * 2 - 1
    if not hermitian:
        coeff += (np.random.rand(num_terms) * 2 - 1) * 1j
    qubit_operators = tableau_to_pauli(pauli_tableau, coeff=coeff)
    return QubitOperator.accumulate(qubit_operators)
