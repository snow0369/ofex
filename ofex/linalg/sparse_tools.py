from __future__ import annotations

from typing import Union

import numpy as np
from openfermion import QubitOperator, LinearQubitOperator
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

from ofex.state.binary_fock import STATE_LSB_FIRST
from ofex.state.state_tools import to_scipy_sparse, get_num_qubits, state_type_transform, to_dense, is_zero
from ofex.state.types import is_sparse_state, State, is_dense_state, type_state, ScipySparse
from ofex.utils.binary import hamming_weight


def transition_amplitude(operator: Union[QubitOperator, spmatrix, LinearOperator],
                         state1: State,
                         state2: State,
                         sparse: bool = False) -> complex:
    """Compute the transitional amplitude, <φ1|O|φ2>.

    Args:
        operator: (QubitOperator or scipy.sparse.spmatrix or scipy.sparse.linalg.LinearOperator)
        state1: (ofex.state.SparseStateDict, numpy.ndarray or scipy.sparse.spmatrix): A numpy array
                representing a pure state or a sparse matrix representing a density
                matrix. If `unitary` is a LinearOperator, then this must be a
                numpy array.
        state2: If None, then the expectation value of the state1 is calculated.
        sparse:

    Returns:
        A complex number giving the transitional amplitude.
    """
    if sparse:
        return state_dot(state1, sparse_apply_operator(operator, state2))
    else:
        return state_dot(state1, apply_operator(operator, state2))


def expectation(operator: Union[QubitOperator, spmatrix, LinearOperator],
                state: State,
                sparse: bool = False) -> complex:
    return transition_amplitude(operator, state, state, sparse)


def apply_operator(operator: Union[QubitOperator, spmatrix, LinearOperator],
                   state: State) -> State:
    if is_zero(state):
        return state
    input_type = type_state(state)
    if isinstance(operator, QubitOperator):
        # This generally takes a long time.
        # If the operator is used multiple times, consider to transform before apply_operator().
        operator = LinearQubitOperator(operator, n_qubits=get_num_qubits(state))
    if input_type in ['sparse_dict', 'scipy_sparse']:
        state = to_dense(state)
        app_st = (operator @ state).T  # This goes to mat-vec mult in scipy.sparse.linalg
        return state_type_transform(app_st, input_type)
    elif input_type == 'dense':
        return state_type_transform(operator @ state, input_type)  # This goes to numpy mat-vec.
    else:
        raise AssertionError


def sparse_apply_operator(operator: QubitOperator,
                          state: State) -> State:
    input_type = type_state(state)
    if input_type == 'sparse_dict':
        new_dict = dict()
        for fock, v in state.items():
            for pauli, coeff in operator.terms.items():
                fock_new, phase = fock.apply_pauli(pauli)
                if fock_new not in new_dict:
                    new_dict[fock_new] = coeff * v * phase
                else:
                    new_dict[fock_new] += coeff * v * phase
        return new_dict
    elif input_type in ['dense', 'scipy_sparse']:
        state = to_scipy_sparse(state)
        n_qubits = get_num_qubits(state)
        n_dim = 2 ** n_qubits
        new_state = ScipySparse((1, n_dim), dtype=np.complex128)
        for pauli, coeff in operator.terms.items():
            p_x, p_z = 0, 0
            for idx, op in pauli:
                bit_add = 2 ** idx if STATE_LSB_FIRST else 2 ** (n_qubits - idx - 1)
                if op == 'X':
                    p_x ^= bit_add
                elif op == 'Y':
                    p_x ^= bit_add
                    p_z ^= bit_add
                elif op == 'Z':
                    p_z ^= bit_add
                elif op == 'I':
                    continue
                else:
                    raise AssertionError
            p_y = p_x & p_z
            p_zo = p_z & (~p_x)
            for _, int_fock in zip(*state.nonzero()):
                st_coeff = state[0, int_fock]
                st_coeff = complex(st_coeff)

                new_int_fock = int_fock ^ p_x
                phase_real = int_fock & p_zo
                phase_imag_m = int_fock & p_y
                phase_imag_p = (~int_fock) & p_y
                phase_tot = (hamming_weight(phase_real) * 2
                             + hamming_weight(phase_imag_p)
                             + hamming_weight(phase_imag_m) * 3)
                new_state[0, new_int_fock] += (1j ** phase_tot) * coeff * st_coeff

        return state_type_transform(new_state, input_type)
    else:
        raise AssertionError


def state_dot(state_1: State, state_2: State) -> complex:
    if is_zero(state_1) or is_zero(state_2):
        return 0.0
    dense_1, dense_2 = is_dense_state(state_1), is_dense_state(state_2)
    if (not dense_1) and (not dense_2):
        state_1, state_2 = to_scipy_sparse(state_1), to_scipy_sparse(state_2)
        return state_1.conj().dot(state_2.T)[0, 0]
    elif (not dense_1) and dense_2:
        state_1 = to_scipy_sparse(state_1)
        return state_1.conj().dot(state_2.T)[0, 0]
    elif dense_1 and (not dense_2):
        state_2 = to_scipy_sparse(state_2)
        return state_2.conj().dot(state_1.T)[0, 0].conjugate()
    else:
        return np.dot(state_1.conj(), state_2)
