"""
This module provides utilities and functions for handling quantum state operations
and sparse matrix computations. Key functionalities include quantum state transitions, 
expectation value calculations, operator applications, diagonalization of operators, 
and state manipulation in both dense and sparse formats.

The primary goal of this module is to enhance the efficiency and scalability of 
quantum operator applications and computations by leveraging sparse representations 
where applicable.

Definitions:

- State: Representations of quantum states using formats like numpy arrays, sparse matrices, or dictionaries
  as supported by ``ofex.state.types``.

- Operator: Quantum operators applied to states, supporting formats like QubitOperator, scipy sparse matrices,
  or LinearOperators.
"""

from __future__ import annotations

from typing import Union, Tuple, Optional

import numpy as np
import scipy
from openfermion import QubitOperator, LinearQubitOperator, get_sparse_operator, FermionOperator
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

from ofex.state.binary_fock import STATE_LSB_FIRST
from ofex.state.state_tools import to_scipy_sparse, get_num_qubits, state_type_transform, to_dense, is_zero, \
    get_sparsity
from ofex.state.types import State, is_dense_state, type_state, ScipySparse
from ofex.utils.binary import hamming_weight

__all__ = ["transition_amplitude", "expectation", "apply_operator", "diagonalization", "sparse_apply_operator",
           "state_dot"]


def transition_amplitude(operator: Union[QubitOperator, spmatrix, LinearOperator],
                         state1: State,
                         state2: State,
                         sparse1: bool = False,
                         sparse2: bool = False) -> complex:
    """
    Compute the transitional amplitude, <φ1|O|φ2>.
    
    Args:
        operator: The operator to apply, which can be a QubitOperator, a scipy sparse matrix, 
                  or a scipy LinearOperator.
        state1: A quantum state supported as defined in `ofex.state.types` (e.g., a numpy array, 
                sparse matrix, or dict).
        state2: A second quantum state in the same format as `state1`. If set to None, the expectation 
                value of `state1` is calculated instead.
        sparse1: A boolean indicating whether to perform sparse operations for `state1`.
        sparse2: A boolean indicating whether to perform sparse operations for `state2`.
    
    Returns:
        A complex number representing the transitional amplitude between the two states.
    """
    if sparse1 and sparse2:
        s1, s2 = get_sparsity(state1), get_sparsity(state2)
        if s1 > s2:
            sparse1 = False
        else:
            sparse2 = False

    if sparse2:
        return state_dot(state1, sparse_apply_operator(operator, state2))
    elif sparse1:
        return state_dot(state2, sparse_apply_operator(operator, state1)).conjugate()
    else:
        return state_dot(state1, apply_operator(operator, state2))


def expectation(operator: Union[QubitOperator, spmatrix, LinearOperator],
                state: State,
                sparse: bool = False) -> complex:
    """
    Compute the expectation value <φ|O|φ>.

    Args:
        operator: The operator for which the expectation value is calculated. This can be a 
                  QubitOperator, a scipy sparse matrix, or a LinearOperator.
        state: A quantum state supported as defined in `ofex.state.types` (e.g., a numpy array,
               sparse matrix, or dict).
        sparse: A boolean indicating whether to perform sparse operations for `state`.

    Returns:
        A complex number representing the expectation value of the operator for the given state.
    """
    if sparse:
        return state_dot(state, sparse_apply_operator(operator, state))
    else:
        return state_dot(state, apply_operator(operator, state))


def apply_operator(operator: Union[QubitOperator, spmatrix, LinearOperator],
                   state: State) -> State:
    """
    Apply a given operator to a quantum state.

    Args:
        operator: The operator to apply, which can be a QubitOperator, scipy sparse matrix, 
                  or a LinearOperator.
        state: A quantum state supported as defined in `ofex.state.types` (e.g., a numpy array,
               sparse matrix, or dict).

    Returns:
        The resulting quantum state after applying the operator, preserving the input type.
        If the state is zero, it returns the same state without modification.
    """
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
        return operator @ state  # This goes to numpy mat-vec.
    else:
        raise AssertionError


def diagonalization(operator:Union[QubitOperator, FermionOperator],
                    n_qubits: int,
                    sparse_eig: bool,
                    n_eigen: Optional[int] = None,
                    **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of an operator.

    Args:
        operator: QubitOperator or FermionOperator to be diagonalized.
        n_qubits: The number of qubits represented in the operator.
        sparse_eig: A boolean value indicating whether to compute eigenvalues for a sparse 
                    operator representation.
        n_eigen: The number of eigenvalues/vectors to calculate if `sparse_eig` is True.
                 Defaults to None, indicating all eigenvalues/vectors.
        **kwargs: Additional arguments to be passed to the scipy eigensolver.

    Returns:
        If `sparse_eig` is True, returns a tuple (eigenvalues, eigenvectors) for the 
        sparse representation.
        Otherwise, returns a dense numpy array of eigenvalues and eigenvectors.
    """
    operator = get_sparse_operator(operator, n_qubits=n_qubits)
    if sparse_eig:
        which = kwargs.pop('which', "SA")
        return scipy.sparse.linalg.eigsh(operator, k=n_eigen, which=which, **kwargs)
    else:
        operator = operator.todense()
        return scipy.linalg.eigh(operator, **kwargs)


def sparse_apply_operator(operator: QubitOperator,
                          state: State) -> State:
    """
    Apply a QubitOperator to a state using sparse representations.

    Args:
        operator: The QubitOperator to apply.
        state: The quantum state to which the operator will be applied. This can 
               be specified in sparse or dense format.

    Returns:
        The resulting state after applying the operator, preserving the original 
        state format.
    """
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
    """
    Compute the Hermitian dot product <state_1|state_2> between two quantum states.
    
    Args:
        state_1: The first quantum state, which can be in any supported format defined 
                 in `ofex.state.types` (e.g., numpy array, scipy sparse matrix, or dict).
        state_2: The second quantum state in the computation, in the same formats 
                 supported as `state_1`.

    Returns:
        A complex number representing the Hermitian dot product of `state_1` and `state_2`. 
        If either state represents a "zero" state, the result will be zero.
    """
    if is_zero(state_1) or is_zero(state_2):
        return 0.0
    dense_1, dense_2 = is_dense_state(state_1), is_dense_state(state_2)
    if (not dense_1) and (not dense_2):
        state_1, state_2 = to_scipy_sparse(state_1), to_scipy_sparse(state_2)
        return state_1.conj().dot(state_2.T)[0, 0]
    elif (not dense_1) and dense_2:
        state_1 = to_scipy_sparse(state_1)
        return state_1.conj().dot(state_2.T)[0]
    elif dense_1 and (not dense_2):
        state_2 = to_scipy_sparse(state_2)
        return state_2.conj().dot(state_1.T)[0].conjugate()
    else:
        return np.dot(state_1.conj(), state_2)
