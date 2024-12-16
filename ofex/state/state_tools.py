"""
This module provides utilities for working with quantum states, including 
state transformations, normalization, sparsity computations, and comparison.

It supports multiple state formats: dense arrays, sparse dictionaries, 
and Scipy sparse matrices. Additionally, it includes functions for 
Fock vector conversions and pretty-printed representations of quantum states.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Union

import numpy as np
import scipy
from openfermion.config import EQ_TOLERANCE

from ofex.exceptions import OfexTypeError
from ofex.state.binary_fock import BinaryFockVector, int_to_fock, fock_to_int
from ofex.state.types import DenseState, SparseStateDict, is_dense_state, is_sparse_state, State, is_scipy_sparse_state, \
    ScipySparse
from ofex.utils.dict_utils import dict_allclose, compare_dict

__all__ = ["get_num_qubits", "get_state_dim", "get_sparsity", "pretty_print_state", "compare_states",
           "to_dense", "to_scipy_sparse", "to_sparse_dict", "state_type_transform",
           "fock_vector_to_dense_state", "fock_vector_to_scipy_state",
           "compress_sparse", "allclose", "norm", "normalize", "is_zero"]


def get_num_qubits(state: State) -> int:
    """
    Determines the number of qubits in the given quantum state.

    Args:
        state (State): The quantum state, which can be dense, sparse, or a Scipy sparse type.

    Returns:
        int: The number of qubits in the state.

    Raises:
        ValueError: If the dimension of the state is not a power of 2.
        OfexTypeError: If the state type is unsupported.
    """
    if is_dense_state(state) or is_scipy_sparse_state(state):
        if np.log2(state.shape[-1]) != int(np.log2(state.shape[-1])):
            raise ValueError(f"{state.shape[-1]} is not a power of 2.")
        return int(np.log2(state.shape[-1]))
    elif is_sparse_state(state):
        return list(state.keys())[0].num_qubits
    else:
        raise OfexTypeError(state)


def get_state_dim(state: State) -> int:
    """
    Retrieves the dimension of the quantum state.

    Args:
        state (State): The quantum state in dense, sparse, or Scipy sparse format.

    Returns:
        int: The dimension of the state.

    Raises:
        OfexTypeError: If the state type is unsupported.
    """
    if is_dense_state(state) or is_scipy_sparse_state(state):
        return state.shape[-1]
    elif is_sparse_state(state):
        return 2 ** get_num_qubits(state)
    else:
        raise OfexTypeError(state)


def get_sparsity(state: State) -> int:
    """
    Computes the sparsity of the given quantum state.

    Args:
        state (State): The quantum state to analyze.

    Returns:
        int: The number of nonzero components in the state.

    Raises:
        OfexTypeError: If the state type is unsupported.
    """
    if is_sparse_state(state):
        return len(state)
    elif is_dense_state(state):
        return len(state)
    elif is_scipy_sparse_state(state):
        return len(state.nonzero())
    else:
        raise OfexTypeError(state)


def pretty_print_state(state: State, fermion=False) -> str:
    """
    Returns a string representation of the quantum state with coefficients and basis states.

    Args:
        state (State): The quantum state to represent.
        fermion (bool, optional): If True, uses the fermionic representation. Defaults to False.

    Returns:
        str: The pretty-printed representation of the state.

    Raises:
        OfexTypeError: If the state type is unsupported.
    """
    output = list()
    num_qubits = get_num_qubits(state)
    if is_dense_state(state) or is_scipy_sparse_state(state):
        state_dim = get_state_dim(state)
        for i in range(state_dim):
            if abs(state[i]) > EQ_TOLERANCE:
                f = int_to_fock(i, num_qubits)
                coeff_str = f'+{state[i]}'
                output.append(f"{coeff_str} {f.pretty_string(fermion)}")
    elif is_sparse_state(state):
        for f, c in state.items():
            if abs(c) > EQ_TOLERANCE:
                coeff_str = f'+{c}'
                output.append(f"{coeff_str} {f.pretty_string(fermion)}")
    else:
        raise OfexTypeError(state)
    return '\n'.join(output)


def to_dense(state: State) -> DenseState:
    """
    Converts a quantum state to its dense representation.

    Args:
        state (State): The quantum state to convert.

    Returns:
        DenseState: The dense representation of the state.

    Raises:
        OfexTypeError: If the state type is unsupported or cannot be converted.
    """
    if is_dense_state(state):
        return state
    elif is_scipy_sparse_state(state):
        np_state = state.toarray()
        assert np_state.shape[0] == 1
        np_state = np_state[0]
        assert is_dense_state(np_state)
        return np_state
    elif is_sparse_state(state):
        dense_state = np.zeros(get_state_dim(state), dtype=complex)
        for f, c in state.items():
            b = f.to_int()
            assert np.isclose(dense_state[b], 0.0)
            dense_state[b] = c
        return dense_state
    else:
        raise OfexTypeError(state)


def to_scipy_sparse(state: State) -> ScipySparse:
    """
    Converts a quantum state to a Scipy sparse representation.

    Args:
        state (State): The quantum state to convert.

    Returns:
        ScipySparse: The Scipy sparse representation of the state.

    Raises:
        OfexTypeError: If the state type is unsupported or cannot be converted.
    """
    if is_dense_state(state):
        state = ScipySparse(state)
        assert is_scipy_sparse_state(state)
        return state
    elif is_scipy_sparse_state(state):
        return state
    elif is_sparse_state(state):
        state_dim = get_state_dim(state)
        sp_state = ScipySparse(np.zeros(state_dim), (1, state_dim), dtype=np.complex128)
        for k, v in state.items():
            b = fock_to_int(k)
            assert np.isclose(sp_state[0, b], 0.0)
            sp_state[0, b] = v
        assert is_scipy_sparse_state(sp_state)
        return sp_state
    else:
        raise OfexTypeError(state)


def to_sparse_dict(state: State, atol=EQ_TOLERANCE) -> SparseStateDict:
    """
    Converts a quantum state into a sparse dictionary format.

    Args:
        state (State): The quantum state to convert.
        atol (float, optional): Absolute tolerance for nonzero values. Defaults to EQ_TOLERANCE.

    Returns:
        SparseStateDict: The sparse dictionary representation of the state.

    Raises:
        OfexTypeError: If the state type is unsupported or cannot be converted.
    """
    if is_scipy_sparse_state(state):
        state = to_dense(state)
    elif is_sparse_state(state):
        return state
    if not is_dense_state(state):
        raise OfexTypeError(state)
    state_dict = dict()
    n_qubits = get_num_qubits(state)
    for idx, value in np.ndenumerate(state):
        assert len(idx) == 1, idx
        idx = idx[0]
        assert isinstance(idx, int)
        if abs(value) > atol:
            f = int_to_fock(idx, n_qubits)
            assert f not in state_dict, (idx, f)
            state_dict[f] = value
    return state_dict


def state_type_transform(state: State, target_type: str) -> State:
    """
    Transforms the quantum state into a specified representation type.

    Args:
        state (State): The quantum state to transform.
        target_type (str): The target representation type ('dense', 'sparse_dict', 'scipy_sparse').

    Returns:
        State: The transformed quantum state in the specified format.

    Raises:
        ValueError: If the target type is unknown.
    """
    if target_type == 'dense':
        return to_dense(state)
    elif target_type == 'sparse_dict':
        return to_sparse_dict(state)
    elif target_type == 'scipy_sparse':
        return to_scipy_sparse(state)
    else:
        raise ValueError(f"unknown type {target_type}")


def compress_sparse(state: Union[SparseStateDict, ScipySparse], atol=EQ_TOLERANCE,
                    out_normalize=False) \
        -> Union[SparseStateDict, ScipySparse]:
    """
    Compresses a sparse quantum state by removing elements below a tolerance and optionally normalizing it.

    Args:
        state (Union[SparseStateDict, ScipySparse]): The sparse quantum state.
        atol (float, optional): Absolute tolerance for nonzero values. Defaults to EQ_TOLERANCE.
        out_normalize (bool, optional): If True, normalizes the resulting state. Defaults to False.

    Returns:
        Union[SparseStateDict, ScipySparse]: The compressed (and optionally normalized) sparse state.

    Raises:
        OfexTypeError: If the state type is unsupported.
    """
    if is_sparse_state(state):
        new_state = dict()
        for fock, value in state.items():
            if abs(value) > atol:
                new_state[fock] = value
    elif is_scipy_sparse_state(state):
        state_dim = get_state_dim(state)
        new_state = ScipySparse(np.zeros(state_dim), shape=(1, state_dim))
        for idx in state.nonzero():
            idx = idx[0]
            if abs(state[idx]) > atol:
                new_state[idx] = state[idx]
    else:
        raise OfexTypeError(state)

    if out_normalize:
        return normalize(new_state)
    else:
        return new_state


def allclose(state_1: State,
             state_2: State,
             atol=EQ_TOLERANCE) -> bool:
    """
    Checks if two quantum states are approximately equal within a tolerance.

    Args:
        state_1 (State): The first quantum state.
        state_2 (State): The second quantum state.
        atol (float, optional): Absolute tolerance for comparison. Defaults to EQ_TOLERANCE.

    Returns:
        bool: True if the states are approximately equal, False otherwise.
    """
    if is_sparse_state(state_1) and is_sparse_state(state_2):
        return dict_allclose(state_1, state_2, atol)
    else:
        state_1 = to_dense(state_1)
        state_2 = to_dense(state_2)
        return np.allclose(state_1, state_2, atol)


def fock_vector_to_dense_state(fock: BinaryFockVector) -> DenseState:
    """
    Converts a binary Fock vector into a dense quantum state.

    Args:
        fock (BinaryFockVector): The Fock vector to convert.

    Returns:
        DenseState: The dense quantum state corresponding to the Fock vector.
    """
    state = np.zeros(2 ** fock.num_qubits, dtype=complex)
    state[fock.to_int()] = 1.0
    return state


def fock_vector_to_scipy_state(fock: BinaryFockVector) -> ScipySparse:
    """
    Converts a binary Fock vector into a Scipy sparse quantum state.

    Args:
        fock (BinaryFockVector): The Fock vector to convert.

    Returns:
        ScipySparse: The Scipy sparse state corresponding to the Fock vector.
    """
    state = ScipySparse(np.zeros(2 ** fock.num_qubits, dtype=complex), shape=(1, 2 ** fock.num_qubits))
    state[fock.to_int()] = 1.0
    return state


def compare_states(state_1: State, state_2: State,
                   str_len=40, atol=EQ_TOLERANCE, fermion=False) -> str:
    """
    Compares two quantum states and returns a description of their differences.

    Args:
        state_1 (State): The first quantum state.
        state_2 (State): The second quantum state.
        str_len (int, optional): String length limit for individual differences. Defaults to 40.
        atol (float, optional): Absolute tolerance for comparison. Defaults to EQ_TOLERANCE.
        fermion (bool, optional): If True, uses the fermionic representation. Defaults to False.

    Returns:
        str: A string describing the differences between the states.
    """
    def repr_state(k, c):
        return ' '.join((str(c), k.pretty_string(fermion)))

    state_1, state_2 = to_sparse_dict(state_1), to_sparse_dict(state_2)
    return compare_dict(state_1, state_2, repr_state,
                        str_len=str_len, atol=atol)


def norm(state: State) -> float:
    """
    Calculates the L2 norm of the quantum state.

    Args:
        state (State): The quantum state to compute the norm for.

    Returns:
        float: The L2 norm of the state.

    Raises:
        OfexTypeError: If the state type is unsupported.
    """
    if is_sparse_state(state):
        coeffs = np.array(list(state.values()))
        return np.linalg.norm(coeffs, ord=2)
    elif is_dense_state(state):
        return np.linalg.norm(state, ord=2)
    elif is_scipy_sparse_state(state):
        return scipy.sparse.linalg.norm(state, ord='fro')
    else:
        raise OfexTypeError(state)


def normalize(state: State, inplace: bool = False) -> State:
    """
    Normalizes the quantum state to have unit norm.

    Args:
        state (State): The quantum state to normalize.
        inplace (bool, optional): If True, modifies the state in place. Defaults to False.

    Returns:
        State: The normalized quantum state.

    Raises:
        ValueError: If the state is a zero state.
        OfexTypeError: If the state type is unsupported.
    """
    norm_before = norm(state)
    if norm_before < EQ_TOLERANCE:
        raise ValueError("Cannot normalize zero state.")
    if is_sparse_state(state):
        new_state = deepcopy(state)
        for k in state.keys():
            if inplace:
                state[k] /= norm_before
            else:
                new_state[k] = state[k] / norm_before
        if inplace:
            new_state = state
    elif is_dense_state(state) or is_scipy_sparse_state(state):
        if inplace:
            state /= norm_before
            new_state = state
        else:
            new_state = state / norm_before
    else:
        raise OfexTypeError(state)
    return new_state


def is_zero(state: State) -> bool:
    """
    Checks if the quantum state is effectively a zero state.

    Args:
        state (State): The quantum state to check.

    Returns:
        bool: True if the state is a zero state, False otherwise.
    """
    if is_sparse_state(state):
        return len(state) == 0 or np.allclose(list(state.values()), 0.0, atol=EQ_TOLERANCE)
    elif is_dense_state(state):
        return np.allclose(state, 0.0, atol=EQ_TOLERANCE)
    elif is_scipy_sparse_state(state):
        return scipy.sparse.linalg.norm(state, ord='fro') < EQ_TOLERANCE
