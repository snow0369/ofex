"""
This module defines custom types and utility functions to classify and identify 
different quantum state representations used in physics simulations.

It supports three state representations:
1. Dense states (NumPy 1-dimensional arrays).
2. Sparse dictionary-based states (mapping BinaryFockVector to numerical values).
3. SciPy sparse matrix-based states.

Functions are provided for type checking and classification of these state types.
"""

from numbers import Number
from typing import Dict, Any, Union

import numpy as np
from scipy.sparse import lil_matrix, spmatrix

from ofex.exceptions import OfexTypeError
from ofex.state.binary_fock import BinaryFockVector

__all__ = ["DenseState", "SparseStateDict", "ScipySparse", "State",
           "STR_STATE_TYPES", "is_dense_state", "is_sparse_state", "is_scipy_sparse_state", "type_state"]

DenseState = np.ndarray
SparseStateDict = Dict[BinaryFockVector, Number]
ScipySparse = lil_matrix
State = Union[DenseState, SparseStateDict, ScipySparse]
STR_STATE_TYPES = ["dense", "sparse_dict", "scipy_sparse"]


def is_dense_state(state: Any) -> bool:
    """Checks if a given state is a dense state (1-dimensional NumPy array).

    Args:
        state (Any): The state to check.

    Returns:
        bool: True if the state is a 1-dimensional NumPy array, False otherwise.
    """
    return isinstance(state, np.ndarray) and state.ndim == 1


def is_sparse_state(state: Any) -> bool:
    """Checks if a given state is a sparse state represented as a dictionary.

    Args:
        state (Any): The state to check.

    Returns:
        bool: True if the state is a dictionary with keys as BinaryFockVector 
        instances and values as Numbers, False otherwise.
    """
    return isinstance(state, dict) and all([isinstance(k, BinaryFockVector) and isinstance(v, Number)
                                            for k, v in state.items()])


def is_scipy_sparse_state(state: Any) -> bool:
    """Checks if a given state is a sparse state represented as a SciPy sparse matrix.

    Args:
        state (Any): The state to check.

    Returns:
        bool: True if the state is a 1-row SciPy sparse matrix, False otherwise.
    """
    return isinstance(state, spmatrix) and state.shape[0] == 1


def type_state(state: Any) -> str:
    """Determines the type of a given state.

    Args:
        state (Any): The state to identify.

    Returns:
        str: The type of the state, one of "dense", "sparse_dict", or "scipy_sparse".

    Raises:
        OfexTypeError: If the state does not match any recognized type.
    """
    if is_dense_state(state):
        return "dense"
    elif is_sparse_state(state):
        return "sparse_dict"
    elif is_scipy_sparse_state(state):
        return "scipy_sparse"
    else:
        raise OfexTypeError(state)
