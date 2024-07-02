from numbers import Number
from typing import Dict, Any, Union

import numpy as np
from scipy.sparse import lil_matrix, spmatrix

from ofex.exceptions import OfexTypeError
from ofex.state.binary_fock import BinaryFockVector

DenseState = np.ndarray
SparseStateDict = Dict[BinaryFockVector, Number]
ScipySparse = lil_matrix
State = Union[DenseState, SparseStateDict, ScipySparse]
STR_STATE_TYPES = ["dense", "sparse_dict", "scipy_sparse"]


def is_dense_state(state: Any) -> bool:
    return isinstance(state, np.ndarray) and state.ndim == 1


def is_sparse_state(state: Any) -> bool:
    return isinstance(state, dict) and all([isinstance(k, BinaryFockVector) and isinstance(v, Number)
                                            for k, v in state.items()])


def is_scipy_sparse_state(state: Any) -> bool:
    return isinstance(state, spmatrix) and state.shape[0] == 1


def type_state(state: Any) -> str:
    if is_dense_state(state):
        return "dense"
    elif is_sparse_state(state):
        return "sparse_dict"
    elif is_scipy_sparse_state(state):
        return "scipy_sparse"
    else:
        raise OfexTypeError(state)
