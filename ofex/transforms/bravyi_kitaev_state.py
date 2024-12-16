"""
This module provides functions for performing the Bravyi-Kitaev transformation to states,
a method for encoding fermionic states into qubit representations. The transformation 
is described in detail in the paper: https://arxiv.org/abs/1208.5986.

It includes functions to convert between Fock state representations and Bravyi-Kitaev 
encoded representations, as well as utilities for constructing the transformation matrices.
"""

import numpy as np
from galois import FieldArray

from ofex.clifford.clifford_tools import gf
from ofex.state import BinaryFockVector
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import SparseStateDict


def bravyi_kitaev_state(fock_state: SparseStateDict) -> SparseStateDict:
    """
    Convert a Fermionic Fock state representation to its Bravyi-Kitaev encoding.
    Refer to the paper: https://arxiv.org/abs/1208.5986 for more details.

    Args:
        fock_state (SparseStateDict): A sparse representation of the input Fermionic Fock state.

    Returns:
        SparseStateDict: The Bravyi-Kitaev encoded representation of the input state.
    """
    return _bravyi_kitaev_state(fock_state, inv=False)


def inv_bravyi_kitaev_state(fock_state: SparseStateDict) -> SparseStateDict:
    """
    Convert a Bravyi-Kitaev encoded state back to its Fermionic Fock state representation.
    Refer to the paper: https://arxiv.org/abs/1208.5986 for more details.

    Args:
        fock_state (SparseStateDict): A sparse representation of the input Bravyi-Kitaev state.

    Returns:
        SparseStateDict: The Fermionic Fock state representation of the input state.
    """
    return _bravyi_kitaev_state(fock_state, inv=True)


def _bravyi_kitaev_state(input_state: SparseStateDict, inv: bool) -> SparseStateDict:
    """
    Perform the Bravyi-Kitaev or inverse Bravyi-Kitaev transformation on a given state.

    Args:
        input_state (SparseStateDict): A sparse representation of the input state (Fock or Bravyi-Kitaev).
        inv (bool): Whether to apply the inverse transformation.

    Returns:
        SparseStateDict: The transformed state in the chosen encoding.
    """
    ret_dict = dict()
    num_qubits = get_num_qubits(input_state)
    beta_mat = beta_matrix(num_qubits)
    if inv:
        beta_mat = np.linalg.inv(beta_mat)
    for f, c in input_state.items():
        f = BinaryFockVector(list(beta_mat @ gf(f)))
        assert f not in ret_dict.keys()
        ret_dict[f] = c
    return ret_dict


def beta_matrix(n: int, inv: bool = False) -> FieldArray:
    """
    Construct the beta_n matrix used for Bravyi-Kitaev transformations.

    Args:
        n (int): The number of qubits (size of the matrix).
        inv (bool, optional): Whether to construct the inverse of the matrix. Default is False.

    Returns:
        FieldArray: The constructed beta_n matrix or its inverse.
    """
    def _custom_log(x):
        # x = 2^(exp)-res (res >= 0)
        tmp_x = x
        exp = 0
        while tmp_x > 0:
            tmp_x = tmp_x // 2
            exp += 1
        if 2 ** (exp - 1) == x:
            res = 0
            exp -= 1
        else:
            res = (2 ** exp) - x
        return exp, res

    def _beta_matrix(_n, _inv):
        if _n < 1:
            raise ValueError("beta_matrix can not have size of {}".format(_n))

        if _n == 1:
            return np.array([[1]])
        elif _n == 2:
            return np.array([[1, 1], [0, 1]])

        exp, res = _custom_log(_n)
        _ret = np.kron(np.identity(2, dtype=int), _beta_matrix(2 ** (exp - 1), _inv))
        if _inv:
            _ret[0][2 ** (exp - 1)] = 1
        else:
            _ret[0] = np.ones(2 ** exp)
        if res > 0:
            _ret = _ret[-_n:, -_n:]
            # _ret = _ret[0:_n, 0:_n]
        _ret = _ret % 2
        return _ret

    ret = _beta_matrix(n, inv)
    for i, row in enumerate(ret):
        ret[i] = row[::-1]
    ret = ret[::-1]

    return gf(ret)
