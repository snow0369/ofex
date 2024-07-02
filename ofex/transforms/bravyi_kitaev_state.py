import numpy as np
from galois import FieldArray

from ofex.clifford.clifford_tools import gf
from ofex.state.binary_fock import BinaryFockVector
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import SparseStateDict


def bravyi_kitaev_state(fock_state: SparseStateDict) -> SparseStateDict:
    return _bravyi_kitaev_state(fock_state, inv=False)


def inv_bravyi_kitaev_state(fock_state: SparseStateDict) -> SparseStateDict:
    return _bravyi_kitaev_state(fock_state, inv=True)


def _bravyi_kitaev_state(input_state: SparseStateDict, inv: bool) -> SparseStateDict:
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
    # Construct beta_n matrix (size with n)
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
