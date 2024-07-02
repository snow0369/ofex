import warnings
from numbers import Number
from typing import List, Union, Tuple

import galois
import numpy as np
from galois import FieldArray
from openfermion import FermionOperator, MajoranaOperator, QubitOperator, normal_ordered
from openfermion.config import EQ_TOLERANCE

from ofex.operators.fermion_operator_tools import cre_ann
from ofex.operators.qubit_operator_tools import dict_to_operator
from ofex.state.binary_fock import BinaryFockVector
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import SparseStateDict
from ofex.transforms.majorana_fermion import majorana_to_fermion

gf = galois.GF(2)

warnings.warn(DeprecationWarning())


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


def bravyi_kitaev(input_sum: Union[FermionOperator, MajoranaOperator],
                  num_qubits: int) -> QubitOperator:
    # FermionWord, FermionSum, MajoranaWord or MajoranaSum -> PauliSum
    # Convert fermion or majorana to pauli based on the rule of jordan-wigner.
    if isinstance(input_sum, FermionOperator):
        return _bravyi_kitaev_fer(input_sum, num_qubits)
    elif isinstance(input_sum, MajoranaOperator):
        return _bravyi_kitaev_maj(input_sum, num_qubits)
    else:
        raise TypeError(type(input_sum))


def inv_bravyi_kitaev(input_pauli: QubitOperator,
                      num_qubits: int,
                      to_fermion: bool) -> Union[FermionOperator, Tuple[MajoranaOperator, Number]]:
    # PauliSum -> MajoranaSum(to_fermion=False) or FermionSum(to_fermion=True)
    # Convert pauli to fermion or majorana based on the rule of jordan-wigner.
    if not isinstance(input_pauli, QubitOperator):
        raise TypeError

    if to_fermion:
        return _inv_bravyi_kitaev_fer(input_pauli, num_qubits)
    else:
        return _inv_bravyi_kitaev_maj(input_pauli, num_qubits)


def _bravyi_kitaev_fer(fermion_sum: FermionOperator, num_qubits: int) -> QubitOperator:
    fermion_sum = normal_ordered(fermion_sum)

    def _single_bk(f_op, is_ann, u_vec, r_vec, p_vec, _num_qubits):
        pauli_1 = list()
        pauli_2 = list()
        if f_op % 2 == 1:
            rho_vec = r_vec
        else:
            rho_vec = p_vec
        for i in range(_num_qubits):
            update1 = "I"
            update2 = "I"
            if i in u_vec:
                update1 = "X"
                update2 = "X"
            elif i == f_op:
                update1 = "X"
                update2 = "Y"
            if i in p_vec:
                update1 = "Z"
            if i in rho_vec:
                update2 = "Z"

            if update1 != "I":
                pauli_1.append((i, update1))
            if update2 != "I":
                pauli_2.append((i, update2))
        pauli_1, pauli_2 = tuple(pauli_1), tuple(pauli_2)
        phase = 1 if is_ann else -1
        return dict_to_operator({pauli_1: 0.5, pauli_2: 0.5j * phase}, QubitOperator)

    ret_pauli_sum = QubitOperator()
    parity_set, update_set, flip_set, remainder_set = get_bk_sets(num_qubits)
    for f_term, coeff in fermion_sum.terms.items():
        cre, ann = cre_ann(f_term)

        # For constant term
        if len(cre) == 0 and len(ann) == 0:
            if abs(coeff) < EQ_TOLERANCE:
                continue
            else:
                ret_pauli_sum = ret_pauli_sum + coeff
                continue

        # For n-body term
        if max(cre + ann) >= num_qubits:
            raise ValueError("not sufficient qubits")
        tmp_pauli_sum = QubitOperator.identity() * coeff
        # Creation operators
        for cre_op in cre:
            next_pauli = _single_bk(cre_op,
                                    False,
                                    update_set[cre_op],
                                    remainder_set[cre_op],
                                    parity_set[cre_op],
                                    num_qubits)
            tmp_pauli_sum = tmp_pauli_sum * next_pauli

        # Annihilation operators
        for ann_op in ann:
            next_pauli = _single_bk(ann_op,
                                    True,
                                    update_set[ann_op],
                                    remainder_set[ann_op],
                                    parity_set[ann_op],
                                    num_qubits)
            tmp_pauli_sum = tmp_pauli_sum * next_pauli
        ret_pauli_sum = ret_pauli_sum + tmp_pauli_sum

    return ret_pauli_sum


def _bravyi_kitaev_maj(majorana_sum: MajoranaOperator, num_qubits: int):
    ret_pauli_sum = QubitOperator()
    parity_set, update_set, flip_set, remainder_set = get_bk_sets(num_qubits)

    for m_term, coeff in majorana_sum.terms.items():
        m_operator = m_term

        # For constant term
        if len(m_operator) == 0:
            if abs(coeff) < EQ_TOLERANCE:
                continue
            else:
                ret_pauli_sum = ret_pauli_sum + coeff
                continue

        # For n-body term
        if max(m_operator) >= num_qubits * 2:
            print("not sufficient qubits")
            raise ValueError
        tmp_pauli_sum = QubitOperator.identity() * coeff

        for m in m_operator:
            tmp_pauli = list()
            f = m // 2  # Fermion site
            symmetric = m % 2 == 0
            for i in range(num_qubits):
                if (symmetric and i in parity_set[f]) \
                        or ((not symmetric) and i in remainder_set[f]):
                    tmp_pauli.append((i, "Z"))
                elif i in update_set[f] \
                        or (symmetric and i == f):
                    tmp_pauli.append((i, "X"))
                elif (not symmetric) and i == f:
                    tmp_pauli.append((i, "Y"))
            tmp_pauli_sum = tmp_pauli_sum * QubitOperator(tmp_pauli, 1.0)
        ret_pauli_sum = ret_pauli_sum + tmp_pauli_sum

    return ret_pauli_sum


def _inv_bravyi_kitaev_fer(input_pauli: QubitOperator,
                           num_qubits: int) -> FermionOperator:
    maj_op, const = _inv_bravyi_kitaev_maj(input_pauli, num_qubits)
    return majorana_to_fermion(maj_op) + const


def _inv_bravyi_kitaev_maj(input_pauli: QubitOperator,
                           num_qubits: int) -> Tuple[MajoranaOperator, Number]:
    maj_terms = list()
    parity_set, update_set, flip_set, remainder_set = get_bk_sets(num_qubits)

    # Descendants
    des_set: List[List[int]] = [list() for _ in range(num_qubits)]
    for i in range(num_qubits):
        for ch in flip_set[i]:
            des_set[i] += sorted([ch] + des_set[ch])

    # Parent
    par = list()
    for i in range(num_qubits):
        if len(update_set[i]) > 0:
            par.append(min(update_set[i]))
        else:
            par.append(None)

    # F1 set
    bro_set: List[List[int]] = [list() for _ in range(num_qubits)]
    for i in range(num_qubits):
        if par[i] is not None:
            bro_set[i] = [x for x in flip_set[par[i]] if x > i]

    for operator, coeff in input_pauli.terms.items():
        tmp_maj = coeff
        for i, p in operator:
            if p == 'X':
                tmp_operator = list()
                tmp_phase = 0
                # Add F1
                for j in bro_set[i]:
                    tmp_operator = tmp_operator + [2 * k for k in des_set[j]] + [2 * k + 1 for k in des_set[j]] \
                                   + [2 * j, 2 * j + 1]
                    tmp_phase += len(des_set[j]) + 1
                tmp_coeff = (-1j) ** (tmp_phase + 1)
                # Add remaining part
                if par[i] is not None:
                    tmp_operator = tmp_operator + [2 * i + 1, 2 * par[i]]
                else:
                    tmp_operator = tmp_operator + [2 * i + 1]
            elif p == 'Y':
                tmp_operator = list()
                tmp_phase = 0
                # Add F1 and i
                for j in bro_set[i] + [i]:
                    tmp_operator = tmp_operator + [2 * k for k in des_set[j]] + [2 * k + 1 for k in des_set[j]] \
                                   + [2 * j, 2 * j + 1]
                    tmp_phase += len(des_set[j]) + 1
                tmp_coeff = (-1j) ** (tmp_phase + 2)
                # Add remaining part
                if par[i] is not None:
                    tmp_operator = tmp_operator + [2 * i + 1, 2 * par[i]]
                else:
                    tmp_operator = tmp_operator + [2 * i + 1]
            elif p == 'Z':
                tmp_operator = [2 * j for j in des_set[i]] + [2 * j + 1 for j in des_set[i]] + [2 * i, 2 * i + 1]
                tmp_coeff = (-1j) ** (len(tmp_operator) // 2)
            elif p == 'I':
                continue
            else:
                raise ValueError
            tmp_operator = tuple(sorted(tmp_operator))
            if len(tmp_operator) == 0:
                tmp_maj = tmp_maj * tmp_coeff
            else:
                tmp_maj = tmp_maj * MajoranaOperator(tmp_operator, tmp_coeff)
        maj_terms.append(tmp_maj)
    maj_sum = MajoranaOperator()
    const = 0
    for term in maj_terms:
        if isinstance(term, MajoranaOperator):
            maj_sum += term
        elif isinstance(term, Number):
            const += term
        else:
            raise AssertionError
    return maj_sum, const


# Following the arxiv 1208.5986
def get_bk_sets(num_basis: int) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    # Get parity, update, flip, and remainder set for given size(num_basis)
    pi_mat = pi_matrix(num_basis)
    beta_mat = beta_matrix(num_basis)
    beta_mat_inv = beta_matrix(num_basis, True)
    # Get parity set
    p_mat = np.matmul(pi_mat, beta_mat_inv)  # % 2
    # p_mat %= 2
    parity_set = [list(np.reshape(np.argwhere(p_mat[i] == 1), (-1,))) for i in range(num_basis)]

    # Get update set
    u_mat = np.swapaxes(beta_mat, 0, 1) - gf(np.identity(num_basis, dtype=int))
    update_set = [list(np.reshape(np.argwhere(u_mat[i] == 1), (-1,))) for i in range(num_basis)]

    # Get Flip set
    f_mat = beta_mat_inv - gf(np.identity(num_basis, dtype=int))
    flip_set = [list(np.reshape(np.argwhere(f_mat[i] == 1), (-1,))) for i in range(num_basis)]

    # Get remainder set
    remainder_set = list()
    for i in range(num_basis):
        if len(parity_set[i]) != 0:
            remainder_set.append([v for v in parity_set[i] if v not in flip_set[i]])
        else:
            remainder_set.append(list())

    return parity_set, update_set, flip_set, remainder_set


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
            print("beta_matrix can not have size of {}".format(_n))
            raise ValueError

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


def pi_matrix(n: int) -> FieldArray:
    if n < 1:
        print("pi_matrix can not have size of {}".format(n))
        raise ValueError
    if n == 1:
        return gf(np.array([[0]]))
    else:
        return gf(np.array([[i > j for j in range(n)] for i in range(n)], dtype=int))
