from numbers import Number
from typing import Tuple, List

import numpy as np
from openfermion import QubitOperator

from ofex.operators.symbolic_operator_tools import coeff, operator


def _checksum_coeff(pauli_grp_coeff_split, coeff_list, atol) -> None:
    for split_list, c in zip(pauli_grp_coeff_split, coeff_list):
        assert np.isclose(sum(split_list), c, atol=atol)


def _checksum_ham(ham: QubitOperator, grp_operator: List[QubitOperator], atol) -> None:
    assert ham.isclose(QubitOperator.accumulate(grp_operator), atol)


def _groupwise_norm(pauli_grp_coeff_split, pauli_grp_list, num_grp) -> np.ndarray:
    grp_two_norm = np.zeros(num_grp)
    for split_list, grp_list in zip(pauli_grp_coeff_split, pauli_grp_list):
        for c, g in zip(split_list, grp_list):
            grp_two_norm[g] += abs(c) ** 2
    grp_two_norm = np.sqrt(grp_two_norm)
    return grp_two_norm


def _update_coeff_split(old_pauli_grp_coeff_split, pauli_grp_list, coeff_list, num_grp,
                        checksum_atol) -> List[List[Number]]:
    grp_norm_list = _groupwise_norm(old_pauli_grp_coeff_split, pauli_grp_list, num_grp)
    new_pauli_grp_coeff_split = list()
    for grp_list, c in zip(pauli_grp_list, coeff_list):
        new_pauli_grp_coeff_split.append(list())
        tot_norm = sum([grp_norm_list[g] for g in grp_list])
        for grp in grp_list:
            new_pauli_grp_coeff_split[-1].append(c * grp_norm_list[grp] / tot_norm)
    _checksum_coeff(new_pauli_grp_coeff_split, coeff_list, atol=checksum_atol)
    return new_pauli_grp_coeff_split


def _converged(old_pauli_group_coeff_split, new_pauli_group_coeff_split, conv_atol):
    for old_group_coeff_split, new_group_coeff_split in zip(old_pauli_group_coeff_split, new_pauli_group_coeff_split):
        if not np.allclose(old_group_coeff_split, new_group_coeff_split, atol=conv_atol):
            return False
    return True


def _synthesis(pauli_list, pauli_grp_coeff_split, pauli_grp_list, num_grp) -> List[QubitOperator]:
    grp_operator = [QubitOperator() for _ in range(num_grp)]
    for pauli, grp_coeff_split, grp_list in zip(pauli_list, pauli_grp_coeff_split, pauli_grp_list):
        p = operator(pauli)
        for c, grp in zip(grp_coeff_split, grp_list):
            grp_operator[grp] += QubitOperator(p, c)
    return grp_operator


def efficient_ics(ham: QubitOperator,
                  initial_grp: Tuple[List[QubitOperator], List[List[int]], List[List[int]]],
                  conv_th: float = 1e-6,
                  checksum_atol: float = 1e-6,
                  max_iter=10000,
                  ) -> Tuple[List[QubitOperator], float]:
    if ham.constant != 0.0:
        raise ValueError

    # 1. Partitioning into compatible groups
    pauli_list, grp_pauli_list, pauli_grp_list = initial_grp

    # 2. Get coefficients
    num_grp = len(grp_pauli_list)
    num_pauli = len(pauli_list)
    size_pauli = [len(x) for x in pauli_grp_list]
    coeff_list = [coeff(p) for p in pauli_list]
    # Initial uniform split
    pauli_grp_coeff_split = [[coeff_list[i] / size_pauli[i]
                              for _ in pauli_grp_list[i]]
                             for i in range(num_pauli)]
    _checksum_coeff(pauli_grp_coeff_split, coeff_list, checksum_atol)

    # 3. Perform optimization
    converged = False
    n_iter = 0
    while not converged and n_iter < max_iter:
        new_pauli_grp_coeff_split = _update_coeff_split(pauli_grp_coeff_split, pauli_grp_list, coeff_list, num_grp,
                                                        checksum_atol)
        converged = _converged(pauli_grp_coeff_split, new_pauli_grp_coeff_split, conv_th)
        pauli_grp_coeff_split = new_pauli_grp_coeff_split
        n_iter += 1

    # 4. Wrapup
    tot_norm = sum(_groupwise_norm(pauli_grp_coeff_split, pauli_grp_list, num_grp))
    grp_operator = _synthesis(pauli_list, pauli_grp_coeff_split, pauli_grp_list, num_grp)
    _checksum_ham(ham, grp_operator, checksum_atol)
    return grp_operator, tot_norm
