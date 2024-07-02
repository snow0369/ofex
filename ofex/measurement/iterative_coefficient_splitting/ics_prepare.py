from typing import Tuple, List

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.measurement.iterative_coefficient_splitting.ics_utils import _synthesize_group
from ofex.measurement.sorted_insertion import sorted_insertion
from ofex.operators.ordering import order_abs_coeff
from ofex.operators.qubit_operator_tools import single_pauli_commute_chk
from ofex.operators.symbolic_operator_tools import coeff, operator


def init_ics(ham: QubitOperator,
             anticommute: bool = False,
             method="even",
             debug=False) \
        -> Tuple[List[QubitOperator], Tuple[List[QubitOperator], List[List[int]], List[List[int]]], np.ndarray]:
    """
    Produce initial coefficient splitting.

    :param ham:
    :param anticommute: Anti-commutation partitioning
    :param method: "even" | "si" ; Initial coefficient assignment
    :param debug:
    :return:
        ham_frags: Initial coefficient splitting.
        pauli_list: List of Pauli operators.
        grp_pauli_list: List of groups of Pauli operators indices.
        pauli_grp_pauli_list: List of group indices which Pauli operators belong to.
        c_vec: Initial coefficient in numpy vector.
    """
    if ham.constant != 0.0:
        raise ValueError("Hamiltonian should have zero trace.")

    pauli_list = order_abs_coeff(ham, reverse=True)
    for i, p in enumerate(pauli_list):
        assert np.isclose(coeff(p).imag, 0.0)
        if abs(coeff(p)) > EQ_TOLERANCE:
            pauli_list[i] = QubitOperator(operator(p), coeff(p).real)

    init_pauli_grp = sorted_insertion(ham, anticommute)
    grp_pauli_list: List[List[int]] = [list() for _ in range(len(init_pauli_grp))]
    # Contains pauli idx list for each group
    pauli_grp_list: List[List[int]] = [list() for _ in range(len(pauli_list))]
    # Contains group idx list for each pauli
    for idx_p, p in enumerate(pauli_list):
        for idx_grp, grp in enumerate(init_pauli_grp):
            for q in grp:
                if p == q:
                    grp_pauli_list[idx_grp].append(idx_p)
                    pauli_grp_list[idx_p].append(idx_grp)
                    break
            else:  # Not found
                continue
            break
        else:
            print(f"p = {coeff(p)} {operator(p)} ({type(p)})")
            print(pauli_list)
            raise AssertionError

    for idx_p, p in enumerate(pauli_list):
        for idx_grp, grp in enumerate(grp_pauli_list):
            if idx_p in grp:
                continue
            for idx_q in grp:
                q = pauli_list[idx_q]
                if anticommute == single_pauli_commute_chk(p, q):
                    break
            else:
                grp_pauli_list[idx_grp].append(idx_p)
                pauli_grp_list[idx_p].append(idx_grp)

    size_grp = [len(x) for x in grp_pauli_list]
    num_split_pauli = sum(size_grp)

    c_vec = np.zeros(num_split_pauli, dtype=float)
    if method == "even":
        c_idx = 0
        for idx_grp in range(len(grp_pauli_list)):
            for idx_p, p in enumerate(grp_pauli_list[idx_grp]):
                n_included_grp = len(pauli_grp_list[p])
                c_vec[c_idx + idx_p] = coeff(pauli_list[p]).real * (1 / n_included_grp)
            c_idx += size_grp[idx_grp]
    elif method == "si":
        acc_size_grp = [sum(size_grp[:i]) if i > 0 else 0 for i in range(len(size_grp))]
        for p_idx, p_grp_list in enumerate(pauli_grp_list):
            idx_grp = p_grp_list[0]
            curr_grp = grp_pauli_list[idx_grp]
            addr_p = acc_size_grp[idx_grp] + curr_grp.index(p_idx)
            c_vec[addr_p] = coeff(pauli_list[p_idx]).real
    else:
        raise ValueError(f"Unknown method {method}")
    ham_frags = _synthesize_group(c_vec, pauli_list, grp_pauli_list, size_grp, ham, debug)
    initial_grp = pauli_list, grp_pauli_list, pauli_grp_list

    return ham_frags, initial_grp, c_vec
