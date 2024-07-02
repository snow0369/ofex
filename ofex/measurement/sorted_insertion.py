from copy import deepcopy
from typing import List, Tuple

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.operators.ordering import order_abs_coeff
from ofex.operators.qubit_operator_tools import single_pauli_commute_chk
from ofex.operators.symbolic_operator_tools import single_term, operator


def sorted_insertion(op: QubitOperator, anticommute=False) -> List[QubitOperator]:
    op = deepcopy(op)
    if () in op.terms:
        c = op.terms[()]
        del op.terms[()]
        if abs(c) > EQ_TOLERANCE:
            raise Warning('Constant term is ignored in sorted insertion.')
    si_group: List[List[QubitOperator]] = list()
    init_op_list = order_abs_coeff(op, reverse=True)
    for term_new in init_op_list:
        op_new, coeff_new = single_term(term_new)
        if abs(coeff_new.imag) > EQ_TOLERANCE:
            raise ValueError("Input is not Hermitian.")
        for grp in si_group:
            for term_old in grp:
                op_old = operator(term_old)
                if anticommute == single_pauli_commute_chk(op_new, op_old):
                    break
            else:
                grp.append(term_new)
                break
        else:
            si_group.append([term_new])
    si_group: List[QubitOperator] = [QubitOperator.accumulate(grp) for grp in si_group]
    assert op.isclose(QubitOperator.accumulate(si_group))
    return si_group


def iterative_sorted_insertion(op: QubitOperator) -> Tuple[List[QubitOperator], List[QubitOperator]]:
    raise Warning('This is not a good option')
    remaining = op
    isi_group_comm, isi_group_anti = list(), list()

    def get_norm(x):
        return x.induced_norm(order=2)

    while remaining != QubitOperator.zero():
        si_group_comm = sorted_insertion(remaining, anticommute=False)
        largest_comm = max(si_group_comm, key=get_norm)
        si_group_anti = sorted_insertion(remaining, anticommute=True)
        largest_anti = max(si_group_anti, key=get_norm)
        if get_norm(largest_comm) < get_norm(largest_anti):
            isi_group_anti.append(largest_anti)
            remaining -= largest_anti
        else:
            isi_group_comm.append(largest_comm)
            remaining -= largest_comm

    return isi_group_comm, isi_group_anti


def optimal_sorted_insertion(op: QubitOperator,
                             anticommute,
                             init_method="even",
                             norm_atol=1e-5,
                             debug=False) -> List[QubitOperator]:
    """
    This gives a slight reduction of norm bound than sorted insertion.

    Args:
        op:
        anticommute:
        init_method:
        norm_atol:
        debug:

    Returns:

    """
    from ofex.measurement.iterative_coefficient_splitting import init_ics

    op = deepcopy(op)
    if () in op.terms:
        c = op.terms[()]
        del op.terms[()]
        if abs(c) > 1e-5:
            raise Warning('Constant term is ignored in sorted insertion.')

    ham_frags, (pauli_list, grp_pauli_list, pauli_grp_list), c_vec = (
        init_ics(op, anticommute, method=init_method))

    if debug:
        for idx_grp, p_list in enumerate(grp_pauli_list):
            print(f"Pauli operators in {idx_grp}:")
            op_names = list()
            for op_idx in p_list:
                op_names.append(' '.join([o+str(idx) for idx, o in operator(pauli_list[op_idx])]))
            print(f"\t{op_names}")
    n_groups = len(grp_pauli_list)
    prev_ham_frag = ham_frags
    converge = False

    while not converge:
        prev_grp_norm = [frag.induced_norm(order=2) for frag in prev_ham_frag]
        curr_ham_frag = [QubitOperator().zero() for _ in range(n_groups)]
        for pauli_op, shared_grp in zip(pauli_list, pauli_grp_list):
            normalizer = sum([prev_grp_norm[g] for g in shared_grp])
            for g in shared_grp:
                curr_ham_frag[g] += prev_grp_norm[g] / normalizer * pauli_op

        prev_norm = sum(prev_grp_norm)
        curr_norm = sum([frag.induced_norm(order=2) for frag in curr_ham_frag])
        converge = np.isclose(curr_norm, prev_norm, atol=norm_atol)

        prev_ham_frag = deepcopy(curr_ham_frag)

    assert op.isclose(QubitOperator.accumulate(prev_ham_frag))
    return prev_ham_frag
