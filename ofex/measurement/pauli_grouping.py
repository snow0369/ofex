from copy import deepcopy
from typing import List, Tuple

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.measurement.utils import synthesize_group
from ofex.operators.ordering import order_abs_coeff
from ofex.operators.qubit_operator_tools import single_pauli_commute_chk
from ofex.operators.symbolic_operator_tools import single_term, operator, coeff


__all__ = ["sorted_insertion", "iterative_sorted_insertion", "optimal_sorted_insertion", "pauli_split_group"]

def sorted_insertion(op: QubitOperator, anticommute=False) -> List[QubitOperator]:
    """
    Groups terms in the given QubitOperator into measurement-compatible sets.
    
    Args:
        op (QubitOperator): The input operator to be grouped.
        anticommute (bool): If True, grouping is done for anticommuting terms; 
                            otherwise, for commuting terms (default: False).
    
    Returns:
        List[QubitOperator]: A list of grouped operators for efficient measurement, 
        where each group corresponds to a QubitOperator.
    
    References:
        - https://arxiv.org/abs/1908.06942
        - https://arxiv.org/abs/2409.02504
    """
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
    """
    Iteratively groups terms in the given QubitOperator into commuting and anticommuting sets.

    WARNING: This method is not recommended for use due to its ineffectiveness.

    Args:
        op (QubitOperator): The input operator to be grouped.

    Returns:
        Tuple[List[QubitOperator], List[QubitOperator]]:
        A tuple containing two lists:
        - The first list contains QubitOperator groups with commuting terms.
        - The second list contains QubitOperator groups with anticommuting terms.
    """
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
                             max_iter=10000,
                             debug=False) -> List[QubitOperator]:
    """
    Groups terms in the input QubitOperator into measurement-compatible sets 
    by optimizing the norm using an iterative splitting method.
    
    Args:
        op (QubitOperator): The input operator to be grouped.
        anticommute (bool): If True, performs grouping based on anticommuting compatibility; 
                            otherwise, based on commuting compatibility.
        init_method (str): Initialization method for splitting terms into groups. 
                           Possible values:
                           - "even": Performs even splitting of terms into groups,
                           - "si": Uses sorted insertion for initialization.
                           Default is "even".
        norm_atol (float): The convergence tolerance for the norm difference (default: 1e-5).
        debug (bool): If True, logs debug information about the grouping process (default: False).
    
    Returns:
        List[QubitOperator]: A list of grouped operators for efficient measurement, 
        where each group corresponds to a QubitOperator with optimal norm distribution.
    """

    op = deepcopy(op)
    if () in op.terms:
        c = op.terms[()]
        del op.terms[()]
        if abs(c) > 1e-5:
            raise Warning('Constant term is ignored in sorted insertion.')

    ham_frags, (pauli_list, grp_pauli_list, pauli_grp_list) = (
        pauli_split_group(op, anticommute, method=init_method))

    if debug:
        for idx_grp, p_list in enumerate(grp_pauli_list):
            print(f"Pauli operators in {idx_grp}:")
            op_names = list()
            for op_idx in p_list:
                op_names.append(' '.join([str(o)+str(idx) for idx, o in operator(pauli_list[op_idx])]))
            print(f"\t{op_names}")
    n_groups = len(grp_pauli_list)
    prev_ham_frag = ham_frags

    for _ in range(max_iter):
        prev_grp_norm = [frag.induced_norm(order=2) for frag in prev_ham_frag]
        curr_ham_frag = [QubitOperator().zero() for _ in range(n_groups)]
        for pauli_op, shared_grp in zip(pauli_list, pauli_grp_list):
            normalizer = sum([prev_grp_norm[g] for g in shared_grp])
            for g in shared_grp:
                curr_ham_frag[g] += prev_grp_norm[g] / normalizer * pauli_op

        prev_norm = sum(prev_grp_norm)
        curr_norm = sum([frag.induced_norm(order=2) for frag in curr_ham_frag])
        if np.isclose(curr_norm, prev_norm, atol=norm_atol):
            break

        prev_ham_frag = deepcopy(curr_ham_frag)

    assert op.isclose(QubitOperator.accumulate(prev_ham_frag))
    return prev_ham_frag


def pauli_split_group(ham,
                      anticommute: bool = False,
                      method: str = "even",
                      debug: bool = False):
    """
    Splits the given Hamiltonian into groups of Pauli operators based on 
    (anti)commutation compatibility, using the specified initialization method.
    Unlike the `sorted_insertion` methods, this method allows a single Pauli 
    operator to be partitioned across multiple groups, depending on (anti)commutation.
    
    Args:
        ham (QubitOperator): The input Hamiltonian to be split into Pauli operator groups.
        anticommute (bool): Specifies the grouping criterion. If True, groups Pauli 
                            operators based on anticommutation; if False, groups them 
                            based on commutation (default: False).
        method (str): The initialization method for distributing terms among groups:
                      - "even": Distributes terms evenly across different groups.
                      - "si": Initializes groups using a sorted insertion method.
                      The default value is "even."
        debug (bool): If True, prints detailed debug information about the execution 
                      process (default: False).
    
    Returns:
        Tuple[List[QubitOperator], Tuple[List[QubitOperator], List[List[int]], List[List[int]]]]:
        - The first element is a list of the grouped Hamiltonian fragments.
        - The second element is a tuple containing:
          - pauli_list: List of all Pauli operators from the Hamiltonian.
          - grp_pauli_list: List of Pauli operators in each group.
          - pauli_grp_list: List mapping Pauli operators to their respective group indices.
    """
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
                grp.append(idx_p)
                pauli_grp_list[idx_p].append(idx_grp)

    size_grp = [len(x) for x in grp_pauli_list]
    num_split_pauli = sum(size_grp)

    c_vec = np.zeros(num_split_pauli, dtype=float)
    if method == "even":
        c_idx = 0
        for idx_grp, grp in enumerate(grp_pauli_list):
            for idx_p, p in enumerate(grp):
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
    ham_frags = synthesize_group(False, c_vec, pauli_list, grp_pauli_list, size_grp, ham, debug)[0]
    initial_grp = pauli_list, grp_pauli_list, pauli_grp_list
    return ham_frags, initial_grp
