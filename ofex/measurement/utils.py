import numpy as np
from openfermion import QubitOperator

from ofex.linalg.sparse_tools import transition_amplitude, expectation
from ofex.operators.symbolic_operator_tools import operator, dict_to_operator

__all__ = []

def buf_transition_amplitude(op, state1, state2, ph, key, buffer):
    if buffer is None:
        return transition_amplitude(op, state1, state2, sparse1=True) * np.exp(-1j * ph)
    elif key in buffer:
        return buffer[key] * np.exp(-1j * ph)
    else:
        ov = transition_amplitude(op, state1, state2, sparse1=True)
        buffer[key] = ov
        return ov * np.exp(-1j * ph)


def buf_diag_expectation(op, state1, state2, key, buffer):
    if buffer is None:
        return 0.5 * (expectation(op, state1, sparse=True) + expectation(op, state2, sparse=True))
    elif key in buffer:
        return buffer[key]
    else:
        ex = 0.5 * (expectation(op, state1, sparse=True) + expectation(op, state2, sparse=True))
        buffer[key] = ex
        return ex


def buf_expectation(op, state, key, buffer):
    if buffer is None:
        return expectation(op, state, sparse=True)
    elif key in buffer:
        return buffer[key]
    else:
        ex = expectation(op, state, sparse=True)
        buffer[key] = ex
        return ex


def weight_sum_dict(ov_dict_1: dict, ov_dict_2: dict):
    ov_dict_merged = dict()
    tot_keys = set(ov_dict_1.keys()).union(ov_dict_2.keys())
    for k in tot_keys:
        if k in ov_dict_1 and k in ov_dict_2:
            (sh_real_1, sh_imag_1), ov_1 = ov_dict_1[k]
            (sh_real_2, sh_imag_2), ov_2 = ov_dict_2[k]
            sh_real_tot, sh_imag_tot = sh_real_1 + sh_real_2, sh_imag_1 + sh_imag_2
            if sh_real_tot > 0:
                real_avg = (ov_1.real * sh_real_1 + ov_2.real * sh_real_2) / sh_real_tot
            else:
                real_avg = 0.0
            if sh_imag_tot > 0:
                imag_avg = (ov_1.imag * sh_imag_1 + ov_2.imag * sh_imag_2) / sh_imag_tot
            else:
                imag_avg = 0.0
            ov_dict_merged[k] = ((sh_real_tot, sh_imag_tot), real_avg + imag_avg * 1j)
        elif k in ov_dict_1 and k not in ov_dict_2:
            ov_dict_merged[k] = ov_dict_1[k]
        elif k not in ov_dict_1 and k in ov_dict_2:
            ov_dict_merged[k] = ov_dict_2[k]
        else:
            raise AssertionError
    return ov_dict_merged


def synthesize_group(sep_reim, c_opt, pauli_list, grp_pauli_list, size_grp, ham, debug, correct_sum=True):
    # single_terms, leftover, anticommute, debug):
    grp_ham_re, grp_ham_im = list(), list()
    if sep_reim:
        grp_ham_list = [grp_ham_re, grp_ham_im]
        num_split_pauli = c_opt.shape[0] // 2
    else:
        grp_ham_list = [grp_ham_re]
        num_split_pauli = c_opt.shape[0]

    for j, grp_ham in enumerate(grp_ham_list):
        offset = j * num_split_pauli
        count_pauli = 0
        for idx_grp, grp in enumerate(grp_pauli_list):
            block_coeff = c_opt[count_pauli + offset: count_pauli + offset + size_grp[idx_grp]]
            p_dict = dict()
            for idx_p, alloc_coeff in zip(grp, block_coeff):
                p_dict[operator(pauli_list[idx_p])] = alloc_coeff
            grp_ham.append(dict_to_operator(p_dict, QubitOperator))
            count_pauli += size_grp[idx_grp]

    if not correct_sum:
        return grp_ham_list

    # Checksum
    for j, grp_ham in enumerate(grp_ham_list):
        if len(grp_ham) > 1:
            ham_check = sum(grp_ham[1:], grp_ham[0])
            if not ham_check == ham:
                # Make Correction
                diff = ham_check - ham
                if debug:
                    print(f"Mismatching hamiltonaian : diff_norm={diff.induced_norm(order=2)}")
                    print(diff)
                for op, coeff in diff.terms.items():
                    included_grp = [i for i in range(len(grp_ham)) if op in grp_ham[i]]
                    if len(included_grp) == 0:
                        continue
                    coeff_split = coeff / len(included_grp)
                    for i in included_grp:
                        grp_ham_list[j][i] = grp_ham[i] + QubitOperator(op, coeff_split)
                assert ham.isclose(QubitOperator.accumulate(grp_ham), tol=1e-6), \
                    QubitOperator.accumulate(grp_ham) - ham
        else:
            assert grp_ham[0] == ham
    return grp_ham_list
