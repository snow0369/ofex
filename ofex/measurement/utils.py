import numpy as np

from ofex.linalg.sparse_tools import transition_amplitude, expectation


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
