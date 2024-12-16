from itertools import product
from typing import Tuple, List, Optional

import numpy as np
from scipy.linalg import eigh

from ofex.operators.symbolic_operator_tools import operator


def _truncate(mat: np.ndarray):
    # TODO: check usage
    d, v = eigh(mat)
    assert np.allclose(d.imag, 0.0)
    d = d.real
    # sort_idx = d.argsort()[::-1]
    # d = d[sort_idx]
    for x in d:
        assert np.isclose(x, 0.0) or x > 0, d
    d[d < 0] = 0
    return v @ np.diag(d) @ (v.T.conj())


def _calculate_groupwise_std(transition: bool,
                             sep_reim: bool,
                             num_grp: int,
                             size_grp: List[int],
                             c_opt: np.ndarray,
                             cov_list_real: List[np.ndarray],
                             cov_list_imag: Optional[List[np.ndarray]]) \
        -> Tuple[List[float], List[float]]:
    count_pauli = 0
    std_real_list = list()
    std_imag_list = list() if transition else None
    half_c = c_opt.shape[0] // 2
    for idx_grp in range(num_grp):
        sz = size_grp[idx_grp]
        c_block_re = c_opt[count_pauli: count_pauli + sz]
        ov_re = 0.5 * c_block_re.T @ cov_list_real[idx_grp] @ c_block_re
        if ov_re < 0 and np.isclose(ov_re, 0.0, atol=1e-4):
            std_real_list.append(0.0)
        elif ov_re >= 0:
            std_real_list.append(np.sqrt(ov_re))
        else:
            raise ValueError((ov_re, c_block_re, cov_list_real[idx_grp]))
        if transition:
            if sep_reim:
                c_block_im = c_opt[count_pauli + half_c: count_pauli + half_c + sz]
            else:
                c_block_im = c_block_re
            ov_im = 0.5 * c_block_im.T @ cov_list_imag[idx_grp] @ c_block_im
            if ov_im < 0 and np.isclose(ov_im, 0.0, atol=1e-4):
                std_imag_list.append(0.0)
            elif ov_im >= 0:
                std_imag_list.append(np.sqrt(ov_im))
            else:
                raise ValueError((ov_im, c_block_im, cov_list_imag[idx_grp]))
        count_pauli += sz
    return std_real_list, std_imag_list


def _generate_cov_list(cov_dict, pauli_list, grp_pauli_list, size_grp, transition):
    cov_list_real: List[np.ndarray] = list()
    cov_list_imag: Optional[List[np.ndarray]] = list() if transition else None
    for idx_grp, grp in enumerate(grp_pauli_list):
        v_block_real = np.zeros((size_grp[idx_grp], size_grp[idx_grp]))
        v_block_imag = np.zeros((size_grp[idx_grp], size_grp[idx_grp])) \
            if transition else None
        for (i, idx_p), (j, idx_q) in product(enumerate(grp), repeat=2):
            """
            if idx_p > idx_q:
                v_block_real[i, j] = v_block_real[j, i]
                if transition:
                    v_block_imag[i, j] = v_block_imag[j, i]
                continue
            """
            p, q = pauli_list[idx_p], pauli_list[idx_q]
            p_op, q_op = operator(p), operator(q)
            cov_key = (p_op, q_op)
            if cov_key not in cov_dict:
                cov_key = (q_op, p_op)
            if transition:
                v_block_real[i, j], v_block_imag[i, j] = cov_dict[cov_key]
            else:
                v_block_real[i, j] = cov_dict[cov_key]
        v_block_real = _truncate(v_block_real)
        cov_list_real.append(v_block_real)
        if transition:
            v_block_imag = _truncate(v_block_imag)
            cov_list_imag.append(v_block_imag)
    return cov_list_real, cov_list_imag


def _add_epsilon_shot(m_arr, eps=1e-5):
    prev_sum = np.sum(m_arr)
    m_arr = np.array(m_arr)
    rich_list = list()
    for i, m in enumerate(m_arr):
        assert m >= 0
        if m < eps * 2:
            m_arr[i] += eps
        else:
            rich_list.append(i)
    assert len(rich_list) > 0
    distr = eps * (len(m_arr) - len(rich_list)) / len(rich_list)
    for i in rich_list:
        m_arr[i] -= distr
    assert np.allclose(np.sum(m_arr), prev_sum), (np.sum(m_arr), prev_sum)
    assert np.allclose(m_arr.imag, 0.0, atol=1e-1), m_arr
    return m_arr.real
