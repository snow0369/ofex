import math
import os
import pickle
from itertools import product
from multiprocessing import Manager, Pool
from time import time
from typing import Optional, List, Union, Tuple

import numpy as np
from filelock import FileLock
from openfermion import QubitOperator, LinearQubitOperator

from ofex.linalg.sparse_tools import expectation, apply_operator, state_dot
from ofex.measurement.types import PauliCovDict, TransitionPauliCovDict
from ofex.measurement.utils import buf_transition_amplitude, buf_diag_expectation, buf_expectation
from ofex.operators.symbolic_operator_tools import operator, coeff
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import State


def pauli_variance(true_ref1: Optional[State],
                   true_ref2: Optional[State],
                   grp_ham: List[QubitOperator],
                   m_opt_real: np.ndarray,
                   m_opt_imag: Optional[np.ndarray],
                   true_cov_dict: Optional[Union[PauliCovDict, TransitionPauliCovDict]] = None,
                   anticommute: bool = False,
                   ) -> float:
    transition = m_opt_imag is not None
    tot_var = 0.0
    if true_cov_dict is not None:
        for idx, grp in enumerate(grp_ham):
            for p, q in product(grp.get_operators(), repeat=2):
                cov_key = (operator(p), operator(q))
                if cov_key not in true_cov_dict:
                    cov_key = (cov_key[1], cov_key[0])
                if cov_key not in true_cov_dict:
                    true_cov_dict[cov_key] = _calc_pauli_cov(p, q, true_ref1, true_ref2, anticommute)
                coeff_p, coeff_q = coeff(p), coeff(q)
                assert np.isclose(coeff_p.imag, 0.0)
                assert np.isclose(coeff_q.imag, 0.0)
                if not transition:
                    tot_var += 0.5 * coeff_p.real * coeff_q.real * true_cov_dict[cov_key] / m_opt_real[idx]
                else:
                    re, im = true_cov_dict[cov_key]
                    tot_var += 0.5 * coeff_p.real * coeff_q.real * (re / m_opt_real[idx] + im / m_opt_imag[idx])
    else:
        assert true_ref1 is not None
        assert transition == (true_ref2 is not None)
        n_qubits = get_num_qubits(true_ref1)
        for idx, grp in enumerate(grp_ham):
            op_grp = LinearQubitOperator(grp, n_qubits=n_qubits)
            if true_ref2 is None:
                if anticommute:
                    ov = expectation(op_grp, true_ref1)
                    ov2 = grp.induced_norm(order=2) ** 2
                else:
                    op_grp2 = LinearQubitOperator(grp * grp, n_qubits=n_qubits)
                    ov = expectation(op_grp, true_ref1)
                    ov2 = expectation(op_grp2, true_ref1)
                    assert np.isclose(ov.imag, 0.0)
                    assert np.isclose(ov2.imag, 0.0)
                tot_var += (ov2.real - ov.real ** 2) / m_opt_real[idx]
            else:
                app_ref1 = apply_operator(op_grp, true_ref1)
                app_ref2 = apply_operator(op_grp, true_ref2)
                ov = state_dot(app_ref1, true_ref2)
                if anticommute:
                    ov2 = grp.induced_norm(order=2) ** 2
                else:
                    ov2 = 0.5 * (state_dot(app_ref1, app_ref1) + state_dot(app_ref2, app_ref2))
                mr, mi = m_opt_real[idx], m_opt_imag[idx]
                assert np.isclose(ov2.imag, 0.0)
                ov2 = ov2.real
                tot_var += 0.5 * ((ov2 - ov.real ** 2) / mr + (ov2 - ov.imag ** 2) / mi)
    return tot_var


def pauli_covariance(initial_grp: Tuple[List[QubitOperator], List[List[int]], List[List[int]]],
                     ref1: State,
                     ref2: Optional[State],
                     num_workers=1,
                     anticommute=False,
                     cov_buf_path: Optional[str] = None,
                     debug: bool = False) \
        -> Union[PauliCovDict, TransitionPauliCovDict]:
    """
    Calculates Pauli covariance and optionally saves it as a pickle file(if cov_buf_path provided).
    Parallelism can be used for each group of Pauli terms.

    1) ref2 is None (P and Q commute.):
        Cov[P,Q] = <r1|PQ|r1> - <r1|P|r1><r1|Q|r1>
    2) ref2 is not None:
        Cov[P,Q]_R = 1/2 (<r1|{P, Q}|r1> + <r2|{P, Q}|r2>) - Re <r1|P|r2> * Re <r1|Q|r2>
        Cov[P,Q]_I = 1/2 (<r1|{P, Q}|r1> + <r2|{P, Q}|r2>) - Im <r1|P|r2> * Im <r1|Q|r2>

    :param pauli_list: Obtained from ics_partition_hamiltonian
    :param grp_pauli_list: Obtained from ics_partition_hamiltonian
    :param ref1: Bra-side unitary
    :param ref2: Ket-side unitary. If None, identical to ref1.
    :param num_workers: Number of parallel workers.
    :param anticommute: Anti-commutation partitioning
    :param cov_buf_path: Path(.pkl) to save covariance dictionary.
    :param debug:
    :return:
        covariance_dict: Dictionary of Pauli covariance (key=(P,Q), value=cov or (cov_R, cov_I)).
    """
    pauli_list, grp_pauli_list, pauli_grp_list = initial_grp

    t = time()
    mngr = Manager()
    cov_dict = mngr.dict()
    if cov_buf_path is not None and os.path.isfile(cov_buf_path):
        with open(cov_buf_path, "rb") as f:
            loaded_cov_dict = pickle.load(f)
        cov_dict.update(loaded_cov_dict)
        if debug:
            print(f"cov_dict Loaded from {cov_buf_path}")

    prod_dict1, prod_dict2 = mngr.dict(), mngr.dict()
    pool = Pool(processes=num_workers)
    pool.starmap(_pauli_covariance_sub, [(idx, cov_dict, (prod_dict1, prod_dict2), grp, pauli_list, ref1, ref2,
                                          anticommute, cov_buf_path, debug)
                                         for idx, grp in enumerate(grp_pauli_list)])
    pool.close()
    pool.join()
    if debug:
        print(f"cov_dict time = {time() - t}")
    return dict(cov_dict)


def _pauli_covariance_sub(idx,
                          cov_dict,
                          prod_dict,
                          grp: List[int],
                          pauli_list: List[QubitOperator],
                          ref1: State,
                          ref2: Optional[State],
                          anticommute: bool,
                          buf_path: Optional[str],
                          debug=False):
    prod_dict_off, prod_dict_diag = prod_dict  # Contains off-diagonal and diagonal transition amplitudes of pauli.
    if debug:
        print(f"grp idx = {idx}, len grp = {len(grp)} started.")

    update_buf = False

    for idx_p, idx_q in product(grp, repeat=2):
        if idx_p > idx_q:
            continue
        p, q = pauli_list[idx_p], pauli_list[idx_q]
        cov_key = (operator(p), operator(q))
        if cov_key in cov_dict:
            continue
        update_buf = True
        cov_dict[cov_key] = _calc_pauli_cov(p, q, ref1, ref2, anticommute, prod_dict_off, prod_dict_diag)
    if buf_path is not None and update_buf:
        fl = FileLock(buf_path + ".lock")
        with fl:
            with open(buf_path, "wb") as f:
                pickle.dump(dict(cov_dict), f)


def empirical_pauli_covariance(idx,
                               ov_dict: dict,
                               cov_dict: dict,
                               grp: List[int],
                               pauli_list: List[QubitOperator],
                               calc_diag: bool,
                               debug=False):
    prod_dict_off, prod_dict_diag = _parse_ov_dict(ov_dict)
    if debug:
        print(f"grp idx = {idx}, {len(grp)}")
    for idx_p, idx_q in product(grp, repeat=2):
        if idx_p > idx_q:
            continue
        p, q = pauli_list[idx_p], pauli_list[idx_q]
        p_op, q_op = operator(p), operator(q)
        cov_key = (p_op, q_op)
        if cov_key in cov_dict:
            continue
        if calc_diag:
            if idx_p == idx_q:
                ov = prod_dict_diag[p_op]
                assert np.isclose(ov.imag, 0.0)
                cov_dict[cov_key] = 1 - ov.real ** 2
            else:
                pq = QubitOperator(p_op, 1.0) * QubitOperator(q_op, 1.0)
                ov_p = prod_dict_diag[p_op]
                ov_q = prod_dict_diag[q_op]
                ov_pq = prod_dict_diag[operator(pq)] * coeff(pq)
                assert np.allclose([ov_p.imag, ov_q.imag, ov_pq.imag], 0.0)
                cov_dict[cov_key] = ov_pq.real - ov_p * ov_q
        else:
            if idx_p == idx_q:
                ov = prod_dict_off[p_op]
                cov_dict[cov_key] = (1 - ov.real ** 2), (1 - ov.imag ** 2)
            else:
                pq = QubitOperator(p_op, 1.0) * QubitOperator(q_op, 1.0)
                ov_p = prod_dict_off[p_op]
                ov_q = prod_dict_off[q_op]
                ov_pq = prod_dict_diag[operator(pq)]  # * pq.coeff
                assert np.isclose(ov_pq.imag, 0.0)
                cov_dict[cov_key] = ((ov_pq.real - ov_p.real * ov_q.real),
                                     (ov_pq.real - ov_p.imag * ov_q.imag))
            if math.isnan(cov_dict[cov_key][0]):
                print((1, idx, cov_key, cov_dict[cov_key]))
            if math.isnan(cov_dict[cov_key][0]):
                print((2, idx, cov_key, cov_dict[cov_key]))


def _calc_pauli_cov(op1, op2, ref1, ref2, anticommute,
                    prod_dict_off=None,
                    prod_dict_diag=None,):
    op1_key, op2_key = operator(op1), operator(op2)
    op1, op2 = QubitOperator(op1_key, 1.0), QubitOperator(op2_key, 1.0)
    if ref2 is not None:  # Transition Amplitude
        if op1_key == op2_key:  # 1 - ov^2
            ov = buf_transition_amplitude(op1, ref1, ref2, op1_key, prod_dict_off)
            return (1 - ov.real ** 2), (1 - ov.imag ** 2)
        else:  # <p1 p2> - ov^2
            ov_p = buf_transition_amplitude(op1, ref1, ref2, op1_key, prod_dict_off)
            ov_q = buf_transition_amplitude(op2, ref1, ref2, op2_key, prod_dict_off)
            if anticommute:
                return (- ov_p.real * ov_q.real), (- ov_p.imag * ov_q.imag)
            else:
                pq = op1 * op2
                pq_op = operator(pq)
                new_pq = QubitOperator(pq_op, 1.0)
                ex_pq = buf_diag_expectation(new_pq, ref1, ref2, pq_op, prod_dict_diag)
                ex_pq *= coeff(pq)
                assert np.isclose(ex_pq.imag, 0.0)
                return (ex_pq.real - ov_p.real * ov_q.real), \
                    (ex_pq.real - ov_p.imag * ov_q.imag)
    else:  # Expectation Value
        if op1_key == op2_key:  # 1 - ov^2
            ov = buf_expectation(op1, ref1, op1_key, prod_dict_off)
            assert np.isclose(ov.imag, 0.0)
            return 1 - ov.real ** 2
        else:  # <op1 op2> - ov^2
            if anticommute:
                sq = 0.0
            else:
                pq = op1 * op2
                pq_op = operator(pq)
                new_pq = QubitOperator(pq_op, 1.0)
                sq = buf_expectation(new_pq, ref1, pq_op, prod_dict_diag)
                sq *= coeff(pq)
                assert np.isclose(sq.imag, 0.0)
            ov_p = buf_expectation(op1, ref1, op1_key, prod_dict_off)
            ov_q = buf_expectation(op2, ref1, op2_key, prod_dict_off)
            assert np.isclose(ov_p.imag, 0.0)
            assert np.isclose(ov_q.imag, 0.0)
            return sq.real - ov_p.real * ov_q.real


def _parse_ov_dict(ov_dict):
    off_d, diag_d = dict(), dict()
    for k, v in ov_dict.items():
        op, q = tuple(k.split("_"))
        if q == "off":
            off_d[op] = v[-1]
        elif q == "diag":
            diag_d[op] = v[-1]
        else:
            raise ValueError
    return off_d, diag_d
