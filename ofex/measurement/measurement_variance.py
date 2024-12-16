import math
import os
import pickle
from itertools import product
from multiprocessing import Manager, Pool
from time import time
from typing import Optional, List, Union, Tuple, Dict

import numpy as np
from filelock import FileLock
from openfermion import QubitOperator, LinearQubitOperator

from ofex.linalg.sparse_tools import expectation, apply_operator, state_dot
from ofex.measurement.types import PauliCovDict, TransitionPauliCovDict, PhasedTransitionalPauliCovDict
from ofex.measurement.utils import buf_transition_amplitude, buf_diag_expectation, buf_expectation
from ofex.operators.symbolic_operator_tools import operator, coeff
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import State


def fragment_variance(grp_ham: List[QubitOperator],
                      state1: Optional[State],
                      state2: Optional[State],
                      shots: np.ndarray,
                      true_cov_dict: Optional[Union[PauliCovDict, TransitionPauliCovDict]] = None,
                      anticommute: bool = False) -> float:
    """
    Computes the variance of Hamiltonian fragments based on reference states, grouped Hamiltonian terms,
    and measurement shot counts.
    
    For the j-th fragment `H_j` in `grp_ham`, the total variance is computed as:
        V_tot = ∑_j (V_{R,j} / m_{R,j} + V_{I,j} / m_{I,j}),
    where:
        - V_{R,j} = 0.5 * (<φ1|H_j²|φ1> + <φ2|H_j²|φ2>) − (Re[<φ1|H_j|φ2>]²),
        - V_{I,j} = 0.5 * (<φ1|H_j²|φ1> + <φ2|H_j²|φ2>) − (Im[<φ1|H_j|φ2>]²).
        
    Here, m_{R,j} (shots_real) and m_{I,j} (shots_imag) represent the real and imaginary shot counts for `H_j`.
    Note that V_{(R,I),j} = ‖H_j‖₂² − (Re(Im)[<φ1|H_j|φ2>]²) if anticommute=True, indicating the Pauli operators
    supporting the fragment are anticommutative, where ‖H_j‖₂ is the 2nd order induced norm of H_j(L2 norm of Pauli
    coefficients or equivalently √Tr[H_j²]).

    **Case Explanation:**
        - For single-state variance calculations (`state2=None` and `shots.ndim==1`):
            V_tot = ∑_j (V_{R,j} / m_{R,j}).
    
        - For transition-state variance calculations (`state2` provided and `shots.ndim==2`):
            Both real and imaginary contributions are included in the computation.
    
    For further details, see [ArXiv:2409.02504](https://arxiv.org/abs/2409.02504).
    
    Args:
        grp_ham (List[QubitOperator]): A list of Hamiltonian fragments represented by QubitOperators.
        state1 (Optional[State]): The first reference state (required for computation).
        state2 (Optional[State]): The second reference state (optional; needed for transition-state variance).
        shots (np.ndarray): Array defining measurement shot counts:
            - If 1D (shape `[n]`): Represents real shot counts `m_{R,j}`; applicable for single-state variance.
            - If 2D (shape `[n, 2]`): Represents real `m_{R,j}` and imaginary `m_{I,j}` shot counts; used for 
              transition-state variance.
        true_cov_dict (Optional[Union[PauliCovDict, TransitionPauliCovDict]]): 
            An optional precomputed dictionary storing covariances of Pauli operators.
        anticommute (bool): If True, applies anticommutation rules during the computation.
    
    Returns:
        float: Total computed variance, calculated as a sum of contributions from all Pauli operator groups.
    """
    if shots.ndim == 2:
        shots_real = shots[:, 0]
        shots_imag = shots[:, 1]
        transition = True
    elif shots.ndim == 1:
        shots_real = shots[:, ]
        shots_imag = None
        transition = False
    else:
        raise ValueError("Invalid shots array dimensions. Expected 1D or 2D array.")
    tot_var = 0.0
    if true_cov_dict is not None:
        for idx, grp in enumerate(grp_ham):
            for p, q in product(grp.get_operators(), repeat=2):
                cov_key = (operator(p), operator(q))
                if cov_key not in true_cov_dict:
                    cov_key = (cov_key[1], cov_key[0])
                if cov_key not in true_cov_dict:
                    true_cov_dict[cov_key] = _calc_pauli_cov(p, q, state1, state2, ph=0, anticommute=anticommute)
                coeff_p, coeff_q = coeff(p), coeff(q)
                assert np.isclose(coeff_p.imag, 0.0)
                assert np.isclose(coeff_q.imag, 0.0)
                if not transition:
                    tot_var += 0.5 * coeff_p.real * coeff_q.real * true_cov_dict[cov_key] / shots_real[idx]
                else:
                    re, im = true_cov_dict[cov_key]
                    tot_var += 0.5 * coeff_p.real * coeff_q.real * (re / shots_real[idx] + im / shots_imag[idx])
    else:
        if state1 is None:
            raise ValueError(f"Input state(s) should be given unless true_cov_dict is provided.")
        if transition != (state2 is not None):
            raise ValueError("state2 must be provided when transition calculations are needed and"
                             "must be None otherwise.")
        n_qubits = get_num_qubits(state1)
        for idx, grp in enumerate(grp_ham):
            op_grp = LinearQubitOperator(grp, n_qubits=n_qubits)
            if state2 is None:
                if anticommute:
                    ov = expectation(op_grp, state1)
                    ov2 = grp.induced_norm(order=2) ** 2
                else:
                    op_grp2 = LinearQubitOperator(grp * grp, n_qubits=n_qubits)
                    ov = expectation(op_grp, state1)
                    ov2 = expectation(op_grp2, state1)
                    assert np.isclose(ov.imag, 0.0)
                    assert np.isclose(ov2.imag, 0.0)
                tot_var += (ov2.real - ov.real ** 2) / shots_real[idx]
            else:
                app_state1 = apply_operator(op_grp, state1)
                app_state2 = apply_operator(op_grp, state2)
                ov = state_dot(app_state1, state2)
                if anticommute:
                    ov2 = grp.induced_norm(order=2) ** 2
                else:
                    ov2 = 0.5 * (state_dot(app_state1, app_state1) + state_dot(app_state2, app_state2))
                mr, mi = shots_real[idx], shots_imag[idx]
                assert np.isclose(ov2.imag, 0.0)
                ov2 = ov2.real
                tot_var += 0.5 * ((ov2 - ov.real ** 2) / mr + (ov2 - ov.imag ** 2) / mi)
    return tot_var


def pauli_covariance(pauli_list: List[QubitOperator],
                     grp_pauli_list: List[List[int]],
                     state1: State,
                     state2: Optional[State],
                     num_workers=1,
                     anticommute=False,
                     cov_buf_dir: Optional[str] = None,
                     phase_list: Optional[List[float]] = None,
                     debug: bool = False) \
        -> Union[PauliCovDict, TransitionPauliCovDict, PhasedTransitionalPauliCovDict]:
    """
    Computes the Pauli covariance matrix for a given set of Pauli operators and states.
    
    This function supports parallel computation and optionally saves covariance results to files.
    The covariance formulas depend on whether a single reference state (`state2=None`) or transition
    states (`state2` provided) are used.
    
    Covariance cases (∀P,Q ∈ pauli_list[grp_pauli_list[grp_idx]] ∀grp_idx):
        1. Single-state (`state2=None`; operators commute):
            Cov[P,Q] = <φ1|PQ|φ1> - <φ1|P|φ1><φ1|Q|φ1>
        2. Transition states (`state2` provided):
            Cov[P,Q]_R = 1/2 (<φ1|{P, Q}|φ1> + <φ2|{P, Q}|φ2>) - Re[<φ1|P|φ2>] * Re[<φ1|Q|φ2>]
            Cov[P,Q]_I = 1/2 (<φ1|{P, Q}|φ1> + <φ2|{P, Q}|φ2>) - Im[<φ1|P|φ2>] * Im[<φ1|Q|φ2>].
        For transitional states, note that anticommute=True, the first term is ignored.

    If phase_list is given, Cov[P,Q] * exp(-1j * ph) for ph in phase_list is calculated, which is implemented to
    avoid duplicated computation for different phase values.

    For further details, see [ArXiv:2409.02504](https://arxiv.org/abs/2409.02504).

    Args:
        pauli_list (List[QubitOperator]): A list of Pauli operators for which covariances are calculated.
        grp_pauli_list (List[List[int]]): A list of groups, where each group is a list of indices referring to subsets 
            of `pauli_list`. Each subset indicates Pauli operator groups over which covariance calculation 
            should be carried out. This grouping allows efficient parallel computation or targeted subset evaluations.
        state1 (State): The first reference state, provided as input.
        state2 (Optional[State]): The second reference state, required for calculating transition covariances.
        num_workers (int): Number of parallel workers to distribute computation.
        anticommute (bool): Enforces anticommutation rules if True.
        cov_buf_dir (Optional[str]): Directory path to save intermediate covariance results as pickle files.
        phase_list (Optional[List[float]]): List of phase values for covariance calculations.
        debug (bool): If True, prints debugging information during execution.
    
    Returns:
        Union[PauliCovDict, TransitionPauliCovDict, PhasedTransitionalPauliCovDict]:
            A dictionary where keys are Pauli operator pairs and values are covariance coefficients.
            - PauliCovDict: For single-state covariance calculations.
            - TransitionPauliCovDict: For transition-state covariance calculations without phase shift.
            - PhasedTransitionalPauliCovDict: For phased transition-state covariance calculations.
    """
    was_ph_none = False
    if phase_list is None:
        phase_list = [0.0]
        was_ph_none = True

    t = time()
    mngr = Manager()
    cov_dict_list = {p: mngr.dict() for p in phase_list}
    fname_list = None
    if cov_buf_dir is not None:
        fname_list = {p: os.path.join(cov_buf_dir, "phase={:.6f}.pkl".format(p)) for p in phase_list}
        if not os.path.isdir(cov_buf_dir):
            os.mkdir(cov_buf_dir)
        for p, fname in fname_list.items():
            if os.path.isfile(fname):
                with open(fname, 'rb') as f:
                    loaded_cov_dict = pickle.load(f)
                cov_dict_list[p].update(loaded_cov_dict)
                if debug:
                    print(f"cov_dict Loaded from {fname}")

    prod_dict1, prod_dict2 = mngr.dict(), mngr.dict()
    pool = Pool(processes=num_workers)
    pool.starmap(_pauli_covariance_sub, [(idx, cov_dict_list, (prod_dict1, prod_dict2), grp, pauli_list, state1, state2,
                                          anticommute, fname_list, debug)
                                         for idx, grp in enumerate(grp_pauli_list)])
    pool.close()
    pool.join()
    if debug:
        print(f"cov_dict time = {time() - t}")

    if was_ph_none:
        return dict(cov_dict_list[0.0])
    else:
        return {p: dict(cov_dict_list[p]) for p in phase_list}


def _pauli_covariance_sub(idx,
                          cov_dict_list,
                          prod_dict,
                          grp: List[int],
                          pauli_list: List[QubitOperator],
                          state1: State,
                          state2: Optional[State],
                          anticommute: bool,
                          fname_list: Optional[Dict[float, str]],
                          debug=False):
    """
    Subroutine for computing covariance of a subset of Pauli operators in parallel.
    
    This function iterates through paired Pauli operator groups, calculating the covariance values,
    and optionally updates a covariance buffer for intermediate storage.
    
    Args:
        idx (int): Index of the current group being processed in the parallel run.
        cov_dict_list (dict): Shared manager dictionary storing computed covariance values indexed by phase.
        prod_dict (tuple): Tuple containing shared dictionaries for diagonal and off-diagonal products.
        grp (List[int]): List of integer indices representing a group of Pauli operators.
        pauli_list (List[QubitOperator]): List of all available Pauli operators.
        state1 (State): The first reference state required for covariance computation.
        state2 (Optional[State]): The second reference state for transition covariance computations.
        anticommute (bool): If True, applies anticommutative calculations in operator grouping.
        fname_list (Optional[Dict[float, str]]): Dictionary mapping phase values to buffer filenames for saving results.
        debug (bool): Enables verbose logging for debugging purposes.
    """
    prod_dict_off, prod_dict_diag = prod_dict  # Contains off-diagonal and diagonal transition amplitudes of pauli.
    if debug:
        print(f"grp idx = {idx}, len grp = {len(grp)} started.")

    update_buf = False

    for idx_p, idx_q in product(grp, repeat=2):
        if idx_p > idx_q:
            continue
        p, q = pauli_list[idx_p], pauli_list[idx_q]
        cov_key = (operator(p), operator(q))
        for ph, cov_dict in cov_dict_list.items():
            if cov_key in cov_dict:
                continue
            update_buf = True
            cov_dict[cov_key] = _calc_pauli_cov(p, q, state1, state2, ph, anticommute, prod_dict_off, prod_dict_diag)
    if fname_list is not None and update_buf:
        for p, buf_path in fname_list.items():
            fl = FileLock(buf_path + ".lock")
            with fl:
                with open(buf_path, "wb") as f:
                    pickle.dump(dict(cov_dict_list[p]), f)


def _calc_pauli_cov(op1, op2, state1, state2, ph, anticommute,
                    prod_dict_off=None,
                    prod_dict_diag=None, ):
    """
    Computes covariance between two Pauli operators for single-state or transition-state cases.
    
    This function allows calculation of diagonal or off-diagonal covariance terms based on provided
    operator inputs and states. The outputs depend on whether anticommute rules are applied.
    
    Args:
        op1 (QubitOperator): The first Pauli operator.
        op2 (QubitOperator): The second Pauli operator.
        state1 (State): The first reference state required for overlap computation.
        state2 (Optional[State]): The second reference state for calculating transition amplitude terms.
        ph (float): Phase shift used during the computation.
        anticommute (bool): If True, applies anticommutative covariance calculations.
        prod_dict_off (Optional[dict]): Dictionary of precomputed off-diagonal products (optional).
        prod_dict_diag (Optional[dict]): Dictionary of diagonal expectation values (optional).
    
    Returns:
        Union[float, Tuple[float, float]]: Covariance values for single or transition (real, imaginary) cases.
    """
    op1_key, op2_key = operator(op1), operator(op2)
    op1, op2 = QubitOperator(op1_key, 1.0), QubitOperator(op2_key, 1.0)
    if state2 is not None:  # Transition Amplitude
        if op1_key == op2_key:  # 1 - ov^2
            ov = buf_transition_amplitude(op1, state1, state2, ph, op1_key, prod_dict_off)
            return (1 - ov.real ** 2), (1 - ov.imag ** 2)
        else:  # <p1 p2> - ov^2
            ov_p = buf_transition_amplitude(op1, state1, state2, ph, op1_key, prod_dict_off)
            ov_q = buf_transition_amplitude(op2, state1, state2, ph, op2_key, prod_dict_off)
            if anticommute:
                return (- ov_p.real * ov_q.real), (- ov_p.imag * ov_q.imag)
            else:
                pq = op1 * op2
                pq_op = operator(pq)
                new_pq = QubitOperator(pq_op, 1.0)
                ex_pq = buf_diag_expectation(new_pq, state1, state2, pq_op, prod_dict_diag)
                ex_pq *= coeff(pq)
                assert np.isclose(ex_pq.imag, 0.0)
                return (ex_pq.real - ov_p.real * ov_q.real), \
                    (ex_pq.real - ov_p.imag * ov_q.imag)
    else:  # Expectation Value
        if op1_key == op2_key:  # 1 - ov^2
            ov = buf_expectation(op1, state1, op1_key, prod_dict_off)
            assert np.isclose(ov.imag, 0.0)
            return 1 - ov.real ** 2
        else:  # <op1 op2> - ov^2
            if anticommute:
                sq = 0.0
            else:
                pq = op1 * op2
                pq_op = operator(pq)
                new_pq = QubitOperator(pq_op, 1.0)
                sq = buf_expectation(new_pq, state1, pq_op, prod_dict_diag)
                sq *= coeff(pq)
                assert np.isclose(sq.imag, 0.0)
            ov_p = buf_expectation(op1, state1, op1_key, prod_dict_off)
            ov_q = buf_expectation(op2, state1, op2_key, prod_dict_off)
            assert np.isclose(ov_p.imag, 0.0)
            assert np.isclose(ov_q.imag, 0.0)
            return sq.real - ov_p.real * ov_q.real


def empirical_pauli_covariance(idx,
                               ov_dict: dict,
                               cov_dict: dict,
                               grp: List[int],
                               pauli_list: List[QubitOperator],
                               calc_diag: bool,
                               debug=False):
    """
    Computes empirical covariance of Pauli operators using a provided overlap dictionary.

    This function uses overlap dictionaries to compute diagonal and off-diagonal covariance
    terms for the specified group of Pauli operators.

    Args:
        idx (int): Numerical index identifying the operator group being processed.
        ov_dict (dict): Dictionary containing precomputed overlap values for Pauli operators.
        cov_dict (dict): Output dictionary to store the resulting covariance values.
        grp (List[int]): List of indices indicating the subset of operators being processed.
        pauli_list (List[QubitOperator]): List of available Pauli operators used in the computation.
        calc_diag (bool): When True, computes diagonal terms in addition to off-diagonal covariance terms.
        debug (bool): If True, enables detailed logging for debugging the function's execution.
    """
    # Parse ov_dict
    prod_dict_off, prod_dict_diag = dict(), dict()
    for k, v in ov_dict.items():
        op, q = tuple(k.split("_"))
        if q == "off":
            prod_dict_off[op] = v[-1]
        elif q == "diag":
            prod_dict_diag[op] = v[-1]
        else:
            raise ValueError

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

