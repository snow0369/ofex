from typing import List, Optional, Tuple, Union

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE
from scipy.linalg import eigh

from ofex.measurement.iterative_coefficient_splitting.ics_utils import (_calculate_groupwise_std, _generate_cov_list,
                                                                        _add_epsilon_shot)
from ofex.measurement.types import PauliCovDict, TransitionPauliCovDict
from ofex.measurement.utils import synthesize_group
from ofex.operators.symbolic_operator_tools import coeff, compare_operators


def run_ics(ham: QubitOperator,
            initial_grp: Tuple[List[QubitOperator], List[List[int]], List[List[int]]],
            cov_dict: Union[PauliCovDict, TransitionPauliCovDict],
            transition: bool = False,
            sep_reim: bool = False,
            conv_atol: float = 1e-6,
            conv_rtol: Optional[float] = 1e-4,
            max_iter=10000,
            lstsq_rcond=1e-6,
            initial_c: Optional[np.ndarray] = None,
            debug: bool = False, ) \
        -> Tuple[List[QubitOperator], np.ndarray, float, Optional[np.ndarray]]:
    """
    Runs the Iterative Coefficient Splitting (ICS) optimization algorithm for Hamiltonian decomposition.

    References:
        - https://arxiv.org/abs/2201.01471
        - https://arxiv.org/abs/2409.02504

    Args:
        ham (QubitOperator): Initial Hamiltonian that needs to be decomposed.
        initial_grp (Tuple[List[QubitOperator], List[List[int]], List[List[int]]]): 
            Initial grouping of the Hamiltonian into compatible groups:
            - pauli_list: List of all Pauli operators from the Hamiltonian.
            - grp_pauli_list: List of Pauli operators in each group.
            - pauli_grp_list: List mapping Pauli operators to their respective group indices.
            Those are the output of ofex.measurement.iterative_coefficient_splitting.init_ics
        cov_dict (Union[PauliCovDict, TransitionPauliCovDict]): 
            A dictionary of covariances. The keys are tuples of Pauli strings, and the values are either real parts
            or a combination of real and imaginary parts (for `transition=True`).
        transition (bool, optional): If set to True, calculates transition amplitudes (applies when `ref1` and 
            `ref2` are different states).
        sep_reim (bool, optional): If True, separates the real and imaginary covariance elements. Valid only with 
            `transition=True`.
        conv_atol (float, optional): Absolute tolerance for convergence. Default is 1e-6.
        conv_rtol (Optional[float], optional): Relative tolerance for convergence (default: 1e-4). If None, only absolute
            tolerance will be used.
        max_iter (int, optional): Maximum number of iterations allowed for the ICS optimization loop. Default is 10,000.
        lstsq_rcond (float, optional): Cut-off value for singular values in least-squares optimization. Default: 1e-6.
        initial_c (Optional[np.ndarray], optional): Initial coefficients for optimization. If None, automatically populated.
        debug (bool, optional): If True, prints debugging information to the console.
    
    Returns:
        Tuple:
            - List[QubitOperator]: Grouped Hamiltonians obtained from the optimization process.
            - np.ndarray: If `transition` is `True`, a 2D array with shape `(2, n_frag)` containing shot allocations 
              for the real and imaginary parts. If `transition` is `False`, a 1D array with shot allocations for 
              the real part only.
            - float: Final optimal measurement variance achieved.
            - Optional[np.ndarray]: Optimized coefficients for the group Hamiltonians.
    """
    if ham.constant != 0.0:
        raise ValueError
    if sep_reim and not transition:
        raise ValueError

    # 1. Partitioning into compatible groups
    pauli_list, grp_pauli_list, pauli_grp_list = initial_grp

    # 2. Generate constraints : A c = b (c = splitted coefficients, b = merged coefficients)
    num_pauli = len(pauli_list)
    size_grp = [len(x) for x in grp_pauli_list]
    num_split_pauli = sum(size_grp)
    num_grp = len(grp_pauli_list)
    # 2-1. A matrix
    a_mat = np.zeros((num_pauli, num_split_pauli), dtype=int)
    count_pauli = 0
    for idx_grp in range(num_grp):
        for idx_p, p in enumerate(grp_pauli_list[idx_grp]):
            a_mat[p, count_pauli + idx_p] = 1
        count_pauli += size_grp[idx_grp]
    a_mat = a_mat.astype(float)
    # 2-2. b vector
    b_vec = np.array([coeff(p) for p in pauli_list])
    if not np.allclose(b_vec.imag, 0.0):
        raise ValueError
    b_vec = b_vec.real

    if sep_reim:
        new_amat = np.zeros((num_pauli * 2, num_split_pauli * 2), dtype=float)
        new_amat[:a_mat.shape[0], :a_mat.shape[1]] = a_mat
        new_amat[a_mat.shape[0]:, a_mat.shape[1]:] = a_mat
        new_bvec = np.zeros(b_vec.shape[0] * 2, dtype=float)
        new_bvec[:b_vec.shape[0]] = b_vec
        new_bvec[b_vec.shape[0]:] = b_vec
        a_mat = new_amat
        b_vec = new_bvec

    # 3. Generate variance
    # List of covariance matrix for each groups
    cov_list_real, cov_list_imag = _generate_cov_list(cov_dict, pauli_list, grp_pauli_list, size_grp, transition)
    """
    if debug:
        cond_r = np.linalg.cond(v_block_real)
        dr, _ = eigh(v_block_real)
        pd = all([x > 0 or np.isclose(x, 0.0) for x in dr])
        print(f"{idx_grp} : cond(VR) = {cond_r}, POS = {pd}")
        if transition:
            cond_i = np.linalg.cond(v_block_imag)
            di, _ = eigh(v_block_real)
            pd = all([x > 0 or np.isclose(x, 0.0) for x in di])
            print(f"{idx_grp} : cond(VI) = {cond_i}, POS = {pd}")
        else:
            cond_i = 0
    """
    if debug:
        print("cov_dict done")

    # 4. perform optimization
    # 4-0. Initial m allocation
    if initial_c is not None:
        init_std_real_list, init_std_imag_list = _calculate_groupwise_std(
            transition, False, num_grp, size_grp, initial_c, cov_list_real, cov_list_imag
        )
        if transition:
            init_sum_std = sum(init_std_real_list) + sum(init_std_imag_list)
            m_opt_real = np.array([s / init_sum_std for s in init_std_real_list])
            m_opt_imag = np.array([s / init_sum_std for s in init_std_imag_list])
        else:
            init_sum_std = sum(init_std_real_list)
            m_opt_real = np.array([s / init_sum_std for s in init_std_real_list])
            m_opt_imag = None
        if debug:
            print(f"init_var : {init_sum_std ** 2}")
    elif transition:
        m_opt_real = np.ones(num_grp, dtype=float) / (2 * num_grp)
        m_opt_imag = np.ones(num_grp, dtype=float) / (2 * num_grp)
    else:
        m_opt_real = np.ones(num_grp, dtype=float) / num_grp
        m_opt_imag = None

    m_opt_real = _add_epsilon_shot(m_opt_real)
    if m_opt_imag is not None:
        m_opt_imag = _add_epsilon_shot(m_opt_imag)

    m_tot = sum(m_opt_real) + sum(m_opt_imag) if transition else sum(m_opt_real)
    assert np.isclose(m_tot, 1.0), m_tot

    # Prepare modified quadratic programming for non-full ranked problem.
    # To satisfy the constraint : c_opt = phi_s + projector @ phi
    a_pinv = a_mat.T @ np.linalg.inv(a_mat @ a_mat.T)  # pseudo inverse of A
    phi_s = a_pinv @ b_vec
    if sep_reim:
        projector = np.eye(num_split_pauli * 2) - a_pinv @ a_mat
    else:
        projector = np.eye(num_split_pauli) - a_pinv @ a_mat

    # Perform iterative optimization
    sum_std = None
    converge = False
    max_trial = 10
    trial = 1
    add_shot_eps = 1e-6
    truncate_epsilon = 1e-9
    while True:
        m_opt_real = _add_epsilon_shot(m_opt_real, eps=add_shot_eps)
        if m_opt_imag is not None:
            m_opt_imag = _add_epsilon_shot(m_opt_imag, eps=add_shot_eps)

        m_tot = sum(m_opt_real) + sum(m_opt_imag) if transition else sum(m_opt_real)
        assert np.isclose(m_tot, 1.0), m_tot
        try:
            if sep_reim and initial_c is not None:
                c_opt = np.zeros(initial_c.shape[0] * 2, dtype=float)
                c_opt[:initial_c.shape[0]] = initial_c
                c_opt[initial_c.shape[0]:] = initial_c
            else:
                c_opt = initial_c
            for it in range(max_iter):
                # 4-1. construct total covariance matrix
                if sep_reim:
                    tot_vmat = np.zeros((2 * num_split_pauli, 2 * num_split_pauli))
                else:
                    tot_vmat = np.zeros((num_split_pauli, num_split_pauli))
                count_pauli = 0
                for idx_grp in range(num_grp):
                    end_idx = count_pauli + size_grp[idx_grp]
                    if not np.isclose(m_opt_real[idx_grp], 0):
                        tot_vmat[count_pauli: end_idx, count_pauli: end_idx] \
                            = cov_list_real[idx_grp] / m_opt_real[idx_grp]
                    if transition and (not sep_reim) and (not np.isclose(m_opt_imag[idx_grp], 0)):
                        tot_vmat[count_pauli: end_idx, count_pauli: end_idx] \
                            += cov_list_imag[idx_grp] / m_opt_imag[idx_grp]
                    elif sep_reim and not np.isclose(m_opt_imag[idx_grp], 0):
                        tot_vmat[count_pauli + num_split_pauli: end_idx + num_split_pauli,
                                 count_pauli + num_split_pauli: end_idx + num_split_pauli] \
                            = cov_list_imag[idx_grp] / m_opt_imag[idx_grp]

                    count_pauli += size_grp[idx_grp]
                if converge:
                    break

                # 4-2. Solve coefficients
                mu = projector.T @ tot_vmat @ phi_s
                vmat_tilde = projector.T @ tot_vmat @ projector
                diag_v, u_v = eigh(vmat_tilde)
                for i, dv in enumerate(diag_v):
                    if dv < truncate_epsilon:
                        diag_v[i] = 0.0
                vmat_tilde = u_v @ np.diag(diag_v) @ u_v.T.conj()
                phi, res, rank, sing = np.linalg.lstsq(vmat_tilde, -mu, rcond=lstsq_rcond)
                """
                if debug:
                    d, _ =eigh(tot_vmat)
                    if not all([np.isclose(x, 0.0) or x > 0 for x in d]):
                        print(f"Non-proper singular values : {[x for x in d if x < 0 and not np.isclose(x, 0.0)]}")
                """
                new_c_opt = phi_s + projector @ phi

                # 4-3. Solve measurement allocation
                # 4-3-1. Calculate group-wise standard deviations
                std_real_list, std_imag_list = _calculate_groupwise_std(transition, sep_reim, num_grp, size_grp,
                                                                        new_c_opt, cov_list_real, cov_list_imag)

                # 4-3-2. Check Convergence
                new_sum_std = sum(std_real_list)
                if transition:
                    new_sum_std += sum(std_imag_list)
                if sum_std is not None and new_sum_std > sum_std:
                    break

                converge = False
                if sum_std is not None:
                    if conv_rtol is not None and abs(sum_std) > EQ_TOLERANCE:
                        rtol_check = abs(new_sum_std - sum_std) / sum_std < conv_rtol
                    else:
                        rtol_check = False
                    converge = (abs(new_sum_std - sum_std) < conv_atol) or rtol_check

                sum_std = new_sum_std
                c_opt = new_c_opt

                # 4-3-3. Allocate shots
                m_opt_real = np.array([s / sum_std for s in std_real_list])
                if transition:
                    m_opt_imag = np.array([s / sum_std for s in std_imag_list])

                m_opt_real = _add_epsilon_shot(m_opt_real, eps=add_shot_eps)
                if m_opt_imag is not None:
                    m_opt_imag = _add_epsilon_shot(m_opt_imag, eps=add_shot_eps)

                m_tot = sum(m_opt_real) + sum(m_opt_imag) if transition else sum(m_opt_real)
                assert np.isclose(m_tot, 1.0), m_tot

                if debug:
                    print(f"it {it} : {sum_std ** 2}")
                    grp_ham_list = synthesize_group(sep_reim, c_opt, pauli_list, grp_pauli_list, size_grp, ham, debug,
                                                    correct_sum=False)
                    for grp_ham in grp_ham_list:
                        checksum = QubitOperator.accumulate(grp_ham)
                        if not checksum.isclose(ham):
                            print(f"mismatch during optimization : {(checksum - ham).induced_norm(order=2)}")
                            print(compare_operators(checksum, ham, str_len=300))
            else:
                raise Warning("Reached Max Iteration")
            break
        except np.linalg.LinAlgError as e:
            if trial < max_trial:
                print(f"Error occured, retry: {trial}")
                print(e)
                trial += 1
                add_shot_eps *= 2.0
            else:
                raise e

    # 5. Synthesize group
    grp_ham_list = synthesize_group(sep_reim, c_opt, pauli_list, grp_pauli_list, size_grp, ham,
                                    debug)  # , leftover, anticommute, debug)
    final_variance = 0.5 * c_opt.T @ tot_vmat @ c_opt

    n_frag = len(grp_ham_list[0])
    if transition:
        shots = np.zeros((2, n_frag), dtype=float)
        shots[0, :] = m_opt_real
        shots[1, :] = m_opt_imag
    else:
        shots = m_opt_real

    if not sep_reim:
        return grp_ham_list[0], shots, final_variance, c_opt
    else:
        return grp_ham_list, shots, final_variance, c_opt
