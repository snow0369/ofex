import copy
import os
import pickle
from itertools import product
from typing import Union, Optional, Tuple

import numpy as np
import numpy.typing as npt
from openfermion import QubitOperator
from scipy.sparse import spmatrix

from algorithms.qksd.qksd_utils import toeplitz_arr_to_mat
from ofex.linalg.sparse_tools import apply_operator, state_dot, expectation, sparse_apply_operator, transition_amplitude
from ofex.sampling_simulation.hadamard_test import hadamard_test_qubit_operator, hadamard_test_general
from ofex.sampling_simulation.qksd_extended_swap_test import qksd_extended_swap_test, prepare_qksd_est_op, \
    prepare_qksd_est_state
from ofex.sampling_simulation.sampling_base import ProbDist
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import State

FH, LCU = 0, 1
REAL, IMAG = 0, 1
H, S = 0, 1


def ideal_qksd_toeplitz(pham: QubitOperator,
                        prop: Union[spmatrix, np.ndarray],
                        ref: State,
                        n_krylov: int) -> Tuple[np.ndarray, np.ndarray]:
    ksd_state = copy.deepcopy(ref)
    h_ref = apply_operator(pham, ref)
    s_arr, h_arr = np.zeros(n_krylov, dtype=complex), np.zeros(n_krylov, dtype=complex)

    for i in range(n_krylov):
        s_arr[i] = state_dot(ref, ksd_state)
        h_arr[i] = state_dot(h_ref, ksd_state)
        if i != n_krylov - 1:
            ksd_state = apply_operator(prop, ksd_state)

    s_mat = toeplitz_arr_to_mat(s_arr)
    h_mat = toeplitz_arr_to_mat(h_arr)

    return h_mat, s_mat


def ideal_qksd_nontoeplitz(pham: QubitOperator,
                           prop: Union[spmatrix, np.ndarray],
                           ref: State,
                           n_krylov: int) -> Tuple[np.ndarray, np.ndarray]:
    basis = list()
    h_basis = list()
    ksd_state = copy.deepcopy(ref)
    for i in range(n_krylov):
        basis.append(ksd_state)
        h_basis.append(apply_operator(pham, ksd_state))
        if i != n_krylov - 1:
            ksd_state = apply_operator(prop, ksd_state)

    s_mat, h_mat = np.zeros((n_krylov, n_krylov), dtype=complex), np.zeros((n_krylov, n_krylov), dtype=complex)

    for i, j in product(range(n_krylov), repeat=2):
        if j < i:
            s_mat[i, j] = s_mat[j, i].conjugate()
            h_mat[i, j] = h_mat[j, i].conjugate()
        else:
            s_mat[i, j] = state_dot(basis[i], basis[j])
            h_mat[i, j] = state_dot(basis[i], h_basis[j])

    return h_mat, s_mat


def sample_qksd_toeplitz(ham_frag: npt.NDArray[QubitOperator],
                         prop: Union[spmatrix, np.ndarray],
                         ref: State,
                         n_krylov: int,
                         meas_type: str,
                         shot_list: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                         sample_buf_dir: Optional[str] = None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    # For FH
    # shot_list[idx_n_krylov][idx_ham_frag][real=0, imag=1]

    # For LCU,
    # shot_list[h=0][idx_n_krylov][idx_unitary ][real=0, imag=1]
    # shot_list[s=1][idx_n_krylov][real=0, imag=1]

    # Check input
    meas_type = meas_type.upper()
    if meas_type not in ["LCU", "FH"]:
        raise ValueError("meas_type must be 'LCU' or 'FH'.")

    n_qubits = get_num_qubits(ref)

    # ham_frag.ndim == 1 : The fragmentation is uniform across the krylov indices and real/imaginary part.
    #                       ( ham_frag[idx_frag] )
    # ham_frag.ndim == 2 : The fragmentation is differed by krylov indices.
    #                       ( ham_frag[idx_krylov][idx_frag] )
    # ham_frag.ndim == 3 : The fragmentation is differed by krylov indices and real/imaginary part.
    #                       ( ham_frag[idx_krylov][REAL/IMAG][idx_frag] )
    ham_frag = np.array(ham_frag)
    n_frag = ham_frag.shape[-1]

    # Find hamiltonian by accumulating the fragments.
    if ham_frag.ndim == 1:
        pham = QubitOperator.accumulate(ham_frag)
    elif ham_frag.ndim in [2, 3]:
        first_idx = tuple(0 for _ in range(ham_frag.ndim - 1))
        pham = QubitOperator.accumulate(ham_frag[first_idx])
    else:
        raise ValueError

    # Initial preparation
    ksd_state = copy.deepcopy(ref)
    s_arr, h_arr = np.zeros(n_krylov, dtype=complex), np.zeros(n_krylov, dtype=complex)

    # Assign the 0th elements without sampling.
    h_arr[0] = expectation(pham, ref, sparse=True)
    s_arr[0] = 1.0
    ksd_state = apply_operator(prop, ksd_state)

    # Load prob. dist. for the simulation from the previously generated files.
    loaded_prob_dist = dict()
    use_prob_buffer = sample_buf_dir is not None
    if sample_buf_dir is not None:
        if os.path.exists(sample_buf_dir):
            f_list = os.listdir(sample_buf_dir)
            for fname in f_list:
                idx_krylov = int(fname.split('.')[0])
                assert idx_krylov not in loaded_prob_dist
                with open(os.path.join(sample_buf_dir, fname), 'rb') as f:
                    loaded_prob_dist[idx_krylov] = pickle.load(f)
        else:
            os.mkdir(sample_buf_dir)

    # Run the sampling
    if meas_type == "LCU":
        # Normalize to make it unitary
        norm_list = np.zeros(ham_frag.shape, dtype=float)
        for idx in product(*[range(s) for s in ham_frag.shape]):
            norm_list[idx] = ham_frag[idx].induced_norm(order=2)
            ham_frag[idx] = ham_frag[idx] / norm_list[idx]

        # Prepare objects for the Hadamard Test simulation.
        ref_evol_1d, norm_re_1d, norm_im_1d = None, None, None
        if ham_frag.ndim == 1:
            ref_evol_1d = [sparse_apply_operator(h, ref) for h in ham_frag]
            norm_re_1d, norm_im_1d = norm_list, norm_list

        # Sampling simulation
        for i in range(1, n_krylov):
            # Evaluate the sampling if not loaded from the buffer.
            if i not in loaded_prob_dist:
                # Prepare overlaps
                # Note that the probability distribution of Hadamard test outcomes are determined
                # by the overlap values.
                if ham_frag.ndim == 1:
                    ov_list = [state_dot(ref_evol, ksd_state) for ref_evol in ref_evol_1d]
                elif ham_frag.ndim == 2:
                    ref_evol_1d = [sparse_apply_operator(h, ref) for h in ham_frag[i]]
                    ov_list = [state_dot(ref_evol, ksd_state) for ref_evol in ref_evol_1d]
                    norm_re_1d, norm_im_1d = norm_list[i], norm_list[i]
                elif ham_frag.ndim == 3:
                    ref_evol_re_1d = [sparse_apply_operator(h, ref) for h in ham_frag[i][REAL]]
                    ref_evol_im_1d = [sparse_apply_operator(h, ref) for h in ham_frag[i][IMAG]]
                    ov_list = [state_dot(ref_evol_re, ksd_state).real + state_dot(ref_evol_im, ksd_state).imag * 1j
                               for ref_evol_re, ref_evol_im in zip(ref_evol_re_1d, ref_evol_im_1d)]
                    norm_re_1d, norm_im_1d = norm_list[i][REAL], norm_list[i][IMAG]
                else:
                    raise AssertionError
                prob_dist_now = list()
                for j, ov in enumerate(ov_list):
                    shot_j_re, shot_j_im = int(shot_list[H][i][j][REAL]), int(shot_list[H][i][j][IMAG])
                    prob_real_h = hadamard_test_general(ov, imaginary=False, coeff=float(norm_re_1d[j]))
                    prob_imag_h = hadamard_test_general(ov, imaginary=True, coeff=float(norm_im_1d[j]))
                    if shot_j_re > 0:
                        h_arr[i] += prob_real_h.empirical_average(shot_j_re)
                    if shot_j_im > 0:
                        h_arr[i] += prob_imag_h.empirical_average(shot_j_im) * 1j

                    prob_dist_now.append((prob_real_h.pickle(), prob_imag_h.pickle()))

                prob_real_s, prob_imag_s = hadamard_test_qubit_operator(ref, ksd_state, sparse_1=True)
                shot_s_re, shot_s_im = int(shot_list[S][i][REAL]), int(shot_list[S][i][IMAG])
                if shot_s_re > 0:
                    s_arr[i] += prob_real_s.empirical_average(shot_s_re)
                if shot_s_im > 0:
                    s_arr[i] += prob_imag_s.empirical_average(shot_s_im) * 1j

                loaded_prob_dist[i] = {"H": prob_dist_now, "S": (prob_real_s.pickle(), prob_imag_s.pickle())}

                if use_prob_buffer:
                    fname = f'{i}.pkl'
                    with open(os.path.join(sample_buf_dir, fname), 'wb') as f:
                        pickle.dump(prob_dist_now, f)

            elif i in loaded_prob_dist:
                prob_dist_now = loaded_prob_dist[i]["H"]
                for j, (prob_real_h, prob_imag_h) in enumerate(prob_dist_now):
                    prob_real_h, prob_imag_h = ProbDist.unpickle(prob_real_h), ProbDist.unpickle(prob_imag_h)
                    shot_j_re, shot_j_im = int(shot_list[H][i][j][REAL]), int(shot_list[H][i][j][IMAG])
                    if shot_j_re > 0:
                        h_arr[i] += prob_real_h.empirical_average(shot_j_re)
                    if shot_j_im > 0:
                        h_arr[i] += prob_imag_h.empirical_average(shot_j_im) * 1j

                shot_s_re, shot_s_im = int(shot_list[S][i][REAL]), int(shot_list[S][i][IMAG])
                prob_real_s, prob_imag_s = loaded_prob_dist[i]["S"]
                prob_real_s, prob_imag_s = ProbDist.unpickle(prob_real_s), ProbDist.unpickle(prob_imag_s)
                if shot_s_re > 0:
                    s_arr[i] += prob_real_s.empirical_average(shot_s_re)
                if shot_s_im > 0:
                    s_arr[i] += prob_imag_s.empirical_average(shot_s_im) * 1j

            else:
                raise AssertionError

            if i != n_krylov - 1:
                ksd_state = apply_operator(prop, ksd_state)

    elif meas_type == "FH":
        ref1_prepared = np.zeros(ham_frag.shape, dtype=object)
        for idx in product(*[range(s) for s in ham_frag.shape]):
            ham_frag[idx] = prepare_qksd_est_op(ham_frag[idx], n_qubits)
            ref1_prepared[idx] = prepare_qksd_est_state(ref, *(ham_frag[idx][1:]))

        for i in range(1, n_krylov):
            for part in [REAL, IMAG]:
                shots_now = sum([int(shot_list[i][j][part]) for j in range(n_frag)])
                for j in range(n_frag):
                    if ham_frag.ndim == 1:
                        idx_h = j
                    elif ham_frag.ndim == 2:
                        idx_h = (i, j)
                    else:
                        idx_h = (i, part, j)

                    shot_j = int(shot_list[i][j][part])
                    if shot_j > 0:
                        state1, h_frag = ref1_prepared[idx_h], ham_frag[idx_h]
                        loaded_prob_dist = qksd_extended_swap_test(state1, ksd_state, h_frag, imaginary=(part == IMAG),
                                                                   prepared_op=True,
                                                                   prepared_state=(True, False))
                        samp = loaded_prob_dist.empirical_average(shot_j)
                        h_arr[i] += samp["H"] if part == REAL else (1j * samp["H"])
                        inc_s = samp["S"] * shot_j / shots_now
                        s_arr[i] += inc_s if part == REAL else (1j * inc_s)

            if i != n_krylov - 1:
                ksd_state = apply_operator(prop, ksd_state)
    else:
        raise AssertionError

    s_mat = toeplitz_arr_to_mat(s_arr)
    h_mat = toeplitz_arr_to_mat(h_arr)

    return h_mat, s_mat


def sample_qksd_nontoeplitz(ham_frag: npt.NDArray[QubitOperator],
                            prop: Union[spmatrix, np.ndarray],
                            ref: State,
                            n_krylov: int,
                            meas_type: str,
                            shot_list: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
                            ) -> Tuple[np.ndarray, np.ndarray]:
    # For FH
    # shot_list[idx1_n_krylov, idx2][idx_ham_frag][real=0, imag=1]

    # For LCU,
    # shot_list[h=0][idx_n_krylov, idx2][idx_unitary ][real=0, imag=1]
    # shot_list[s=1][idx_n_krylov, idx2][real=0, imag=1]

    n_qubits = get_num_qubits(ref)

    meas_type = meas_type.upper()
    if meas_type not in ["LCU", "FH"]:
        raise ValueError("meas_type must be 'LCU' or 'FH'.")

    basis = list()
    ksd_state = copy.deepcopy(ref)
    for i in range(n_krylov):
        basis.append(ksd_state)
        if i != n_krylov - 1:
            ksd_state = apply_operator(prop, ksd_state)

    ham_frag = np.array(ham_frag)
    if ham_frag.ndim == 1:
        pham = QubitOperator.accumulate(ham_frag)
    elif ham_frag.ndim in [2, 3]:
        first_idx = tuple(0 for _ in range(ham_frag.ndim - 1))
        pham = QubitOperator.accumulate(ham_frag[first_idx])
    else:
        raise ValueError
    n_frag = ham_frag.shape[-1]

    s_mat, h_mat = np.zeros((n_krylov, n_krylov), dtype=complex), np.zeros((n_krylov, n_krylov), dtype=complex)

    if meas_type == "LCU":
        norm_list = np.zeros(ham_frag.shape, dtype=float)
        for idx in product(*[range(s) for s in ham_frag.shape]):
            norm_list[idx] = ham_frag[idx].induced_norm(order=2)
            ham_frag[idx] = ham_frag[idx] / norm_list[idx]

        for i1 in range(n_krylov):
            basis1 = basis[i1]
            ref_evol_list = [apply_operator(h, basis1) for h in ham_frag] if ham_frag.ndim == 1 else None
            for i2 in range(n_krylov):
                if i1 == i2 == 0:
                    s_mat[i1, i2] = 1.0
                    h_mat[i1, i2] = expectation(pham, ref, sparse=True)
                    continue
                elif i1 > i2:
                    s_mat[i1, i2] = s_mat[i2, i1].conjugate()
                    h_mat[i1, i2] = h_mat[i2, i1].conjugate()
                    continue
                elif i1 == i2:
                    s_mat[i1, i2] = 1.0

                basis2 = basis[i2]
                if ham_frag.ndim == 1:
                    ov_list = np.array([state_dot(ref_evol, basis2) for ref_evol in ref_evol_list], dtype=np.complex128)
                    norm_re, norm_im = norm_list, norm_list
                elif ham_frag.ndim == 2:
                    ov_list = np.array([transition_amplitude(h, basis1, basis2) for h in ham_frag[i2 - i1]],
                                       dtype=np.complex128)
                    norm_re, norm_im = norm_list[i2 - i1], norm_list[i2 - i1]
                elif ham_frag.ndim == 3:
                    ov_list = np.array([transition_amplitude(h, basis1, basis2).real for h in ham_frag[i2 - i1][REAL]],
                                       dtype=np.complex128)
                    if i1 != i2:
                        ov_list += \
                            np.array([transition_amplitude(h, basis1, basis2).imag for h in ham_frag[i2 - i1][IMAG]],
                                     dtype=np.complex128) * 1j
                    norm_re, norm_im = norm_list[i2 - i1][REAL], norm_list[i2 - i1][IMAG]
                else:
                    raise AssertionError
                for j, ov in enumerate(ov_list):
                    shot_h_re, shot_h_im = int(shot_list[H][i1, i2][j][REAL]), int(shot_list[H][i1, i2][j][IMAG])
                    if shot_h_re > 0:
                        prob_real_h = hadamard_test_general(ov, imaginary=False, coeff=float(norm_re[j]))
                        h_mat[i1, i2] += prob_real_h.empirical_average(shot_h_re)
                    if i1 != i2 and shot_h_im > 0:
                        prob_imag_h = hadamard_test_general(ov, imaginary=True, coeff=float(norm_im[j]))
                        h_mat[i1, i2] += prob_imag_h.empirical_average(int(shot_list[H][i1, i2][j][IMAG])) * 1j

                if i1 != i2:
                    shot_s_re, shot_s_im = int(shot_list[S][i1, i2][REAL]), int(shot_list[S][i1, i2][IMAG])
                    prob_real_s, prob_imag_s = hadamard_test_qubit_operator(basis1, basis2)
                    if shot_s_re > 0:
                        s_mat[i1, i2] += prob_real_s.empirical_average(shot_s_re)
                    if shot_s_im > 0:
                        s_mat[i1, i2] += prob_imag_s.empirical_average(shot_s_im) * 1j

    elif meas_type == "FH":
        prepared_state1 = False
        for idx in product(*[range(s) for s in ham_frag.shape]):
            ham_frag[idx] = prepare_qksd_est_op(ham_frag[idx], n_qubits)

        for i1 in range(n_krylov):
            basis1 = basis[i1]
            ref1_prepared = None
            if ham_frag.ndim == 1:
                ref1_prepared = [prepare_qksd_est_state(basis1, *(h[1:])) for h in ham_frag]
                prepared_state1 = True
            for i2 in range(n_krylov):
                if i1 == i2 == 0:
                    s_mat[i1, i2] = 1.0
                    h_mat[i1, i2] = expectation(pham, ref, sparse=True)
                    continue
                elif i1 > i2:
                    s_mat[i1, i2] = s_mat[i2, i1].conjugate()
                    h_mat[i1, i2] = h_mat[i2, i1].conjugate()
                    continue
                elif i1 == i2:
                    s_mat[i1, i2] = 1.0

                basis2 = basis[i2]
                if ham_frag.ndim == 2:
                    ref1_prepared = [prepare_qksd_est_state(basis1, *(h[1:])) for h in ham_frag[i2 - i1]]
                    prepared_state1 = True

                for part in [REAL, IMAG]:
                    shots = sum([int(shot_list[i1, i2][j][part]) for j in range(len(shot_list[i1, i2]))])
                    for j in range(n_frag):
                        if ham_frag.ndim == 1:
                            idx_h = j
                        elif ham_frag.ndim == 2:
                            idx_h = (i2 - i1, j)
                        else:
                            idx_h = (i2 - i1, part, j)
                        shot_j = int(shot_list[i1, i2][j][part])
                        if shot_j > 0:
                            state1 = ref1_prepared[j] if prepared_state1 else basis1
                            prop_dist = qksd_extended_swap_test(state1, basis2, ham_frag[idx_h],
                                                                imaginary=bool(part),
                                                                prepared_op=True,
                                                                prepared_state=(prepared_state1, False))
                            samp = prop_dist.empirical_average(shot_j)

                            h_mat[i1, i2] += samp["H"] if part == REAL else samp["H"] * 1j
                            if i1 != i2:
                                inc = samp["S"] * shot_j / shots
                                s_mat[i1, i2] += inc if part == REAL else inc * 1j

    else:
        raise AssertionError

    return h_mat, s_mat


def sample_qksd(ham_frag: npt.NDArray[QubitOperator],
                prop: Union[spmatrix, np.ndarray],
                ref: State,
                n_krylov: int,
                is_toeplitz: bool,
                meas_type: str,
                shot_list: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
                ) -> Tuple[np.ndarray, np.ndarray]:
    if is_toeplitz:
        return sample_qksd_toeplitz(ham_frag, prop, ref, n_krylov, meas_type, shot_list)
    else:
        return sample_qksd_nontoeplitz(ham_frag, prop, ref, n_krylov, meas_type, shot_list)


def qksd_shot_allocation(tot_shots: Union[int, float],
                         ham_frag,
                         n_krylov: int,
                         meas_type: str,
                         is_toeplitz: bool,
                         frag_shot_alloc: Optional[np.ndarray] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # frag_shot_alloc : [n_frag] or [n_frag][2] or [n_krylov][n_frag][2]
    meas_type = meas_type.upper()
    if meas_type not in ["LCU", "FH"]:
        raise ValueError("meas_type must be 'LCU' or 'FH'.")

    ham_frag = np.array(ham_frag)
    n_frag = ham_frag.shape[-1]

    if frag_shot_alloc is not None and frag_shot_alloc.ndim not in [1, 2, 3]:
        raise ValueError
    elif frag_shot_alloc is None:
        default_fsha = np.zeros(ham_frag.shape)
        for idx in product(*[range(s) for s in ham_frag.shape]):
            default_fsha[idx] = ham_frag[idx].induced_norm(order=2)
        frag_shot_alloc = default_fsha
    fndim = frag_shot_alloc.ndim

    if is_toeplitz:
        if fndim == 3:
            shot_h = np.array(frag_shot_alloc)
            shot_h[0, :, :] = 0
        else:
            shot_h = np.zeros((n_krylov, n_frag, 2), dtype=float)
            for i in range(1, n_krylov):
                if fndim == 1:
                    shot_h[i, :, REAL] = frag_shot_alloc
                    shot_h[i, :, IMAG] = frag_shot_alloc
                elif fndim == 2:
                    shot_h[i, :, :] = frag_shot_alloc
                else:
                    raise AssertionError
    else:
        shot_h = np.zeros((n_krylov, n_krylov, n_frag, 2), dtype=float)
        for i1, i2 in product(range(n_krylov), repeat=2):
            if i1 == i2 == 0:
                continue
            elif i1 == i2:
                if fndim == 3:
                    shot_h[i1, i2, :, REAL] = frag_shot_alloc[0, :, REAL]
                elif fndim == 2:
                    shot_h[i1, i2, :, REAL] = frag_shot_alloc[:, REAL]
                else:
                    shot_h[i1, i2, :, REAL] = frag_shot_alloc
            elif i1 < i2:
                if fndim == 3:
                    shot_h[i1, i2, :, :] = frag_shot_alloc[i2 - i1, :, :]
                elif fndim == 2:
                    shot_h[i1, i2, :, :] = frag_shot_alloc
                else:
                    shot_h[i1, i2, :, REAL] = frag_shot_alloc
                    shot_h[i1, i2, :, IMAG] = frag_shot_alloc

    shot_h /= np.sum(shot_h)

    if meas_type == "LCU":
        if is_toeplitz:
            shot_s = np.zeros((n_krylov, 2), dtype=float)
            shot_s[1:, REAL] = 1.0
            shot_s[1:, IMAG] = 1.0
        else:
            shot_s = np.zeros((n_krylov, n_krylov, 2), dtype=float)
            for i1, i2 in product(range(n_krylov), repeat=2):
                if i1 == i2:
                    continue
                elif i1 < i2:
                    shot_s[i1, i2, REAL] = 1.0
                    shot_s[i1, i2, IMAG] = 1.0

        shot_s /= np.sum(shot_s)
        return shot_h * tot_shots / 2, shot_s * tot_shots / 2
    elif meas_type == "FH":
        return shot_h * tot_shots
    else:
        raise AssertionError
