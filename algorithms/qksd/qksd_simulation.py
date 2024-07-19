import copy
import os
import pickle
from itertools import product
from typing import Union, Optional, Tuple, Dict

import numpy as np
import numpy.typing as npt
from openfermion import QubitOperator
from scipy.sparse import spmatrix

from algorithms.qksd.qksd_utils import toeplitz_arr_to_mat
from ofex.linalg.sparse_tools import apply_operator, state_dot, expectation, sparse_apply_operator
from ofex.sampling_simulation.hadamard_test import hadamard_test_general
from ofex.sampling_simulation.qksd_extended_swap_test import qksd_extended_swap_test, prepare_qksd_est_op, \
    prepare_qksd_est_state
from ofex.sampling_simulation.sampling_base import ProbDist, JointProbDist
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


# noinspection PyTypeChecker
def _sample_qksd_toeplitz(ham_frag: npt.NDArray[QubitOperator],
                          prop: Union[spmatrix, np.ndarray],
                          ref: State,
                          n_krylov: int,
                          meas_type: str,
                          shot_list: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                          n_batch: int = 1,
                          sample_buf_dir: Optional[str] = None,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    # For FH
    # shot_list[idx_n_krylov][real=0, imag=1][idx_ham_frag]

    # For LCU,
    # shot_list[h=0][idx_n_krylov][real=0, imag=1][idx_unitary]
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

    # Load prob. dist. for the simulation from the previously generated files.
    loaded_prob_dist = list()
    use_prob_buffer = sample_buf_dir is not None
    if sample_buf_dir is not None:
        if os.path.exists(sample_buf_dir):
            f_list = os.listdir(sample_buf_dir)
            for fname in f_list:
                idx_prob = tuple([int(x) for x in fname.split('.')[0].split('_')])
                loaded_prob_dist.append(idx_prob)
        else:
            os.mkdir(sample_buf_dir)

    # Initial preparation
    ksd_state = copy.deepcopy(ref)
    s_arr, h_arr = np.zeros((n_batch, n_krylov), dtype=complex), np.zeros((n_batch, n_krylov), dtype=complex)

    # Assign the 0th elements without sampling.
    h_arr[:, 0] = expectation(pham, ref, sparse=True)
    s_arr[:, 0] = 1.0
    ksd_state = apply_operator(prop, ksd_state)

    # Find basis which is not involved in the loaded prob. dist.
    if use_prob_buffer:
        if meas_type == "LCU":
            is_all_loaded_h = np.array([True] + [all([(H, i, part, j) in loaded_prob_dist
                                                      for part, j in product([REAL, IMAG], range(n_frag))])
                                                 for i in range(1, n_krylov)], dtype=bool)
            is_all_loaded_s = np.array([True] + [all([(S, i, part) in loaded_prob_dist
                                                      for part in [REAL, IMAG]])
                                                 for i in range(1, n_krylov)], dtype=bool)
            is_all_loaded = np.logical_and(is_all_loaded_h, is_all_loaded_s)
        else:
            is_all_loaded = np.array([True] + [all([(i, part, j) in loaded_prob_dist
                                                    for part, j in product([REAL, IMAG], range(n_frag))])
                                               for i in range(1, n_krylov)], dtype=bool)
        basis_required = np.logical_not(is_all_loaded)
        idxs_basis_req = [idx[0] for idx in np.argwhere(basis_required)]
        if len(idxs_basis_req) > 0:
            max_basis_required = max(idxs_basis_req)
        else:
            max_basis_required = 0
    else:
        max_basis_required = n_krylov - 1
    # Run sampling
    if meas_type == "LCU":
        # Normalize to make it unitary
        norm_list = np.zeros((n_krylov, 2, n_frag), dtype=float)
        for idx in product(*[range(s) for s in ham_frag.shape]):
            nrm = ham_frag[idx].induced_norm(order=2)
            ham_frag[idx] = ham_frag[idx] / nrm
            idx_h_list = list()
            if ham_frag.ndim == 1:
                j = idx[0]
                for i, part in product(range(n_krylov), [REAL, IMAG]):
                    idx_h_list.append((i, part, j))
            elif ham_frag.ndim == 2:
                i, j = idx[0], idx[1]
                idx_h_list = [(i, REAL, j), (i, IMAG, j)]
            elif ham_frag.ndim == 3:
                idx_h_list = [idx]
            else:
                raise AssertionError
            for idx_h in idx_h_list:
                norm_list[idx_h] = nrm

        # Prepare objects for the Hadamard Test simulation.
        # Note that the overlaps are the only information to simulate the Hadamard test.
        ov_h_list = np.zeros((n_krylov, n_frag), dtype=complex)
        ov_h_prepared = np.zeros((n_krylov, n_frag), dtype=bool)
        ov_s_list = np.zeros(n_krylov, dtype=complex)
        ov_s_prepared = np.zeros(n_krylov, dtype=bool)

        ref_evol_1d = None
        if ham_frag.ndim == 1:
            ref_evol_1d = [sparse_apply_operator(h, ref) for h in ham_frag]
        for i in range(1, n_krylov):
            # Overlaps for H matrix
            for j in range(n_frag):
                idx_prob_list = [(H, i, REAL, j), (H, i, IMAG, j)]
                if (not use_prob_buffer) or (not all([idx_prob in loaded_prob_dist for idx_prob in idx_prob_list])):
                    assert i <= max_basis_required
                    if ham_frag.ndim == 1:
                        ov_h_list[i, j] = state_dot(ref_evol_1d[j], ksd_state)
                    elif ham_frag.ndim == 2:
                        ref_evol = sparse_apply_operator(ham_frag[i, j], ref)
                        ov_h_list[i, j] = state_dot(ref_evol, ksd_state)
                    elif ham_frag.ndim == 3:
                        # noinspection PyTypeChecker
                        ref_evol_re = sparse_apply_operator(ham_frag[i][REAL][j], ref)
                        ref_evol_im = sparse_apply_operator(ham_frag[i][IMAG][j], ref)
                        ov_h_list[i, j] = state_dot(ref_evol_re, ksd_state).real + \
                                          state_dot(ref_evol_im, ksd_state).imag * 1j
                    else:
                        raise AssertionError
                    ov_h_prepared[i, j] = True

            # Overlaps for S matrix
            idx_prob_list = [(S, i, REAL), (S, i, IMAG)]
            if (not use_prob_buffer) or (not all([idx_prob in loaded_prob_dist for idx_prob in idx_prob_list])):
                assert i <= max_basis_required
                ov_s_list[i] = state_dot(ref, ksd_state)
                ov_s_prepared[i] = True

            # Evolve the state
            if i < max_basis_required:
                ksd_state = apply_operator(prop, ksd_state)

        # Perform the Sampling simulation
        for i, part in product(range(1, n_krylov), [REAL, IMAG]):
            # Simulate H matrix sampling
            for j in range(n_frag):
                shot_h = int(shot_list[H][i][part][j])
                idx_prob_h = (H, i, part, j)
                if idx_prob_h not in loaded_prob_dist:
                    assert ov_h_prepared[i, j]
                    prob_h = hadamard_test_general(ov_h_list[i, j], imaginary=(part == IMAG),
                                                   coeff=norm_list[i, part, j])
                    if use_prob_buffer:
                        fname = "_".join([str(x) for x in idx_prob_h]) + ".pkl"
                        with open(os.path.join(sample_buf_dir, fname), 'wb') as f:
                            pickle.dump(prob_h.pickle(), f)
                else:
                    fname = "_".join([str(x) for x in idx_prob_h]) + ".pkl"
                    with open(os.path.join(sample_buf_dir, fname), 'rb') as f:
                        prob_h = ProbDist.unpickle(pickle.load(f))
                if shot_h > 0:
                    inc = np.array(prob_h.batched_empirical_average(shot_h, n_batch), dtype=complex)
                    h_arr[:, i] += inc if part == REAL else (inc * 1j)

            # Simulate S matrix sampling
            shot_s = int(shot_list[S][i][part])
            idx_prob_s = (S, i, part)
            if idx_prob_s not in loaded_prob_dist:
                assert ov_s_prepared[i]
                prob_s = hadamard_test_general(ov_s_list[i], imaginary=(part == IMAG))
                if use_prob_buffer:
                    fname = "_".join([str(x) for x in idx_prob_s]) + ".pkl"
                    with open(os.path.join(sample_buf_dir, fname), 'wb') as f:
                        pickle.dump(prob_s.pickle(), f)
            else:
                fname = "_".join([str(x) for x in idx_prob_s]) + ".pkl"
                with open(os.path.join(sample_buf_dir, fname), 'rb') as f:
                    prob_s = ProbDist.unpickle(pickle.load(f))
            if shot_s > 0:
                inc = np.array(prob_s.batched_empirical_average(shot_s, n_batch), dtype=complex)
                s_arr[:, i] += inc if part == REAL else (inc * 1j)

    elif meas_type == "FH":
        # Prepare the fragmentation for extended swap test
        ref1_prepared = np.zeros(ham_frag.shape, dtype=object)
        op_prepared = np.zeros(ham_frag.shape, dtype=bool)
        if use_prob_buffer:
            for idx in product(*[range(s) for s in ham_frag.shape]):
                # For each fragment, find the corresponding indices for probability distribution.
                if ham_frag.ndim == 1:  # all idx_krylov and part for each fragment.
                    idx_prob_list = list()
                    for i in range(1, n_krylov):
                        idx_prob_list += [(i, REAL, idx[0]), (i, IMAG, idx[0])]
                elif ham_frag.ndim == 2:  # Real and imaginary parts.
                    idx_prob_list = [(idx[0], REAL, idx[1]), (idx[0], IMAG, idx[1])]
                elif ham_frag.ndim == 3:  # Individual
                    idx_prob_list = [idx]
                else:
                    raise AssertionError
                # If any operators are not included in the loaded probability distribution, prepare the operator.
                if not all([idx_prob in loaded_prob_dist for idx_prob in idx_prob_list]):
                    assert not op_prepared[idx]
                    ham_frag[idx] = prepare_qksd_est_op(ham_frag[idx], n_qubits)
                    ref1_prepared[idx] = prepare_qksd_est_state(ref, *(ham_frag[idx][1:]))
                    op_prepared[idx] = True
        else:  # If not using the prob buffer
            for idx in product(*[range(s) for s in ham_frag.shape]):
                ham_frag[idx] = prepare_qksd_est_op(ham_frag[idx], n_qubits)
                ref1_prepared[idx] = prepare_qksd_est_state(ref, *(ham_frag[idx][1:]))
                op_prepared[idx] = True

        # Perform the sampling simulation
        for i in range(1, n_krylov):
            for part in [REAL, IMAG]:
                shots_now = sum([int(shot_list[i][part][j]) for j in range(n_frag)])
                for j in range(n_frag):
                    if ham_frag.ndim == 1:
                        idx_h = j
                    elif ham_frag.ndim == 2:
                        idx_h = (i, j)
                    else:
                        idx_h = (i, part, j)
                    idx_prob = (i, part, j)
                    shot = int(shot_list[i][part][j])
                    if shot > 0:
                        if idx_prob not in loaded_prob_dist:
                            assert op_prepared[idx_h]
                            assert i <= max_basis_required
                            state1, h_frag = ref1_prepared[idx_h], ham_frag[idx_h]
                            prob_dist = qksd_extended_swap_test(state1, ksd_state, h_frag, imaginary=(part == IMAG),
                                                                prepared_op=True,
                                                                prepared_state=(True, False))
                            if use_prob_buffer:
                                fname = "_".join([str(x) for x in idx_prob]) + ".pkl"
                                with open(os.path.join(sample_buf_dir, fname), 'wb') as f:
                                    pickle.dump(prob_dist.pickle(), f)
                        else:
                            fname = "_".join([str(x) for x in idx_prob]) + ".pkl"
                            with open(os.path.join(sample_buf_dir, fname), 'rb') as f:
                                prob_dist = JointProbDist.unpickle(pickle.load(f))
                        samp = prob_dist.batched_empirical_average(shot, n_batch)
                        samp_h = np.array([x["H"] for x in samp], dtype=complex)
                        h_arr[:, i] += samp_h if part == REAL else (samp_h * 1j)

                        samp_s = np.array([x["S"] for x in samp], dtype=complex)
                        inc_s = samp_s * shot / shots_now
                        s_arr[:, i] += inc_s if part == REAL else (inc_s * 1j)

            if i != max_basis_required:
                ksd_state = apply_operator(prop, ksd_state)
    else:
        raise AssertionError

    s_mat = toeplitz_arr_to_mat(s_arr)
    h_mat = toeplitz_arr_to_mat(h_arr)

    return h_mat, s_mat


# noinspection PyTypeChecker
def _sample_qksd_nontoeplitz(ham_frag: npt.NDArray[QubitOperator],
                             prop: Union[spmatrix, np.ndarray],
                             ref: State,
                             n_krylov: int,
                             meas_type: str,
                             shot_list: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                             n_batch: int = 1,
                             sample_buf_dir: Optional[str] = None,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    # For FH
    # shot_list[idx1_n_krylov, idx2][real=0, imag=1][idx_ham_frag]

    # For LCU,
    # shot_list[h=0][idx_n_krylov, idx2][real=0, imag=1][idx_unitary ]
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

    # Find Hamiltonian by accumulating the fragments.
    if ham_frag.ndim == 1:
        pham = QubitOperator.accumulate(ham_frag)
    elif ham_frag.ndim in [2, 3]:
        first_idx = tuple(0 for _ in range(ham_frag.ndim - 1))
        pham = QubitOperator.accumulate(ham_frag[first_idx])
    else:
        raise ValueError

    # Load prob. dist. for the simulation from the previously generated files.
    loaded_prob_dist = list()
    use_prob_buffer = sample_buf_dir is not None
    if sample_buf_dir is not None:
        if os.path.exists(sample_buf_dir):
            f_list = os.listdir(sample_buf_dir)
            for fname in f_list:
                idx_prob = tuple([int(x) for x in fname.split('.')[0].split('_')])
                loaded_prob_dist.append(idx_prob)
        else:
            os.mkdir(sample_buf_dir)

    # Find basis which is not involved in the loaded prob. dist.
    if use_prob_buffer:
        basis_required = np.zeros(n_krylov, dtype=bool)
        for i1, i2 in product(range(n_krylov), range(n_krylov)):
            if i1 == i2 == 0 or i1 > i2:
                continue
            if basis_required[i1] and basis_required[i2]:
                continue

            if i1 != i2:
                if meas_type == "LCU":
                    is_loaded = all([(H, i1, i2, part, j) in loaded_prob_dist
                                     for part, j in product([REAL, IMAG], range(n_frag))])  # H loaded
                    is_loaded = is_loaded and all([(S, i2 - i1, part) in loaded_prob_dist
                                                   for part in [REAL, IMAG]])  # S loaded
                else:  # FH
                    is_loaded = all([(i1, i2, part, j) in loaded_prob_dist
                                     for part, j in product([REAL, IMAG], range(n_frag))])
            else:  # i1 == i2
                if meas_type == "LCU":
                    is_loaded = all([(H, i1, i2, part, j) in loaded_prob_dist
                                     for part, j in product([REAL], range(n_frag))])  # H loaded
                else:  # FH
                    is_loaded = all([(i1, i2, part, j) in loaded_prob_dist
                                     for part, j in product([REAL], range(n_frag))])
            basis_required[i1] = basis_required[i1] or not is_loaded
            basis_required[i2] = basis_required[i2] or not is_loaded

    else:  # All the basis are required.
        basis_required = np.ones(n_krylov, dtype=bool)

    idxs_basis_req = [idx[0] for idx in np.argwhere(basis_required)]
    if len(idxs_basis_req) > 0:
        max_basis_required = max(idxs_basis_req)
    else:
        max_basis_required = 0

    # Basis Preparation
    basis: Dict[int, Optional[State]] = {i: None for i in range(n_krylov)}
    ksd_state = copy.deepcopy(ref)
    for i in range(max_basis_required + 1):
        if basis_required[i]:
            basis[i] = ksd_state
        if i < max_basis_required:
            ksd_state = apply_operator(prop, ksd_state)

    h_mat = np.zeros((n_batch, n_krylov, n_krylov), dtype=complex)
    s_arr = np.zeros((n_batch, n_krylov), dtype=complex)
    s_arr[:, 0] = 1.0

    # Run sampling
    if meas_type == "LCU":
        # Normalize to make it unitary
        norm_list = np.zeros((n_krylov, 2, n_frag), dtype=float)
        for idx in product(*[range(s) for s in ham_frag.shape]):
            nrm = ham_frag[idx].induced_norm(order=2)
            ham_frag[idx] = ham_frag[idx] / nrm
            idx_h_list = list()
            if ham_frag.ndim == 1:
                j = idx[0]
                for i, part in product(range(n_krylov), [REAL, IMAG]):
                    idx_h_list.append((i, part, j))
            elif ham_frag.ndim == 2:
                i, j = idx[0], idx[1]
                idx_h_list = [(i, REAL, j), (i, IMAG, j)]
            elif ham_frag.ndim == 3:
                idx_h_list = [idx]
            else:
                raise AssertionError
            for idx_h in idx_h_list:
                norm_list[idx_h] = nrm

        # Prepare objects for the Hadamard Test simulation.
        # Note that the overlaps are the only information to simulate the Hadamard test.
        ov_h_list = np.zeros((n_krylov, n_krylov, n_frag), dtype=complex)
        ov_h_prepared = np.zeros((n_krylov, n_krylov, n_frag), dtype=bool)
        ov_s_list = np.zeros(n_krylov, dtype=complex)
        ov_s_prepared = np.zeros(n_krylov, dtype=bool)

        for i1 in range(n_krylov):
            basis1 = basis[i1]
            if basis1 is None:
                continue
            # Overlaps for H matrix
            ref_evol_list = [apply_operator(h, basis1) for h in ham_frag] if ham_frag.ndim == 1 else None
            for i2 in range(n_krylov):
                if i1 == i2 == 0 or i1 > i2:
                    continue
                basis2 = basis[i2]
                if basis2 is None:
                    continue
                for j in range(n_frag):
                    idx_prob_list = [(H, i1, i2, REAL, j), (H, i1, i2, IMAG, j)]
                    if (not use_prob_buffer) or (not all([idx_prob in loaded_prob_dist for idx_prob in idx_prob_list])):
                        if ham_frag.ndim == 1:
                            ov_h_list[i1, i2, j] = state_dot(ref_evol_list[j], basis2)
                        elif ham_frag.ndim == 2:
                            ref_evol = sparse_apply_operator(ham_frag[i2 - i1, j], basis1)
                            ov_h_list[i1, i2, j] = state_dot(ref_evol, basis2)
                        elif ham_frag.ndim == 3:
                            ref_evol_re = sparse_apply_operator(ham_frag[i2 - i1][REAL][j], basis1)
                            ref_evol_im = sparse_apply_operator(ham_frag[i2 - i1][IMAG][j], basis1)
                            ov_h_list[i1, i2, j] = state_dot(ref_evol_re, basis2).real + \
                                                   state_dot(ref_evol_im, basis2).imag * 1j
                        else:
                            raise AssertionError
                    ov_h_prepared[i1, i2, j] = True

            # Overlaps for S matrix
            idx_prob_list = [(S, i1, REAL), (S, i1, IMAG)]
            if (not use_prob_buffer) or (not all([idx_prob in loaded_prob_dist for idx_prob in idx_prob_list])):
                ov_s_list[i1] = state_dot(ref, basis1)
                ov_s_prepared[i1] = True

        # Perform the Sampling simulation
        for i1 in range(n_krylov):
            # Simulate H matrix sampling
            for i2 in range(n_krylov):
                if i1 == i2 == 0:
                    h_mat[:, i1, i2] = expectation(pham, ref, sparse=True)
                    continue
                elif i1 > i2:
                    h_mat[:, i1, i2] = h_mat[:, i2, i1].conjugate()
                    continue

                for (j, ov), part in product(enumerate(ov_h_list[i1, i2, :]), [REAL, IMAG]):
                    shot_h = int(shot_list[H][i1, i2][part][j])
                    idx_prob_h = (H, i1, i2, part, j)
                    if idx_prob_h not in loaded_prob_dist:
                        assert ov_h_prepared[i1, i2, j]
                        prob_h = hadamard_test_general(ov, imaginary=(part == IMAG), coeff=norm_list[i2 - i1, part, j])
                        if use_prob_buffer:
                            fname = "_".join([str(x) for x in idx_prob_h]) + ".pkl"
                            with open(os.path.join(sample_buf_dir, fname), 'wb') as f:
                                pickle.dump(prob_h.pickle(), f)
                    else:
                        fname = "_".join([str(x) for x in idx_prob_h]) + ".pkl"
                        with open(os.path.join(sample_buf_dir, fname), 'rb') as f:
                            prob_h = ProbDist.unpickle(pickle.load(f))
                    if shot_h > 0:
                        inc = np.array(prob_h.batched_empirical_average(shot_h, n_batch), dtype=complex)
                        h_mat[:, i1, i2] += inc if part == REAL else (inc * 1j)

            # Simulate S matrix sampling
            if i1 == 0:
                continue
            for part in [REAL, IMAG]:
                shot_s = int(shot_list[S][i1][part])
                idx_prob_s = (S, i1, part)
                if idx_prob_s not in loaded_prob_dist:
                    assert ov_s_prepared[i1]
                    prob_s = hadamard_test_general(ov_s_list[i1], imaginary=(part == IMAG))
                    if use_prob_buffer:
                        fname = "_".join([str(x) for x in idx_prob_s]) + ".pkl"
                        with open(os.path.join(sample_buf_dir, fname), 'wb') as f:
                            pickle.dump(prob_s.pickle(), f)
                else:
                    fname = "_".join([str(x) for x in idx_prob_s]) + ".pkl"
                    with open(os.path.join(sample_buf_dir, fname), 'rb') as f:
                        prob_s = ProbDist.unpickle(pickle.load(f))
                if shot_s > 0:
                    inc = np.array(prob_s.batched_empirical_average(shot_s, n_batch))
                    s_arr[:, i1] += inc if part == REAL else (inc * 1j)

    elif meas_type == "FH":
        # Prepare the fragmentation for extended swap test
        # idx_prob = (i1, i2, part, j)
        op_prepared = np.zeros(ham_frag.shape, dtype=bool)
        if use_prob_buffer:
            for idx in product(*[range(s) for s in ham_frag.shape]):
                # For each fragment, find the corresponding indices for probability distribution.
                if ham_frag.ndim == 1 and not any(basis_required):
                    continue
                elif ham_frag.ndim == 2:
                    i2, j = idx
                    if all([(i1, i1 + i2, part, j) in loaded_prob_dist
                            for i1, part in product(range(n_krylov - i2), [REAL, IMAG])]):
                        continue
                elif ham_frag.ndim == 3:
                    i2, j, part = idx
                    if all([(i1, i1 + i2, part, j) in loaded_prob_dist for i1 in range(n_krylov - i2)]):
                        continue
                assert not op_prepared[idx]
                ham_frag[idx] = prepare_qksd_est_op(ham_frag[idx], n_qubits)
                op_prepared[idx] = True
        else:  # If not using the prob buffer
            for idx in product(*[range(s) for s in ham_frag.shape]):
                ham_frag[idx] = prepare_qksd_est_op(ham_frag[idx], n_qubits)
                op_prepared[idx] = True

        # Perform the sampling simulation
        shot_s = np.zeros((n_krylov, 2), dtype=int)
        shot_s[0, REAL] = 1
        shot_s[0, IMAG] = 1
        for i1 in range(n_krylov):
            basis1 = basis[i1]
            ref1_prepared, prepared_state1 = None, False
            if ham_frag.ndim == 1 and basis1 is not None:
                assert np.all(op_prepared)
                ref1_prepared = [prepare_qksd_est_state(basis1, *(h[1:])) for h in ham_frag]
                prepared_state1 = True

            for i2 in range(n_krylov):
                if i1 == i2 == 0:
                    h_mat[:, i1, i2] = expectation(pham, ref, sparse=True)
                    continue
                elif i1 > i2:
                    h_mat[:, i1, i2] = h_mat[:, i2, i1].conjugate()
                    continue

                basis2 = basis[i2]
                if ham_frag.ndim == 2 and basis1 is not None and basis2 is not None:
                    ref1_prepared = [prepare_qksd_est_state(basis1, *(h[1:])) for h in ham_frag[i2 - i1]]
                    prepared_state1 = True

                for part in [REAL, IMAG]:
                    # shots = sum([int(shot_list[i1, i2][j][part]) for j in range(len(shot_list[i1, i2]))])
                    for j in range(n_frag):
                        idx_prob = (i1, i2, part, j)
                        if idx_prob not in loaded_prob_dist:
                            if ham_frag.ndim == 1:
                                idx_h = j
                            elif ham_frag.ndim == 2:
                                idx_h = (i2 - i1, j)
                            else:
                                idx_h = (i2 - i1, part, j)
                            state1 = ref1_prepared[j] if prepared_state1 else basis1
                            prob_dist = qksd_extended_swap_test(state1, basis2, ham_frag[idx_h],
                                                                imaginary=bool(part),
                                                                prepared_op=True,
                                                                prepared_state=(prepared_state1, False))
                            if use_prob_buffer:
                                fname = "_".join([str(x) for x in idx_prob]) + ".pkl"
                                with open(os.path.join(sample_buf_dir, fname), 'wb') as f:
                                    pickle.dump(prob_dist.pickle(), f)
                        else:
                            fname = "_".join([str(x) for x in idx_prob]) + ".pkl"
                            with open(os.path.join(sample_buf_dir, fname), 'rb') as f:
                                prob_dist = JointProbDist.unpickle(pickle.load(f))

                        shot_j = int(shot_list[i1, i2][part][j])
                        if shot_j > 0:
                            samp = prob_dist.batched_empirical_average(shot_j, n_batch)
                            samp_h = np.array([x["H"] for x in samp], dtype=complex)
                            h_mat[:, i1, i2] += samp_h if part == REAL else (samp_h * 1j)
                            if i1 != i2:
                                i_diff = i2 - i1
                                samp_s = np.array([x["S"] for x in samp], dtype=complex) * shot_j
                                s_arr[:, i_diff] += samp_s if part == REAL else (samp_s * 1j)
                                shot_s[i_diff, part] += shot_j

        s_arr = s_arr.real / shot_s[:, REAL] + s_arr.imag * 1j / shot_s[:, IMAG]

    else:
        raise AssertionError

    s_mat = toeplitz_arr_to_mat(s_arr)

    return h_mat, s_mat


def sample_qksd(ham_frag: npt.NDArray[QubitOperator],
                prop: Union[spmatrix, np.ndarray],
                ref: State,
                n_krylov: int,
                is_toeplitz: bool,
                meas_type: str,
                shot_list: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                n_batch: Optional[int] = None,
                sample_buf_dir: Optional[str] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs circuit simulation to generate QKSD matrices.

    Args:
        ham_frag: Array of Hamiltonian fragmentation (meas_type == "FH") or LCU (meas_type == "LCU")
                  The dimension of the array should be one of the follows:
                    1D - ham_frag[idx_frag]
                    2D - ham_frag[idx_krylov][idx_frag]
                    3D - ham_frag[idx_krylov][REAL/IMAG][idx_frag]

                  idx_frag: Index of FH or LCU (j: H=ΣH_j or H=Σj β_jU_j)
                  idx_krylov: Index of Krylov basis. (k: <φ|H e^{-iH k Δt}|φ>)
                  REAL/IMAG: Whether to measure Re[<φ|H e^{-iH k Δt}|φ>] or Im[<φ|H e^{-iH k Δt}|φ>].

                  Note: The fragmentation may differ by krylov index / Re/Im measurements.
        prop: Propagator (e^{-iH k Δt}) or its Trotterized operator
        ref: Reference State, such as Hartree Fock ground state.
        n_krylov: Krylov order, i.e. size of Krylov matrices.
        is_toeplitz: Whether to perform Toeplitz measurement for H matrix.

                     Toeplitz: H_{k,l} = <φ|H e^{-iH (l-k) Δt}|φ>  -> O(n_krylov) elements
                     NonToeplitz: H_{k,l} = <φ|e^{iH k Δt} H e^{-iH l Δt}|φ>  -> O(n_krylov^2) elements

                     Note : If the exact propagator is assigned in prop, the expectation of H_{k,l} are identical,
                            while the variance is larger in Non-Toeplitz case because the shots are distributed to more
                            elements. However, the expectations are different when the Trotterized operator is asigned
                            to prop because prop no longer commutes with H.
        meas_type: "FH" or "LCU", depending on the type of ham_frag.
        shot_list: Shot allocation.
                   For FH, S and H matrices are measured simultaneously. Shots are not distinguished by two matrices.
                        shot_list[idx_krylov][real=0, imag=1][idx_ham_frag]

                   For LCU, S and H are measured individually.
                        shot_list[H=0][idx_krylov][REAL/IMAG][idx_unitary]
                        shot_list[S=1][idx_krylov][REAL/IMAG]
        n_batch: Number of batch.
                 The function will generate n_batch pairs of (H, S) matrices, which are sampled independently.
                 If None, only single matrix pair is generated.
        sample_buf_dir: Directory to the buffer for sampling distribution.
                        If specified, the probability distributions of the circuit simulation are saved as files.
                        If those files already exist, sampling can be done quickly by loading the results which are
                        previously calculated.
                        If None, such operation doesn't occur.

    Returns:
        (List of) H and S matrices.
    """
    if n_batch is None:
        was_n_batch_none = True
        n_batch = 1
    else:
        was_n_batch_none = False

    if is_toeplitz:
        hmat, smat = _sample_qksd_toeplitz(ham_frag, prop, ref, n_krylov, meas_type, shot_list, n_batch, sample_buf_dir)
    else:
        hmat, smat = _sample_qksd_nontoeplitz(ham_frag, prop, ref, n_krylov, meas_type, shot_list, n_batch,
                                              sample_buf_dir)

    if was_n_batch_none:
        return hmat[0], smat[0]
    else:
        return hmat, smat


def qksd_shot_allocation(tot_shots: Union[int, float],
                         ham_frag,
                         n_krylov: int,
                         meas_type: str,
                         is_toeplitz: bool,
                         frag_shot_alloc: Optional[np.ndarray] = None) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Allocates the shots to each element and hamiltonian fragment.

    Args:
        tot_shots: Number of total shots. Those will be allocated to each element.
        ham_frag: Hamiltonian fragments (Refer to the input in sample_qksd).
        n_krylov: Krylov order, i.e. size of Krylov matrices.
        meas_type: "FH" or "LCU", depending on the type of ham_frag.
        is_toeplitz: Whether to perform Toeplitz measurement for H matrix. (Refer to the input in sample_qksd).
        frag_shot_alloc: Base shot allocation to each element.
                        1D - frag_shot_alloc[idx_frag]
                        2D - frag_shot_alloc[REAL/IMAG][idx_frag]
                        3D - frag_shot_alloc[idx_krylov][REAL/IMAG][idx_frag]
                        If None, 1D array is assigned based on the fragment norm.
    Returns:
        shot_allocation: Refer to the input in sample_qksd.

    """
    # frag_shot_alloc : [n_frag] or [2][n_frag] or [n_krylov][2][n_frag]
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
            shot_h = np.zeros((n_krylov, 2, n_frag), dtype=float)
            for i in range(1, n_krylov):
                if fndim == 1:
                    shot_h[i, REAL, :] = frag_shot_alloc
                    shot_h[i, IMAG, :] = frag_shot_alloc
                elif fndim == 2:
                    shot_h[i, :, :] = frag_shot_alloc
                else:
                    raise AssertionError
    else:
        shot_h = np.zeros((n_krylov, n_krylov, 2, n_frag), dtype=float)
        for i1, i2 in product(range(n_krylov), repeat=2):
            if i1 == i2 == 0:
                continue
            elif i1 == i2:
                if fndim == 3:
                    shot_h[i1, i2, REAL, :] = frag_shot_alloc[0, REAL, :]
                elif fndim == 2:
                    shot_h[i1, i2, REAL, :] = frag_shot_alloc[REAL, :]
                else:
                    shot_h[i1, i2, REAL, :] = frag_shot_alloc
            elif i1 < i2:
                if fndim == 3:
                    shot_h[i1, i2, :, :] = frag_shot_alloc[i2 - i1, :, :]
                elif fndim == 2:
                    shot_h[i1, i2, :, :] = frag_shot_alloc
                else:
                    shot_h[i1, i2, REAL, :] = frag_shot_alloc
                    shot_h[i1, i2, IMAG, :] = frag_shot_alloc

    shot_h /= np.sum(shot_h)

    if meas_type == "LCU":
        # S matrix is toeplitz regardless the measurement.
        shot_s = np.ones((n_krylov, 2), dtype=float)
        shot_s[0, :] = 0.0
        shot_s /= np.sum(shot_s)
        return shot_h * tot_shots / 2, shot_s * tot_shots / 2
    elif meas_type == "FH":
        return shot_h * tot_shots
    else:
        raise AssertionError
