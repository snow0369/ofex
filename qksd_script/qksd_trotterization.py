import copy
from itertools import product
from typing import Union, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from openfermion import QubitOperator
from scipy.sparse import spmatrix

from ofex.linalg.sparse_tools import apply_operator, state_dot, expectation
from ofex.sampling_simulation.hadamard_test import hadamard_test_qubit_operator
from ofex.sampling_simulation.qksd_extended_swap_test import qksd_extended_swap_test
from ofex.state.types import State
from qksd_script.qksd_utils import toeplitz_arr_to_mat

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
                         ) -> Tuple[np.ndarray, np.ndarray]:
    # For FH
    # shot_list[idx_n_krylov][idx_ham_frag][real=0, imag=1]

    # For LCU,
    # shot_list[h=0][idx_n_krylov][idx_unitary ][real=0, imag=1]
    # shot_list[s=1][idx_n_krylov][real=0, imag=1]

    meas_type = meas_type.upper()
    if meas_type not in ["LCU", "FH"]:
        raise ValueError("meas_type must be 'LCU' or 'FH'.")

    ksd_state = copy.deepcopy(ref)
    s_arr, h_arr = np.zeros(n_krylov, dtype=complex), np.zeros(n_krylov, dtype=complex)

    ham_frag = np.array(ham_frag)
    if ham_frag.ndim == 1:
        pham = QubitOperator.accumulate(ham_frag)
    else:
        first_idx = (0 for _ in range(ham_frag.ndim - 1))
        pham = QubitOperator.accumulate(ham_frag[first_idx])

    n_frag = ham_frag.shape[-1]

    h_arr[0] = expectation(pham, ref, sparse=True)
    s_arr[0] = 1.0
    ksd_state = apply_operator(prop, ksd_state)

    if meas_type == "LCU":
        norm_list = np.zeros(ham_frag.shape, dtype=float)
        for idx in product(*[range(s) for s in ham_frag.shape]):
            norm_list[idx] = ham_frag[idx].induced_norm(order=2)
            ham_frag[idx] /= norm_list[idx]
        for i in range(1, n_krylov):
            for j, h_frag in enumerate(ham_frag):
                prob_real_h, prob_imag_h = hadamard_test_qubit_operator(ref, ksd_state, h_frag, norm_list[j],
                                                                        sparse_1=True)
                h_arr[i] += prob_real_h.empirical_average(int(shot_list[H][i][j][REAL]))
                h_arr[i] += prob_imag_h.empirical_average(int(shot_list[H][i][j][IMAG])) * 1j
            prob_real_s, prob_imag_s = hadamard_test_qubit_operator(ref, ksd_state,
                                                                    sparse_1=True)
            s_arr[i] += prob_real_s.empirical_average(int(shot_list[S][i][REAL]))
            s_arr[i] += prob_imag_s.empirical_average(int(shot_list[S][i][IMAG])) * 1j

            if i != n_krylov - 1:
                ksd_state = apply_operator(prop, ksd_state)

    elif meas_type == "FH":



        for i in range(1, n_krylov):
            shots_re = sum([int(shot_list[i][j][REAL]) for j in range(n_frag)])
            shots_im = sum([int(shot_list[i][j][IMAG]) for j in range(n_frag)])
            for j, h_frag in enumerate(ham_frag):
                prob_real, prob_imag = qksd_extended_swap_test(ref, ksd_state, h_frag)
                re_samp = prob_real.empirical_average(int(shot_list[i][j][REAL]))
                im_samp = prob_imag.empirical_average(int(shot_list[i][j][IMAG]))
                h_arr[i] += re_samp["H"] + 1j * im_samp["H"]
                s_arr[i] += ((re_samp["S"] * int(shot_list[i][j][REAL])) / shots_re
                             + 1j * im_samp["S"] * int(shot_list[i][j][IMAG]) / shots_im)

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

    meas_type = meas_type.upper()
    if meas_type not in ["LCU", "FH"]:
        raise ValueError("meas_type must be 'LCU' or 'FH'.")

    basis = list()
    ksd_state = copy.deepcopy(ref)
    for i in range(n_krylov):
        basis.append(ksd_state)
        if i != n_krylov - 1:
            ksd_state = apply_operator(prop, ksd_state)

    pham = QubitOperator.accumulate(ham_frag)

    if meas_type == "LCU":
        norm_list = [h.induced_norm(order=2) for h in ham_frag]
        ham_frag = [h / n for h, n in zip(ham_frag, norm_list)]
    else:
        norm_list = None

    s_mat, h_mat = np.zeros((n_krylov, n_krylov), dtype=complex), np.zeros((n_krylov, n_krylov), dtype=complex)

    for i1, i2 in product(range(n_krylov), repeat=2):
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

        basis1, basis2 = basis[i1], basis[i2]

        if meas_type == "FH":
            shots_re = sum([int(shot_list[i1, i2][j][REAL]) for j in range(len(ham_frag))])
            shots_im = sum([int(shot_list[i1, i2][j][IMAG]) for j in range(len(ham_frag))])
            for j, h_frag in enumerate(ham_frag):
                prop_real, prob_imag = qksd_extended_swap_test(basis1, basis2, h_frag)
                re_samp = prop_real.empirical_average(int(shot_list[i1, i2][j][REAL]))
                im_samp = prob_imag.empirical_average(int(shot_list[i1, i2][j][IMAG]))
                h_mat[i1, i2] += re_samp["H"] + 1j * im_samp["H"]
                if i1 != i2:
                    s_mat[i1, i2] += ((re_samp["S"] * int(shot_list[i1, i2][j][REAL])) / shots_re
                                      + 1j * im_samp["S"] * int(shot_list[i1, i2][j][IMAG]) / shots_im)
        elif meas_type == "LCU":
            for j, h_frag in enumerate(ham_frag):
                prob_real_h, prob_imag_h = hadamard_test_qubit_operator(basis1, basis2, h_frag, norm_list[j])
                h_mat[i1, i2] += prob_real_h.empirical_average(int(shot_list[H][i1, i2][j][REAL]))
                if i1 != i2:
                    h_mat[i1, i2] += prob_imag_h.empirical_average(int(shot_list[H][i1, i2][j][IMAG])) * 1j
            if i1 != i2:
                prob_real_s, prob_imag_s = hadamard_test_qubit_operator(basis1, basis2)
                s_mat[i1, i2] += prob_real_s.empirical_average(int(shot_list[S][i1, i2][REAL]))
                s_mat[i1, i2] += prob_imag_s.empirical_average(int(shot_list[S][i1, i2][IMAG])) * 1j
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


def _example_prepare():
    import inspect

    from ofex.utils.chem import molecule_example, run_driver
    from openfermion import get_fermion_operator
    from ofex.transforms.fermion_qubit import fermion_to_qubit_operator
    from ofex.state.chem_ref_state import hf_ground
    from ofex.operators.qubit_operator_tools import normalize_by_lcu_norm
    from ofex.propagator.exact import exact_rte
    from ofex.propagator.trotter import trotter_rte_by_si_lcu
    from ofex.measurement.sorted_insertion import sorted_insertion

    def retrieve_name(var):
        callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

    mol_name = "H4"
    transform = "bravyi_kitaev"
    n_krylov = 10
    time_step = 3 * np.pi / n_krylov
    n_trotter = 2

    mol = molecule_example(mol_name)
    mol = run_driver(mol, run_fci=True, driver="PSI4")
    fham = mol.get_molecular_hamiltonian()
    fham = get_fermion_operator(fham)

    # keyword arguments required in ofex.transforms.fermion_to_qubit_operator
    f2q_kwargs = {"n_qubits": mol.n_qubits}

    pham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
    p_const = pham.constant
    pham = pham - p_const
    assert np.isclose(p_const.imag, 0.0)
    p_const = p_const.real
    pham_prop, norm = normalize_by_lcu_norm(pham, level=1)
    # pham_prop = pham

    ref = hf_ground(mol, fermion_to_qubit_map=transform, **f2q_kwargs)

    prop_rte = exact_rte(pham_prop, time_step)
    prop_trot = trotter_rte_by_si_lcu(pham_prop, time_step, mol.n_qubits,
                                      n_trotter=n_trotter)

    ham_frag = sorted_insertion(pham, anticommute=False)
    lcu_frag = sorted_insertion(pham, anticommute=True)

    return {retrieve_name(v): v for v in [pham, prop_rte, prop_trot, ref, n_krylov,
                                          ham_frag, lcu_frag,
                                          mol, p_const]}


def _example_script_trotter(pham, prop_rte, prop_trot, ref, n_krylov, mol, p_const, **_):
    from qksd_utils import trunc_eigh

    ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_krylov)
    toep_h, toep_s = ideal_qksd_toeplitz(pham, prop_trot, ref, n_krylov)
    nontoep_h, nontoep_s = ideal_qksd_nontoeplitz(pham, prop_trot, ref, n_krylov)

    val, vec = trunc_eigh(ideal_h, ideal_s, epsilon=1e-14)
    ideal_gnd = np.min(val)
    print("=== Trotter Perturbation Analysis ===")
    print("")
    print(f"\tIdealQKSD - FCI = {ideal_gnd + p_const - mol.fci_energy}")

    toep_h_pert = np.linalg.norm(ideal_h - toep_h, ord=2)
    toep_s_pert = np.linalg.norm(ideal_s - toep_s, ord=2)
    val, vec = trunc_eigh(toep_h, toep_s, epsilon=toep_s_pert)
    print("")
    print(f"\tTrotter Toeplitz ‖ΔH‖ = {toep_h_pert}")
    print(f"\tTrotter Toeplitz ‖ΔS‖ = {toep_s_pert}")
    print(f"\tEigenvalue Perturbation = {np.min(val) - ideal_gnd}")

    ntoep_h_pert = np.linalg.norm(ideal_h - nontoep_h, ord=2)
    ntoep_s_pert = np.linalg.norm(ideal_s - nontoep_s, ord=2)
    val, vec = trunc_eigh(nontoep_h, nontoep_s, epsilon=ntoep_s_pert)
    print("")
    print(f"\tTrotter NonToeplitz ‖ΔH‖ = {ntoep_h_pert}")
    print(f"\tTrotter NonToeplitz ‖ΔS‖ = {ntoep_s_pert}")
    print(f"\tEigenvalue Perturbation = {np.min(val) - ideal_gnd}")

    print("")


def _example_script_sample(pham, prop_rte, ref, n_krylov, ham_frag, lcu_frag, **_):
    from qksd_utils import trunc_eigh
    from time import time

    tot_shots = 1e8

    ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_krylov)
    val, vec = trunc_eigh(ideal_h, ideal_s, epsilon=1e-14)
    ideal_gnd = np.min(val)

    print("=== Sampling Perturbation Analysis ===")
    print("")

    for is_toeplitz, meas_type in product([True, False], ["FH", "LCU"]):
        frag = ham_frag if meas_type == "FH" else lcu_frag

        t = time()
        shot_alloc = qksd_shot_allocation(tot_shots, frag, n_krylov, meas_type, is_toeplitz)
        checksum_shot = np.sum(shot_alloc) if isinstance(shot_alloc, np.ndarray) \
            else (np.sum(shot_alloc[0]) + np.sum(shot_alloc[1]))
        assert np.isclose(checksum_shot, tot_shots)
        samp_h, samp_s = sample_qksd(frag, prop_rte, ref, n_krylov, is_toeplitz, meas_type, shot_alloc)
        t = time() - t

        pert_h = np.linalg.norm(samp_h - ideal_h, ord=2)
        pert_s = np.linalg.norm(samp_s - ideal_s, ord=2)
        val, vec = trunc_eigh(samp_h, samp_s, epsilon=pert_s)

        print(f"\t{meas_type}, Toeplitz = {is_toeplitz} ({t} sec)")
        print(f"\t\t‖ΔH‖ = {pert_h}")
        print(f"\t\t‖ΔS‖ = {pert_s}")
        print(f"\t\t ΔE  = {np.min(val) - ideal_gnd}")
        print("")


def _profile_script_sample(prop_rte, ref, n_krylov, ham_frag, lcu_frag, **_):
    from cProfile import Profile
    from pstats import Stats

    meas_type = "FH"
    is_toeplitz = False
    tot_shots = 1e8

    frag = ham_frag if meas_type == "FH" else lcu_frag
    shot_alloc = qksd_shot_allocation(tot_shots, frag, n_krylov, meas_type, is_toeplitz)

    def _test():
        return sample_qksd(frag, prop_rte, ref, n_krylov, is_toeplitz, meas_type, shot_alloc)

    profiler = Profile()
    profiler.runcall(_test)

    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats()


if __name__ == '__main__':
    _kwargs = _example_prepare()
    # _example_script_trotter(**_kwargs)
    _example_script_sample(**_kwargs)
    # _profile_script_sample(**_kwargs)
