import inspect
import os
import pickle
from cProfile import Profile
from itertools import product
from pstats import Stats
from time import time

import numpy as np
from openfermion import get_fermion_operator, FermionOperator, QubitOperator

from algorithms.qksd.qksd_simulation import ideal_qksd_toeplitz, ideal_qksd_nontoeplitz, qksd_shot_allocation, \
    sample_qksd
from algorithms.qksd.qksd_utils import trunc_eigh
from ofex.measurement.iterative_coefficient_splitting import init_ics, run_ics
from ofex.measurement.killer_shift import killer_shift_opt_fermion_hf
from ofex.measurement.sorted_insertion import sorted_insertion
from ofex.operators.qubit_operator_tools import normalize_by_lcu_norm
from ofex.propagator.exact import exact_rte
from ofex.propagator.trotter import trotter_rte_by_si_lcu
from ofex.state.chem_ref_state import hf_ground, cisd_ground
from ofex.state.state_tools import get_num_qubits
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator, fermion_to_qubit_state
from ofex.utils.chem import molecule_example, run_driver


def _prepare():
    def retrieve_name(var):
        callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
        return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

    mol_name = "H2O"
    transform = "symmetry_conserving_bravyi_kitaev"
    n_krylov = 10
    time_step = 3 * np.pi / n_krylov
    n_trotter = 2

    mol = molecule_example(mol_name)
    mol.load()
    mol = run_driver(mol, run_cisd=True, run_fci=True, driver="pyscf")

    # keyword arguments required in ofex.transforms.fermion_to_qubit_operator
    if transform == "bravyi_kitaev":
        f2q_kwargs = {"n_qubits": mol.n_qubits}
    elif transform == "symmetry_conserving_bravyi_kitaev":
        f2q_kwargs = {"active_fermions": mol.n_electrons,
                      "active_orbitals": mol.n_qubits}
    elif transform == "jordan_wigner":
        f2q_kwargs = dict()
    else:
        raise NotImplementedError

    ref = hf_ground(mol, fermion_to_qubit_map=transform, **f2q_kwargs)
    f_ref = hf_ground(mol, fermion_to_qubit_map=None)

    # PySCF results may be different for every run. Thus, the objects need to be pickled.
    fname = f"./tmp_{mol_name}_{transform}.pkl"
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            fham_t, f_const, pham_t, p_const, cisd_state, shift_op_pikl = pickle.load(f)
        fham, pham = FermionOperator(), QubitOperator()
        fham.terms = fham_t
        pham.terms = pham_t

        shift_op = dict()
        for oplvl, (fh_t, ph_t, c) in shift_op_pikl.items():
            fh, ph = FermionOperator(), QubitOperator()
            fh.terms = fh_t
            ph.terms = ph_t
            shift_op[oplvl] = (fh, ph, c)

    else:
        cisd_state = cisd_ground(mol)
        cisd_state = fermion_to_qubit_state(cisd_state, transform, **f2q_kwargs)

        fham = mol.get_molecular_hamiltonian()
        fham = get_fermion_operator(fham)
        f_const = fham.constant
        fham = fham - f_const
        pham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
        p_const = pham.constant
        assert np.isclose(p_const.imag, 0.0)
        p_const = p_const.real
        pham = pham - p_const

        shift_op = dict()
        repeat_opt = 10
        hf_vector = list(f_ref.keys())[0]
        for optimization_level in [0, 1, 2]:
            shift_fham, shift_pham, shift_const = killer_shift_opt_fermion_hf(fham, hf_vector, transform,
                                                                              optimization_level,
                                                                              repeat_opt,
                                                                              f2q_kwargs)
            shift_op[optimization_level] = (shift_fham, shift_pham, shift_const)

        shift_op_pikl = {oplvl: (fh.terms, ph.terms, c) for oplvl, (fh, ph, c) in shift_op.items()}
        with open(fname, 'wb') as f:
            pickle.dump((fham.terms, f_const, pham.terms, p_const, cisd_state, shift_op_pikl), f)

    pham_prop, norm = normalize_by_lcu_norm(pham, level=1)
    # pham_prop = pham

    n_qubits = get_num_qubits(ref)
    prop_rte = exact_rte(pham_prop, time_step)
    prop_trot = None  # trotter_rte_by_si_lcu(pham_prop, time_step, n_qubits, n_trotter=n_trotter)

    ham_frag = sorted_insertion(pham, anticommute=False)
    lcu_frag = sorted_insertion(pham, anticommute=True)

    return {retrieve_name(v): v for v in [fham, pham, f_const, p_const,
                                          prop_rte, prop_trot,
                                          ref, f_ref, cisd_state,
                                          n_krylov, time_step, norm,
                                          ham_frag, lcu_frag,
                                          shift_op,
                                          mol, mol_name, transform, f2q_kwargs,]}


def _trotter_perturbation(pham, prop_rte, prop_trot, ref, n_krylov, mol, p_const, f_const, **_):
    ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_krylov)
    toep_h, toep_s = ideal_qksd_toeplitz(pham, prop_trot, ref, n_krylov)
    nontoep_h, nontoep_s = ideal_qksd_nontoeplitz(pham, prop_trot, ref, n_krylov)

    val, vec = trunc_eigh(ideal_h, ideal_s, epsilon=1e-14)
    ideal_gnd = np.min(val)
    print("=== Trotter Perturbation Analysis ===")
    print("")
    print(f"\tIdealQKSD - FCI = {ideal_gnd + p_const + f_const - mol.fci_energy}")

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


def _sample_perturbation(pham, prop_rte, ref, n_krylov, ham_frag, lcu_frag, mol_name, transform, **_):
    tot_shots = 1e8
    n_batch = 100
    save_buffer = True
    load_buffer = True

    prob_buf_dir = f'./tmp_prob_buf_{mol_name}_{transform}/'
    if not os.path.isdir(prob_buf_dir):
        os.mkdir(prob_buf_dir)

    ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_krylov)
    val, vec = trunc_eigh(ideal_h, ideal_s, epsilon=1e-14)
    ideal_gnd = np.min(val)

    print("=== Sampling Perturbation Analysis ===")
    print("")

    for is_toeplitz, meas_type in product([True, False], ["FH", "LCU"]):
        frag = ham_frag if meas_type == "FH" else lcu_frag

        buf_dir = os.path.join(prob_buf_dir, f"./toeplitz_{is_toeplitz}_{meas_type}/")
        if not save_buffer:
            buf_dir = None
        elif not load_buffer and os.path.isdir(buf_dir):
            os.remove(buf_dir)

        t = time()
        shot_alloc = qksd_shot_allocation(tot_shots, frag, n_krylov, meas_type, is_toeplitz)
        checksum_shot = np.sum(shot_alloc) if isinstance(shot_alloc, np.ndarray) \
            else (np.sum(shot_alloc[0]) + np.sum(shot_alloc[1]))
        assert np.isclose(checksum_shot, tot_shots)
        samp_h, samp_s = sample_qksd(frag, prop_rte, ref, n_krylov, is_toeplitz, meas_type, shot_alloc,
                                     n_batch, buf_dir)
        t = time() - t

        pert_h = np.linalg.norm(samp_h - ideal_h, ord=2, axis=(1, 2))
        pert_s = np.linalg.norm(samp_s - ideal_s, ord=2, axis=(1, 2))
        eigvals = np.zeros(n_batch, dtype=float)
        for i in range(n_batch):
            val, vec = trunc_eigh(samp_h[i, :, :], samp_s[i, :, :], epsilon=pert_s[i])
            eigvals[i] = np.min(val)

        print(f"\t{meas_type}, Toeplitz = {is_toeplitz} ({t} sec)")
        print(f"\t\t‖ΔH‖ AVG = {np.mean(pert_h)} | STD = {np.std(pert_h)}")
        print(f"\t\t‖ΔS‖ AVG = {np.mean(pert_s)} | STD = {np.std(pert_s)}")
        print(f"\t\t ΔE  AVG = {np.mean(eigvals - ideal_gnd)} | STD = {np.std(eigvals - ideal_gnd)}")
        print("")


def _profile_sample(prop_rte, ref, n_krylov, ham_frag, lcu_frag, **_):
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


def _killer_shift(pham, fham, prop_rte, ref, mol_name, n_krylov, shift_op, transform,
                  **_):
    tot_shots = 1e8
    n_batch = 100
    is_toeplitz = True
    save_buffer = True
    load_buffer = True

    prob_buf_dir = f'./tmp_prob_buf_{mol_name}_{transform}/'
    if not os.path.isdir(prob_buf_dir):
        os.mkdir(prob_buf_dir)

    ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_krylov)
    val, vec = trunc_eigh(ideal_h, ideal_s, epsilon=1e-14)
    ideal_gnd = np.min(val)

    print("=== Killer-Shift Sampling Perturbation Analysis ===")
    print("")

    for optimization_level in [None, 0, 1, 2]:
        if optimization_level is None:
            shift_fham, shift_pham, shift_const = fham, pham, 0.0
        else:
            shift_fham, shift_pham, shift_const = shift_op[optimization_level]
        sh_ideal_h, _ = ideal_qksd_toeplitz(shift_pham, prop_rte, ref, n_krylov)

        for meas_type in ["FH", "LCU"]:
            buf_dir = os.path.join(prob_buf_dir, f"./ks{optimization_level}_toeplitz_{is_toeplitz}_{meas_type}/")
            if not save_buffer:
                buf_dir = None
            elif not load_buffer and os.path.isdir(buf_dir):
                os.remove(buf_dir)

            frag = sorted_insertion(shift_pham, anticommute=meas_type == "LCU")
            shot_alloc = qksd_shot_allocation(tot_shots, frag, n_krylov, meas_type, is_toeplitz)
            samp_h, samp_s = sample_qksd(frag, prop_rte, ref, n_krylov, is_toeplitz,
                                         meas_type, shot_alloc, n_batch, buf_dir)
            pert_h = np.linalg.norm(samp_h - sh_ideal_h, ord=2, axis=(1, 2))
            pert_s = np.linalg.norm(samp_s - ideal_s, ord=2, axis=(1, 2))
            eigvals = np.zeros(n_batch, dtype=float)
            for i in range(n_batch):
                val, vec = trunc_eigh(samp_h[i, :, :], samp_s[i, :, :], epsilon=pert_s[i])
                eigvals[i] = np.min(val) + shift_const

            print(f"\topt_level = {optimization_level}  ({meas_type})")
            print(f"\t\t‖H‖β = {sum([h.induced_norm(order=2) for h in frag])}")
            print(f"\t\t‖ΔH‖ AVG = {np.mean(pert_h)} | STD = {np.std(pert_h)}")
            print(f"\t\t‖ΔS‖ AVG = {np.mean(pert_s)} | STD = {np.std(pert_s)}")
            print(f"\t\t ΔE  AVG = {np.mean(eigvals - ideal_gnd)} | STD = {np.std(eigvals - ideal_gnd)}")
            print("")


def _iterative_coefficient_split(mol, mol_name, pham, ref, n_krylov, prop_rte, time_step, norm, cisd_state, shift_op,
                                 transform, **_):
    tot_shots = 1e8
    num_workers = 8
    remove_buf = False
    skip_count = 0

    n_batch = 100
    save_buffer = True
    load_buffer = True

    prob_buf_dir = f'./tmp_prob_buf_{mol_name}_{transform}/'
    if not os.path.isdir(prob_buf_dir):
        os.mkdir(prob_buf_dir)

    _, shift_pham, shift_const = shift_op[2]

    ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_krylov)
    val, vec = trunc_eigh(ideal_h, ideal_s, epsilon=1e-14)
    ideal_gnd = np.min(val)

    sh_ideal_h, _ = ideal_qksd_toeplitz(shift_pham, prop_rte, ref, n_krylov)

    cisd_energy = mol.cisd_energy
    cisd_phase = np.array([time_step * cisd_energy * k / norm for k in range(n_krylov)])

    print("=== ICS Sampling Perturbation Analysis ===")
    print("")
    idx_ex = skip_count
    for is_toeplitz, meas_type in product([True, False], ["FH", "LCU"]):
        print(f"\tToeplitz = {is_toeplitz} meas type = {meas_type}")
        for is_shifted, ics_level in product([False, True], [0, 1, 2, 3]):
            if (not is_toeplitz) and is_shifted:  # Shift technique only available for Toeplitz construction
                continue
            if skip_count > 0:
                skip_count -= 1
                continue
            curr_pham = shift_pham if is_shifted else pham
            curr_ideal_h = sh_ideal_h if is_shifted else ideal_h
            curr_shift_const = shift_const if is_shifted else 0.0
            anticommute = meas_type == "LCU"
            cov_buf_dir = f"./tmp_cov_{mol_name}_{meas_type}_sh={is_shifted}_icslv={ics_level}/"

            print(f"\tshift={is_shifted}, ics_level={ics_level} ({idx_ex})")
            idx_ex += 1

            buf_dir = os.path.join(prob_buf_dir, f"./ks_{is_shifted}_ics{ics_level}_toeplitz_{is_toeplitz}_{meas_type}/")
            if not save_buffer:
                buf_dir = None
            elif not load_buffer and os.path.isdir(buf_dir):
                os.remove(buf_dir)

            perform_ics, include_phase, sep_reim = False, False, False
            if ics_level == 0:  # No ICS
                pass
            elif ics_level == 1:  # ICS with <HF|O|CISD>
                perform_ics = True
            elif ics_level == 2:  # ICS with <HF|O|e^{-i E_{CISD} k Δt}|CISD>
                perform_ics = True
                include_phase = True
            elif ics_level == 3:  # ICS with <HF|O|e^{-i E_{CISD} k Δt}|CISD> and seperated RE/IM.
                perform_ics = True
                include_phase = True
                sep_reim = True

            if perform_ics:
                if cov_buf_dir is not None and os.path.exists(cov_buf_dir):
                    print("\t\tCov buf exists.")

                t = time()
                initial_grp, cov_dict = init_ics(curr_pham, ref, cisd_state, num_workers, anticommute,
                                                 cov_buf_dir=cov_buf_dir,
                                                 phase_list=cisd_phase if include_phase else None)
                t = time() - t
                print(f"\t\tCov Pauli Took {t}")
                if include_phase:
                    grp_ham_list, frag_shots_list = list(), list()
                    t = time()
                    for j, ph in enumerate(cisd_phase):
                        grp_ham, frag_shots, final_var, c_opt = run_ics(curr_pham, initial_grp, cov_dict[ph],
                                                                        sep_reim=sep_reim,
                                                                        transition=True, conv_atol=1e-5, conv_rtol=1e-3,
                                                                        debug=False)
                        grp_ham_list.append(grp_ham)
                        frag_shots_list.append(frag_shots)
                    t = time() - t
                    print(f"\t\tICS took {t}")
                    grp_ham = np.array(grp_ham_list)
                    frag_shots = np.array(frag_shots_list)
                else:
                    t = time()
                    grp_ham, frag_shots, final_var, c_opt = run_ics(curr_pham, initial_grp, cov_dict,
                                                                    sep_reim=sep_reim,
                                                                    transition=True, conv_atol=1e-5, conv_rtol=1e-2)
                    t = time() - t
                    print(f"\t\tICS took {t}")
            else:
                grp_ham = sorted_insertion(curr_pham, anticommute)
                frag_shots = None

            shot_alloc = qksd_shot_allocation(tot_shots, grp_ham, n_krylov, meas_type, is_toeplitz, frag_shots)
            samp_h, samp_s = sample_qksd(grp_ham, prop_rte, ref, n_krylov, is_toeplitz,
                                         meas_type, shot_alloc, n_batch, buf_dir)
            pert_h = np.linalg.norm(samp_h - curr_ideal_h, ord=2, axis=(1, 2))
            pert_s = np.linalg.norm(samp_s - ideal_s, ord=2, axis=(1, 2))
            eigvals = np.zeros(n_batch, dtype=float)
            for i in range(n_batch):
                val, vec = trunc_eigh(samp_h[i, :, :], samp_s[i, :, :], epsilon=pert_s[i])
                eigvals[i] = np.min(val) + curr_shift_const

            print(f"\t\t‖ΔH‖ AVG = {np.mean(pert_h)} | STD = {np.std(pert_h)}")
            print(f"\t\t‖ΔS‖ AVG = {np.mean(pert_s)} | STD = {np.std(pert_s)}")
            print(f"\t\t ΔE  AVG = {np.mean(eigvals - ideal_gnd)} | STD = {np.std(eigvals - ideal_gnd)}")
            print("")

            if cov_buf_dir is not None and remove_buf and os.path.exists(cov_buf_dir):
                os.remove(cov_buf_dir)


def _experiment(mol, mol_name, transform, shift_op, pham, prop_rte, ref, n_krylov, norm, time_step, cisd_state, **_):
    tot_shots = 1e8
    num_workers = 8

    n_batch = 100
    save_buffer = True
    load_buffer = True

    meas_type = "FH"

    prob_buf_dir = f'./tmp_prob_buf_{mol_name}_{transform}/'
    if not os.path.isdir(prob_buf_dir):
        os.mkdir(prob_buf_dir)

    result_dir = f"./tmp_{mol_name}_{transform}_result_matrices/"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    _, shift_pham, shift_const = shift_op[2]

    ideal_h, ideal_s = ideal_qksd_toeplitz(pham, prop_rte, ref, n_krylov)
    val, vec = trunc_eigh(ideal_h, ideal_s, epsilon=1e-14)
    ideal_gnd = np.min(val)

    sh_ideal_h, _ = ideal_qksd_toeplitz(shift_pham, prop_rte, ref, n_krylov)

    cisd_energy = mol.cisd_energy
    cisd_phase = np.array([time_step * cisd_energy * k / norm for k in range(n_krylov)])

    print(f"=== Experiment {mol_name}, {transform} ===")

    for is_shifted, perform_ics in product([True, False], [True, False]):
        print(f"is_shifte={is_shifted}, perform_ics={perform_ics}")
        curr_pham = shift_pham if is_shifted else pham
        curr_ideal_h = sh_ideal_h if is_shifted else ideal_h
        curr_shift_const = shift_const if is_shifted else 0.0
        anticommute = meas_type == "LCU"
        cov_buf_dir = f"./tmp_icscov_{mol_name}_{meas_type}_sh={is_shifted}/"
        buf_dir = os.path.join(prob_buf_dir, f"./ks_{is_shifted}_ics_{perform_ics}_{meas_type}/")
        if not save_buffer:
            buf_dir = None
        elif not load_buffer and os.path.isdir(buf_dir):
            os.remove(buf_dir)
        if perform_ics:
            if cov_buf_dir is not None and os.path.exists(cov_buf_dir):
                print("\t\tCov buf exists.")

            t = time()
            initial_grp, cov_dict = init_ics(curr_pham, ref, cisd_state, num_workers, anticommute,
                                             cov_buf_dir=cov_buf_dir,
                                             phase_list=cisd_phase)
            t = time() - t
            print(f"\t\tCov Pauli Took {t}")
            grp_ham_list, frag_shots_list = list(), list()
            t = time()
            for j, ph in enumerate(cisd_phase):
                grp_ham, frag_shots, final_var, c_opt = run_ics(curr_pham, initial_grp, cov_dict[ph],
                                                                sep_reim=True,
                                                                transition=True, conv_atol=1e-5, conv_rtol=1e-4,
                                                                debug=False)
                grp_ham_list.append(grp_ham)
                frag_shots_list.append(frag_shots)
            t = time() - t
            print(f"\t\tICS took {t}")

            grp_ham = np.array(grp_ham_list)
            frag_shots = np.array(frag_shots_list)
        else:
            grp_ham = sorted_insertion(curr_pham, anticommute)
            frag_shots = None

        shot_alloc = qksd_shot_allocation(tot_shots, grp_ham, n_krylov, meas_type,
                                          is_toeplitz=True,
                                          frag_shot_alloc=frag_shots)
        samp_h, samp_s = sample_qksd(grp_ham, prop_rte, ref, n_krylov, is_toeplitz=True,
                                     meas_type=meas_type, shot_list=shot_alloc, n_batch=n_batch, sample_buf_dir=buf_dir)
        pert_h = np.linalg.norm(samp_h - curr_ideal_h, ord=2, axis=(1, 2))
        pert_s = np.linalg.norm(samp_s - ideal_s, ord=2, axis=(1, 2))
        eigvals = np.zeros(n_batch, dtype=float)
        for i in range(n_batch):
            val, vec = trunc_eigh(samp_h[i, :, :], samp_s[i, :, :], epsilon=pert_s[i])
            eigvals[i] = np.min(val) + curr_shift_const

        print(f"\t\t‖ΔH‖ AVG = {np.mean(pert_h)} | STD = {np.std(pert_h)}")
        print(f"\t\t‖ΔS‖ AVG = {np.mean(pert_s)} | STD = {np.std(pert_s)}")
        print(f"\t\t ΔE  AVG = {np.mean(eigvals - ideal_gnd)} | STD = {np.std(eigvals - ideal_gnd)}")
        print("")
        with open(os.path.join(result_dir, f"shift={is_shifted}_ics={perform_ics}.pkl"), 'wb') as f:
            samp_h_list = samp_h.tolist()
            samp_s_list = samp_s.tolist()
            pickle.dump((samp_h_list, samp_s_list), f)


if __name__ == '__main__':
    _kwargs = _prepare()
    # _trotter_perturbation(**_kwargs)
    # _sample_perturbation(**_kwargs)
    # _profile_script_sample(**_kwargs)
    # _killer_shift(**_kwargs)
    # _iterative_coefficient_split(**_kwargs)
    _experiment(**_kwargs)
