from typing import List, Optional, Tuple, Union

import numpy as np
from openfermion import QubitOperator
from scipy.linalg import eigh

from ofex.measurement.iterative_coefficient_splitting.ics_utils import _synthesize_group, \
    _calculate_groupwise_std, _generate_cov_list, _add_epsilon_shot
from ofex.measurement.pauli_variance import pauli_variance, pauli_covariance
from ofex.measurement.types import PauliCovDict, TransitionPauliCovDict
from ofex.operators.symbolic_operator_tools import coeff


def run_ics(ham: QubitOperator,
            initial_grp: Tuple[List[QubitOperator], List[List[int]], List[List[int]]],
            cov_dict: Union[PauliCovDict, TransitionPauliCovDict],
            transition: bool = False,
            conv_th: float = 1e-6,
            max_iter=10000,
            lstsq_rcond=1e-6,
            initial_c: Optional[np.ndarray] = None,
            debug: bool = False, ) \
        -> Tuple[List[QubitOperator], np.ndarray, float, Optional[np.ndarray]]:
    """

    Args:
        ham: Initial Hamiltonian
        initial_grp:
        cov_dict: Covariances, (key : tuple of pauli string, value : real (and imaginary for transition = True))
        transition: Calculation for transition amplitude (ref1 and ref2 are different)
        conv_th: Threshold for the optimization convergence.
        max_iter: Maximum number of iteration in ics optimization
        lstsq_rcond:
        initial_c:
        debug: STDOUT for debugging

    Returns:
        group_hamiltonian, shot_allocation_real, shot_allocation_imag, optimal_variance, c_opt

    """
    if ham.constant != 0.0:
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
            transition, num_grp, size_grp, initial_c, cov_list_real, cov_list_imag
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
            c_opt = initial_c
            for it in range(max_iter):
                # 4-1. construct total covariance matrix
                tot_vmat = np.zeros((num_split_pauli, num_split_pauli))
                count_pauli = 0
                for idx_grp in range(num_grp):
                    if np.isclose(m_opt_real[idx_grp], 0):
                        block = np.zeros(cov_list_real[idx_grp].shape)
                    else:
                        block = cov_list_real[idx_grp] / m_opt_real[idx_grp]
                    if transition:
                        if not np.isclose(m_opt_imag[idx_grp], 0):
                            block += cov_list_imag[idx_grp] / m_opt_imag[idx_grp]
                    end_idx = count_pauli + size_grp[idx_grp]
                    tot_vmat[count_pauli: end_idx, count_pauli: end_idx] = block
                    count_pauli += size_grp[idx_grp]
                if converge:
                    break

                # 4-2. Solve coefficients
                mu = projector.T @ tot_vmat @ phi_s
                vmat_tilde = projector.T @ tot_vmat @ projector
                diag_v, u_v = eigh(vmat_tilde)
                for i in range(len(diag_v)):
                    if diag_v[i] < truncate_epsilon:
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
                std_real_list, std_imag_list = _calculate_groupwise_std(transition, num_grp, size_grp, new_c_opt,
                                                                        cov_list_real, cov_list_imag)

                # 4-3-2. Check Convergence
                new_sum_std = sum(std_real_list)
                if transition:
                    new_sum_std += sum(std_imag_list)
                if sum_std is not None and new_sum_std > sum_std:
                    break
                converge = abs(
                    new_sum_std - sum_std) < conv_th if sum_std is not None else False
                sum_std = new_sum_std
                c_opt = new_c_opt

                # 4-3-3. Allocate measurement
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
                    grp_ham = _synthesize_group(c_opt, pauli_list, grp_pauli_list, size_grp, ham, debug,
                                                correct_sum=False)
                    checksum = sum(grp_ham[1:], grp_ham[0])
                    if not checksum == ham:
                        print(f"mismatch during optimization : {(checksum - ham).two_norm}")
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
    grp_ham = _synthesize_group(c_opt, pauli_list, grp_pauli_list, size_grp, ham,
                                debug)  # , leftover, anticommute, debug)
    final_variance = 0.5 * c_opt.T @ tot_vmat @ c_opt

    n_frag = len(grp_ham)
    if transition:
        shots = np.zeros((n_frag, 2), dtype=float)
        shots[:, 0] = m_opt_real
        shots[:, 1] = m_opt_imag
    else:
        shots = m_opt_real

    return grp_ham, shots, final_variance, c_opt


if __name__ == "__main__":
    import json

    from openfermion import get_fermion_operator
    from ofex.utils.chem import molecule_example
    from ofex.transforms.fermion_qubit import fermion_to_qubit_operator, fermion_to_qubit_state
    from ofex.state.chem_ref_state import hf_ground
    from ofex.measurement.sorted_insertion import sorted_insertion
    from ofex.propagator.exact import exact_rte
    from ofex.measurement.iterative_coefficient_splitting import init_ics
    from ofex.linalg.sparse_tools import apply_operator
    from ofex.state.state_tools import pretty_print_state


    def ics_test():
        mol_name = "LiH"
        transform = 'symmetry_conserving_bravyi_kitaev'
        debug = True
        anticommute = False

        mol = molecule_example(mol_name)
        fham = mol.get_molecular_hamiltonian()
        fham = get_fermion_operator(fham)
        n_qubits = mol.n_qubits
        if transform == "jordan_wigner":
            kwargs = dict()
        elif transform == "bravyi_kitaev":
            kwargs = {'n_qubits': mol.n_qubits}
        elif transform == 'symmetry_conserving_bravyi_kitaev':
            kwargs = {'active_fermions': mol.n_electrons,
                      'active_orbitals': mol.n_qubits}
            n_qubits -= 2
        else:
            raise AssertionError

        pham = fermion_to_qubit_operator(fham, transform, **kwargs)
        pham = pham - pham.constant

        norm = sum([u.induced_norm(order=2) for u in sorted_insertion(pham, anticommute=True)])
        prop = exact_rte(pham, t=6 * np.pi / norm, n_qubits=n_qubits)

        ref1 = fermion_to_qubit_state(hf_ground(mol), transform, **kwargs)
        print(prop.shape)
        print(pretty_print_state(ref1))
        ref2 = apply_operator(prop, ref1)

        _, initial_grp, _ = init_ics(pham, anticommute=anticommute, debug=debug)
        cov_dict = pauli_covariance(initial_grp, ref1, ref2, anticommute=anticommute, num_workers=15, debug=debug)

        grp_ham, shots, final_var, c_opt = run_ics(pham,
                                                   initial_grp,
                                                   cov_dict,
                                                   transition=True,
                                                   conv_th=1e-6,
                                                   debug=debug)
        print(final_var)
        print("== ICS VAR ==")
        ev_var = pauli_variance(ref1, ref2, grp_ham, shots, )
        print(ev_var)
        print(np.sum(shots))

        print("== SI  VAR ==")
        si_group = sorted_insertion(pham, anticommute)
        m_si_real = np.array([x.induced_norm(order=2) for x in si_group])
        m_si_imag = np.array(m_si_real)
        div = sum(m_si_real) + sum(m_si_imag)
        m_si_real, m_si_imag = m_si_real / div, m_si_imag / div
        ev_var_si = pauli_variance(ref1, ref2, si_group, shots)
        print(ev_var_si)
        print(sum(m_si_real) + sum(m_si_imag))


    ics_test()
