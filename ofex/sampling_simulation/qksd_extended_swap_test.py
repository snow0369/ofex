from copy import deepcopy
from typing import Union, Tuple

import numpy as np
from openfermion import QubitOperator, FermionOperator, get_linear_qubit_operator_diagonal

from ofex.clifford.clifford_tools import tableau_to_pauli
from ofex.clifford.pauli_diagonalization import diagonalizing_clifford
from ofex.clifford.simulation import clifford_simulation
from ofex.exceptions import OfexTypeError
from ofex.linalg.sparse_tools import state_dot
from ofex.operators.fermion_operator_tools import is_number_only
from ofex.operators.qubit_operator_tools import is_z_only
from ofex.operators.symbolic_operator_tools import clean_imaginary
from ofex.sampling_simulation.sampling_base import JointProbDist
from ofex.state.state_tools import get_num_qubits, norm, normalize, to_dense
from ofex.state.types import State
from ofex.transforms.fermion_factorization import FermionFragment
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator, fermion_to_qubit_state, qubit_to_fermion_state
from ofex.transforms.fermion_rotation import fermion_rotation_state


def _fermion_to_pauli(ref_state_1: State,
                      ref_state_2: State,
                      num_op, u, transform, **kwargs) -> Tuple[QubitOperator, State, State]:
    num_qubits = get_num_qubits(ref_state_1)
    if num_qubits == u.shape[0]:
        is_spin = True
    elif num_qubits == u.shape[0] * 2:
        is_spin = False
    else:
        raise ValueError(f"unitary shape({u.shape[0]}) not matched with the state({num_qubits}).")
    op_pauli = fermion_to_qubit_operator(num_op, transform, **kwargs)

    def _mix_basis_pauli(r: State):
        # Qubit to fermion
        r = qubit_to_fermion_state(r, transform=transform, **kwargs)
        # Mix
        r = fermion_rotation_state(r, u.T.conj(), spatial_v=(not is_spin))
        return fermion_to_qubit_state(r, transform, **kwargs)

    ref1_sim = _mix_basis_pauli(ref_state_1)
    ref2_sim = _mix_basis_pauli(ref_state_2)
    return op_pauli, ref1_sim, ref2_sim


def prepare_qksd_est(ref_state_1: State,
                     ref_state_2: State,
                     operator: Union[QubitOperator, FermionOperator],
                     **kwargs):
    ref_state_1, ref_state_2 = to_dense(ref_state_1), to_dense(ref_state_2)
    if not np.isclose(state_dot(ref_state_1, ref_state_1), 1.0):
        raise ValueError("Ref1 is not normalized.")
    if not np.isclose(state_dot(ref_state_2, ref_state_2), 1.0):
        raise ValueError("Ref2 is not normalized.")
    n_qubits = get_num_qubits(ref_state_1)
    if n_qubits != get_num_qubits(ref_state_2):
        raise ValueError("Not matching number of qubits.")

    if isinstance(operator, QubitOperator):  # Pauli simulation
        # Find clifford
        op_mat, op_coeff, cl_hist = diagonalizing_clifford(operator, n_qubits)
        op_pauli = tableau_to_pauli(op_mat, None, op_coeff)
        op_pauli = QubitOperator.accumulate(op_pauli)
        # Clifford Simulation of ref vectors
        ref1_sim = clifford_simulation(ref_state_1, cl_hist)
        ref2_sim = clifford_simulation(ref_state_2, cl_hist)
    elif isinstance(operator, FermionOperator):  # Fermion simulation
        num_op, u = operator
        num_op: FermionOperator
        assert is_number_only(num_op) and isinstance(u, np.ndarray)
        op_pauli, ref1_sim, ref2_sim = _fermion_to_pauli(
            ref_state_1, ref_state_2, num_op, u, **kwargs
        )
    else:
        raise OfexTypeError(operator)
    assert is_z_only(op_pauli)
    assert np.isclose(op_pauli.constant, 0.0)

    op_pauli = clean_imaginary(op_pauli)
    op_pauli = get_linear_qubit_operator_diagonal(op_pauli, n_qubits=n_qubits)
    assert np.allclose(op_pauli.imag, 0.0)
    op_pauli = op_pauli.real
    return ref1_sim, ref2_sim, op_pauli


def qksd_extended_swap_test(ref_state_1: State,  # Qubit state
                            ref_state_2: State,  # Qubit state
                            operator: Union[QubitOperator, FermionFragment, np.ndarray],
                            imaginary: bool = False,
                            verbose=False,
                            eig_degen_tol=1e-8,
                            prepared=False,
                            **kwargs) -> Tuple[JointProbDist, JointProbDist]:
    real_prob_dict, imag_prob_dict = dict(), dict()

    # 1. Prepare diagonalized operators and reference states in the diagonalizing basis.
    if prepared:
        ref1_sim, ref2_sim, op_pauli = deepcopy(ref_state_1), deepcopy(ref_state_2), deepcopy(operator)
    else:
        ref1_sim, ref2_sim, op_pauli = prepare_qksd_est(ref_state_1, ref_state_2, operator, **kwargs)

    # 2. Probabilities of ancilla qubit
    ov = state_dot(ref_state_1, ref_state_2)

    p_ancilla_x_0, p_ancilla_y_0 = 0.5 * (1 + ov.real), 0.5 * (1 + ov.imag)
    p_ancilla_x_1, p_ancilla_y_1 = 1 - p_ancilla_x_0, 1 - p_ancilla_y_0

    # 3. Collapsed state after measuring ancilla
    state_x_0 = ref2_sim + ref1_sim
    state_x_1 = ref2_sim + ref1_sim * (-1)
    state_y_0 = ref2_sim + ref1_sim * 1j
    state_y_1 = ref2_sim + ref1_sim * (-1j)
    for st in [state_x_0, state_x_1, state_y_0, state_y_1]:
        if not np.isclose(norm(st), 0.0):
            normalize(st, inplace=True)

    # 4. Probability distribution for the system qubits with computational basis
    p_system_x_0 = abs(state_x_0) ** 2
    p_system_x_1 = abs(state_x_1) ** 2
    p_system_y_0 = abs(state_y_0) ** 2
    p_system_y_1 = abs(state_y_1) ** 2

    for p_ancilla, p_system in zip([p_ancilla_x_0, p_ancilla_x_1, p_ancilla_y_0, p_ancilla_y_1],
                                   [p_system_x_0, p_system_x_1, p_system_y_0, p_system_y_1]):
        if not np.isclose(p_ancilla, 0.0):
            assert np.isclose(sum(p_system), 1.0)

    # 5. Compute the probability distributions
    if verbose:  # Compute for each Pauli's.
        raise NotImplementedError
    else:
        key_list = ["H", "S"]
        # Range over all system outcomes (eigenvalues)
        for idx, (px0, px1, py0, py1) in enumerate(zip(p_system_x_0, p_system_x_1,
                                                       p_system_y_0, p_system_y_1)):
            # Range over real and imaginary
            for pr_list, storage in zip([[px0 * p_ancilla_x_0, px1 * p_ancilla_x_1],
                                         [py0 * p_ancilla_y_0, py1 * p_ancilla_y_1]],
                                        [real_prob_dict, imag_prob_dict]):
                # Range over ancilla outcomes
                for sign, pr in zip([1.0, -1.0], pr_list):
                    if np.isclose(pr, 0.0):
                        continue

                    # Event pair
                    ev_list = (op_pauli[idx] * sign, sign)  # H and S
                    if len(storage) > 0:
                        # Check duplicated event.
                        kl = list(storage.keys())
                        for k in kl:
                            if np.allclose(k, ev_list, atol=eig_degen_tol):
                                ev_list = k
                                break
                    if ev_list in storage:
                        storage[ev_list] += pr
                    else:
                        storage[ev_list] = pr
    return JointProbDist(key_list, real_prob_dict), JointProbDist(key_list, imag_prob_dict)
