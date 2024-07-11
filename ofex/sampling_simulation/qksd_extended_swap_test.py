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


def _fermion_to_pauli(ref_state: State,
                      u, transform, **kwargs) -> State:
    num_qubits = get_num_qubits(ref_state)
    if num_qubits == u.shape[0]:
        is_spin = True
    elif num_qubits == u.shape[0] * 2:
        is_spin = False
    else:
        raise ValueError(f"unitary shape({u.shape[0]}) not matched with the state({num_qubits}).")

    # Qubit to fermion
    ref_state = qubit_to_fermion_state(ref_state, transform=transform, **kwargs)
    # Mix
    ref_state = fermion_rotation_state(ref_state, u.T.conj(), spatial_v=(not is_spin))
    return fermion_to_qubit_state(ref_state, transform, **kwargs)


def prepare_qksd_est_op(operator: Union[QubitOperator, FermionFragment],
                        n_qubits: int,
                        **f2q_kwargs):
    if isinstance(operator, QubitOperator):  # Pauli simulation
        # Find clifford
        op_mat, op_coeff, cl_hist = diagonalizing_clifford(operator, n_qubits)
        op_pauli = tableau_to_pauli(op_mat, None, op_coeff)
        op_pauli = QubitOperator.accumulate(op_pauli)
        # Clifford Simulation of ref vectors
        sim_obj = cl_hist
        sim_type = "PAULI"
    elif isinstance(operator[0], FermionOperator):  # Fermion simulation
        num_op, u = operator
        assert is_number_only(num_op) and isinstance(u, np.ndarray)
        transform = f2q_kwargs.pop('transform', '')
        op_pauli = fermion_to_qubit_operator(num_op, transform, **f2q_kwargs)
        sim_obj = u
        sim_type = "FERMION"
    else:
        raise OfexTypeError(operator)
    assert is_z_only(op_pauli)
    assert np.isclose(op_pauli.constant, 0.0)

    op_pauli = clean_imaginary(op_pauli)
    op_pauli = get_linear_qubit_operator_diagonal(op_pauli, n_qubits=n_qubits)
    assert np.allclose(op_pauli.imag, 0.0)
    op_pauli = op_pauli.real

    return op_pauli, sim_obj, sim_type


def prepare_qksd_est_state(ref_state: State,
                           sim_obj,
                           sim_type: str,
                           **f2q_kwargs):
    ref_state = to_dense(ref_state)
    if not np.isclose(norm(ref_state), 1.0):
        raise ValueError("Ref is not normalized.")
    if sim_type == "PAULI":
        ref_sim = clifford_simulation(ref_state, sim_obj)
    elif sim_type == "FERMION":
        transform = f2q_kwargs.pop('transform', '')
        ref_sim = _fermion_to_pauli(ref_state, sim_obj, transform, **f2q_kwargs)
    else:
        raise ValueError
    return ref_sim


def qksd_extended_swap_test(ref_state_1: State,  # Qubit state
                            ref_state_2: State,  # Qubit state
                            operator,  # : Union[QubitOperator, FermionFragment, np.ndarray],
                            imaginary: bool,
                            eig_degen_tol=1e-8,
                            prepared_op=False,
                            prepared_state: Tuple[bool, bool] = (False, False),
                            verbose_prob=False,
                            **f2q_kwargs) -> JointProbDist:
    prob_dict = dict()

    # 1. Prepare diagonalized operators and reference states in the diagonalizing basis.
    n_qubits = get_num_qubits(ref_state_1)

    if prepared_op:
        op_pauli, sim_obj, sim_type = operator
    else:
        op_pauli, sim_obj, sim_type = prepare_qksd_est_op(operator, n_qubits, **f2q_kwargs)

    if prepared_state[0]:
        ref1_sim = ref_state_1
    else:
        ref1_sim = prepare_qksd_est_state(ref_state_1, sim_obj, sim_type, **f2q_kwargs)
    if prepared_state[1]:
        ref2_sim = ref_state_2
    else:
        ref2_sim = prepare_qksd_est_state(ref_state_2, sim_obj, sim_type, **f2q_kwargs)

    # 2. Probabilities of ancilla qubit
    ov = state_dot(ref1_sim, ref2_sim)

    p_ancilla_0 = 0.5 * (1 + ov.real) if not imaginary else 0.5 * (1 + ov.imag)
    p_ancilla_1 = 1 - p_ancilla_0

    # 3. Collapsed state after measuring ancilla
    ref1_sim, ref2_sim = to_dense(ref1_sim), to_dense(ref2_sim)
    ref1_sim = ref1_sim * 1j if imaginary else ref1_sim
    state_0 = ref2_sim + ref1_sim
    state_1 = ref2_sim + ref1_sim * (-1)
    for st in [state_0, state_1]:
        if not np.isclose(norm(st), 0.0):
            normalize(st, inplace=True)

    # 4. Probability distribution for the system qubits with computational basis
    p_system_0 = abs(state_0) ** 2
    p_system_1 = abs(state_1) ** 2

    # Check sum
    for p_ancilla, p_system in zip([p_ancilla_0, p_ancilla_1],
                                   [p_system_0, p_system_1]):
        if not np.isclose(p_ancilla, 0.0):
            assert np.isclose(sum(p_system), 1.0)

    # 5. Compute the probability distributions
    if verbose_prob:  # Compute for each Pauli's.
        raise NotImplementedError
    else:
        key_list = ["H", "S"]
        # Range over all system outcomes (eigenvalues)
        for idx, (p0, p1) in enumerate(zip(p_system_0, p_system_1)):
            # Range over ancilla outcomes
            for sign, pr in zip([1.0, -1.0], [p0 * p_ancilla_0, p1 * p_ancilla_1]):
                if np.isclose(pr, 0.0):
                    continue
                # Event pair
                ev_list = (op_pauli[idx] * sign, sign)  # H and S
                if len(prob_dict) > 0:
                    # Check duplicated event.
                    kl = list(prob_dict.keys())
                    for k in kl:
                        if np.allclose(k, ev_list, atol=eig_degen_tol):
                            ev_list = k
                            break
                if ev_list in prob_dict:
                    prob_dict[ev_list] += pr
                else:
                    prob_dict[ev_list] = pr
    return JointProbDist(key_list, prob_dict)
