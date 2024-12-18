from typing import Union, Tuple, Any

import numpy as np
from openfermion import QubitOperator, FermionOperator, get_linear_qubit_operator_diagonal

from ofex.clifford import tableau_to_pauli, diagonalizing_clifford, clifford_simulation
from ofex.exceptions import OfexTypeError
from ofex.linalg.sparse_tools import state_dot
from ofex.operators.fermion_operator_tools import is_number_only
from ofex.operators.qubit_operator_tools import is_z_only
from ofex.operators.symbolic_operator_tools import clean_imaginary
from ofex.sampling_simulation.sampling_base import JointProbDist
from ofex.state.state_tools import get_num_qubits, norm, normalize, to_dense
from ofex.state.types import State
from ofex.transforms.fermion_factorization import FermionFragment
from ofex.transforms import (fermion_to_qubit_operator, fermion_to_qubit_state, qubit_to_fermion_state,
                             fermion_rotation_state)


__all__ = ["qksd_extended_swap_test_run", "prepare_qksd_est_op", "prepare_qksd_est_state"]

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
                        **f2q_kwargs)\
        -> Tuple[np.ndarray, Any, str]:
    """
    Prepares the diagonalized operator representation for simulation of the Extended Swap Test (EST) for
    Quantum Krylov Subspace Diagonalization (QKSD).

    This function generates a diagonalized operator in the qubit computational basis, allowing for efficient 
    simulation and probability computation. The operator can either be a QubitOperator (representing Pauli 
    operators) or a FermionFragment (used for fermionic systems). The function identifies the type of the 
    operator, diagonalizes it, and returns the corresponding diagonal representation alongside the simulation 
    object and type metadata.

    Args:
        operator (Union[QubitOperator, FermionFragment]): The operator to be diagonalized with polynomial
            resources. Can be either a QubitOperator with mutually commuting Pauli operators, or a FermionFragment,
            which is a tuple of (FermionOperator, np.ndarray) (Refer to ofex.transform.double_factorization).
        n_qubits (int): The number of qubits in the system.
        **f2q_kwargs: Additional keyword arguments passed to fermion-to-qubit transformations, including
            'transform' keyword, required for FermionFragment input only. See ofex.transforms.fermion_to_qubit_operator.

    Returns:
        Tuple[np.ndarray, Any, str]: A tuple containing:
            - The diagonal representation of the operator as a real NumPy array.
            - The simulation object (Clifford history or orbital permutation matrix).
            - The simulation type ("PAULI" for Pauli-based simulations, "FERMION" for fermionic simulations).
    """
    if isinstance(operator, QubitOperator):  # Pauli simulation
        # Find clifford
        op_mat, op_coeff, cl_hist = diagonalizing_clifford(operator, n_qubits)
        op_pauli = tableau_to_pauli(op_mat, op_coeff, None)
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
    """
    Prepares the QKSD Extended Swap Test state in the diagonalized operator basis. This method transforms 
    the input reference state into the diagonalizing basis based on the simulation type (PAULI or FERMION).

    Args:
        ref_state (State): The input reference state.
        sim_obj (Any): The simulation object; output of prepare_qksd_est_op. For PAULI simulations,
            this is the Clifford history; for FERMION simulations, this is the orbital permutation matrix.
        sim_type (str): Type of simulation; output of prepare_qksd_est_op. Must be either "PAULI" (Pauli
            simulation) or "FERMION" (fermionic simulation).
        **f2q_kwargs: Additional keyword arguments passed to fermion-to-qubit transformations, including
            'transform' keyword, required for FermionFragment input only. See ofex.transforms.fermion_to_qubit_operator.

    Returns:
        State: The transformed state in the diagonalizing basis of the given operator.
    """
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


def qksd_extended_swap_test_run(ref_state_1: State,  # Qubit state
                                ref_state_2: State,  # Qubit state
                                operator: Union[QubitOperator, FermionFragment, Tuple[np.ndarray, Any, str]],
                                imaginary: bool,
                                eig_degen_tol=1e-8,
                                prepared_op=False,
                                prepared_state: Tuple[bool, bool] = (False, False),
                                verbose_prob=False,
                                **f2q_kwargs) -> JointProbDist:
    r"""
    Performs the Quantum Krylov Subspace Diagonalization (QKSD) Extended Swap Test.

    This function calculates the joint probability distribution of outcomes for the simultaneous measurement of
    :math:`\braket{ \phi(0) | \phi(t) }` and :math:`\braket{ \phi(0) | O | \phi(t) }` under the QKSD
    framework using the Extended Swap Test. The method supports operators that are either pre-diagonalized or require
    diagonalization during runtime, and it works with both Pauli and fermionic operator types.

    Args:
        ref_state_1 (State): The first input reference state (qubit state) :math:`\ket{ \phi(0) }`.
        ref_state_2 (State): The second input reference state (qubit state) :math:`\ket{ \phi(t) }`.
        operator (Union[QubitOperator, FermionFragment, Tuple[np.ndarray, Any, str]]): The operator used in the
            Extended Swap Test. It can be:

            - A ``QubitOperator`` representing Pauli operators.

            - A ``FermionFragment``: A tuple of ``(FermionOperator, np.ndarray)`` for fermionic systems.

            - A tuple ``(np.ndarray, Any, str)`` representing a pre-diagonalized operator, generated using
              the `prepare_qksd_est_op` function.

        imaginary (bool): If True, computes the imaginary part of the Extended Swap Test; otherwise, computes
            the real part.
        eig_degen_tol (float, optional): Tolerance for eigenvalue degeneracy when detecting similar probability
            events. Defaults to 1e-8.
        prepared_op (bool, optional): Indicates whether the operator has already been diagonalized. Defaults to False.
        prepared_state (Tuple[bool, bool], optional): Specifies whether the input reference states are already prepared
            in the diagonalizing basis. Defaults to ``(False, False)``.
        verbose_prob (bool, optional): If True, computes individual probability contributions for each Pauli term.
            **Note**: This is currently not implemented but may be added in the future to support bootstrapping
            covariance estimation. Defaults to False.
        **f2q_kwargs: Additional keyword arguments for fermion-to-qubit transformations. These are mandatory for
            fermionic systems when unprepared objects are provided.

    Returns:
        JointProbDist: An object containing the joint probability distribution of the computed probabilities.
    """

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
