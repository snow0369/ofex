from itertools import product
from typing import Union, Dict, Sequence, Callable

import numpy as np
from openfermion import QubitOperator, FermionOperator, jordan_wigner, bravyi_kitaev, bravyi_kitaev_tree, \
    binary_code_transform, symmetry_conserving_bravyi_kitaev, up_then_down
from openfermion.config import EQ_TOLERANCE

from ofex.linalg.sparse_tools import sparse_apply_operator
from ofex.state import BinaryFockVector
from ofex.state.state_tools import to_sparse_dict, to_dense, compress_sparse, get_num_qubits, state_type_transform, \
    pretty_print_state
from ofex.state.types import DenseState, SparseStateDict, State, type_state
from ofex.transforms.bravyi_kitaev_state import bravyi_kitaev_state, inv_bravyi_kitaev_state
from ofex.transforms.bravyi_kitaev_tree_state import bravyi_kitaev_tree_state, inv_bravyi_kitaev_tree_state


def fermion_to_qubit_operator(fermion_op: FermionOperator,
                              transform: str,
                              **kwargs) -> QubitOperator:
    """
    Convert a FermionOperator to a QubitOperator using a specified transformation.
    
    Args:
        fermion_op (FermionOperator): The input operator in the fermionic basis to be transformed.
        transform (str): The transformation method to be applied. Supported options:
                         "jordan_wigner", "bravyi_kitaev", "bravyi_kitaev_tree",
                         "binary_code_transform", "symmetry_conserving_bravyi_kitaev".
        kwargs: Additional arguments required for some transformations, such as:
                - "n_qubits" (int): Number of qubits for transformations "bravyi_kitaev" and "bravyi_kitaev_tree".
                - "code" (BinaryCode): The binary code for "binary_code_transform".
                - "active_fermions" (int) and "active_orbitals" (int): For "symmetry_conserving_bravyi_kitaev".
    
    Returns:
        QubitOperator: The transformed operator in the qubit representation.
    
    Raises:
        ValueError: If the provided transform method is not supported.
                    If required arguments in kwargs are missing for certain transformations.
    """
    # TODO: Complete bksf for FermionOperator
    if transform == "jordan_wigner":
        return jordan_wigner(fermion_op)
    elif transform == "bravyi_kitaev":
        if 'n_qubits' not in kwargs:
            raise ValueError("n_qubits must be provided in kwargs.")
        return bravyi_kitaev(fermion_op, n_qubits=kwargs['n_qubits'])
    elif transform == "bravyi_kitaev_tree":
        if 'n_qubits' not in kwargs:
            raise ValueError("n_qubits must be provided in kwargs.")
        return bravyi_kitaev_tree(fermion_op, n_qubits=kwargs['n_qubits'])
    elif transform == "binary_code_transform":
        if 'code' not in kwargs:
            raise ValueError("code must be provided in kwargs.")
        return binary_code_transform(fermion_op, code=kwargs['code'])
    elif transform == "symmetry_conserving_bravyi_kitaev":
        # Two-qubit reduction.
        if 'active_fermions' not in kwargs:
            raise ValueError("active_fermions must be provided in kwargs.")
        if 'active_orbitals' not in kwargs:
            raise ValueError("active_orbitals must be provided in kwargs.")
        return symmetry_conserving_bravyi_kitaev(fermion_op,
                                                 active_fermions=kwargs['active_fermions'],
                                                 active_orbitals=kwargs['active_orbitals'])
    else:
        raise ValueError(f"{transform} is not a supported transform.")


def fermion_to_qubit_state(fermion_state: State,
                           transform: str,
                           **kwargs) -> State:
    """
    Convert a fermionic state to a qubit state using a specified transformation.

    Args:
        fermion_state (State): The input state in the fermionic basis to be transformed.
        transform (str): The transformation method to be applied. Supported options:
                         "jordan_wigner", "bravyi_kitaev", "bravyi_kitaev_tree",
                         "symmetry_conserving_bravyi_kitaev".
        kwargs: Additional arguments required for specific transformations, such as:
                - "active_orbitals" (int): For symmetry-conserving transformations. If not provided, the number
                    of full orbitals is used.

    Returns:
        State: The transformed state in the qubit representation.
    """
    if transform == "jordan_wigner":
        return fermion_state

    input_type = type_state(fermion_state)
    fermion_state = to_sparse_dict(fermion_state)

    if transform == "bravyi_kitaev":
        qubit_state = bravyi_kitaev_state(fermion_state)
    elif transform == "bravyi_kitaev_tree":
        qubit_state = bravyi_kitaev_tree_state(fermion_state)
    elif transform == "symmetry_conserving_bravyi_kitaev":
        #n_orbitals = get_num_qubits(fermion_state)
        #parity_set = [node.index for node in fenwick_tree.get_parity_set(index)]
        #ancestors = [node.index for node in fenwick_tree.get_update_set(index)]
        #ancestor_children = [node.index for node in fenwick_tree.get_remainder_set(index)]
        active_orbitals = kwargs.get("active_orbitals", get_num_qubits(fermion_state))
        fermion_state_reorder = reorder_state(fermion_state, up_then_down, phase=True)

        # fenwick_tree = FenwickTree(active_orbitals)
        # qubit_excitation = QubitOperator()
        # for fock, coeff in fermion_state_reorder.items():
        #     f_term = [(i, 1) for i, f in enumerate(fock) if f][::-1]
        #     qubit_excitation += _transform_operator_term(f_term, coeff, fenwick_tree)
        # qubit_zero = {BinaryFockVector([0 for _ in range(active_orbitals)]): 1.0}
        # qubit_state = sparse_apply_operator(qubit_excitation, qubit_zero)

        qubit_state = bravyi_kitaev_tree_state(fermion_state_reorder)
        qubit_state = remove_indices_state(qubit_state, (active_orbitals // 2 - 1, active_orbitals - 1))
    else:
        qubit_state = fermion_to_qubit_state_general(fermion_state, transform, **kwargs)

    return state_type_transform(qubit_state, input_type)


def fermion_to_qubit_state_general(state: SparseStateDict,
                                   transform: str,
                                   **kwargs) -> SparseStateDict:
    """
    Convert a general fermionic state to a qubit state using the specified transformation.
    Because this function transforms the creation operator to qubit form and multiplies it by the state,
    it is computationally expensive.
    This function is not recommended for large systems; and used for debugging and other transform
    not specified in the fermion_to_qubit_state function.

    Args:
        state (SparseStateDict): The input state in the fermionic basis to be transformed.
        transform (str): The transformation to apply. Supported options include
                         "jordan_wigner", "bravyi_kitaev", "bravyi_kitaev_tree",
                         and others. The "symmetry_conserving_bravyi_kitaev" 
                         transformation is not implemented.
        kwargs: Additional arguments required for some transformations, such as:
                - "n_qubits" (int): Number of qubits for transformations "bravyi_kitaev" and "bravyi_kitaev_tree".
                - "code" (BinaryCode): The binary code for "binary_code_transform".
                - "active_fermions" (int) and "active_orbitals" (int): For "symmetry_conserving_bravyi_kitaev".
    Returns:
        SparseStateDict: The transformed state in the qubit basis.
    
    Raises:
        NotImplementedError: If the "symmetry_conserving_bravyi_kitaev" transformation is requested.
    """
    # Inefficient
    new_state = dict()

    if transform == "symmetry_conserving_bravyi_kitaev":
        raise NotImplementedError

    for fock, coeff in state.items():
        cre_op = FermionOperator(tuple([(idx, 1) for idx, f in enumerate(fock) if f]))
        pauli_cre_op = fermion_to_qubit_operator(cre_op, transform, **kwargs)
        num_qubits = len(fock) if transform != 'symmetry_conserving_bravyi_kitaev' else len(fock) - 2
        state_now = {BinaryFockVector(tuple([0 for _ in range(num_qubits)])): coeff}
        state_now = compress_sparse(sparse_apply_operator(pauli_cre_op, state_now))
        assert len(state_now) == 1, state_now
        fock, coeff = list(state_now.items())[0]
        if transform == "symmetry_conserving_bravyi_kitaev":
            if fock in new_state and not np.isclose(coeff, 0.0, atol=EQ_TOLERANCE):
                raise ValueError("Symmetry is broken.")
        new_state[fock] = coeff
    return new_state


def qubit_to_fermion_state(qubit_state: State,
                           transform: str,
                           **kwargs) -> Union[DenseState, SparseStateDict]:
    """
    Convert a qubit state to a fermionic state using a specified transformation.

    Args:
        qubit_state (State): Input state in the qubit basis to be transformed.
        transform (str): The transformation method to apply. Supported options:
                         "jordan_wigner", "bravyi_kitaev", "bravyi_kitaev_tree",
                         "symmetry_conserving_bravyi_kitaev".
        kwargs: Additional arguments required for some transformations, such as:
                - "active_fermions" (int): Number of active fermions for the symmetry-conserving transformation.
                - "active_orbitals" (int): For symmetry-conserving transformations. If not provided, the number
                    of full orbitals is used.

    Returns:
        Union[DenseState, SparseStateDict]: Transformed fermionic state in sparse or dense form.

    Raises:
        ValueError: If required arguments for specific transformations are missing.
        NotImplementedError: For unspecified transformation methods.
    """
    if transform == "jordan_wigner":
        return qubit_state

    is_dense_input = isinstance(qubit_state, DenseState)
    if is_dense_input:
        qubit_state = to_sparse_dict(qubit_state)

    if transform == "bravyi_kitaev":
        fermion_state = inv_bravyi_kitaev_state(qubit_state)
    elif transform == "bravyi_kitaev_tree":
        fermion_state = inv_bravyi_kitaev_tree_state(qubit_state)
    elif transform == "symmetry_conserving_bravyi_kitaev":
        if 'active_fermions' not in kwargs:
            raise ValueError("active_fermions must be provided in kwargs.")
        active_fermions = kwargs['active_fermions']
        active_orbitals = kwargs.get("active_orbitals", get_num_qubits(qubit_state) + 2)
        remainder = active_fermions % 4
        if remainder == 0:
            final_orb = 0
            middle_orb = 0
        elif remainder == 1:
            final_orb = 1
            middle_orb = 1
        elif remainder == 2:
            final_orb = 0
            middle_orb = 1
        else:
            final_orb = 1
            middle_orb = 0
        active_orbitals = int(active_orbitals)
        qubit_state = recover_indices_state(qubit_state, {active_orbitals // 2: middle_orb, active_orbitals: final_orb})
        fermion_state = inv_bravyi_kitaev_tree_state(qubit_state)
        fermion_state = reorder_state(fermion_state, up_then_down, reverse=True)
    else:
        raise NotImplementedError

    if is_dense_input:
        return to_dense(fermion_state)
    else:
        return fermion_state


def reorder_state(state: SparseStateDict,
                  order_function: Callable[[int, int], int],
                  reverse=False,
                  phase=False) -> SparseStateDict:
    """
    Reorder the modes of a fermionic state according to a specified ordering function.

    Args:
        state (SparseStateDict): The input fermionic state as a sparse dictionary.
        order_function (Callable[[int, int], int]): A function that determines the new positions 
                        of the current modes based on the total number of modes.
        reverse (bool): If True, undoes the reordering using the inverse mode map. Default is False.
        phase (bool): If True, accounts for the phase changes incurred during reordering. Default is False.

    Returns:
        SparseStateDict: A new fermionic state with reordered modes according to the specified function.
    """
    num_modes = get_num_qubits(state)
    inv_mode_map = {mode_idx: order_function(mode_idx, num_modes) for mode_idx in range(num_modes)}
    mode_map = {val: key for key, val in inv_mode_map.items()}
    if reverse:
        inv_mode_map, mode_map = mode_map, inv_mode_map

    new_state = dict()
    for fock, coeff in state.items():
        if phase:
            pairs = [(i, inv_mode_map[i]) for i, f in enumerate(fock) if f]
            n_cross_hf = int((len(pairs) - 2) * len(pairs) // 8)
            count_cross = 0
            for (idx_p1, p1), (idx_p2, p2) in product(enumerate(pairs), enumerate(pairs)):
                if idx_p1 <= idx_p2:
                    continue
                i1, j1 = p1
                i2, j2 = p2
                assert i1 != i2
                assert j1 != j2
                count_cross += int((i1 < i2) != (j1 < j2))
            if (count_cross % 2) != (n_cross_hf % 2):
                coeff = -coeff
        new_fock = BinaryFockVector([fock[mode_map[i]] for i in range(num_modes)])
        new_state[new_fock] = coeff
    return new_state


def remove_indices_state(state: SparseStateDict, indices: Sequence[int],
                         check_symmetry_conserved=True) -> SparseStateDict:
    """
    Removes specified mode indices from a fermionic state, projecting the state
    while checking for symmetry conservation if required.

    Args:
        state (SparseStateDict): The input state in the sparse fermionic basis.
        indices (Sequence[int]): Indices of the modes to be removed from the state.
        check_symmetry_conserved (bool): Whether to enforce symmetry conservation 
            in the removed modes. If symmetry is not conserved, raises a ValueError.

    Returns:
        SparseStateDict: A new fermionic state with the specified modes removed
        and optionally verified for symmetry conservation.
    
    Raises:
        ValueError: If symmetry conservation is violated when check_symmetry_conserved
        is set to True.
    """
    new_state = dict()
    check_f = None
    num_modes = get_num_qubits(state)
    remaining_indices = sorted(set(range(num_modes)).difference(indices))
    for fock, coeff in state.items():
        if abs(coeff) < EQ_TOLERANCE:
            continue
        new_fock = BinaryFockVector([fock[idx] for idx in remaining_indices])
        if check_symmetry_conserved:
            check_f_now = BinaryFockVector([fock[idx] for idx in indices])
            if check_f is None:
                check_f = check_f_now
            elif check_f != check_f_now:
                raise ValueError(f"Symmetry is not conserved. {new_fock} {check_f} \n{pretty_print_state(new_state)}")
        assert new_fock not in new_state
        new_state[new_fock] = coeff
    return new_state


def recover_indices_state(state: SparseStateDict,
                          pos_fill: Dict[int, Union[int, bool]]) -> SparseStateDict:
    """
    Restore selected modes to a sparse fermionic state by filling specific
    positions with predefined values.

    Args:
        state (SparseStateDict): The input sparse state to be modified. It contains
                                 the current fermionic configuration with some indices removed.
        pos_fill (Dict[int, Union[int, bool]]): A dictionary specifying the indices to be
                                 restored and the corresponding fixed values to insert at those positions.

    Returns:
        SparseStateDict: A new fermionic state with the specified positions restored.
                         The inserted positions are filled with the provided values in pos_fill.
    """
    new_state = dict()
    for fock, coeff in state.items():
        new_fock = list(fock)
        for inserted, pos in enumerate(sorted(pos_fill.keys())):
            fill = pos_fill[pos]
            new_fock.insert(pos + inserted, fill)
        new_state[BinaryFockVector(new_fock)] = coeff
    return new_state
