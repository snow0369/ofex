from itertools import product

import numpy as np
from numpy.random import rand
from openfermion import FermionOperator, get_fermion_operator, normal_ordered, get_sparse_operator

from ofex.linalg.sparse_tools import expectation
from ofex.operators.symbolic_operator_tools import compare_operators
from ofex.state.chem_ref_state import hf_ground
from ofex.state.state_tools import compare_states, pretty_print_state, to_dense
from ofex.test_scripts.random_object import random_state_spdict
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator, fermion_to_qubit_state
from ofex.transforms.bravyi_kitaev_deprecated import bravyi_kitaev as bravyi_kitaev_original, \
    inv_bravyi_kitaev_state
from ofex.transforms.bravyi_kitaev_deprecated import bravyi_kitaev_state
from ofex.transforms.bravyi_kitaev_tree_state import bravyi_kitaev_tree_state, \
    inv_bravyi_kitaev_tree_state
from ofex.utils.chem import molecule_example
from ofex.utils.dict_utils import dict_allclose


def bravyi_kitaev_test(num_qubits):
    rand_coeff = rand() * 2 - 1
    for idx in range(num_qubits):
        cre_op_f = FermionOperator((idx, 1), rand_coeff)
        cre_op_p1 = fermion_to_qubit_operator(cre_op_f, 'bravyi_kitaev', n_qubits=num_qubits)
        cre_op_p2 = bravyi_kitaev_original(cre_op_f, num_qubits)
        try:
            assert cre_op_p1 == cre_op_p2
        except AssertionError as e:
            print(f"Testing {idx}/{num_qubits}")
            compare_operators(cre_op_p1, cre_op_p2)
            raise e


def state_transform_test(num_qubits, transform, **kwargs):
    fermion_state = random_state_spdict(num_qubits)
    qubit_state_1 = fermion_to_qubit_state(fermion_state, transform, **kwargs)
    if transform == "bravyi_kitaev":
        qubit_state_2 = bravyi_kitaev_state(fermion_state)
        fermion_state_2 = inv_bravyi_kitaev_state(qubit_state_2)
    elif transform == "bravyi_kitaev_tree":
        qubit_state_2 = bravyi_kitaev_tree_state(fermion_state)
        fermion_state_2 = inv_bravyi_kitaev_tree_state(qubit_state_2)
    elif transform == "jordan_wigner":
        qubit_state_2 = fermion_state
        fermion_state_2 = qubit_state_2
    else:
        raise NotImplementedError(transform)
    assert dict_allclose(qubit_state_1, qubit_state_2), '\n' + compare_states(qubit_state_1, qubit_state_2)
    assert dict_allclose(fermion_state, fermion_state_2), '\n' + compare_states(fermion_state, fermion_state_2)


def hf_test(mol_name, transform, **kwargs):
    mol = molecule_example(mol_name)
    num_qubits = mol.n_qubits
    fham = get_fermion_operator(mol.get_molecular_hamiltonian())
    fham = normal_ordered(fham)
    qham = fermion_to_qubit_operator(fham, transform, n_qubits=num_qubits, **kwargs)
    hf_fermion = hf_ground(mol)
    hf_state = hf_ground(mol, fermion_to_qubit_map=transform, n_qubits=num_qubits, **kwargs)
    est_hf_energy_direct = expectation(qham, hf_state)
    est_hf_energy_sparse = expectation(get_sparse_operator(qham, n_qubits=num_qubits), hf_state)
    assert np.isclose(est_hf_energy_direct, est_hf_energy_sparse), (est_hf_energy_direct, est_hf_energy_sparse)
    assert np.isclose(est_hf_energy_direct.imag, 0.0)
    est_hf_energy_direct = est_hf_energy_direct.real
    assert np.isclose(est_hf_energy_direct, mol.hf_energy), '\n'.join([pretty_print_state(hf_fermion),
                                                                pretty_print_state(hf_state),
                                                                str(to_dense(hf_fermion)),
                                                                f"{est_hf_energy_direct - mol.hf_energy:.4f}"])


if __name__ == "__main__":
    for n_qubits in range(1, 13):
        print(f"\nn_qubits: {n_qubits}")
        bravyi_kitaev_test(n_qubits)  # Passed
        state_transform_test(n_qubits, 'bravyi_kitaev', n_qubits=n_qubits)  # Passed
        state_transform_test(n_qubits, 'bravyi_kitaev_tree', n_qubits=n_qubits)
        state_transform_test(n_qubits, 'jordan_wigner', n_qubits=n_qubits)
        print('Passed')

    for mn, tr in product(["H2", "H4"], ['jordan_wigner', 'bravyi_kitaev', 'bravyi_kitaev_tree']):
        print(f"\nHF energy test : {mn} | {tr}")
        hf_test(mn, tr)
        print("Passed")
