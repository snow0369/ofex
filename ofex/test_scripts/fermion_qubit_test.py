import os
import pickle
from itertools import product

import numpy as np
from numpy.random import rand
from openfermion import FermionOperator, get_fermion_operator, normal_ordered

from ofex.linalg.sparse_tools import expectation
from ofex.operators.symbolic_operator_tools import compare_operators
from ofex.state.binary_fock import BinaryFockVector
from ofex.state.chem_ref_state import hf_ground, cisd_ground
from ofex.state.state_tools import compare_states, pretty_print_state
from ofex.test_scripts.random_object import random_state_spdict
from ofex.transforms.bravyi_kitaev_deprecated import bravyi_kitaev as bravyi_kitaev_original, \
    inv_bravyi_kitaev_state
from ofex.transforms.bravyi_kitaev_deprecated import bravyi_kitaev_state
from ofex.transforms.bravyi_kitaev_tree_state import bravyi_kitaev_tree_state, \
    inv_bravyi_kitaev_tree_state
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator, fermion_to_qubit_state
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


def energy_test(mol, transform, cisd_state):
    if transform in ["bravyi_kitaev", "bravyi_kitaev_tree"]:
        f2q_kwargs = {"n_qubits": mol.n_qubits}
    elif transform == "symmetry_conserving_bravyi_kitaev":
        f2q_kwargs = {"active_fermions": mol.n_electrons,
                      "active_orbitals": mol.n_qubits}
    elif transform == "jordan_wigner":
        f2q_kwargs = dict()
    else:
        raise NotImplementedError

    mol_dir = "./tmp_mol_data/"
    if not os.path.isdir(mol_dir):
        os.mkdir(mol_dir)
    fname = os.path.join(mol_dir, f"{mn}.pkl")

    if not os.path.isfile(fname):
        fham = get_fermion_operator(mol.get_molecular_hamiltonian())
        fham = normal_ordered(fham)
        pkl_fham = fham.terms
        pkl_cisd = {tuple(f):v for f, v in cisd_state.items()}
        with open(fname, 'wb') as f:
            pickle.dump((pkl_fham, pkl_cisd), f)
    else:
        with open(fname, 'rb') as f:
            pkl_fham, pkl_cisd = pickle.load(f)
        fham = FermionOperator()
        fham.terms = pkl_fham
        cisd_state = {BinaryFockVector(f): v for f, v in pkl_cisd.items()}

    qham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
    f_hf_state = hf_ground(mol)
    hf_state = fermion_to_qubit_state(f_hf_state, transform, **f2q_kwargs)
    print("HF=")
    print(pretty_print_state(hf_state))
    est_hf_energy = expectation(qham, hf_state)
    assert np.isclose(est_hf_energy.imag, 0.0)
    est_hf_energy = est_hf_energy.real
    assert np.isclose(est_hf_energy, mol.hf_energy, atol=1e-7), '\n'.join(
        [f"{est_hf_energy:.4f} {mol.hf_energy:.4f}", pretty_print_state(hf_state)])

    cisd_state_qubit = fermion_to_qubit_state(cisd_state, transform, **f2q_kwargs)

    if len(cisd_state) < 20:
        print("CISDFERMION=")
        print(pretty_print_state(cisd_state))
        print("CISDQUBIT=")
        print(pretty_print_state(cisd_state_qubit))
    else:
        f_hf_vector = list(f_hf_state.keys())[0]
        hf_vector = list(hf_state.keys())[0]
        print(f"CISDFERMION[HF]= {cisd_state[f_hf_vector]}")
        print(f"CISDQUBIT[HF]  = {cisd_state_qubit[hf_vector]}")

    est_cisd_energy = expectation(qham, cisd_state_qubit)
    assert np.isclose(est_cisd_energy.imag, 0.0)
    est_cisd_energy = est_cisd_energy.real
    assert np.isclose(est_cisd_energy, mol.cisd_energy, atol=1e-7), (est_cisd_energy, mol.cisd_energy, mol.hf_energy)


if __name__ == "__main__":
    for n_qubits in range(1, 13):
        print(f"\nn_qubits: {n_qubits}")
        bravyi_kitaev_test(n_qubits)  # Passed
        state_transform_test(n_qubits, 'bravyi_kitaev', n_qubits=n_qubits)  # Passed
        state_transform_test(n_qubits, 'bravyi_kitaev_tree', n_qubits=n_qubits)
        state_transform_test(n_qubits, 'jordan_wigner', n_qubits=n_qubits)
        print('Passed')

    tr_list = ['jordan_wigner', 'bravyi_kitaev', 'bravyi_kitaev_tree', 'symmetry_conserving_bravyi_kitaev']
    mol_name_list = ["H2", "H4", "LiH", "BeH2", "H2O"]

    mol_list = {mn: molecule_example(mn) for mn in mol_name_list}
    cisd_state_list = {mn: cisd_ground(mol_list[mn]) for mn in mol_name_list}

    for mn, tr in product(mol_name_list, tr_list):
        print(f"\nHF energy test : {mn} | {tr}")
        energy_test(mol_list[mn], tr, cisd_state_list[mn])
        print("Passed")
