from time import time

import numpy as np
from openfermion import get_fermion_operator

from ofex.measurement import sorted_insertion, pauli_split_group, pauli_covariance, fragment_variance
from ofex.transforms import fermion_to_qubit_operator
from ofex.utils.chem import molecule_example, run_driver
from test_scripts.random_object import random_state_nparray

mol = molecule_example("LiH")
mol.load()
mol = run_driver(mol, run_cisd=True, run_fci=True, driver="pyscf")

fham = get_fermion_operator(mol.get_molecular_hamiltonian())
fham = fham - fham.constant
pham = fermion_to_qubit_operator(fham, "symmetry_conserving_bravyi_kitaev",
                                 active_fermions = mol.n_electrons,
                                 active_orbitals = mol.n_qubits)
pham = pham - pham.constant

si = sorted_insertion(pham, anticommute=False)
shots = np.zeros((2, len(si)), dtype=float)
shots[0, :] = np.array([frag.induced_norm(order=2) for frag in si])
shots[1, :] = np.array([frag.induced_norm(order=2) for frag in si])
shots /= np.sum(shots)

state1 = random_state_nparray(mol.n_qubits - 2, seed=10)
state2 = random_state_nparray(mol.n_qubits - 2, seed=20)

ham_frags, (pauli_list, grp_pauli_list, _) = pauli_split_group(pham, anticommute=False, method='even')
cov_dict = pauli_covariance(pauli_list, grp_pauli_list, state1, state2, num_workers=7,
                            anticommute=False, )

t = time()
v_wo_cov = fragment_variance(si, state1, state2, shots, anticommute=False )
print( time() - t )

t = time()
v_w_cov = fragment_variance(si, state1, state2, shots, cov_dict, anticommute=False)
print( time() - t)

print(v_wo_cov, v_w_cov)
