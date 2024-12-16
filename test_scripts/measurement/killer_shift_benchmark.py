import pytest
from itertools import product

from openfermion import QubitOperator, get_fermion_operator

from ofex.measurement import sorted_insertion, killer_shift_opt_fermion_hf
from ofex.transforms import fermion_to_qubit_operator
from ofex.utils.chem import molecule_example
from ofex.state.chem_ref_state import hf_ground


def betas(pham: QubitOperator):
    fh = sorted_insertion(pham, anticommute=False)
    lcu = sorted_insertion(pham, anticommute=True)
    return sum([x.induced_norm(order=2) for x in fh]), sum([x.induced_norm(order=2) for x in lcu])

mol_list = ["H2", "LiH", "BeH2", "H2O"]
tr_list = ["jordan_wigner", "bravyi_kitaev", "symmetry_conserving_bravyi_kitaev"]

@pytest.mark.parametrize("mol_name, transform", list(product(mol_list, tr_list)))
def test_shift(mol_name, transform):
    print(f"mol = {mol_name}, transform = {transform}")
    mol = molecule_example(mol_name)
    hf = hf_ground(mol)
    assert len(hf) == 1
    hf = list(hf.keys())[0]
    fham = mol.get_molecular_hamiltonian()
    fham = get_fermion_operator(fham)
    if transform == "jordan_wigner":
        kwargs = dict()
    elif transform == "bravyi_kitaev":
        kwargs = {'n_qubits': mol.n_qubits}
    elif transform == 'symmetry_conserving_bravyi_kitaev':
        kwargs = {'active_fermions': mol.n_electrons,
                  'active_orbitals': mol.n_qubits}
    else:
        raise AssertionError
    pham = fermion_to_qubit_operator(fham, transform, **kwargs)
    pham = pham - pham.constant
    beta, antibeta = betas(pham)
    print("\tOriginal")
    print(f"\t\tnorm     = {pham.induced_norm(order=1)}")
    print(f"\t\tSI norm  = {beta}")
    print(f"\t\tASI norm = {antibeta}")
    # print(pham.pretty_string(num_tabs=2))

    _, shifted_pham_0, const = killer_shift_opt_fermion_hf(fham, hf, transform,
                                                           optimization_level=0,
                                                           f2q_kwargs=kwargs)
    beta, antibeta = betas(shifted_pham_0)
    print("\tShifted 0")
    print(f"\t\tnorm     = {shifted_pham_0.induced_norm(order=1)}")
    print(f"\t\tSI norm  = {beta}")
    print(f"\t\tASI norm = {antibeta}")
    print(f"\t\tshift_const = {const}")
    # print(shifted_pham_0.pretty_string(num_tabs=2))

    _, shifted_pham_1, const = killer_shift_opt_fermion_hf(fham, hf, transform,
                                                           optimization_level=1,
                                                           f2q_kwargs=kwargs)
    beta, antibeta = betas(shifted_pham_1)
    print("\tShifted 1")
    print(f"\t\tnorm     = {shifted_pham_1.induced_norm(order=1)}")
    # print(shifted_pham_1.pretty_string(num_tabs=2))
    print(f"\t\tSI norm  = {beta}")
    print(f"\t\tASI norm = {antibeta}")
    print(f"\t\tshift_const = {const}")

    _, shifted_pham_2, const = killer_shift_opt_fermion_hf(fham, hf, transform,
                                                           optimization_level=2,
                                                           repeat_opt=5,
                                                           f2q_kwargs=kwargs)
    beta, antibeta = betas(shifted_pham_2)
    print("\tShifted 2")
    print(f"\t\tnorm     = {shifted_pham_2.induced_norm(order=1)}")
    # print(shifted_pham_1.pretty_string(num_tabs=2))
    print(f"\t\tSI norm  = {beta}")
    print(f"\t\tASI norm = {antibeta}")
    print(f"\t\tshift_const = {const}")

