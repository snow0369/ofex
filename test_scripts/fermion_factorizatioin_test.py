from openfermion import get_fermion_operator, normal_ordered

from ofex.operators.symbolic_operator_tools import compare_operators
from ofex.transforms.fermion_factorization import ham_to_ei_spin, ei_to_ham_spin, ham_to_ei_spatial, ei_to_ham_spatial, \
    double_factorization, number_factorization_to_reflection, double_factorization_to_hamiltonian
from ofex.utils.chem import molecule_example


def extract_ei_test(spin):
    mol_names = ["H2", "LiH", "BeH2", "H2O"]
    for mn in mol_names:
        mol = molecule_example(mn)
        print(mn)
        n_spinorb = mol.n_qubits
        fham = get_fermion_operator(mol.get_molecular_hamiltonian())
        fham = normal_ordered(fham)
        if spin:
            oei, tei, const = ham_to_ei_spin(fham, n_spinorb)
            _fham = ei_to_ham_spin(oei, tei, const)
        else:
            oei, tei, const = ham_to_ei_spatial(fham, n_spinorb)
            _fham = ei_to_ham_spatial(oei, tei, const)
        try:
            assert fham == _fham
        except AssertionError as e:
            print(compare_operators(fham, _fham))
            raise e
        print("passed")


def low_rank_factorization_test(spin):
    mol_names = ["H2", "LiH", "BeH2", "H2O"]
    atol = 1e-6
    for mn in mol_names:
        mol = molecule_example(mn)
        print(mn)
        n_spinorb = mol.n_qubits
        fham = get_fermion_operator(mol.get_molecular_hamiltonian())
        fham = normal_ordered(fham)
        fham = fham - fham.constant
        if spin:
            oei, tei, const = ham_to_ei_spin(fham, n_spinorb)
        else:
            oei, tei, const = ham_to_ei_spatial(fham, n_spinorb)
        h_list, u_list, const_out = double_factorization(oei, tei, reflection=False)
        _fham1 = double_factorization_to_hamiltonian(h_list, u_list, is_spin=spin, is_reflection=False)
        try:
            assert fham.isclose(_fham1, tol=atol)
        except AssertionError as e:
            print(compare_operators(fham, _fham1, atol=atol))
            raise e
        print(f"\tPassed double factorization (num, is_spin_orb={spin})")
        h_list_ref, u_list_ref, const_ref = number_factorization_to_reflection(h_list, u_list, is_spin=spin)
        _fham2 = double_factorization_to_hamiltonian(h_list_ref, u_list_ref, is_spin=spin, is_reflection=True) \
                 + const_ref
        try:
            assert fham.isclose(_fham2, tol=atol)
        except AssertionError as e:
            print(compare_operators(fham, _fham2, atol=atol))
            raise e
        print(f"\tPassed double factorization (ref, is_spin_orb={spin})")


if __name__ == '__main__':
    extract_ei_test(spin=False)  # Passed
    extract_ei_test(spin=True)  # Passed
    low_rank_factorization_test(spin=True)  # Passed
    low_rank_factorization_test(spin=False)  # Passed
