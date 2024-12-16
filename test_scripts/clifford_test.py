import galois
import numpy as np
import pytest
from openfermion import get_fermion_operator, normal_ordered, QubitOperator, get_linear_qubit_operator_diagonal, \
    get_sparse_operator

from ofex.clifford import pauli_to_tableau, tableau_to_pauli, diagonalizing_clifford, clifford_simulation, dot_tableau, \
    is_zero_gf, is_equal_gf, clifford_unitary_mat
from ofex.measurement.pauli_grouping import sorted_insertion
from ofex.operators.symbolic_operator_tools import clean_imaginary, compare_operators
from ofex.state.state_tools import pretty_print_state, get_num_qubits
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator
from ofex.utils.chem import molecule_example

gf = galois.GF(2)

@pytest.mark.parametrize("pham, num_qubits", [
    (QubitOperator("X0 Z1"), 2),
    (QubitOperator("Y0 Z2 X3", 0.5), 4),
    (QubitOperator("X0 Z1") + QubitOperator("Y0"), 2),
    (QubitOperator("X0 Y1", 0.7) + QubitOperator("Z2 X3", -0.3), 4),
    (QubitOperator("Z0") + QubitOperator("X1 Y2", -0.5) + QubitOperator("Z3 X4", 0.3), 5),
    (QubitOperator("", 1.0), 1),
])
def test_pauli_tableau(pham: QubitOperator, num_qubits):
    tableau, coeff_arr = pauli_to_tableau(pham, num_qubits)
    recovered_pham = tableau_to_pauli(tableau, coeff_arr)
    recovered_pham = QubitOperator.accumulate(recovered_pham)
    assert pham == recovered_pham, (f"Original and recovered QubitOperators differ: \n"
                                    f"{compare_operators(pham, recovered_pham, atol=1e-6)}")
    pham_list = [QubitOperator(*p) for p in pham.terms.items()]
    tableau, coeff_arr = pauli_to_tableau(pham_list, num_qubits)
    recovered_pham = tableau_to_pauli(tableau, coeff_arr)
    recovered_pham = QubitOperator.accumulate(recovered_pham)
    assert pham == recovered_pham, (f"Original and recovered QubitOperators differ for pham_list: \n"
                                    f"{compare_operators(pham, recovered_pham, atol=1e-6)}")


@pytest.mark.parametrize("tab1, tab2, expected_result", [
    (gf([0, 1]), gf([1, 0]), 1),
    (gf([0, 0]), gf([0, 0]), 0),
    (gf([1, 1]), gf([1, 1]), 0),
    (gf([1, 0, 1, 1]), gf([0, 1, 1, 0]), 0),
    (gf([1, 0, 0, 1]), gf([1, 1, 0, 0]), 1),
    (gf([1, 0, 0, 1]), gf([0, 1, 0, 0]), 1),
    (gf([1, 1, 1, 0]), gf([0, 1, 0, 0]), 0),
])
def test_dot_tableau(tab1, tab2, expected_result):
    result = dot_tableau(tab1, tab2)  # Replace with actual function logic
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

@pytest.mark.parametrize("gf_element, expected", [
    (gf([0, 0]), True),
    (gf([[0, 0], [0, 0]]), True),
    (gf([[1, 0], [0, 1]]), False),
])
def test_is_zero_gf(gf_element, expected):
    result = is_zero_gf(gf_element)  # Replace with actual function logic
    assert result == expected, f"Expected {expected}, but got {result}"

@pytest.mark.parametrize("gf_element1, gf_element2, expected", [
    (gf([1, 0]), gf([1, 0]), True),
    (gf([1, 0]), gf([0, 1]), False),
])
def test_is_equal_gf(gf_element1, gf_element2, expected):
    result = is_equal_gf(gf_element1, gf_element2)  # Replace with actual function logic
    assert result == expected, f"Expected {expected}, but got {result}"

@pytest.mark.parametrize("mn", ["H2", "H4"])
def test_clifford_diagonalization_simulation(mn:str):
    mol = molecule_example(mn)
    num_qubits = mol.n_qubits
    fham = get_fermion_operator(mol.get_molecular_hamiltonian())
    fham = normal_ordered(fham)
    qham = fermion_to_qubit_operator(fham, "bravyi_kitaev", n_qubits=num_qubits)
    qham -= qham.constant
    frag = sorted_insertion(qham)

    for idx_frag, f in enumerate(frag):
        f_mat = get_sparse_operator(f).toarray()
        f_ev, f_vec = np.linalg.eigh(f_mat)
        f_argsort = np.argsort(f_ev)
        f_ev, f_vec = f_ev[f_argsort], f_vec[:, f_argsort]

        a_mat, a_coeff, clif_hist = diagonalizing_clifford(f, num_qubits)

        op_pauli = tableau_to_pauli(a_mat, a_coeff, None)
        op_pauli = QubitOperator.accumulate(op_pauli)
        op_pauli = clean_imaginary(op_pauli)
        op_pauli = get_linear_qubit_operator_diagonal(op_pauli, n_qubits=num_qubits)
        diag_pauli = np.diag(op_pauli)

        assert np.allclose(np.sort(op_pauli), np.sort(f_ev)), (op_pauli, f_ev)
        for idx in range(len(f_ev)):
            v = f_vec[:, idx]
            clif_v = clifford_simulation(v, clif_hist, inv=False)
            clif_u = clifford_unitary_mat(clif_hist, num_qubits, inv=False)
            assert np.allclose(clif_u @ v, clif_v)
            applied_clif_v = diag_pauli @ clif_v
            try:
                assert np.allclose(applied_clif_v, f_ev[idx] * clif_v)
            except AssertionError as e:
                print(f"Delta = {np.linalg.norm(applied_clif_v - f_ev[idx] * clif_v)}")
                print(np.diag(diag_pauli))
                print("clif_v")
                print(pretty_print_state(clif_v))
                print('applied_clif_v')
                print(pretty_print_state(applied_clif_v))
                print(f_ev[idx])
                raise e
