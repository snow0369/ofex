import numpy as np
from openfermion import get_fermion_operator, normal_ordered, QubitOperator, get_linear_qubit_operator_diagonal, \
    get_sparse_operator

from ofex.clifford.clifford_tools import tableau_to_pauli
from ofex.clifford.pauli_diagonalization import diagonalizing_clifford
from ofex.clifford.simulation import clifford_simulation
from ofex.measurement.sorted_insertion import sorted_insertion
from ofex.operators.symbolic_operator_tools import clean_imaginary
from ofex.state.state_tools import pretty_print_state
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator
from ofex.utils.chem import molecule_example


def pauli_diagonlization_test():
    mol_names = ["H2", "H4"]  # "LiH", "BeH2", "H2O"]
    for mn in mol_names:
        mol = molecule_example(mn)
        print(mn)
        num_qubits = mol.n_qubits
        fham = get_fermion_operator(mol.get_molecular_hamiltonian())
        fham = normal_ordered(fham)
        qham = fermion_to_qubit_operator(fham, "bravyi_kitaev", n_qubits=num_qubits)
        frag = sorted_insertion(qham)

        for idx_frag, f in enumerate(frag):
            print(f"idx_frag = {idx_frag}")
            f_mat = get_sparse_operator(f).toarray()
            f_ev, f_vec = np.linalg.eigh(f_mat)
            f_argsort = np.argsort(f_ev)
            f_ev, f_vec = f_ev[f_argsort], f_vec[:, f_argsort]
            a_mat, a_coeff, clif_hist = diagonalizing_clifford(f, num_qubits)
            op_pauli = tableau_to_pauli(a_mat, None, a_coeff)
            op_pauli = QubitOperator.accumulate(op_pauli)
            op_pauli = clean_imaginary(op_pauli)
            op_pauli = get_linear_qubit_operator_diagonal(op_pauli, n_qubits=num_qubits)
            diag_pauli = np.diag(op_pauli)
            assert np.allclose(np.sort(op_pauli), np.sort(f_ev)), (op_pauli, f_ev)
            print(f)
            print(clif_hist)
            for idx in range(len(f_ev)):
                print(f"idx_ev = {idx}")
                v = f_vec[:, idx]
                clif_v = clifford_simulation(v, clif_hist, inv=False)
                applied_clif_v = diag_pauli @ clif_v
                # clif_v = to_sparse_dict(clif_v)
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


if __name__ == "__main__":
    pauli_diagonlization_test()
