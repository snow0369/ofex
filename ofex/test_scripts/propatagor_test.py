import numpy as np
from openfermion import get_sparse_operator
from tqdm import tqdm

from ofex.operators.qubit_operator_tools import normalize_by_lcu_norm
from ofex.propagator.exact import exact_rte, exact_ite
from ofex.test_scripts.random_object import random_qubit_operator
from ofex.utils.chem import molecule_example, run_driver
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator


def exact_prop_test():
    mol_name = "LiH"
    mol = molecule_example(mol_name)
    mol = run_driver(mol, run_fci=True, driver='psi4')

    fham = mol.get_molecular_hamiltonian()
    n_qubits = mol.n_qubits
    pham = fermion_to_qubit_operator(fham, 'jordan_wigner')
    const = pham.constant
    pham = pham - const

    mat = get_sparse_operator(pham, n_qubits=n_qubits)
    # print(mat.shape)
    mat = mat.toarray()
    d, u = np.linalg.eigh(mat)
    assert np.allclose(mat, u @ np.diag(d) @ u.T.conj())
    print(f"Lowest : {np.min(d) + const}, FCI = {mol.fci_energy} (Diff={np.min(d) + const - mol.fci_energy})")
    pham, norm = normalize_by_lcu_norm(pham, level=2)
    print(f"Spectral Norm : {np.max(np.abs(d))} Bound : {norm}")

    d = d / norm

    t = 1.0
    true_rte = u @ np.diag(np.exp(d * -1j * t)) @ u.T.conj()
    # true_propagator = scipy.linalg.expm(-1j * beta * mat)
    sparse_rte_op = exact_rte(pham, t, n_qubits, exact_sparse=False)
    exact_rte_op = exact_rte(pham, t, n_qubits, exact_sparse=True)
    print(f"diff_true_exact = {np.linalg.norm(exact_rte_op - true_rte, ord=2)}")
    print(f"diff_true_sparse = {np.linalg.norm(sparse_rte_op - true_rte, ord=2)}")

    beta = 1.0
    true_ite = u @ np.diag(np.exp(d * -beta)) @ u.T.conj()
    sparse_ite_op = exact_ite(pham, t, n_qubits, exact_sparse=False)
    exact_ite_op = exact_ite(pham, t, n_qubits, exact_sparse=True)
    print(f"diff_true_exact = {np.linalg.norm(exact_ite_op - true_ite, ord=2)}")
    print(f"diff_true_sparse = {np.linalg.norm(sparse_ite_op - true_ite, ord=2)}")


def single_prop_test():
    n_qubits = 6
    trial = 20
    t = 1.0
    beta = 1.0
    for _ in tqdm(range(trial)):
        op = random_qubit_operator(n_qubits, weight=1, hermitian=True)
        mat = get_sparse_operator(op, n_qubits=n_qubits).toarray()
        d, u = np.linalg.eigh(mat)
        assert np.allclose(mat, u @ np.diag(d) @ u.T.conj())

        true_rte = u @ np.diag(np.exp(d * -1j * t)) @ u.T.conj()
        rte_op = exact_rte(op, t, n_qubits, exact_sparse=True)
        diff_norm = np.linalg.norm(rte_op - true_rte, ord=2)
        assert np.isclose(diff_norm, 0.0), diff_norm

        true_ite = u @ np.diag(np.exp(d * -beta)) @ u.T.conj()
        ite_op = exact_ite(op, t, n_qubits, exact_sparse=True)
        diff_norm = np.linalg.norm(ite_op - true_ite, ord=2)
        assert np.isclose(diff_norm, 0.0), diff_norm


if __name__ == '__main__':
    # exact_prop_test()
    single_prop_test()
