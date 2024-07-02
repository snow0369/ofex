import json
from time import time

import numpy as np
from matplotlib import pyplot as plt

from ofex.operators.qubit_operator_tools import normalize_by_lcu_norm
from ofex.propagator.exact import exact_rte, exact_ite
from ofex.propagator.trotter import trotter_rte_by_si_lcu, trotter_ite_by_si_lcu, trotter_rte_by_single_pauli, \
    trotter_rte_by_si_comm, trotter_ite_by_si_comm, trotter_ite_by_single_pauli
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator
from ofex.utils.chem import molecule_example


def trotter_comparison():
    mol_names = ["H2", "H4", "LiH"]
    max_trotter = 10
    t, beta = 1.0, 1.0
    exact_sparse = False

    results = dict()

    for mol_name in mol_names:
        print(f"mol_name = {mol_name}")
        results[mol_name] = dict()
        mol = molecule_example(mol_name)

        fham = mol.get_molecular_hamiltonian()
        n_qubits = mol.n_qubits
        pham = fermion_to_qubit_operator(fham, 'jordan_wigner')
        const = pham.constant
        pham = pham - const
        pham, norm = normalize_by_lcu_norm(pham, level=1)

        rte_exact = exact_rte(pham, t, n_qubits, exact_sparse)
        ite_exact = exact_ite(pham, beta, n_qubits, exact_sparse)
        results[mol_name]['time'] = {'rte_lcu': dict(), 'rte_comm': dict(), 'rte_pauli': dict(),
                                     'ite_lcu': dict(), 'ite_comm': dict(), 'ite_pauli': dict()}
        results[mol_name]['norm'] = {'rte_lcu': dict(), 'rte_comm': dict(), 'rte_pauli': dict(),
                                     'ite_lcu': dict(), 'ite_comm': dict(), 'ite_pauli': dict()}

        for n_trotter in range(1, max_trotter + 1):
            print(f"n_trotter = {n_trotter} / {max_trotter}")
            tm = time()
            rte_trot_lcu = trotter_rte_by_si_lcu(pham, t, n_qubits, n_trotter, exact_sparse)
            tm_rte_trot_lcu = time() - tm
            diff_rte_lcu = np.linalg.norm((rte_exact - rte_trot_lcu).toarray(), ord=2)
            print(f"\tdiff_rte_lcu   = {diff_rte_lcu} (time = {tm_rte_trot_lcu})")

            tm = time()
            rte_trot_comm = trotter_rte_by_si_comm(pham, t, n_qubits, n_trotter, exact_sparse)
            tm_rte_trot_comm = time() - tm
            diff_rte_comm = np.linalg.norm((rte_exact - rte_trot_comm).toarray(), ord=2)
            print(f"\tdiff_rte_comm  = {diff_rte_comm} (time = {tm_rte_trot_comm})")

            tm = time()
            rte_trot_pauli = trotter_rte_by_single_pauli(pham, t, n_qubits, n_trotter, exact_sparse)
            tm_rte_trot_pauli = time() - tm
            diff_rte_pauli = np.linalg.norm((rte_exact - rte_trot_pauli).toarray(), ord=2)
            print(f"\tdiff_rte_pauli = {diff_rte_pauli} (time = {tm_rte_trot_pauli})")

            tm = time()
            ite_trot_lcu = trotter_ite_by_si_lcu(pham, beta, n_qubits, n_trotter, exact_sparse)
            tm_ite_trot_lcu = time() - tm
            diff_ite_lcu = np.linalg.norm((ite_exact - ite_trot_lcu).toarray(), ord=2)
            print(f"\tdiff_ite_lcu   = {diff_ite_lcu} (time = {tm_ite_trot_lcu})")

            tm = time()
            ite_trot_comm = trotter_ite_by_si_comm(pham, beta, n_qubits, n_trotter, exact_sparse)
            tm_ite_trot_comm = time() - tm
            diff_ite_comm = np.linalg.norm((ite_exact - ite_trot_comm).toarray(), ord=2)
            print(f"\tdiff_ite_comm  = {diff_ite_comm} (time = {tm_ite_trot_comm})")

            tm = time()
            ite_trot_pauli = trotter_ite_by_single_pauli(pham, beta, n_qubits, n_trotter, exact_sparse)
            tm_ite_trot_pauli = time() - tm
            diff_ite_pauli = np.linalg.norm((ite_exact - ite_trot_pauli).toarray(), ord=2)
            print(f"\tdiff_ite_pauli = {diff_ite_pauli} (time = {tm_ite_trot_pauli})")

            results[mol_name]['norm']['rte_lcu'][n_trotter] = diff_rte_lcu
            results[mol_name]['norm']['rte_comm'][n_trotter] = diff_rte_comm
            results[mol_name]['norm']['rte_pauli'][n_trotter] = diff_rte_pauli
            results[mol_name]['norm']['ite_lcu'][n_trotter] = diff_ite_lcu
            results[mol_name]['norm']['ite_comm'][n_trotter] = diff_ite_comm
            results[mol_name]['norm']['ite_pauli'][n_trotter] = diff_ite_pauli

            results[mol_name]['time']['rte_lcu'][n_trotter] = tm_rte_trot_lcu
            results[mol_name]['time']['rte_comm'][n_trotter] = tm_rte_trot_comm
            results[mol_name]['time']['rte_pauli'][n_trotter] = tm_rte_trot_pauli
            results[mol_name]['time']['ite_lcu'][n_trotter] = tm_ite_trot_lcu
            results[mol_name]['time']['ite_comm'][n_trotter] = tm_ite_trot_comm
            results[mol_name]['time']['ite_pauli'][n_trotter] = tm_ite_trot_pauli

    with open('trotter_comparison.json', 'w') as outfile:
        json.dump(results, outfile)


def draw_comparison(draw_what='norm'):
    with open("trotter_comparison.json", "r") as infile:
        results = json.load(infile)
    mol_names = list(results.keys())

    colors = {'rte_lcu': 'r', 'rte_comm': 'g', 'rte_pauli': 'b',
              'ite_lcu': 'r', 'ite_comm': 'g', 'ite_pauli': 'b'}
    n_rows = int(np.ceil(np.sqrt(len(mol_names))))
    n_cols = n_rows

    fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_cols * 6, n_rows * 6))
    fig.suptitle(draw_what)
    for idx_plt, mol_name in enumerate(mol_names):
        if len(mol_names) == 1:
            ax = axes
        else:
            ax = axes.flat[idx_plt]
        for label, item in results[mol_name][draw_what].items():
            x_list = list(item.keys())
            y_list = [item[x] for x in x_list]
            ax.plot(x_list, y_list, label=label, color=colors[label], linestyle='dashed' if 'ite' in label else 'solid')
        ax.set_title(mol_name)
        if idx_plt == 0:
            ax.legend()
    plt.show()


if __name__ == '__main__':
    # trotter_comparison()
    draw_comparison('norm')
    draw_comparison('time')
