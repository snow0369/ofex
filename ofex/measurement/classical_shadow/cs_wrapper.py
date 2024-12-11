import os
import subprocess

import numpy as np
from openfermion import QubitOperator, get_fermion_operator

from ofex.transforms.fermion_qubit import fermion_to_qubit_operator
from ofex.utils.chem import molecule_example, run_driver

PATH_CS = "./predicting-quantum-properties/"
PATH_DAQ = os.path.join(PATH_CS, "data_acquisition_shadow")
PATH_GEN = os.path.join(PATH_CS, "generate_observables")
PATH_PRE = os.path.join(PATH_CS, "prediction_shadow")


def classical_shadow(n_meas, n_qubits):
    # proc_daq = subprocess.Popen([PATH_DAQ, '-r', str(n_meas), str(n_qubits)])
    # proc_daq.wait()
    raise NotImplementedError


def derandomized_measurement(pham: QubitOperator, n_meas_per_obs, n_qubits):
    OBSERVABLE_FILE = os.path.join(PATH_CS, "pauli_observables.txt")
    SCHEME_FILE = os.path.join(PATH_CS, "pauli_scheme.txt")
    MEAS_FILE = os.path.join(PATH_CS, "pauli_meas.txt")

    weight_order = 1

    # Generate observable file
    pham = pham / pham.induced_norm(order=weight_order)
    with open(OBSERVABLE_FILE, "w") as f:
        f.write(f"{n_qubits}\n")
        for op, coeff in pham.terms.items():
            op_str = [f"{x[1].upper()} {x[0]}" for x in op if x[1].upper() != "I"]
            locality = len(op_str)
            if locality > 0:
                op_str = " ".join(op_str)
                f.write(f"{locality} {op_str} {abs(coeff) ** weight_order:.10f}\n")

    command = [PATH_DAQ, '-d', str(n_meas_per_obs), OBSERVABLE_FILE]
    with open(SCHEME_FILE, "w") as of, open(MEAS_FILE, "w") as ef:
        proc_daq = subprocess.Popen(command, stdout=of, stderr=ef)
    proc_daq.wait()


if __name__ == "__main__":
    mol_name = "H2O"
    transform = "symmetry_conserving_bravyi_kitaev"
    n_meas_per_obs = 1000

    mol = molecule_example(mol_name)
    mol.load()
    mol = run_driver(mol, run_cisd=True, run_fci=True, driver="pyscf")

    if transform == "bravyi_kitaev":
        f2q_kwargs = {"n_qubits": mol.n_qubits}
        n_qubits = mol.n_qubits
    elif transform == "symmetry_conserving_bravyi_kitaev":
        f2q_kwargs = {"active_fermions": mol.n_electrons,
                      "active_orbitals": mol.n_qubits}
        n_qubits = mol.n_qubits - 2
    elif transform == "jordan_wigner":
        f2q_kwargs = dict()
        n_qubits = mol.n_qubits
    else:
        raise NotImplementedError

    fham = mol.get_molecular_hamiltonian()
    fham = get_fermion_operator(fham)
    pham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
    p_const = pham.constant
    assert np.isclose(p_const.imag, 0.0)
    p_const = p_const.real
    pham = pham - p_const

    derandomized_measurement(pham, n_meas_per_obs, n_qubits)
