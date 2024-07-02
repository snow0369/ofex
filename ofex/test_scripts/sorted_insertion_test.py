from itertools import product
from typing import List

from openfermion import get_fermion_operator, normal_ordered, QubitOperator

from ofex.measurement.iterative_coefficient_splitting import init_ics
from ofex.measurement.iterative_coefficient_splitting.efficient_ics import efficient_ics
from ofex.measurement.sorted_insertion import sorted_insertion
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator
from ofex.utils.chem import molecule_example


def _grp_norm(list_operator: List[QubitOperator]):
    return sum([x.induced_norm(order=2) for x in list_operator])


def compare_si_eics_test():
    transform = "bravyi_kitaev"
    kwargs = {"n_qubits": None}
    mol_names = ["H2", "LiH", "BeH2", "H2O"]
    for mn, anticommute in product(mol_names, [False, True]):
        mol = molecule_example(mn)
        print(f"{mn}  {'LCU' if anticommute else 'FH'}")
        if "n_qubits" in kwargs:
            kwargs["n_qubits"] = mol.n_qubits
        fham = get_fermion_operator(mol.get_molecular_hamiltonian())
        fham = normal_ordered(fham)
        pham = fermion_to_qubit_operator(fham, transform, **kwargs)
        pham -= pham.constant
        pham.compress()

        si = sorted_insertion(pham, anticommute=anticommute)
        _, initial_grp, _ = init_ics(pham, anticommute, debug=False)
        eics, _ = efficient_ics(pham, initial_grp)
        print(f"\tQW  = {pham.induced_norm(order=1)}")
        print(f"\tSI  = {_grp_norm(si)}")
        print(f"\tEICS= {_grp_norm(eics)}")


if __name__ == "__main__":
    compare_si_eics_test()
