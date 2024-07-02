from openfermion import MolecularData, get_fermion_operator, jordan_wigner, normal_ordered

from ofex.operators.symbolic_operator_tools import operator

if __name__ == "__main__":
    diatomic_bond_length = 1.45
    geometry = [("Li", (0., 0., 0.)), ("H", (0., 0., diatomic_bond_length))]
    basis = 'sto-3g'
    multiplicity = 1

    molecule = MolecularData(geometry, basis, multiplicity, description=str(diatomic_bond_length))
    molecule.load()

    active_space_start = 1
    active_space_stop = 3
    occupied_indices = list(range(active_space_start))
    active_indices = list(range(active_space_start, active_space_stop))

    molecular_hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices, active_indices)
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()

    fermion_hamiltonian = normal_ordered(fermion_hamiltonian)
    print(fermion_hamiltonian.is_normal_ordered())

    for f_term in fermion_hamiltonian.get_operators():
        for idx, dag in operator(f_term):
            print(f"a{idx}"+("†" if dag else ""), end=' ')
        print('')

    print(qubit_hamiltonian)
    print(fermion_hamiltonian)
