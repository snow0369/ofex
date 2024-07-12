from openfermion import MolecularData, get_fermion_operator, jordan_wigner, normal_ordered

from ofex.linalg.sparse_tools import expectation
from ofex.operators.symbolic_operator_tools import operator
from ofex.state.chem_ref_state import cisd_ground
from ofex.state.state_tools import pretty_print_state

if __name__ == "__main__":
    diatomic_bond_length = 1.45
    geometry = [("Li", (0., 0., 0.)), ("H", (0., 0., diatomic_bond_length))]
    basis = 'sto-3g'
    multiplicity = 1

    molecule = MolecularData(geometry, basis, multiplicity, description=str(diatomic_bond_length))
    molecule.load()

    print(molecule.name)

    # active_space_start = 1
    # active_space_stop = 3
    occupied_indices = None  # list(range(active_space_start))
    active_indices = None  # list(range(active_space_start, active_space_stop))

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

    cisd_state = cisd_ground(molecule)
    print(pretty_print_state(cisd_state))
    cisd_energy_true = molecule.cisd_energy
    cisd_energy = expectation(qubit_hamiltonian, cisd_state)
    print(molecule.hf_energy)
    print(cisd_energy_true)
    print(cisd_energy)
