from ofex.state.chem_ref_state import generate_coupling_coeff, csf_states
from ofex.state.state_tools import pretty_print_state


def _test_generate_coupling_coeff():
    T_list, P_list, coeff_mat = generate_coupling_coeff(total_spin=1,
                                                        projected_spin=0,
                                                        n_open=4)
    print(T_list)
    print(P_list)
    print(coeff_mat)


def _test_generate_reference_state():
    n_orbitals = 6
    n_electrons = 4
    multiplicity = 3
    projected_spin = 0.0
    n_open = 2
    cs_excitation = 1
    os_excitation = 2
    states = csf_states(n_orbital=n_orbitals, n_electrons=n_electrons,
                        multiplicity=multiplicity, projected_spin=projected_spin,
                        n_open=n_open,
                        cs_excitation=cs_excitation,
                        os_excitation=os_excitation)
    for s in states:
        print(pretty_print_state(s))
