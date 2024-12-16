import numpy as np
from openfermion import get_fermion_operator

from ofex.linalg.sparse_tools import apply_operator
from ofex.measurement import sorted_insertion, fragment_variance
from ofex.measurement.iterative_coefficient_splitting import run_ics, init_ics
from ofex.propagator import exact_rte
from ofex.state.chem_ref_state import hf_ground
from ofex.state.state_tools import pretty_print_state
from ofex.transforms import fermion_to_qubit_operator, fermion_to_qubit_state
from ofex.utils.chem import molecule_example


def ics_test(mol_name, transform, anticommute, debug):
    mol = molecule_example(mol_name)
    fham = mol.get_molecular_hamiltonian()
    fham = get_fermion_operator(fham)
    n_qubits = mol.n_qubits
    if transform == "jordan_wigner":
        kwargs = dict()
    elif transform == "bravyi_kitaev":
        kwargs = {'n_qubits': mol.n_qubits}
    elif transform == 'symmetry_conserving_bravyi_kitaev':
        kwargs = {'active_fermions': mol.n_electrons,
                  'active_orbitals': mol.n_qubits}
        n_qubits -= 2
    else:
        raise AssertionError

    pham = fermion_to_qubit_operator(fham, transform, **kwargs)
    pham = pham - pham.constant

    norm = sum([u.induced_norm(order=2) for u in sorted_insertion(pham, anticommute=True)])
    prop = exact_rte(pham, t=6 * np.pi / norm, n_qubits=n_qubits)

    ref1 = fermion_to_qubit_state(hf_ground(mol), transform, **kwargs)
    print(prop.shape)
    print(pretty_print_state(ref1))
    ref2 = apply_operator(prop, ref1)

    initial_grp, cov_dict = init_ics(pham, ref1, ref2,
                                     num_workers=15,
                                     anticommute=anticommute, debug=debug)

    grp_ham, shots, final_var, c_opt = run_ics(pham,
                                               initial_grp,
                                               cov_dict,
                                               transition=True,
                                               conv_atol=1e-6,
                                               debug=debug)
    print(final_var)
    print("== ICS VAR ==")
    ev_var = fragment_variance(grp_ham, ref1, ref2, shots)
    print(ev_var)
    print(np.sum(shots))

    print("== SI  VAR ==")
    si_group = sorted_insertion(pham, anticommute)
    m_si_real = np.array([x.induced_norm(order=2) for x in si_group])
    m_si_imag = np.array(m_si_real)
    div = sum(m_si_real) + sum(m_si_imag)
    m_si_real, m_si_imag = m_si_real / div, m_si_imag / div
    ev_var_si = fragment_variance(si_group, ref1, ref2, shots)
    print(ev_var_si)
    print(sum(m_si_real) + sum(m_si_imag))
