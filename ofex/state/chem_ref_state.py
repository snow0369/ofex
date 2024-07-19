from itertools import product, combinations
from typing import List, Tuple, Optional

import numpy as np
from openfermion import MolecularData, FermionOperator, jordan_wigner
from openfermion.config import EQ_TOLERANCE

from ofex.linalg.sparse_tools import sparse_apply_operator
from ofex.operators.fermion_operator_tools import one_body_excitation
from ofex.state.binary_fock import BinaryFockVector
from ofex.state.state_tools import pretty_print_state, norm
from ofex.state.types import SparseStateDict
from ofex.transforms.fermion_qubit import fermion_to_qubit_state
from ofex.utils.chem import run_driver


def hf_ground(mol: MolecularData,
              active_idx: Optional[List[int]] = None,
              fermion_to_qubit_map: Optional[str] = None,
              **kwargs) -> SparseStateDict:
    n_spinorb, n_electrons = mol.n_qubits, mol.n_electrons
    if active_idx is None:
        active_idx = list(range(n_spinorb))
    fock_vector = [1 if idx < n_electrons else 0 for idx in active_idx]
    state = dict({BinaryFockVector(fock_vector): 1.0})
    if fermion_to_qubit_map is not None:
        state = fermion_to_qubit_state(state, fermion_to_qubit_map, **kwargs)
    return state


def cisd_ground(mol: MolecularData, debug=False):
    mol.load()
    if 'cisd' not in mol._pyscf_data or mol._pyscf_data['cisd'] is None:
        mol = run_driver(mol, run_cisd=True, driver='pyscf')
    cisd_data = mol._pyscf_data['cisd']
    c0, c1, c2 = cisd_data.cisdvec_to_amplitudes(cisd_data.ci)
    nocc, nvirt = c1.shape
    occ, virt = list(range(nocc)), list(range(nocc, nocc + nvirt))

    fermion_ex = FermionOperator.identity() * c0
    for (ai, a), (ri, r) in product(enumerate(occ), enumerate(virt)):
        if abs(c1[ai, ri]) > EQ_TOLERANCE:
            fermion_ex += one_body_excitation(r, a, spin_idx=False, hermitian=False) * c1[ai, ri]
    for (ai, a), (bi, b), (ri, r), (si, s) in product(enumerate(occ), enumerate(occ),
                                                      enumerate(virt), enumerate(virt)):
        if abs(c2[ai, bi, ri, si]) > EQ_TOLERANCE:
            two_ex = one_body_excitation(r, a, spin_idx=False, hermitian=False) * \
                     one_body_excitation(s, b, spin_idx=False, hermitian=False)
            if b == r:
                two_ex -= one_body_excitation(s, a, spin_idx=False, hermitian=False)
            fermion_ex += two_ex * 0.5 * c2[ai, bi, ri, si]

    pauli_ex = jordan_wigner(fermion_ex)
    cisd_state = sparse_apply_operator(pauli_ex, hf_ground(mol))

    assert np.isclose(norm(cisd_state), 1.0)
    return cisd_state


def generate_csf(n_orbital: int,
                 n_electrons: int,
                 multiplicity: int = 1,
                 projected_spin: float = 0.0,
                 n_open: Optional[int] = None,
                 cs_excitation: int = 0,
                 os_excitation: int = 0) -> List[SparseStateDict]:
    """
    Generate configuration state functions.

    Args:
        n_orbital: Number of spatial orbitals
        n_electrons: Number of total electrons
        multiplicity: Multiplicity (integer >=1)
        projected_spin: Value of M, integer multiplication of +-1/2
        n_open: Number of open shell
        cs_excitation: Number of excited closed shells
        os_excitation: Number of excited open shells

    Returns:
        csf_states: 3D integer np array. csf_state[i,j] is the Fock vector of determinant j in the csf i.
        coeff: 2D float np array. coeff[i % N,j] is the coefficient of csf_state[i, j] where N = coeff.shape[0]

    """
    # return : List[List[List[int]]]
    #  First qubit = lowest energy
    if (n_electrons - n_open) % 2 != 0:
        raise ValueError
    if n_open is None:
        n_open = multiplicity - 1
    if n_open + 2 * cs_excitation > n_electrons:
        raise ValueError
    if n_open < os_excitation:
        raise ValueError
    T_list, P_list, coeff = generate_coupling_coeff(total_spin=float(multiplicity - 1) / 2,
                                                    projected_spin=projected_spin,
                                                    n_open=n_open)
    # Convert P_list to fock vectors
    inc_P_list = [[P_list[i][j] - P_list[i][j - 1] if j > 0 else P_list[i][j] for j in range(n_open)]
                  for i in range(len(P_list))]

    states = list()
    num_closed = (n_electrons - n_open) // 2
    num_frozen = num_closed - cs_excitation
    frozen_indices = set(range(num_frozen))
    active_indices = set(range(num_frozen, n_orbital))
    for cse in combinations(active_indices, cs_excitation):  # Sample from active_indices.
        cse = set(cse)  # indices of excited closed shell
        not_cs_indices = active_indices.difference(cse)
        for ose in combinations(not_cs_indices, os_excitation):
            ose = set(ose)  # indices of excited open shell
            not_os_indicies = not_cs_indices.difference(ose)
            op = set(sorted(list(not_os_indicies))[:n_open - os_excitation]).union(ose)  # indices of open shell
            f_vector_list = list()
            for inc_P in inc_P_list:  # add dets in a csf
                f_vector = list()
                j = 0
                for i in range(n_orbital):
                    if i in frozen_indices or i in cse:
                        f_vector += [1, 1]
                        assert i not in op
                    elif i in op:
                        if inc_P[j] == 1:
                            f_vector += [1, 0]
                        elif inc_P[j] == -1:
                            f_vector += [0, 1]
                        else:
                            raise ValueError
                        j += 1
                    else:  # Virtual Orbital
                        f_vector += [0, 0]
                try:
                    assert sum(f_vector) == n_electrons
                except AssertionError:
                    print(cse, op, f_vector, sum(f_vector))
                    raise AssertionError
                f_vector_list.append(BinaryFockVector(f_vector))
            for coeff_set in coeff:
                states.append({f: c for f, c in zip(f_vector_list, coeff_set)})
    return states


def generate_coupling_coeff(total_spin: float, projected_spin: float, n_open: int) \
        -> Tuple[List[List[int]], List[List[int]], np.ndarray]:
    """

    Args:
        total_spin: value of S, integer(>=0) multiple of 1/2
        projected_spin: value of M, integer(>=0) multiple of +-1/2
        n_open: number of opened shell

    Returns:
        T_list : the list of double of original T vector
        P_list : the list of double of original P vector
        coeff_mat: 2D float np array, coeff_mat[i,j] = the coefficient of determinant j in the csf i.

    """

    def _genealogical_coeff(S: int, M: int, tn: int, sigma: int):
        S, M, tn, sigma = float(S) / 2, float(M) / 2, float(tn) / 2, float(sigma) / 2
        if np.isclose(tn, 0.5):
            return np.sqrt(0.5 + sigma * M / S)
        elif np.isclose(tn, -0.5):
            inner = 0.5 - sigma * M / (S + 1)
            if inner < 0.0:
                return 0.0
            else:
                return -2 * sigma * np.sqrt(inner)
        else:
            raise ValueError(tn)

    # All S and M values are integers doubled from the original values
    total_spin = int(2 * total_spin)
    projected_spin = int(2 * projected_spin)

    if total_spin < 0:
        raise ValueError
    if total_spin > n_open:
        raise ValueError
    if abs(projected_spin) > total_spin:
        raise ValueError

    if n_open == 0:
        return [[]], [[]], np.ones((1, 1), dtype=float)

    T_table = [{1: [[1, ]]}]  # [S(n) : [ T_vectors ] for n in n_open]
    P_table = [{1: [[1, ]], -1: [[-1, ]]}]  # [M(n) : [ T_vectors ] for n in n_open]
    for i in range(1, n_open):
        new_Ts = dict()
        for s, T_list in T_table[i - 1].items():
            if s - 1 >= 0:
                if s - 1 not in new_Ts:
                    new_Ts[s - 1] = list()
                new_Ts[s - 1] += [T_vec + [s - 1] for T_vec in T_list]
            if s + 1 not in new_Ts:
                new_Ts[s + 1] = list()
            new_Ts[s + 1] += [T_vec + [s + 1] for T_vec in T_list]
        T_table.append(new_Ts)

        new_Ps = dict()
        for m, P_list in P_table[i - 1].items():
            if m - 1 not in new_Ps:
                new_Ps[m - 1] = list()
            new_Ps[m - 1] += [P_vec + [m - 1] for P_vec in P_list]
            if m + 1 not in new_Ps:
                new_Ps[m + 1] = list()
            new_Ps[m + 1] += [P_vec + [m + 1] for P_vec in P_list]
        P_table.append(new_Ps)

    T_list = T_table[n_open - 1][total_spin]
    P_list = P_table[n_open - 1][projected_spin]
    coeff_mat = np.zeros((len(T_list), len(P_list)), dtype=float)
    for i, j in product(range(len(T_list)), range(len(P_list))):
        d = 1.0
        for k in range(n_open):
            d *= _genealogical_coeff(S=T_list[i][k], M=P_list[j][k],
                                     tn=T_list[i][k] if k == 0 else (T_list[i][k] - T_list[i][k - 1]),
                                     sigma=P_list[j][k] if k == 0 else (P_list[j][k] - P_list[j][k - 1]))
        coeff_mat[i, j] = d
    return T_list, P_list, coeff_mat


if __name__ == "__main__":
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
        states = generate_csf(n_orbital=n_orbitals, n_electrons=n_electrons,
                              multiplicity=multiplicity, projected_spin=projected_spin,
                              n_open=n_open,
                              cs_excitation=cs_excitation,
                              os_excitation=os_excitation)
        for s in states:
            print(pretty_print_state(s))


    # _test_generate_coupling_coeff()
    _test_generate_reference_state()
