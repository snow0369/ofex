"""
Module containing functions and utilities for manipulating quantum chemical states.

This module provides methods to compute Hartree Fock ground states, configuration interaction (CISD)
wavefunctions, and generate configuration state functions (CSFs) for quantum chemistry. It integrates
tools for handling fermionic and qubit states, as well as utilities for processing molecular data.
"""

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

__all__ = ["hf_ground", "cisd_ground", "csf_states"]


def hf_ground(mol: MolecularData,
              active_idx: Optional[List[int]] = None,
              fermion_to_qubit_map: Optional[str] = None,
              **kwargs) -> SparseStateDict:
    # TODO: access attributes of `mol` and find the ground state.
    """
    Generate the Hartree-Fock ground state as a sparse state dictionary.

    Args:
        mol: An instance of MolecularData which contains molecule information.
        active_idx: List of indices specifying the active orbitals. If None, all orbitals are active.
        fermion_to_qubit_map: Optional string specifying the mapping from fermions to qubits.
        **kwargs: Additional arguments for fermion to qubit mapping.

    Returns:
        SparseStateDict: A dictionary representing the Hartree-Fock ground state.
    """
    n_spinorb, n_electrons = mol.n_qubits, mol.n_electrons
    if active_idx is None:
        active_idx = list(range(n_spinorb))
    fock_vector = [1 if idx < n_electrons else 0 for idx in active_idx]
    state = dict({BinaryFockVector(fock_vector): 1.0})
    if fermion_to_qubit_map is not None:
        state = fermion_to_qubit_state(state, fermion_to_qubit_map, **kwargs)
    return state


def cisd_ground(mol: MolecularData) -> SparseStateDict:
    """
    Compute the CISD (Configuration Interaction with Single and Double excitations)
    ground state as a sparse state dictionary.

    Args:
        mol: An instance of MolecularData with molecule information.

    Returns:
        SparseStateDict: A dictionary representing the CISD ground state.
    """
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


def csf_states(n_orbital: int,
               n_electrons: int,
               multiplicity: int = 1,
               projected_spin: float = 0.0,
               n_open: Optional[int] = None,
               cs_excitation: int = 0,
               os_excitation: int = 0) -> List[SparseStateDict]:
    """
    Generate Configuration State Functions (CSFs) based on specified spin and electron constraints.
    Refer to Helgaker, T., et al. (2000). Spin in Second Quantization. In Molecular Electronic-Structure Theory.
    https://doi.org/10.1002/9781119019572.ch2

    Args:
        n_orbital: Number of spatial orbitals.
        n_electrons: Total number of electrons in the system.
        multiplicity: Spin multiplicity of the states (e.g., 1 for singlet, 3 for triplet).
        projected_spin: Projected value of the total spin (M_s), in half-integer steps.
        n_open: Number of open shells. If None, defaults to multiplicity - 1.
        cs_excitation: Number of excited closed shell electrons.
        os_excitation: Number of excited open shell electrons.

    Returns:
        List[SparseStateDict]: A list of sparse state dictionaries representing the CSFs.
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
    _, p_list, coeff = generate_coupling_coeff(total_spin=float(multiplicity - 1) / 2,
                                                    projected_spin=projected_spin,
                                                    n_open=n_open)
    # Convert P_list to fock vectors
    inc_p_list = [[p_list[i][j] - p_list[i][j - 1] if j > 0 else p_list[i][j] for j in range(n_open)]
                  for i in range(len(p_list))]

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
            for inc_p in inc_p_list:  # add dets in a csf
                f_vector = list()
                j = 0
                for i in range(n_orbital):
                    if i in frozen_indices or i in cse:
                        f_vector += [1, 1]
                        assert i not in op
                    elif i in op:
                        if inc_p[j] == 1:
                            f_vector += [1, 0]
                        elif inc_p[j] == -1:
                            f_vector += [0, 1]
                        else:
                            raise ValueError
                        j += 1
                    else:  # Virtual Orbital
                        f_vector += [0, 0]
                try:
                    assert sum(f_vector) == n_electrons
                except AssertionError as exc:
                    print(cse, op, f_vector, sum(f_vector))
                    raise exc
                f_vector_list.append(BinaryFockVector(f_vector))
            for coeff_set in coeff:
                states.append({f: c for f, c in zip(f_vector_list, coeff_set)})
    return states


def generate_coupling_coeff(total_spin: float, projected_spin: float, n_open: int) \
        -> Tuple[List[List[int]], List[List[int]], np.ndarray]:
    """
    Generate the coupling coefficients, T-vectors, and P-vectors for open shell electrons.
    Refer to Helgaker, T., et al. (2000). Spin in Second Quantization. In Molecular Electronic-Structure Theory.
    https://doi.org/10.1002/9781119019572.ch2

    Args:
        total_spin: The value of total spin (S), given as a non-negative half-integer multiple.
        projected_spin: The projected spin value (M), given as a half-integer multiple.
        n_open: The number of open shells in the system.

    Returns:
        Tuple:
            T_list (List[List[int]]): The list of doubled T-vectors representing spin couplings.
            P_list (List[List[int]]): The list of doubled P-vectors representing spin projections.
            coeff_mat (np.ndarray): A 2D array where each element is the coefficient of 
                                    a determinant in a given CSF.
    """

    def _genealogical_coeff(_s: int, _m: int, _tn: int, _sigma: int):
        _s, _m, _tn, _sigma = float(_s) / 2, float(_m) / 2, float(_tn) / 2, float(_sigma) / 2
        if np.isclose(_tn, 0.5):
            return np.sqrt(0.5 + _sigma * _m / _s)
        elif np.isclose(_tn, -0.5):
            inner = 0.5 - _sigma * _m / (_s + 1)
            if inner < 0.0:
                return 0.0
            else:
                return -2 * _sigma * np.sqrt(inner)
        else:
            raise ValueError(_tn)

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

    t_table = [{1: [[1, ]]}]  # [S(n) : [ T_vectors ] for n in n_open]
    p_table = [{1: [[1, ]], -1: [[-1, ]]}]  # [M(n) : [ T_vectors ] for n in n_open]
    for i in range(1, n_open):
        new_ts = dict()
        for s, t_list in t_table[i - 1].items():
            if s - 1 >= 0:
                if s - 1 not in new_ts:
                    new_ts[s - 1] = list()
                new_ts[s - 1] += [t_vec + [s - 1] for t_vec in t_list]
            if s + 1 not in new_ts:
                new_ts[s + 1] = list()
            new_ts[s + 1] += [t_vec + [s + 1] for t_vec in t_list]
        t_table.append(new_ts)

        new_ps = dict()
        for m, p_list in p_table[i - 1].items():
            if m - 1 not in new_ps:
                new_ps[m - 1] = list()
            new_ps[m - 1] += [p_vec + [m - 1] for p_vec in p_list]
            if m + 1 not in new_ps:
                new_ps[m + 1] = list()
            new_ps[m + 1] += [p_vec + [m + 1] for p_vec in p_list]
        p_table.append(new_ps)

    t_list = t_table[n_open - 1][total_spin]
    p_list = p_table[n_open - 1][projected_spin]
    coeff_mat = np.zeros((len(t_list), len(p_list)), dtype=float)
    for i, j in product(range(len(t_list)), range(len(p_list))):
        d = 1.0
        for k in range(n_open):
            d *= _genealogical_coeff(_s=t_list[i][k], _m=p_list[j][k],
                                     _tn=t_list[i][k] if k == 0 else (t_list[i][k] - t_list[i][k - 1]),
                                     _sigma=p_list[j][k] if k == 0 else (p_list[j][k] - p_list[j][k - 1]))
        coeff_mat[i, j] = d
    return t_list, p_list, coeff_mat
