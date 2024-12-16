"""
This module provides functions and tools for manipulating and factorizing fermionic Hamiltonians.

The module contains utilities to convert fermion Hamiltonians into various representations,
perform double factorization of electron integrals, and handle reflection or number operator
transformations. It also includes helper functions for one-body and two-body terms, eigenvalue 
decomposition of tensors, and verification of results.

Core functionalities include:
1. Hamiltonian conversion to/from electron integral tensors in spatial and spin orbital formats.
2. Double factorization of electron integrals with optional reflection operators.
3. Construction of Hamiltonians from factorized forms.
4. Calculation of norms for reflection-based factorizations.

Common operations performed by the module involve manipulating electron integrals, normal-ordering
fermion operators, and expressing Hamiltonians in highly factorized forms.

Available functions:
- ham_to_ei_spatial, ei_to_ham_spatial
- ham_to_ei_spin, ei_to_ham_spin
- double_factorization
- number_factorization_to_reflection
- double_factorization_to_hamiltonian
- calculate_reflect_norm
"""
from collections import Counter
from functools import partial
from itertools import product
from typing import Tuple, List

import numpy as np
from openfermion import FermionOperator, normal_ordered
from openfermion.config import EQ_TOLERANCE
from scipy.linalg import eigh

from ofex.operators.fermion_operator_tools import cre_ann, one_body_excitation, one_body_number, one_body_reflection, \
    two_body_reflection
from ofex.operators.symbolic_operator_tools import dict_to_operator
from ofex.operators.types import SingleFermion
from ofex.transforms.fermion_rotation import fermion_rotation_operator

__all__ = ["ham_to_ei_spatial", "ei_to_ham_spatial", "ham_to_ei_spin", "ei_to_ham_spin",
           "double_factorization", "number_factorization_to_reflection",
           "double_factorization_to_hamiltonian",
           "calculate_reflect_norm", "FermionFragment"]

FermionFragment = Tuple[FermionOperator, np.ndarray]


def _sorted_tuple(p: int, q: int, r: int, s: int) -> SingleFermion:
    """
    Generate a normal-ordered tuple of creation and annihilation operators.

    Args:
        p, q: Indices for creation operators.
        r, s: Indices for annihilation operators.

    Returns:
        A tuple of normal-ordered creation and annihilation operators,
        first sorted by index for creation operators, then for annihilation operators.
    """
    return tuple(sorted([(p, 1), (q, 1)], reverse=True, key=lambda x: x[0]) +
                 sorted([(r, 0), (s, 0)], reverse=True, key=lambda x: x[0]))


def _bring_all_two_spatial(p_s, q_s, r_s, s_s, twobody: FermionOperator, new_twobody=None):
    """
    Collect two-electron interaction terms for spatial orbitals from a FermionOperator.

    Args:
        p_s, q_s, r_s, s_s: Indices of the spatial orbitals.
        twobody: Original FermionOperator containing two-body terms.
        new_twobody: Partial dictionary of terms to append.

    Returns:
        Normal-ordered FermionOperator including relevant two-electron terms.
    """
    def spatial_twobody(_p_s, _q_s, _r_s, _s_s):
        _op_list = list()
        _op_list.append(_sorted_tuple(2 * _p_s, 2 * _q_s, 2 * _r_s, 2 * _s_s))
        _op_list.append(_sorted_tuple(2 * _p_s + 1, 2 * _q_s, 2 * _r_s + 1, 2 * _s_s))
        _op_list.append(_sorted_tuple(2 * _p_s, 2 * _q_s + 1, 2 * _r_s, 2 * _s_s + 1))
        _op_list.append(_sorted_tuple(2 * _p_s + 1, 2 * _q_s + 1, 2 * _r_s + 1, 2 * _s_s + 1))

        _op_list.append(_sorted_tuple(2 * _p_s, 2 * _q_s + 1, 2 * _r_s + 1, 2 * _s_s))
        _op_list.append(_sorted_tuple(2 * _p_s + 1, 2 * _q_s, 2 * _r_s, 2 * _s_s + 1))
        return _op_list

    op_list = list()
    op_list += spatial_twobody(p_s, q_s, r_s, s_s)  #
    op_list += spatial_twobody(r_s, q_s, p_s, s_s)  # p <-> r
    op_list += spatial_twobody(p_s, s_s, r_s, q_s)  # q <-> s
    op_list += spatial_twobody(r_s, s_s, p_s, q_s)  # p <-> r q <-> s
    op_list += spatial_twobody(p_s, r_s, q_s, s_s)  # q <-> r
    op_list += spatial_twobody(s_s, q_s, r_s, p_s)  # p <-> s

    if new_twobody is None:
        new_twobody = dict()
    else:
        new_twobody = dict(new_twobody)

    for op in op_list:
        if op in twobody.terms and op not in new_twobody:
            new_twobody[op] = twobody.terms[op]
    return normal_ordered(dict_to_operator(new_twobody, FermionOperator))


def _bring_all_two_spin(p, q, r, s, twobody: FermionOperator, new_twobody=None):
    """
    Collect two-electron interaction terms for spin orbitals from a FermionOperator.

    Args:
        p, q, r, s: Indices of the spin orbitals.
        twobody: Original FermionOperator containing two-body terms.
        new_twobody: Partial dictionary of terms to append.

    Returns:
        Normal-ordered FermionOperator including relevant two-electron terms.
    """
    op_list = [
        _sorted_tuple(p, q, r, s),
        _sorted_tuple(r, q, p, s),
        _sorted_tuple(p, s, r, q),
        _sorted_tuple(r, s, p, q),
        _sorted_tuple(p, r, q, s),
        _sorted_tuple(s, q, r, p)
    ]
    if new_twobody is None:
        new_twobody = dict()
    else:
        new_twobody = dict(new_twobody)

    for op in op_list:
        if op in twobody.terms and op not in new_twobody:
            new_twobody[op] = twobody.terms[op]
    return normal_ordered(dict_to_operator(new_twobody, FermionOperator))


def _one_body_excitation_spatial(idx1, idx2):
    """
    Generate a one-body excitation operator for spatial orbitals.

    Args:
        idx1: Index of the creation operator.
        idx2: Index of the annihilation operator.

    Returns:
        FermionOperator: One-body operator for the given indices.
    """
    if idx1 == idx2:
        return one_body_excitation(idx1, idx2, spin_idx=False, hermitian=False) / 2
    return one_body_excitation(idx1, idx2, spin_idx=False, hermitian=False)


def ham_to_ei_spatial(fham: FermionOperator,
                      n_spinorb: int)\
        -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Converts a Fermion Hamiltonian into spatial one-body and two-body electron integral tensors.

    This function processes the input Fermion Hamiltonian into spatial orbital tensors, separating the one-body
    and two-body electronic integrals as well as the constant energy contribution. It ensures normal ordering
    of the Hamiltonian terms and constructs the corresponding tensors for spatial orbitals.

    Notation:
        E_{pq} = a†_{p↑}a_{q↑} + a†_{p↓}a_{q↓}

        H = ∑_{pq} oei[p,q] E_{pq}
          + (1/2) ∑_{pqrs} tei[p,q,r,s] E_{pq} * E_{rs}
          + const

    Args:
        fham (FermionOperator): The input Fermion Hamiltonian.
        n_spinorb (int): The total number of spin orbitals (must be an even number).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - spatial_oei (ndarray): A 2D tensor with one-body electron integrals for spatial orbitals.
            - spatial_tei (ndarray): A 4D tensor with two-body electron integrals for spatial orbitals.
            - const (float): The constant term in the Hamiltonian.
    """
    spatial_oei = np.zeros((n_spinorb // 2, n_spinorb // 2), dtype=complex)
    const = 0.0
    two_body = dict()
    fham = normal_ordered(fham)

    # 1. Build one-body tensor and collect two-body terms
    for op, coeff in fham.terms.items():
        cre, ann = cre_ann(op)
        assert len(cre) == len(ann)
        if len(cre) == 1:
            c, a = cre[0], ann[0]
            assert c % 2 == a % 2
            c, a = c // 2, a // 2
            if np.isclose(spatial_oei[c, a], 0.0, atol=EQ_TOLERANCE):
                spatial_oei[c, a] = coeff
            else:
                assert np.isclose(spatial_oei[c, a], coeff)
        elif len(cre) == 2:
            two_body[op] = coeff
        elif len(cre) == 0:
            const = coeff
        else:
            raise ValueError("More than two body.")

    # 2. Build two-body tensor
    two_body = dict_to_operator(two_body, FermionOperator)  # Two-body only terms
    check_sum = FermionOperator()  # for verification
    added = list()  # added indices, to prevent duplication.
    spatial_tei = np.zeros((n_spinorb // 2, n_spinorb // 2, n_spinorb // 2, n_spinorb // 2), dtype=float)
    # Iterate over all two-body terms
    for op, coeff in two_body.terms.items():
        (p, q), (r, s) = cre_ann(op)
        p_s, q_s, r_s, s_s = p // 2, q // 2, r // 2, s // 2  # Index for spatial orbitals
        unique_key = tuple(sorted((p_s, q_s, r_s, s_s)))
        if unique_key in added:
            continue
        added.append(unique_key)

        # From the entire two-body operators, extract terms related to the current index.
        two_body_now = _bring_all_two_spatial(p_s, q_s, r_s, s_s, two_body)
        check_sum = check_sum + two_body_now  # Outer side verification
        # Classify the type of indices, based on the count for each orbital.
        # Example) (a,a,b,b) -> {a:2, b:2}
        idxs = Counter(unique_key)
        counts = sorted(list(idxs.values()))
        assert sum(counts) == 4

        if len(idxs) == 1:  # aaaa
            # Verify
            check_two_body = (_one_body_excitation_spatial(p_s, p_s) * _one_body_excitation_spatial(p_s, p_s)
                              - _one_body_excitation_spatial(p_s, p_s))
            check_two_body = normal_ordered(check_two_body * (coeff / 2))
            if two_body_now != check_two_body:
                coeff = -coeff
                check_two_body = -check_two_body
            assert two_body_now == check_two_body

            # Assign
            spatial_tei[p_s, p_s, p_s, p_s] = coeff
            spatial_oei[p_s, p_s] -= coeff / 2

        elif len(idxs) == 2 and counts[0] != counts[1]:  # aaab
            a, b = tuple(idxs.keys())
            if idxs[a] == 3 and idxs[b] == 1:
                pass
            elif idxs[a] == 1 and idxs[b] == 3:
                a, b = b, a
            else:
                raise AssertionError

            # Determine coefficient(s).
            check_two_body = _one_body_excitation_spatial(a, a) * _one_body_excitation_spatial(a, b) + \
                             _one_body_excitation_spatial(a, a) * _one_body_excitation_spatial(b, a) + \
                             _one_body_excitation_spatial(a, b) * _one_body_excitation_spatial(a, a) + \
                             _one_body_excitation_spatial(b, a) * _one_body_excitation_spatial(a, a)
            check_two_body = check_two_body - (_one_body_excitation_spatial(a, b) + _one_body_excitation_spatial(b, a))
            idx_check = _sorted_tuple(2 * a, 2 * a + 1, 2 * a, 2 * b + 1)
            check_two_body = normal_ordered(check_two_body)
            new_coeff = two_body_now.terms[idx_check] * check_two_body.terms[idx_check] / 2

            # Verify
            check_two_body = check_two_body * (new_coeff / 2)
            assert two_body_now == check_two_body

            # Assign
            for tensor_idx in [(a, a, a, b), (a, a, b, a), (a, b, a, a), (b, a, a, a)]:
                spatial_tei[tensor_idx] = new_coeff
            spatial_oei[a, b] -= new_coeff / 2
            spatial_oei[b, a] -= new_coeff / 2

        elif len(idxs) == 2 and counts[0] == counts[1]:  # aabb, abab
            a, b = tuple(idxs.keys())
            assert idxs[a] == idxs[b] == 2

            # Determine coefficient(s).
            check_ex = (_one_body_excitation_spatial(a, b) * _one_body_excitation_spatial(a, b) +
                        _one_body_excitation_spatial(a, b) * _one_body_excitation_spatial(b, a) +
                        _one_body_excitation_spatial(b, a) * _one_body_excitation_spatial(a, b) +
                        _one_body_excitation_spatial(b, a) * _one_body_excitation_spatial(b, a))
            check_ex = check_ex - (_one_body_excitation_spatial(a, a) + _one_body_excitation_spatial(b, b))
            check_num = (_one_body_excitation_spatial(a, a) * _one_body_excitation_spatial(b, b) +
                         _one_body_excitation_spatial(b, b) * _one_body_excitation_spatial(a, a))
            check_num, check_ex = normal_ordered(check_num), normal_ordered(check_ex)
            idx_ex = _sorted_tuple(2 * a, 2 * b + 1, 2 * a + 1, 2 * b)
            assert idx_ex not in check_num
            coeff_ex = two_body_now.terms[idx_ex] * check_ex.terms[idx_ex] / 2
            idx_num = _sorted_tuple(2 * a, 2 * b + 1, 2 * a, 2 * b + 1)
            assert idx_num not in check_ex
            coeff_num = two_body_now.terms[idx_num] * check_num.terms[idx_num] / 2

            # Verify
            check_two_body = check_ex * (coeff_ex / 2) + check_num * (coeff_num / 2)
            assert two_body_now == check_two_body

            # Assign
            for tensor_idx in [(a, b, a, b), (a, b, b, a), (b, a, a, b), (b, a, b, a)]:
                spatial_tei[tensor_idx] = coeff_ex
            for tensor_idx in [(a, a, b, b), (b, b, a, a)]:
                spatial_tei[tensor_idx] = coeff_num
            spatial_oei[a, a] -= coeff_ex / 2
            spatial_oei[b, b] -= coeff_ex / 2

        elif len(two_body_now.terms) == 16:  # aabc, abac
            assert len(idxs) == 3
            a, b, c = tuple(idxs.keys())
            if idxs[a] == 2:
                pass
            elif idxs[b] == 2:
                a, b = b, a
            elif idxs[c] == 2:
                a, c = c, a
            else:
                raise AssertionError

            # Determine coefficient(s).
            check_exex = (_one_body_excitation_spatial(a, b) * _one_body_excitation_spatial(a, c) +
                          _one_body_excitation_spatial(a, b) * _one_body_excitation_spatial(c, a) +
                          _one_body_excitation_spatial(b, a) * _one_body_excitation_spatial(a, c) +
                          _one_body_excitation_spatial(b, a) * _one_body_excitation_spatial(c, a) +
                          _one_body_excitation_spatial(a, c) * _one_body_excitation_spatial(a, b) +
                          _one_body_excitation_spatial(a, c) * _one_body_excitation_spatial(b, a) +
                          _one_body_excitation_spatial(c, a) * _one_body_excitation_spatial(a, b) +
                          _one_body_excitation_spatial(c, a) * _one_body_excitation_spatial(b, a) -
                          _one_body_excitation_spatial(b, c) - _one_body_excitation_spatial(c, b)
                          )
            check_numex = (_one_body_excitation_spatial(a, a) * _one_body_excitation_spatial(b, c) +
                           _one_body_excitation_spatial(a, a) * _one_body_excitation_spatial(c, b) +
                           _one_body_excitation_spatial(b, c) * _one_body_excitation_spatial(a, a) +
                           _one_body_excitation_spatial(c, b) * _one_body_excitation_spatial(a, a)
                           )
            check_exex, check_numex = normal_ordered(check_exex), normal_ordered(check_numex)
            idx_exex = _sorted_tuple(2 * a, 2 * a + 1, 2 * b + 1, 2 * c)
            coeff_exex = two_body_now.terms[idx_exex] * check_exex.terms[idx_exex] / 2
            idx_numex = _sorted_tuple(2 * a + 1, 2 * b, 2 * c, 2 * a + 1)
            coeff_numex = two_body_now.terms[idx_numex] * check_numex.terms[idx_numex] / 2

            # Verify
            check_two_body = check_exex * (coeff_exex / 2) + check_numex * (coeff_numex / 2)
            assert two_body_now == check_two_body

            # Assign
            for tensor_idx in [(a, b, a, c), (a, b, c, a), (b, a, a, c), (b, a, c, a),
                               (a, c, a, b), (a, c, b, a), (c, a, a, b), (c, a, b, a)]:
                spatial_tei[tensor_idx] = coeff_exex
            for tensor_idx in [(a, a, b, c), (a, a, c, b), (b, c, a, a), (c, b, a, a)]:
                spatial_tei[tensor_idx] = coeff_numex
            spatial_oei[b, c] -= coeff_exex / 2
            spatial_oei[c, b] -= coeff_exex / 2

        elif len(idxs) == 4:  # abcd
            a, b, c, d = tuple(idxs.keys())
            assert idxs[a] == idxs[b] == idxs[c] == idxs[d] == 1

            # Determine coefficient(s).
            check_1 = ((_one_body_excitation_spatial(a, b) + _one_body_excitation_spatial(b, a)) *
                       (_one_body_excitation_spatial(c, d) + _one_body_excitation_spatial(d, c)))
            check_2 = ((_one_body_excitation_spatial(b, c) + _one_body_excitation_spatial(c, b)) *
                       (_one_body_excitation_spatial(a, d) + _one_body_excitation_spatial(d, a)))
            check_3 = ((_one_body_excitation_spatial(a, c) + _one_body_excitation_spatial(c, a)) *
                       (_one_body_excitation_spatial(b, d) + _one_body_excitation_spatial(d, b)))
            check_1, check_2, check_3 = normal_ordered(check_1), normal_ordered(check_2), normal_ordered(check_3)
            idx_1 = _sorted_tuple(2 * a + 1, 2 * d, 2 * c, 2 * b + 1)
            assert idx_1 not in check_2.terms
            assert idx_1 not in check_3.terms
            idx_2 = _sorted_tuple(2 * b + 1, 2 * d, 2 * c + 1, 2 * a)
            assert idx_2 not in check_1.terms
            assert idx_2 not in check_3.terms
            idx_3 = _sorted_tuple(2 * c + 1, 2 * d, 2 * b, 2 * a + 1)
            coeff_1 = two_body_now.terms[idx_1] * check_1.terms[idx_1] * 2
            coeff_2 = two_body_now.terms[idx_2] * check_2.terms[idx_2] * 2
            coeff_3 = two_body_now.terms[idx_3] * check_3.terms[idx_3] * 2

            # Verify
            check_two_body = check_1 * (coeff_1 / 2) + check_2 * (coeff_2 / 2) + check_3 * (coeff_3 / 2)
            assert two_body_now == check_two_body

            # Assign
            for tensor_idx in [(a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c),
                               (c, d, a, b), (c, d, b, a), (d, c, a, b), (d, c, b, a)]:
                spatial_tei[tensor_idx] = coeff_1 / 2
            for tensor_idx in [(c, b, a, d), (b, c, a, d), (c, b, d, a), (b, c, d, a),
                               (a, d, c, b), (a, d, b, c), (d, a, c, b), (d, a, b, c)]:
                spatial_tei[tensor_idx] = coeff_2 / 2
            for tensor_idx in [(a, c, b, d), (c, a, b, d), (a, c, d, b), (c, a, d, b),
                               (b, d, a, c), (b, d, c, a), (d, b, a, c), (d, b, c, a)]:
                spatial_tei[tensor_idx] = coeff_3 / 2

        else:  # Assure the other case for the two-body index doesn'beta appear.
            raise AssertionError
    assert two_body == check_sum
    return spatial_oei, spatial_tei, const


def ei_to_ham_spatial(spatial_oei: np.ndarray,
                      spatial_tei: np.ndarray,
                      const: float)\
        -> FermionOperator:
    """
    Constructs a Fermion Hamiltonian from the spatial one-body and two-body electron integral tensors.
    (Inverse function of ham_to_ei_spatial)

    This method uses the provided one-body and two-body integral tensors, as well as a constant term, to
    construct a normalized Fermion Hamiltonian.

    Notation:
        E_{pq} = a†_{p↑}a_{q↑} + a†_{p↓}a_{q↓}

        H = ∑_{p,q} oei[p, q] E_{pq}
            + 1/2 ∑_{pqrs} tei[p, q, r, s] E_{pq} * E_{rs}
            + const

    Args:
        spatial_oei (np.ndarray): A 2D array containing the spatial one-body electron integrals.
        spatial_tei (np.ndarray): A 4D array containing the spatial two-body electron integrals.
        const (float): A constant energy term to include in the Hamiltonian.

    Returns:
        FermionOperator: The resulting Fermion Hamiltonian.
    """
    ham = FermionOperator()
    n_orb = spatial_oei.shape[0]
    for p, q in product(range(n_orb), repeat=2):
        ham = ham + _one_body_excitation_spatial(p, q) * spatial_oei[p, q]

    for p, q, r, s in product(range(n_orb), repeat=4):
        if abs(spatial_tei[p, q, r, s]) > EQ_TOLERANCE:
            ham = ham + _one_body_excitation_spatial(p, q) * _one_body_excitation_spatial(r, s) * (
                    spatial_tei[p, q, r, s] / 2)
    ham = normal_ordered(ham)
    return ham + const


def ham_to_ei_spin(fham: FermionOperator,
                   n_spinorb: int)\
        -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Converts a Fermion Hamiltonian into spin one-body and two-body electron integral tensors.

    This function separates the provided Fermion Hamiltonian into spin orbital tensors, which include 
    one-body and two-body electron integrals as well as a constant energy term. The method ensures 
    normal ordering of the Hamiltonian terms and calculates the appropriate tensors for spin orbitals.

    Mathematical Notation:
        E_{pq} = a†_{p}a_{q}

        H = ∑_{p,q} oei[p, q] * E_{pq}
            + (1/2) ∑_{p,q,r,s} tei[p, q, r, s] * E_{pq} * E_{rs}
            + const

    Args:
        fham (FermionOperator): The input Fermion Hamiltonian in normal order.
        n_spinorb (int): The total number of spin orbitals.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: A tuple containing:
            - spin_oei (np.ndarray): A 2D array (n_spinorb x n_spinorb) representing spin one-body electron integrals.
            - spin_tei (np.ndarray): A 4D array (n_spinorb x n_spinorb x n_spinorb x n_spinorb) 
                                     representing spin two-body electron integrals.
            - const (float): The constant energy term in the Hamiltonian.
    """
    spin_oei = np.zeros((n_spinorb, n_spinorb), dtype=complex)
    two_body = dict()
    const = 0.0
    fham = normal_ordered(fham)

    for op, coeff in fham.terms.items():
        cre, ann = cre_ann(op)
        assert len(cre) == len(ann)
        if len(cre) == 1:
            c, a = cre[0], ann[0]
            spin_oei[c, a] += coeff
        elif len(cre) == 2:
            two_body[op] = coeff
        elif len(cre) == 0:
            const = coeff
        else:
            raise ValueError(f"More than 2-body : {cre}")

    two_body = dict_to_operator(two_body, FermionOperator)
    two_body = normal_ordered(two_body)
    check_sum = FermionOperator()
    added = list()
    spin_tei = np.zeros((n_spinorb, n_spinorb, n_spinorb, n_spinorb), dtype=complex)

    for op, coeff in two_body.terms.items():
        (p, q), (r, s) = cre_ann(op)
        unique_key = tuple(sorted((p, q, r, s)))
        if unique_key in added:
            continue
        added.append(unique_key)

        two_body_now = _bring_all_two_spin(p, q, r, s, two_body)
        check_sum = check_sum + two_body_now

        idxs = Counter(unique_key)
        counts = sorted(list(idxs.values()))
        assert sum(counts) == 4

        if len(idxs) == 1:  # aaaa
            raise AssertionError
        elif len(idxs) == 2 and counts[0] != counts[1]:  # aaab
            raise AssertionError
        elif len(idxs) == 2 and counts[0] == counts[1]:  # abab
            a, b = tuple(idxs.keys())
            # abab = - n_a n_b

            # Verify
            check = one_body_number(a, spin_idx=True) * one_body_number(b, spin_idx=True) * coeff
            check = normal_ordered(check)
            if two_body_now != check:
                coeff = -coeff
                check = -check
            assert two_body_now == check, (check, two_body_now)

            # Assign
            for tensor_idx in [(a, a, b, b), (b, b, a, a)]:
                spin_tei[tensor_idx] = coeff
        elif len(idxs) == 3:  # abac, abbc, abca, abcb
            a, b, c = tuple(idxs.keys())
            if idxs[a] == 2:
                pass
            elif idxs[b] == 2:
                a, b = b, a
            elif idxs[c] == 2:
                a, c = c, a

            # Verify
            check = one_body_number(a) * one_body_excitation(b, c) * coeff
            check = normal_ordered(check)
            if two_body_now != check:
                coeff = -coeff
                check = check * -1
            assert two_body_now == check

            # Assign
            for tensor_idx in [(a, a, b, c), (a, a, c, b), (c, b, a, a), (b, c, a, a)]:
                spin_tei[tensor_idx] = coeff

        elif len(idxs) == 4:  # abcd
            # Verify
            check_1 = one_body_excitation(p, r) * one_body_excitation(q, s)
            check_2 = one_body_excitation(p, s) * one_body_excitation(q, r)
            check_1, check_2 = normal_ordered(check_1), normal_ordered(check_2)
            idx_1 = _sorted_tuple(p, s, q, r)
            idx_2 = _sorted_tuple(p, r, q, s)
            assert idx_1 in check_1.terms, (idx_1, list(check_1.terms.keys()))
            assert idx_2 in check_2.terms
            coeff_1 = two_body_now.terms[idx_1] * check_1.terms[idx_1] if idx_1 in two_body_now.terms else 0.0
            coeff_2 = two_body_now.terms[idx_2] * check_2.terms[idx_2] if idx_2 in two_body_now.terms else 0.0
            check = check_1 * coeff_1 + check_2 * coeff_2
            for tensor_idx in [(p, r, q, s), (r, p, q, s), (p, r, s, q), (r, p, s, q),
                               (s, q, r, p), (q, s, r, p), (s, q, p, r), (q, s, p, r)]:
                spin_tei[tensor_idx] = coeff_1
            for tensor_idx in [(p, s, q, r), (s, p, q, r), (p, s, r, q), (s, p, r, q),
                               (r, q, s, p), (q, r, s, p), (r, q, p, s), (q, r, p, s)]:
                spin_tei[tensor_idx] = coeff_2
            try:
                assert two_body_now == check
            except AssertionError as e:
                print(p, q, r, s)
                print(two_body_now.pretty_string())
                print(check.pretty_string())
                print(check_1.pretty_string())
                print(check_2.pretty_string())
                raise e
        else:
            raise ValueError

    assert two_body == check_sum
    return spin_oei, spin_tei, const


def ei_to_ham_spin(spin_oei: np.ndarray,
                   spin_tei: np.ndarray, const: float)\
        -> FermionOperator:
    """
    Constructs a Fermion Hamiltonian from the spin one-body and two-body electron integral tensors.
    (Inverse function of ham_to_ei_spin)
    
    This function takes one-body and two-body spin integral tensors, along with a constant term, and converts
    them back into a FermionOperator representation of the Hamiltonian. The result includes creation and
    annihilation operators for spin orbitals.
    
    Notation:
        E_{pq} = a†_{p}a_{q}
    
        H = ∑_{p,q} oei[p, q] E_{pq}
            + (1/2) ∑_{p,q,r,s} tei[p, q, r, s] * E_{pq} * E_{rs}
            + const
    
    Args:
        spin_oei (np.ndarray): A 2D array (n_spinorb x n_spinorb) representing spin one-body electron integrals.
        spin_tei (np.ndarray): A 4D array (n_spinorb x n_spinorb x n_spinorb x n_spinorb)
                               representing spin two-body electron integrals.
        const (float): The constant energy term in the Hamiltonian.
    
    Returns:
        FermionOperator: The resulting Fermion Hamiltonian.
    """
    ham = FermionOperator()
    n_spinorb = spin_oei.shape[0]
    for c, a in product(range(n_spinorb), repeat=2):
        if np.isclose(spin_oei[c, a], 0.0):
            continue
        ham = ham + FermionOperator(((c, 1), (a, 0)), spin_oei[c, a])
    for c1, a1, c2, a2 in product(range(n_spinorb), repeat=4):
        if np.isclose(spin_tei[c1, a1, c2, a2], 0.0):
            continue
        ham = (ham + FermionOperator(((c1, 1), (a1, 0)), spin_tei[c1, a1, c2, a2] / 2) *
               FermionOperator(((c2, 1), (a2, 0)), 1.0))
    ham = normal_ordered(ham)
    return ham + const


def double_factorization(oei: np.ndarray,
                         tei: np.ndarray,
                         reflection: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """
    Perform double factorization of a Fermionic Hamiltonian using one-body and two-body spatial/spin electron
    integral tensors.
    
    This method factorizes a Fermionic Hamiltonian, represented by the one-body (oei) and two-body (tei) 
    electron integrals, into a sequence of matrices (h_list) and unitary operators (u_list). The factorization 
    can be performed with either number operators or reflection operators, based on the optional argument 
    `reflection`.
    
    Notation:
        - n_p = n_{p↑} + n_{p↓} (number operator)
        - r_{p↑} = 2n_{p↑} - 1, r_{p↓} = 2n_{p↓} - 1 (reflection operators)
        - r_p = r_{p↑} + r_{p↓}
    
    Double-Factorized Hamiltonian (H):
        1) If reflection=False:
            H = u0 (∑_{p} h[0][p] n_p) u0† 
                + 1/2 ∑_{m >= 1} um (∑_{pq} h[m][p,q] n_p n_q) um†
        2) If reflection=True:
            H = u0 (∑_{p} h[0][p] r_p) u0† 
                + 1/2 ∑_{m >= 1} um (∑_{pq} h[m][p,q] r_p r_q) um† + const
        
        Here, um represents a unitary rotation operator corresponding to spatial/spin orbital transformations.
    
    Args:
        oei (np.ndarray): 2D tensor of one-body electron integrals (dimension: n_orb x n_orb) in spin or spatial
            orbitals.
        tei (np.ndarray): 4D tensor of two-body electron integrals (dimension: n_orb x n_orb x n_orb x n_orb) in
            spin or spatial orbitals.
        reflection (bool): If True, factorizes in terms of reflection operators; defaults to False.
    
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], float]: 
            - h_list: List of coefficient matrices; h_list[0] is a 1D array (one-body coefficients), 
              while h_list[m > 0] are 2D arrays (two-body coefficients).
            - u_list: List of unitary rotation matrices corresponding to orbital rotations.
            - const: Scalar constant representing the remaining contribution, which is zero if reflection=False.
    """
    n_orb = oei.shape[0]
    h_0, u_0 = eigh(oei)
    h_list, u_list = [h_0], [u_0]
    w_mat = tei.reshape((n_orb ** 2, n_orb ** 2))
    w, g = eigh(w_mat)
    for w_l, g_l in zip(w, g.T.conj()):
        # assert w_l > 0 or np.isclose(w_l, 0.0)
        if np.isclose(w_l, 0.0):
            continue
        g_l = g_l.reshape((n_orb, n_orb))
        h_l, u_l = eigh(g_l)
        h_l = w_l * np.outer(h_l, h_l)
        h_list.append(h_l)
        u_list.append(u_l)
    if not reflection:
        return h_list, u_list, 0.0
    else:
        return number_factorization_to_reflection(h_list, u_list, is_spin=False)


def number_factorization_to_reflection(h_list: List[np.ndarray],
                                       u_list: List[np.ndarray],
                                       is_spin: bool) \
        -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """
    Transforms the factorization with number operators into that of reflection operators.

    Notation:
        - n_p = n_{p↑} + n_{p↓} (number operator for spatial orbitals)
        - r_{p↑} = 2n_{p↑} - 1, r_{p↓} = 2n_{p↓} - 1 (reflection operators for spin orbitals)
        - r_p = r_{p↑} + r_{p↓} (reflection operator for spatial orbitals)

    Transformation:
        Given a factorization with number operators:
        H = u0 (∑_{p} h[0][p] n_p) u0† + ∑_{m≥1} um (∑_{pq} h[m][p,q] n_p n_q) um†,

        this function transforms h and u such that the resulting Hamiltonian uses 
        reflection operators:
        H = u0 (∑_{p} h'[0][p] r_p) u0† + ∑_{m≥1} u'm (∑_{pq} h'[m][p,q] r_p r_q) u'm† + const.

    Similar transformations apply when the inputs are specified in terms of spin orbital operators.

    Args:
        h_list (List[np.ndarray]): A list of coefficient matrices for the input Hamiltonian in
                                   the number operator form. The first element is a 1D array
                                   representing the one-body coefficients, and subsequent elements
                                   are 2D arrays representing two-body coefficients.
        u_list (List[np.ndarray]): A list of unitary rotation matrices corresponding to orbital 
                                   transformations, where each matrix has dimensions n_orb x n_orb.
        is_spin (bool): Indicates if the inputs (h_list, u_list) correspond to spin orbital operators.
                        - True: Spin orbital operators.
                        - False: Spatial orbital operators.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], float]:
            - ref_h_list (List[np.ndarray]): The transformed coefficient matrices for the Hamiltonian
                                             in the reflection operator form.
            - ref_u_list (List[np.ndarray]): The list of rotation unitary matrices corresponding to
                                             the reflection operator representation.
            - const (float): The constant energy term arising during the transformation, equivalent
                             to the trace of the original one-body coefficient matrix. This term 
                             ensures energy conservation.

    Raises:
        ValueError: If the resulting constant term is not a real number.
    """
    n_orb = h_list[0].shape[0]
    h0, u0 = h_list[0], u_list[0]
    oei = u0 @ np.diag(h0) / 2 @ u0.T.conj()
    new_h_list = list()
    if is_spin:
        const = np.sum(h0) / 2
    else:
        const = np.sum(h0)
    for h, u in zip(h_list[1:], u_list[1:]):
        # Transition of twobody to onebody
        for p, q in product(range(n_orb), repeat=2):
            for r, s in product(range(n_orb), repeat=2):
                if is_spin:
                    oei[r, s] += h[p, q] / 8 * (u[r, p] * u[s, p].conj() + u[r, q] * u[s, q].conj())
                else:
                    oei[r, s] += h[p, q] / 4 * (u[r, p] * u[s, p].conj() + u[r, q] * u[s, q].conj())
        if is_spin:
            const += np.sum(h) / 8
        else:
            const += np.sum(h) / 2
        # h = h - np.diag(h)
        # Transition of twobody to const (p=q case)
        # const += np.sum(np.diag(h)) / 4
        # h = h - np.diag(np.diag(h))
        new_h_list.append(h / 4)
    if not np.isclose(const.imag, 0.0):
        raise ValueError(f"{const} is not real.")
    const = float(const.real)
    h0, u0 = eigh(oei)
    ref_h_list = [h0] + new_h_list
    ref_u_list = [u0] + u_list[1:]
    return ref_h_list, ref_u_list, const


def calculate_reflect_norm(ref_h_list: List[np.ndarray],
                           frag_type: str)\
        -> float:
    """
    Calculate the norm of a factorized Hamiltonian represented in the reflection operator form.

    This method computes the norm for a Hamiltonian decomposed into reflection operators, based on the 
    specified fragment type ("LCU" or "FH"). The norm is calculated based on either the L1 or L2 norm, 
    depending on the fragment type because the terms are individually counted in LCU while they are grouped
    in FH type.

    Args:
        ref_h_list (List[np.ndarray]): A list of coefficient matrices for the factorized Hamiltonian in
                                       reflection operator form. The first element corresponds to the
                                       one-body coefficients, while subsequent elements are two-body
                                       coefficient matrices.
        frag_type (str): The type of fragment used in the factorization. 
                         - "LCU": Use the L1 norm for the norm calculation.
                         - "FH": Use the L2 norm for the norm calculation.

    Returns:
        float: The computed norm of the Hamiltonian based on the given fragment type.

    Raises:
        ValueError: If the provided `frag_type` is not "LCU" or "FH".
    """
    if frag_type not in ["LCU", "FH"]:
        raise ValueError(f"Invalid frag_type: {frag_type}")
    if frag_type == "LCU":
        norm_func = partial(np.linalg.norm, ord=1)
    else:
        norm_func = partial(np.linalg.norm, ord=2)
    norm = norm_func(ref_h_list[0]) * 2
    for h in ref_h_list[1:]:
        if frag_type == "LCU":
            norm += norm_func(h.flatten()) / 2 - norm_func(np.diag(h)) / 4
        else:
            norm += norm_func((h - np.diag(np.diag(h)) / 2).flatten()) / 2
    return norm


def double_factorization_to_hamiltonian(h_list: List[np.ndarray],
                                        u_list: List[np.ndarray],
                                        is_spin: bool,
                                        is_reflection: bool = False)\
        -> FermionOperator:
    """
    Transform lists of coefficient matrices (h_list) and unitary operators (u_list) into a single 
    FermionOperator Hamiltonian.
    
    This method reconstructs a Hamiltonian from its factorized representation, using the provided
    coefficient matrices and orbital rotation matrices. Depending on the input parameters, it supports
    both spin and spatial orbital representations, as well as reflection-based operators.
    
    Args:
        h_list (List[np.ndarray]): A list of coefficient matrices for the factorized Hamiltonian.
            - h_list[0] is a 1D array for one-body coefficients.
            - h_list[m > 0] are 2D arrays for two-body coefficients.
        u_list (List[np.ndarray]): A list of unitary matrices corresponding to orbital rotations.
            Each matrix should have dimensions n_orb x n_orb.
        is_spin (bool): Indicates whether the coefficient matrices and unitaries are in terms of spin orbitals.
            - True: Spin orbital representation.
            - False: Spatial orbital representation.
        is_reflection (bool): Specifies whether the coefficient matrices correspond to reflection operators.
            - True: Use reflection operators.
            - False: Use number operators.
    
    Returns:
        FermionOperator: A FermionOperator representing the reconstructed Hamiltonian.
    """
    ham = FermionOperator()
    n_orb = h_list[0].shape[0]
    for h, u in zip(h_list, u_list):
        tmp_ham = FermionOperator()
        if len(h.shape) == 1:  # One body
            for p in range(n_orb):
                if not is_reflection:
                    tmp_ham = tmp_ham + one_body_number(p, spin_idx=is_spin) * h[p]
                else:
                    tmp_ham = tmp_ham + one_body_reflection(p, spin_idx=is_spin) * h[p]
        elif len(h.shape) == 2:  # Two body
            for p, q in product(range(n_orb), repeat=2):
                if not is_reflection:
                    tmp_ham = tmp_ham + one_body_number(p, spin_idx=is_spin) * one_body_number(q, spin_idx=is_spin) * (
                            h[p, q] / 2)
                else:
                    tmp_ham = tmp_ham + two_body_reflection(p, q, spin_idx=is_spin) * h[p, q] / 2
        else:
            raise ValueError(f"Invalid n-body: {len(h.shape)}")
        ham = ham + fermion_rotation_operator(tmp_ham, u, spatial_v=not is_spin)
    return ham
