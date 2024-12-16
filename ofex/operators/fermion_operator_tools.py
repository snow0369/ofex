"""
This module contains a collection of tools for manipulating and analyzing fermionic operators.

It provides utility functions and operator constructs to handle creation and annihilation
operators, number operators, reflection operators, and excitation operators.
"""

from typing import List, Tuple, Union

from openfermion import FermionOperator

from ofex.operators.symbolic_operator_tools import operator, dict_to_operator
from ofex.operators.types import SingleFermion

__all__ = ["cre_ann", "is_number_only", "normal_ordered_single",
           "one_body_number", "one_body_excitation", "one_body_reflection",
           "two_body_reflection"]


def cre_ann(fermion_op: Union[SingleFermion, FermionOperator]) -> Tuple[List[int], List[int]]:
    """
    Extracts the lists of creation and annihilation operators from a fermionic operator.

    Args:
        fermion_op: A SingleFermion or FermionOperator to extract operators from.

    Returns:
        A tuple containing two lists:
            - The list of the creation operators indices.
            - The list of the annihilation operators indices.
    """
    cre, ann = list(), list()
    if isinstance(fermion_op, FermionOperator):
        fermion_op = operator(fermion_op)
    for idx, dag in fermion_op:
        if dag:
            cre.append(idx)
        else:
            ann.append(idx)
    return cre, ann


def is_number_only(f: FermionOperator) -> bool:
    """
    Checks whether a FermionOperator is composed exclusively of number operators.
    
    Args:
        f: The FermionOperator to check.
    
    Returns:
        True if all terms in the operator represent valid number operators; otherwise, False.
    """
    for op in f.terms:
        cre, ann = cre_ann(op)
        if sorted(cre) != sorted(ann):
            return False
    return True


def normal_ordered_single(cre: List[int], ann: List[int]) -> SingleFermion:
    """
    Generates a SingleFermion operator in normal order.

    In our convention, normal ordering implies terms are ordered
    from the highest tensor factor (on left) to lowest (on right).
    Also, ladder operators come first.

    Args:
        cre: A list of indices for the creation operators.
        ann: A list of indices for the annihilation operators.

    Returns:
        A tuple representing the normal-ordered SingleFermion operator.
    """
    ops = [(c, 1) for c in sorted(cre, reverse=True)] + [(a, 0) for a in sorted(ann, reverse=True)]
    return tuple(ops)


def one_body_excitation(p: int, q: int,
                        spin_idx: bool = True, hermitian: bool = True) -> FermionOperator:
    """
    Generates a one-body excitation operator for the specified indices.
    
    If `spin_idx` is True, the indices are treated as spin-indexed, and the operator is defined as:
        E_{p,q} = a†_p a_q (+ h.c.)
    If `spin_idx` is False, the indices are treated as spatially indexed, and the operator is defined as:
        E_{p,q} = a†_p↑ a_q↑ + a†_p↓ a_q↓ (+ h.c.)
    Here, "h.c." refers to the Hermitian conjugate, which is included only when `hermitian` is True, and thus
    p and q becomes indistinguishable in the resulting operator.
    
    Args:
        p (int): The source fermion index.
        q (int): The target fermion index.
        spin_idx (bool): Indicates whether the indices are spin-indexed. Defaults to True.
        hermitian (bool): Specifies whether the resulting operator is Hermitian. Defaults to True.
    
    Returns:
        FermionOperator: The generated one-body excitation operator.
    """
    if p == q:
        return one_body_number(p, spin_idx) * 2
    if hermitian:
        return one_body_excitation(p, q, spin_idx, hermitian=False) + \
            one_body_excitation(q, p, spin_idx, hermitian=False)
    if spin_idx:
        return dict_to_operator({((p, 1), (q, 0)): 1.0}, base=FermionOperator)
    else:
        return dict_to_operator(
            {((2 * p, 1), (2 * q, 0)): 1.0,
             ((2 * p + 1, 1), (2 * q + 1, 0)): 1.0},
            base=FermionOperator
        )


def one_body_number(p: int, spin_idx: bool = True) -> FermionOperator:
    """
    Generates a one-body number operator for the specified index.

    For spin-indexed systems:
        n_{p} (spin)    = a†_{p} a_{p}

    For spatially indexed systems:
        n_{p} (spatial) = n_{p,↑} + n_{p,↓} = a†_{p,↑} a_{p,↑} + a†_{p,↓} a_{p,↓}

    Args:
        p: The index of the fermion.
        spin_idx: Whether the index is spin-indexed or spatially indexed. Defaults to True.

    Returns:
        A FermionOperator representing the number operator for the given index.
    """
    if spin_idx:
        return dict_to_operator({((p, 1), (p, 0)): 1.0}, base=FermionOperator)
    else:
        return dict_to_operator({((2 * p, 1), (2 * p, 0)): 1.0,
                                 ((2 * p + 1, 1), (2 * p + 1, 0)): 1.0}, base=FermionOperator)


def one_body_reflection(p: int, spin_idx: bool = True) -> FermionOperator:
    """
    Computes the reflection operator for a one-body term.

    Reflections are expressed as:
    - r_{p} (spin)    = 2 * n_{p} (spin) - 1
    - r_{p} (spatial) = r_{p,↑} + r_{p,↓}

    Args:
        p: The index of the fermion.
        spin_idx: Whether to generate the operator in spin indexing. Defaults to True.

    Returns:
        The FermionOperator representing the one-body reflection.
    """
    if spin_idx:
        return one_body_number(p, spin_idx=True) * 2 - 1
    else:
        return one_body_number(p, spin_idx=False) * 2 - 2


def two_body_reflection(p: int, q: int, spin_idx: bool = True) -> FermionOperator:
    """
    Computes the two-body reflection operator as the product of two reflection terms.

    Args:
        p: The first fermion index.
        q: The second fermion index.
        spin_idx: Whether to use spin indexing. Defaults to True.

    Returns:
        The FermionOperator representing r_{p} * r_{q}.
    """
    if spin_idx:
        if p == q:
            return FermionOperator.identity()
        else:
            return one_body_reflection(p, spin_idx=True) * one_body_reflection(q, spin_idx=True)
    else:
        if p == q:
            return 4 * (2 * one_body_number(2 * p) * one_body_number(2 * p + 1) - one_body_number(p,
                                                                                                  spin_idx=False) + 1)
        else:
            return one_body_reflection(p, spin_idx=False) * one_body_reflection(q, spin_idx=False)
