from typing import List, Tuple, Union

from openfermion import FermionOperator

from ofex.operators.qubit_operator_tools import dict_to_operator
from ofex.operators.symbolic_operator_tools import operator
from ofex.operators.types import SingleFermion


def cre_ann(fermion_op: Union[SingleFermion, FermionOperator]) -> Tuple[List[int], List[int]]:
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
    for op, coeff in f.terms.dict():
        cre, ann = cre_ann(op)
        if sorted(cre) != sorted(ann):
            return False
    return True


def normal_ordered_single(cre: List[int], ann: List[int]) -> SingleFermion:
    ops = [(c, 1) for c in sorted(cre, reverse=True)] + [(a, 0) for a in sorted(ann, reverse=True)]
    return tuple(ops)


def one_body_excitation(p: int, q: int,
                        spin_idx: bool = True, hermitian: bool = True) -> FermionOperator:
    if p == q:
        return one_body_number(p, spin_idx)
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
    n_{p} (spin)    = a†_{p} a_{p}
    n_{p} (spatial) = n_{p,↑} + n_{p,↓}
    """
    if spin_idx:
        return dict_to_operator({((p, 1), (p, 0)): 1.0}, FermionOperator)
    else:
        return dict_to_operator({((2 * p, 1), (2 * p, 0)): 1.0,
                                 ((2 * p + 1, 1), (2 * p + 1, 0)): 1.0}, base=FermionOperator)


def one_body_reflection(p: int, spin_idx: bool = True) -> FermionOperator:
    """
    r_{p} (spin)    = 2 * n_{p} (spin) - 1
    r_{p} (spatial) = r_{p,↑} + r_{p,↓}
    """
    if spin_idx:
        return one_body_number(p, spin_idx=True) * 2 - 1
    else:
        return one_body_number(p, spin_idx=False) * 2 - 2


def two_body_reflection(p: int, q: int, spin_idx: bool = True) -> FermionOperator:
    """
    Returns r_{p} * r_{q}
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
