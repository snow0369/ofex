from copy import deepcopy

from openfermion import SymbolicOperator
from openfermion.config import EQ_TOLERANCE

from ofex.operators.types import Operators
from ofex.utils.dict_utils import compare_dict


def coeff(op: Operators):
    t = list(op.terms.values())
    if len(t) == 0:
        return 0.0
    elif len(t) != 1:
        raise ValueError
    return t[0]


def operator(op: Operators):
    t = list(op.terms.keys())
    if len(t) == 0:
        return op.identity()
    elif len(t) != 1:
        raise ValueError
    return t[0]


def single_term(op: Operators):
    return operator(op), coeff(op)


def compare_operators(op1: SymbolicOperator, op2: SymbolicOperator,
                      str_len=40, atol=EQ_TOLERANCE) -> str:
    def repr_op(k, c):
        return str(op1.__class__(k, c))

    return compare_dict(op1.terms, op2.terms, repr_op,
                        str_len=str_len, atol=atol)


def clean_imaginary(op: SymbolicOperator, atol=EQ_TOLERANCE) -> SymbolicOperator:
    op = deepcopy(op)
    for k, v in op.terms.items():
        if abs(v.imag) > atol:
            raise ValueError
        op.terms[k] = v.real
    return op


def is_constant(op: SymbolicOperator) -> bool:
    op.compress()
    try:
        _ = coeff(op)
    except ValueError:
        return False
    return True
