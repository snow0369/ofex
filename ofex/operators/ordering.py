from typing import List

from openfermion.config import EQ_TOLERANCE

from ofex.operators.types import Operators


def order_abs_coeff(op: Operators, reverse=False, atol=EQ_TOLERANCE) -> List[Operators]:
    op_list = [(single_op, coeff) for (single_op, coeff) in op.terms.items()]
    op_list = sorted(op_list, key=lambda x: abs(x[1]), reverse=reverse)
    return [op.__class__(single_op, coeff) for single_op, coeff in op_list if abs(coeff) > atol]
