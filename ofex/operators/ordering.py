"""
This module provides various functionality for ordering operators. It includes a utility function `order_abs_coeff`
that orders operators by absolute value of their coefficients.
"""

from typing import List

from openfermion.config import EQ_TOLERANCE

from ofex.operators.types import Operators

__all__ = ["order_abs_coeff"]


def order_abs_coeff(op: Operators,
                    reverse=False,
                    atol=EQ_TOLERANCE) -> List[Operators]:
    """
    Sorts the terms of an operator by the absolute value of their coefficients.

    Parameters:
        op (Operators): The operator containing terms to be sorted.
        reverse (bool, optional): Whether to sort in descending order. Defaults to False.
        atol (float, optional): Absolute tolerance; terms with coefficients below this tolerance are excluded.
            Defaults to EQ_TOLERANCE.

    Returns:
        List[Operators]: A list of operators with terms sorted by absolute coefficient values.
    """
    op_list = list(op.terms.items())
    op_list = sorted(op_list, key=lambda x: abs(x[1]), reverse=reverse)
    return [op.__class__(single_op, coeff) for single_op, coeff in op_list if abs(coeff) > atol]
