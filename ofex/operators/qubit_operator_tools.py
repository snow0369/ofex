"""
This module provides utility functions for operations related to QubitOperator and SinglePauli objects.

Functions included:
- is_z_only: Checks if a QubitOperator contains only Z-type Pauli operators.
- single_pauli_commute_chk: Determines whether two SinglePauli or QubitOperator objects commute.
- normalize_by_lcu_norm: Normalizes a QubitOperator using Linear Combination of Unitaries (LCU) norms to
ensure its eigenspectrum resides in [-1, 1].
"""

from typing import Dict, Union

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.operators.symbolic_operator_tools import operator, coeff
from ofex.operators.types import SinglePauli


__all__ = ["is_z_only", "single_pauli_commute_chk"]


def _single_pauli_to_dict(op: SinglePauli) -> Dict[int, str]:
    idx_op = dict()
    for idx, single_op in op:
        assert idx not in idx_op
        idx_op[idx] = single_op
    return idx_op


def is_z_only(op: QubitOperator) -> bool:
    """
    Check if a given QubitOperator consists only of Z-type Pauli operators,

    Args:
        op (QubitOperator): The QubitOperator to check.

    Returns:
        bool: True if the operator consists only of Z-type operators, False otherwise.
    """
    for op in op.terms.keys():
        if any([x[1].upper() not in ["I", "Z"] for x in op]):
            return False
    return True


def single_pauli_commute_chk(op1: Union[SinglePauli, QubitOperator],
                             op2: Union[SinglePauli, QubitOperator]) -> bool:
    """
    Check if two single Pauli operators or QubitOperators commute.

    Args:
        op1 (Union[SinglePauli, QubitOperator]): The first operator to compare.
        op2 (Union[SinglePauli, QubitOperator]): The second operator to compare.

    Returns:
        bool: True if the operators commute, False otherwise.

    Raises:
        ValueError: If either operator has an imaginary coefficient.
    """
    if isinstance(op1, QubitOperator):
        if not np.isclose(coeff(op1).imag, 0.0, atol=EQ_TOLERANCE):
            raise ValueError("Coeff should be real.")
        op1 = operator(op1)
    if isinstance(op2, QubitOperator):
        if not np.isclose(coeff(op2).imag, 0.0, atol=EQ_TOLERANCE):
            raise ValueError("Coeff should be real.")
        op2 = operator(op2)

    if len(op1) == 0 or len(op2) == 0:
        return True
    op1 = _single_pauli_to_dict(op1)
    op2 = _single_pauli_to_dict(op2)

    op1, op2 = (op1, op2) if len(op1) < len(op2) else (op2, op1)
    commute = True
    for idx1, pauli1 in op1.items():
        if idx1 in op2 and pauli1 != 'I' and op2[idx1] != 'I':
            commute = not commute if pauli1 != op2[idx1] else commute
    return commute


