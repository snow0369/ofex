"""
This module provides utility functions for operations related to QubitOperator and SinglePauli objects.

Functions included:
- is_z_only: Checks if a QubitOperator contains only Z-type Pauli operators.
- single_pauli_commute_chk: Determines whether two SinglePauli or QubitOperator objects commute.
- normalize_by_lcu_norm: Normalizes a QubitOperator using Linear Combination of Unitaries (LCU) norms to
  ensure its eigenspectrum resides in [-1, 1].
"""

from typing import Dict, Union, Tuple

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.operators.symbolic_operator_tools import operator, coeff
from ofex.operators.types import SinglePauli


__all__ = ["is_z_only", "single_pauli_commute_chk", "normalize_by_lcu_norm"]


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


def normalize_by_lcu_norm(ham: QubitOperator,
                          level: int = 1,
                          **kwargs) -> Tuple[QubitOperator, float]:
    """
    Normalize a QubitOperator using different methods to calculate its Linear Combination of Unitaries (LCU) norm.

    The eigenspectrum of the normalized operator is guaranteed to be less than one, while this computation is
    much more efficient than normalization by spectral norm.
    Better normalization makes the spectrum more separated within the range [-1, 1], which corresponds to 
    normalization using smaller norms.

    This function supports multiple levels of normalization strategies. Before normalization, it ensures that
    any constant term in the Hamiltonian is removed.

    Args:
        ham (QubitOperator): The Hamiltonian to normalize.
        level (int, optional): Specifies the normalization strategy:
            - level = 0: Uses the Pauli 1-norm for normalization.
            - level = 1: Applies the Sorted Insertion (SI) method based on commutation relationships.
            - level = 2: Uses the optimally sorted insertion (SI) method that minimizes norms.
        **kwargs: Additional arguments for optimization when `level=2`.

    Returns:
        Tuple[QubitOperator, float]: A tuple where the first element is the normalized Hamiltonian,
                                     and the second element is the calculated norm.

    Raises:
        ValueError: If the Hamiltonian contains a constant term or if the specified level is invalid.
    """
    from ofex.measurement import sorted_insertion, optimal_sorted_insertion

    if () in ham.terms:
        if np.isclose(ham.terms[()], 0.0):
            del ham.terms[()]
        else:
            raise ValueError("Make trace zero before the normalization (Remove the constant term).")

    if level == 0:
        norm = ham.induced_norm(order=1)
    elif level in (1, 2):
        if level == 1:
            si_out = sorted_insertion(ham, anticommute=True)
        else:
            si_out = optimal_sorted_insertion(ham, anticommute=True, **kwargs)
        l2_norm_list = [frag.induced_norm(order=2) for frag in si_out]
        norm = sum(l2_norm_list)
    else:
        raise ValueError
    return ham/norm, norm
