from copy import deepcopy
from typing import Union

from openfermion import SymbolicOperator, QubitOperator, FermionOperator
from openfermion.config import EQ_TOLERANCE
from openfermion.ops.operators.symbolic_operator import COEFFICIENT_TYPES

from ofex.operators.types import Operators
from ofex.utils.dict_utils import compare_dict

__all__ = ["coeff", "operator", "single_term", "compare_operators", "clean_imaginary", "is_constant",
           "dict_to_operator"]


def coeff(op: Operators):
    """
    Extracts the coefficient of a single-term operator.

    Args:
        op (Operators): The symbolic operator instance.

    Returns:
        float: The coefficient of the single term if present.

    Raises:
        ValueError: If the operator has zero or multiple terms.
    """
    t = list(op.terms.values())
    if len(t) == 0:
        return 0.0
    elif len(t) != 1:
        raise ValueError
    return t[0]


def operator(op: Operators):
    """
    Extracts the term (keys) of a single-term operator.

    Args:
        op (Operators): The symbolic operator instance.

    Returns:
        The term of the single-term operator.

    Raises:
        ValueError: If the operator has zero or multiple terms.
    """
    t = list(op.terms.keys())
    if len(t) == 0:
        return op.identity()
    elif len(t) != 1:
        raise ValueError
    return t[0]


def single_term(op: Operators):
    """
    Extracts a single term and its coefficient from the operator.

    Args:
        op (Operators): The symbolic operator instance.

    Returns:
        tuple: A tuple containing the term and its coefficient.

    Raises:
        ValueError: If the operator has zero or multiple terms.
    """
    return operator(op), coeff(op)


def compare_operators(op1: SymbolicOperator, op2: SymbolicOperator,
                      str_len=40, atol=EQ_TOLERANCE) -> str:
    """
    Compares two symbolic operators and returns a string summary of their differences.

    Args:
        op1 (SymbolicOperator): The first symbolic operator for comparison.
        op2 (SymbolicOperator): The second symbolic operator for comparison.
        str_len (int): Maximum length for string representation of terms.
        atol (float): Absolute tolerance for comparing coefficients.

    Returns:
        str: A formatted string showing the differences between the terms of the two operators.
    """
    def repr_op(k, c):
        return str(op1.__class__(k, c))

    return compare_dict(op1.terms, op2.terms, repr_op,
                        str_len=str_len, atol=atol)


def clean_imaginary(op: SymbolicOperator, atol=EQ_TOLERANCE) -> SymbolicOperator:
    """
    Removes small imaginary components from operator coefficients.

    Args:
        op (SymbolicOperator): The symbolic operator to clean.
        atol (float): Absolute tolerance for the imaginary part.

    Returns:
        SymbolicOperator: A new operator with cleaned coefficients.

    Raises:
        ValueError: If any coefficient has an imaginary component exceeding the tolerance.
    """
    op = deepcopy(op)
    for k, v in op.terms.items():
        if abs(v.imag) > atol:
            raise ValueError
        op.terms[k] = v.real
    return op


def is_constant(op: SymbolicOperator) -> bool:
    """
    Checks if the operator is constant.

    Args:
        op (SymbolicOperator): The symbolic operator to check.

    Returns:
        bool: True if the operator is constant, False otherwise.
    """
    op.compress()
    try:
        _ = coeff(op)
    except ValueError:
        return False
    return True


def dict_to_operator(op_dict, base) -> Union[QubitOperator, FermionOperator]:
    """
    Converts a dictionary of terms to a symbolic operator.

    Args:
        op_dict (dict): Dictionary where keys are terms and values are coefficients.
        base (type): The base operator type, either QubitOperator or FermionOperator.

    Returns:
        Union[QubitOperator, FermionOperator]: The constructed symbolic operator.

    Raises:
        TypeError: If the base type is not supported.
        ValueError: If coefficients are not numeric or terms are invalid.
    """
    if base not in [QubitOperator, FermionOperator]:
        raise TypeError(f"Unknown base type {base}.")
    op = base()

    new_op_dict = dict()
    for term, coefficient in op_dict.items():
        if not isinstance(coefficient, COEFFICIENT_TYPES):
            raise ValueError(
                'Coefficient must be a numeric type. Got {}'.format(
                    type(coefficient)))
        if term is None:
            continue
        elif isinstance(term, (list, tuple)):
            term = op._parse_sequence(term)
        elif isinstance(term, str):
            term = op._parse_string(term)
        else:
            raise ValueError('term specified incorrectly.')
        coefficient, term = op._simplify(term, coefficient=coefficient)
        new_op_dict[term] = coefficient
    op.terms = new_op_dict
    return op
