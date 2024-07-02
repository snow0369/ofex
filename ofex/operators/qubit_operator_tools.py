from typing import Dict, Union, Tuple

import numpy as np
from openfermion import QubitOperator, FermionOperator
from openfermion.config import EQ_TOLERANCE
from openfermion.ops.operators.symbolic_operator import COEFFICIENT_TYPES

from ofex.operators.symbolic_operator_tools import operator, coeff
from ofex.operators.types import SinglePauli


def _single_pauli_to_dict(op: SinglePauli) -> Dict[int, str]:
    idx_op = dict()
    for idx, single_op in op:
        assert idx not in idx_op
        idx_op[idx] = single_op
    return idx_op


def is_z_only(op: QubitOperator) -> bool:
    for op in op.terms.keys():
        if any([x[1].upper() not in ["I", "Z"] for x in op]):
            return False
    return True


def single_pauli_commute_chk(op1: Union[SinglePauli, QubitOperator],
                             op2: Union[SinglePauli, QubitOperator], ):
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


def dict_to_operator(op_dict, base) -> Union[QubitOperator, FermionOperator]:
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


def normalize_by_lcu_norm(ham: QubitOperator,
                          level: int = 1,
                          **kwargs) -> Tuple[QubitOperator, float]:
    """
    level = 0 : Pauli 1 norm
    level = 1 : Sorted Insertion
    level = 2 : SI with norm optimization
    """
    from ofex.measurement.sorted_insertion import sorted_insertion, optimal_sorted_insertion

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
