"""
This module provides functionality for computing the matrix exponential of Pauli operators.

It supports both single-term Pauli operators and reflective operators, enabling efficient
generation of exponential matrix representations for use in quantum computing.

Functions included:
- single_pauli_expm: Computes the exponential of a single Pauli operator.
- reflective_pauli_expm: Efficiently computes the exponential of reflective Pauli operators.
"""

from typing import Optional

import numpy as np
from openfermion import QubitOperator, count_qubits, get_sparse_operator
from scipy.sparse import spmatrix, identity

from ofex.operators.symbolic_operator_tools import is_constant

__all__ = ["single_pauli_expm", "reflective_pauli_expm"]


def single_pauli_expm(pauli: QubitOperator,
                      n_qubits: Optional[int] = None) -> spmatrix:
    """
    Compute the exponential of a single Pauli operator.

    This function generates the matrix representation of `exp(pauli)`
    for a given single-term Pauli operator expressed as a `QubitOperator`.

    Args:
        pauli (QubitOperator): A single-term QubitOperator representing the Pauli operator.
        n_qubits (Optional[int]): The number of qubits in the system. If not specified, 
                                  the number of qubits is inferred from the operator.

    Returns:
        spmatrix: A sparse matrix representing the exponential of the given Pauli operator.

    Raises:
        ValueError: If the number of qubits specified is invalid or if the operator 
                    contains more than one term.
    """
    if n_qubits is None:
        n_qubits = count_qubits(pauli)
    if n_qubits < count_qubits(pauli):
        raise ValueError('Invalid number of qubits specified.')
    if len(pauli.terms) != 1:
        raise ValueError('Invalid number of terms.')
    return reflective_pauli_expm(pauli, n_qubits, check_reflective=False)


def reflective_pauli_expm(pauli: QubitOperator,
                          n_qubits: Optional[int] = None,
                          check_reflective: bool = True) -> spmatrix:
    """
    Compute the exponential of reflective qubit operator efficiently.

    This function generates the matrix representation of `exp(pauli)` for a
    given Pauli operator, under the assumption that the operator is reflective 
    (i.e., the square of the operator is constant).
    For example, a sum of mutually anti-commuting Pauli operators or an LCU-fragment is reflective.
    The exponential of a reflective operator can be computed efficiently.

    Args:
        pauli (QubitOperator): The operator for which the exponential is to
                               be calculated.
        n_qubits (Optional[int]): The number of qubits in the system. If not specified, 
                                  the number of qubits is inferred from the operator.
        check_reflective (bool): Whether to validate that the operator is reflective.
                                 Set to False to bypass this validation.

    Returns:
        spmatrix: A sparse matrix representing the exponential of the given Pauli operator.

    Raises:
        ValueError: If the number of qubits specified is invalid or if the operator 
                    is not reflective when `check_reflective` is True.
    """
    if n_qubits is None:
        n_qubits = count_qubits(pauli)
    if n_qubits < count_qubits(pauli):
        raise ValueError('Invalid number of qubits specified.')
    if check_reflective:
        sq_pauli = pauli * pauli
        if not is_constant(sq_pauli):
            raise ValueError("Not a reflective Pauli operator.")
    beta = np.emath.sqrt(sum([v**2 for v in pauli.terms.values()]))
    c_re, c_im = beta.real, beta.imag
    mat = get_sparse_operator(pauli / beta, n_qubits)
    i_mat = identity(mat.shape[0], format=mat.format)
    ret = (np.cosh(c_re) * np.cos(c_im) + 1j * np.sinh(c_re) * np.sin(c_im)) * i_mat
    ret += (np.sinh(c_re) * np.cos(c_im) + 1j * np.cosh(c_re) * np.sin(c_im)) * mat
    return ret
