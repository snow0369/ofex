from typing import Optional

import numpy as np
from openfermion import QubitOperator, count_qubits, get_sparse_operator
from scipy.sparse import spmatrix, identity

from ofex.operators.symbolic_operator_tools import coeff, is_constant


def single_pauli_expm(pauli: QubitOperator,
                      n_qubits: Optional[int] = None) -> spmatrix:
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
