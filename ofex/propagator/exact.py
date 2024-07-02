from typing import Union, Optional

import numpy as np
import scipy
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse import spmatrix

from ofex.exceptions import OfexTypeError
from ofex.linalg.pauli_expm import single_pauli_expm, reflective_pauli_expm


def exact_expop(op: Union[QubitOperator, np.ndarray, spmatrix],
                n_qubits: Optional[int] = None,
                exact_sparse=False,
                reflective: bool = False,
                check_reflective: bool = True) -> Union[spmatrix, np.ndarray]:
    if isinstance(op, QubitOperator):
        if len(op.terms) == 1:
            return single_pauli_expm(op, n_qubits)
        elif reflective:
            return reflective_pauli_expm(op, n_qubits, check_reflective)
        else:
            op = get_sparse_operator(op, n_qubits=n_qubits)
    if isinstance(op, np.ndarray):
        return scipy.linalg.expm(op)
    elif isinstance(op, spmatrix) and exact_sparse:
        op = op.toarray()
        return scipy.linalg.expm(op)
    elif isinstance(op, spmatrix):
        return scipy.sparse.linalg.expm(op)
    else:
        raise OfexTypeError(op)


def exact_rte(ham: Union[QubitOperator, np.ndarray, spmatrix],
              t: float,
              n_qubits: Optional[int] = None,
              exact_sparse=False):
    return exact_expop(ham * -1j * t, n_qubits, exact_sparse)


def exact_ite(ham: Union[QubitOperator, np.ndarray, spmatrix],
              beta: float,
              n_qubits: Optional[int] = None,
              exact_sparse=False):
    return exact_expop(ham * -beta, n_qubits, exact_sparse)
