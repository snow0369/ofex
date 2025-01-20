from typing import Union, Optional

import numpy as np
import scipy
from openfermion import QubitOperator, get_sparse_operator
from scipy.sparse import spmatrix

from ofex.exceptions import OfexTypeError
from ofex.linalg.pauli_expm import single_pauli_expm, reflective_pauli_expm


__all__ = ["exact_rte", "exact_ite"]

def exact_expop(op: Union[QubitOperator, np.ndarray, spmatrix],
                n_qubits: Optional[int] = None,
                exact_sparse=False,
                reflective: bool = False,
                check_reflective: bool = True) -> Union[spmatrix, np.ndarray]:
    """
    Compute the matrix exponential of a given operator (e^{op}).
    
    Parameters:
        op (Union[QubitOperator, np.ndarray, spmatrix]): The operator for which the exponential is calculated.
            Can be a `QubitOperator`, dense matrix (`np.ndarray`), or sparse matrix (`spmatrix`).
        n_qubits (Optional[int]): The number of qubits, required in some cases to determine the matrix dimensions.
        exact_sparse (bool): If True, converts a sparse matrix to dense for exact exponential computation.
            Defaults to False.
        reflective (bool): If True, uses the efficient reflective exponentiation method, limited to `QubitOperator`
            inputs that satisfy op*op = constant. Defaults to False.
        check_reflective (bool): If True, performs a check to verify the applicability of reflective exponentiation.
            Defaults to True.

    Returns:
        Union[spmatrix, np.ndarray]: The resulting matrix exponential as a dense or sparse matrix.

    Raises:
        OfexTypeError: If the input operator type is invalid.
    """
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
              exact_sparse=False)\
        -> Union[spmatrix, np.ndarray]:
    """
    Compute the real-time evolution operator for a given Hamiltonian.

    Parameters:
        ham (Union[QubitOperator, np.ndarray, spmatrix]): The Hamiltonian operator.
        t (float): The evolution time parameter.
        n_qubits (Optional[int]): The number of qubits, required in some cases to determine the matrix dimensions.
        exact_sparse (bool): If True, converts a sparse matrix to dense for exact exponential computation.
            Defaults to False.

    Returns:
        Union[spmatrix, np.ndarray]: The resulting matrix exponential representing the time evolution operator.
    """
    return exact_expop(ham * -1j * t, n_qubits, exact_sparse)


def exact_ite(ham: Union[QubitOperator, np.ndarray, spmatrix],
              beta: float,
              n_qubits: Optional[int] = None,
              exact_sparse=False)\
        -> Union[spmatrix, np.ndarray]:
    """
    Compute the imaginary-time evolution operator for a given Hamiltonian.

    Parameters:
        ham (Union[QubitOperator, np.ndarray, spmatrix]): The Hamiltonian operator.
        beta (float): The evolution parameter, representing inverse temperature.
        n_qubits (Optional[int]): The number of qubits, required in some cases to determine the matrix dimensions.
        exact_sparse (bool): If True, converts a sparse matrix to dense for exact exponential computation.
            Defaults to False.

    Returns:
        Union[spmatrix, np.ndarray]: The resulting matrix exponential representing the imaginary-time evolution operator.
    """
    return exact_expop(ham * -beta, n_qubits, exact_sparse)
