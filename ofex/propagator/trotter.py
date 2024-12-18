from typing import List

import numpy as np
from openfermion import QubitOperator
from scipy import sparse

from ofex.exceptions import OfexTypeError
from ofex.measurement import sorted_insertion
from ofex.operators.ordering import order_abs_coeff
from ofex.propagator.exact import exact_expop


__all__ = [
    "trotter_rte_by_si_ref", "trotter_rte_by_si_comm", "trotter_rte_by_single_pauli",
    "trotter_ite_by_si_ref", "trotter_ite_by_si_comm", "trotter_ite_by_single_pauli"
]

def trotter_rte_by_si_ref(ham: QubitOperator,
                          t: float,
                          n_qubits: int,
                          n_trotter: int,
                          exact_sparse: bool = False):
    """
    Perform real-time evolution using the Suzuki-Trotter decomposition 
    with sorted insertion and a reflective strategy.
    
    The reflective approach is more efficient for classical simulation 
    compared to the commuting grouping strategy.
    
    Args:
        ham (QubitOperator): The Hamiltonian operator.
        t (float): The evolution time parameter.
        n_qubits (int): The total number of qubits in the system.
        n_trotter (int): The number of Trotter steps for the decomposition.
        exact_sparse (bool, optional): If True, use an exact sparse matrix 
            representation for computation. Defaults to False.
    
    Returns:
        ndarray or spmatrix: The evolution operator.
    """
    lcu_ham = sorted_insertion(ham, anticommute=True)
    lcu_ham = sorted(lcu_ham, key=lambda x: x.induced_norm(order=2), reverse=True)
    lcu_ham = [frag * -1j * t for frag in lcu_ham]
    return _trotter_by_frag(lcu_ham, n_qubits, n_trotter, reflective=True, check_reflective=False,
                            exact_sparse=exact_sparse)


def trotter_rte_by_si_comm(ham: QubitOperator,
                           t: float,
                           n_qubits: int,
                           n_trotter: int,
                           exact_sparse: bool = False):
    """
    Perform real-time evolution using the Suzuki-Trotter decomposition
    with sorted insertion and commuting terms grouping.

    Args:
        ham (QubitOperator): The Hamiltonian operator.
        t (float): The evolution time parameter.
        n_qubits (int): The total number of qubits in the system.
        n_trotter (int): The number of Trotter steps for the decomposition.
        exact_sparse (bool, optional): If True, use an exact sparse matrix
            representation for computation. Defaults to False.

    Returns:
        ndarray or spmatrix: The evolution operator.
    """
    comm_ham = sorted_insertion(ham, anticommute=False)
    comm_ham = sorted(comm_ham, key=lambda x: x.induced_norm(order=2), reverse=True)
    comm_ham = [frag * -1j * t for frag in comm_ham]
    return _trotter_by_frag(comm_ham, n_qubits, n_trotter, reflective=False, check_reflective=False,
                            exact_sparse=exact_sparse)


def trotter_rte_by_single_pauli(ham: QubitOperator,
                                t: float,
                                n_qubits: int,
                                n_trotter: int,
                                exact_sparse: bool = False):
    """
    Perform real-time evolution using the Suzuki-Trotter decomposition 
    with a single Pauli operator ordering.

    Args:
        ham (QubitOperator): The Hamiltonian operator.
        t (float): The evolution time parameter.
        n_qubits (int): The total number of qubits in the system.
        n_trotter (int): The number of Trotter steps for the decomposition.
        exact_sparse (bool, optional): If True, use an exact sparse matrix
            representation for computation. Defaults to False.

    Returns:
        ndarray or spmatrix: The evolution operator.
    """
    pauli_list = order_abs_coeff(ham, reverse=True)
    pauli_list = [frag * -1j * t for frag in pauli_list]
    return _trotter_by_frag(pauli_list, n_qubits, n_trotter, reflective=True, check_reflective=False,
                            exact_sparse=exact_sparse)


def trotter_ite_by_si_ref(ham: QubitOperator,
                          beta: float,
                          n_qubits: int,
                          n_trotter: int,
                          exact_sparse: bool = False):
    """
    Perform imaginary-time evolution using the Suzuki-Trotter decomposition
    with sorted insertion and a reflective strategy.

    The reflective approach is more efficient for classical simulation
    compared to the commuting grouping strategy.

    Args:
        ham (QubitOperator): The Hamiltonian operator.
        beta (float): Imaginary time evolution parameter.
        n_qubits (int): The total number of qubits in the system.
        n_trotter (int): The number of Trotter steps for the decomposition.
        exact_sparse (bool, optional): If True, use an exact sparse matrix
            representation for computation. Defaults to False.

    Returns:
        ndarray or spmatrix: The evolution operator.
    """
    lcu_ham = sorted_insertion(ham, anticommute=True)
    lcu_ham = sorted(lcu_ham, key=lambda x: x.induced_norm(order=2), reverse=True)
    lcu_ham = [frag * -beta for frag in lcu_ham]
    return _trotter_by_frag(lcu_ham, n_qubits, n_trotter, reflective=True, check_reflective=False,
                            exact_sparse=exact_sparse)


def trotter_ite_by_si_comm(ham: QubitOperator,
                           beta: float,
                           n_qubits: int,
                           n_trotter: int,
                           exact_sparse: bool = False):
    """
    Perform imaginary-time evolution using the Suzuki-Trotter decomposition
    with sorted insertion and commuting terms grouping.

    Args:
        ham (QubitOperator): The Hamiltonian operator.
        beta (float): Imaginary time evolution parameter.
        n_qubits (int): The total number of qubits in the system.
        n_trotter (int): The number of Trotter steps for the decomposition.
        exact_sparse (bool, optional): If True, use an exact sparse matrix
            representation for computation. Defaults to False.

    Returns:
        ndarray or spmatrix: The evolution operator.
    """
    comm_ham = sorted_insertion(ham, anticommute=False)
    comm_ham = sorted(comm_ham, key=lambda x: x.induced_norm(order=2), reverse=True)
    comm_ham = [frag * -beta for frag in comm_ham]
    return _trotter_by_frag(comm_ham, n_qubits, n_trotter, reflective=False, check_reflective=False,
                            exact_sparse=exact_sparse)


def trotter_ite_by_single_pauli(ham: QubitOperator,
                                beta: float,
                                n_qubits: int,
                                n_trotter: int,
                                exact_sparse: bool = False):
    """
    Perform imaginary-time evolution using the Suzuki-Trotter decomposition 
    with a single Pauli operator ordering.

    Args:
        ham (QubitOperator): The Hamiltonian operator.
        beta (float): Imaginary time evolution parameter.
        n_qubits (int): Number of qubits.
        n_trotter (int): The number of Trotter steps.
        exact_sparse (bool, optional): Flag to indicate if exact sparse 
            matrix representation should be used. Defaults to False.

    Returns:
        ndarray or spmatrix: The resulting matrix after simulation.
    """
    pauli_list = order_abs_coeff(ham, reverse=True)
    pauli_list = [frag * -beta for frag in pauli_list]
    return _trotter_by_frag(pauli_list, n_qubits, n_trotter, reflective=True, check_reflective=False,
                            exact_sparse=exact_sparse)


def _trotter_by_frag(frag_list: List[QubitOperator],
                     n_qubits: int,
                     n_trotter: int,
                     reflective: bool,
                     check_reflective: bool,
                     exact_sparse: bool = False,
                     spmatrix_format: str = 'csc'):
    if len(frag_list) == 0:
        if exact_sparse:
            return np.eye(2 ** n_qubits, dtype=complex)
        else:
            return sparse.identity(2 ** n_qubits, dtype=complex, format=spmatrix_format)

    trot = exact_expop(frag_list[0] / n_trotter, n_qubits, exact_sparse, reflective, check_reflective)
    for frag in frag_list[1:]:
        trot = exact_expop(frag / n_trotter, n_qubits, exact_sparse, reflective, check_reflective) @ trot

    if n_trotter > 1:
        if isinstance(trot, sparse.spmatrix):
            return sparse.linalg.matrix_power(trot, n_trotter)
        elif isinstance(trot, np.ndarray):
            return np.linalg.matrix_power(trot, n_trotter)
        else:
            raise OfexTypeError(trot)
    else:
        return trot
