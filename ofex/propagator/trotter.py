from typing import List

import numpy as np
from openfermion import QubitOperator
from scipy import sparse

from ofex.exceptions import OfexTypeError
from ofex.measurement.sorted_insertion import sorted_insertion
from ofex.operators.ordering import order_abs_coeff
from ofex.propagator.exact import exact_expop


def trotter_rte_by_si_lcu(ham: QubitOperator,
                          t: float,
                          n_qubits: int,
                          n_trotter: int,
                          exact_sparse: bool = False):
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
    pauli_list = order_abs_coeff(ham, reverse=True)
    pauli_list = [frag * -1j * t for frag in pauli_list]
    return _trotter_by_frag(pauli_list, n_qubits, n_trotter, reflective=True, check_reflective=False,
                            exact_sparse=exact_sparse)


def trotter_ite_by_si_lcu(ham: QubitOperator,
                          beta: float,
                          n_qubits: int,
                          n_trotter: int,
                          exact_sparse: bool = False):
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
