"""
This module provides tools and utilities for working with Clifford operators and Pauli operators
in the context of quantum computation. 

The module includes:

    - Conversion utilities between Pauli operators and tableau form (`pauli_to_tableau`, `tableau_to_pauli`).
    - Functions for performing and visualizing tableau arithmetic (`dot_tableau`, `str_tableau`, `str_tableau_side_by_side`).
    - Methods for working with finite fields (`is_zero_gf`, `is_equal_gf`).
    - Algorithms for diagonalizing sets of mutually-commuting Pauli operators (`diagonalizing_clifford`).
    - Simulation and unitary matrix application for Clifford operators (`clifford_apply`, `clifford_qiskit`, `clifford_simulation`, `clifford_unitary_mat`).

"""
from .clifford_tools import (pauli_to_tableau, tableau_to_pauli, dot_tableau, str_tableau, str_tableau_side_by_side,
                             is_zero_gf, is_equal_gf)
from .pauli_diagonalization import diagonalizing_clifford
from .simulation import clifford_apply, clifford_qiskit, clifford_simulation, clifford_unitary_mat

__all__ = [
    "pauli_to_tableau", "tableau_to_pauli", "dot_tableau", "str_tableau", "str_tableau_side_by_side",
    "is_zero_gf", "is_equal_gf", "diagonalizing_clifford", "clifford_apply", "clifford_qiskit", "clifford_simulation",
    "clifford_unitary_mat"
]
