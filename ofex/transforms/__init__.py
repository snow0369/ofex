"""
This module provides functions and utilities for transforming and working 
with fermions and qubits.

It includes:

- Conversions between fermion and qubit states/operators.
- Majorana to fermion conversion.
- Fermion rotation operations.
- Factorization-related tools for fermions.

Modules:

- **fermion_qubit**: Functions for fermion ↔ qubit state/operator conversions.
- **fermion_rotation**: Functions for handling fermion rotations.
- **majorana_fermion**: Tools for converting Majorana fermions to regular fermions.
- **fermion_factorization**: Additional utilities related to fermion factorization.
"""

from .fermion_qubit import (fermion_to_qubit_state, fermion_to_qubit_operator, fermion_to_qubit_state_general,
                            qubit_to_fermion_state)
from .fermion_rotation import fermion_rotation_state, fermion_rotation_operator
from .majorana_fermion import majorana_to_fermion
from . import fermion_factorization

__all__ = [
    "fermion_to_qubit_state", "fermion_to_qubit_operator", "fermion_to_qubit_state_general", "qubit_to_fermion_state",
    "fermion_rotation_state", "fermion_rotation_operator",
    "fermion_factorization",
    "majorana_fermion"
]
