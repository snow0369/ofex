"""
This module defines common data types and aliases used for working with symbolic, 
qubit, and fermionic operators in quantum computing.

Exports:
    - Operators: A union of SymbolicOperator, QubitOperator, and FermionOperator.
    - SinglePauli: A tuple describing a single Pauli operator with its index and type.
    - SingleFermion: A tuple describing a fermion operator with its index and dagger status.
"""

from typing import Union, Tuple

from openfermion import SymbolicOperator, QubitOperator, FermionOperator

__all__ = ["Operators", "SinglePauli", "SingleFermion"]


Operators = Union[SymbolicOperator, QubitOperator, FermionOperator]
SinglePauli = Tuple[Tuple[int, str], ...]  # First int = idx, Second idx = "X", "Y", "Z", "I"
SingleFermion = Tuple[Tuple[int, int], ...]  # First int = idx, Second idx = dagger
