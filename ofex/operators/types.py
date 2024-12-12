from typing import Union, Tuple

from openfermion import SymbolicOperator, QubitOperator, FermionOperator

__all__ = ["Operators", "SinglePauli", "SingleFermion"]


Operators = Union[SymbolicOperator, QubitOperator, FermionOperator]
SinglePauli = Tuple[Tuple[int, str], ...]  # First int = idx, Second idx = "X", "Y", "Z", "I"
SingleFermion = Tuple[Tuple[int, int], ...]  # First int = idx, Second idx = dagger
