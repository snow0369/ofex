"""
This module defines type aliases for various dictionaries related to Pauli operators covariance calculations.
For detailed information about these types and their usage, please refer to `ofex.measurement.pauli_covariance` method.
"""

from typing import Dict, Tuple
from ofex.operators.types import SinglePauli

PauliCovDict = Dict[Tuple[SinglePauli, SinglePauli], float]
TransitionPauliCovDict = Dict[Tuple[SinglePauli, SinglePauli], Tuple[float, float]]
PhasedTransitionalPauliCovDict = Dict[float, TransitionPauliCovDict]

__all__ = ["PauliCovDict", "TransitionPauliCovDict", "PhasedTransitionalPauliCovDict"]
