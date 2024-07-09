from typing import Dict, Tuple

from ofex.operators.types import SinglePauli

PauliCovDict = Dict[Tuple[SinglePauli, SinglePauli], float]
TransitionPauliCovDict = Dict[Tuple[SinglePauli, SinglePauli], Tuple[float, float]]
PhasedTransitionalPauliCovDict = Dict[float, TransitionPauliCovDict]
