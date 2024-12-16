"""
This module provides implementations of various quantum physics and chemistry models.

The module includes:
- Functions for defining 1D Pauli Hamiltonian models (`zz_1d`, `heisenberg_1d`, and `heisenberg_1d_ring`).
- A class `PolyacenePPP` for representing and working with polyacene molecules using the Pariser-Parr-Pople (PPP) model.
"""

from .pauli_hamiltonian import zz_1d, heisenberg_1d, heisenberg_1d_ring
from .polyacene_ppp import PolyacenePPP

__all__ = [
    "zz_1d", "heisenberg_1d", "heisenberg_1d_ring", "PolyacenePPP"
]
