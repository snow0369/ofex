"""
This module provides various techniques for optimal quantum measurements.

It includes functionality for managing measurement variance, grouping Pauli operators,
and other related tools for efficient quantum computations. Specific techniques,
such as the iterative coefficient splitting and various grouping algorithms, are
also provided.
"""

from .killer_shift import killer_shift_opt_fermion_hf
from .measurement_variance import fragment_variance, pauli_covariance, empirical_pauli_covariance
from .pauli_grouping import sorted_insertion, iterative_sorted_insertion, optimal_sorted_insertion, pauli_split_group
from . import iterative_coefficient_splitting, types

__all__ = [
    "killer_shift_opt_fermion_hf", "fragment_variance", "pauli_covariance", "empirical_pauli_covariance",
    "sorted_insertion", "iterative_sorted_insertion", "optimal_sorted_insertion", "pauli_split_group",
    "iterative_coefficient_splitting", "types"
]
