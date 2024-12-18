"""
This module initializes and simulates some quantum algorithm's building blocks.

The module contains imports and function declarations for performing
Hadamard tests, extended SWAP tests for QKSD (Quantum Krylov Subspace Diagonalization).

Available components:

- hadamard_test_general, hadamard_test_qubit_operator, hadamard_test_fermion_operator:
  Functions for performing Hadamard-based quantum tests.

- prepare_qksd_est_op, prepare_qksd_est_state, qksd_extended_swap_test:
  Tools for preparing and performing QKSD extended SWAP tests.

- sampling_base: Provides a base framework for sampling from probability distributions.
"""

from .hadamard_test import hadamard_test_general, hadamard_test_qubit_operator, hadamard_test_fermion_operator
from .qksd_extended_swap_test import prepare_qksd_est_op, prepare_qksd_est_state, qksd_extended_swap_test_run
from . import sampling_base

__all__ = ["hadamard_test_general", "hadamard_test_qubit_operator", "hadamard_test_fermion_operator",
           "prepare_qksd_est_op", "prepare_qksd_est_state", "qksd_extended_swap_test_run",
           "sampling_base"]
