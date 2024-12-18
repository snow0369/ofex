# API Overview Document

---

## Introduction

This document provides an overview of the APIs available within the repository.  
The repository consists of two substructures, `ofex` (OpenFermion EXpansion) and `ofex_algorithms`.  
Leveraging OpenFermion and numerical libraries, `ofex` includes useful examples, techniques, and operations for quantum simulation.  
Based on this, `ofex_algorithms` includes quantum algorithms of interest to the project.

---

## Package Structure

The repository is organized as follows:
```text
ofex/
│
├── clifford/ # Clifford group operations and tools
│   ├── clifford_tools.py # Tools for manipulating Pauli tableaus
│   ├── pauli_diagonalization.py # Diagonalize Pauli operators using Clifford operations
│   ├── simulation.py # Clifford-based quantum simulation tools
│   └── standard_operators.py # Standard Clifford operations: Hadamard, S, CX, CZ, QSW
│
├── hamiltonian/ # Pauli Hamiltonian construction and related tools
│   └── pauli_hamiltonian.py # Build and manipulate Pauli Hamiltonians (chain, ring, Heisenberg models)
│
├── linalg/ # Linear algebra utilities
│   └── sparse_tools.py # Sparse matrix operations, expectation values, and diagonalizations
│
├── measurement/ # Measurement techniques and iterative coefficient splitting (ICS)
│   ├── killer_shift.py # Killer shift optimization for fermionic Hamiltonians
│   ├── iterative_coefficient_splitting/ 
│   │   ├── ics.py # Core ICS implementation 
│   │   ├── ics_prepare.py # Preparation routines for ICS 
│   │   └── ics_utils.py # Utility functions for ICS
│   └── sorted_insertion.py # Optimized sorted insertion for measurements
│
├── operators/ # Operator-related utilities
│   ├── fermion_operator_tools.py # Tools for manipulating Fermion operators 
│   ├── qubit_operator_tools.py # Tools for Qubit operators
│   └── symbolic_operator_tools.py # General symbolic operator manipulations 
│ 
├── propagator/ # Quantum propagators
│   ├── exact.py # Exact imaginary/real-time evolution
│   └── trotter.py # Trotter approximations for propagators
│
├── sampling_simulation/ # Sampling simulation and estimation tools
│   ├── hadamard_test.py # Hadamard test implementation for overlap estimation 
│   ├── qksd_extended_swap_test.py # Extended QKSD swap test 
│   └── sampling_base.py # Base utilities for sampling 
│
├── state/ # Quantum state tools
│   ├── binary_fock.py # Binary Fock state representation 
│   ├── chem_ref_state.py # Chemical reference states 
│   ├── state_tools.py # General state manipulation tools 
│   └── types.py # State type definitions 
│
├── transforms/ # Fermion-to-Qubit transformations
│   ├── fermion_qubit.py # Transform Fermion operators/states to Qubit representations
│   ├── fermion_rotation.py # Fermionic rotations
│   ├── fermion_factorization.py # Fermion operator factorizations
│   └── majorana_fermion.py # Majorana to Fermion transformations
│
└── utils/ # Miscellaneous utilities
    ├── binary.py # Binary utilities
    ├── chem.py # Chemical utilities
    └── dict_utils.py # Dictionary utilities

ofex_algorithms/ # Quantum algorithms
└── qksd/
    ├── qksd_simulation.py # QKSD simulation framework
    ├── qksd_utils.py # Utilities for QKSD algorithms
    └── example_script.py # Example QKSD simulation script

```