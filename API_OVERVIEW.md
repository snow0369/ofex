# API Overview Document

------

## Introduction
This document provides an overview of the APIs available within the repository.
The repository consists of two substructures, ```ofex```(OpenFermion EXpansion) and ```ofex_algorithms```.
Leveraging OpenFermion and numerical libraries,  ```ofex``` includes useful examples, techniques and operations for quantum simulation.
Based on this, ```ofex_algorithms``` includes quantum algorithms which are mainly interested in my project.

------

## Package Structure
The repository is organized as follows:
```text
ofex/
│
├── clifford/                # Clifford group operations and tools
│   └── clifford_tools.py
│
├── hamiltonian/             # Pauli Hamiltonian construction
│   └── pauli_hamiltonian.py
│
├── linalg/                  # Linear algebra utilities
│
├── measurement/             # Measurement simulations and iterative algorithms
│   ├── iterative_coefficient_splitting/
│   │   └── iterative_coefficient_splitting.py
│   └── ics_prepare.py
│
├── operators/               # Operator-related utilities
│
├── propagator/              # Quantum propagators
│
├── sampling_simulation/     # Sampling simulations
│
├── state/                   # Quantum state tools
│
├── transforms/              # Transformations (fermion to qubit, etc.)
│
└── utils/                   # Miscellaneous utilities

ofex_algorithms/         # Quantum Krylov subspace algorithms
└── qksd/
    ├── qksd_simulation.py
    ├── qksd_utils.py
    └── example_script.py


```