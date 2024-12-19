# OFEX: OpenFermion EXpansion Toolkit

---

## **Overview**

**OFEX** (OpenFermion EXpansion) is a Python toolkit that builds upon OpenFermion to provide tools for
**classical quantum simulation**, **measurement optimization**, **efficient classical data structures for quantum objects**
and **quantum algorithms** like QKSD (Quantum Krylov Subspace Diagonalization).  

It is modular, flexible, and focused on reducing quantum measurement costs and improving simulation performance.

---

## **Key Features**

- **Quantum State Tools**: Manipulate quantum states, Fock states, and chemical reference states.
- **Clifford Operations**: Simulate Clifford-based transformations and standard operations.
- **Measurement Optimization**: Perform Iterative Coefficient Splitting (ICS) and Killer Shift optimizations.
- **Quantum Operation Utilities**: Perform diagonalizations, expectation values, and operator applications.
- **Pauli Hamiltonians**: Construct chain, ring, and Heisenberg models.
- **Quantum Krylov Subspace Diagonalization (QKSD)**: Tools for simulating Krylov subspaces and advanced sampling.

---

## **Installation**

To install OFEX, clone this repository and install the dependencies:

```bash
git clone https://github.com/snow0369/ofex.git
cd ofex
pip install -r requirements.txt
```

We recommend to install `psi4` with 'Installer' option in [here](https://psicode.org/installs/latest/).
After then, add environment variable `PSI4PATH` for example:
```bash
# ~/.bashrc
...
export PSI4PATH=/home/username/path/to/psi4conda/bin/psi4
...
```

Also, replace `openfermionpyscf/_pyscf_molecular_data.py` with [this version](https://github.com/snow0369/OpenFermion-PySCF/blob/master/openfermionpyscf/_pyscf_molecular_data.py).

### **Dependencies**
- Python 3.8+
- NumPy
- SciPy
- OpenFermion
- Refer to requirements.txt for further dependencies.
---

## **Getting Started**

### **1. Expectation Value of Sparse Operators**
Operation with sparse and dense states.
```python
import numpy as np
from openfermion import QubitOperator

from ofex.linalg.sparse_tools import expectation
from ofex.state import BinaryFockVector, int_to_fock
from ofex.state.state_tools import state_allclose, pretty_print_state

# Define operators
z0, z1 = QubitOperator("Z0"), QubitOperator("Z1")

# Define states  |01>
state_0_dense = np.array([0, 0, 1, 0])  # A state can be np array
state_0_sparse = {BinaryFockVector((1, 0)): 1}  # or dict of (BinaryFockVector, complex)
state_0_int_to_fock = {int_to_fock(2, num_qubits=2): 1}  # Binary Fock vector can be initialized by an integer
print(state_allclose(state_0_dense, state_0_sparse)) # True
print(state_allclose(state_0_dense, state_0_int_to_fock)) # True  - They are all close to each other.
print(pretty_print_state(state_0_dense)) # +1 |01>

# Define a state  |10>
state_1_sparse = {BinaryFockVector((0, 1)): 1}

# Compute expectation value
print(expectation(z0, state_0_sparse))  # -1.0
print(expectation(z1, state_0_sparse))  #  1.0
print(expectation(z0, state_1_sparse))  #  1.0
print(expectation(z1, state_1_sparse))  # -1.0
```

### **2. Clifford Diagonalization**
```python

```


### **2. Iterative Coefficient Splitting (ICS)**
```python

```

### **4. Quantum Krylov Subspace Diagonalization (QKSD)**
```python

```

---

## **API Overview**

[Full API Documentation](https://snow0369.github.io/ofex/html/index.html)

| Module                          | Description                                                                                 |
|---------------------------------|---------------------------------------------------------------------------------------------|
| `ofex.clifford`                 | Clifford operations and tools (tableaus, Pauli diagonalization).                            |
| `ofex.hamiltonian`              | Hamiltonian examples (chain, ring, Heisenberg, PPP-Polyacene).                              |
| `ofex.linalg.sparse_tools`      | Sparse matrix tools: expectation, diagonalization, transitional amplitudes, etc.            |
| `ofex.measurement`              | Measurement optimization (Sorted Insertion, Iterative Coefficient Splitting, Killer Shift). |
| `ofex.propagator`               | Exact and Trotter-based propagators, with both of real and imaginary time evolutions.       |
| `ofex.sampling_simulation`      | Sampling-based simulation tools, including Hadamard tests.                                  |
| `ofex.state`                    | Quantum state tools, including binary Fock states.                                          |
| `ofex.transforms`               | Fermion-to-Qubit transformations and Fermion rotations and factorization.                   |
| `ofex_algorithms.qksd`          | QKSD algorithms and simulation tools.                                                       |

---

## **Contributing**

We are not yet prepared to accept contributions at this time.


---

## **License**

This project is licensed under the **MIT License**. See the LICENSE file for details.

---

## **Contact**

For issues, feedback, or contributions, reach out via the repository's [Issues page](https://github.com/snow0369/ofex/issues).

---

### **Acknowledgments**

This project builds upon the powerful OpenFermion library and numerical tools provided by NumPy and SciPy.
