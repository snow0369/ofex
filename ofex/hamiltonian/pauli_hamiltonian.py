import openfermion as of

__all__ = ["ring_pauli", "zz_1d", "heisenberg_1d", "heisenberg_1d_ring"]

def chain_pauli(num_qubits: int, p_type: str) -> of.QubitOperator:
    """
    Generate the Pauli-based chain Hamiltonian as a QubitOperator.

    Args:
        num_qubits (int): The number of qubits in the system.
        p_type (str): The type of Pauli operator ('X', 'Y', 'Z').

    Returns:
        of.QubitOperator: The QubitOperator representing the chain Hamiltonian.
    """
    p_type = p_type.upper()
    if p_type not in ("X", "Y", "Z"):
        raise ValueError(f"Unknown pauli")
    qubit_op = of.QubitOperator()
    for idx_site in range(num_qubits - 1):
        qubit_op += of.QubitOperator(f"{p_type}{idx_site} {p_type}{idx_site + 1}")
    return qubit_op


def ring_pauli(num_qubits: int, p_type: str) -> of.QubitOperator:
    """
    Generate the Pauli-based ring Hamiltonian as a QubitOperator.

    Args:
        num_qubits (int): The number of qubits in the system.
        p_type (str): The type of Pauli operator ('X', 'Y', 'Z').

    Returns:
        of.QubitOperator: The QubitOperator representing the ring Hamiltonian, 
                          where the last qubit also interacts with the first.
    """
    p_type = p_type.upper()
    if p_type not in ("X", "Y", "Z"):
        raise ValueError(f"Unknown pauli")
    qubit_op = of.QubitOperator()
    for idx_site in range(num_qubits):
        if idx_site < num_qubits - 1:
            qubit_op += of.QubitOperator(f"{p_type}{idx_site} {p_type}{(idx_site + 1)}")
        else:
            qubit_op += of.QubitOperator(f"{p_type}{idx_site} {p_type}{0}")
    return qubit_op


def zz_1d(num_qubits: int) -> of.QubitOperator:
    """
    Generate the Z-Z interaction Hamiltonian for a 1D chain of qubits.

    Args:
        num_qubits (int): The number of qubits in the system.

    Returns:
        of.QubitOperator: The QubitOperator representing the Z-Z Hamiltonian.
    """
    qubit_op = chain_pauli(num_qubits, "Z")
    # return get_sparse_operator(qubit_op, num_qubits)
    return qubit_op


def heisenberg_1d(num_qubits) -> of.QubitOperator:
    """
    Generate the Heisenberg Hamiltonian for a 1D open chain of spins.

    Args:
        num_qubits (int): The number of qubits in the system.

    Returns:
        of.QubitOperator: The QubitOperator representing the Heisenberg Hamiltonian.
    """
    qubit_op = chain_pauli(num_qubits, "X") + chain_pauli(num_qubits, "Y") + chain_pauli(num_qubits, "Z")
    # return get_sparse_operator(qubit_op, num_qubits)
    return qubit_op


def heisenberg_1d_ring(num_qubits) -> of.QubitOperator:
    """
    Generate the Heisenberg Hamiltonian for a 1D closed ring of spins.

    Args:
        num_qubits (int): The number of qubits in the system.

    Returns:
        of.QubitOperator: The QubitOperator representing the Heisenberg Hamiltonian on a closed ring.
    """
    qubit_op = ring_pauli(num_qubits, "X") + ring_pauli(num_qubits, "Y") + ring_pauli(num_qubits, "Z")
    # return get_sparse_operator(qubit_op, num_qubits)
    return qubit_op
