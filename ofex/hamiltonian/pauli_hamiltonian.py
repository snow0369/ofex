import openfermion as of


def chain_pauli(num_qubits: int, p_type: str) -> of.QubitOperator:
    p_type = p_type.upper()
    if p_type not in ("X", "Y", "Z"):
        raise ValueError(f"Unknown pauli")
    qubit_op = of.QubitOperator()
    for idx_site in range(num_qubits - 1):
        qubit_op += of.QubitOperator(f"{p_type}{idx_site} {p_type}{idx_site + 1}")
    return qubit_op


def ring_pauli(num_qubits: int, p_type: str) -> of.QubitOperator:
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


def ZZ_1D(num_qubits: int) -> of.QubitOperator:
    qubit_op = chain_pauli(num_qubits, "Z")
    # return get_sparse_operator(qubit_op, num_qubits)
    return qubit_op


def heisenberg_1D(num_qubits) -> of.QubitOperator:
    qubit_op = chain_pauli(num_qubits, "X") + chain_pauli(num_qubits, "Y") + chain_pauli(num_qubits, "Z")
    # return get_sparse_operator(qubit_op, num_qubits)
    return qubit_op


def heisenberg_1D_ring(num_qubits) -> of.QubitOperator:
    qubit_op = ring_pauli(num_qubits, "X") + ring_pauli(num_qubits, "Y") + ring_pauli(num_qubits, "Z")
    # return get_sparse_operator(qubit_op, num_qubits)
    return qubit_op
