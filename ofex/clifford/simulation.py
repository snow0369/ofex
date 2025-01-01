from typing import Optional, List, Tuple

import numpy as np
from galois import FieldArray
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from ofex.clifford.clifford_tools import gf
from ofex.clifford.standard_operators import hadamard, s_gate, cx, cz
from ofex.state.state_tools import get_num_qubits, to_dense
from ofex.state.types import State

__all__ = ["clifford_apply", "clifford_qiskit", "clifford_simulation", "clifford_unitary_mat"]


def clifford_apply(mat: FieldArray,
                   ph: Optional[FieldArray],
                   clifford_hist: List[str])\
        ->Tuple[FieldArray, FieldArray]:
    """
    Applies a sequence of Clifford operations to a stabilizer matrix and phase vector.

    Args:
        mat (FieldArray): The stabilizer matrix representing the quantum state in a symplectic form.
        ph (Optional[FieldArray]): The phase vector associated with the quantum state. If None, 
                                   a zero vector with appropriate dimensions is used.
        clifford_hist (List[str]): A list of Clifford operations to apply, specified in string format. 
                                   Supported operations:
                                   - "H_<index>" for Hadamard gate applied to qubit at <index>.
                                   - "S_<index>" for S gate applied to qubit at <index>.
                                   - "CX_<index1>_<index2>" for CNOT gate from control qubit <index1> to target qubit <index2>.
                                   - "CZ_<index1>_<index2>" for CZ gate between qubits at <index1> and <index2>.
                                   - "QSW_<index1>_<index2>" for qubit label swap between qubits at <index1> and <index2>.

    Returns:
        Tuple[FieldArray, FieldArray]: A tuple containing the updated stabilizer matrix and phase vector after 
                                       applying the Clifford operations.
    """
    
    if ph is None:
        ph = gf.Zeros(mat.shape[1])
    num_qubits = mat.shape[0] // 2
    q_lbl: List[int] = list(range(num_qubits))
    for op in clifford_hist:
        if op.startswith("H"):
            idx = int(op.split("_")[-1])
            mat, ph = hadamard(mat, ph, q_lbl[idx])
        elif op.startswith("S"):
            idx = int(op.split("_")[-1])
            mat, ph = s_gate(mat, ph, q_lbl[idx])
        elif op.startswith("CX"):
            idx1, idx2 = int(op.split("_")[-2]), int(op.split("_")[-1])
            mat, ph = cx(mat, ph, q_lbl[idx1], q_lbl[idx2])
        elif op.startswith("CZ"):
            idx1, idx2 = int(op.split("_")[-2]), int(op.split("_")[-1])
            mat, ph = cz(mat, ph, q_lbl[idx1], q_lbl[idx2])
        elif op.startswith("QSW"):
            idx1, idx2 = int(op.split("_")[-2]), int(op.split("_")[-1])
            q_lbl[idx1], q_lbl[idx2] = q_lbl[idx2], q_lbl[idx1]
        else:
            raise ValueError
    return mat, ph


def clifford_qiskit(num_qubits: int,
                    clifford_hist: List[str],
                    init_state: Optional[State],
                    inv=False)\
        -> QuantumCircuit:
    """
    Constructs the Qiskit equivalent of a quantum circuit from a given sequence of Clifford operations.

    Args:
        num_qubits (int): The total number of qubits in the circuit.
        clifford_hist (List[str]): A list of Clifford operations (e.g., ``"H_<index>"``, ``"S_<index>"``,
                                   ``"CX_<index1>_<index2>"``, ``"CZ_<index1>_<index2>"``,
                                   ``"QSW_<index1>_<index2>"``).
        init_state (Optional[State]): The initial quantum state. If None, the circuit starts in the
                                      default ``|0...0⟩`` state.
        inv (bool): If True, the operations in the Clifford history are applied in reverse order.
                    Defaults to False.

    Returns:
        QuantumCircuit: A Qiskit quantum circuit representing the applied Clifford operations.
    """
    if init_state is not None:
        num_qubits = get_num_qubits(init_state)
        init_state = to_dense(init_state)
    q_lbl: List[int] = list(range(num_qubits))[::-1]
    if inv:
        clifford_hist = clifford_hist[::-1]
    ckt = QuantumCircuit(num_qubits)
    if init_state is not None:
        ckt.initialize(init_state, ckt.qubits)
    for op in clifford_hist:
        if op.startswith("H"):
            idx = int(op.split("_")[-1])
            ckt.h(q_lbl[idx])
        elif op.startswith("S"):
            idx = int(op.split("_")[-1])
            if inv:
                ckt.sdg(q_lbl[idx])
            else:
                ckt.s(q_lbl[idx])
        elif op.startswith("CX"):
            idx1, idx2 = int(op.split("_")[-2]), int(op.split("_")[-1])
            ckt.cx(q_lbl[idx1], q_lbl[idx2])
        elif op.startswith("CZ"):
            idx1, idx2 = int(op.split("_")[-2]), int(op.split("_")[-1])
            ckt.cz(q_lbl[idx1], q_lbl[idx2])
        elif op.startswith("QSW"):
            idx1, idx2 = int(op.split("_")[-2]), int(op.split("_")[-1])
            q_lbl[idx1], q_lbl[idx2] = q_lbl[idx2], q_lbl[idx1]
        else:
            raise ValueError
    return ckt


def clifford_simulation(init_state: State,
                        clifford_history: List[str],
                        inv=False)\
        -> np.ndarray:
    """
    Simulates the action of a sequence of Clifford operations on an initial quantum state.

    Args:
        init_state (State): The initial quantum state vector as a NumPy array.
        clifford_history (List[str]): A list of Clifford operations (e.g., "H_<index>", "S_<index>", 
                                      "CX_<index1>_<index2>", "CZ_<index1>_<index2>", 
                                      "QSW_<index1>_<index2>"), applied in order.
        inv (bool): If True, the operations in the Clifford history are applied in reverse order. 
                    Defaults to False.

    Returns:
        np.ndarray: The resulting quantum state vector after applying the Clifford operations.
    """
    init_state = to_dense(init_state)
    ckt = clifford_qiskit(get_num_qubits(init_state), clifford_history, init_state, inv)
    ckt.save_statevector()
    sv_sim = Aer.get_backend("aer_simulator_statevector")
    ckt = transpile(ckt, sv_sim)
    sv = sv_sim.run(ckt).result().get_statevector(ckt).data
    return sv


def clifford_unitary_mat(clifford_history,
                         num_qubits,
                         inv=False)\
        -> np.ndarray:
    """
    Constructs and returns the unitary matrix representation of a sequence of Clifford operations.

    Args:
        clifford_history (List[str]): A list of Clifford operations to apply, specified in string format, 
                                      e.g., "H_<index>", "S_<index>", "CX_<index1>_<index2>", 
                                      "CZ_<index1>_<index2>", "QSW_<index1>_<index2>".
        num_qubits (int): The total number of qubits for which the unitary matrix is to be constructed.
        inv (bool): If True, the operations in the Clifford history are applied in reverse order. Defaults to False.

    Returns:
        np.ndarray: The unitary matrix representation of the specified Clifford operations.
    """
    ckt = clifford_qiskit(num_qubits, clifford_history, init_state=None, inv=inv)
    unitary_sim = Aer.get_backend("unitary_simulator")
    ckt = transpile(ckt, unitary_sim)
    unitary = np.array(unitary_sim.run(ckt).result().get_unitary(ckt))
    return unitary
