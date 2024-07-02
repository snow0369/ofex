from typing import Optional, List

import numpy as np
from galois import FieldArray
from qiskit import QuantumCircuit, transpile, Aer

from ofex.clifford.clifford_tools import gf
from ofex.clifford.standard_operators import hadamard, s_gate, cx, cz
from ofex.state.state_tools import get_num_qubits, pretty_print_state


def clifford_apply(mat: FieldArray,
                   ph: Optional[FieldArray],
                   clifford_hist: List[str]):
    if ph is None:
        ph = gf(np.zeros(mat.shape[1], dtype=int))
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


def clifford_qiskit(num_qubits: int, clifford_hist: List[str], init_state: Optional[np.ndarray], inv=False):
    if init_state is not None:
        num_qubits = get_num_qubits(init_state)
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


def clifford_simulation(init_state: np.ndarray, clifford_history: List[str], inv=False) -> np.ndarray:
    from qiskit.providers.aer import Aer
    ckt = clifford_qiskit(get_num_qubits(init_state), clifford_history, init_state, inv)
    ckt.save_statevector()
    sv_sim = Aer.get_backend("aer_simulator_statevector")
    ckt = transpile(ckt, sv_sim)
    sv = sv_sim.run(ckt).result().get_statevector(ckt).data
    return sv


def clifford_unitary_mat(clifford_history, num_qubits, inv=False) -> np.ndarray:
    ckt = clifford_qiskit(num_qubits, clifford_history, init_state=None, inv=inv)
    unitary_sim = Aer.get_backend("unitary_simulator")
    ckt = transpile(ckt, unitary_sim)
    unitary = np.array(unitary_sim.run(ckt).result().get_unitary(ckt))
    return unitary


if __name__ == "__main__":
    def qiskit_test():
        from qiskit.providers.aer import Aer
        from openfermion import QubitOperator, get_sparse_operator

        ckt = QuantumCircuit(3)
        ckt.h(2)
        ckt.save_statevector()
        sv_sim = Aer.get_backend("aer_simulator_statevector")
        ckt = transpile(ckt, sv_sim)
        sv = sv_sim.run(ckt).result().get_statevector(ckt).data
        qubit_op = QubitOperator("X2")
        qubit_op = get_sparse_operator(qubit_op).toarray()
        print(np.allclose(qubit_op @ sv, sv))
        print(pretty_print_state(sv))


    qiskit_test()
