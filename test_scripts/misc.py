import numpy as np
from openfermion import QubitOperator, get_sparse_operator
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


def qiskit_test():
    ckt = QuantumCircuit(3)
    ckt.h(2)
    ckt.save_statevector()
    sv_sim = Aer.get_backend("aer_simulator_statevector")
    ckt = transpile(ckt, sv_sim)
    sv = sv_sim.run(ckt).result().get_statevector(ckt).data
    qubit_op = QubitOperator("X2")
    qubit_op = get_sparse_operator(qubit_op).toarray()
    assert np.allclose(qubit_op @ sv, sv)
