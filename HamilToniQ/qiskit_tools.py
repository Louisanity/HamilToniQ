"""
This file contains the tools needed in the benchmarking of IBMQ
"""

from typing import Callable

from qiskit.providers.ibmq import IBMQ
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.result.counts import Counts

def get_fake_backend(backend_name: str) -> AerSimulator:
    # get a real backend from a real provider
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmq_manila')

    # generate a simulator that mimics the real quantum system with the latest calibration results
    backend_sim = AerSimulator.from_backend(backend)

    return backend_sim

def qiskit_sampler(backend: AerSimulator, n_samples: int, circuit: QuantumCircuit) -> Counts:
    # get the sampling results from either a simulator or a real hardware
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit)
    counts = job.result().get_counts()

    return Counts

def Ising_to_ansatz(pauli_terms, weights, nlayers, params):
    n_qubits = len(pauli_terms[0])
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.h(i)
    for j in range(nlayers):
        for k in range(len(pauli_terms)):
            term = pauli_terms[k]
            index_of_ones = []
            for l in range(len(term)):
                if term[l] == 1:
                    index_of_ones.append(l)
            if len(index_of_ones) == 1:
                circuit.rz(qubit=index_of_ones[0], phi=2 * weights[k] * params[2 * j])
            elif len(index_of_ones) == 2:
                circuit.rzz(
                    qubit1=index_of_ones[0],
                    qubit2=index_of_ones[1],
                    theta=weights[k] * params[2 * j],
                )
            else:
                raise ValueError("Invalid number of Z terms")

        for i in range(n_qubits):
            circuit.rx(i, theta=params[2 * j + 1])  # mixing terms
    return circuit