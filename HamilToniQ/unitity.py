from typing import List, Any, Callable, Dict

import numpy as np
from functools import partial
from qiskit import QuantumCircuit
from qiskit.providers.ibmq import IBMQ
from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.result.counts import Counts

matrix = Any
circuit = Any
counts = Dict


def return_hardness(return_vec: list) -> float:
    """
    A function calculating the QAOA hardness according of a return vector
    Args:
        return_vec (List(float)): the return vector
    return:
        hardness (float): the hardness in the range between -1 and 1.
    """
    hardness = np.var(return_vec)
    return hardness


def covariance_hardness(covariance: list) -> float:
    """
    A function calculating the QAOA hardness according of a covariance matrix
    Args:
        covariance (List(List(float)): the covariance matrix
    return:
        hardness (float): the hardness in the range between -1 and 1.
    """
    normalized_covariance = [
        covariance[i, j] / np.sqrt(covariance[i, i] * covariance[j, j])
        for j in range(i + 1)
        for i in range(len(covariance))
    ]
    hardness = np.var(normalized_covariance)
    return hardness


class instance_generator:
    def get_Q_without_constraint():
        Q = 0
        return Q

    def get_Q_with_constraint():
        Q = 0
        return Q


def counts_to_cost(counts: counts, Q: matrix) -> float:
    cost = 0
    n_samples = 0
    for key, value in counts.items():
        vector = [int(i) for i in key]
        cost += np.dot(vector, np.dot(Q, vector)) * value
        n_samples += value

    return cost / n_samples


def params_to_distribution(
    params: list[float],
    ansatz: Callable[[List[float]], circuit],
    Q: matrix,
    simulator: Callable[[circuit], counts],
) -> float:
    circuit = ansatz(params)
    counts = simulator(circuit)
    distribution = {}
    for key, value in counts.item:
        vector = [int(i) for i in key]
        energy = np.dot(vector, np.dot(Q, vector))
        distribution[energy] = value

    return distribution


def get_fake_backend(backend_name: str) -> AerSimulator:
    # get a real backend from a real provider
    provider = IBMQ.load_account()
    backend = provider.get_backend("ibmq_manila")

    # generate a simulator that mimics the real quantum system with the latest calibration results
    backend_sim = AerSimulator.from_backend(backend)

    return backend_sim


def qiskit_sampler(
    backend: AerSimulator, n_samples: int, circuit: QuantumCircuit
) -> Counts:
    # get the sampling results from either a simulator or a real hardware
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit)
    counts = job.result().get_counts()

    return counts


def Q_to_Ising(Q):
    # input is nxn symmetric numpy array corresponding to QUBO matrix Q

    n = np.shape(Q)[0]

    offset = np.triu(Q, 0).sum() / 2
    pauli_terms = []
    weights = []

    weights = -np.sum(Q, axis=1) / 2

    for i in range(n):
        term = np.zeros(n)
        term[i] = 1
        pauli_terms.append(term)

    for i in range(n - 1):
        for j in range(i + 1, n):
            term = np.zeros(n)
            term[i] = 1
            term[j] = 1
            pauli_terms.append(term)

            weight = Q[i][j] / 2
            weights = np.concatenate((weights, weight), axis=None)

    return pauli_terms, weights, offset


def Ising_to_ansatz(pauli_terms, weights, n_layers, params):
    n_qubits = len(pauli_terms[0])
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        circuit.h(i)
    for j in range(n_layers):
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
            circuit.rx(theta=params[2 * j + 1], qubit=i)  # mixing terms
    return circuit


def cost_func(sampler, Q, n_layers, params) -> float:
    pauli_terms, weights, offset = Q_to_Ising(Q)
    circuit = Ising_to_ansatz(pauli_terms, weights, n_layers, params)
    counts = sampler(circuit)
    return counts_to_cost(counts, Q)

def build_cost_func(sampler, Q, n_layers) -> Callable:
    return partial(cost_func, sampler, Q, n_layers)

def load_qiskit_sampler(backend, n_samples) -> Callable:
    return partial(qiskit_sampler, backend, n_samples)