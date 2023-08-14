from typing import List, Any, Callable, Dict

import numpy as np
from qiskit import QuantumCircuit

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
    for key, value in counts.items:
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

def Q_to_Ising(Q):
    # input is nxn symmetric numpy array corresponding to QUBO matrix Q

    n = Q.shape[0]

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