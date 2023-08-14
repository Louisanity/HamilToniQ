import networkx as nx
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.circuit import Parameter
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import XXPlusYYGate
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.providers.fake_provider import FakeMelbourneV2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
import numpy as np
import qiskit
import math
import argparse

from typing import List

Circuit = qiskit.QuantumCircuit
Graph = nx.Graph


class Toniq:
    def __init__(self):
        self.shots = 2048
        self.problems = ["Standard", "Three-Regular", "SK"]
        self.mixers = ["x_mixer", "xy_mixer_ring", "xy_mixer_parity", "xy_mixer_full"]
        self.gamma_values = np.linspace(-1.5, 1.5, 10)
        self.beta_values = np.linspace(-1.5, 1.5, 10)
        self.backendStr = "qasm_simulator"

    # Grid/Graph creator - creates the problem which is solved
    def QAOA_Graph_Creator(self, label: str = "Standard") -> Graph:
        graph = nx.Graph()
        if label == "Standard":
            graph.add_nodes_from([0, 1, 2, 3])
            graph.add_edges_from(
                [(0, 1), (1, 2), (2, 3), (3, 0), (4, 1), (4, 2), (4, 3)]
            )
        elif label == "Three-Regular":
            graph.add_nodes_from(range(8))  # This will add nodes 0 to 7
            graph.add_edges_from(
                [
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 4),
                    (1, 5),
                    (2, 6),
                    (2, 7),
                    (3, 4),
                    (3, 7),
                    (4, 6),
                    (5, 6),
                    (5, 7),
                ]
            )
        elif label == "SK":  # SK = Sherrington-Kirkpatrick
            num_nodes = 7
            graph = nx.complete_graph(num_nodes)
        return graph

    def get_Graph_Complexity(self, label: str = "Standard"):
        adjacency = nx.adjacency_matrix(self.QAOA_Graph_Creator(label)).todense()
        return np.var(adjacency)

    # X-Mixer
    def x_mixer(self, qc: Circuit, n: int, b: float):
        for i in range(n):
            qc.rx(2 * b, i)
        return qc

    # XY-Mixer
    def xy_mixer_ring(self, qc, n, b):
        for i in range(n - 1):
            qc.append(XXPlusYYGate(2 * b, 0), [i, i + 1])
        qc.append(XXPlusYYGate(2 * b, 0), [0, n - 1])
        return qc

    def xy_mixer_full(self, qc, n, b):
        for i in range(n - 1):
            for j in range(i + 1, n):
                qc.append(XXPlusYYGate(2 * b), [i, j])
        return qc

    def xy_mixer_parity(self, qc, n, b):
        for i in range(n - 1)[::2]:
            qc.append(XXPlusYYGate(2 * b), [i, i + 1])
        for i in range(n - 1)[1::2]:
            qc.append(XXPlusYYGate(2 * b), [i, i + 1])
        return qc

    def get_mixer_from_string(self, mixerStr, qc, nqubits, beta):
        if mixerStr == "xy_mixer_full":
            qc = self.xy_mixer_full(qc, nqubits, beta)
        elif mixerStr == "xy_mixer_parity":
            qc = self.xy_mixer_parity(qc, nqubits, beta)
        elif mixerStr == "xy_mixer_ring":
            qc = self.xy_mixer_ring(qc, nqubits, beta)
        elif mixerStr == "x_mixer":
            qc = self.x_mixer(qc, nqubits, beta)
        return qc

    def maxcut_obj(self, solution, graph):
        obj = 0
        for i, j in graph.edges():
            if solution[i] != solution[j]:
                obj -= 1
        return obj

    def compute_expectation(self, counts, graph):
        avg = 0
        sum_count = 0
        for bit_string, count in counts.items():
            obj = self.maxcut_obj(bit_string, graph)
            avg += obj * count
            sum_count += count
        return avg / sum_count

    def create_qaoa_circ(self, graph, mixerStr, theta, no_meas=False):
        nqubits = len(graph.nodes())
        n_layers = len(theta) // 2  # number of alternating unitaries
        beta = theta[:n_layers]
        gamma = theta[n_layers:]

        qc = QuantumCircuit(nqubits)

        # initial_state
        qc.h(range(nqubits))

        for layer_index in range(n_layers):
            # problem unitary
            for _, pair in enumerate(list(graph.edges())):
                qc.rzz(2 * gamma[layer_index], pair[0], pair[1])
            # mixer unitary
            qc = self.get_mixer_from_string(mixerStr, qc, nqubits, beta[layer_index])

        if no_meas is False:
            qc.measure_all()
        return qc

    def get_expectation(self, graph, mixerStr, backendStr, shots):
        if backendStr == "fakeM":
            backend = FakeMelbourneV2()
        else:
            backend = Aer.get_backend(backendStr)
        backend.shots = shots

        def execute_circ(theta):
            qc = self.create_qaoa_circ(graph, mixerStr, theta)
            result = execute(qc, backend, shots=shots).result()
            counts = result.get_counts()
            return self.compute_expectation(counts, graph)

        return execute_circ

    def compute_expectation_sv(self, counts, graph):
        """Computes expectation value based on measurement results
        Args:
            counts: (dict) key as bit string, val as count
            graph: networkx graph
        Returns:
            avg: float
                 expectation value
        """
        avg = 0
        for bit_string, prob in counts.items():
            obj = self.maxcut_obj(bit_string, graph)
            avg += obj * prob
        return avg

    def get_expectation_sv(self, theta, graph, mixerStr, backendStr, shots):
        """Runs parametrized circuit
        Args:
            graph: networkx graph
        """
        if backendStr == "fakeM":
            backend = FakeMelbourneV2()
        else:
            backend = Aer.get_backend(backendStr)
        backend.shots = shots

        def execute_circ(theta):
            qc = self.create_qaoa_circ(graph, mixerStr, theta, no_meas=True)
            sv_result = Statevector(qc)
            sv = {}
            for i in range(2**qc.num_qubits):
                a = f"{bin(i)[2:]:0>{qc.num_qubits}}"
                sv[a] = np.abs(sv_result[i]) ** 2
            return self.compute_expectation_sv(sv, graph)

        return execute_circ(theta)

    def get_function_values_sv(
        self, gamma_values, beta_values, graph, mixer, backendStr, shots
    ):
        # Initialize an array to store the function values
        fun_values = np.zeros((len(gamma_values), len(beta_values)))

        # Run the QAOA circuit for each pair of gamma and beta values
        for i, gamma in enumerate(gamma_values):
            for j, beta in enumerate(beta_values):
                theta = [beta, gamma]
                expectation = self.get_expectation_sv(
                    theta, self.QAOA_Graph_Creator(graph), mixer, backendStr, shots
                )
                fun_values[i, j] = expectation
        return fun_values

    def detailed_analysis(self):
        # Calculating variances to compare the complexity of the graphs for benchmarking
        for problem in self.problems:
            print(self.get_Graph_Complexity(problem))

        # Analysis with noise-free simulator
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)
        backendStr = "qasm_simulator"

        for i in range(len(self.mixers)):
            axs[i, 0].set_ylabel("Beta")
            for j in range(len(self.problems)):
                values = self.get_function_values_sv(
                    self.gamma_values,
                    self.beta_values,
                    self.problems[j],
                    self.mixers[i],
                    backendStr,
                    self.shots,
                )
                axs[i, j].contourf(self.gamma_values, self.beta_values, values)
                axs[i, j].set_adjustable("box")

        for i in range(len(self.problems)):
            axs[0, i].set_title(self.problems[i])
            axs[len(self.mixers) - 1, i].set_xlabel("Gamma")

        plt.tight_layout()
        plt.savefig(
            "detailed_analysis.png", dpi=300
        )  # Save the figure in high resolution
        plt.show()

    def noisy_simulator_analysis(self):
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)
        backendStr = "fakeM"

        for i in range(len(self.mixers)):
            axs[i, 0].set_ylabel("Beta")
            for j in range(len(self.problems)):
                values = self.get_function_values_sv(
                    self.gamma_values,
                    self.beta_values,
                    self.problems[j],
                    self.mixers[i],
                    backendStr,
                    self.shots,
                )
                axs[i, j].contourf(self.gamma_values, self.beta_values, values)
                axs[i, j].set_adjustable("box")

        for i in range(len(self.problems)):
            axs[0, i].set_title(self.problems[i])
            axs[len(self.mixers) - 1, i].set_xlabel("Gamma")

        plt.tight_layout()
        plt.savefig(
            "noisy_simulator_analysis.png", dpi=300
        )  # Save the figure in high resolution
        plt.show()

    def statevector_analysis(self):
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)

        # Define the range of gamma and beta values - more than before for finer/better results
        gamma_values = np.linspace(-1.5, 1.5, 20)
        beta_values = np.linspace(-1.5, 1.5, 20)

        for i in range(len(self.mixers)):
            axs[i, 0].set_ylabel("Beta")
            for j in range(len(self.problems)):
                values = self.get_function_values_sv(
                    gamma_values,
                    beta_values,
                    self.problems[j],
                    self.mixers[i],
                    self.backendStr,
                    self.shots,
                )
                axs[i, j].contourf(gamma_values, beta_values, values)
                axs[i, j].set_adjustable("box")

        for i in range(len(self.problems)):
            axs[0, i].set_title(self.problems[i])
            axs[len(self.mixers) - 1, i].set_xlabel("Gamma")

        plt.tight_layout()
        plt.savefig(
            "statevector_analysis.png", dpi=300
        )  # Save the figure in high resolution
        plt.show()


# Usage:
toniq_instance = Toniq()
# toniq_instance.run()  # Uncomment if you want a run method
toniq_instance.detailed_analysis()  # Uncomment if you want a detailed_analysis method
toniq_instance.noisy_simulator_analysis()
toniq_instance.statevector_analysis()
