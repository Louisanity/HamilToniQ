"""
This Class is defined to give an overall score of the QAOA performance on a quantum hardware
"""

from typing import Callable, List, Any

import random
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer
from qiskit.primitives import Sampler, Estimator, BackendSampler
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA
from functools import partial

from unitity import (
    build_cost_func,
    load_qiskit_sampler,
    load_circuit,
    plot_energy_distribution,
    get_overlap,
    Q_to_paulis,
    store_process,
    symmetric_matrix_generator,
)
from matrices import *

Counts = Any
Circuit = Any
Hardware_Backend = Any


class Toniq:
    def __init__(self) -> None:
        self.Q = dim_4_var_6
        self.le = le_4_6
        self.rep = 1000
        self.n_layers = 6
        self.maxiter = 1000
        pass

    def standard_QAOA(self, backend: Hardware_Backend) -> None:
        # sampler = load_qiskit_sampler(backend, 1000)
        cost_func = build_cost_func(backend, self.Q, self.n_layers)
        optimizer = COBYLA(maxiter=self.maxiter)

        def get_result():
            init_params = [
                random.random() * 2 * np.pi for i in range(2 * self.n_layers)
            ]
            result = optimizer.minimize(fun=cost_func, x0=init_params)
            loaded_circuit = load_circuit(self.Q, self.n_layers, result.x)
            return get_overlap(loaded_circuit, self.le)

        # plot_energy_distribution(loaded_circuit, self.Q)
        overlaps = [get_result() for i in range(self.rep)]
        return overlaps

    def default_QAOA(self, backend):
        sampler = BackendSampler(backend=backend)
        optimizer = COBYLA(maxiter=self.maxiter)
        self.param_list = []
        self.energy_list = []
        callback = partial(store_process, self.param_list, self.energy_list)
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.n_layers,
            initial_point=[
                random.random() * 2 * np.pi for i in range(2 * self.n_layers)
            ],
            callback=callback,
        )
        op, _ = Q_to_paulis(self.Q)
        result = qaoa.compute_minimum_eigenvalue(op)
        return result

    def converge_distribution(self, dim, n_layers, n_cores=8):
        from multiprocessing import Pool

        backend = Aer.get_backend("aer_simulator")
        self.n_layers = n_layers

        def find_converge(dim):
            self.Q = symmetric_matrix_generator(dim)
            self.default_QAOA(backend)
            lowest = abs(np.mean(self.energy_list[-20:-1]))
            return np.where(self.energy_list < 0.99 * lowest)

        #with Pool(n_cores) as p:
        p = Pool(n_cores)
        distribution = p.map(find_converge, [dim for i in range(1000)])

        return distribution
    
    def score_auc(reference, data):
        # return the score of a backend at a certain dim and n_layers
        # This score is calculated using auc (area under the curve)
        # `reference` is pre-calculated data using noiseless simulator
        n_boxes = 1000 # the number of boxed
        prob = [i/n_boxes for i in range(n_boxes)] # the boundaries of boxes
        hist_data, _ = np.histogram(data, prob)
        hist_ref, _ = np.histogram(reference, prob)
        hist_data = [i/np.size(data, 0) for i in hist_data] # normalization
        hist_ref = [i/np.size(reference, 0) for i in hist_ref]
        score = 0
        for i, j in zip(hist_ref, hist_data):
            score += np.min(i, j)
        return score


    def portfolio_optimization(self, backend: Hardware_Backend) -> None:
        pass

    def total_benchmark(self, backend: Hardware_Backend) -> float:
        self.standard_QAOA(backend)
        self.portfolio_optimization(backend)
