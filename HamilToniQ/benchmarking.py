"""
This Class is defined to give an overall score of the QAOA performance on a quantum hardware
"""

from typing import Callable, List, Any
import sys

sys.path.append("..")

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer
from qiskit.algorithms.optimizers import COBYLA
from functools import partial

from .unitity import build_cost_func, load_qiskit_sampler, load_circuit, plot_energy_distribution
from .matrices import *

Counts = Any
Circuit = Any
Hardware_Backend = Any


class Toniq:
    def __init__(self) -> None:
        self.Q = dim_4_var_8
        self.rep = 3
        self.n_layers = 6
        pass

    def standard_QAOA(self, backend: Hardware_Backend) -> None:
        #sampler = load_qiskit_sampler(backend, 1000)
        cost_func = build_cost_func(backend, self.Q, self.n_layers)
        optimizer = COBYLA(maxiter=1000)
        init_params = [2 for i in range(2 * self.n_layers)]
        result = optimizer.minimize(fun=cost_func, x0=init_params)
        loaded_circuit = load_circuit(self.Q, self.n_layers, result.x)
        plot_energy_distribution(loaded_circuit, self.Q)
        return result.x

    def portfolio_optimization(self, backend: Hardware_Backend) -> None:
        pass

    def total_benchmark(self, backend: Hardware_Backend) -> float:
        self.standard_QAOA(backend)
        self.portfolio_optimization(backend)
