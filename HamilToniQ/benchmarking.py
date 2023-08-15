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

from .unitity import build_cost_func, load_qiskit_sampler
from .matrices import *

Counts = Any
Circuit = Any


class Toniq:
    def __init__(self) -> None:
        self.Q = dim_4_var_8
        self.rep = 3
        self.n_layers = 2
        pass

    def run(
        self,
        #ansatz: Callable[[list, list, int], Circuit],
        #sampler: Callable[[Circuit], Counts],
    ) -> float:
        backend = Aer.get_backend('statevector_simulator')
        sampler = load_qiskit_sampler(backend, 1000)
        cost_func = build_cost_func(sampler, self.Q, self.n_layers)
        optimizer = COBYLA(maxiter=1000)
        init_params = [2 for i in range(2*self.n_layers)]
        result = optimizer.minimize(fun=cost_func, x0=init_params)
        return result.x
