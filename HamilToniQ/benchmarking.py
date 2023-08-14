"""
This Class is defined to give an overall score of the QAOA performance on a quantum hardware
"""

from typing import Callable, List, Any
import sys
sys.path.append('..')

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from functools import partial

from .qiskit_tools import Ising_to_ansatz, qiskit_sampler
from .unitity import Q_to_Ising
from .matrices import *

Counts = Any
Circuit = Any

class Toniq:
    def __init__(self) -> None:
        self.Q = dim_4_var_8
        self.rep = 3
        pass

    def run(self, ansatz: Callable[[list, list, int], Circuit], sampler: Callable[[Circuit], Counts]) -> float:
        paulis_terms, weights, offset = Q_to_Ising(self.Q)
        loaded_ansatz = partial(ansatz, paulis_terms, weights, self.rep)

        counts = sampler(loaded_ansatz)
        cost = self.counts_to_cost(counts, Q)
        