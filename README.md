# HamilToniQ: Comprehensive Optimization and Benchmarking for Mixer Hamiltonian in QAOA with Error Mitigation
This is the repository for IBM Quantum Hackathon 2023 at the World of Quantum with the topic of encoding (Mixer Hamiltonian)

![alt text](https://github.com/Louisanity/HamilToniQ//blob/main/pictures/overview.png?raw=true)

## Project Summary
1. Problem Formulation
2. Mixer Hamiltonian Formulation
3. Post-processing of mixer Hamiltonian
4. Error mitigation
5. QAOA Circuit Depth Reduction
7. QASM, Noisy Simulator and Real Hardware Benchmark

### Motivation
Quantum computing is an emerging field with the potential to revolutionize various sectors. The Quantum Approximate Optimization Algorithm (QAOA) is a promising quantum algorithm for near-term devices, but its performance is often limited by the depth of the quantum circuit (Hamiltonian Formation). To address this, we are developing an open-source benchmarking kit for QAOA based on Qiskit, a popular quantum computing framework. This project, the first of its kind on GitHub, will focus on depth reduction techniques for QAOA circuits to improve their performance on real quantum devices. Additionally, we will leverage the Qiskit Runtime function and incorporate error mitigation techniques, enabling comprehensive benchmarking on both Qiskit's simulators and real quantum hardware. Our project aims to contribute to the practical development of quantum computing by providing a valuable resource for the quantum computing community.

### Quantum Approximate Optimization Algorithm (QAOA)
The Quantum Approximate Optimization Algorithm (QAOA)is a hybrid quantum-classical algorithm that has been proposed as a practical method to solve combinatorial optimization problems on near-term, noisy intermediate-scale quantum (NISQ) devices. QAOA operates by approximating the ground state of a problem Hamiltonian (H_P), which encodes the optimization problem to be solved.

### Problem formulation
We study Quantum Approximate Optimization Algorithm (QAOA) on three distinct
problem types: Hardware Grid problem, the Three Regular problem, and the Sherrington-Kirkpatrick (SK) model problem.
![alt text](https://github.com/Louisanity/HamilToniQ//blob/main/pictures/problem_set.png?raw=true)

- SK model is a well-known model in statistical mechanics, representing a system of spins with random interactions.
In the context of QAOA, the SK model problem is a fully connected graph, where each node (or spin) can interact with
every other.
- Three Regular problem has each node in the graph connected to exactly three others, forming a ’3-regular’ graph. It represents a balance between the highly constrained hardware grid problem and the fully connected SK model problem.
- Hardware Grid problem represents physical layout of qubits in quantum hardware. The qubits are arranged in a
two-dimensional grid, and interactions are allowed between neighboring qubits. This problem type is particularly
relevant for near-term quantum devices, as it mirrors the connectivity constraints of actual quantum hardware.

### Error Mitigation 

We compare X and XY mixers with and without error mitigation by setting the resilience_level to 0 (no error mitigation) or 1 ( error mitigation). Particular error mitigation we used was T-REx.

## References
[(1) Benchmarking the performance of portfolio optimization with QAOA https://arxiv.org/abs/2207.10555 <br>
[(2) A Quantum Approximate Optimization Algorithm https://arxiv.org/abs/1411.4028 <br>
[(3) Model-free readout-error mitigation for quantum expectation values https://arxiv.org/abs/2012.09738
