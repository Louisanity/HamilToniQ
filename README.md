# HamilToniQ
This is the repository for IBM Quantum Hackathon 2023 at the World of Quantum with the topic of encoding (Mixer Hamiltonian)

## Project Summary
1. Problem Formulation
2. Mixer Hamiltonian Formulation
3. Post-processing of mixer Hamiltonian
4. Error mitigation
6. Scale plot of $\beta$ and $\gamma$
7. QASM, noisy simulator and real harware benchmark

### Problem formulation
![alt text](https://github.com/Louisanity/HamilToniQ//blob/main/pictures/problem_set.png?raw=true)

### Error Mitigation 

We compare X and XY mixers with and without error mitigation by setting the resilience_level to 0 (no error mitigation) or 1 ( error mitigation). Particular error mitigation we used was T-REx.

## References
[(1) Benchmarking the performance of portfolio optimization with QAOA https://arxiv.org/abs/2207.10555 <br>
[(2) A Quantum Approximate Optimization Algorithm https://arxiv.org/abs/1411.4028 <br>
[(3) Model-free readout-error mitigation for quantum expectation values https://arxiv.org/abs/2012.09738
