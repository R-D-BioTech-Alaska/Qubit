[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-blue?style=for-the-badge)](https://discord.gg/sr9QBj3k36)

# Qubit: HybridQubit & Cubit Classes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.7%2B-blue)  ![Qiskit](https://img.shields.io/badge/Qiskit-1.4.2-orange)  ![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.15.1-green)  ![GitHub Stars](https://img.shields.io/github/stars/R-D-BioTech-Alaska/Qubit?style=social)

Welcome to **Qubit**, a Python toolkit for quantum logic, supporting both Qiskit-based simulation and direct CPU statevector emulation. This repository provides two complete, fully documented classes:

* **HybridQubit** (`Qubit.py`): Qiskit/Aer-powered, statevector-level logic with logical subspaces, custom noise, hidden storage, and custom unitaries.
* **Cubit** (`Cubit.py`): Fast, pure-NumPy logical qubit class, allowing all quantum logic on classical CPUs with no Qiskit or QPU dependency—built for AI/ML, QELM, and more.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation & Requirements](#installation--requirements)
4. [Usage: Getting Started](#usage-getting-started)

   * [HybridQubit (Qiskit/Aer)](#hybridqubit-qiskitaer-mode)
   * [Cubit (Pure CPU Mode)](#cubit-pure-cpu-mode)
5. [Class Interface & Methods](#class-interface--methods)
6. [Advanced Techniques](#advanced-techniques)
7. [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)
8. [Contributing](#contributing)
9. [License](#license)

---

## Overview

In modern quantum computing, a qubit is typically two-dimensional (|0⟩, |1⟩). However, real systems can leak into extra dimensions, and many advanced AI/ML or simulation pipelines need CPU-only access, or classical-quantum hybrid architectures.

This toolkit provides both approaches:

* **HybridQubit** (Qiskit backend, in `Qubit.py`): Operates with Qiskit’s AerSimulator for full statevector fidelity, custom logical/hidden subspaces, engineered quantum noise, and all the advanced quantum logic you expect.
* **Cubit** (CPU backend, in `Cubit.py`): Replicates the same logic—superposition, phase, noise, subspace storage—purely in NumPy, enabling "quantum logic" as a class in any classical Python environment, with no Qiskit or hardware dependency.

---

## Features

* **Arbitrary-Dimensional Qubits**: Any number of basis states (2+), not just binary.
* **Custom Logical Subspaces**: Choose which basis states act as your “logical” |0> and |1>.
* **Amplitude Storage/Retrieval**: Hide, recover, or manipulate information in hidden quantum states.
* **Noise & Decoherence Models**: Add phase noise, depolarizing noise, or build your own.
* **Logical Operations**: Logical X, Z, arbitrary unitaries, measurement, and more.
* **Backends**:

  * Qiskit/Aer-powered (`HybridQubit`)
  * Pure NumPy, CPU-only (`Cubit`)

---

## Installation & Requirements

* **HybridQubit:** Python 3.7+, Qiskit, NumPy
* **Cubit:** Python 3.7+, NumPy only

```bash
pip install qiskit numpy
# For Cubit only:
pip install numpy
```

Clone the repository:

```bash
git clone https://github.com/R-D-BioTech-Alaska/Qubit.git
cd Qubit
```

---

## Usage: Getting Started

Both classes offer a very similar interface. Below you’ll find a quick-start guide for each.

### HybridQubit (Qiskit/Aer Mode)

This class provides a **real quantum simulation** using the Qiskit AerSimulator.

**Typical use case:**
You want to test quantum logic and advanced subspace tricks using a real quantum backend, with all the features of Qiskit (visualization, noise, transpilation, etc).

```python
from Qubit import HybridQubit

# Create a 4D hybrid qubit (supports hidden subspaces)
qubit = HybridQubit(dimension=4, logical_zero_idx=0, logical_one_idx=1, name="DemoHybrid")

# Check the full quantum statevector
print("Initial state:", qubit.get_statevector())

# Apply a logical X (swap |0⟩ and |1⟩)
qubit.apply_logical_x()
print("After X:", qubit.get_statevector())

# Store amplitude in a hidden state (e.g., index 2)
qubit.store_amplitude_in_subspace(target_subspace_idx=2)

# Retrieve it later back to the logical subspace
qubit.retrieve_amplitude_from_subspace(source_subspace_idx=2)

# Apply quantum noise
qubit.apply_noise_to_subspace(gamma=0.05)

# Logical measurement (returns 0 or 1)
result = qubit.measure_logical()
print("Logical measurement:", result)
```

See [Qubit.py](Qubit.py) for the full class reference.

---

### Cubit (Pure CPU Mode)

This class allows you to use **all the same quantum logic, purely on a CPU**, with no Qiskit required. Ideal for quantum-inspired AI, simulation, or when you want *true QPU independence*.

**Typical use case:**
You want to build/experiment with quantum state logic, but deploy it *everywhere*—on laptops, clusters, in QELM or hybrid AI pipelines, and not just where Qiskit is available.

```python
from Cubit import Cubit

# Create a 4D Cubit (identical logic as HybridQubit)
cubit = Cubit(dimension=4, logical_zero_idx=0, logical_one_idx=1, name="DemoCubit")

# Direct statevector access (NumPy)
print("Initial state:", cubit.state)

# Apply logical operations
cubit.apply_logical_x()
print("After X:", cubit.state)

# Hide/retrieve quantum information
cubit.store_amplitude_in_subspace(2)
cubit.retrieve_amplitude_from_subspace(2)

# Add phase noise to hidden states
cubit.apply_phase_noise(gamma=0.03)

# Logical measurement (simulated quantum measurement)
result = cubit.measure_logical()
print("Logical measurement:", result)
```

See [Cubit.py](Cubit.py) for a full class and API reference.

---

## Class Interface & Methods

Both `HybridQubit` and `Cubit` expose nearly identical methods and logic:

* `get_statevector()` or `.state`
* `apply_logical_x()`, `apply_logical_z()`
* `store_amplitude_in_subspace()`, `retrieve_amplitude_from_subspace()`
* `apply_noise_to_subspace()` / `apply_phase_noise()`
* `measure_logical()`
* And more!

Full documentation for each method is in the class files.

---

## Advanced Techniques

* **High-Dimensional Quantum Logic**: Use `dimension > 2` for qudit, error-mitigating, or multi-level experiments.
* **Hybrid Classical-Quantum AI**: Plug Cubit directly into QELM, AI, or hybrid pipelines for quantum-inspired processing.
* **Custom Noise Models**: Extend noise/decoherence to simulate realistic environments or train for robustness.

---

## Troubleshooting & Common Pitfalls

* **Dependencies**: HybridQubit needs Qiskit; Cubit only needs NumPy.
* **Indices/Dimension**: Invalid logical or subspace indices raise errors for safety.
* **Normalization**: Both classes auto-normalize, but direct amplitude edits require caution.
* **State Access**: HybridQubit uses Qiskit’s statevector, Cubit exposes raw NumPy arrays for speed.

---

## Contributing

Fork, branch, and submit a pull request. Open issues for bugs or feature requests—new modules (QPU, GPU, etc.) encouraged!

---

## License

MIT License. See [LICENSE](LICENSE).

---
