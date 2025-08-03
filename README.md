[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-blue?style=for-the-badge)](https://discord.gg/sr9QBj3k36)

# Qubit: HybridQubit & Cubit Classes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.7%2B-blue) ![Qiskit](https://img.shields.io/badge/Qiskit-1.4.2-orange) ![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.15.1-green) ![GitHub Stars](https://img.shields.io/github/stars/R-D-BioTech-Alaska/Qubit?style=social)

Welcome to **Qubit**, a Python toolkit for quantum logic, supporting both Qiskit-based simulation and direct CPU statevector emulation. This repository provides two robust, feature-complete classes:

* **HybridQubit** (`Qubit.py`): Statevector-level quantum logic powered by Qiskit/Aer, with logical subspaces, engineered noise, amplitude storage, and full custom unitaries.
* **Cubit** (`Cubit.py`): A pure-NumPy, high-speed logical qubit class, allowing advanced quantum operations on any classical CPU—no Qiskit or hardware dependencies required. Designed for AI/ML integration, QELM, and rapid development.

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

Quantum computing relies on the flexible, powerful properties of qubits—quantum bits that exist in superposition and can interact in high-dimensional spaces. Real quantum systems often require hybrid or classical-quantum workflows. **Qubit** provides two complete approaches:

* **HybridQubit** (Qiskit/Aer backend): Operates with Qiskit’s simulator for full quantum fidelity, supporting custom logical/hidden subspaces, noise engineering, and programmable quantum logic.
* **Cubit** (CPU backend): Brings the same logical and mathematical operations—superposition, measurement, noise, subspace storage—directly to NumPy arrays, for 100% QPU-independent quantum logic. Cubit is ideal for AI/ML, simulation, and anywhere Qiskit isn’t practical.

---

## Features

* **Arbitrary-Dimensional Qubits**: Work with standard qubits (2D) or extend to multi-level quantum systems (qudits).
* **Logical Subspace Selection**: Assign any basis states to act as logical |0⟩ and |1⟩.
* **Amplitude Storage/Retrieval**: Move and recover information between logical and hidden subspaces.
* **Comprehensive Noise/Decoherence**: Apply phase noise, depolarizing noise, amplitude damping, and more.
* **Full Logical Operations**: Logical X, Z, Hadamard, Rx, Ry, Rz, custom unitaries, and advanced measurement routines.
* **Flexible Backends**:

  * Qiskit/Aer-powered simulation (HybridQubit)
  * Pure-CPU, NumPy-only simulation (Cubit)

---

## Installation & Requirements

**HybridQubit:** Python 3.7+, Qiskit, NumPy
**Cubit:** Python 3.7+, NumPy

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

The classes offer a unified, simple interface. Here are quick-start examples:

### HybridQubit (Qiskit/Aer Mode)

Uses Qiskit’s simulator for real quantum state fidelity, with all Qiskit features available.

```python
from Qubit import HybridQubit

qubit = HybridQubit(dimension=4, logical_zero_idx=0, logical_one_idx=1, name="DemoHybrid")
print("Initial state:", qubit.get_statevector())
qubit.apply_logical_x()
print("After X:", qubit.get_statevector())
qubit.store_amplitude_in_subspace(target_subspace_idx=2)
qubit.retrieve_amplitude_from_subspace(source_subspace_idx=2)
qubit.apply_noise_to_subspace(gamma=0.05)
result = qubit.measure_logical()
print("Logical measurement:", result)
```

---

### Cubit (Pure CPU Mode)

All quantum logic, direct on CPU. No external dependencies beyond NumPy.

```python
from Cubit import Cubit

cubit = Cubit(dimension=4, logical_zero_idx=0, logical_one_idx=1, name="DemoCubit")
print("Initial state:", cubit.state)
cubit.apply_logical_x()
print("After X:", cubit.state)
cubit.store_amplitude_in_subspace(2)
cubit.retrieve_amplitude_from_subspace(2)
cubit.apply_phase_noise(gamma=0.03)
result = cubit.measure_logical()
print("Logical measurement:", result)
```

---

## Class Interface & Methods

Both classes expose nearly identical method sets for logical and subspace quantum operations:

* Statevector access: `.get_statevector()` or `.state`
* Logical operations: `.apply_logical_x()`, `.apply_logical_z()`, `.apply_h()`, `.apply_rx()`, etc.
* Subspace manipulation: `.store_amplitude_in_subspace()`, `.retrieve_amplitude_from_subspace()`
* Noise models: `.apply_noise_to_subspace()` / `.apply_phase_noise()`, `.apply_amplitude_damping()`, `.apply_depolarizing()`
* Measurement: `.measure_logical()`, `.measure_full()`, `.measure_multiple_shots()`

See each class file for full details.

---

## Advanced Techniques

* **Qudit Experiments**: Use higher dimensions for advanced simulation, error models, or logical redundancy.
* **Classical-Quantum AI**: Use Cubit with QELM or similar AI pipelines—enabling quantum-inspired models without hardware.
* **Noise Calibration**: Simulate real hardware environments by calibrating Cubit noise to real QPU specs.

---

## Troubleshooting & Common Pitfalls

* **Dependencies**: HybridQubit requires Qiskit; Cubit requires only NumPy.
* **Indices/Dimensions**: All basis state, logical, and subspace indices are checked for safety—invalid choices raise errors.
* **Normalization**: State normalization is handled automatically after all operations.
* **Direct State Access**: HybridQubit uses Qiskit’s internal statevectors; Cubit uses direct NumPy arrays for speed and transparency.

---

## Contributing

Fork, branch, and open a pull request. Issues and feature requests are welcome—multi-qubit support and new modules encouraged.

---

## License

MIT License. See [LICENSE](LICENSE).

---
