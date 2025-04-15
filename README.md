[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-blue?style=for-the-badge)](https://discord.gg/sr9QBj3k36)

# Qubit: Hybrid-Qubit Class
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.7%2B-blue)  ![Qiskit](https://img.shields.io/badge/Qiskit-1.4.2-orange)  ![Qiskit Aer](https://img.shields.io/badge/Qiskit_Aer-0.15.1-green)  ![GitHub Stars](https://img.shields.io/github/stars/R-D-BioTech-Alaska/Qubit?style=social)

Welcome to **Qubit**, a Python library and demonstration for creating and manipulating higher-dimensional qubit states in Qiskit. This repository showcases the `HybridQubit` class, which embeds a 2-dimensional *logical* subspace (\|0_L>, \|1_L>) inside a higher-dimensional Hilbert space, allowing you to explore advanced quantum concepts such as “hidden” states, custom unitaries, and engineered noise.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation & Requirements](#installation--requirements)  
4. [Usage](#usage)  
5. [Class Interface & Methods](#class-interface--methods)  
6. [Advanced Techniques](#advanced-techniques)  
7. [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## Overview

In standard quantum computing, a qubit has exactly two basis states (\|0> and \|1>). However, many real quantum systems can *leak* into higher-dimensional states or incorporate auxiliary levels for error mitigation or advanced protocols. 

The **`Hybrid-Qubit`** class solves this by:
- Explicitly embedding a logical qubit (\|0_L>, \|1_L>) into a larger dimensional space.  
- Providing methods to store and retrieve amplitude in and out of the *hidden* states.  
- Offering a range of “gate-like” operations (logical X/Z, custom unitaries, noise models) fully at the statevector level.

---

## Features

- **Flexible Dimension**: Instantiate a HybridQubit with any dimension \(\geq 2\).  
- **Logical-Index Mapping**: Choose which basis states represent your logical \|0_L> and \|1_L>.  
- **Advanced Subspace Operations**: Store amplitude in or retrieve it from designated hidden states.  
- **Built-in Noise**: Apply random phase noise selectively to hidden subspace indices.  
- **Measurement**: Measure only in the logical basis, ignoring hidden states.  
- **Full Statevector Access**: Directly get/set the entire statevector, using Qiskit's `AerSimulator`.

---

## Installation & Requirements

1. **Python 3.7+**  
2. **Qiskit** (particularly `qiskit-terra` and `qiskit-aer`)  
3. **NumPy**  

Install these dependencies if you haven’t already:

```bash
pip install qiskit numpy
```

### Cloning the Repository

```bash
git clone https://github.com/R-D-BioTech-Alaska/Qubit.git
cd Qubit
```

Inside the repository, the core code lives in **`Qubit.py`**.

---

## Usage

### 1. Import and Create a HybridQubit

```python
from Qubit import HybridQubit

# Create a 4-dimensional HybridQubit with logical indices 0, 1
qubit = HybridQubit(dimension=4, logical_zero_idx=0, logical_one_idx=1, name="MyHybridQubit")
```

### 2. Basic Operations

```python
# Check the initial state (should be \|0_L>):
initial_sv = qubit.get_statevector()
print("Initial statevector:", initial_sv)

# Apply a logical X (swap \|0_L> and \|1_L>)
qubit.apply_logical_x()
print("After logical X:", qubit.get_statevector())

# Measure the logical subspace
result = qubit.measure_logical()
print(f"Logical measurement = {result}")  # 0 or 1
```

### 3. Storing and Retrieving Amplitude

```python
# Suppose you want to store amplitude in basis index 2 (a hidden state)
qubit.store_amplitude_in_subspace(target_subspace_idx=2)
print("After storing in subspace:", qubit.get_statevector())

# Then retrieve that amplitude back into the logical \|1_L> state
qubit.retrieve_amplitude_from_subspace(source_subspace_idx=2)
print("After retrieving from subspace:", qubit.get_statevector())
```

### 4. Applying Noise and Custom Unitaries

```python
# Apply random phase noise to the hidden subspace
qubit.apply_noise_to_subspace(gamma=0.05)

# Apply a custom phase shift to indices [2, 3]
phases = [0.3, -1.2]
indices = [2, 3]
qubit.apply_arbitrary_subspace_unitary(subspace_indices=indices, phases=phases)
```

---

## Class Interface & Methods

Below is a quick reference for the `HybridQubit` class. For details, see [Qubit.py](Qubit.py).

### Constructor

```python
HybridQubit(
    dimension: int,
    logical_zero_idx: int = 0,
    logical_one_idx: int = 1,
    name: str = "HybridQubit"
)
```

- **dimension** (int): Size of the total Hilbert space (>=2).  
- **logical_zero_idx** (int): Which basis index represents \|0_L>.  
- **logical_one_idx** (int): Which basis index represents \|1_L>.  
- **name** (str): A human-friendly name for logs/debugging.

### Methods (Selected)

1. **`get_statevector() -> np.ndarray`**  
   Returns a NumPy array containing the statevector.

2. **`reset_to_logical_zero()`** / **`reset_to_logical_one()`**  
   Re-initializes the qubit to \|0_L> or \|1_L>.

3. **`apply_logical_x()`** / **`apply_logical_z()`**  
   Performs a Pauli-X or Pauli-Z on the *logical* subspace only.

4. **`measure_logical() -> int`**  
   Measures and returns 0 or 1 according to the logical amplitude.

5. **`store_amplitude_in_subspace(target_subspace_idx: int)`** /  
   **`retrieve_amplitude_from_subspace(source_subspace_idx: int)`**  
   Moves amplitude between the hidden and logical subspace.

6. **`apply_noise_to_subspace(gamma: float = 0.05)`**  
   Adds a small random phase to each hidden state.

7. **`apply_arbitrary_subspace_unitary(subspace_indices: list, phases: list)`**  
   Applies user-specified phase shifts to selected basis states.

8. **Population Queries**  
   - `get_population(basis_idx: int) -> float`  
   - `get_logical_pops() -> (float, float)`  
   - `get_hidden_population_sum() -> float`

9. **Renaming**  
   - `rename(new_name: str)` changes the qubit’s name for circuit labeling.

---

## Advanced Techniques

1. **High-Dimensional Embeddings**  
   Use `dimension` > 2 to explore qudit or leakage-based simulations. For example, dimension = 8 means three qubits are allocated internally, but only \|logical_zero_idx> and \|logical_one_idx> are “logical.”

2. **Combining HybridQubits**  
   If you want multiple `HybridQubit` objects, keep in mind each is backed by its own `QuantumCircuit`. You may need to manually combine or tensor circuits if your use case requires multi-hybrid-qubit interactions.

3. **Experimenting with Noise Models**  
   - Extend `apply_noise_to_subspace()` or write your own method for amplitude damping, depolarizing, etc., using Qiskit’s built-in gates or direct statevector manipulation.

4. **Monitoring Leakage**  
   - `get_hidden_population_sum()` reveals how much probability mass is outside \|0_L> and \|1_L>.  
   - Perfect gates yield zero leakage (unless you deliberately store amplitude in hidden states).

---

## Troubleshooting & Common Pitfalls

1. **Index Errors**  
   If you accidentally set `logical_zero_idx` or `logical_one_idx` outside `[0, dimension - 1]`, a `SubspaceDefinitionError` is raised.  

2. **Dimension Mismatch**  
   Ensure your `dimension` parameter makes sense for your embedded qubit. Attempting to store amplitude in an index >= `dimension` also raises `SubspaceDefinitionError`.

3. **Renormalization in Measurements**  
   If all amplitude leaks into hidden states, `measure_logical()` defaults to a random 0 or 1. Check `get_hidden_population_sum()` to track amplitude outside \|0_L>, \|1_L>.

4. **Circuit Clearing**  
   Each method re-initializes the entire circuit based on the updated statevector. This design is intentional for direct state manipulation but means you can’t chain gates in the typical Qiskit sense. Instead, you might combine them at the statevector level or write additional logic that composes them in a single shot.

---

## Contributing

We welcome contributions and suggestions! To add new functionality or fix issues:

1. [Fork the repository](https://github.com/R-D-BioTech-Alaska/Qubit/fork)  
2. Create a feature branch (`git checkout -b feature-name`)  
3. Commit changes (`git commit -m "Add new feature"`)  
4. Push to your fork (`git push origin feature-name`)  
5. Create a Pull Request  

Feel free to open an [Issue](https://github.com/R-D-BioTech-Alaska/Qubit/issues) for bug reports or feature requests.

---

## License

This repository is licensed under the [MIT License](LICENSE). You’re free to use, modify, and distribute this code for personal or commercial purposes but ensure credit is given.

---
