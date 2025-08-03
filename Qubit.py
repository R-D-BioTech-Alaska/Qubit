import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library.standard_gates import XGate, ZGate, CXGate, RZGate
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from typing import Optional, Union


class SubspaceDefinitionError(Exception):

    pass

class HybridQubit:

    def __init__(self,
                 dimension: int,
                 logical_zero_idx: int = 0,
                 logical_one_idx: int = 1,
                 name: str = "HybridQubit"):
        if dimension < 2:
            raise ValueError("Dimension must be >= 2 for a qubit.")
        if logical_zero_idx < 0 or logical_zero_idx >= dimension:
            raise SubspaceDefinitionError("logical_zero_idx out of valid range.")
        if logical_one_idx < 0 or logical_one_idx >= dimension:
            raise SubspaceDefinitionError("logical_one_idx out of valid range.")
        if logical_one_idx == logical_zero_idx:
            raise SubspaceDefinitionError("logical_zero_idx and logical_one_idx must differ.")

        self.dimension = dimension
        self.logical_zero_idx = logical_zero_idx
        self.logical_one_idx = logical_one_idx
        self.name = name

        # Number of qubits needed to represent 'dimension' basis states
        self.num_qubits = int(np.ceil(np.log2(self.dimension)))

        # Prepare the Qiskit circuit
        self.circuit = QuantumCircuit(self.num_qubits, name=self.name)
        # We'll use AerSimulator statevector by default (for now)
        self.backend = AerSimulator(method="statevector")

        # By default, set the system to the logical |0_L>
        # which is the basis state: self.logical_zero_idx
        self._initialize_to_basis_state(self.logical_zero_idx)

    def _initialize_to_basis_state(self, basis_idx: int) -> None:

        if basis_idx < 0 or basis_idx >= self.dimension:
            raise SubspaceDefinitionError(f"Requested basis_idx {basis_idx} out of [0, {self.dimension-1}].")

        # Clear the circuit instructions
        self.circuit.data.clear()

        # Build a pure basis state vector of length 2^num_qubits
        full_size = 2**self.num_qubits
        state_vector = np.zeros(full_size, dtype=complex)
        state_vector[basis_idx] = 1.0

        # Use the Qiskit initialize instruction
        self.circuit.initialize(state_vector, range(self.num_qubits))
        self.circuit.save_statevector(label="hybridqubit_init")
    def get_statevector(self) -> np.ndarray:
        transpiled = transpile(self.circuit, self.backend, optimization_level=1)
        result = self.backend.run(transpiled).result()
        return np.array(result.get_statevector(transpiled))
    def reset_to_logical_zero(self) -> None:
        self._initialize_to_basis_state(self.logical_zero_idx)
    def reset_to_logical_one(self) -> None:
        self._initialize_to_basis_state(self.logical_one_idx)

    def apply_logical_x(self) -> None:
        # First, get the current statevector
        current_sv = self.get_statevector()
        dim = self.dimension
        extended_sv = current_sv.copy()  # We'll mutate a copy

        # Swap amplitude for the two subspace indices
        amp0 = extended_sv[self.logical_zero_idx] 
        amp1 = extended_sv[self.logical_one_idx] 
        extended_sv[self.logical_zero_idx] = amp1
        extended_sv[self.logical_one_idx]  = amp0

        # Other indices remain as is
        self.circuit.data.clear()
        self.circuit.initialize(extended_sv, range(self.num_qubits))
        self.circuit.save_statevector(label="after_logical_x")

    def apply_logical_z(self) -> None:

        current_sv = self.get_statevector()
        extended_sv = current_sv.copy()

        # Multiply the amplitude at logical_one_idx by -1
        extended_sv[self.logical_one_idx] *= -1.0

        self.circuit.data.clear()
        self.circuit.initialize(extended_sv, range(self.num_qubits))
        self.circuit.save_statevector(label="after_logical_z")

    def measure_logical(self) -> int:

        sv = self.get_statevector()
        amp0 = sv[self.logical_zero_idx]
        amp1 = sv[self.logical_one_idx]

        p0 = np.abs(amp0)**2
        p1 = np.abs(amp1)**2
        norm = p0 + p1

        if norm < 1e-14:
            return np.random.choice([0,1])
        else:
            # Renormalize
            p0 /= norm
            p1 /= norm
            return int(np.random.choice([0,1], p=[p0, p1]))

    def store_amplitude_in_subspace(self, target_subspace_idx: int) -> None:

        if target_subspace_idx < 0 or target_subspace_idx >= self.dimension:
            raise SubspaceDefinitionError("target_subspace_idx out of range.")
        if target_subspace_idx == self.logical_zero_idx or target_subspace_idx == self.logical_one_idx:
            raise SubspaceDefinitionError("target_subspace_idx must not be logical 0 or 1 index.")

        current_sv = self.get_statevector()
        extended_sv = current_sv.copy()

        amp_hidden = extended_sv[target_subspace_idx]
        amp_logical1 = extended_sv[self.logical_one_idx]

        # Move amplitude from logical_one_idx to target_subspace_idx
        extended_sv[self.logical_one_idx] = 0.0
        extended_sv[target_subspace_idx] = amp_hidden + amp_logical1

        self.circuit.data.clear()
        self.circuit.initialize(extended_sv, range(self.num_qubits))
        self.circuit.save_statevector(label="after_store_in_subspace")

    def retrieve_amplitude_from_subspace(self, source_subspace_idx: int) -> None:

        if source_subspace_idx < 0 or source_subspace_idx >= self.dimension:
            raise SubspaceDefinitionError("source_subspace_idx out of range.")
        if source_subspace_idx == self.logical_zero_idx or source_subspace_idx == self.logical_one_idx:
            raise SubspaceDefinitionError("source_subspace_idx must not be logical 0 or 1 index.")

        current_sv = self.get_statevector()
        extended_sv = current_sv.copy()

        amp_logical1 = extended_sv[self.logical_one_idx]
        amp_subspace = extended_sv[source_subspace_idx]

        # Move amplitude from source_subspace_idx to logical_one_idx
        extended_sv[source_subspace_idx] = 0.0
        extended_sv[self.logical_one_idx] = amp_logical1 + amp_subspace

        self.circuit.data.clear()
        self.circuit.initialize(extended_sv, range(self.num_qubits))
        self.circuit.save_statevector(label="after_retrieve_from_subspace")

    def apply_noise_to_subspace(self, gamma: float = 0.05) -> None:

        if gamma < 0:
            raise ValueError("Gamma must be non-negative.")

        current_sv = self.get_statevector()
        extended_sv = current_sv.copy()

        for idx in range(self.dimension):
            # skip logical subspace
            if idx == self.logical_zero_idx or idx == self.logical_one_idx:
                continue
            # random small phase
            phase_shift = np.random.uniform(-gamma, gamma)
            extended_sv[idx] *= np.exp(1j * phase_shift)

        self.circuit.data.clear()
        self.circuit.initialize(extended_sv, range(self.num_qubits))
        self.circuit.save_statevector(label="after_phase_noise")

    def apply_arbitrary_subspace_unitary(self, 
                                         subspace_indices: list, 
                                         phases: list) -> None:

        if len(subspace_indices) != len(phases):
            raise ValueError("subspace_indices and phases must have same length.")

        current_sv = self.get_statevector()
        extended_sv = current_sv.copy()

        for idx, phi in zip(subspace_indices, phases):
            if idx < 0 or idx >= self.dimension:
                raise SubspaceDefinitionError(f"Index {idx} out of subspace range.")
            # multiply amplitude by e^{i phi}
            extended_sv[idx] *= np.exp(1j * phi)

        self.circuit.data.clear()
        self.circuit.initialize(extended_sv, range(self.num_qubits))
        self.circuit.save_statevector(label="after_custom_subspace_unitary")

    def get_population(self, basis_idx: int) -> float:

        if basis_idx < 0 or basis_idx >= 2**self.num_qubits:
            raise ValueError(f"basis_idx {basis_idx} out of 0..{2**self.num_qubits - 1}.")
        if basis_idx >= self.dimension:
            # conceptually out of range subspace
            return 0.0
        sv = self.get_statevector()
        return float(np.abs(sv[basis_idx])**2)

    def get_logical_pops(self) -> (float, float):

        sv = self.get_statevector()
        p0 = np.abs(sv[self.logical_zero_idx])**2
        p1 = np.abs(sv[self.logical_one_idx])**2
        return (float(p0), float(p1))

    def get_hidden_population_sum(self) -> float:

        sv = self.get_statevector()
        total = 0.0
        for idx in range(self.dimension):
            if idx == self.logical_zero_idx or idx == self.logical_one_idx:
                continue
            total += np.abs(sv[idx])**2
        return float(total)

    def rename(self, new_name: str) -> None:

        self.name = new_name
        self.circuit.name = new_name
