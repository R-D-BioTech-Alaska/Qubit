import numpy as np

class CPUQubitError(Exception): pass

class CPUQubit:
    def __init__(self, dimension=2, logical_zero_idx=0, logical_one_idx=1, name="CPUQubit"):
        if dimension < 2 or logical_zero_idx < 0 or logical_zero_idx >= dimension or \
           logical_one_idx < 0 or logical_one_idx >= dimension or logical_one_idx == logical_zero_idx:
            raise CPUQubitError("Invalid dimension or logical indices.")
        self.dimension = int(dimension)
        self.logical_zero_idx = int(logical_zero_idx)
        self.logical_one_idx = int(logical_one_idx)
        self.name = str(name)
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.reset_to_logical_zero()
    def initialize(self, state):
        state = np.asarray(state, dtype=np.complex128)
        if state.shape != (self.dimension,) or not np.isclose(np.linalg.norm(state), 1.0, atol=1e-12):
            raise CPUQubitError("Statevector shape or norm error.")
        self.state = state.copy()
    def reset_to_logical_zero(self):
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.state[self.logical_zero_idx] = 1.0
    def reset_to_logical_one(self):
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.state[self.logical_one_idx] = 1.0
    def reset_to_basis(self, idx):
        if idx < 0 or idx >= self.dimension: raise CPUQubitError("Index out of range.")
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.state[idx] = 1.0
    def apply_logical_x(self):
        sv = self.state.copy()
        sv[self.logical_zero_idx], sv[self.logical_one_idx] = sv[self.logical_one_idx], sv[self.logical_zero_idx]
        self.state = sv
    def apply_logical_z(self):
        sv = self.state.copy()
        sv[self.logical_one_idx] *= -1.0
        self.state = sv
    def apply_logical_hadamard(self):
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        sv = self.state.copy()
        sv[self.logical_zero_idx] = (a + b)/np.sqrt(2)
        sv[self.logical_one_idx] = (a - b)/np.sqrt(2)
        self.state = sv
    def apply_subspace_unitary(self, indices, phases):
        if len(indices) != len(phases): raise CPUQubitError("Length mismatch.")
        sv = self.state.copy()
        for idx, phi in zip(indices, phases):
            if idx < 0 or idx >= self.dimension: raise CPUQubitError("Index out of range.")
            sv[idx] *= np.exp(1j*phi)
        self.state = sv
    def apply_unitary(self, U):
        U = np.asarray(U, dtype=np.complex128)
        if U.shape != (self.dimension, self.dimension) or not np.allclose(U.conj().T @ U, np.eye(self.dimension), atol=1e-12):
            raise CPUQubitError("Unitary shape or property error.")
        self.state = U @ self.state
    def store_amplitude_in_subspace(self, target_idx):
        if target_idx < 0 or target_idx >= self.dimension or target_idx in (self.logical_zero_idx, self.logical_one_idx):
            raise CPUQubitError("target_idx out of range.")
        sv = self.state.copy()
        sv[target_idx] += sv[self.logical_one_idx]
        sv[self.logical_one_idx] = 0.0
        self.state = sv
    def retrieve_amplitude_from_subspace(self, source_idx):
        if source_idx < 0 or source_idx >= self.dimension or source_idx in (self.logical_zero_idx, self.logical_one_idx):
            raise CPUQubitError("source_idx out of range.")
        sv = self.state.copy()
        sv[self.logical_one_idx] += sv[source_idx]
        sv[source_idx] = 0.0
        self.state = sv
    def apply_phase_noise(self, gamma=0.05):
        if gamma < 0: raise CPUQubitError("Negative gamma.")
        sv = self.state.copy()
        for idx in range(self.dimension):
            if idx in (self.logical_zero_idx, self.logical_one_idx): continue
            sv[idx] *= np.exp(1j * np.random.uniform(-gamma, gamma))
        self.state = sv
    def measure_logical(self):
        amp0, amp1 = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        p0, p1 = np.abs(amp0)**2, np.abs(amp1)**2
        norm = p0 + p1
        if norm < 1e-14: return np.random.choice([0,1])
        p0, p1 = p0/norm, p1/norm
        return int(np.random.choice([0,1], p=[p0, p1]))
    def measure_full(self):
        probs = np.abs(self.state)**2
        probs = probs / np.sum(probs)
        return int(np.random.choice(np.arange(self.dimension), p=probs))
    def get_population(self, idx):
        if idx < 0 or idx >= self.dimension: raise CPUQubitError("Index out of range.")
        return float(np.abs(self.state[idx])**2)
    def get_logical_populations(self):
        return (self.get_population(self.logical_zero_idx), self.get_population(self.logical_one_idx))
    def get_hidden_population_sum(self):
        hidden = [i for i in range(self.dimension) if i not in (self.logical_zero_idx, self.logical_one_idx)]
        return float(np.sum(np.abs(self.state[hidden])**2))
    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm < 1e-14: raise CPUQubitError("Norm near zero.")
        self.state /= norm
    def rename(self, new_name): self.name = str(new_name)
