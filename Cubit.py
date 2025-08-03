import numpy as np
import scipy.linalg
import warnings

class CubitError(Exception): pass

class Cubit:
    def __init__(self, dimension=2, logical_zero_idx=0, logical_one_idx=1, name="Cubit"):
        if dimension < 2 or logical_zero_idx < 0 or logical_zero_idx >= dimension or \
           logical_one_idx < 0 or logical_one_idx >= dimension or logical_one_idx == logical_zero_idx:
            raise CubitError("Invalid dimension or logical indices.")
        self.dimension = int(dimension)
        self.logical_zero_idx = int(logical_zero_idx)
        self.logical_one_idx = int(logical_one_idx)
        self.name = str(name)
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.T1 = None
        self.T2 = None
        self.gate_error_rates = None
        self.reset_to_logical_zero()

    def initialize(self, state):
        state = np.asarray(state, dtype=np.complex128)
        if state.shape != (self.dimension,) or not np.isclose(np.linalg.norm(state), 1.0, atol=1e-12):
            raise CubitError("Statevector shape or norm error.")
        self.state = state.copy()

    def initialize_superposition(self, weights, normalize=True):
        weights = np.asarray(weights, dtype=np.complex128)
        if weights.shape != (self.dimension,): raise CubitError("Invalid weights shape.")
        if normalize:
            norm = np.linalg.norm(weights)
            if norm < 1e-14: raise CubitError("Norm near zero.")
            weights = weights / norm
        self.state = weights.copy()

    def reset_to_logical_zero(self):
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.state[self.logical_zero_idx] = 1.0

    def reset_to_logical_one(self):
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.state[self.logical_one_idx] = 1.0

    def reset_to_basis(self, idx):
        if idx < 0 or idx >= self.dimension: raise CubitError("Index out of range.")
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

    def apply_rx(self, theta):
        U = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                      [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        self._apply_to_logical_subspace(U)

    def apply_ry(self, theta):
        U = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                      [np.sin(theta/2),  np.cos(theta/2)]], dtype=np.complex128)
        self._apply_to_logical_subspace(U)

    def apply_rz(self, theta):
        U = np.array([[np.exp(-1j*theta/2), 0],
                      [0, np.exp(1j*theta/2)]], dtype=np.complex128)
        self._apply_to_logical_subspace(U)

    def apply_h(self):
        U = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        self._apply_to_logical_subspace(U)

    def apply_s(self):
        U = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        self._apply_to_logical_subspace(U)

    def apply_t(self):
        U = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128)
        self._apply_to_logical_subspace(U)

    def _apply_to_logical_subspace(self, U):
        sv = self.state.copy()
        indices = [self.logical_zero_idx, self.logical_one_idx]
        sub_sv = sv[indices]
        sv[indices] = U @ sub_sv
        self.state = sv

    def apply_subspace_unitary(self, indices, phases):
        indices = np.asarray(indices)
        phases = np.asarray(phases)
        if indices.shape != phases.shape or np.any(indices < 0) or np.any(indices >= self.dimension):
            raise CubitError("Subspace index error.")
        self.state[indices] *= np.exp(1j * phases)

    def apply_unitary(self, U, check='fast'):
        U = np.asarray(U, dtype=np.complex128)
        if U.shape != (self.dimension, self.dimension): raise CubitError("Unitary shape error.")
        if check == 'full':
            if not np.allclose(U.conj().T @ U, np.eye(self.dimension), atol=1e-12):
                raise CubitError("Unitary property error.")
        elif check == 'fast':
            eigvals = np.linalg.eigvals(U)
            if not np.allclose(np.abs(eigvals), 1, atol=1e-8):
                raise CubitError("Unitary eigenvalue property error.")
        self.state = U @ self.state

    def apply_amplitude_damping(self, dt):
        if self.T1 is None: raise CubitError("T1 not calibrated.")
        gamma = 1 / self.T1
        decay = np.exp(-gamma * dt)
        amp1 = self.state[self.logical_one_idx]
        self.state[self.logical_one_idx] = amp1 * decay
        self.state[self.logical_zero_idx] += amp1 * (1 - decay)
        self.normalize()

    def apply_phase_damping(self, dt):
        if self.T2 is None: raise CubitError("T2 not calibrated.")
        gamma = 1 / self.T2
        decay = np.exp(-gamma * dt)
        for i in [self.logical_zero_idx, self.logical_one_idx]:
            self.state[i] *= decay
        self.normalize()

    def apply_depolarizing(self, p):
        if not (0 <= p <= 1): raise CubitError("Invalid depolarizing probability.")
        d = self.dimension
        sv = self.state.copy()
        uniform_noise = np.ones(d, dtype=np.complex128) / np.sqrt(d)
        self.state = np.sqrt(1-p)*sv + np.sqrt(p)*uniform_noise
        self.normalize()

    def apply_coherent_error(self, phase):
        self.apply_subspace_unitary([self.logical_one_idx], [phase])

    def apply_phase_noise(self, gamma=0.05):
        if gamma < 0: raise CubitError("Negative gamma.")
        sv = self.state.copy()
        mask = np.ones(self.dimension, dtype=bool)
        mask[[self.logical_zero_idx, self.logical_one_idx]] = False
        sv[mask] *= np.exp(1j * np.random.uniform(-gamma, gamma, size=np.count_nonzero(mask)))
        self.state = sv

    def apply_hamiltonian(self, H, t):
        H = np.asarray(H, dtype=np.complex128)
        if H.shape != (self.dimension, self.dimension): raise CubitError("Invalid Hamiltonian shape.")
        U = scipy.linalg.expm(-1j * H * t)
        self.apply_unitary(U, check='fast')

    def calibrate_to_hardware(self, T1, T2, gate_error_rates):
        self.T1 = T1
        self.T2 = T2
        self.gate_error_rates = gate_error_rates

    def apply_cnot(self, target_qubit):
        if not isinstance(target_qubit, Cubit): raise CubitError("Invalid target.")
        d1, d2 = self.dimension, target_qubit.dimension
        if d1 != 2 or d2 != 2: raise CubitError("CNOT only for 2-level qubits.")
        joint_state = np.kron(self.state, target_qubit.state)
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
        new_joint = CNOT @ joint_state
        self.state = np.sum(new_joint.reshape(2,2), axis=1)
        target_qubit.state = np.sum(new_joint.reshape(2,2), axis=0)
        self.normalize()
        target_qubit.normalize()

    def measure_logical(self, rng=None, warn_hidden=True):
        amp0, amp1 = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        p0, p1 = np.abs(amp0)**2, np.abs(amp1)**2
        norm = p0 + p1
        hidden_sum = np.sum([np.abs(self.state[i])**2 for i in range(self.dimension) if i not in (self.logical_zero_idx, self.logical_one_idx)])
        if warn_hidden and hidden_sum > 1e-4:
            warnings.warn(f"Significant population in hidden states: {hidden_sum:.4f}")
        if norm < 1e-14: return int((rng or np.random).choice([0,1]))
        p0, p1 = p0/norm, p1/norm
        if rng:
            return int(rng.choices([0,1], weights=[p0, p1])[0])
        return int(np.random.choice([0,1], p=[p0, p1]))

    def measure_full(self, rng=None):
        probs = np.abs(self.state)**2
        probs = probs / np.sum(probs)
        if rng:
            idx = rng.choices(np.arange(self.dimension), weights=probs)[0]
        else:
            idx = int(np.random.choice(np.arange(self.dimension), p=probs))
        return idx

    def measure_multiple_shots(self, n_shots, logical=True, rng=None):
        measure_func = self.measure_logical if logical else self.measure_full
        return [measure_func(rng) for _ in range(n_shots)]

    def get_population(self, idx):
        if idx < 0 or idx >= self.dimension: raise CubitError("Index out of range.")
        return float(np.abs(self.state[idx])**2)

    def get_logical_populations(self):
        return (self.get_population(self.logical_zero_idx), self.get_population(self.logical_one_idx))

    def get_hidden_population_sum(self):
        hidden = [i for i in range(self.dimension) if i not in (self.logical_zero_idx, self.logical_one_idx)]
        return float(np.sum(np.abs(self.state[hidden])**2))

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm < 1e-14: raise CubitError("Norm near zero.")
        self.state /= norm

    def rename(self, new_name): self.name = str(new_name)

    def store_amplitude_in_subspace(self, target_idx):
        if target_idx < 0 or target_idx >= self.dimension or target_idx in (self.logical_zero_idx, self.logical_one_idx):
            raise CubitError("target_idx out of range.")
        sv = self.state.copy()
        sv[target_idx] += sv[self.logical_one_idx]
        sv[self.logical_one_idx] = 0.0
        self.state = sv

    def retrieve_amplitude_from_subspace(self, source_idx):
        if source_idx < 0 or source_idx >= self.dimension or source_idx in (self.logical_zero_idx, self.logical_one_idx):
            raise CubitError("source_idx out of range.")
        sv = self.state.copy()
        sv[self.logical_one_idx] += sv[source_idx]
        sv[source_idx] = 0.0
        self.state = sv

    def to_qiskit_state(self):
        from qiskit.quantum_info import Statevector
        return Statevector(self.state)
