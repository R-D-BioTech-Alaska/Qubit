#This is full implemented into Qelm as a working method. There are many uses for this kind of method, but it can definitely be improved. PR for help if needed.

from __future__ import annotations
import numpy as np
import scipy.linalg as la
import secrets
from typing import List, Tuple, Optional, Sequence, Iterable, Union, Dict

try:
    import h5py
    _H5PY_AVAILABLE = True
except Exception:
    _H5PY_AVAILABLE = False

try:
    from qiskit.quantum_info import DensityMatrix as _QDM, Statevector as _QSV
    from qiskit.circuit import QuantumCircuit as _QC
    _QISKIT_AVAILABLE = True
except Exception:
    _QISKIT_AVAILABLE = False  

__all__ = ['QuantumEmulator', 'RealQubit', 'QubitError']


class QubitError(Exception):
    pass

def _np_complex_dtype(dtype='complex128'): # this sometimes gives errors, pr if it does
    return np.complex64 if str(dtype) in ('float32','complex64') else np.complex128

def _rng_from_serets():
    return _rng_from_secrets()

def _rng_from_secrets():
    seed = int.from_bytes(secrets.token_bytes(16), 'big')
    ss = np.random.SeedSequence(seed)
    return np.random.default_rng(ss)

def _single_qubit_unitary(name: str, theta: Optional[float]=None) -> np.ndarray:
    name = name.upper()
    if name == 'I':  return np.eye(2, dtype=np.complex128)
    if name == 'X':  return np.array([[0,1],[1,0]], dtype=np.complex128)
    if name == 'Y':  return np.array([[0,-1j],[1j,0]], dtype=np.complex128)
    if name == 'Z':  return np.array([[1,0],[0,-1]], dtype=np.complex128)
    if name == 'H':  return (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=np.complex128)
    if name == 'S':  return np.array([[1,0],[0,1j]], dtype=np.complex128)
    if name == 'SDG':return np.array([[1,0],[0,-1j]], dtype=np.complex128)
    if name == 'T':  return np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=np.complex128)
    if name == 'TDG':return np.array([[1,0],[0,np.exp(-1j*np.pi/4)]], dtype=np.complex128)
    if name == 'RX':
        if theta is None: raise QubitError("RX requires theta.")
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
    if name == 'RY':
        if theta is None: raise QubitError("RY requires theta.")
        return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2),  np.cos(theta/2)]], dtype=np.complex128)
    if name == 'RZ':
        if theta is None: raise QubitError("RZ requires theta.")
        return np.array([[np.exp(-1j*theta/2), 0],
                         [0, np.exp(1j*theta/2)]], dtype=np.complex128)
    raise QubitError(f"Unknown single-qubit gate '{name}'.")

def _two_qubit_unitary(name: str) -> np.ndarray:
    name = name.upper()
    if name == 'CNOT':
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
    if name == 'CZ':
        return np.diag([1,1,1,-1]).astype(np.complex128)
    if name == 'SWAP':
        return np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex128)
    raise QubitError(f"Unknown two-qubit gate '{name}'.")

def _embed_1q(U: np.ndarray, q: int, n: int) -> np.ndarray:
    if U.shape != (2,2): raise QubitError("U must be 2×2.")
    mats = [np.eye(2, dtype=np.complex128)]*n
    mats[q] = U
    res = mats[0]
    for m in mats[1:]:
        res = np.kron(res, m)
    return res

def _embed_2q_adjacent(U4: np.ndarray, q: int, n: int) -> np.ndarray:
    if q < 0 or q+1 >= n:
        raise QubitError("Adjacent pair (q,q+1) out of range.")
    mats = []
    for i in range(n):
        if i == q:
            mats.append(U4)
        elif i == q+1:
            continue
        else:
            mats.append(np.eye(2, dtype=np.complex128))
    res = mats[0]
    for m in mats[1:]:
        res = np.kron(res, m)
    return res

_PAULI = {
    'I': _single_qubit_unitary('I'),
    'X': _single_qubit_unitary('X'),
    'Y': _single_qubit_unitary('Y'),
    'Z': _single_qubit_unitary('Z'),
}
_SIGMA_PLUS  = np.array([[0,1],[0,0]], dtype=np.complex128)   # |1><0|
_SIGMA_MINUS = np.array([[0,0],[1,0]], dtype=np.complex128)   # |0><1|


def _parse_pauli_string(pauli: str, n: int) -> List[str]:
    s = pauli.strip().upper()
    if not s:
        raise QubitError("Empty Pauli string.")
    if any(ch.isdigit() for ch in s):
        paulis = ['I']*n
        tokens = s.replace(',', ' ').split()
        for tok in tokens:
            p = tok[0]
            try:
                q = int(tok[1:])
            except ValueError:
                raise QubitError(f"Bad token '{tok}'. Use like 'X0' or 'Z3'.")
            if not (0 <= q < n): raise QubitError("Qubit index out of range in Pauli string.")
            if p not in 'IXYZ': raise QubitError("Unknown Pauli in string.")
            paulis[q] = p
        return paulis
    if len(s) != n or any(ch not in 'IXYZ' for ch in s):
        raise QubitError(f"Pauli string must be length {n} with I/X/Y/Z or tokenized form.")
    return list(s)

def _liouvillian(H: np.ndarray, collapses: Sequence[np.ndarray]) -> np.ndarray:
    d = H.shape[0]
    I = np.eye(d, dtype=np.complex128)
    L = -1j*(np.kron(I, H) - np.kron(H.T, I))
    for c in collapses:
        c = np.asarray(c, dtype=np.complex128)
        cd = c.conj().T
        L += np.kron(c, c.conj()) - 0.5*(np.kron(I, (cd@c).T) + np.kron((cd@c), I))
    return L

def _vec(rho: np.ndarray) -> np.ndarray:
    return rho.reshape(-1, 1, order='F')

def _unvec(v: np.ndarray, d: int) -> np.ndarray:
    return v.reshape(d, d, order='F')

class QuantumEmulator:
    def __init__(self, num_qubits: int = 1, dtype: str='complex128', seed: Optional[int]=None):
        if num_qubits < 1: raise QubitError("num_qubits must be >=1.")
        self.n   = int(num_qubits)
        self.dim = 2**self.n
        self.dtype = _np_complex_dtype(dtype)
        psi0 = np.zeros(self.dim, dtype=self.dtype); psi0[0]=1.0+0j
        self._rho = np.outer(psi0, psi0.conj()).astype(self.dtype, copy=False)
        self.T1  = [None]*self.n
        self.T2  = [None]*self.n
        self.p_eq = [0.0]*self.n  
        self.readout_matrix = [np.array([[1.0,0.0],[0.0,1.0]], dtype=float) for _ in range(self.n)]
        self.detuning = [0.0]*self.n
        self.Jzz: Dict[Tuple[int,int], float] = {} 
        self.gate_error_rates = {}
        self.rng = _rng_from_secrets() if seed is None else np.random.default_rng(seed)
        self._drift_enabled = False
        self._drift_sigmas = {'T1':0.0, 'T2':0.0, 'detuning':0.0}

    def _apply_U(self, U: np.ndarray):
        U = np.asarray(U, dtype=self.dtype)
        self._rho = U @ self._rho @ U.conj().T
        self._renorm()

    def _renorm(self):
        tr = np.trace(self._rho)
        if not np.isfinite(tr) or abs(tr) < 1e-14:
            raise QubitError("Density matrix trace invalid.")
        self._rho /= tr

    def reset(self, thermal: bool=False):
        if thermal and any(self.p_eq):
            probs = np.array([1.0], dtype=float)
            for q in range(self.n):
                p1 = float(self.p_eq[q]); p0 = 1.0 - p1
                probs = np.kron(probs, np.array([p0, p1], dtype=float))
            self._rho = np.diag(probs).astype(self.dtype)
            self._renorm()
            return self
        psi0 = np.zeros(self.dim, dtype=self.dtype); psi0[0]=1.0+0j
        self._rho = np.outer(psi0, psi0.conj()).astype(self.dtype, copy=False)
        return self

    def calibrate(self, T1: Optional[Union[float, Sequence[Optional[float]]]]=None,
                  T2: Optional[Union[float, Sequence[Optional[float]]]]=None,
                  gate_error_rates: Optional[dict]=None):
        if T1 is not None:
            if isinstance(T1, (list, tuple, np.ndarray)):
                if len(T1) != self.n: raise QubitError("T1 length mismatch.")
                self.T1 = [None if t is None else float(t) for t in T1]
            else:
                self.T1 = [float(T1)]*self.n
        if T2 is not None:
            if isinstance(T2, (list, tuple, np.ndarray)):
                if len(T2) != self.n: raise QubitError("T2 length mismatch.")
                self.T2 = [None if t is None else float(t) for t in T2]
            else:
                self.T2 = [float(T2)]*self.n
        if gate_error_rates is not None:
            self.gate_error_rates.update(gate_error_rates)
        return self

    def calibrate_temperature(self, p_excited: Union[float, Sequence[float]]):
        if isinstance(p_excited, (list, tuple, np.ndarray)):
            if len(p_excited) != self.n: raise QubitError("p_excited length mismatch.")
            self.p_eq = [float(p) for p in p_excited]
        else:
            self.p_eq = [float(p_excited)]*self.n
        return self

    def calibrate_readout(self, q: int, M: np.ndarray):
        M = np.asarray(M, dtype=float)
        if M.shape != (2,2): raise QubitError("Readout matrix must be 2×2.")
        if not np.allclose(M.sum(axis=0), 1.0, atol=1e-6):
            raise QubitError("Columns must each sum to 1.")
        self.readout_matrix[q] = M
        return self

    def calibrate_couplings(self, Jzz: Dict[Tuple[int,int], float]):
        clean = {}
        for (i,j), J in Jzz.items():
            i, j = int(i), int(j)
            if i==j: continue
            if not (0<=i<self.n and 0<=j<self.n): raise QubitError("Coupling index out of range.")
            a,b = (i,j) if i<j else (j,i)
            clean[(a,b)] = float(J)
        self.Jzz = clean
        return self

    def calibrate_detuning(self, delta: Dict[int, float]):
        for q, d in delta.items():
            if not (0<=q<self.n): raise QubitError("Detuning index out of range.")
            self.detuning[q] = float(d)
        return self

    def enable_drift(self, T1_sigma=0.0, T2_sigma=0.0, detuning_sigma=0.0):
        self._drift_enabled = True
        self._drift_sigmas = {'T1':float(T1_sigma), 'T2':float(T2_sigma), 'detuning':float(detuning_sigma)}
        return self

    def disable_drift(self):
        self._drift_enabled = False
        return self

    def apply_gate(self, name: str, targets: Sequence[int], theta: Optional[float]=None):
        nameU = name.upper()
        targets = list(targets)
        if len(targets)==1:
            q = targets[0]
            if not (0 <= q < self.n): raise QubitError("Target out of range.")
            U = _single_qubit_unitary(nameU, theta)
            U_full = _embed_1q(U, q, self.n)
            self._apply_U(U_full)
        elif len(targets)==2:
            q1, q2 = targets
            if not (0 <= q1 < self.n and 0 <= q2 < self.n): raise QubitError("Target out of range.")
            if q1 == q2: raise QubitError("Distinct targets required.")
            U4 = _two_qubit_unitary(nameU)
            self._apply_two_qubit(U4, q1, q2)
        else:
            raise QubitError("Only 1- or 2-qubit gates supported.")
        if nameU in self.gate_error_rates and self.gate_error_rates[nameU] > 0:
            p = float(self.gate_error_rates[nameU])
            if len(targets)==1:
                self.apply_depolarizing(p, targets[0])
            else:
                self.apply_depolarizing(p, targets[0])
                self.apply_depolarizing(p, targets[1])
        return self

    def _apply_two_qubit(self, U4: np.ndarray, q1: int, q2: int):
        q1, q2 = int(q1), int(q2)
        if q1 == q2: raise QubitError("q1 and q2 must be distinct.")
        if q1 > q2: q1, q2 = q2, q1
        SW = _two_qubit_unitary('SWAP')
        for k in range(q2-1, q1, -1):
            self._apply_U(_embed_2q_adjacent(SW, k-1, self.n))
        self._apply_U(_embed_2q_adjacent(U4, q1, self.n))
        for k in range(q1+1, q2):
            self._apply_U(_embed_2q_adjacent(SW, k-1, self.n))

    def apply_unitary(self, U: np.ndarray):
        U = np.asarray(U, dtype=self.dtype)
        if U.shape != (self.dim, self.dim): raise QubitError("Unitary shape mismatch.")
        if not np.allclose(U.conj().T @ U, np.eye(self.dim), atol=1e-10):
            raise QubitError("Matrix is not unitary within tolerance.")
        self._apply_U(U)
        return self

    def evolve_hamiltonian(self, H: np.ndarray, t: float):
        H = np.asarray(H, dtype=self.dtype)
        if H.shape != (self.dim, self.dim): raise QubitError("Hamiltonian shape mismatch.")
        U = la.expm(-1j*H*t).astype(self.dtype, copy=False)
        self._apply_U(U)
        return self

    def _collapse_ops_from_T(self) -> List[np.ndarray]:
        cs = []
        for q in range(self.n):
            if self.T1[q] is not None and self.T1[q] > 0:
                tot = 1.0/float(self.T1[q])
                p = float(self.p_eq[q])
                gamma_up   = p*tot
                gamma_down = (1.0-p)*tot
                if gamma_down > 0:
                    c_down = np.sqrt(gamma_down) * _SIGMA_MINUS
                    cs.append(_embed_1q(c_down, q, self.n))
                if gamma_up > 0:
                    c_up = np.sqrt(gamma_up) * _SIGMA_PLUS
                    cs.append(_embed_1q(c_up, q, self.n))
            if self.T2[q] is not None and self.T2[q] > 0:
                invT2 = 1.0/float(self.T2[q])
                invT1 = 1.0/float(self.T1[q]) if self.T1[q] not in (None,0) else 0.0
                gamma_phi = max(0.0, invT2 - 0.5*invT1)
                if gamma_phi > 0:
                    c_phi = np.sqrt(gamma_phi) * _PAULI['Z']
                    cs.append(_embed_1q(c_phi, q, self.n))
        return cs

    def _static_H(self) -> np.ndarray:
        H = np.zeros((self.dim, self.dim), dtype=self.dtype)
        for q, dz in enumerate(self.detuning):
            if dz != 0.0:
                H += 0.5*dz * _embed_1q(_PAULI['Z'], q, self.n)
        for (i,j), J in self.Jzz.items():
            if abs(i-j)==1:
                H += (J/4.0) * _embed_2q_adjacent(np.kron(_PAULI['Z'], _PAULI['Z']), min(i,j), self.n)
            else:
                H += self._embed_nonadjacent_two_qubit(np.kron(_PAULI['Z'], _PAULI['Z']), i, j)
        return H

    def _embed_nonadjacent_two_qubit(self, U4: np.ndarray, q1: int, q2: int) -> np.ndarray:
        q1, q2 = (q1,q2) if q1<q2 else (q2,q1)
        SW = _two_qubit_unitary('SWAP')
        S = np.eye(self.dim, dtype=self.dtype)
        for k in range(q2-1, q1, -1):
            S = _embed_2q_adjacent(SW, k-1, self.n) @ S
        U_full_adj = _embed_2q_adjacent(U4, q1, self.n)
        for k in range(q1+1, q2):
            S = _embed_2q_adjacent(SW, k-1, self.n) @ S
        return S @ U_full_adj @ S.conj().T

    def evolve_lindblad(self, H: np.ndarray, dt: float, steps: int=1, extra_collapses: Optional[Sequence[np.ndarray]]=None):
        H = np.asarray(H, dtype=self.dtype)
        if H.shape != (self.dim, self.dim): raise QubitError("Hamiltonian shape mismatch.")
        collapses = self._collapse_ops_from_T()
        if extra_collapses:
            collapses = list(collapses) + [np.asarray(C, dtype=self.dtype) for C in extra_collapses]
        L = _liouvillian(H, collapses)
        E = la.expm(L*dt)
        v = _vec(self._rho)
        for _ in range(int(steps)):
            v = E @ v
            if self._drift_enabled:
                self._apply_drift_step()
        self._rho = _unvec(v, self.dim).astype(self.dtype, copy=False)
        self._renorm()
        return self

    def idle(self, dt: float, steps: int=1):
        H = self._static_H()
        return self.evolve_lindblad(H, dt, steps=steps)

    def apply_pulse_schedule(self, schedule: List[dict], include_decoherence: bool=True):
        for seg in schedule:
            dt = float(seg.get('dt', 0.0))
            if dt <= 0: continue
            drive = seg.get('drive', {})
            H = self._static_H()
            for q, d in drive.items():
                q = int(q)
                ox = float(d.get('ox', 0.0)); oy = float(d.get('oy', 0.0)); dz = float(d.get('dz', 0.0))
                H += 0.5*ox*_embed_1q(_PAULI['X'], q, self.n)
                H += 0.5*oy*_embed_1q(_PAULI['Y'], q, self.n)
                H += 0.5*dz*_embed_1q(_PAULI['Z'], q, self.n)
            if include_decoherence:
                self.evolve_lindblad(H, dt, steps=1)
            else:
                self.evolve_hamiltonian(H, dt)
        return self

    def apply_amplitude_damping(self, dt: float, qubit: int):
        T1 = self.T1[qubit]
        if T1 is None or T1<=0: raise QubitError(f"T1 not calibrated for qubit {qubit}.")
        gamma = 1 - np.exp(-dt/float(T1))
        K0 = np.array([[1,0],[0,np.sqrt(1-gamma)]], dtype=self.dtype)
        K1 = np.array([[0,np.sqrt(gamma)],[0,0]], dtype=self.dtype)
        self._apply_kraus([K0, K1], qubit); return self

    def apply_generalized_amplitude_damping(self, dt: float, qubit: int, p_eq: Optional[float]=None):
        if p_eq is None: p_eq = float(self.p_eq[qubit])
        T1 = self.T1[qubit]
        if T1 is None or T1<=0: raise QubitError(f"T1 not calibrated for qubit {qubit}.")
        gamma = 1 - np.exp(-dt/float(T1))
        p = float(p_eq); p = min(max(p,0.0),1.0)
        K0 = np.sqrt(p) * np.array([[np.sqrt(1-gamma),0],[0,1]], dtype=self.dtype)
        K1 = np.sqrt(p) * np.array([[0,0],[np.sqrt(gamma),0]], dtype=self.dtype)
        K2 = np.sqrt(1-p) * np.array([[1,0],[0,np.sqrt(1-gamma)]], dtype=self.dtype)
        K3 = np.sqrt(1-p) * np.array([[0,np.sqrt(gamma)],[0,0]], dtype=self.dtype)
        self._apply_kraus([K0,K1,K2,K3], qubit); return self

    def apply_phase_damping(self, dt: float, qubit: int):
        T2 = self.T2[qubit]
        if T2 is None or T2<=0: raise QubitError(f"T2 not calibrated for qubit {qubit}.")
        p = 1 - np.exp(-dt/float(T2))
        K0 = np.sqrt(1-p)*np.eye(2, dtype=self.dtype)
        K1 = np.sqrt(p)*_PAULI['Z']
        self._apply_kraus([K0, K1], qubit); return self

    def apply_depolarizing(self, p: float, qubit: int):
        if not (0 <= p <= 1): raise QubitError("Depolarizing p must be in [0,1].")
        I,X,Y,Z = _PAULI['I'], _PAULI['X'], _PAULI['Y'], _PAULI['Z']
        self._mix_pauli([(1-p), (p/3), (p/3), (p/3)], [I, X, Y, Z], qubit); return self

    def _apply_kraus(self, Ks: Iterable[np.ndarray], qubit: int):
        rho_new = np.zeros_like(self._rho)
        for K in Ks:
            U_full = _embed_1q(K, qubit, self.n)
            rho_new += U_full @ self._rho @ U_full.conj().T
        self._rho = rho_new; self._renorm()

    def _mix_pauli(self, weights: Sequence[float], ops: Sequence[np.ndarray], qubit: int):
        if len(weights)!=len(ops): raise QubitError("weights/ops length mismatch")
        if not np.isclose(sum(weights), 1.0, atol=1e-9):
            raise QubitError("Pauli mix weights must sum to 1.")
        rho_new = np.zeros_like(self._rho)
        for w, P in zip(weights, ops):
            U_full = _embed_1q(P, qubit, self.n)
            rho_new += w*(U_full @ self._rho @ U_full.conj().T)
        self._rho = rho_new; self._renorm()

    def _basis_rotation_matrix(self, bases: Sequence[str]) -> np.ndarray:
        if len(bases) != self.n: raise QubitError("bases length must equal num qubits.")
        mats = []
        for q, b in enumerate(bases):
            b = b.upper()
            if b == 'Z': mats.append(_single_qubit_unitary('I'))
            elif b == 'X': mats.append(_single_qubit_unitary('H'))
            elif b == 'Y': mats.append(_single_qubit_unitary('H') @ _single_qubit_unitary('SDG'))
            else: raise QubitError("Unknown basis (use 'X','Y','Z').")
        U = mats[0]
        for m in mats[1:]: U = np.kron(U, m)
        return U

    def _apply_readout_matrix_to_bit(self, q: int, true_bit: int) -> int:
        M = self.readout_matrix[q]  
        p_meas0 = M[0, true_bit]
        return int(self.rng.choice([0,1], p=[p_meas0, 1-p_meas0]))

    def sample_bitstrings(self, shots: int, bases: Union[str, Sequence[str]]='Z', collapse: bool=False, apply_readout_error: bool=True) -> List[str]:
        if isinstance(bases, str): bases = [bases]*self.n
        bases = [b.upper() for b in bases]
        U = self._basis_rotation_matrix(bases)
        rho_rot = U @ self._rho @ U.conj().T
        diag = np.real_if_close(np.diag(rho_rot)).astype(float)
        p = diag / float(np.sum(diag))
        outcomes = []
        for _ in range(shots):
            idx = int(self.rng.choice(self.dim, p=p))
            bits_true = [ (idx >> q) & 1 for q in range(self.n) ]
            if apply_readout_error:
                bits_meas = [ self._apply_readout_matrix_to_bit(q, bits_true[q]) for q in range(self.n) ]
            else:
                bits_meas = bits_true
            outcomes.append(''.join(str(b) for b in bits_meas))
            if collapse:
                proj = np.zeros((self.dim, self.dim), dtype=self.dtype); proj[idx, idx] = 1.0
                self._rho = U.conj().T @ proj @ U
                self._renorm()
        return outcomes

    def measure(self, qubit: int, shots: int=1, basis: str='Z', collapse: bool=True, apply_readout_error: bool=True) -> List[int]:
        if not (0 <= qubit < self.n): raise QubitError("qubit out of range.")
        basis = basis.upper()
        if basis == 'Z': Uloc = _single_qubit_unitary('I')
        elif basis == 'X': Uloc = _single_qubit_unitary('H')
        elif basis == 'Y': Uloc = _single_qubit_unitary('H') @ _single_qubit_unitary('SDG')
        else: raise QubitError("basis must be 'X','Y','Z'.")
        U = _embed_1q(Uloc, qubit, self.n)
        rho_rot = U @ self._rho @ U.conj().T
        diag = np.real_if_close(np.diag(rho_rot))
        idx = np.arange(self.dim, dtype=int)
        mask1 = ((idx >> qubit) & 1) == 1
        p1 = float(np.sum(diag[mask1]).real); p0 = float(np.sum(diag[~mask1]).real)
        s = p0+p1; p0, p1 = ((0.5,0.5) if s<=0 else (p0/s, p1/s))
        outs: List[int] = []
        for _ in range(shots):
            true_bit = int(self.rng.choice([0,1], p=[p0,p1]))
            meas_bit = self._apply_readout_matrix_to_bit(qubit, true_bit) if apply_readout_error else true_bit
            outs.append(meas_bit)
            if collapse:
                proj_loc = np.array([[1,0],[0,0]], dtype=self.dtype) if true_bit==0 else np.array([[0,0],[0,1]], dtype=self.dtype)
                P = _embed_1q(proj_loc, qubit, self.n)
                self._rho = U.conj().T @ (P @ rho_rot @ P) @ U
                self._renorm()
        return outs

    def measure_all(self, shots: int=1, basis: str='Z', collapse: bool=True, apply_readout_error: bool=True) -> List[str]:
        return self.sample_bitstrings(shots, bases=basis, collapse=collapse, apply_readout_error=apply_readout_error)

    def mitigate_readout_counts(self, counts: Dict[str, int]) -> Dict[str, float]:
        p_meas = np.zeros(self.dim, dtype=float)
        total = sum(counts.values()) if counts else 0
        if total == 0: return {k:0.0 for k in counts}
        for bitstr, c in counts.items():
            idx = sum((int(bitstr[q])<<q) for q in range(self.n))
            p_meas[idx] += c/total
        M = np.array([[1.0]], dtype=float)
        for q in range(self.n):
            M = np.kron(M, self.readout_matrix[q])
        MtM = M.T @ M
        lam = 1e-8
        Minv = np.linalg.solve(MtM + lam*np.eye(MtM.shape[0]), M.T)
        p_true = Minv @ p_meas
        p_true = np.clip(p_true, 0, None)
        if p_true.sum()>0: p_true /= p_true.sum()
        out = {}
        for idx, prob in enumerate(p_true):
            bits = ''.join(str((idx>>q)&1) for q in range(self.n))
            out[bits] = float(prob)
        return out

    def reduced_density(self, keep: Sequence[int]) -> np.ndarray:
        keep = sorted(set(int(q) for q in keep))
        if any(q<0 or q>=self.n for q in keep): raise QubitError("keep indices out of range.")
        traced = [q for q in range(self.n) if q not in keep]
        k = len(keep)
        if k == 0: raise QubitError("Must keep at least one qubit.")
        rho = self._rho.reshape([2]*self.n*2)
        current_n = self.n
        for q in sorted(traced, reverse=True):
            rho = np.trace(rho, axis1=q, axis2=current_n+q)
            current_n -= 1
        rho = rho.reshape(2**k, 2**k).astype(self.dtype, copy=False)
        rho /= float(np.trace(rho))
        return rho

    def bloch_vector(self, qubit: int) -> Tuple[float,float,float]:
        rho1 = self.reduced_density([qubit])
        rx = float(np.real(np.trace(rho1 @ _PAULI['X'])))
        ry = float(np.real(np.trace(rho1 @ _PAULI['Y'])))
        rz = float(np.real(np.trace(rho1 @ _PAULI['Z'])))
        return (rx, ry, rz)

    def expval_pauli_string(self, pauli: str) -> float:
        ops = _parse_pauli_string(pauli, self.n)
        O = np.array([[1]], dtype=self.dtype)
        for q in range(self.n):
            O = np.kron(O, _PAULI[ops[q]])
        val = np.trace(self._rho @ O)
        return float(np.real_if_close(val))

    def tomography_qubit(self, qubit: int, shots_per_basis: int=0) -> Tuple[np.ndarray, Tuple[float,float,float]]:
        if shots_per_basis and shots_per_basis>0:
            est = {}
            for B in ('X','Y','Z'):
                outcomes = self.measure(qubit, shots=shots_per_basis, basis=B, collapse=False, apply_readout_error=False)
                mean = np.mean([1 if o==0 else -1 for o in outcomes])
                est[B] = float(mean)
            rx, ry, rz = est['X'], est['Y'], est['Z']
        else:
            rx, ry, rz = self.bloch_vector(qubit)
        rho = 0.5*( _PAULI['I'] + rx*_PAULI['X'] + ry*_PAULI['Y'] + rz*_PAULI['Z'] )
        w, V = np.linalg.eigh(rho); w=np.clip(w,0,None); w=w/np.sum(w); rho=(V@np.diag(w)@V.conj().T).astype(self.dtype, copy=False)
        return rho, (rx,ry,rz)

    def tomography_two_qubit(self, q1: int, q2: int, shots_per_setting: int=0) -> np.ndarray:
        if shots_per_setting and shots_per_setting>0:
            paulis = ['I','X','Y','Z']; coeffs = {}
            for a in paulis:
                for b in paulis:
                    def basis_for(p): return {'I':'Z','X':'X','Y':'Y','Z':'Z'}[p]
                    bases = ['Z']*self.n; bases[q1]=basis_for(a); bases[q2]=basis_for(b)
                    samples = self.sample_bitstrings(shots=shots_per_setting, bases=bases, collapse=False, apply_readout_error=False)
                    vals = []
                    for s in samples:
                        v1 = 1 if int(s[q1])==0 else -1
                        v2 = 1 if int(s[q2])==0 else -1
                        vals.append(v1*v2)
                    coeffs[(a,b)] = float(np.mean(vals))
            P = {'I':_PAULI['I'], 'X':_PAULI['X'], 'Y':_PAULI['Y'], 'Z':_PAULI['Z']}
            rho2 = np.zeros((4,4), dtype=self.dtype)
            for a in paulis:
                for b in paulis:
                    rho2 += (coeffs[(a,b)]/4.0) * np.kron(P[a], P[b])
            w,V = np.linalg.eigh(rho2); w=np.clip(w,0,None); w=w/np.sum(w); rho2=(V@np.diag(w)@V.conj().T)
            return rho2.astype(self.dtype, copy=False)
        else:
            return self.reduced_density([q1,q2])

    def purity(self) -> float:
        return float(np.real_if_close(np.trace(self._rho @ self._rho)))

    def entropy_vn(self, base: float=2.0) -> float:
        w = np.clip(np.linalg.eigvalsh(self._rho), 0, 1); w = w[w>1e-15]
        s = -np.sum(w * (np.log(w)/np.log(base)))
        return float(np.real_if_close(s))

    def fidelity(self, other: Union[np.ndarray, 'QuantumEmulator']) -> float:
        if isinstance(other, QuantumEmulator):
            sigma = other._rho
        else:
            arr = np.asarray(other)
            if arr.ndim == 1:
                ket = arr.reshape(-1,1).astype(self.dtype); sigma = ket @ ket.conj().T
            else:
                sigma = arr.astype(self.dtype)
        A = la.sqrtm(self._rho); B = A @ sigma @ A; C = la.sqrtm(B); F = (np.real(np.trace(C)))**2
        return float(np.clip(F, 0.0, 1.0))

    def concurrence_two_qubit(self, q1: int, q2: int) -> float:
        rho2 = self.reduced_density([q1,q2])
        Y = _PAULI['Y']; YY = np.kron(Y, Y)
        R = rho2 @ YY @ rho2.conj() @ YY
        w = np.sqrt(np.maximum(la.eigvals(R).real, 0)); w.sort()
        C = max(0.0, w[-1]-w[-2]-w[-3]-w[-4])
        return float(C)

    def peek_state(self) -> np.ndarray:
        return self._rho.copy()

    def populations(self) -> np.ndarray:
        return np.real_if_close(np.diag(self._rho)).astype(float)

    def save(self, filename: str):
        if _H5PY_AVAILABLE and filename.lower().endswith(('.h5','.hdf5')):
            with h5py.File(filename, 'w') as f:
                f.create_dataset('rho', data=self._rho); f.attrs['num_qubits'] = self.n
        else:
            np.savez_compressed(filename, rho=self._rho, num_qubits=self.n)

    def load(filename: str) -> 'QuantumEmulator':
        if _H5PY_AVAILABLE and filename.lower().endswith(('.h5','.hdf5')):
            with h5py.File(filename, 'r') as f:
                rho = f['rho'][:]; n = int(f.attrs['num_qubits'])
        else:
            data = np.load(filename, allow_pickle=True); rho = data['rho']; n = int(data['num_qubits'])
        qe = QuantumEmulator(num_qubits=n)
        if rho.shape != (qe.dim, qe.dim): raise QubitError("File dimensions inconsistent.")
        qe._rho = rho.astype(qe.dtype, copy=False); qe._renorm(); return qe

    def to_qiskit_densitymatrix(self):
        if not _QISKIT_AVAILABLE: raise QubitError("Qiskit not available.")
        return _QDM(self._rho)

    def from_qiskit(obj: Union[' _QDM ', ' _QSV ']) -> 'QuantumEmulator':
        if not _QISKIT_AVAILABLE: raise QubitError("Qiskit not available.")
        if isinstance(obj, _QDM):
            rho = np.asarray(obj.data, dtype=np.complex128)
        elif isinstance(obj, _QSV):
            ket = np.asarray(obj.data, dtype=np.complex128).reshape(-1,1); rho = ket @ ket.conj().T
        else:
            raise QubitError("Unsupported Qiskit object (use DensityMatrix or Statevector).")
        n = int(np.log2(rho.shape[0])); qe = QuantumEmulator(n); qe._rho = rho.astype(qe.dtype, copy=False); qe._renorm(); return qe

    def apply_qiskit_circuit(self, circuit: ' _QC '):
        if not _QISKIT_AVAILABLE: raise QubitError("Qiskit not available.")
        for inst, qargs, _ in circuit.data:
            name = inst.name.lower()
            if name in ('x','y','z','h','s','sdg','t','tdg'):
                q = circuit.find_bit(qargs[0]).index; self.apply_gate(name.upper(), [q])
            elif name in ('rx','ry','rz'):
                theta = float(inst.params[0]); q = circuit.find_bit(qargs[0]).index; self.apply_gate(name.upper(), [q], theta=theta)
            elif name == 'cx':
                c = circuit.find_bit(qargs[0]).index; t = circuit.find_bit(qargs[1]).index; self.apply_gate('CNOT', [c,t])
            elif name == 'cz':
                c = circuit.find_bit(qargs[0]).index; t = circuit.find_bit(qargs[1]).index; self.apply_gate('CZ', [c,t])
            elif name == 'swap':
                a = circuit.find_bit(qargs[0]).index; b = circuit.find_bit(qargs[1]).index; self.apply_gate('SWAP', [a,b])
            elif name in ('barrier','measure','reset','id','delay'):
                continue
            else:
                raise QubitError(f"Unsupported Qiskit instruction: {inst.name}")

    def _apply_drift_step(self):
        sT1 = self._drift_sigmas['T1']; sT2 = self._drift_sigmas['T2']; sD = self._drift_sigmas['detuning']
        for q in range(self.n):
            if sT1>0 and self.T1[q] not in (None,0):
                self.T1[q] = max(1e-9, float(self.T1[q]) * np.exp(self.rng.normal(0, sT1)))
            if sT2>0 and self.T2[q] not in (None,0):
                self.T2[q] = max(1e-9, float(self.T2[q]) * np.exp(self.rng.normal(0, sT2)))
            if sD>0:
                self.detuning[q] += float(self.rng.normal(0, sD))


class RealQubit:
    def __init__(self, dtype: str='complex128', seed: Optional[int]=None):
        self._emu = QuantumEmulator(1, dtype=dtype, seed=seed)

    def x(self): self._emu.apply_gate('X', [0]); return self
    def y(self): self._emu.apply_gate('Y', [0]); return self
    def z(self): self._emu.apply_gate('Z', [0]); return self
    def h(self): self._emu.apply_gate('H', [0]); return self
    def s(self): self._emu.apply_gate('S', [0]); return self
    def sdg(self): self._emu.apply_gate('SDG', [0]); return self
    def t(self): self._emu.apply_gate('T', [0]); return self
    def tdg(self): self._emu.apply_gate('TDG', [0]); return self
    def rx(self, theta: float): self._emu.apply_gate('RX', [0], theta); return self
    def ry(self, theta: float): self._emu.apply_gate('RY', [0], theta); return self
    def rz(self, theta: float): self._emu.apply_gate('RZ', [0], theta); return self

    def pulse(self, ox=0.0, oy=0.0, dz=0.0, dt=0.0, decoherence=True):
        schedule = [{'dt':dt, 'drive':{0:{'ox':ox,'oy':oy,'dz':dz}}}]
        self._emu.apply_pulse_schedule(schedule, include_decoherence=decoherence); return self

    def idle(self, dt: float): self._emu.idle(dt); return self

    def calibrate(self, T1: Optional[float]=None, T2: Optional[float]=None, gate_error_rates: Optional[dict]=None):
        self._emu.calibrate([T1] if T1 is not None else None,
                            [T2] if T2 is not None else None,
                            gate_error_rates); return self
    def temperature(self, p_excited: float): self._emu.calibrate_temperature(p_excited); return self
    def readout(self, matrix_2x2: np.ndarray): self._emu.calibrate_readout(0, matrix_2x2); return self

    def measure(self, shots: int=1, basis: str='Z', collapse: bool=True, apply_readout_error: bool=True) -> List[int]:
        return self._emu.measure(0, shots=shots, basis=basis, collapse=collapse, apply_readout_error=apply_readout_error)
    def sample(self, shots: int=1, basis: str='Z', collapse: bool=False, apply_readout_error: bool=True) -> List[int]:
        bits = self._emu.sample_bitstrings(shots, bases=[basis], collapse=collapse, apply_readout_error=apply_readout_error)
        return [int(b[0]) for b in bits]
    def populations(self) -> Tuple[float,float]:
        p = self._emu.populations(); return float(p[0]), float(p[1])
    def bloch(self) -> Tuple[float,float,float]: return self._emu.bloch_vector(0)
    def tomography(self, shots_per_basis: int=0): return self._emu.tomography_qubit(0, shots_per_basis=shots_per_basis)
    def purity(self) -> float: return self._emu.purity()
    def entropy(self, base: float=2.0) -> float: return self._emu.entropy_vn(base=base)

    def save(self, filename: str): self._emu.save(filename)
    def load(filename: str) -> 'RealQubit':
        emu = QuantumEmulator.load(filename)
        if emu.n != 1: raise QubitError("Saved state is not a single qubit.")
        rq = RealQubit(); rq._emu = emu; return rq
