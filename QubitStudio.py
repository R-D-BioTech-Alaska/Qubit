import os
import sys
import json
import math
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any, Union
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
try:
    from qiskit import QuantumCircuit, transpile
except Exception as e:
    QuantumCircuit = None
    transpile = None

_AER_AVAILABLE = True
try:
    from qiskit_aer import AerSimulator
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.noise.errors import pauli_error, thermal_relaxation_error, depolarizing_error
    except Exception:
        NoiseModel = None
        pauli_error = thermal_relaxation_error = depolarizing_error = None
except Exception:
    _AER_AVAILABLE = False
    AerSimulator = None
    NoiseModel = None
    pauli_error = thermal_relaxation_error = depolarizing_error = None

try:
    from qiskit.quantum_info import Statevector, DensityMatrix
except Exception:
    Statevector = None
    DensityMatrix = None

MATPLOTLIB_AVAILABLE = True
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
except Exception:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvasTkAgg = None
    plt = None

_QASM2_LOADS = None
_QASM2_DUMPS = None
try:
    from qiskit.qasm2 import loads as _qasm2_loads, dumps as _qasm2_dumps
    _QASM2_LOADS = _qasm2_loads
    _QASM2_DUMPS = _qasm2_dumps
except Exception:
    _QASM2_LOADS = None
    _QASM2_DUMPS = None

_IBM_AVAILABLE = True
try:
    from qiskit_ibm_provider import IBMProvider  
except Exception:
    IBMProvider = None
    _IBM_AVAILABLE = False

def safe_complex_list_to_pairs(vec: np.ndarray) -> List[List[float]]:
    out = []
    for a in vec:
        out.append([float(np.real(a)), float(np.imag(a))])
    return out

def pairs_to_complex_list(pairs: List[List[float]]) -> np.ndarray:
    return np.array([complex(r, i) for r, i in pairs], dtype=np.complex128)

def normalize_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm > 0:
        return state / norm
    return state

def clamp01(x: float) -> float:
    if x < 0: return 0.0
    if x > 1: return 1.0
    return x

class ToolTip:
    def __init__(self, widget, text: str, delay: int = 600):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.id = None
        self.tip = None
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)

    def _enter(self, _):
        self._schedule()

    def _leave(self, _):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self.id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def _show(self):
        if self.tip or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") if self.widget.winfo_ismapped() else (0,0,0,0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + cy + self.widget.winfo_rooty() + 20
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#111111", foreground="#f0f0f0",
                         relief=tk.SOLID, borderwidth=1,
                         font=("Segoe UI", 9))
        label.pack(ipadx=6, ipy=3, padx=1, pady=1)

    def _hide(self):
        if self.tip:
            self.tip.destroy()
            self.tip = None

class Cubit:
    def __init__(self, name: str, dimension: int = 2, logical_zero_idx: int = 0, logical_one_idx: int = 1):
        self.name = name
        self.dimension = int(dimension)
        self.logical_zero_idx = int(logical_zero_idx)
        self.logical_one_idx  = int(logical_one_idx)
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.state[self.logical_zero_idx] = 1.0
        self.T1: Optional[float] = None
        self.T2: Optional[float] = None
        self.gate_errors: Dict[str, float] = {}
        self._history: List[np.ndarray] = []
        self._future: List[np.ndarray] = []

    def _snapshot(self):
        self._history.append(self.state.copy())
        self._future.clear()

    def undo(self) -> bool:
        if not self._history:
            return False
        self._future.append(self.state.copy())
        self.state = self._history.pop()
        return True

    def redo(self) -> bool:
        if not self._future:
            return False
        self._history.append(self.state.copy())
        self.state = self._future.pop()
        return True

    def reset_to_logical_zero(self):
        self._snapshot()
        self.state[:] = 0
        self.state[self.logical_zero_idx] = 1.0

    def reset_to_logical_one(self):
        self._snapshot()
        self.state[:] = 0
        self.state[self.logical_one_idx] = 1.0

    def reset_to_basis(self, idx: int):
        self._snapshot()
        self.state[:] = 0
        self.state[idx] = 1.0

    def initialize(self, amplitudes: Union[List[complex], np.ndarray]):
        self._snapshot()
        self.state = np.array(amplitudes, dtype=np.complex128)
        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1.0):
            raise ValueError("State must be normalized")

    def get_statevector(self) -> np.ndarray:
        return self.state.copy()

    def set_T1(self, val: float): self.T1 = float(val)
    def set_T2(self, val: float): self.T2 = float(val)
    def set_gate_error(self, g: str, v: float):  self.gate_errors[g] = float(v)

    def apply_logical_x(self):
        self._snapshot()
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = b, a

    def apply_logical_z(self):
        self._snapshot()
        self.state[self.logical_one_idx] *= -1

    def apply_logical_hadamard(self):
        self._snapshot()
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        c = (a + b)/np.sqrt(2)
        d = (a - b)/np.sqrt(2)
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = c, d

    def apply_logical_rx(self, theta: float):
        self._snapshot()
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        m, n = np.cos(theta/2), -1j*np.sin(theta/2)
        U = np.array([[m, n], [n, m]], dtype=complex)
        new = U @ np.array([a, b])
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = new

    def apply_logical_ry(self, theta: float):
        self._snapshot()
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        m, n = np.cos(theta/2), np.sin(theta/2)
        U = np.array([[m, -n], [n, m]], dtype=complex)
        new = U @ np.array([a, b])
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = new

    def apply_logical_rz(self, theta: float):
        self._snapshot()
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        self.state[self.logical_zero_idx] = a * np.exp(-1j*theta/2)
        self.state[self.logical_one_idx]  = b * np.exp(1j*theta/2)

    def apply_logical_s(self):
        self._snapshot()
        self.state[self.logical_one_idx] *= 1j

    def apply_logical_t(self):
        self._snapshot()
        self.state[self.logical_one_idx] *= np.exp(1j*np.pi/4)

    def apply_amplitude_damping(self, dt: float):
        if self.T1 is None:
            raise ValueError("T1 not set")
        self._snapshot()
        gamma = dt / self.T1
        decay = np.exp(-gamma)
        pop1 = abs(self.state[self.logical_one_idx])**2
        self.state[self.logical_one_idx] *= decay
        self.state[self.logical_zero_idx] += np.sqrt(max(0.0, 1 - decay**2)) * np.sqrt(pop1)
        self.state = normalize_state(self.state)

    def apply_phase_damping(self, dt: float):
        if self.T2 is None:
            raise ValueError("T2 not set")
        self._snapshot()
        gamma = dt / self.T2
        decay = np.exp(-gamma)
        self.state *= decay
        self.state = normalize_state(self.state)

    def apply_depolarizing(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("p out of range")
        self._snapshot()
        uniform = np.ones(self.dimension, dtype=complex)/np.sqrt(self.dimension)
        self.state = np.sqrt(1-p)*self.state + np.sqrt(p)*uniform
        self.state = normalize_state(self.state)

    def apply_phase_noise(self, gamma: float):
        self._snapshot()
        for i in range(self.dimension):
            if i not in (self.logical_zero_idx, self.logical_one_idx):
                phi = np.random.uniform(-gamma, gamma)
                self.state[i] *= np.exp(1j*phi)
        self.state = normalize_state(self.state)

    def measure_logical(self) -> int:
        p0 = abs(self.state[self.logical_zero_idx])**2
        p1 = abs(self.state[self.logical_one_idx])**2
        denom = p0 + p1
        if denom <= 0:
            res = int(np.random.rand() < 0.5)
        else:
            res = 0 if np.random.rand() < p0/denom else 1
        if res == 0: self.reset_to_logical_zero()
        else: self.reset_to_logical_one()
        return res

    def measure_full(self) -> int:
        probs = abs(self.state)**2
        probs = probs / max(1e-12, np.sum(probs))
        idx   = int(np.random.choice(self.dimension, p=probs))
        self.reset_to_basis(idx)
        return idx

    def get_hidden_population_sum(self) -> float:
        return float(sum(abs(self.state[i])**2 for i in range(self.dimension)
                         if i not in (self.logical_zero_idx, self.logical_one_idx)))

class HybridQubit:
    def __init__(self, name: str, dimension: int = 2, logical_zero_idx: int = 0, logical_one_idx: int = 1, backend=None):
        self.name = name
        self.dimension = int(dimension)
        self.logical_zero_idx = int(logical_zero_idx)
        self.logical_one_idx  = int(logical_one_idx)
        self.num_qubits = int(np.ceil(np.log2(max(2, self.dimension))))
        self.backend = backend or (AerSimulator() if AerSimulator else None)
        self.circuit = QuantumCircuit(self.num_qubits) if QuantumCircuit else None

        if self.circuit is not None:
            init = np.zeros(2**self.num_qubits, dtype=complex)
            init[self.logical_zero_idx] = 1.0
            self.circuit.initialize(init, range(self.num_qubits))

        self._history: List[np.ndarray] = []
        self._future: List[np.ndarray] = []

    def _push_sv(self, sv: np.ndarray):
        self._history.append(sv.copy())
        self._future.clear()

    def _set_sv(self, sv: np.ndarray):
        if self.circuit is None:
            return
        self.circuit = QuantumCircuit(self.num_qubits)
        self.circuit.initialize(sv, range(self.num_qubits))

    def get_statevector(self) -> np.ndarray:
        if self.circuit is None or Statevector is None:
            return np.zeros(self.dimension, dtype=complex)
        sv = Statevector.from_instruction(self.circuit)
        return np.array(sv.data)

    def reset_to_logical_zero(self):
        sv = self.get_statevector(); self._push_sv(sv)
        sv[:] = 0; sv[self.logical_zero_idx] = 1.0
        self._set_sv(sv)

    def reset_to_logical_one(self):
        sv = self.get_statevector(); self._push_sv(sv)
        sv[:] = 0; sv[self.logical_one_idx] = 1.0
        self._set_sv(sv)

    def reset_to_basis(self, idx: int):
        sv = self.get_statevector(); self._push_sv(sv)
        sv[:] = 0; sv[idx] = 1.0
        self._set_sv(sv)

    def initialize(self, amplitudes: Union[List[complex], np.ndarray]):
        sv = self.get_statevector(); self._push_sv(sv)
        amps = np.array(amplitudes, dtype=complex)
        if not np.isclose(np.linalg.norm(amps), 1.0):
            raise ValueError("State must be normalized")
        self._set_sv(amps)

    def _apply_2d(self, U: np.ndarray):
        sv = self.get_statevector(); self._push_sv(sv)
        a, b = sv[self.logical_zero_idx], sv[self.logical_one_idx]
        new = U @ np.array([a, b])
        sv[self.logical_zero_idx], sv[self.logical_one_idx] = new
        self._set_sv(sv)

    def apply_logical_x(self):
        U = np.array([[0, 1],[1, 0]], dtype=complex)
        self._apply_2d(U)

    def apply_logical_z(self):
        U = np.array([[1, 0],[0,-1]], dtype=complex)
        self._apply_2d(U)

    def apply_logical_hadamard(self):
        U = (1/np.sqrt(2))*np.array([[1, 1],[1,-1]], dtype=complex)
        self._apply_2d(U)

    def apply_logical_rx(self, theta: float):
        m, n = np.cos(theta/2), -1j*np.sin(theta/2)
        U = np.array([[m, n],[n, m]], dtype=complex)
        self._apply_2d(U)

    def apply_logical_ry(self, theta: float):
        m, n = np.cos(theta/2), np.sin(theta/2)
        U = np.array([[m, -n],[n, m]], dtype=complex)
        self._apply_2d(U)

    def apply_logical_rz(self, theta: float):
        U = np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype=complex)
        self._apply_2d(U)

    def apply_logical_s(self):  self.apply_logical_rz(np.pi/2)
    def apply_logical_t(self):  self.apply_logical_rz(np.pi/4)

    def measure_logical(self) -> int:
        sv = self.get_statevector()
        p0 = abs(sv[self.logical_zero_idx])**2
        p1 = abs(sv[self.logical_one_idx])**2
        denom = p0 + p1
        if denom <= 0: res = int(np.random.rand() < 0.5)
        else: res = 0 if np.random.rand() < p0/denom else 1
        if res == 0: self.reset_to_logical_zero()
        else: self.reset_to_logical_one()
        return res

    def measure_full(self) -> int:
        sv = self.get_statevector()
        probs = abs(sv)**2; probs = probs / max(1e-12, np.sum(probs))
        idx = int(np.random.choice(len(sv), p=probs))
        self.reset_to_basis(idx)
        return idx

    def get_hidden_population_sum(self) -> float:
        sv = self.get_statevector()
        total = 0.0
        for i in range(len(sv)):
            if i in (self.logical_zero_idx, self.logical_one_idx): continue
            total += abs(sv[i])**2
        return float(total)

    def undo(self) -> bool:
        if not self._history: return False
        cur = self.get_statevector()
        self._future.append(cur.copy())
        last = self._history.pop()
        self._set_sv(last)
        return True

    def redo(self) -> bool:
        if not self._future: return False
        cur = self.get_statevector()
        self._history.append(cur.copy())
        nxt = self._future.pop()
        self._set_sv(nxt)
        return True

class QuantumChannel:
    def __init__(self, name: str, num_subbits: int):
        self.name = name
        self.subbits: List[Cubit] = [Cubit(f"{name}_s{i+1}") for i in range(int(num_subbits))]

    def encode(self, thetas: List[float], phis: Optional[List[float]] = None):
        if phis is None:
            phis = [0]*len(thetas)
        for s, t, p in zip(self.subbits, thetas, phis):
            s.reset_to_logical_zero()
            s.apply_logical_ry(float(t))
            s.apply_logical_rz(float(p))

    def decode(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for s in self.subbits:
            a = s.state[s.logical_zero_idx]
            b = s.state[s.logical_one_idx]
            norm = np.sqrt(abs(a)**2 + abs(b)**2)
            if norm <= 1e-12:
                out.append((0.0, 0.0))
                continue
            a, b = a/norm, b/norm
            theta = 2*np.arccos(abs(a))
            phi   = float(np.angle(b))
            out.append((float(theta), float(phi)))
        return out

class QuantumManager:
    def __init__(self):
        self.qubits: Dict[str, Union[Cubit, HybridQubit]] = {}
        self.channels: Dict[str, QuantumChannel] = {}

    def add_cubit(self, name: str, dimension: int = 2, lz: int = 0, lo: int = 1):
        if name in self.qubits:
            raise ValueError("Duplicate qubit name")
        self.qubits[name] = Cubit(name, dimension, lz, lo)

    def add_hybrid(self, name: str, dimension: int = 2, lz: int = 0, lo: int = 1, backend=None):
        if name in self.qubits:
            raise ValueError("Duplicate qubit name")
        self.qubits[name] = HybridQubit(name, dimension, lz, lo, backend)

    def add_channel(self, name: str, num_subbits: int):
        if name in self.channels:
            raise ValueError("Duplicate channel name")
        self.channels[name] = QuantumChannel(name, num_subbits)

    def remove(self, name: str):
        self.qubits.pop(name, None)
        self.channels.pop(name, None)

class QbtFile:
    @staticmethod
    def save_object(obj: Union[Cubit, HybridQubit], path: str):
        if isinstance(obj, Cubit):
            d = {
                "type": "Cubit",
                "name": obj.name,
                "dimension": obj.dimension,
                "logical_zero_idx": obj.logical_zero_idx,
                "logical_one_idx": obj.logical_one_idx,
                "statevector": safe_complex_list_to_pairs(obj.state),
                "T1": obj.T1,
                "T2": obj.T2,
                "gate_errors": obj.gate_errors
            }
        elif isinstance(obj, HybridQubit):
            sv = obj.get_statevector()
            d = {
                "type": "HybridQubit",
                "name": obj.name,
                "dimension": obj.dimension,
                "logical_zero_idx": obj.logical_zero_idx,
                "logical_one_idx": obj.logical_one_idx,
                "statevector": safe_complex_list_to_pairs(sv)
            }
        else:
            raise ValueError("Can only save Cubit or HybridQubit")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
            
    def load_object(path: str) -> Union[Cubit, HybridQubit]:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        t = d["type"]
        if t == "Cubit":
            c = Cubit(d["name"], d["dimension"], d["logical_zero_idx"], d["logical_one_idx"])
            amps = pairs_to_complex_list(d["statevector"])
            c.initialize(amps)
            c.T1 = d.get("T1")
            c.T2 = d.get("T2")
            c.gate_errors = d.get("gate_errors", {})
            return c
        elif t == "HybridQubit":
            h = HybridQubit(d["name"], d["dimension"], d["logical_zero_idx"], d["logical_one_idx"])
            amps = pairs_to_complex_list(d["statevector"])
            h.initialize(amps)
            return h
        else:
            raise ValueError("Unsupported QBT file type")

    def save_project(qm: QuantumManager, path: str):
        data = {
            "qubits": [],
            "channels": []
        }
        for nm, q in qm.qubits.items():
            if isinstance(q, Cubit):
                data["qubits"].append({
                    "kind": "cubit",
                    "payload": {
                        "name": q.name,
                        "dimension": q.dimension,
                        "lz": q.logical_zero_idx,
                        "lo": q.logical_one_idx,
                        "statevector": safe_complex_list_to_pairs(q.get_statevector())
                    }
                })
            else:
                data["qubits"].append({
                    "kind": "hybrid",
                    "payload": {
                        "name": q.name,
                        "dimension": q.dimension,
                        "lz": q.logical_zero_idx,
                        "lo": q.logical_one_idx,
                        "statevector": safe_complex_list_to_pairs(q.get_statevector())
                    }
                })
        for nm, ch in qm.channels.items():
            data["channels"].append({
                "name": nm,
                "subbits": [safe_complex_list_to_pairs(s.get_statevector()) for s in ch.subbits]
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_project(path: str) -> QuantumManager:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        qm = QuantumManager()
        for item in data.get("qubits", []):
            kind = item["kind"]
            pl = item["payload"]
            if kind == "cubit":
                q = Cubit(pl["name"], pl["dimension"], pl["lz"], pl["lo"])
                q.initialize(pairs_to_complex_list(pl["statevector"]))
                qm.qubits[q.name] = q
            else:
                q = HybridQubit(pl["name"], pl["dimension"], pl["lz"], pl["lo"])
                q.initialize(pairs_to_complex_list(pl["statevector"]))
                qm.qubits[q.name] = q
        for ch in data.get("channels", []):
            qc = QuantumChannel(ch["name"], len(ch["subbits"]))
            for sv_pairs, sub in zip(ch["subbits"], qc.subbits):
                sub.initialize(pairs_to_complex_list(sv_pairs))
            qm.channels[qc.name] = qc
        return qm

class IBMQManager: #Works but dicey, currently experencing issues with ibm provider v1 backends compatibility (waiting for qiskit fix)
    def __init__(self):
        self.connected = False
        self.provider  = None
        self.backends  = []
        self.api_error = None

    def connect(self, token: Optional[str] = None):
        if not _IBM_AVAILABLE:
            self.api_error = (
                "IBMProvider not available. Install qiskit-ibm-provider and ensure Qiskit compatibility."
            )
            self.connected = False
            return
        try:
            if token:
                prov = IBMProvider(token=token)
            else:
                prov = IBMProvider()
            self.provider = prov
            self.backends = list(prov.backends())
            self.connected = True
            self.api_error = None
        except Exception as e:
            self.api_error = str(e)
            self.connected = False

    def get_backends(self):
        return self.backends

    def get_backend(self, name: str):
        for b in self.backends:
            try:
                if getattr(b, "name", None) == name or getattr(b, "name", lambda: None)() == name:
                    return b
            except Exception:
                pass
        return None

class Exporter: #Used from a given build (I dont use qasm much) pr if issues or needs updated
    def export_qasm2(circuit: QuantumCircuit) -> Optional[str]:
        try:
            if _QASM2_DUMPS is not None:
                return _QASM2_DUMPS(circuit)
        except Exception:
            pass
        try:
            return circuit.qasm()
        except Exception:
            return None

    def import_qasm2(text: str) -> Optional[QuantumCircuit]:
        try:
            if _QASM2_LOADS is not None:
                return _QASM2_LOADS(text)
        except Exception:
            pass
        return None

    def export_notebook(path: str, title: str, circuits: List[QuantumCircuit], notes: str = "") -> bool:
        try:
            import nbformat as nbf
        except Exception:
            return False
        nb = nbf.v4.new_notebook()
        cells = []
        cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n{notes}"))
        cells.append(nbf.v4.new_code_cell("from qiskit import QuantumCircuit, transpile\nfrom qiskit_aer import AerSimulator\nfrom qiskit.visualization import plot_histogram, plot_bloch_multivector\nfrom qiskit.quantum_info import Statevector\n%matplotlib inline"))
        for idx, qc in enumerate(circuits):
            qasm = Exporter.export_qasm2(qc)
            if qasm is None: qasm = "# QASM export unavailable for this circuit/version."
            code = f"""# Circuit {idx+1}
qasm_text = r'''{qasm}'''
try:
    from qiskit.qasm2 import loads as qasm2_loads
    qc = qasm2_loads(qasm_text)
except Exception:
    qc = None
if qc is None:
    print("Unable to parse QASM. Skipping execution for this circuit.")
else:
    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    result = sim.run(tqc).result()
    sv = Statevector.from_instruction(qc)
    display(plot_bloch_multivector(sv))
    counts = result.get_counts(tqc) if hasattr(result, "get_counts") else None
    if counts:
        display(plot_histogram(counts))
"""
            cells.append(nbf.v4.new_code_cell(code))
        nb["cells"] = cells
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        return True

class BackgroundTask:
    name: str
    target: Any
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)

class TaskRunner:
    """Run long tasks off the Tk main thread and post results back."""
    def __init__(self, ui_callback):
        self.ui_callback = ui_callback  
        self._tasks = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def post(self, task: BackgroundTask):
        self._tasks.put(task)

    def shutdown(self):
        self._stop.set()
        self._tasks.put(None)

    def _loop(self):
        while not self._stop.is_set():
            task = self._tasks.get()
            if task is None:
                break
            try:
                res = task.target(*task.args, **task.kwargs)
                self.ui_callback("task_done", {"name": task.name, "result": res, "ok": True})
            except Exception as e:
                self.ui_callback("task_done", {"name": task.name, "error": str(e), "ok": False})

class AppModel:
    def __init__(self):
        self.qm = QuantumManager()
        self.ibm = IBMQManager()
        self.selected: Optional[Union[Cubit, HybridQubit, QuantumChannel]] = None
        self._circuits_cache: List[QuantumCircuit] = []

    def set_selected(self, obj): self.selected = obj

    def build_example_circuit(self) -> Optional[QuantumCircuit]:
        if QuantumCircuit is None: return None
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0,1)
        return qc

    def get_circuits(self) -> List[QuantumCircuit]:
        circs: List[QuantumCircuit] = []
        for nm, q in self.qm.qubits.items():
            if QuantumCircuit is None: continue
            n = int(np.ceil(np.log2(max(2, q.dimension if isinstance(q, Cubit) else q.dimension))))
            qc = QuantumCircuit(n, name=nm)
            sv = q.get_statevector()
            if np.isclose(np.linalg.norm(sv), 0):
                sv = np.zeros(2**n, dtype=complex); sv[0] = 1.0
            try:
                qc.initialize(sv, range(n))
            except Exception:
                continue
            circs.append(qc)
        self._circuits_cache = circs
        return circs

class ChannelDialog(simpledialog.Dialog):
    def __init__(self, parent, title="Add Channel"):
        super().__init__(parent, title=title)

    def body(self, master):
        try:
            master.winfo_toplevel().attributes("-topmost", True)
            master.winfo_toplevel().after(10, lambda: master.winfo_toplevel().attributes("-topmost", False))
        except Exception:
            pass
        master.columnconfigure(1, weight=1)
        ttk.Label(master, text="Channel Name:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(master, textvariable=self.name_var, width=28)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=8)

        ttk.Label(master, text="Subbits (≥1):").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.n_var = tk.IntVar(value=2)
        self.n_spin = tk.Spinbox(master, from_=1, to=1024, textvariable=self.n_var, width=8, justify="center")
        self.n_spin.grid(row=1, column=1, sticky="w", padx=10, pady=8)

        return self.name_entry 

    def validate(self):
        nm = self.name_var.get().strip()
        try:
            n = int(self.n_var.get())
        except Exception:
            n = 0
        if not nm or n < 1:
            messagebox.showwarning("Invalid Input", "Provide a name and subbits ≥ 1.")
            return False
        self.result = (nm, n)
        return True

class QubitDialog(simpledialog.Dialog):
    def __init__(self, parent, title="Add Hybrid Qubit"):
        super().__init__(parent, title=title)

    def body(self, master):
        try:
            master.winfo_toplevel().attributes("-topmost", True)
            master.winfo_toplevel().after(10, lambda: master.winfo_toplevel().attributes("-topmost", False))
        except Exception:
            pass
        master.columnconfigure(1, weight=1)
        ttk.Label(master, text="Qubit Name:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(master, textvariable=self.name_var, width=28)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=8)

        ttk.Label(master, text="Dimension (≥2):").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.dim_var = tk.IntVar(value=2)
        self.dim_spin = tk.Spinbox(master, from_=2, to=4096, textvariable=self.dim_var, width=8, justify="center")
        self.dim_spin.grid(row=1, column=1, sticky="w", padx=10, pady=8)

        return self.name_entry

    def validate(self):
        nm = self.name_var.get().strip()
        try:
            d = int(self.dim_var.get())
        except Exception:
            d = 0
        if not nm or d < 2:
            messagebox.showwarning("Invalid Input", "Provide a name and dimension ≥ 2.")
            return False
        self.result = (nm, d)
        return True

class CubitDialog(simpledialog.Dialog):
    def __init__(self, parent, title="Add Cubit"):
        super().__init__(parent, title=title)

    def body(self, master):
        try:
            master.winfo_toplevel().attributes("-topmost", True)
            master.winfo_toplevel().after(10, lambda: master.winfo_toplevel().attributes("-topmost", False))
        except Exception:
            pass
        master.columnconfigure(1, weight=1)
        ttk.Label(master, text="Cubit Name:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(master, textvariable=self.name_var, width=28)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=8)

        ttk.Label(master, text="Dimension (≥2):").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.dim_var = tk.IntVar(value=2)
        self.dim_spin = tk.Spinbox(master, from_=2, to=4096, textvariable=self.dim_var, width=8, justify="center")
        self.dim_spin.grid(row=1, column=1, sticky="w", padx=10, pady=8)

        return self.name_entry

    def validate(self):
        nm = self.name_var.get().strip()
        try:
            d = int(self.dim_var.get())
        except Exception:
            d = 0
        if not nm or d < 2:
            messagebox.showwarning("Invalid Input", "Provide a name and dimension ≥ 2.")
            return False
        self.result = (nm, d)
        return True

class QubitStudioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Qubit Studio")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)
        self.style = ttk.Style(self.root)
        self.theme_mode = tk.StringVar(value="light")
        self._apply_theme()
        self.model = AppModel()
        self.runner = TaskRunner(self._on_task_done)
        self.status_var = tk.StringVar(value="Ready.")
        self._build_menu()
        self._build_layout()
        self._refresh_resources()
        self._refresh_all_tabs()

    def _apply_theme(self):
        mode = self.theme_mode.get()
        if mode == "dark":
            bg = "#0f1216"; fg = "#e9eef6"; accent = "#4f8cff"; card = "#171b21"
            self.style.theme_use("clam")
            self.style.configure(".", background=bg, foreground=fg)
            self.style.configure("TFrame", background=bg)
            self.style.configure("TLabel", background=bg, foreground=fg)
            self.style.configure("TButton", background=card, foreground=fg, padding=6, relief="flat")
            self.style.configure("Card.TFrame", background=card, relief="raised", borderwidth=1)
            self.style.map("TButton", background=[("active", "#263040")])
        else:
            bg = "#f6f8fc"; fg = "#0e1116"; accent = "#1d6ef2"; card = "#ffffff"
            self.style.theme_use("clam")
            self.style.configure(".", background=bg, foreground=fg)
            self.style.configure("TFrame", background=bg)
            self.style.configure("TLabel", background=bg, foreground=fg)
            self.style.configure("TButton", background=card, foreground="#0f172a", padding=6, relief="flat")
            self.style.configure("Card.TFrame", background=card, relief="raised", borderwidth=1)
            self.style.map("TButton", background=[("active", "#e2e8f0")])

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="New Project", command=self._new_project)
        filem.add_command(label="Open Project...", command=self._open_project)
        filem.add_command(label="Save Project As...", command=self._save_project_as)
        filem.add_separator()
        filem.add_command(label="Import QASM...", command=self._import_qasm)
        filem.add_command(label="Export Selected → QASM", command=self._export_selected_qasm)
        filem.add_command(label="Export All → Notebook", command=self._export_notebook)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=filem)

        viewm = tk.Menu(menubar, tearoff=0)
        viewm.add_radiobutton(label="Light Mode", variable=self.theme_mode, value="light", command=self._apply_theme)
        viewm.add_radiobutton(label="Dark Mode",  variable=self.theme_mode, value="dark",  command=self._apply_theme)
        menubar.add_cascade(label="View", menu=viewm)

        ibmm = tk.Menu(menubar, tearoff=0)
        ibmm.add_command(label="Connect IBM (Token)...", command=self._connect_ibm)
        menubar.add_cascade(label="IBM Quantum", menu=ibmm)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=helpm)

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.pw.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.left = ttk.Frame(self.pw, width=420, style="TFrame")
        self.left.grid_propagate(False)
        self.right = ttk.Frame(self.pw, style="TFrame")
        try:
            self.pw.add(self.left, weight=0)
            self.pw.add(self.right, weight=1)
        except Exception:
            self.pw.add(self.left)
            self.pw.add(self.right)
    
        hdr = ttk.Frame(self.left, style="TFrame"); hdr.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(hdr, text="Resources", font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        ttk.Button(hdr, text="➕ Channel", command=self._add_channel_popup).pack(side=tk.RIGHT, padx=4)
        ttk.Button(hdr, text="➕ Cubit", command=self._add_cubit_popup).pack(side=tk.RIGHT, padx=4)
        ttk.Button(hdr, text="➕ Qubit", command=self._add_qubit_popup).pack(side=tk.RIGHT, padx=4)

        self.res_canvas = tk.Canvas(self.left, borderwidth=0, highlightthickness=0)
        self.res_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=10, pady=(0,10))
        self.res_scroll = ttk.Scrollbar(self.left, orient="vertical", command=self.res_canvas.yview)
        self.res_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(0,10))
        self.res_canvas.configure(yscrollcommand=self.res_scroll.set)
        self.res_frame = ttk.Frame(self.res_canvas, style="TFrame")
        self._res_window = self.res_canvas.create_window((0, 0), window=self.res_frame, anchor="nw")
        self.res_frame.bind("<Configure>", lambda e: self.res_canvas.configure(scrollregion=self.res_canvas.bbox("all")))
        self.res_canvas.bind("<Configure>", lambda e: self.res_canvas.itemconfigure(self._res_window, width=e.width))
        self.nb = ttk.Notebook(self.right)
        self.nb.pack(fill=tk.BOTH, expand=1, padx=0, pady=0)
        self.tab_overview = ttk.Frame(self.nb, style="TFrame")
        self.tab_explore  = ttk.Frame(self.nb, style="TFrame")
        self.tab_channels = ttk.Frame(self.nb, style="TFrame")
        self.tab_ibm      = ttk.Frame(self.nb, style="TFrame")
        self.tab_lessons  = ttk.Frame(self.nb, style="TFrame")
        self.nb.add(self.tab_overview, text="Overview")
        self.nb.add(self.tab_explore,  text="Qubit Explorer")
        self.nb.add(self.tab_channels, text="Channel Ops")
        self.nb.add(self.tab_ibm,      text="IBM Cloud")
        self.nb.add(self.tab_lessons,  text="Lessons")
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._refresh_current_tab())
    
    def _refresh_resources(self):
    
            for w in self.res_frame.winfo_children():
                w.destroy()
    
            if not self.model.qm.qubits and not self.model.qm.channels:
                ttk.Label(self.res_frame, text="No resources.\nUse the buttons above to add items.",
                          justify=tk.CENTER).pack(pady=20)
                return
    
            def card(title: str, subtitle: str, obj):
                frm = ttk.Frame(self.res_frame, style="Card.TFrame")
                frm.pack(fill=tk.X, pady=6)
                ttk.Label(frm, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(8,0))
                ttk.Label(frm, text=subtitle, font=("Consolas", 9), wraplength=380, justify=tk.LEFT).pack(anchor="w", padx=10, pady=(2,8))
                btns = ttk.Frame(frm, style="Card.TFrame"); btns.pack(fill=tk.X, padx=8, pady=(0,8))
                ttk.Button(btns, text="Select", command=lambda o=obj: self._select_object(o)).pack(side=tk.LEFT, padx=3)
                ttk.Button(btns, text="Save", command=lambda o=obj: self._save_object(o)).pack(side=tk.LEFT, padx=3)
                ttk.Button(btns, text="Remove", command=lambda o=obj: self._remove_object(o)).pack(side=tk.RIGHT, padx=3)
                return frm
    
            for nm in sorted(self.model.qm.qubits):
                qb = self.model.qm.qubits[nm]
                sv = qb.get_statevector()
                summary = ", ".join(f"{np.real(a):.2f}{np.imag(a):+.2f}j" for a in sv[:min(12,len(sv))])
                tp = "HybridQubit" if isinstance(qb, HybridQubit) else "Cubit"
                card(f"{tp}: {qb.name}", f"State[{len(sv)}]: [{summary}{'...' if len(sv)>8 else ''}]", qb)
    
            for nm in sorted(self.model.qm.channels):
                ch = self.model.qm.channels[nm]
                card(f"Channel: {ch.name}", f"Subbits: {len(ch.subbits)}", ch)
    
    def _select_object(self, obj):
        self.model.set_selected(obj)
        self.status_var.set(f"Selected: {getattr(obj, 'name', type(obj).__name__)}")
        self._refresh_all_tabs()

    def _save_object(self, obj):
        p = filedialog.asksaveasfilename(defaultextension=".qbt", filetypes=[("Qubit Files", "*.qbt")])
        if not p: return
        try:
            QbtFile.save_object(obj, p)
            messagebox.showinfo("Saved", "Saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _remove_object(self, obj):
        name = getattr(obj, "name", None)
        if not name:
            return
        self.model.qm.remove(name)
        if self.model.selected is obj:
            self.model.selected = None
        self._refresh_resources()
        self._refresh_all_tabs()

    def _refresh_all_tabs(self):
        self._show_overview()
        self._show_explore()
        self._show_channels()
        self._show_ibm()
        self._show_lessons()

    def _refresh_current_tab(self):
        idx = self.nb.index("current")
        if idx == 0: self._show_overview()
        elif idx == 1: self._show_explore()
        elif idx == 2: self._show_channels()
        elif idx == 3: self._show_ibm()
        elif idx == 4: self._show_lessons()

    def _show_overview(self):
        tab = self.tab_overview
        for w in tab.winfo_children():
            w.destroy()
        pad = 12
        ttk.Label(tab, text="Qubit Studio — Dashboard", font=("Segoe UI", 18, "bold")).pack(pady=pad)
        msg = (
            f"Qubits/Cubits: {len(self.model.qm.qubits)}    "
            f"Channels: {len(self.model.qm.channels)}\n\n"
            "• Build and explore quantum states visually\n"
            "• Export circuits to OpenQASM / Jupyter Notebook\n"
            "• Run locally (Aer) or on IBM Quantum devices\n"
            "• Noise models, step-through, undo/redo\n"
            "• Lessons from beginner to advanced"
        )
        ttk.Label(tab, text=msg, font=("Segoe UI", 12), justify=tk.LEFT).pack(pady=pad)

        actions = ttk.Frame(tab, style="TFrame"); actions.pack(pady=pad)
        ttk.Button(actions, text="Add Example Entangler (2q)", command=self._add_example_entangler).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Export All → Notebook", command=self._export_notebook).pack(side=tk.LEFT, padx=6)

    def _add_example_entangler(self):
        try:
            self.model.qm.add_hybrid("BellPair", 4, 0, 1)
        except Exception:
            pass
        qb = self.model.qm.qubits.get("BellPair")
        if isinstance(qb, HybridQubit):
            sv = qb.get_statevector()
            if len(sv) >= 4:
                qb.apply_logical_hadamard()
        self._refresh_resources()
        self._refresh_all_tabs()

    def _show_explore(self):
        tab = self.tab_explore
        for w in tab.winfo_children():
            w.destroy()

        if not self.model.qm.qubits:
            ttk.Label(tab, text="No qubits/cubits created yet.", font=("Segoe UI", 12)).pack(pady=24)
            return

        for nm in sorted(self.model.qm.qubits):
            qb = self.model.qm.qubits[nm]
            card = ttk.Frame(tab, style="Card.TFrame")
            card.pack(fill=tk.X, padx=16, pady=10)

            tp = "HybridQubit" if isinstance(qb, HybridQubit) else "Cubit"
            header = ttk.Frame(card, style="Card.TFrame"); header.pack(fill=tk.X, padx=10, pady=(8,4))
            ttk.Label(header, text=f"{tp}: {qb.name}", font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT)
            ttk.Button(header, text="Undo", command=lambda q=qb: self._undo_q(q)).pack(side=tk.RIGHT, padx=4)
            ttk.Button(header, text="Redo", command=lambda q=qb: self._redo_q(q)).pack(side=tk.RIGHT, padx=4)

            vis = ttk.Frame(card, style="Card.TFrame")
            vis.pack(fill=tk.X, padx=10, pady=4)
            sv = qb.get_statevector()
            self._plot_statevector(vis, sv)

            stats = self._state_stats(qb)
            ttk.Label(card, text=stats, font=("Consolas", 10)).pack(anchor="w", padx=12, pady=4)

            ops = ttk.Frame(card, style="Card.TFrame"); ops.pack(fill=tk.X, padx=10, pady=(4,12))
            r1 = ttk.Frame(ops, style="Card.TFrame"); r1.pack(fill=tk.X, pady=2)
            ttk.Button(r1, text="X",  command=lambda q=qb:self._op(q,"x")).pack(side=tk.LEFT, padx=3)
            ToolTip(r1.winfo_children()[-1], "Pauli-X")
            ttk.Button(r1, text="Z",  command=lambda q=qb:self._op(q,"z")).pack(side=tk.LEFT, padx=3)
            ToolTip(r1.winfo_children()[-1], "Pauli-Z")
            ttk.Button(r1, text="H",  command=lambda q=qb:self._op(q,"h")).pack(side=tk.LEFT, padx=3)
            ToolTip(r1.winfo_children()[-1], "Hadamard")
            ttk.Button(r1, text="S",  command=lambda q=qb:self._op(q,"s")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r1, text="T",  command=lambda q=qb:self._op(q,"t")).pack(side=tk.LEFT, padx=3)

            r2 = ttk.Frame(ops, style="Card.TFrame"); r2.pack(fill=tk.X, pady=2)
            ttk.Button(r2, text="Rxθ", command=lambda q=qb:self._op_angle(q,"rx")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r2, text="Ryθ", command=lambda q=qb:self._op_angle(q,"ry")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r2, text="Rzθ", command=lambda q=qb:self._op_angle(q,"rz")).pack(side=tk.LEFT, padx=3)

            r3 = ttk.Frame(ops, style="Card.TFrame"); r3.pack(fill=tk.X, pady=2)
            ttk.Button(r3, text="Reset |0⟩", command=lambda q=qb:self._op(q,"reset0")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r3, text="Reset |1⟩", command=lambda q=qb:self._op(q,"reset1")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r3, text="Measure Logical", command=lambda q=qb:self._measure(q,"logical")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r3, text="Measure Full",    command=lambda q=qb:self._measure(q,"full")).pack(side=tk.LEFT, padx=3)

            r4 = ttk.Frame(ops, style="Card.TFrame"); r4.pack(fill=tk.X, pady=2)
            ttk.Button(r4, text="Amplitude Damping", command=lambda q=qb:self._noise(q,"amp")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r4, text="Phase Damping",     command=lambda q=qb:self._noise(q,"phase")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r4, text="Depolarizing",      command=lambda q=qb:self._noise(q,"depol")).pack(side=tk.LEFT, padx=3)
            ttk.Button(r4, text="Phase Noise",       command=lambda q=qb:self._noise(q,"phasen")).pack(side=tk.LEFT, padx=3)

            r5 = ttk.Frame(ops, style="Card.TFrame"); r5.pack(fill=tk.X, pady=2)
            ttk.Button(r5, text="Export QASM (init-only)", command=lambda q=qb:self._export_qasm_for_qubit(q)).pack(side=tk.LEFT, padx=3)

    def _plot_statevector(self, parent, sv: np.ndarray):
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(parent, text="matplotlib not available — install for plots!", foreground="#c00").pack()
            return
        fig, ax = plt.subplots(figsize=(5.0, 2.0), dpi=100)
        prob = np.abs(sv)**2
        ax.bar(range(len(sv)), prob)
        ax.set_ylim(0, 1)
        ax.set_ylabel("|a|²")
        ax.set_xlabel("State index")
        ax.set_title("Statevector Probabilities")
        for i, h in enumerate(prob):
            ax.text(i, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.X)
        plt.close(fig)

    def _state_stats(self, qb: Union[Cubit, HybridQubit]) -> str:
        sv = qb.get_statevector()
        p0 = abs(sv[qb.logical_zero_idx])**2 if len(sv) > qb.logical_zero_idx else 0.0
        p1 = abs(sv[qb.logical_one_idx])**2  if len(sv) > qb.logical_one_idx  else 0.0
        hid = qb.get_hidden_population_sum()
        return f"|0⟩={p0:.3f}  |1⟩={p1:.3f}  hidden={hid:.3f}  (dim={len(sv)})"

    def _undo_q(self, qb):
        if qb.undo():
            self._refresh_all_tabs()

    def _redo_q(self, qb):
        if qb.redo():
            self._refresh_all_tabs()

    def _op(self, qb, op: str):
        try:
            if op=="x": qb.apply_logical_x()
            elif op=="z": qb.apply_logical_z()
            elif op=="h": qb.apply_logical_hadamard()
            elif op=="s": qb.apply_logical_s()
            elif op=="t": qb.apply_logical_t()
            elif op=="reset0": qb.reset_to_logical_zero()
            elif op=="reset1": qb.reset_to_logical_one()
            self._refresh_all_tabs()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _op_angle(self, qb, op: str):
        th = simpledialog.askfloat("Angle", "Angle (radians):")
        if th is None: return
        try:
            if op=="rx": qb.apply_logical_rx(th)
            elif op=="ry": qb.apply_logical_ry(th)
            elif op=="rz": qb.apply_logical_rz(th)
            self._refresh_all_tabs()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _measure(self, qb, mode: str):
        if mode=="logical":
            res = qb.measure_logical()
            messagebox.showinfo("Measurement", f"Logical result: {res}")
        else:
            res = qb.measure_full()
            messagebox.showinfo("Measurement", f"Full basis result: {res}")
        self._refresh_all_tabs()

    def _noise(self, qb, op: str):
        try:
            if op=="amp":
                dt = simpledialog.askfloat("dt","Time dt:")
                if dt is None: return
                qb.apply_amplitude_damping(dt)
            elif op=="phase":
                dt = simpledialog.askfloat("dt","Time dt:")
                if dt is None: return
                qb.apply_phase_damping(dt)
            elif op=="depol":
                p  = simpledialog.askfloat("p","Probability [0–1]:")
                if p is None: return
                qb.apply_depolarizing(clamp01(p))
            elif op=="phasen":
                g  = simpledialog.askfloat("gamma","Gamma:")
                if g is None: return
                qb.apply_phase_noise(g)
            self._refresh_all_tabs()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_qasm_for_qubit(self, qb):
        if QuantumCircuit is None:
            messagebox.showerror("Error", "Qiskit not available.")
            return
        n = int(np.ceil(np.log2(max(2, qb.dimension if isinstance(qb, Cubit) else qb.dimension))))
        qc = QuantumCircuit(n, name=qb.name)
        sv = qb.get_statevector()
        try:
            qc.initialize(sv, range(n))
        except Exception as e:
            messagebox.showerror("Error", f"Unable to build circuit: {e}")
            return
        text = Exporter.export_qasm2(qc)
        if not text:
            messagebox.showerror("Error", "QASM export unavailable.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".qasm", filetypes=[("OpenQASM", "*.qasm")])
        if not p: return
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        messagebox.showinfo("Exported", "OpenQASM saved.")

    def _show_channels(self):
        tab = self.tab_channels
        for w in tab.winfo_children():
            w.destroy()
        if not self.model.qm.channels:
            ttk.Label(tab, text="No channels created yet.", font=("Segoe UI", 12)).pack(pady=24)
            return

        for nm in sorted(self.model.qm.channels):
            ch = self.model.qm.channels[nm]
            card = ttk.Frame(tab, style="Card.TFrame"); card.pack(fill=tk.X, padx=16, pady=10)

            ttk.Label(card, text=f"Channel: {ch.name}", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=10, pady=(8,2))
            ttk.Label(card, text=f"Subbits: {len(ch.subbits)}", font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(0,8))

            txt = []
            for i, sub in enumerate(ch.subbits):
                sv = sub.get_statevector()
                s = f"Subbit {i+1}: [{np.real(sv[0]):.3f}{np.imag(sv[0]):+.3f}j, {np.real(sv[1]):.3f}{np.imag(sv[1]):+.3f}j]"
                txt.append(s)
            ttk.Label(card, text="\n".join(txt), font=("Consolas", 9)).pack(anchor="w", padx=12, pady=(0,8))

            ctr = ttk.Frame(card, style="Card.TFrame"); ctr.pack(pady=6, padx=10, fill=tk.X)
            ttk.Button(ctr, text="Encode", command=lambda c=ch:self._encode_channel(c)).pack(side=tk.LEFT, padx=6)
            ttk.Button(ctr, text="Decode", command=lambda c=ch:self._decode_channel(c)).pack(side=tk.LEFT, padx=6)

    def _encode_channel(self, ch: QuantumChannel):
        n = len(ch.subbits)
        ths, phs = [], []
        for i in range(n):
            th = simpledialog.askfloat(f"Subbit {i+1} θ", f"Enter θ for subbit {i+1}:")
            if th is None: return
            ph = simpledialog.askfloat(f"Subbit {i+1} φ", f"Enter φ for subbit {i+1}:", initialvalue=0.0)
            if ph is None: return
            ths.append(th); phs.append(ph)
        ch.encode(ths, phs)
        self._refresh_all_tabs()

    def _decode_channel(self, ch: QuantumChannel):
        res = ch.decode()
        txt = "\n".join(f"{i+1}: θ={th:.3f}, φ={ph:.3f}" for i, (th, ph) in enumerate(res))
        messagebox.showinfo("Decoded", txt)
        self._refresh_all_tabs()

    def _show_ibm(self):
        tab = self.tab_ibm
        for w in tab.winfo_children():
            w.destroy()

        ttk.Label(tab, text="IBM Quantum Cloud", font=("Segoe UI", 16, "bold")).pack(pady=8)

        status = "Connected ✓" if self.model.ibm.connected else ("Error: "+self.model.ibm.api_error if self.model.ibm.api_error else "Not Connected")
        ttk.Label(tab, text=f"Status: {status}", font=("Segoe UI", 12)).pack(pady=4)
        ttk.Button(tab, text="Connect / Retry", command=self._connect_ibm).pack(pady=4)

        if self.model.ibm.connected:
            wrap = ttk.Frame(tab, style="TFrame"); wrap.pack(fill=tk.BOTH, expand=1, padx=14, pady=8)
            wrap.columnconfigure(0, weight=1); wrap.columnconfigure(1, weight=1)
            ttk.Label(wrap, text="Available backends:", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
            lst = tk.Listbox(wrap, height=12)
            lst.grid(row=1, column=0, sticky="nsew", padx=(0,8))
            for b in self.model.ibm.get_backends():
                try:
                    name = getattr(b, "name", None) or b.name()
                except Exception:
                    name = str(b)
                lst.insert(tk.END, name)
            ctl = ttk.Frame(wrap, style="TFrame"); ctl.grid(row=1, column=1, sticky="nsew")
            ttk.Button(ctl, text="Run Selected on Backend", command=lambda:self._run_selected_on_backend(lst)).pack(pady=4, fill=tk.X)
            ttk.Button(ctl, text="Refresh Status", command=self._connect_ibm).pack(pady=4, fill=tk.X)
            ttk.Label(ctl, text="(Submission uses transpile + job monitor,\nthen results histogram shown if available)",
                      font=("Segoe UI",9)).pack(pady=6)

    def _connect_ibm(self):
        tok = simpledialog.askstring("IBM Quantum Token (optional)",
                                     "Enter IBM Quantum API token (leave blank to use saved account):")
        self.status_var.set("Connecting to IBM Quantum...")
        self.progress.start(10)
        def do():
            self.model.ibm.connect(tok or None)
            return True
        self.runner.post(BackgroundTask("ibm_connect", do))

    def _run_selected_on_backend(self, listbox: tk.Listbox):
        if not self.model.qm.qubits:
            messagebox.showinfo("Run", "No qubits to run. Create one first.")
            return
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("Run", "Select a backend first.")
            return
        try:
            name = listbox.get(sel[0])
        except Exception:
            messagebox.showerror("Run", "Invalid selection.")
            return
        backend = self.model.ibm.get_backend(name)
        if backend is None:
            messagebox.showerror("Run", "Backend not found.")
            return
        circs = self.model.get_circuits()
        if not circs:
            messagebox.showinfo("Run", "No circuits to submit.")
            return

        self.status_var.set(f"Submitting {len(circs)} circuit(s) to {name}...")
        self.progress.start(10)

        def do():
            from qiskit import transpile
            tqc = [transpile(qc, backend) for qc in circs]
            job = backend.run(tqc)
            while True:
                try:
                    st = job.status()
                    if str(st).lower() in ("jobstatus.done", "done", "jobstatus.done"):
                        break
                    time.sleep(2.0)
                except Exception:
                    time.sleep(2.0)
            try:
                res = job.result()
                return {"backend": name, "counts": [res.get_counts(t) for t in tqc if hasattr(res, "get_counts")]}
            except Exception as e:
                return {"backend": name, "counts": None, "error": str(e)}
        self.runner.post(BackgroundTask("ibm_run", do))

    def _show_lessons(self):
        tab = self.tab_lessons
        for w in tab.winfo_children():
            w.destroy()
        ttk.Label(tab, text="Lessons", font=("Segoe UI", 16, "bold")).pack(pady=8)
        desc = ("Beginner → single-qubit intuition (Bloch, H/X/Z, measurement)\n"
                "Intermediate → entanglement, two-qubit gates, histograms\n"
                "Advanced → algorithms, noise, hardware runs")
        ttk.Label(tab, text=desc).pack(pady=4)

        acts = ttk.Frame(tab, style="TFrame"); acts.pack(pady=8)
        ttk.Button(acts, text="Beginner: Create |+⟩ and measure many times", command=self._lesson_beginner_plus).pack(side=tk.LEFT, padx=6)
        ttk.Button(acts, text="Intermediate: Bell State (H+CX) demo", command=self._lesson_bell).pack(side=tk.LEFT, padx=6)

    def _lesson_beginner_plus(self):
        name = "LessonPlus"
        try:
            self.model.qm.add_hybrid(name, 2, 0, 1)
        except Exception:
            pass
        qb = self.model.qm.qubits.get(name)
        if isinstance(qb, HybridQubit):
            qb.reset_to_logical_zero()
            qb.apply_logical_hadamard()
        self._refresh_resources()
        self._refresh_all_tabs()
        messagebox.showinfo("Lesson", "Built |+⟩. Try measuring repeatedly to see ~50/50 outcomes.")

    def _lesson_bell(self):
        name = "LessonBell"
        try:
            self.model.qm.add_hybrid(name, 4, 0, 1)
        except Exception:
            pass
        qb = self.model.qm.qubits.get(name)
        if isinstance(qb, HybridQubit):
            qb.reset_to_logical_zero()
            qb.apply_logical_hadamard()
        self._refresh_resources()
        self._refresh_all_tabs()
        messagebox.showinfo("Lesson", "Constructed a Bell-like state (toy). Explore and measure.")

    def _new_project(self):
        self.model = AppModel()
        self._refresh_resources()
        self._refresh_all_tabs()

    def _open_project(self):
        p = filedialog.askopenfilename(defaultextension=".qproj", filetypes=[("QubitStudio Project", "*.qproj"),("JSON","*.json"),("All","*.*")])
        if not p: return
        try:
            qm = QbtFile.load_project(p)
            self.model.qm = qm
            self.model.selected = None
            self._refresh_resources()
            self._refresh_all_tabs()
            messagebox.showinfo("Project", "Project loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_project_as(self):
        p = filedialog.asksaveasfilename(defaultextension=".qproj", filetypes=[("QubitStudio Project", "*.qproj")])
        if not p: return
        try:
            QbtFile.save_project(self.model.qm, p)
            messagebox.showinfo("Project", "Project saved.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _import_qasm(self):
        p = filedialog.askopenfilename(defaultextension=".qasm", filetypes=[("OpenQASM", "*.qasm"),("All","*.*")])
        if not p: return
        text = ""
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read: {e}")
            return
        qc = Exporter.import_qasm2(text)
        if qc is None:
            messagebox.showerror("Import", "OpenQASM parsing not available in this environment/Qiskit version.")
            return
        if Statevector is None:
            messagebox.showerror("Import", "Statevector tools unavailable.")
            return
        sv = Statevector.from_instruction(qc).data
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            self.model.qm.add_hybrid(name, len(sv), 0, 1)
        except Exception:
            pass
        qh = self.model.qm.qubits.get(name)
        if isinstance(qh, HybridQubit):
            try:
                qh.initialize(sv)
            except Exception:
                pass
        self._refresh_resources()
        self._refresh_all_tabs()
        messagebox.showinfo("Import", "Imported QASM into a HybridQubit.")

    def _export_notebook(self):
        p = filedialog.asksaveasfilename(defaultextension=".ipynb", filetypes=[("Jupyter Notebook", "*.ipynb")])
        if not p: return
        circs = self.model.get_circuits()
        if not circs:
            messagebox.showinfo("Export", "No circuits to export.")
            return
        ok = Exporter.export_notebook(p, "Qubit Studio Export", circs, notes="Generated from Qubit Studio.")
        if ok: messagebox.showinfo("Export", "Notebook saved.")
        else:  messagebox.showerror("Export", "nbformat not available. Install it to enable notebook export.")

    def _export_selected_qasm(self):
        obj = self.model.selected
        if obj is None:
            messagebox.showinfo("Export", "No selection.")
            return
        self._export_qasm_for_qubit(obj)
    
    def _add_qubit_popup(self):
        dlg = QubitDialog(self.root, "Add Hybrid Qubit")
        if getattr(dlg, "result", None):
            nm, d = dlg.result
            try:
                self.model.qm.add_hybrid(nm, d)
                self._refresh_resources(); self._refresh_all_tabs()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _add_cubit_popup(self):
        dlg = CubitDialog(self.root, "Add Cubit")
        if getattr(dlg, "result", None):
            nm, d = dlg.result
            try:
                self.model.qm.add_cubit(nm, d)
                self._refresh_resources(); self._refresh_all_tabs()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _add_channel_popup(self):
        dlg = ChannelDialog(self.root, "Add Channel")
        if getattr(dlg, "result", None):
            nm, n = dlg.result
            try:
                self.model.qm.add_channel(nm, n)
                self._refresh_resources(); self._refresh_all_tabs()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _about(self):

        messagebox.showinfo("About", "Qubit Studio — Tkinter + Qiskit educational app.\nIBM Quantum integration (best effort until qiskit is fixed -.-). R&D BioTech Alaska")

    def _on_task_done(self, event: str, payload: dict):
        self.root.after(0, self._on_task_done_main, event, payload)

    def _on_task_done_main(self, event: str, payload: dict):
        self.progress.stop()
        if event == "task_done":
            name = payload.get("name")
            ok = payload.get("ok", False)
            if name == "ibm_connect":
                if ok and self.model.ibm.connected:
                    self.status_var.set("Connected to IBM Quantum.")
                else:
                    self.status_var.set(f"IBM Connect failed: {self.model.ibm.api_error}")
                self._refresh_all_tabs()
            elif name == "ibm_run":
                if ok:
                    info = payload.get("result", {})
                    counts_list = info.get("counts")
                    if counts_list:
                        try:
                            if MATPLOTLIB_AVAILABLE:
                                fig, ax = plt.subplots(figsize=(5,2), dpi=100)
                                from collections import Counter
                                c = Counter()
                                for d in counts_list:
                                    if isinstance(d, dict):
                                        c.update(d)
                                if c:
                                    keys = list(c.keys()); vals = [c[k] for k in keys]
                                    ax.bar(range(len(keys)), vals)
                                    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha="right")
                                    ax.set_title(f"Results on {info.get('backend','backend')}")
                                    fig.tight_layout()
                                    win = tk.Toplevel(self.root)
                                    win.title("Execution Results")
                                    canvas = FigureCanvasTkAgg(fig, master=win)
                                    canvas.draw()
                                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
                                    plt.close(fig)
                                    self.status_var.set("Job finished; results displayed.")
                                else:
                                    messagebox.showinfo("Run", "Job finished but no counts available.")
                            else:
                                messagebox.showinfo("Run", "Job finished. (Install matplotlib for charts.)")
                        except Exception:
                            messagebox.showinfo("Run", "Job finished.")
                    else:
                        err = info.get("error")
                        if err:
                            messagebox.showerror("Run", f"Job finished with error: {err}")
                        else:
                            messagebox.showinfo("Run", "Job finished.")
                else:
                    messagebox.showerror("Run", f"Execution failed: {payload.get('error')}")
            else:
                self.status_var.set("Done.")
        else:
            self.status_var.set("Done.")

def main():
    root = tk.Tk()
    app = QubitStudioApp(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

if __name__ == "__main__":
    main()
