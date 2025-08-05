import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import json

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class Cubit:
    def __init__(self, name, dimension=2, logical_zero_idx=0, logical_one_idx=1):
        self.name = name
        self.dimension = dimension
        self.logical_zero_idx = logical_zero_idx
        self.logical_one_idx = logical_one_idx
        self.state = np.zeros(self.dimension, dtype=np.complex128)
        self.state[self.logical_zero_idx] = 1.0
        self.T1 = None
        self.T2 = None
        self.gate_errors = {}

    def reset_to_logical_zero(self):
        self.state[:] = 0
        self.state[self.logical_zero_idx] = 1.0

    def reset_to_logical_one(self):
        self.state[:] = 0
        self.state[self.logical_one_idx] = 1.0

    def reset_to_basis(self, idx):
        self.state[:] = 0
        self.state[idx] = 1.0

    def initialize(self, amplitudes):
        self.state = np.array(amplitudes, dtype=np.complex128)
        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1.0):
            raise ValueError("State must be normalized")

    def normalize(self):
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

    def get_statevector(self):
        return np.copy(self.state)

    def set_T1(self, val):     self.T1 = val
    def set_T2(self, val):     self.T2 = val
    def set_gate_error(self, g, v):  self.gate_errors[g] = v

    def apply_logical_x(self):
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = b, a

    def apply_logical_z(self):
        self.state[self.logical_one_idx] *= -1

    def apply_logical_hadamard(self):
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        c = (a + b)/np.sqrt(2)
        d = (a - b)/np.sqrt(2)
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = c, d

    def apply_logical_rx(self, theta):
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        m, n = np.cos(theta/2), -1j*np.sin(theta/2)
        U = np.array([[m, n], [n, m]], dtype=complex)
        new = U @ np.array([a, b])
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = new

    def apply_logical_ry(self, theta):
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        m, n = np.cos(theta/2), np.sin(theta/2)
        U = np.array([[m, -n], [n, m]], dtype=complex)
        new = U @ np.array([a, b])
        self.state[self.logical_zero_idx], self.state[self.logical_one_idx] = new

    def apply_logical_rz(self, theta):
        a, b = self.state[self.logical_zero_idx], self.state[self.logical_one_idx]
        self.state[self.logical_zero_idx] = a * np.exp(-1j*theta/2)
        self.state[self.logical_one_idx]  = b * np.exp(1j*theta/2)

    def apply_logical_s(self):
        self.state[self.logical_one_idx] *= 1j

    def apply_logical_t(self):
        self.state[self.logical_one_idx] *= np.exp(1j*np.pi/4)

    def apply_amplitude_damping(self, dt):
        if self.T1 is None:
            raise ValueError("T1 not set")
        gamma = dt/self.T1
        decay = np.exp(-gamma)
        pop1 = abs(self.state[self.logical_one_idx])**2
        self.state[self.logical_one_idx] *= decay
        self.state[self.logical_zero_idx] += np.sqrt(1 - decay**2) * np.sqrt(pop1)
        self.normalize()

    def apply_phase_damping(self, dt):
        if self.T2 is None:
            raise ValueError("T2 not set")
        gamma = dt/self.T2
        decay = np.exp(-gamma)
        self.state *= decay
        self.normalize()

    def apply_depolarizing(self, p):
        if not (0 <= p <= 1):
            raise ValueError("p out of range")
        uniform = np.ones(self.dimension, dtype=complex)/np.sqrt(self.dimension)
        self.state = np.sqrt(1-p)*self.state + np.sqrt(p)*uniform
        self.normalize()

    def apply_phase_noise(self, gamma):
        for i in range(self.dimension):
            if i not in (self.logical_zero_idx, self.logical_one_idx):
                phi = np.random.uniform(-gamma, gamma)
                self.state[i] *= np.exp(1j*phi)

    def measure_logical(self):
        p0 = abs(self.state[self.logical_zero_idx])**2
        p1 = abs(self.state[self.logical_one_idx])**2
        r  = np.random.rand()
        if r < p0/(p0 + p1):
            self.reset_to_logical_zero()
            return 0
        else:
            self.reset_to_logical_one()
            return 1

    def measure_full(self):
        probs = abs(self.state)**2
        idx   = np.random.choice(self.dimension, p=probs/np.sum(probs))
        self.reset_to_basis(idx)
        return idx

    def get_hidden_population_sum(self):
        return sum(abs(self.state[i])**2 for i in range(self.dimension)
                   if i not in (self.logical_zero_idx, self.logical_one_idx))


class HybridQubit:
    def __init__(self, name, dimension=2, logical_zero_idx=0, logical_one_idx=1, backend=None):
        self.name = name
        self.dimension = dimension
        self.logical_zero_idx = logical_zero_idx
        self.logical_one_idx  = logical_one_idx
        self.num_qubits = int(np.ceil(np.log2(self.dimension)))
        self.backend = backend or AerSimulator()
        self.circuit = QuantumCircuit(self.num_qubits)
        init = np.zeros(2**self.num_qubits, dtype=complex)
        init[self.logical_zero_idx] = 1.0
        self.circuit.initialize(init, range(self.num_qubits))

    def reset_to_logical_zero(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        init = np.zeros(2**self.num_qubits, dtype=complex)
        init[self.logical_zero_idx] = 1.0
        self.circuit.initialize(init, range(self.num_qubits))

    def reset_to_logical_one(self):
        self.circuit = QuantumCircuit(self.num_qubits)
        init = np.zeros(2**self.num_qubits, dtype=complex)
        init[self.logical_one_idx] = 1.0
        self.circuit.initialize(init, range(self.num_qubits))

    def reset_to_basis(self, idx):
        self.circuit = QuantumCircuit(self.num_qubits)
        init = np.zeros(2**self.num_qubits, dtype=complex)
        init[idx] = 1.0
        self.circuit.initialize(init, range(self.num_qubits))

    def initialize(self, amplitudes):
        self.circuit = QuantumCircuit(self.num_qubits)
        amps = np.array(amplitudes, dtype=complex)
        norm = np.linalg.norm(amps)
        if not np.isclose(norm, 1.0):
            raise ValueError("State must be normalized")
        self.circuit.initialize(amps, range(self.num_qubits))

    def get_statevector(self):
        sv = Statevector.from_instruction(self.circuit)
        return np.array(sv.data)

    def apply_logical_x(self):
        sv = self.get_statevector()
        a, b = sv[self.logical_zero_idx], sv[self.logical_one_idx]
        sv[self.logical_zero_idx], sv[self.logical_one_idx] = b, a
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def apply_logical_z(self):
        sv = self.get_statevector()
        sv[self.logical_one_idx] *= -1
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def apply_logical_hadamard(self):
        sv = self.get_statevector()
        a, b = sv[self.logical_zero_idx], sv[self.logical_one_idx]
        c = (a + b)/np.sqrt(2)
        d = (a - b)/np.sqrt(2)
        sv[self.logical_zero_idx], sv[self.logical_one_idx] = c, d
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def apply_logical_rx(self, theta):
        sv = self.get_statevector()
        a, b = sv[self.logical_zero_idx], sv[self.logical_one_idx]
        m, n = np.cos(theta/2), -1j*np.sin(theta/2)
        U = np.array([[m, n], [n, m]], dtype=complex)
        new = U @ np.array([a, b])
        sv[self.logical_zero_idx], sv[self.logical_one_idx] = new
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def apply_logical_ry(self, theta):
        sv = self.get_statevector()
        a, b = sv[self.logical_zero_idx], sv[self.logical_one_idx]
        m, n = np.cos(theta/2), np.sin(theta/2)
        U = np.array([[m, -n], [n, m]], dtype=complex)
        new = U @ np.array([a, b])
        sv[self.logical_zero_idx], sv[self.logical_one_idx] = new
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def apply_logical_rz(self, theta):
        sv = self.get_statevector()
        a, b = sv[self.logical_zero_idx], sv[self.logical_one_idx]
        sv[self.logical_zero_idx] = a * np.exp(-1j*theta/2)
        sv[self.logical_one_idx]  = b * np.exp(1j*theta/2)
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def apply_logical_s(self):
        sv = self.get_statevector()
        sv[self.logical_one_idx] *= 1j
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def apply_logical_t(self):
        sv = self.get_statevector()
        sv[self.logical_one_idx] *= np.exp(1j*np.pi/4)
        qc = QuantumCircuit(self.num_qubits)
        qc.initialize(sv, range(self.num_qubits))
        self.circuit = qc

    def measure_logical(self):
        sv = self.get_statevector()
        p0 = abs(sv[self.logical_zero_idx])**2
        p1 = abs(sv[self.logical_one_idx])**2
        r  = np.random.rand()
        if r < p0/(p0+p1):
            self.reset_to_logical_zero()
            return 0
        else:
            self.reset_to_logical_one()
            return 1

    def measure_full(self):
        sv = self.get_statevector()
        probs = abs(sv)**2
        idx   = np.random.choice(len(sv), p=probs/np.sum(probs))
        self.reset_to_basis(idx)
        return idx

    def get_hidden_population_sum(self):
        sv = self.get_statevector()
        return sum(abs(sv[i])**2 for i in range(len(sv))
                   if i not in (self.logical_zero_idx, self.logical_one_idx))


class QuantumChannel:
    def __init__(self, name, num_subbits):
        self.name = name
        self.subbits = [Cubit(f"{name}_s{i+1}") for i in range(num_subbits)]

    def encode(self, thetas, phis=None):
        if phis is None:
            phis = [0]*len(thetas)
        for s, t, p in zip(self.subbits, thetas, phis):
            s.reset_to_logical_zero()
            s.apply_logical_ry(t)
            s.apply_logical_rz(p)

    def decode(self):
        out = []
        for s in self.subbits:
            a = s.state[s.logical_zero_idx]
            b = s.state[s.logical_one_idx]
            norm = np.sqrt(abs(a)**2 + abs(b)**2)
            a, b = a/norm, b/norm
            theta = 2*np.arccos(abs(a))
            phi   = np.angle(b)
            out.append((theta, phi))
        return out


class QuantumManager:
    def __init__(self):
        self.qubits   = {}
        self.channels = {}

    def add_cubit(self, name, dimension=2, lz=0, lo=1):
        if name in self.qubits:
            raise ValueError("Duplicate qubit name")
        self.qubits[name] = Cubit(name, dimension, lz, lo)

    def add_hybrid(self, name, dimension=2, lz=0, lo=1, backend=None):
        if name in self.qubits:
            raise ValueError("Duplicate qubit name")
        self.qubits[name] = HybridQubit(name, dimension, lz, lo, backend)

    def add_channel(self, name, num_subbits):
        if name in self.channels:
            raise ValueError("Duplicate channel name")
        self.channels[name] = QuantumChannel(name, num_subbits)

    def remove(self, name):
        self.qubits.pop(name, None)
        self.channels.pop(name, None)


class QbtFile:
    @staticmethod
    def save(obj, path):
        if isinstance(obj, Cubit):
            d = {
                "type": "Cubit",
                "name": obj.name,
                "dimension": obj.dimension,
                "logical_zero_idx": obj.logical_zero_idx,
                "logical_one_idx": obj.logical_one_idx,
                "statevector": [[float(np.real(a)), float(np.imag(a))] for a in obj.state],
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
                "statevector": [[float(np.real(a)), float(np.imag(a))] for a in sv]
            }
        else:
            raise ValueError("Can only save Cubit or HybridQubit")
        with open(path, "w") as f:
            json.dump(d, f)

    @staticmethod
    def load(path):
        with open(path, "r") as f:
            d = json.load(f)
        t = d["type"]
        if t == "Cubit":
            c = Cubit(d["name"], d["dimension"], d["logical_zero_idx"], d["logical_one_idx"])
            amps = [complex(r, i) for r, i in d["statevector"]]
            c.initialize(amps)
            c.T1 = d.get("T1")
            c.T2 = d.get("T2")
            c.gate_errors = d.get("gate_errors", {})
            return c
        elif t == "HybridQubit":
            h = HybridQubit(d["name"], d["dimension"], d["logical_zero_idx"], d["logical_one_idx"])
            amps = [complex(r, i) for r, i in d["statevector"]]
            h.initialize(amps)
            return h
        else:
            raise ValueError("Unsupported QBT file type")


class IBMQManager:
    def __init__(self):
        self.connected = False
        self.provider  = None
        self.backends  = []
        self.api_error = None

    def connect(self, token):
        try:
            from qiskit_ibm_provider import IBMProvider
            prov = IBMProvider(token=token)
            self.provider = prov
            self.backends = list(prov.backends())
            self.connected = True
            self.api_error = None
        except ImportError:
            self.api_error = (
                "IBMProvider not compatible with Qiskit 2.x\n"
                "See https://quantum.cloud.ibm.com/docs/en/guides/latest-updates"
            )
            self.connected = False
        except Exception as e:
            self.api_error = str(e)
            self.connected = False

    def get_backends(self):
        return self.backends

    def get_backend(self, name):
        for b in self.backends:
            if b.name == name:
                return b
        return None


class StudioWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Qubit Studio")
        self.root.geometry("1200x700")

        self.qm  = QuantumManager()
        self.ibm = IBMQManager()
        self.selected_obj = None

        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.style.configure('QubitCard.TFrame',   background='#eaf1fc', relief='raised', borderwidth=2)
        self.style.configure('CubitCard.TFrame',   background='#eafde8', relief='raised', borderwidth=2)
        self.style.configure('ChannelCard.TFrame', background='#f5e9ff', relief='raised', borderwidth=2)
        self.style.configure('Studio.TLabelframe', font=("Segoe UI", 11, "bold"))

        self.make_layout()
        self.refresh_resource_panel()
        self.refresh_all_tabs()

    def make_layout(self):
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=1)

        self.left = ttk.Frame(pane, width=260)
        self.left.pack_propagate(0)
        pane.add(self.left)

        btns = ttk.Frame(self.left)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Add Qubit",   command=self.add_qubit_popup).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Add Cubit",   command=self.add_cubit_popup).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Add Channel", command=self.add_channel_popup).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Save",        command=self.save_selected_popup).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Load",        command=self.load_popup).pack(side=tk.LEFT, padx=2)
        ttk.Separator(self.left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=3)

        self.res_canvas = tk.Canvas(self.left, borderwidth=0, background="#f0f0f0")
        self.res_frame  = ttk.Frame(self.res_canvas)
        self.res_scroll = ttk.Scrollbar(self.left, orient="vertical", command=self.res_canvas.yview)
        self.res_canvas.configure(yscrollcommand=self.res_scroll.set)
        self.res_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.res_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.res_canvas.create_window((0,0), window=self.res_frame, anchor='nw')
        self.res_frame.bind("<Configure>", lambda e: self.res_canvas.configure(scrollregion=self.res_canvas.bbox("all")))


        self.ws = ttk.Notebook(pane)
        pane.add(self.ws)
        self.tab_o = ttk.Frame(self.ws)
        self.tab_q = ttk.Frame(self.ws)
        self.tab_c = ttk.Frame(self.ws)
        self.tab_i = ttk.Frame(self.ws)
        self.ws.add(self.tab_o, text="Overview")
        self.ws.add(self.tab_q, text="Qubit Explorer")
        self.ws.add(self.tab_c, text="Channel Ops")
        self.ws.add(self.tab_i, text="IBM Cloud")
        self.ws.bind("<<NotebookTabChanged>>", lambda e: self.refresh_current_tab())

    def refresh_resource_panel(self):
        for w in self.res_frame.winfo_children():
            w.destroy()

        if not self.qm.qubits and not self.qm.channels:
            ttk.Label(self.res_frame,
                      text="No resources.\nUse the add buttons above.",
                      justify=tk.CENTER, foreground="#888").pack(pady=20)
            return

        r = 0
        for nm in sorted(self.qm.qubits):
            qb = self.qm.qubits[nm]
            frm = ttk.Frame(self.res_frame,
                            style="QubitCard.TFrame" if isinstance(qb, HybridQubit) else "CubitCard.TFrame",
                            relief=tk.RAISED, borderwidth=2)
            frm.grid(row=r, column=0, sticky="ew", pady=3, padx=4)
            frm.columnconfigure(0, weight=1)
            tp = "Qubit" if isinstance(qb, HybridQubit) else "Cubit"
            ttk.Label(frm, text=f"{tp}: {qb.name}", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            sv = qb.get_statevector()
            summary = ", ".join(f"{np.real(a):.2f}{np.imag(a):+.2f}j" for a in sv)
            ttk.Label(frm, text=f"State: [{summary}]", font=("Consolas", 9)).pack(anchor="w")
            ttk.Button(frm, text="Remove", command=lambda n=nm: self.remove_and_refresh(n)).pack(side=tk.RIGHT, padx=2, pady=2)
            r += 1

        for nm in sorted(self.qm.channels):
            ch = self.qm.channels[nm]
            frm = ttk.Frame(self.res_frame, style="ChannelCard.TFrame", relief=tk.RAISED, borderwidth=2)
            frm.grid(row=r, column=0, sticky="ew", pady=3, padx=4)
            frm.columnconfigure(0, weight=1)
            ttk.Label(frm, text=f"Channel: {ch.name}", font=("Segoe UI", 10, "bold"), foreground="#800080").pack(anchor="w")
            ttk.Label(frm, text=f"Subbits: {len(ch.subbits)}", font=("Segoe UI", 9)).pack(anchor="w")
            ttk.Button(frm, text="Remove", command=lambda n=nm: self.remove_and_refresh(n)).pack(side=tk.RIGHT, padx=2, pady=2)
            r += 1

    def remove_and_refresh(self, nm):
        self.qm.remove(nm)
        self.refresh_resource_panel()
        self.refresh_all_tabs()

    def add_qubit_popup(self):
        nm = simpledialog.askstring("Add Hybrid Qubit", "Qubit Name:")
        d  = simpledialog.askinteger("Dimension", "Dimension:", initialvalue=2)
        if nm and d:
            try:
                self.qm.add_hybrid(nm, d)
                self.refresh_resource_panel()
                self.refresh_all_tabs()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def add_cubit_popup(self):
        nm = simpledialog.askstring("Add Cubit", "Cubit Name:")
        d  = simpledialog.askinteger("Dimension", "Dimension:", initialvalue=2)
        if nm and d:
            try:
                self.qm.add_cubit(nm, d)
                self.refresh_resource_panel()
                self.refresh_all_tabs()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def add_channel_popup(self):
        nm = simpledialog.askstring("Add Channel", "Channel Name:")
        n  = simpledialog.askinteger("Subbits", "Number:", initialvalue=2)
        if nm and n:
            try:
                self.qm.add_channel(nm, n)
                self.refresh_resource_panel()
                self.refresh_all_tabs()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def save_selected_popup(self):
        if not self.selected_obj:
            messagebox.showinfo("Save", "No object selected.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".qbt",
                                         filetypes=[("Qubit Files", "*.qbt")])
        if p:
            QbtFile.save(self.selected_obj, p)
            messagebox.showinfo("Saved", "Saved successfully.")

    def load_popup(self):
        p = filedialog.askopenfilename(defaultextension=".qbt",
                                       filetypes=[("Qubit Files", "*.qbt")])
        if p:
            obj = QbtFile.load(p)
            self.qm.qubits[obj.name] = obj
            self.refresh_resource_panel()
            self.refresh_all_tabs()
            messagebox.showinfo("Loaded", f"Loaded '{obj.name}'.")

    def refresh_all_tabs(self):
        self.show_dashboard()
        self.show_qubits_tab()
        self.show_channels_tab()
        self.show_ibm_cloud_panel()

    def refresh_current_tab(self):
        i = self.ws.index("current")
        if   i == 0: self.show_dashboard()
        elif i == 1: self.show_qubits_tab()
        elif i == 2: self.show_channels_tab()
        elif i == 3: self.show_ibm_cloud_panel()

    def show_dashboard(self):
        for w in self.tab_o.winfo_children():
            w.destroy()
        ttk.Label(self.tab_o, text="Qubit Studio — Dashboard", font=("Segoe UI", 16, "bold")).pack(pady=14)
        msg = (
            f"Qubits/Cubits: {len(self.qm.qubits)}\n"
            f"Channels:      {len(self.qm.channels)}\n\n"
            "Use the Resource Panel (left) to add or load Cubits, Qubits, and Channels.\n"
            "All qubits and channels appear in their dedicated tabs below."
        )
        ttk.Label(self.tab_o, text=msg, font=("Segoe UI", 12), justify=tk.LEFT, wraplength=800).pack(pady=12)

    def show_qubits_tab(self):
        for w in self.tab_q.winfo_children():
            w.destroy()
        if not self.qm.qubits:
            ttk.Label(self.tab_q, text="No qubits/cubits created yet.", font=("Segoe UI", 12)).pack(pady=24)
            return

        for nm in sorted(self.qm.qubits):
            qb = self.qm.qubits[nm]
            card = ttk.Frame(self.tab_q,
                             style="QubitCard.TFrame" if isinstance(qb, HybridQubit) else "CubitCard.TFrame",
                             relief=tk.RAISED, borderwidth=2)
            card.pack(fill=tk.X, padx=14, pady=8)

            tp = "Qubit" if isinstance(qb, HybridQubit) else "Cubit"
            ttk.Label(card, text=f"{tp}: {qb.name}", font=("Segoe UI", 12, "bold")).pack(anchor="w")

            self.show_statevector_plot(card, qb.get_statevector())

            stats = self._state_stats(qb)
            ttk.Label(card, text=stats, font=("Consolas", 10)).pack(anchor="w", pady=1)

            self.build_qubit_operations(card, qb)

            ttk.Button(card, text="Save", command=lambda q=qb: self.save_object(q)).pack(side=tk.RIGHT, padx=4, pady=6)
            self.selected_obj = qb

    def save_object(self, obj):
        p = filedialog.asksaveasfilename(defaultextension=".qbt",
                                         filetypes=[("Qubit Files", "*.qbt")])
        if p:
            QbtFile.save(obj, p)
            messagebox.showinfo("Saved", "Saved successfully.")

    def show_channels_tab(self):
        for w in self.tab_c.winfo_children():
            w.destroy()
        if not self.qm.channels:
            ttk.Label(self.tab_c, text="No channels created yet.", font=("Segoe UI", 12)).pack(pady=24)
            return

        for nm in sorted(self.qm.channels):
            ch = self.qm.channels[nm]
            card = ttk.Frame(self.tab_c, style="ChannelCard.TFrame", relief=tk.RAISED, borderwidth=2)
            card.pack(fill=tk.X, padx=14, pady=10)

            ttk.Label(card, text=f"Channel: {ch.name}",
                      font=("Segoe UI", 12, "bold"), foreground="#800080").pack(anchor="w")
            ttk.Label(card, text=f"Subbits: {len(ch.subbits)}",
                      font=("Segoe UI", 10)).pack(anchor="w")

            state_text = ""
            for i, sub in enumerate(ch.subbits):
                sv = sub.get_statevector()
                state_text += (f"Subbit {i+1}: "
                               f"[{np.real(sv[0]):.3f}{np.imag(sv[0]):+.3f}j, "
                               f"{np.real(sv[1]):.3f}{np.imag(sv[1]):+.3f}j]\n")
            ttk.Label(card, text=state_text, font=("Consolas", 9),
                      background="#f5e9ff").pack(pady=3)

            ctr = ttk.Frame(card)
            ctr.pack(pady=6)
            ttk.Button(ctr, text="Encode",
                       command=lambda c=ch: self.encode_channel(c)).pack(side=tk.LEFT, padx=6)
            ttk.Button(ctr, text="Decode",
                       command=lambda c=ch: self.decode_channel(c)).pack(side=tk.LEFT, padx=6)

            ttk.Button(card, text="Save",
                       command=lambda c=ch: self.save_object(c)).pack(side=tk.RIGHT, padx=4, pady=6)
            self.selected_obj = ch

    def show_ibm_cloud_panel(self):
        for w in self.tab_i.winfo_children():
            w.destroy()
        ttk.Label(self.tab_i, text="IBM Quantum Cloud",
                  font=("Segoe UI", 16, "bold"), foreground="#0055b8").pack(pady=8)

        if self.ibm.connected:
            status = "Connected ✓"
        elif self.ibm.api_error:
            status = f"Not Connected\n\nError: {self.ibm.api_error}"
        else:
            status = "Not Connected"

        ttk.Label(self.tab_i, text=f"Status: {status}", font=("Segoe UI", 12)).pack(pady=6)
        ttk.Button(self.tab_i, text="Connect/Retry", command=self.connect_ibm_popup).pack(pady=4)

        if self.ibm.connected:
            ttk.Label(self.tab_i, text="Available backends:", font=("Segoe UI", 11, "bold")).pack()
            for b in self.ibm.get_backends():
                ttk.Label(self.tab_i, text=b.name, font=("Consolas", 10)).pack(anchor="w")
        else:
            ttk.Label(self.tab_i,
                      text=("IBMProvider not compatible with Qiskit 2.x\n"
                            "See https://quantum.cloud.ibm.com/docs/en/guides/latest-updates"),
                      foreground="#c00").pack(pady=12)

    def connect_ibm_popup(self):
        tok = simpledialog.askstring("IBM Quantum Token", "Enter IBM Quantum API token:")
        if tok:
            self.ibm.connect(tok)
            self.show_ibm_cloud_panel()

    def show_statevector_plot(self, parent, sv):
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(parent,
                      text="matplotlib not available — install for live plots!",
                      foreground="#c00").pack()
            return
        fig, ax = plt.subplots(figsize=(4.5, 1.7), dpi=95)
        prob = abs(sv)**2
        bars = ax.bar(range(len(sv)), prob, color="#4477ee")
        ax.set_ylim(0, 1)
        ax.set_ylabel("|a|²")
        ax.set_xlabel("State index")
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{prob[i]:.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.X)
        plt.close(fig)

    def _state_stats(self, qb):
        sv = qb.get_statevector()
        p0 = abs(sv[qb.logical_zero_idx])**2
        p1 = abs(sv[qb.logical_one_idx])**2
        hid = qb.get_hidden_population_sum()
        return f"|0>={p0:.3f} |1>={p1:.3f} Hidden={hid:.3f}"

    def build_qubit_operations(self, frame, qb):
        g1 = ttk.Frame(frame); g1.pack(pady=2, fill=tk.X)
        ttk.Button(g1, text="X",  command=lambda:self._op(qb,"x")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g1, text="Z",  command=lambda:self._op(qb,"z")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g1, text="H",  command=lambda:self._op(qb,"h")).pack(side=tk.LEFT, padx=3)  
        ttk.Button(g1, text="S",  command=lambda:self._op(qb,"s")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g1, text="T",  command=lambda:self._op(qb,"t")).pack(side=tk.LEFT, padx=3)
        g2 = ttk.Frame(frame); g2.pack(pady=2, fill=tk.X)
        ttk.Button(g2, text="Rx", command=lambda:self._op_angle(qb,"rx")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g2, text="Ry", command=lambda:self._op_angle(qb,"ry")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g2, text="Rz", command=lambda:self._op_angle(qb,"rz")).pack(side=tk.LEFT, padx=3)
        g3 = ttk.Frame(frame); g3.pack(pady=2, fill=tk.X)
        ttk.Button(g3, text="Reset0", command=lambda:self._op(qb,"reset0")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g3, text="Reset1", command=lambda:self._op(qb,"reset1")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g3, text="Measure Logical", command=lambda:self._measure(qb,"logical")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g3, text="Measure Full",    command=lambda:self._measure(qb,"full")).pack(side=tk.LEFT, padx=3)
        g4 = ttk.Frame(frame); g4.pack(pady=2, fill=tk.X)
        ttk.Button(g4, text="Amp Damping",  command=lambda:self._noise(qb,"amp")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g4, text="Phase Damping",command=lambda:self._noise(qb,"phase")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g4, text="Depolarizing", command=lambda:self._noise(qb,"depol")).pack(side=tk.LEFT, padx=3)
        ttk.Button(g4, text="Phase Noise",  command=lambda:self._noise(qb,"phasen")).pack(side=tk.LEFT, padx=3)

    def _op(self, qb, op):
        try:
            if op=="x": qb.apply_logical_x()
            elif op=="z": qb.apply_logical_z()
            elif op=="h": qb.apply_logical_hadamard()
            elif op=="s": qb.apply_logical_s()
            elif op=="t": qb.apply_logical_t()
            elif op=="reset0": qb.reset_to_logical_zero()
            elif op=="reset1": qb.reset_to_logical_one()
            self.refresh_all_tabs()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _op_angle(self, qb, op):
        th = simpledialog.askfloat("Angle", "Angle (radians):")
        try:
            if op=="rx": qb.apply_logical_rx(th)
            elif op=="ry": qb.apply_logical_ry(th)
            elif op=="rz": qb.apply_logical_rz(th)
            self.refresh_all_tabs()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _measure(self, qb, mode):
        if mode=="logical":
            res = qb.measure_logical()
            messagebox.showinfo("Measurement", f"Logical result: {res}")
        else:
            res = qb.measure_full()
            messagebox.showinfo("Measurement", f"Full basis result: {res}")
        self.refresh_all_tabs()

    def _noise(self, qb, op):
        try:
            if op=="amp":
                dt = simpledialog.askfloat("dt","Time dt:")
                qb.apply_amplitude_damping(dt)
            elif op=="phase":
                dt = simpledialog.askfloat("dt","Time dt:")
                qb.apply_phase_damping(dt)
            elif op=="depol":
                p  = simpledialog.askfloat("p","Probability [0–1]:")
                qb.apply_depolarizing(p)
            elif op=="phasen":
                g  = simpledialog.askfloat("gamma","Gamma:")
                qb.apply_phase_noise(g)
            self.refresh_all_tabs()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def encode_channel(self, ch):
        n = len(ch.subbits)
        ths, phs = [], []
        for i in range(n):
            th = simpledialog.askfloat(f"Subbit {i+1} θ", f"Enter θ for subbit {i+1}:")
            ph = simpledialog.askfloat(f"Subbit {i+1} φ", f"Enter φ for subbit {i+1}:", initialvalue=0)
            ths.append(th); phs.append(ph)
        ch.encode(ths, phs)
        self.refresh_all_tabs()

    def decode_channel(self, ch):
        res = ch.decode()
        txt = "\n".join(f"{i+1}: θ={th:.3f}, φ={ph:.3f}" for i, (th, ph) in enumerate(res))
        messagebox.showinfo("Decoded", txt)
        self.refresh_all_tabs()


def main():
    root = tk.Tk()
    StudioWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
