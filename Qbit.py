#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

"""
Qubit - An addon to test Qubits in set simulations for Qelm. (Can be used for other simulations as well).
"""

import sys
import os
import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# Qiskit Imports
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator


class QubitNeuron:

    def __init__(
        self,
        neuron_id: int,
        num_qubits: int = 50,
        sim_method: str = 'statevector',
        angle_l1: float = np.pi / 6,
        angle_l3: float = np.pi / 12,
        shots: int = 1,
        enable_logging: bool = True
    ):
        self.neuron_id = neuron_id
        self.num_qubits = num_qubits
        self.sim_method = sim_method
        self.angle_l1 = angle_l1  # Rotation angle for Layer 1
        self.angle_l3 = angle_l3  # Rotation angle for Layer 3
        self.shots = shots
        self.enable_logging = enable_logging
        self.last_error = None

        # Initialize Quantum and Classical Registers
        self.qr = QuantumRegister(self.num_qubits, f"qr_{neuron_id}")
        self.cr = ClassicalRegister(self.num_qubits, f"cr_{neuron_id}")
        self.circuit = QuantumCircuit(self.qr, self.cr, name=f"Neuron_{neuron_id}")

        # Initialize Simulator Backend
        try:
            self.backend = AerSimulator(method=self.sim_method)
            if self.enable_logging:
                print(f"[Neuron {self.neuron_id}] Using AerSimulator(method='{self.sim_method}').")
        except Exception:
            self.last_error = traceback.format_exc()

        # Build the 3-layer quantum structure
        self._build_3_layers()

    def _build_3_layers(self):

        try:
            half_n = self.num_qubits // 2

            # Layer 1: RY rotations on the first half qubits
            for qubit_idx in range(half_n):
                self.circuit.ry(self.angle_l1, self.qr[qubit_idx])

            # Layer 2: Hadamard gates on all qubits for superposition
            for qubit_idx in range(self.num_qubits):
                self.circuit.h(self.qr[qubit_idx])

            # Layer 3: RX rotations on the second half qubits
            for qubit_idx in range(half_n, self.num_qubits):
                self.circuit.rx(self.angle_l3, self.qr[qubit_idx])

            # Save the statevector for simulation
            self.circuit.save_statevector()

        except Exception:
            self.last_error = traceback.format_exc()

    def run_circuit(self):

        try:
            job = self.backend.run(self.circuit, shots=self.shots)
            result = job.result()
            final_state = result.get_statevector(self.circuit)
            return final_state.data  # Returns a complex numpy array
        except Exception:
            self.last_error = traceback.format_exc()
            return None

    def apply_grover_search(self, target_index: int):

        try:
            if target_index < 0 or target_index >= 2**self.num_qubits:
                raise ValueError(
                    f"[Neuron {self.neuron_id}] target_index {target_index} out of valid range "
                    f"for {2**self.num_qubits} possible states."
                )

            grover_circuit = QuantumCircuit(self.qr, self.cr, name=f"Grover_{self.neuron_id}")

            # Minimal Oracle (phase flip for the target state)
            grover_circuit.x(self.qr[0])
            grover_circuit.cz(self.qr[0], self.qr[1])
            grover_circuit.x(self.qr[0])

            # Minimal Diffusion Operator
            for idx in range(self.num_qubits):
                grover_circuit.h(self.qr[idx])
                grover_circuit.x(self.qr[idx])
            grover_circuit.h(self.qr[0])
            grover_circuit.cx(self.qr[1], self.qr[0])
            grover_circuit.h(self.qr[0])
            for idx in range(self.num_qubits):
                grover_circuit.x(self.qr[idx])
                grover_circuit.h(self.qr[idx])

            # Merge Grover's circuit with the neuron's existing circuit
            self.circuit = self.circuit.compose(grover_circuit)
            self.circuit.save_statevector()

        except Exception:
            self.last_error = traceback.format_exc()


class QuantumBrain:


    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.neurons = []
        self.last_error = None
        if self.enable_logging:
            print("[QuantumBrain] Initialized empty brain structure.")

    def create_neuron(self, **kwargs):

        try:
            neuron_id = len(self.neurons) + 1
            new_neuron = QubitNeuron(neuron_id=neuron_id, enable_logging=self.enable_logging, **kwargs)
            self.neurons.append(new_neuron)
            if self.enable_logging:
                print(f"[QuantumBrain] Neuron {neuron_id} created.")
            return new_neuron
        except Exception:
            self.last_error = traceback.format_exc()
            return None

    def delete_neuron(self, neuron_id: int):

        found = False
        for i, neuron in enumerate(self.neurons):
            if neuron.neuron_id == neuron_id:
                self.neurons.pop(i)
                found = True
                if self.enable_logging:
                    print(f"[QuantumBrain] Neuron {neuron_id} deleted.")
                break
        if not found:
            self.last_error = f"[QuantumBrain] Neuron ID {neuron_id} not found."

    def run_all_neurons(self):

        outputs = {}
        for neuron in self.neurons:
            sv = neuron.run_circuit()
            outputs[neuron.neuron_id] = sv
        return outputs

    def apply_grover_global(self, target_idx: int):

        try:
            for neuron in self.neurons:
                neuron.apply_grover_search(target_idx)
        except Exception:
            self.last_error = traceback.format_exc()

    def apply_grover_to_neuron(self, neuron_id: int, target_idx: int):

        found = False
        for neuron in self.neurons:
            if neuron.neuron_id == neuron_id:
                neuron.apply_grover_search(target_idx)
                found = True
                break
        if not found:
            self.last_error = f"[QuantumBrain] Neuron ID {neuron_id} not found."


class QubitCreatorGUI:

    def __init__(self, master):
        self.master = master
        self.master.title("Qubit Creator Extended - Quantum Brain")
        self.master.geometry("1600x900")
        self.master.resizable(False, False)

        self.brain = QuantumBrain(enable_logging=True)

        # Default UI variables
        self.num_qubits_var = tk.IntVar(value=50)
        self.sim_method_var = tk.StringVar(value='statevector')
        self.shots_var = tk.IntVar(value=1)
        self.angle_l1_var = tk.DoubleVar(value=np.pi / 6)
        self.angle_l3_var = tk.DoubleVar(value=np.pi / 12)

        self.target_index_var = tk.StringVar(value="0")
        self.delete_neuron_id_var = tk.StringVar(value="1")
        self.grover_neuron_id_var = tk.StringVar(value="1")

        self.build_gui()

    def build_gui(self):

        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel: Configuration and Actions
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side="left", fill="y", padx=5, pady=5)

        # Right panel: Logs and Outputs
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Configuration Frame
        config_frame = ttk.LabelFrame(left_panel, text="Neuron Configuration")
        config_frame.pack(fill="x", padx=5, pady=5)

        # Number of Qubits
        ttk.Label(config_frame, text="Number of Qubits:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        qspin = ttk.Spinbox(config_frame, from_=1, to=64, textvariable=self.num_qubits_var, width=10)
        qspin.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        # Simulation Method
        ttk.Label(config_frame, text="Simulation Method:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.sim_method_combo = ttk.Combobox(
            config_frame,
            textvariable=self.sim_method_var,
            values=('statevector', 'matrix_product_state', 'density_matrix', 'stabilizer', 'extended_stabilizer'),
            state='readonly'
        )
        self.sim_method_combo.grid(row=1, column=1, sticky='w', padx=5, pady=5)

        # Shots
        ttk.Label(config_frame, text="Shots:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        shotspin = ttk.Spinbox(config_frame, from_=1, to=100000, textvariable=self.shots_var, width=10)
        shotspin.grid(row=2, column=1, sticky='w', padx=5, pady=5)

        # Layer 1 Angle
        ttk.Label(config_frame, text="Layer1 RY Angle (rad):").grid(row=3, column=0, sticky='e', padx=5, pady=5)
        self.angle_l1_entry = ttk.Entry(config_frame, textvariable=self.angle_l1_var, width=12)
        self.angle_l1_entry.grid(row=3, column=1, sticky='w', padx=5, pady=5)

        # Layer 3 Angle
        ttk.Label(config_frame, text="Layer3 RX Angle (rad):").grid(row=4, column=0, sticky='e', padx=5, pady=5)
        self.angle_l3_entry = ttk.Entry(config_frame, textvariable=self.angle_l3_var, width=12)
        self.angle_l3_entry.grid(row=4, column=1, sticky='w', padx=5, pady=5)

        # Create Neuron Button
        create_btn = ttk.Button(config_frame, text="Create Neuron", command=self.create_neuron)
        create_btn.grid(row=5, column=0, columnspan=2, pady=10)

        # Deletion Frame
        del_frame = ttk.LabelFrame(left_panel, text="Neuron Deletion")
        del_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(del_frame, text="Delete Neuron ID:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        del_entry = ttk.Entry(del_frame, textvariable=self.delete_neuron_id_var, width=10)
        del_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        del_btn = ttk.Button(del_frame, text="Delete Neuron", command=self.delete_neuron)
        del_btn.grid(row=1, column=0, columnspan=2, pady=5)

        # Grover Frame
        grover_frame = ttk.LabelFrame(left_panel, text="Grover's Search")
        grover_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(grover_frame, text="Target Index:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
        target_entry = ttk.Entry(grover_frame, textvariable=self.target_index_var, width=12)
        target_entry.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        grover_all_btn = ttk.Button(
            grover_frame,
            text="Apply Grover to ALL",
            command=self.grover_all
        )
        grover_all_btn.grid(row=1, column=0, columnspan=2, pady=5)

        ttk.Label(grover_frame, text="Single Neuron ID:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        groverid_entry = ttk.Entry(grover_frame, textvariable=self.grover_neuron_id_var, width=10)
        groverid_entry.grid(row=2, column=1, sticky='w', padx=5, pady=5)

        grover_one_btn = ttk.Button(
            grover_frame,
            text="Apply Grover to One Neuron",
            command=self.grover_one
        )
        grover_one_btn.grid(row=3, column=0, columnspan=2, pady=5)

        # Run Frame
        run_frame = ttk.LabelFrame(left_panel, text="Simulation")
        run_frame.pack(fill="x", padx=5, pady=5)

        run_btn = ttk.Button(run_frame, text="Run All Neurons", command=self.run_all_neurons)
        run_btn.pack(pady=5, fill="x")

        # Logs / Outputs
        log_frame = ttk.LabelFrame(right_panel, text="Logs and Outputs")
        log_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.log_area = scrolledtext.ScrolledText(log_frame, wrap='word', state='disabled')
        self.log_area.pack(fill="both", expand=True, padx=5, pady=5)

        # Configure Styles
        self.configure_styles()

        self.log_message("Qubit Creator Extended GUI initialized.\n")

    def configure_styles(self):

        style = ttk.Style()
        style.theme_use('clam')

        # Define custom styles
        style.configure("TFrame", background="#2C3E50")
        style.configure("TLabelFrame", background="#34495E", foreground="white")
        style.configure("TLabel", background="#2C3E50", foreground="white")
        style.configure("TButton", background="#34495E", foreground="white", padding=6, relief="flat")
        style.configure("TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
        style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
        style.map("TButton",
                  foreground=[('active', 'white')],
                  background=[('active', '#1F2A36')])

    def log_message(self, msg: str):

        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, msg)
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def create_neuron(self):

        try:
            num_qubits = self.num_qubits_var.get()
            sim_method = self.sim_method_var.get()
            shots = self.shots_var.get()
            angle_l1 = self.angle_l1_var.get()
            angle_l3 = self.angle_l3_var.get()

            # Validate simulation method for large qubits
            if num_qubits > 30 and sim_method == 'statevector':
                messagebox.showwarning(
                    "High Qubit Count Warning",
                    f"Simulating {num_qubits} qubits with 'statevector' method may require excessive memory.\n"
                    f"Consider using 'matrix_product_state' or reducing the number of qubits."
                )

            neuron = self.brain.create_neuron(
                num_qubits=num_qubits,
                sim_method=sim_method,
                shots=shots,
                angle_l1=angle_l1,
                angle_l3=angle_l3
            )

            if neuron is None:
                err = self.brain.last_error or "[ERROR] Could not create neuron."
                self.log_message(f"{err}\n")
            else:
                self.log_message(f"Created Neuron {neuron.neuron_id} with {num_qubits} qubits, method='{sim_method}'.\n")
        except Exception as e:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")

    def delete_neuron(self):

        try:
            nid = int(self.delete_neuron_id_var.get())
            self.brain.delete_neuron(nid)
            if self.brain.last_error:
                self.log_message(f"{self.brain.last_error}\n")
                self.brain.last_error = None
            else:
                self.log_message(f"Deleted Neuron {nid}.\n")
        except ValueError:
            self.log_message("[ERROR] Invalid Neuron ID for deletion.\n")
        except Exception:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")

    def grover_all(self):

        try:
            t_idx = int(self.target_index_var.get())
            self.brain.apply_grover_global(t_idx)
            if self.brain.last_error:
                self.log_message(f"{self.brain.last_error}\n")
                self.brain.last_error = None
            else:
                self.log_message(f"Applied Grover's search to ALL neurons with target index={t_idx}.\n")
        except ValueError:
            self.log_message("[ERROR] Invalid target index.\n")
        except Exception:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")

    def grover_one(self):

        try:
            t_idx = int(self.target_index_var.get())
            nid = int(self.grover_neuron_id_var.get())
            self.brain.apply_grover_to_neuron(nid, t_idx)
            if self.brain.last_error:
                self.log_message(f"{self.brain.last_error}\n")
                self.brain.last_error = None
            else:
                self.log_message(f"Applied Grover's search to Neuron {nid} with target index={t_idx}.\n")
        except ValueError:
            self.log_message("[ERROR] Invalid Neuron ID or target index.\n")
        except Exception:
            self.log_message(f"[EXCEPTION] {traceback.format_exc()}\n")

    def run_all_neurons(self):

        results = self.brain.run_all_neurons()
        for nid, sv in results.items():
            if sv is None:
                err = self.brain.last_error or f"[ERROR] Running Neuron {nid} returned None."
                self.log_message(f"{err}\n")
            else:
                self.log_message(f"\n--- Neuron {nid} Statevector ---\n")
                if len(sv) > 16:
                    self.log_message(f"Statevector length: {len(sv)}. (Truncated display)\n")
                else:
                    self.log_message(f"{sv}\n")


def main():
    root = tk.Tk()
    app = QubitCreatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
