from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

# -----------------------------
# Settings
# -----------------------------
UNITARY_FILE = "targets/haar_random_q2_seed3047.npy"  # change to your filename
BASIS_GATES = ["rz", "sx", "cx"]
OPTIMIZATION_LEVEL = 3
SEED_TRANSPILER = 1234

# -----------------------------
# Load unitary from same folder as this script
# -----------------------------
script_dir = Path(__file__).resolve().parent
unitary_path = script_dir / UNITARY_FILE

U = np.load(unitary_path)

# -----------------------------
# Basic validation
# -----------------------------
if U.shape != (4, 4):
    raise ValueError(f"Expected a 4x4 unitary for 2 qubits, got shape {U.shape}")

I = np.eye(4, dtype=complex)
if not np.allclose(U.conj().T @ U, I, atol=1e-8):
    raise ValueError("The loaded matrix is not unitary")

# -----------------------------
# Build and transpile circuit
# -----------------------------
qc = QuantumCircuit(2)
qc.unitary(U, [0, 1], label="U")

tqc = transpile(
    qc,
    basis_gates=BASIS_GATES,
    optimization_level=OPTIMIZATION_LEVEL,
    seed_transpiler=SEED_TRANSPILER,
)

# -----------------------------
# Compute fidelity
# -----------------------------
U_compiled = Operator(tqc).data
d = U.shape[0]

# Process fidelity
trace_overlap = np.trace(U.conj().T @ U_compiled)
process_fidelity = (np.abs(trace_overlap) ** 2) / (d**2)

# Average gate fidelity
average_gate_fidelity = (d * process_fidelity + 1) / (d + 1)

# Optional: phase-insensitive matrix error
global_phase = np.angle(trace_overlap)
U_compiled_aligned = U_compiled * np.exp(-1j * global_phase)
frobenius_error = np.linalg.norm(U - U_compiled_aligned, ord="fro")

# -----------------------------
# Report
# -----------------------------
counts = tqc.count_ops()

print("Loaded file           :", unitary_path)
print("Total gates           :", tqc.size())
print("Depth                 :", tqc.depth())
print("CX count              :", counts.get("cx", 0))
print("SX count              :", counts.get("sx", 0))
print("RZ count              :", counts.get("rz", 0))
print()
print("Process fidelity      :", process_fidelity)
print("Average gate fidelity :", average_gate_fidelity)
print("Frobenius error       :", frobenius_error)
print()
print(tqc.draw(output="text"))
