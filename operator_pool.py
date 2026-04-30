from itertools import permutations
from typing import Iterable, List, Tuple

import numpy as np
from qiskit.circuit.library import CXGate, RXGate, RYGate, RZGate, SXGate

_DEFAULT_ANGLE = np.pi / 4


def get_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Return the 2×2 unitary for RX, RY, or RZ at the given angle."""
    if axis == "RX":
        gate = RXGate(angle)
    elif axis == "RY":
        gate = RYGate(angle)
    elif axis == "RZ":
        gate = RZGate(angle)
    else:
        raise ValueError(f"Unknown rotation axis: {axis!r}")
    return np.array(gate.to_matrix(), dtype=np.complex128)


def get_fixed_gate_matrix(name: str) -> np.ndarray:
    """Return the matrix for a non-parameterized gate: SX or CNOT."""
    if name == "SX":
        return np.array(SXGate().to_matrix(), dtype=np.complex128)
    if name == "CNOT":
        return np.array(CXGate().to_matrix(), dtype=np.complex128)
    raise ValueError(f"Unknown fixed gate: {name!r}")


def _embed_single_qubit(
    gate_2x2: np.ndarray,
    qubit: int,
    num_qubits: int,
) -> np.ndarray:
    """Embed a one-qubit gate using the project's MSB-first qubit convention."""
    factors = [
        gate_2x2 if q == qubit else np.eye(2, dtype=np.complex128)
        for q in range(num_qubits)
    ]
    result = factors[0]
    for f in factors[1:]:
        result = np.kron(result, f)
    return result


def _embed_two_qubit(
    gate_4x4: np.ndarray,
    control: int,
    target: int,
    num_qubits: int,
) -> np.ndarray:
    """Embed a two-qubit gate for arbitrary ordered control and target qubits."""
    d = 2**num_qubits
    full = np.zeros((d, d), dtype=np.complex128)

    for col in range(d):
        bits = [(col >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        two_q_in = (bits[control] << 1) | bits[target]

        for two_q_out in range(4):
            amp = gate_4x4[two_q_out, two_q_in]
            if abs(amp) < 1e-15:
                continue
            new_bits = bits.copy()
            new_bits[control] = (two_q_out >> 1) & 1
            new_bits[target] = two_q_out & 1
            row = sum(new_bits[q] << (num_qubits - 1 - q) for q in range(num_qubits))
            full[row, col] += amp

    return full


_VALID_ROTATION_AXES = {"RX", "RY", "RZ"}


def _normalize_rotation_axes(rotation_gates: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize configured rotation-gate names to uppercase pool token prefixes."""
    if rotation_gates is None:
        return ("RZ",)

    normalized: list[str] = []
    seen: set[str] = set()
    for gate in rotation_gates:
        if not isinstance(gate, str):
            raise ValueError("rotation_gates entries must be strings")
        axis = gate.strip().upper()
        if axis not in _VALID_ROTATION_AXES:
            raise ValueError(
                f"Invalid rotation gate {gate!r}. Allowed values: {sorted(_VALID_ROTATION_AXES)}"
            )
        if axis in seen:
            raise ValueError(f"Duplicate rotation gate: {axis!r}")
        seen.add(axis)
        normalized.append(axis)

    if not normalized:
        raise ValueError("rotation_gates must contain at least one gate")
    return tuple(normalized)


def build_operator_pool(
    num_qubits: int,
    rotation_gates: Iterable[str] | None = None,
) -> List[Tuple[str, np.ndarray]]:
    """Return the gate-token vocabulary and reference matrices.

    The trainer uses token names for circuit evaluation. The matrices remain
    useful for target construction and Qiskit-facing utilities; rotation
    matrices use a fixed placeholder angle because sampled angles live in the
    policy output, not in the pool.
    """
    rotation_axes = _normalize_rotation_axes(rotation_gates)
    pool: List[Tuple[str, np.ndarray]] = []

    for qubit in range(num_qubits):
        for axis in rotation_axes:
            base = get_rotation_matrix(axis, _DEFAULT_ANGLE)
            full = _embed_single_qubit(base, qubit, num_qubits)
            pool.append((f"{axis}_q{qubit}", full))

        sx = get_fixed_gate_matrix("SX")
        pool.append((f"SX_q{qubit}", _embed_single_qubit(sx, qubit, num_qubits)))

    cnot = get_fixed_gate_matrix("CNOT")
    for ctrl, tgt in permutations(range(num_qubits), 2):
        pool.append(
            (f"CNOT_q{ctrl}_q{tgt}", _embed_two_qubit(cnot, ctrl, tgt, num_qubits))
        )

    return pool
