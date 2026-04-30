"""Host-side closed-form sweep updates for single-axis rotation tokens."""

from __future__ import annotations

import numpy as np

from circuit import (
    GATE_TYPE_CNOT,
    GATE_TYPE_NOOP,
    GATE_TYPE_RX,
    GATE_TYPE_RY,
    GATE_TYPE_RZ,
    GATE_TYPE_SX,
    _CNOT,
    _PAULI_X,
    _PAULI_Y,
    _PAULI_Z,
    _SX,
    _embed_single_qubit,
    _embed_two_qubit,
)


_PAULI_BY_AXIS = {
    GATE_TYPE_RX: _PAULI_X.astype(np.complex128),
    GATE_TYPE_RY: _PAULI_Y.astype(np.complex128),
    GATE_TYPE_RZ: _PAULI_Z.astype(np.complex128),
}


def _build_static_cache(evaluator) -> dict:
    """Cache evaluator metadata needed by the sweep implementation."""
    n = int(evaluator.num_qubits)
    d = 2 ** n
    gate_types = np.asarray(evaluator._tok_gate_type, dtype=np.int32)
    qubit0 = np.asarray(evaluator._tok_qubit0, dtype=np.int32)
    pair_idx = np.asarray(evaluator._tok_cnot_pair, dtype=np.int32)
    cnot_pairs = list(evaluator._cnot_pairs)
    u_target = np.asarray(evaluator.u_target, dtype=np.complex128)
    cnot_full = [
        _embed_two_qubit(_CNOT, c, t, n).astype(np.complex128)
        for c, t in cnot_pairs
    ]
    sx_full = [
        _embed_single_qubit(_SX, q, n).astype(np.complex128)
        for q in range(n)
    ]
    return {
        "n": n,
        "d": d,
        "gate_types": gate_types,
        "q0": qubit0,
        "pair_idx": pair_idx,
        "cnot_full": cnot_full,
        "sx_full": sx_full,
        "u_target": u_target,
        "u_target_dag": u_target.conj().T,
    }


def _rotation_full(axis: int, angle: float, qubit: int, n: int) -> np.ndarray:
    pauli = _PAULI_BY_AXIS[axis]
    half = float(angle) * 0.5
    g2 = (
        np.cos(half) * np.eye(2, dtype=np.complex128)
        - 1j * np.sin(half) * pauli
    )
    return _embed_single_qubit(g2, qubit, n)


def _gate_full(tok: int, angle: float, cache: dict) -> np.ndarray:
    gtype = int(cache["gate_types"][tok])
    if gtype == GATE_TYPE_RX or gtype == GATE_TYPE_RY or gtype == GATE_TYPE_RZ:
        return _rotation_full(gtype, angle, int(cache["q0"][tok]), cache["n"])
    if gtype == GATE_TYPE_SX:
        return cache["sx_full"][int(cache["q0"][tok])]
    if gtype == GATE_TYPE_CNOT:
        return cache["cnot_full"][int(cache["pair_idx"][tok])]
    return np.eye(cache["d"], dtype=np.complex128)


def _partial_trace_to_qubit(M: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Trace out every qubit except the selected one."""
    Mr = M.reshape((2,) * (2 * n))
    perm = list(range(2 * n))
    perm.remove(qubit)
    perm.insert(0, qubit)
    perm.remove(n + qubit)
    perm.insert(1, n + qubit)
    Mp = np.transpose(Mr, perm)
    d_prime = 2 ** (n - 1)
    Mp = Mp.reshape(2, 2, d_prime, d_prime)
    return np.trace(Mp, axis1=2, axis2=3)


def _optimal_angle(M: np.ndarray, axis: int, qubit: int, n: int) -> float:
    A = _partial_trace_to_qubit(M, qubit, n)
    pauli = _PAULI_BY_AXIS[axis]
    a = np.trace(A)
    b = -1j * np.trace(pauli @ A)
    X = float((np.abs(a) ** 2) - (np.abs(b) ** 2))
    Y = float(2.0 * (np.conjugate(a) * b).real)
    if abs(X) < 1e-15 and abs(Y) < 1e-15:
        return 0.0
    return float(np.arctan2(Y, X))


def sweep_refine_one(
    token_ids: np.ndarray,
    angles: np.ndarray,
    cache: dict,
    *,
    num_sweeps: int = 2,
) -> np.ndarray:
    """Per-circuit forward-sweep refinement; returns updated angles."""
    n = cache["n"]
    d = cache["d"]
    L = int(token_ids.shape[0])
    u_target_dag = cache["u_target_dag"]
    angles = np.asarray(angles, dtype=np.float64).copy()

    for _ in range(int(num_sweeps)):
        gates = [_gate_full(int(token_ids[i]), float(angles[i]), cache) for i in range(L)]
        R = [None] * L
        if L > 0:
            R[L - 1] = np.eye(d, dtype=np.complex128)
            for p in range(L - 2, -1, -1):
                R[p] = R[p + 1] @ gates[p + 1]

        L_prev = np.eye(d, dtype=np.complex128)
        for p in range(L):
            tok = int(token_ids[p])
            gtype = int(cache["gate_types"][tok])
            if gtype in (GATE_TYPE_RX, GATE_TYPE_RY, GATE_TYPE_RZ):
                M = L_prev @ u_target_dag @ R[p]
                qubit = int(cache["q0"][tok])
                theta_new = _optimal_angle(M, gtype, qubit, n)
                angles[p] = theta_new
                gates[p] = _rotation_full(gtype, theta_new, qubit, n)
            L_prev = gates[p] @ L_prev

    return angles


def sweep_refine_batch(
    evaluator,
    token_ids_batch: np.ndarray,
    angles_batch: np.ndarray,
    *,
    num_sweeps: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Run sequential sweep refinement on each row and return angles plus fidelity."""
    cache = _build_static_cache(evaluator)
    B, T = angles_batch.shape
    out_angles = np.array(angles_batch, dtype=np.float32).copy()
    if B == 0:
        return out_angles, np.zeros((0,), dtype=np.float32)
    for b in range(B):
        new_angles = sweep_refine_one(
            np.asarray(token_ids_batch[b], dtype=np.int32),
            angles_batch[b],
            cache,
            num_sweeps=num_sweeps,
        )
        out_angles[b] = new_angles.astype(np.float32)
    fids = np.array(
        evaluator.fidelity_batch(token_ids_batch, out_angles),
        dtype=np.float32,
        copy=True,
    )
    return out_angles, fids
