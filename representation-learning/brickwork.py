"""Brickwork-Haar unitary generator (local copy of the parent's logic).

This is intentionally re-implemented here rather than importing
``target.brickwork_haar_generator`` so the dataset can be reproduced from a
seed alone, without the parent project's config dataclasses.
"""

from __future__ import annotations

import numpy as np


def haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-uniform unitary on dimension ``d`` (Mezzadri's QR trick)."""
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return q * phase[np.newaxis, :]


def _embed_adjacent_2q(
    gate_4x4: np.ndarray,
    q0: int,
    num_qubits: int,
) -> np.ndarray:
    """Embed an adjacent two-qubit gate under the project's MSB-first convention."""
    if not (0 <= q0 < num_qubits - 1):
        raise ValueError(
            f"adjacent 2q embedding needs 0 <= q0 < n-1, got q0={q0}, n={num_qubits}"
        )
    left = np.eye(2 ** q0, dtype=gate_4x4.dtype)
    right = np.eye(2 ** (num_qubits - q0 - 2), dtype=gate_4x4.dtype)
    return np.kron(np.kron(left, gate_4x4), right)


def brickwork_haar(
    num_qubits: int,
    depth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compose a brickwork (alternating-NN) Haar circuit and return the full unitary."""
    if num_qubits < 2:
        raise ValueError("brickwork_haar requires num_qubits >= 2")
    if depth <= 0:
        raise ValueError("depth must be positive")
    d = 2 ** num_qubits
    u = np.eye(d, dtype=np.complex128)
    for layer in range(depth):
        offset = layer % 2
        for q0 in range(offset, num_qubits - 1, 2):
            haar_2q = haar_unitary(4, rng).astype(np.complex128)
            full_gate = _embed_adjacent_2q(haar_2q, q0, num_qubits)
            u = full_gate @ u
    return u
