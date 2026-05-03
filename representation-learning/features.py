"""Pre-processing for unitary inputs.

A target unitary ``U`` and its global-phase rotation ``e^{i phi} U`` describe
the same physical operation. Without normalization the model would waste
capacity learning this equivalence. We strip the global phase by dividing out
``det(U)^{1/d}`` (chosen so the result has determinant 1), then flatten to a
real feature vector.

Optional Pauli-transfer-matrix features are provided for callers that want a
basis-aware representation; they are not required by the default encoder.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


def infer_num_qubits(d: int) -> int:
    if d <= 0 or d & (d - 1):
        raise ValueError(f"unitary dimension must be a power of 2, got {d}")
    return d.bit_length() - 1


def phase_normalize(u: np.ndarray) -> np.ndarray:
    """Return ``U / det(U)^{1/d}`` so the result has unit determinant.

    The d-th root is chosen with the principal branch (smallest absolute
    argument). Two unitaries that differ only by a global phase map to the
    same normalized matrix up to an at-most ``d``-fold ambiguity (a discrete
    choice of root of unity), which is fine for downstream feature use.
    """
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {u.shape}")
    d = u.shape[0]
    det = np.linalg.det(u)
    if not np.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("unitary determinant has near-zero magnitude")
    phase = np.angle(det) / d
    return (u * np.exp(-1j * phase)).astype(np.complex128, copy=False)


def flatten_complex(u: np.ndarray) -> np.ndarray:
    """Concatenate real and imaginary parts of a flattened complex matrix."""
    flat = u.reshape(-1)
    return np.concatenate([flat.real, flat.imag]).astype(np.float32)


def feature_dim(num_qubits: int) -> int:
    d = 2 ** num_qubits
    return 2 * d * d


def unitary_to_features(u: np.ndarray) -> np.ndarray:
    """Phase-normalize and flatten a unitary into a real feature vector."""
    return flatten_complex(phase_normalize(u))


def batch_unitary_to_features(matrices: Iterable[np.ndarray]) -> np.ndarray:
    """Batched version of :func:`unitary_to_features`."""
    rows = [unitary_to_features(np.asarray(u)) for u in matrices]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows, axis=0)


_PAULI = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _pauli_basis(num_qubits: int) -> np.ndarray:
    """Return the ``4**n`` normalized Pauli operators on ``n`` qubits."""
    d = 2 ** num_qubits
    norm = 1.0 / np.sqrt(d)
    basis = []
    for labels in product("IXYZ", repeat=num_qubits):
        op = _PAULI[labels[0]]
        for label in labels[1:]:
            op = np.kron(op, _PAULI[label])
        basis.append(op * norm)
    return np.stack(basis, axis=0)


def pauli_transfer_matrix(u: np.ndarray) -> np.ndarray:
    """Return ``R[i, j] = Tr(P_i U P_j U^dagger)`` (real ``4**n`` x ``4**n``).

    The PTM is invariant under global phase by construction, and tends to be
    a more parsimonious representation for unitaries close to Clifford-like
    targets. It is offered as an alternative or complementary feature to the
    flattened complex matrix.
    """
    n = infer_num_qubits(u.shape[0])
    basis = _pauli_basis(n)
    udag = u.conj().T
    intermediate = np.einsum("kab,bc->kac", basis, udag)
    intermediate = np.einsum("ab,kbc->kac", u, intermediate)
    ptm = np.einsum("iab,jba->ij", basis, intermediate)
    return ptm.real.astype(np.float32)


def ptm_dim(num_qubits: int) -> int:
    return (4 ** num_qubits) ** 2


# ---------------------------------------------------------------------------
# Qubit-agnostic per-qubit features
# ---------------------------------------------------------------------------

SITE_FEAT_DIM: int = 32  # 2 * 4 * 4 (flatten_complex of a 4×4 complex Choi)


def per_qubit_choi_features(u: np.ndarray) -> np.ndarray:
    """Return shape ``[n, SITE_FEAT_DIM]`` per-qubit reduced Choi features.

    For each qubit ``q`` the 4×4 reduced Choi matrix is computed by tracing
    out all other qubits from the full Choi matrix of ``U``.  The result is
    flattened (real then imaginary parts) to a vector of length 32.

    The representation is invariant to global phase and is fixed-size per
    qubit regardless of ``n``, so models built on it generalise across qubit
    counts without retraining.
    """
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {u.shape}")
    d = u.shape[0]
    n = infer_num_qubits(d)
    u_norm = phase_normalize(u)
    items: list[np.ndarray] = []
    for q in range(n):
        j_q = _reduced_choi_qubit(u_norm, n, d, q)
        items.append(flatten_complex(j_q))
    return np.stack(items, axis=0).astype(np.float32)


def _reduced_choi_qubit(
    u: np.ndarray,
    n: int,
    d: int,
    q: int,
) -> np.ndarray:
    """Compute the 4×4 reduced Choi matrix for qubit ``q`` of ``u``.

    ``u`` must already be phase-normalised.  Axes convention: the n-qubit
    unitary is indexed in MSB-first (big-endian) order, i.e. row index
    ``r = Σ_k  b_k * 2^{n-1-k}`` where ``b_k`` is the bit for qubit ``k``.
    The returned matrix ``J_q`` satisfies
        ``J_q[i,j; k,l] = (1/d) Σ_{a,b} U_r[i,a,k,b] * conj(U_r[j,a,l,b])``
    where ``U_r`` is ``u`` reshaped as ``(2, d//2, 2, d//2)`` with qubit
    ``q`` as the leading output and input dimension.
    """
    dr = d // 2
    # Reshape U as (out_0, out_1, ..., out_{n-1}, in_0, ..., in_{n-1}).
    u_t = u.reshape([2] * n + [2] * n)
    # Permute so qubit q comes first in both output and input halves.
    out_axes = [q] + [i for i in range(n) if i != q]
    in_axes = [n + q] + [n + i for i in range(n) if i != q]
    u_p = np.transpose(u_t, out_axes + in_axes).reshape(2, dr, 2, dr)
    # Contract over the d//2 "rest" dimensions to get the 4×4 Choi marginal.
    j_q = np.einsum("iakb,jalb->ijkl", u_p, u_p.conj()) / d
    return j_q.reshape(4, 4)
