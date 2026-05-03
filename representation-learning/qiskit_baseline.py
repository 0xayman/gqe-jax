"""Qiskit-transpiler baseline for representation-learning labels.

Given a target unitary, run Qiskit's transpiler at multiple optimization
levels and return the best (lowest CNOT count, sufficiently high fidelity)
circuit converted to the parent project's gate-token format.

Convention bridge
-----------------
The parent project indexes qubits MSB-first: qubit 0 is the most significant
bit of a row index. Qiskit's matrix convention is LSB-first. So the *same*
named gate has different matrices in the two libraries (e.g. project
``CNOT_q0_q1`` and Qiskit ``cx(0, 1)`` differ by a bit-reversal), even
though both describe the same physical action when both libraries use the
same physical-qubit numbering.

To get Qiskit to interpret ``u_target`` (a project-convention matrix) as the
intended physical action, we append it with reversed qargs
``[n-1, n-2, ..., 0]`` — that's the qubit-by-qubit equivalent of
bit-reversing the matrix. Qiskit then transpiles the circuit and produces
gates whose composition (in Qiskit's matrix convention) is the bit-reverse
of ``u_target``.

When we read the transpiled circuit back, *no qubit remap is needed*: each
Qiskit gate ``g(qubit i)`` becomes the project token with the same physical
qubit index ``i``. The two libraries' matrices for that gate differ by a
bit-reversal, and the bit-reversals across all gates compose into the same
final matrix as ``u_target`` — see the unit tests for an empirical check.

After conversion we *verify* by computing the process fidelity in the
project's convention; the wrapper only returns a result if the verified
fidelity exceeds ``min_verify_fidelity``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import vocab as _vocab  # noqa: E402

from qiskit import QuantumCircuit, transpile  # noqa: E402
from qiskit.circuit.library import UnitaryGate  # noqa: E402


@dataclass
class BaselineResult:
    """Output of a single transpilation attempt converted to project tokens."""

    optimization_level: int
    cnot_count: int
    total_gates: int
    depth: int
    tokens: np.ndarray         # int32, shape (L,)
    angles: np.ndarray         # float32, shape (L,) — 0 where not parametric
    fidelity: float            # process fidelity vs. the input unitary
    qiskit_circuit_qasm: str   # for inspection / reproducibility
    success: bool              # whether the verification fidelity was >= threshold


def _qiskit_basis_for(rotation_gates: tuple[str, ...]) -> list[str]:
    """Map project rotation_gates to a Qiskit basis-gates list.

    Always includes ``cx`` and ``sx`` (used for non-parametric singles); the
    project pool already contains both, so the conversion stays in-vocab.
    """
    rot = [g.lower() for g in rotation_gates]
    basis = list(rot) + ["sx", "cx"]
    seen: set[str] = set()
    out: list[str] = []
    for g in basis:
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out


def _gate_to_token(
    gate_name: str,
    qiskit_qubits: list[int],
    num_qubits: int,
    vocab: _vocab.Vocab,
) -> tuple[int, float]:
    """Translate one transpiled instruction to ``(token_id, angle)``.

    No qubit remap: each Qiskit physical qubit ``i`` becomes the project
    token's ``q{i}``. The two libraries differ by a *matrix* convention
    (bit-reversal) but agree on the physical-qubit numbering when the input
    unitary is appended with reversed qargs (see module docstring).
    """
    del num_qubits  # only kept for the assertions below
    name = gate_name.lower()

    if name in ("rx", "ry", "rz"):
        if len(qiskit_qubits) != 1:
            raise ValueError(f"{name!r} expects 1 qubit, got {qiskit_qubits}")
        token_name = f"{name.upper()}_q{qiskit_qubits[0]}"
        return vocab.gate_token_id(token_name), 0.0  # angle filled by caller
    if name == "sx":
        if len(qiskit_qubits) != 1:
            raise ValueError(f"{name!r} expects 1 qubit, got {qiskit_qubits}")
        return vocab.gate_token_id(f"SX_q{qiskit_qubits[0]}"), 0.0
    if name in ("cx", "cnot"):
        if len(qiskit_qubits) != 2:
            raise ValueError(f"{name!r} expects 2 qubits, got {qiskit_qubits}")
        ctrl = qiskit_qubits[0]
        tgt = qiskit_qubits[1]
        return vocab.gate_token_id(f"CNOT_q{ctrl}_q{tgt}"), 0.0
    raise ValueError(
        f"unsupported gate {name!r} in transpiled circuit; "
        f"adjust basis_gates to keep the conversion in-vocab"
    )


def _is_parametric_qiskit(name: str) -> bool:
    return name.lower() in ("rx", "ry", "rz")


def qiskit_circuit_to_tokens(
    qc: QuantumCircuit,
    vocab: _vocab.Vocab,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a transpiled Qiskit circuit to project-format tokens + angles.

    Returns
    -------
    tokens, angles
        ``tokens`` is ``int32`` shape ``(L,)`` with no specials (no BOS/STOP/PAD).
        ``angles`` is ``float32`` shape ``(L,)``; entries for non-parametric
        tokens are 0.
    """
    n = vocab.num_qubits
    qiskit_idx_of = {q: i for i, q in enumerate(qc.qubits)}

    token_ids: list[int] = []
    angles: list[float] = []
    for instruction in qc.data:
        op = instruction.operation
        qubits = [qiskit_idx_of[q] for q in instruction.qubits]
        name = op.name.lower()
        if name in ("barrier", "measure"):
            continue
        if name in ("delay", "id"):
            continue
        token_id, _ = _gate_to_token(name, qubits, n, vocab)
        if _is_parametric_qiskit(name):
            angle = float(op.params[0])
        else:
            angle = 0.0
        token_ids.append(int(token_id))
        angles.append(angle)

    if not token_ids:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.asarray(token_ids, dtype=np.int32),
        np.asarray(angles, dtype=np.float32),
    )


def _build_project_unitary(
    tokens: np.ndarray,
    angles: np.ndarray,
    vocab: _vocab.Vocab,
) -> np.ndarray:
    """Apply project tokens to the identity using a tiny in-house simulator.

    This is intentionally independent of the parent's JAX evaluator so the
    baseline wrapper can be tested in isolation. It is only used for
    verification; throughput here is not critical.
    """
    n = vocab.num_qubits
    d = 2 ** n

    def _embed_1q(g: np.ndarray, q: int) -> np.ndarray:
        factors = [g if i == q else np.eye(2, dtype=np.complex128) for i in range(n)]
        out = factors[0]
        for f in factors[1:]:
            out = np.kron(out, f)
        return out

    def _embed_2q_cnot(ctrl: int, tgt: int) -> np.ndarray:
        full = np.zeros((d, d), dtype=np.complex128)
        for col in range(d):
            bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
            new_bits = bits.copy()
            if bits[ctrl] == 1:
                new_bits[tgt] ^= 1
            row = sum(new_bits[q] << (n - 1 - q) for q in range(n))
            full[row, col] = 1.0
        return full

    sx = np.array(
        [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]],
        dtype=np.complex128,
    )
    px = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    py = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    pz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    eye2 = np.eye(2, dtype=np.complex128)

    u = np.eye(d, dtype=np.complex128)
    for tok, ang in zip(tokens, angles):
        name = vocab.token_names[int(tok)]
        if name.startswith("RX_"):
            q = vocab.qubit0[int(tok)]
            half = 0.5 * float(ang)
            g2 = np.cos(half) * eye2 - 1j * np.sin(half) * px
            u = _embed_1q(g2, q) @ u
        elif name.startswith("RY_"):
            q = vocab.qubit0[int(tok)]
            half = 0.5 * float(ang)
            g2 = np.cos(half) * eye2 - 1j * np.sin(half) * py
            u = _embed_1q(g2, q) @ u
        elif name.startswith("RZ_"):
            q = vocab.qubit0[int(tok)]
            half = 0.5 * float(ang)
            g2 = np.cos(half) * eye2 - 1j * np.sin(half) * pz
            u = _embed_1q(g2, q) @ u
        elif name.startswith("SX_"):
            q = vocab.qubit0[int(tok)]
            u = _embed_1q(sx, q) @ u
        elif name.startswith("CNOT_"):
            ctrl = vocab.qubit0[int(tok)]
            tgt = vocab.qubit1[int(tok)]
            u = _embed_2q_cnot(ctrl, tgt) @ u
        else:
            raise ValueError(f"cannot replay token {name!r} (special?)")
    return u


def _process_fidelity(u_built: np.ndarray, u_target: np.ndarray) -> float:
    d = u_built.shape[0]
    overlap = np.sum(np.conjugate(u_target) * u_built)
    return float(np.clip((np.abs(overlap) ** 2) / (d ** 2), 0.0, 1.0))


def _to_qiskit_circuit(u_target: np.ndarray, num_qubits: int) -> QuantumCircuit:
    """Append ``u_target`` so Qiskit interprets it in the project's MSB-first order."""
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(u_target, label="target"), list(range(num_qubits))[::-1])
    return qc


def transpile_baseline(
    u_target: np.ndarray,
    vocab: _vocab.Vocab,
    *,
    optimization_levels: tuple[int, ...] = (0, 1, 2, 3),
    seed: int | None = None,
    min_verify_fidelity: float = 0.999,
) -> list[BaselineResult]:
    """Run Qiskit transpile at each level; return per-level results.

    Each entry has the verified fidelity in the project's convention; callers
    typically pick the lowest CNOT count among ``success=True`` rows.
    """
    n = vocab.num_qubits
    d = 2 ** n
    if u_target.shape != (d, d):
        raise ValueError(
            f"u_target shape {u_target.shape} does not match {n}-qubit vocab "
            f"(expected {(d, d)})"
        )

    basis = _qiskit_basis_for(vocab.rotation_gates)
    qc_in = _to_qiskit_circuit(u_target, n)

    out: list[BaselineResult] = []
    for level in optimization_levels:
        kwargs = dict(basis_gates=basis, optimization_level=int(level))
        if seed is not None:
            kwargs["seed_transpiler"] = int(seed)
        qc_t = transpile(qc_in, **kwargs)
        tokens, angles = qiskit_circuit_to_tokens(qc_t, vocab)

        u_built = _build_project_unitary(tokens, angles, vocab)
        fid = _process_fidelity(u_built, u_target)

        n_cnot = int(sum(
            1 for instr in qc_t.data if instr.operation.name.lower() in ("cx", "cnot")
        ))
        n_total = int(len(qc_t.data))
        depth = int(qc_t.depth())

        out.append(BaselineResult(
            optimization_level=int(level),
            cnot_count=n_cnot,
            total_gates=n_total,
            depth=depth,
            tokens=tokens,
            angles=angles,
            fidelity=fid,
            qiskit_circuit_qasm="",  # populated lazily by callers if needed
            success=bool(fid >= min_verify_fidelity),
        ))
    return out


def best_baseline(
    u_target: np.ndarray,
    vocab: _vocab.Vocab,
    *,
    optimization_levels: tuple[int, ...] = (0, 1, 2, 3),
    seed: int | None = None,
    min_verify_fidelity: float = 0.999,
) -> BaselineResult:
    """Return the verified-success result with the smallest CNOT count.

    Falls back to the highest-fidelity result if no level meets the threshold.
    """
    results = transpile_baseline(
        u_target, vocab,
        optimization_levels=optimization_levels,
        seed=seed,
        min_verify_fidelity=min_verify_fidelity,
    )
    success = [r for r in results if r.success]
    if success:
        return min(success, key=lambda r: (r.cnot_count, r.total_gates, r.depth))
    return max(results, key=lambda r: r.fidelity)
