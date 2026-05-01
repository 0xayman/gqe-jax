"""Circuit simplification via algebraic rewriting (Rules 1-7).

Rules 1-2 are the user-specified ones; 3-7 are verified extensions:
  1. Merge consecutive same-axis same-qubit rotations: R_P(t1) R_P(t2) = R_P(t1+t2)
  2. Cancel consecutive identical CNOTs: CNOT(c,t) CNOT(c,t) = I
  3. Remove zero-angle rotations: R_P(0) = I
  4. Angle reduction mod 4pi
  5. SX periodicity: SX^2 = RX(pi), SX^4 = I
  6. R_Z commutes past CNOT control  (enables further merging via Rule 1)
  7. R_X commutes past CNOT target   (enables further merging via Rule 1)
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass

import numpy as np

from pareto import ParetoArchive, ParetoPoint

_GATE_OFFSET = 3        # tokens 0/1/2 are BOS/STOP/special
_BOS_TOKEN = 0
_STOP_TOKEN = 1
_PAD_TOKEN = 2
_FOUR_PI = 4.0 * math.pi
_TWO_PI  = 2.0 * math.pi
_ANGLE_TOL = 1e-9
_HASH_ANGLE_SCALE = 1_000_000


# ---------------------------------------------------------------------------
# Internal op representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Op:
    gate_type: str      # 'RX' | 'RY' | 'RZ' | 'SX' | 'CNOT'
    qubits: tuple       # (q,) for 1-qubit gates; (ctrl, tgt) for CNOT
    angle: float = 0.0  # meaningful only for RX / RY / RZ


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reduce_angle(theta: float) -> float:
    """Canonicalise to (-2pi, 2pi] (Rule 4)."""
    theta = math.fmod(theta, _FOUR_PI)
    if theta > _TWO_PI:
        theta -= _FOUR_PI
    elif theta <= -_TWO_PI:
        theta += _FOUR_PI
    return theta


def _commutes_through(gt: str, qubits: tuple, blocker: _Op) -> bool:
    """True if the gate (gt, qubits) can be moved past blocker without error.

    Implements Rules 6 and 7:
      Rule 6: R_Z on CNOT control commutes with CNOT.
      Rule 7: R_X on CNOT target commutes with CNOT.
    """
    if blocker.gate_type != 'CNOT':
        return False
    c, t = blocker.qubits
    return (gt == 'RZ' and qubits == (c,)) or (gt == 'RX' and qubits == (t,))


# ---------------------------------------------------------------------------
# Decode token sequence → list[_Op]
# ---------------------------------------------------------------------------

def _angles_to_bos_layout(tokens: np.ndarray, angles: np.ndarray | None) -> np.ndarray:
    """Return an angle vector indexed like a full BOS-prefixed token row."""
    out = np.zeros(len(tokens), dtype=np.float64)
    if angles is None:
        return out
    raw = np.asarray(angles, dtype=np.float64)
    if raw.size == len(tokens):
        return raw
    n = min(raw.size, max(len(tokens) - 1, 0))
    out[1:1 + n] = raw[:n]
    return out


def _decode(tokens: np.ndarray, angles: np.ndarray, pool) -> list[_Op]:
    ops: list[_Op] = []
    for pos in range(1, len(tokens)):
        tok = int(tokens[pos])
        if tok == _STOP_TOKEN:
            break
        if tok < _GATE_OFFSET:
            continue
        name, _ = pool[tok - _GATE_OFFSET]
        parts = name.split('_')
        gt = parts[0]
        if gt in ('RX', 'RY', 'RZ'):
            q = int(parts[1][1:])
            ops.append(_Op(gt, (q,), float(angles[pos])))
        elif gt == 'SX':
            q = int(parts[1][1:])
            ops.append(_Op('SX', (q,)))
        elif gt == 'CNOT':
            c, t = int(parts[1][1:]), int(parts[2][1:])
            ops.append(_Op('CNOT', (c, t)))
    return ops


# ---------------------------------------------------------------------------
# Encode list[_Op] → token sequence + angles (padded to orig_len)
# ---------------------------------------------------------------------------

def _encode(
    ops: list[_Op],
    pool_name_map: dict[str, int],
    orig_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    toks: list[int] = [_BOS_TOKEN]
    action_angs: list[float] = []
    action_len = max(orig_len - 1, 0)
    for op in ops:
        if len(toks) >= orig_len:
            break
        if op.gate_type in ('RX', 'RY', 'RZ', 'SX'):
            name = f"{op.gate_type}_q{op.qubits[0]}"
        else:
            name = f"CNOT_q{op.qubits[0]}_q{op.qubits[1]}"
        toks.append(pool_name_map[name] + _GATE_OFFSET)
        action_angs.append(op.angle)

    if len(toks) < orig_len:
        toks.append(_STOP_TOKEN)
        action_angs.append(0.0)
    while len(toks) < orig_len:
        toks.append(_PAD_TOKEN)
        action_angs.append(0.0)
    while len(action_angs) < action_len:
        action_angs.append(0.0)
    return (
        np.array(toks[:orig_len], dtype=np.int32),
        np.array(action_angs[:action_len], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# One peephole pass: Rules 1-5 + look-through for Rules 6-7
# ---------------------------------------------------------------------------

def _one_pass(ops: list[_Op], rx_qubits: set[int]) -> tuple[list[_Op], bool]:
    """Scan left-to-right, merging / cancelling adjacent-on-qubit gate pairs."""
    out: list[_Op] = []
    changed = False

    for op in ops:
        gt, qubits = op.gate_type, op.qubits

        if gt in ('RX', 'RY', 'RZ'):
            q = qubits[0]
            placed = False
            for j in range(len(out) - 1, -1, -1):
                prev = out[j]
                if q not in prev.qubits:
                    continue                         # gate does not touch q
                if _commutes_through(gt, qubits, prev):
                    continue                         # Rules 6/7: look further back
                if prev.gate_type == gt and prev.qubits == qubits:
                    out[j] = _Op(gt, qubits, prev.angle + op.angle)  # Rule 1
                    changed = True
                    placed = True
                break                               # blocker found (match or not)
            if not placed:
                out.append(op)

        elif gt == 'SX':
            q = qubits[0]
            placed = False
            for j in range(len(out) - 1, -1, -1):
                prev = out[j]
                if q not in prev.qubits:
                    continue
                if prev.gate_type == 'SX' and prev.qubits == qubits and q in rx_qubits:
                    out[j] = _Op('RX', qubits, math.pi)  # Rule 5: SX² = RX(pi)
                    changed = True
                    placed = True
                break
            if not placed:
                out.append(op)

        elif gt == 'CNOT':
            c, t = qubits
            placed = False
            for j in range(len(out) - 1, -1, -1):
                prev = out[j]
                if c not in prev.qubits and t not in prev.qubits:
                    continue
                if prev.gate_type == 'CNOT' and prev.qubits == qubits:
                    out.pop(j)       # Rule 2: CNOT cancellation
                    changed = True
                    placed = True
                break
            if not placed:
                out.append(op)
        else:
            out.append(op)

    # Rules 3 + 4: reduce angles, drop zeros
    filtered: list[_Op] = []
    for op in out:
        if op.gate_type in ('RX', 'RY', 'RZ'):
            r = _reduce_angle(op.angle)
            if abs(r) < _ANGLE_TOL:
                changed = True
                continue                 # Rule 3: zero angle → identity
            if r != op.angle:
                changed = True
            filtered.append(_Op(op.gate_type, op.qubits, r))
        else:
            filtered.append(op)

    return filtered, changed


# ---------------------------------------------------------------------------
# Fixed-point iteration over passes
# ---------------------------------------------------------------------------

def _simplify_ops(ops: list[_Op], rx_qubits: set[int]) -> list[_Op]:
    for _ in range(200):
        ops, changed = _one_pass(ops, rx_qubits)
        if not changed:
            break
    return ops


# ---------------------------------------------------------------------------
# Metric recomputation
# ---------------------------------------------------------------------------

def _compute_metrics(ops: list[_Op], num_qubits: int) -> tuple[int, int, int]:
    """Return (depth, total_gates, cnot_count) from a simplified op list."""
    qubit_time = [0] * num_qubits
    for op in ops:
        layer = max(qubit_time[q] for q in op.qubits)
        for q in op.qubits:
            qubit_time[q] = layer + 1
    depth      = max(qubit_time) if any(qubit_time) else 0
    total      = len(ops)
    cnot_count = sum(1 for op in ops if op.gate_type == 'CNOT')
    return depth, total, cnot_count


def _canonical_hash(ops: list[_Op]) -> str:
    """Return a stable hash for the canonical simplified operation stream."""
    h = hashlib.sha256()
    h.update(b"gqe-simplified-v1")
    for op in ops:
        h.update(op.gate_type.encode("ascii"))
        h.update(b":")
        h.update(",".join(str(int(q)) for q in op.qubits).encode("ascii"))
        if op.gate_type in ("RX", "RY", "RZ"):
            angle_key = int(round(_reduce_angle(op.angle) * _HASH_ANGLE_SCALE))
            h.update(f":{angle_key}".encode("ascii"))
        h.update(b";")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify_token_sequence(
    token_sequence: np.ndarray,
    angles: np.ndarray | None,
    pool,
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, str]:
    """Simplify one BOS-prefixed token sequence and return canonical metrics.

    The returned angle vector follows the rollout/evaluator convention: it is
    aligned to the action tokens and therefore excludes the BOS position.
    """
    tokens = np.asarray(token_sequence, dtype=np.int32)
    aligned_angles = _angles_to_bos_layout(tokens, angles)

    pool_name_map = {name: idx for idx, (name, _) in enumerate(pool)}
    rx_qubits = {int(n.split('_')[1][1:]) for n, _ in pool if n.startswith('RX_')}

    ops = _decode(tokens, aligned_angles, pool)
    ops = _simplify_ops(ops, rx_qubits)

    depth, total, cnot_count = _compute_metrics(ops, num_qubits)
    canonical_hash = _canonical_hash(ops)
    new_tokens, new_angles = _encode(ops, pool_name_map, orig_len=len(tokens))
    return new_tokens, new_angles, depth, total, cnot_count, canonical_hash


def simplify_point(point: ParetoPoint, pool, num_qubits: int) -> ParetoPoint:
    """Return a new ParetoPoint with its circuit algebraically simplified.

    Fidelity is preserved (rules are exact equivalences).
    Metrics (depth, total_gates, cnot_count) are recomputed from the simplified
    gate sequence.
    """
    new_tokens, new_angles, depth, total, cnot_count, canonical_hash = (
        simplify_token_sequence(point.token_sequence, point.opt_angles, pool, num_qubits)
    )

    return ParetoPoint(
        fidelity=point.fidelity,
        depth=depth,
        total_gates=total,
        cnot_count=cnot_count,
        token_sequence=new_tokens,
        epoch=point.epoch,
        opt_angles=new_angles,
        canonical_hash=canonical_hash,
    )


def simplify_pareto_archive(
    archive: ParetoArchive,
    pool,
    num_qubits: int,
) -> ParetoArchive:
    """Simplify every circuit in the archive and return a rebuilt archive.

    The new archive is rebuilt from scratch so that dominance relationships
    are recomputed after simplification (a simplified circuit may now dominate
    one that it did not dominate before).
    """
    new_arc = ParetoArchive(
        max_size=archive.max_size,
        fidelity_floor=archive.fidelity_floor,
        fidelity_tol=archive.fidelity_tol,
    )
    simplified = [
        simplify_point(p, pool, num_qubits)
        for p in archive.to_sorted_list()
    ]
    new_arc.update_batch(simplified)
    return new_arc
