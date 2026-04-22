"""Structural simplification of token sequences before parameter optimisation.

Operates on a single fixed-length token buffer: cancelled / merged positions
are overwritten with STOP so they map to NOOP downstream (identity in the
fidelity evaluation, skipped in depth / gate counting). The circuit unitary
is preserved up to reparametrisation of surviving rotations; the optimiser
then solves a problem with fewer parametric angles.

Rules, applied left-to-right with one forward scan per anchor gate:

  1. Merge consecutive same-axis rotations on the same qubit
     ``Rz(a)·Rz(b) → Rz(a+b)``.
  2. Cancel adjacent identical CNOTs ``CNOT(c,t)·CNOT(c,t) = I``.
  3. Commute-then-merge: rules 1 and 2 fire across any run of gates that
     commute with the anchor on its support.

Commutation model (conservative — never claim commutation where it fails):
  - Disjoint supports commute.
  - Two single-qubit gates commute only if they are the same axis on the same
    qubit (enabling rule 1).
  - ``Rz`` on qubit ``q`` commutes with ``CNOT`` whose control is ``q``.
  - ``Rx`` / ``SX`` on qubit ``q`` commute with ``CNOT`` whose target is ``q``.
  - Two ``CNOT``s on overlapping supports commute iff they are identical,
    share only the control qubit, or share only the target qubit. Any
    control-vs-target crossing breaks commutation.
"""

from __future__ import annotations

import numpy as np

TOKEN_AXIS_NOOP = 0
TOKEN_AXIS_RX = 1
TOKEN_AXIS_RY = 2
TOKEN_AXIS_RZ = 3
TOKEN_AXIS_SX = 4
TOKEN_AXIS_CNOT = 5


def build_token_axis(pool_names: list[str]) -> np.ndarray:
    """Return a ``(vocab_size,)`` int32 axis-id table keyed by token id."""
    axis = np.zeros((len(pool_names),), dtype=np.int32)
    for i, name in enumerate(pool_names):
        if name.startswith("<"):
            axis[i] = TOKEN_AXIS_NOOP
        elif name.startswith("RX"):
            axis[i] = TOKEN_AXIS_RX
        elif name.startswith("RY"):
            axis[i] = TOKEN_AXIS_RY
        elif name.startswith("RZ"):
            axis[i] = TOKEN_AXIS_RZ
        elif name.startswith("SX"):
            axis[i] = TOKEN_AXIS_SX
        elif name.startswith("CNOT"):
            axis[i] = TOKEN_AXIS_CNOT
        else:
            raise ValueError(f"Unknown pool name: {name!r}")
    return axis


def _commutes(
    axis_a: int, q0_a: int, q1_a: int,
    axis_b: int, q0_b: int, q1_b: int,
) -> bool:
    a_is_two = q1_a >= 0
    b_is_two = q1_b >= 0
    support_a = {q0_a, q1_a} if a_is_two else {q0_a}
    support_b = {q0_b, q1_b} if b_is_two else {q0_b}
    if support_a.isdisjoint(support_b):
        return True

    # Both single-qubit, shared qubit.
    if not a_is_two and not b_is_two:
        return axis_a == axis_b and q0_a == q0_b

    # Single-qubit vs CNOT, overlapping support.
    if not a_is_two and b_is_two:
        if axis_a == TOKEN_AXIS_RZ and q0_a == q0_b:
            return True
        if axis_a in (TOKEN_AXIS_RX, TOKEN_AXIS_SX) and q0_a == q1_b:
            return True
        return False
    if not b_is_two and a_is_two:
        if axis_b == TOKEN_AXIS_RZ and q0_b == q0_a:
            return True
        if axis_b in (TOKEN_AXIS_RX, TOKEN_AXIS_SX) and q0_b == q1_a:
            return True
        return False

    # CNOT vs CNOT on overlapping supports.
    if (q0_a, q1_a) == (q0_b, q1_b):
        return True
    cross = (q0_a == q1_b) or (q1_a == q0_b)
    return not cross


def _is_rotation(axis: int) -> bool:
    return axis == TOKEN_AXIS_RX or axis == TOKEN_AXIS_RY or axis == TOKEN_AXIS_RZ


def _simplify_row_pylists(
    toks: list,
    axis_tbl: list,
    q0_tbl: list,
    q1_tbl: list,
    stop_id: int,
) -> list:
    """Simplify a single sequence working directly on Python lists.

    Avoids numpy scalar ``int(...)`` round-trips in the hot inner loop; all
    comparisons become plain Python int arithmetic, which is ~5× faster than
    repeated numpy scalar indexing for sequences of length ~128.
    """
    n = len(toks)
    is_rot_set = (TOKEN_AXIS_RX, TOKEN_AXIS_RY, TOKEN_AXIS_RZ)
    for i in range(n):
        tok_i = toks[i]
        axis_i = axis_tbl[tok_i]
        if axis_i == TOKEN_AXIS_NOOP:
            continue
        q0_i = q0_tbl[tok_i]
        q1_i = q1_tbl[tok_i]
        a_is_two = q1_i >= 0
        i_is_rot = axis_i in is_rot_set

        j = i + 1
        while j < n:
            tok_j = toks[j]
            axis_j = axis_tbl[tok_j]
            if axis_j == TOKEN_AXIS_NOOP:
                j += 1
                continue
            q0_j = q0_tbl[tok_j]
            q1_j = q1_tbl[tok_j]
            b_is_two = q1_j >= 0

            # Rule 1: merge same-axis rotations on same qubit.
            if (
                i_is_rot
                and axis_i == axis_j
                and not a_is_two
                and not b_is_two
                and q0_i == q0_j
            ):
                toks[j] = stop_id
                j += 1
                continue

            # Rule 2: identical adjacent CNOTs cancel.
            if (
                axis_i == TOKEN_AXIS_CNOT
                and axis_j == TOKEN_AXIS_CNOT
                and q0_i == q0_j
                and q1_i == q1_j
            ):
                toks[i] = stop_id
                toks[j] = stop_id
                break

            # Inline commutation check (hot path — avoids set allocation).
            if a_is_two:
                if b_is_two:
                    disjoint = (q0_i != q0_j and q0_i != q1_j
                                and q1_i != q0_j and q1_i != q1_j)
                else:
                    disjoint = (q0_j != q0_i and q0_j != q1_i)
            else:
                if b_is_two:
                    disjoint = (q0_i != q0_j and q0_i != q1_j)
                else:
                    disjoint = q0_i != q0_j

            if disjoint:
                j += 1
                continue

            # Overlapping support — delegate to full rule (rare branch).
            if not _commutes(axis_i, q0_i, q1_i, axis_j, q0_j, q1_j):
                break
            j += 1

    return toks


def simplify_token_sequence(
    tokens: np.ndarray,
    token_axis: np.ndarray,
    token_q0: np.ndarray,
    token_q1: np.ndarray,
    stop_token_id: int,
) -> np.ndarray:
    """Return a simplified copy of ``tokens`` (shape ``(L,)``)."""
    toks = np.asarray(tokens, dtype=np.int32).tolist()
    axis_tbl = np.asarray(token_axis, dtype=np.int32).tolist()
    q0_tbl = np.asarray(token_q0, dtype=np.int32).tolist()
    q1_tbl = np.asarray(token_q1, dtype=np.int32).tolist()
    simplified = _simplify_row_pylists(
        toks, axis_tbl, q0_tbl, q1_tbl, int(stop_token_id)
    )
    return np.asarray(simplified, dtype=np.int32)


def simplify_token_batch(
    tokens_batch: np.ndarray,
    token_axis: np.ndarray,
    token_q0: np.ndarray,
    token_q1: np.ndarray,
    stop_token_id: int,
) -> np.ndarray:
    """Apply simplification to each row of ``tokens_batch``.

    Metadata tables are converted to Python lists once so the per-row inner
    loop can use native int arithmetic instead of numpy scalar indexing.
    """
    tokens_batch = np.asarray(tokens_batch, dtype=np.int32)
    if tokens_batch.size == 0:
        return tokens_batch.copy()
    axis_tbl = np.asarray(token_axis, dtype=np.int32).tolist()
    q0_tbl = np.asarray(token_q0, dtype=np.int32).tolist()
    q1_tbl = np.asarray(token_q1, dtype=np.int32).tolist()
    stop_id = int(stop_token_id)
    rows = tokens_batch.tolist()
    simplified = [
        _simplify_row_pylists(row, axis_tbl, q0_tbl, q1_tbl, stop_id)
        for row in rows
    ]
    return np.asarray(simplified, dtype=np.int32)
