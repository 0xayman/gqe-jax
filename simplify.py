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


def simplify_token_sequence(
    tokens: np.ndarray,
    token_axis: np.ndarray,
    token_q0: np.ndarray,
    token_q1: np.ndarray,
    stop_token_id: int,
) -> np.ndarray:
    """Return a simplified copy of ``tokens`` (shape ``(L,)``).

    Merged and cancelled positions are rewritten to ``stop_token_id`` so the
    sequence length stays fixed. Surviving positions retain their original
    token id — the optimiser re-parametrises the corresponding rotations.
    """
    tokens = np.asarray(tokens, dtype=np.int32).copy()
    n = int(tokens.shape[0])

    for i in range(n):
        tok_i = int(tokens[i])
        axis_i = int(token_axis[tok_i])
        if axis_i == TOKEN_AXIS_NOOP:
            continue
        q0_i = int(token_q0[tok_i])
        q1_i = int(token_q1[tok_i])

        j = i + 1
        while j < n:
            tok_j = int(tokens[j])
            axis_j = int(token_axis[tok_j])
            if axis_j == TOKEN_AXIS_NOOP:
                j += 1
                continue
            q0_j = int(token_q0[tok_j])
            q1_j = int(token_q1[tok_j])

            # Rule 1: merge same-axis rotations on same qubit — absorb j into i.
            if (
                _is_rotation(axis_i)
                and axis_i == axis_j
                and q1_i < 0
                and q1_j < 0
                and q0_i == q0_j
            ):
                tokens[j] = stop_token_id
                j += 1
                continue

            # Rule 2: identical adjacent (through commutes) CNOTs cancel.
            if (
                axis_i == TOKEN_AXIS_CNOT
                and axis_j == TOKEN_AXIS_CNOT
                and q0_i == q0_j
                and q1_i == q1_j
            ):
                tokens[i] = stop_token_id
                tokens[j] = stop_token_id
                break

            if not _commutes(axis_i, q0_i, q1_i, axis_j, q0_j, q1_j):
                break
            j += 1

    return tokens


def simplify_token_batch(
    tokens_batch: np.ndarray,
    token_axis: np.ndarray,
    token_q0: np.ndarray,
    token_q1: np.ndarray,
    stop_token_id: int,
) -> np.ndarray:
    """Apply :func:`simplify_token_sequence` to each row of ``tokens_batch``."""
    tokens_batch = np.asarray(tokens_batch, dtype=np.int32)
    if tokens_batch.size == 0:
        return tokens_batch.copy()
    out = np.empty_like(tokens_batch)
    for b in range(tokens_batch.shape[0]):
        out[b] = simplify_token_sequence(
            tokens_batch[b], token_axis, token_q0, token_q1, stop_token_id
        )
    return out
