"""Token vocabulary mirroring the parent project's operator pool.

The GQE trainer builds a vocabulary of:

    [<BOS>, <STOP>, <PAD>, <pool gates...>]

with three special tokens occupying ids 0..2 and gate tokens following from
id 3 onward. Inside the pool, gates are ordered:

    for each qubit q in 0..n-1:
        for each axis in rotation_gates:   # e.g. RZ, RY in config order
            "{axis}_q{q}"
        "SX_q{q}"
    for each (ctrl, tgt) in permutations(range(n), 2):
        "CNOT_q{ctrl}_q{tgt}"

This module re-creates that layout without importing the parent project so
the representation-learning checkpoints are reproducible from the rotation
gate set alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

BOS_TOKEN_ID = 0
STOP_TOKEN_ID = 1
PAD_TOKEN_ID = 2
GATE_TOKEN_OFFSET = 3

_VALID_ROTATION_AXES = ("RX", "RY", "RZ")


@dataclass(frozen=True)
class Vocab:
    """Token vocabulary description plus per-token metadata."""

    num_qubits: int
    rotation_gates: tuple[str, ...]   # uppercase, e.g. ("RZ", "RY")
    token_names: tuple[str, ...]      # full vocab including specials
    name_to_id: dict[str, int]
    is_parametric: tuple[bool, ...]   # per token id
    qubit0: tuple[int, ...]           # per token id, -1 for non-gate tokens
    qubit1: tuple[int, ...]           # per token id, -1 for non-CNOT tokens

    @property
    def vocab_size(self) -> int:
        return len(self.token_names)

    def gate_token_id(self, name: str) -> int:
        """Look up a token id by gate name (e.g. ``"RZ_q0"``)."""
        try:
            return self.name_to_id[name]
        except KeyError as exc:
            raise KeyError(
                f"gate {name!r} not in vocabulary (rotation_gates="
                f"{self.rotation_gates}, num_qubits={self.num_qubits})"
            ) from exc


def _normalize_rotation_axes(rotation_gates: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for gate in rotation_gates:
        axis = str(gate).strip().upper()
        if axis not in _VALID_ROTATION_AXES:
            raise ValueError(
                f"invalid rotation gate {gate!r}; allowed: {_VALID_ROTATION_AXES}"
            )
        if axis in seen:
            raise ValueError(f"duplicate rotation gate {axis!r}")
        seen.add(axis)
        normalized.append(axis)
    if not normalized:
        raise ValueError("rotation_gates must be non-empty")
    return tuple(normalized)


def build_vocab(num_qubits: int, rotation_gates: Iterable[str]) -> Vocab:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    axes = _normalize_rotation_axes(rotation_gates)

    names: list[str] = ["<BOS>", "<STOP>", "<PAD>"]
    is_parametric: list[bool] = [False, False, False]
    q0: list[int] = [-1, -1, -1]
    q1: list[int] = [-1, -1, -1]

    for q in range(num_qubits):
        for axis in axes:
            names.append(f"{axis}_q{q}")
            is_parametric.append(True)
            q0.append(q)
            q1.append(-1)
        names.append(f"SX_q{q}")
        is_parametric.append(False)
        q0.append(q)
        q1.append(-1)

    for ctrl, tgt in permutations(range(num_qubits), 2):
        names.append(f"CNOT_q{ctrl}_q{tgt}")
        is_parametric.append(False)
        q0.append(ctrl)
        q1.append(tgt)

    return Vocab(
        num_qubits=int(num_qubits),
        rotation_gates=axes,
        token_names=tuple(names),
        name_to_id={n: i for i, n in enumerate(names)},
        is_parametric=tuple(is_parametric),
        qubit0=tuple(q0),
        qubit1=tuple(q1),
    )


def remap_tokens(
    tokens: "np.ndarray",
    from_vocab: Vocab,
    to_vocab: Vocab,
) -> "np.ndarray":
    """Remap token IDs from one vocab to another using token names as the key."""
    import numpy as np  # local import so vocab.py stays importable without numpy at module level
    result = np.empty_like(tokens)
    for i, tid in enumerate(tokens.flat):
        result.flat[i] = to_vocab.name_to_id[from_vocab.token_names[int(tid)]]
    return result
