"""Vectorized structural constraint enforcement for JAX-based sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax_setup  # noqa: F401
import jax.numpy as jnp


@dataclass(frozen=True)
class _GateInfo:
    gate_type: str
    qubits: Tuple[int, ...]
    qubit: Optional[int]
    ctrl: Optional[int]
    tgt: Optional[int]


class CircuitGrammar:
    def __init__(self, num_qubits: int, pool: List[Tuple[str, Any]]) -> None:
        self.num_qubits = num_qubits
        self.pool_names: List[str] = [name for name, _ in pool]
        self.pool_size = len(self.pool_names)

        self._info: List[_GateInfo] = [self._parse_name(n) for n in self.pool_names]
        self._cnot_pairs: List[Tuple[int, int]] = [
            (g.ctrl, g.tgt) for g in self._info if g.gate_type == "CNOT"
        ]
        self._num_cnot_pairs = len(self._cnot_pairs)
        self._cnot_pair_to_idx = {pair: i for i, pair in enumerate(self._cnot_pairs)}
        self._build_tables()

    @staticmethod
    def _parse_name(name: str) -> _GateInfo:
        if name.startswith("CNOT_"):
            parts = name.split("_q")
            ctrl, tgt = int(parts[1]), int(parts[2])
            return _GateInfo("CNOT", (ctrl, tgt), None, ctrl, tgt)
        if name.startswith("SX_q"):
            q = int(name[4:])
            return _GateInfo("SX", (q,), q, None, None)
        axis = name[:2]
        q = int(name[4:])
        return _GateInfo(axis, (q,), q, None, None)

    def _commutes(self, ia: int, ib: int) -> bool:
        a, b = self._info[ia], self._info[ib]
        qa, qb = set(a.qubits), set(b.qubits)
        if qa.isdisjoint(qb):
            return True
        if a.gate_type == "RZ" and b.gate_type == "CNOT" and a.qubit == b.ctrl:
            return True
        if b.gate_type == "RZ" and a.gate_type == "CNOT" and b.qubit == a.ctrl:
            return True
        if a.gate_type == "RZ" and b.gate_type == "RZ":
            return True
        if a.gate_type == "SX" and b.gate_type == "SX" and a.qubit == b.qubit:
            return True
        return False

    def _build_tables(self) -> None:
        pool_size = self.pool_size
        num_qubits = self.num_qubits
        num_cnot_pairs = self._num_cnot_pairs

        rz_pool, rz_qubit = [], []
        sx_pool, sx_qubit = [], []
        cnot_pool, cnot_pair = [], []
        for pool_idx, gate in enumerate(self._info):
            if gate.gate_type == "RZ":
                rz_pool.append(pool_idx)
                rz_qubit.append(gate.qubit)
            elif gate.gate_type == "SX":
                sx_pool.append(pool_idx)
                sx_qubit.append(gate.qubit)
            elif gate.gate_type == "CNOT":
                cnot_pool.append(pool_idx)
                cnot_pair.append(self._cnot_pair_to_idx[(gate.ctrl, gate.tgt)])

        self._rz_pool = jnp.asarray(rz_pool, dtype=jnp.int32)
        self._rz_qubit = jnp.asarray(rz_qubit, dtype=jnp.int32)
        self._sx_pool = jnp.asarray(sx_pool, dtype=jnp.int32)
        self._sx_qubit = jnp.asarray(sx_qubit, dtype=jnp.int32)
        self._cnot_pool = jnp.asarray(cnot_pool, dtype=jnp.int32)
        self._cnot_pair = jnp.asarray(cnot_pair, dtype=jnp.int32)

        comm = [[self._commutes(i, j) for j in range(pool_size)] for i in range(pool_size)]
        comm = jnp.asarray(comm, dtype=bool)
        indices = jnp.arange(pool_size, dtype=jnp.int32)
        self._canonical_forbidden = comm & (indices[:, None] < indices[None, :])

        rz_set = jnp.zeros((pool_size, num_qubits), dtype=bool)
        rz_clear = jnp.zeros((pool_size, num_qubits), dtype=bool)
        sx_inc = jnp.zeros((pool_size, num_qubits), dtype=bool)
        sx_reset = jnp.zeros((pool_size, num_qubits), dtype=bool)
        cnot_set = jnp.zeros((pool_size, num_cnot_pairs), dtype=bool)
        cnot_clear = jnp.zeros((pool_size, num_cnot_pairs), dtype=bool)

        for pool_idx, gate in enumerate(self._info):
            if gate.gate_type == "RZ":
                rz_set = rz_set.at[pool_idx, gate.qubit].set(True)
            if gate.gate_type == "SX":
                rz_clear = rz_clear.at[pool_idx, gate.qubit].set(True)
                sx_inc = sx_inc.at[pool_idx, gate.qubit].set(True)
            if gate.gate_type == "CNOT":
                rz_clear = rz_clear.at[pool_idx, gate.tgt].set(True)
                sx_reset = sx_reset.at[pool_idx, gate.ctrl].set(True)
                sx_reset = sx_reset.at[pool_idx, gate.tgt].set(True)
                cnot_set = cnot_set.at[
                    pool_idx, self._cnot_pair_to_idx[(gate.ctrl, gate.tgt)]
                ].set(True)
            if gate.gate_type == "RZ":
                sx_reset = sx_reset.at[pool_idx, gate.qubit].set(True)

            for pair_idx, (ctrl, tgt) in enumerate(self._cnot_pairs):
                if gate.gate_type == "RZ" and gate.qubit == tgt:
                    cnot_clear = cnot_clear.at[pool_idx, pair_idx].set(True)
                elif gate.gate_type == "SX" and gate.qubit in (ctrl, tgt):
                    cnot_clear = cnot_clear.at[pool_idx, pair_idx].set(True)
                elif gate.gate_type == "CNOT":
                    if (gate.ctrl, gate.tgt) != (ctrl, tgt) and (
                        gate.ctrl in (ctrl, tgt) or gate.tgt in (ctrl, tgt)
                    ):
                        cnot_clear = cnot_clear.at[pool_idx, pair_idx].set(True)

        self._rz_set = rz_set
        self._rz_clear = rz_clear
        self._sx_inc = sx_inc
        self._sx_reset = sx_reset
        self._cnot_set = cnot_set
        self._cnot_clear = cnot_clear

    def initial_state(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        return {
            "rz_active": jnp.zeros((batch_size, self.num_qubits), dtype=bool),
            "sx_count": jnp.zeros((batch_size, self.num_qubits), dtype=jnp.int8),
            "cnot_active": jnp.zeros((batch_size, self._num_cnot_pairs), dtype=bool),
            "last_gate": jnp.full((batch_size,), -1, dtype=jnp.int32),
        }

    def forbidden_mask(self, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        batch_size = state["rz_active"].shape[0]
        mask = jnp.zeros((batch_size, self.pool_size), dtype=bool)

        if self._rz_pool.size > 0:
            mask = mask.at[:, self._rz_pool].set(state["rz_active"][:, self._rz_qubit])
        if self._cnot_pool.size > 0:
            mask = mask.at[:, self._cnot_pool].set(
                state["cnot_active"][:, self._cnot_pair]
            )
        if self._sx_pool.size > 0:
            mask = mask.at[:, self._sx_pool].set(state["sx_count"][:, self._sx_qubit] == 3)

        valid = state["last_gate"] >= 0
        safe_last = jnp.maximum(state["last_gate"], 0)
        canon = self._canonical_forbidden[safe_last]
        mask = mask | (canon & valid[:, None])

        all_forbidden = jnp.all(mask, axis=1, keepdims=True)
        return mask & (~all_forbidden)

    def update(
        self,
        state: Dict[str, jnp.ndarray],
        gate_indices: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        sets_rz = self._rz_set[gate_indices]
        clears_rz = self._rz_clear[gate_indices]
        inc_sx = self._sx_inc[gate_indices]
        reset_sx = self._sx_reset[gate_indices]
        sets_cnot = self._cnot_set[gate_indices]
        clears_cnot = self._cnot_clear[gate_indices]

        return {
            "rz_active": (state["rz_active"] | sets_rz) & ~clears_rz,
            "sx_count": ((state["sx_count"] + inc_sx.astype(jnp.int8)) % 4) * (
                ~reset_sx
            ),
            "cnot_active": (state["cnot_active"] | sets_cnot) & ~clears_cnot,
            "last_gate": gate_indices.astype(jnp.int32),
        }
