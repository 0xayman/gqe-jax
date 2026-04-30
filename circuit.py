"""JAX circuit evaluator for token and angle sequences."""

from __future__ import annotations

from dataclasses import dataclass

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


GATE_TYPE_RX = 0
GATE_TYPE_RY = 1
GATE_TYPE_RZ = 2
GATE_TYPE_SX = 3
GATE_TYPE_CNOT = 4
GATE_TYPE_NOOP = 5

DEFAULT_ANGLE = float(np.pi / 4)


@dataclass(frozen=True)
class GateSpec:
    """Parsed structural description of a single pool gate."""

    gate_type: str
    qubits: tuple[int, ...]
    is_parametric: bool


def parse_gate_name(name: str) -> GateSpec:
    """Parse a pool token name (e.g. ``"RZ_q0"``, ``"CNOT_q0_q1"``)."""
    if name.startswith("<") and name.endswith(">"):
        return GateSpec("NOOP", (0,), False)
    if name.startswith(("RX", "RY", "RZ")):
        parts = name.split("_")
        return GateSpec(parts[0], (int(parts[1][1:]),), True)
    if name.startswith("SX"):
        qubit = int(name.split("_")[1][1:])
        return GateSpec("SX", (qubit,), False)
    if name.startswith("CNOT"):
        parts = name.split("_")
        return GateSpec("CNOT", (int(parts[1][1:]), int(parts[2][1:])), False)
    raise ValueError(f"Unknown gate name: {name!r}")


_GATE_TYPE_TO_ID = {
    "RX": GATE_TYPE_RX,
    "RY": GATE_TYPE_RY,
    "RZ": GATE_TYPE_RZ,
    "SX": GATE_TYPE_SX,
    "CNOT": GATE_TYPE_CNOT,
    "NOOP": GATE_TYPE_NOOP,
}


_PAULI_X = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_SX = np.asarray(
    [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]],
    dtype=np.complex128,
)
_CNOT = np.asarray(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)


@jax.custom_vjp
def _permute_rows(U: jax.Array, perm: jax.Array, inv_perm: jax.Array) -> jax.Array:
    return U[perm]


def _permute_rows_fwd(U, perm, inv_perm):
    return U[perm], (inv_perm,)


def _permute_rows_bwd(res, g):
    (inv_perm,) = res
    return (g[inv_perm], None, None)


_permute_rows.defvjp(_permute_rows_fwd, _permute_rows_bwd)


def _embed_single_qubit(gate_2x2: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    eye = np.eye(2, dtype=gate_2x2.dtype)
    factors = [gate_2x2 if q == qubit else eye for q in range(num_qubits)]
    out = factors[0]
    for f in factors[1:]:
        out = np.kron(out, f)
    return out


def _embed_two_qubit(gate_4x4: np.ndarray, ctrl: int, tgt: int, num_qubits: int) -> np.ndarray:
    d = 2**num_qubits
    full = np.zeros((d, d), dtype=gate_4x4.dtype)
    for col in range(d):
        bits = [(col >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        two_q_in = (bits[ctrl] << 1) | bits[tgt]
        for two_q_out in range(4):
            amp = gate_4x4[two_q_out, two_q_in]
            if abs(amp) < 1e-15:
                continue
            new_bits = bits.copy()
            new_bits[ctrl] = (two_q_out >> 1) & 1
            new_bits[tgt] = two_q_out & 1
            row = sum(new_bits[q] << (num_qubits - 1 - q) for q in range(num_qubits))
            full[row, col] += amp
    return full


def _build_qubit_perms(num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-qubit row permutations: groups rows by the value of bit ``k``."""
    d = 2**num_qubits
    perm = np.zeros((num_qubits, d), dtype=np.int32)
    inv = np.zeros_like(perm)
    for k in range(num_qubits):
        zeros = [i for i in range(d) if ((i >> (num_qubits - 1 - k)) & 1) == 0]
        ones = [i for i in range(d) if ((i >> (num_qubits - 1 - k)) & 1) == 1]
        perm[k] = np.asarray(zeros + ones, dtype=np.int32)
        for new_pos, old_idx in enumerate(perm[k]):
            inv[k, old_idx] = new_pos
    return perm, inv


def _build_pair_perms(
    cnot_pairs: list[tuple[int, int]],
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(ctrl,tgt)-pair row permutations: groups rows into 4 (cb, tb) classes."""
    d = 2**num_qubits
    if not cnot_pairs:
        return (
            np.arange(d, dtype=np.int32)[None, :],
            np.arange(d, dtype=np.int32)[None, :],
        )
    n_pairs = len(cnot_pairs)
    perm = np.zeros((n_pairs, d), dtype=np.int32)
    inv = np.zeros_like(perm)
    for pi, (c_k, t_k) in enumerate(cnot_pairs):
        groups: list[list[int]] = [[], [], [], []]
        for i in range(d):
            cb = (i >> (num_qubits - 1 - c_k)) & 1
            tb = (i >> (num_qubits - 1 - t_k)) & 1
            groups[cb * 2 + tb].append(i)
        perm[pi] = np.asarray(groups[0] + groups[1] + groups[2] + groups[3], dtype=np.int32)
        for new_pos, old_idx in enumerate(perm[pi]):
            inv[pi, old_idx] = new_pos
    return perm, inv


class CircuitEvaluator:
    """Compile token metadata into JIT functions for fidelity and angle gradients."""

    def __init__(
        self,
        u_target: np.ndarray,
        num_qubits: int,
        pool_token_names: list[str],
        max_gates: int,
    ):
        self.num_qubits = num_qubits
        self.max_gates = max_gates
        self.vocab_size = len(pool_token_names)

        backend = jax.default_backend()
        is_gpu = backend in ("gpu", "cuda")
        self._real_dtype = jnp.float32 if is_gpu else jnp.float64
        self._complex_dtype = jnp.complex64 if is_gpu else jnp.complex128
        self._np_real = np.float32 if is_gpu else np.float64
        self._np_complex = np.complex64 if is_gpu else np.complex128

        self.u_target = jnp.asarray(u_target, dtype=self._complex_dtype)

        self._cnot_pairs = [
            (c, t) for c in range(num_qubits) for t in range(num_qubits) if c != t
        ]
        self._cnot_pair_to_idx = {pair: i for i, pair in enumerate(self._cnot_pairs)}

        perm_1q, inv_perm_1q = _build_qubit_perms(num_qubits)
        perm_2q, inv_perm_2q = _build_pair_perms(self._cnot_pairs, num_qubits)
        self._perm_1q = jnp.asarray(perm_1q, dtype=jnp.int32)
        self._inv_perm_1q = jnp.asarray(inv_perm_1q, dtype=jnp.int32)
        self._perm_2q = jnp.asarray(perm_2q, dtype=jnp.int32)
        self._inv_perm_2q = jnp.asarray(inv_perm_2q, dtype=jnp.int32)

        self._eye_2 = jnp.eye(2, dtype=self._complex_dtype)
        self._sx_2x2 = jnp.asarray(_SX, dtype=self._complex_dtype)
        self._cnot_4x4 = jnp.asarray(_CNOT, dtype=self._complex_dtype)
        self._pauli_xyz_2x2 = jnp.stack(
            [
                jnp.asarray(_PAULI_X, dtype=self._complex_dtype),
                jnp.asarray(_PAULI_Y, dtype=self._complex_dtype),
                jnp.asarray(_PAULI_Z, dtype=self._complex_dtype),
            ],
            axis=0,
        )
        self._imag_unit = jnp.asarray(1.0j, dtype=self._complex_dtype)

        gate_type_ids = np.full((self.vocab_size,), GATE_TYPE_NOOP, dtype=np.int32)
        qubit0 = np.zeros((self.vocab_size,), dtype=np.int32)
        cnot_pair = np.zeros((self.vocab_size,), dtype=np.int32)
        is_parametric = np.zeros((self.vocab_size,), dtype=bool)
        for tok_id, name in enumerate(pool_token_names):
            spec = parse_gate_name(name)
            gate_type_ids[tok_id] = _GATE_TYPE_TO_ID[spec.gate_type]
            qubit0[tok_id] = spec.qubits[0]
            is_parametric[tok_id] = spec.is_parametric
            if spec.gate_type == "CNOT":
                cnot_pair[tok_id] = self._cnot_pair_to_idx[spec.qubits]
        self._tok_gate_type = jnp.asarray(gate_type_ids, dtype=jnp.int32)
        self._tok_qubit0 = jnp.asarray(qubit0, dtype=jnp.int32)
        self._tok_cnot_pair = jnp.asarray(cnot_pair, dtype=jnp.int32)
        self.token_is_parametric_np = is_parametric

        self._build_jit_fns()

    @property
    def cnot_pairs(self) -> list[tuple[int, int]]:
        return list(self._cnot_pairs)

    def _apply_1q(self, U: jax.Array, G: jax.Array, qubit: jax.Array) -> jax.Array:
        d = 2 ** self.num_qubits
        h = d // 2
        perm = self._perm_1q[qubit]
        inv = self._inv_perm_1q[qubit]
        Up = _permute_rows(U, perm, inv)
        U0, U1 = Up[:h], Up[h:]
        Y0 = G[0, 0] * U0 + G[0, 1] * U1
        Y1 = G[1, 0] * U0 + G[1, 1] * U1
        Y = jnp.concatenate([Y0, Y1], axis=0)
        return _permute_rows(Y, inv, perm)

    def _apply_2q(self, U: jax.Array, C: jax.Array, pair: jax.Array) -> jax.Array:
        d = 2 ** self.num_qubits
        q = d // 4
        perm = self._perm_2q[pair]
        inv = self._inv_perm_2q[pair]
        Up = _permute_rows(U, perm, inv)
        U00, U01, U10, U11 = Up[:q], Up[q:2 * q], Up[2 * q:3 * q], Up[3 * q:]
        Y00 = C[0, 0] * U00 + C[0, 1] * U01 + C[0, 2] * U10 + C[0, 3] * U11
        Y01 = C[1, 0] * U00 + C[1, 1] * U01 + C[1, 2] * U10 + C[1, 3] * U11
        Y10 = C[2, 0] * U00 + C[2, 1] * U01 + C[2, 2] * U10 + C[2, 3] * U11
        Y11 = C[3, 0] * U00 + C[3, 1] * U01 + C[3, 2] * U10 + C[3, 3] * U11
        Y = jnp.concatenate([Y00, Y01, Y10, Y11], axis=0)
        return _permute_rows(Y, inv, perm)

    def _build_unitary(self, token_ids: jax.Array, angles: jax.Array) -> jax.Array:
        """Compose one full-length token row into a unitary matrix."""
        d = 2 ** self.num_qubits
        gate_type_ids = self._tok_gate_type[token_ids]
        qubit0 = self._tok_qubit0[token_ids]
        cnot_pair = self._tok_cnot_pair[token_ids]

        half = angles.astype(self._real_dtype) * jnp.asarray(0.5, dtype=self._real_dtype)
        cos_h = jnp.cos(half).astype(self._complex_dtype)
        sin_h = jnp.sin(half).astype(self._complex_dtype)
        axis_idx = jnp.where(gate_type_ids < 3, gate_type_ids, 0)
        pauli_2 = self._pauli_xyz_2x2[axis_idx]
        rot_2 = (
            cos_h[:, None, None] * self._eye_2
            - self._imag_unit * sin_h[:, None, None] * pauli_2
        )
        is_rot = (gate_type_ids < 3)[:, None, None]
        gate_2 = jnp.where(is_rot, rot_2, self._sx_2x2)
        is_1q = gate_type_ids < 4
        is_2q = gate_type_ids == 4

        U0 = jnp.eye(d, dtype=self._complex_dtype)

        def step(U, xs):
            g2, q0, cp, m1, m2 = xs
            U_after_1q = self._apply_1q(U, g2, q0)
            U_after_2q = self._apply_2q(U, self._cnot_4x4, cp)
            new_U = jnp.where(m1, U_after_1q, jnp.where(m2, U_after_2q, U))
            return new_U, None

        U_final, _ = jax.lax.scan(
            step, U0, (gate_2, qubit0, cnot_pair, is_1q, is_2q),
            unroll=4,
        )
        return U_final

    def _process_fidelity(self, U: jax.Array) -> jax.Array:
        d = U.shape[0]
        overlap = jnp.sum(jnp.conjugate(self.u_target) * U)
        return jnp.clip((jnp.abs(overlap) ** 2) / (d ** 2), 0.0, 1.0)

    def _linear_trace_cost(self, U: jax.Array) -> jax.Array:
        d = U.shape[0]
        overlap = jnp.sum(jnp.conjugate(self.u_target) * U)
        return 1.0 - jnp.abs(overlap) / d

    def _build_jit_fns(self) -> None:
        def fidelity_one(token_ids, angles):
            return self._process_fidelity(self._build_unitary(token_ids, angles))

        def loss_one(angles, token_ids):
            return 1.0 - fidelity_one(token_ids, angles)

        def linear_loss_one(angles, token_ids):
            return self._linear_trace_cost(self._build_unitary(token_ids, angles))

        self._fidelity_one = jax.jit(fidelity_one)
        self._loss_one = jax.jit(loss_one)
        self._linear_loss_one = jax.jit(linear_loss_one)
        self._fidelity_batch = jax.jit(jax.vmap(fidelity_one, in_axes=(0, 0)))
        self._loss_value_and_grad_batch = jax.jit(
            jax.vmap(jax.value_and_grad(loss_one), in_axes=(0, 0))
        )
        self._linear_loss_value_and_grad_batch = jax.jit(
            jax.vmap(jax.value_and_grad(linear_loss_one), in_axes=(0, 0))
        )

    def fidelity_batch(
        self,
        token_ids_batch: np.ndarray,
        angles_batch: np.ndarray,
    ) -> np.ndarray:
        """Return process fidelity for each row in a token/angle batch."""
        if token_ids_batch.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        token_jax = jnp.asarray(token_ids_batch, dtype=jnp.int32)
        angle_jax = jnp.asarray(angles_batch, dtype=self._real_dtype)
        f = self._fidelity_batch(token_jax, angle_jax)
        return np.asarray(f, dtype=np.float32)

    def loss_value_and_grad_batch(
        self,
        token_ids_batch: jax.Array,
        angles_batch: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Per-sample (loss, d_loss/d_angles) for a batch."""
        return self._loss_value_and_grad_batch(angles_batch, token_ids_batch)

    def parametric_mask(self, token_ids: np.ndarray) -> np.ndarray:
        """Boolean mask: which positions hold a parametric (rotation) gate."""
        return self.token_is_parametric_np[np.asarray(token_ids, dtype=np.int32)]
