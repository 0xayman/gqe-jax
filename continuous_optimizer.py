from __future__ import annotations

from dataclasses import dataclass

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax

_TWO_PI = 2 * np.pi
_DEFAULT_ANGLE = np.pi / 4

_GATE_TYPE_TO_ID = {
    "RX": 0,
    "RY": 1,
    "RZ": 2,
    "SX": 3,
    "CNOT": 4,
    "NOOP": 5,
}


def _normalize_angle(theta: float) -> float:
    return ((theta + np.pi) % _TWO_PI) - np.pi


@dataclass(frozen=True)
class GateSpec:
    gate_type: str
    qubits: tuple
    initial_angle: float
    is_parametric: bool


def parse_gate_spec(token_name: str) -> GateSpec:
    if token_name.startswith(("RX", "RY", "RZ")):
        parts = token_name.split("_")
        axis = parts[0]
        qubit = int(parts[1][1:])
        return GateSpec(axis, (qubit,), _DEFAULT_ANGLE, is_parametric=True)
    if token_name.startswith("SX"):
        qubit = int(token_name.split("_")[1][1:])
        return GateSpec("SX", (qubit,), 0.0, is_parametric=False)
    if token_name.startswith("CNOT"):
        parts = token_name.split("_")
        ctrl = int(parts[1][1:])
        tgt = int(parts[2][1:])
        return GateSpec("CNOT", (ctrl, tgt), 0.0, is_parametric=False)
    raise ValueError(f"Unknown token name: {token_name!r}")


def _rz(theta: jax.Array) -> jax.Array:
    h = theta / 2
    top = jnp.exp(-1j * h)
    bot = jnp.exp(1j * h)
    return jnp.asarray([[top, 0.0j], [0.0j, bot]], dtype=jnp.complex128)


def _rx(theta: jax.Array) -> jax.Array:
    h = theta / 2
    c = jnp.cos(h)
    s = -1j * jnp.sin(h)
    return jnp.asarray([[c, s], [s, c]], dtype=jnp.complex128)


def _ry(theta: jax.Array) -> jax.Array:
    h = theta / 2
    c = jnp.cos(h)
    s = jnp.sin(h)
    return jnp.asarray([[c, -s], [s, c]], dtype=jnp.complex128)


_ROTATION_BUILDERS = {"RX": _rx, "RY": _ry, "RZ": _rz}
_SX = jnp.asarray(
    [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]],
    dtype=jnp.complex128,
)
_CNOT = jnp.asarray(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=jnp.complex128,
)


def _embed_single(gate_2x2: jax.Array, qubit: int, num_qubits: int) -> jax.Array:
    eye = jnp.eye(2, dtype=jnp.complex128)
    factors = [gate_2x2 if q == qubit else eye for q in range(num_qubits)]
    result = factors[0]
    for factor in factors[1:]:
        result = jnp.kron(result, factor)
    return result


def _embed_cnot(ctrl: int, tgt: int, num_qubits: int) -> jax.Array:
    d = 2**num_qubits
    cnot = np.asarray(_CNOT)
    if num_qubits == 2 and ctrl == 0 and tgt == 1:
        return _CNOT
    if num_qubits == 2 and ctrl == 1 and tgt == 0:
        swap = jnp.asarray(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=jnp.complex128,
        )
        return swap @ _CNOT @ swap

    full = np.zeros((d, d), dtype=np.complex128)
    for col in range(d):
        bits = [(col >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        two_q_in = (bits[ctrl] << 1) | bits[tgt]
        for two_q_out in range(4):
            amp = cnot[two_q_out, two_q_in]
            if abs(amp) < 1e-15:
                continue
            new_bits = bits.copy()
            new_bits[ctrl] = (two_q_out >> 1) & 1
            new_bits[tgt] = two_q_out & 1
            row = sum(new_bits[q] << (num_qubits - 1 - q) for q in range(num_qubits))
            full[row, col] += amp
    return jnp.asarray(full, dtype=jnp.complex128)


def build_circuit_unitary(
    gate_specs: list,
    params: jax.Array,
    num_qubits: int,
) -> jax.Array:
    d = 2**num_qubits
    u = jnp.eye(d, dtype=jnp.complex128)
    param_idx = 0
    for spec in gate_specs:
        if spec.is_parametric:
            theta = params[param_idx]
            param_idx += 1
            gate_full = _embed_single(
                _ROTATION_BUILDERS[spec.gate_type](theta),
                spec.qubits[0],
                num_qubits,
            )
        elif spec.gate_type == "SX":
            gate_full = _embed_single(_SX, spec.qubits[0], num_qubits)
        elif spec.gate_type == "CNOT":
            gate_full = _embed_cnot(spec.qubits[0], spec.qubits[1], num_qubits)
        else:
            raise ValueError(f"Unknown gate type: {spec.gate_type!r}")
        u = gate_full @ u
    return u


def process_fidelity_jax(u_target: jax.Array, u_circuit: jax.Array) -> jax.Array:
    d = u_target.shape[0]
    overlap = jnp.trace(jnp.conjugate(u_target).T @ u_circuit)
    return (jnp.abs(overlap) ** 2) / (d**2)


def simplify_gate_sequence(
    gate_specs: list,
    params: jax.Array,
    threshold: float = 1e-3,
) -> tuple[list, jax.Array]:
    param_list = np.asarray(params, dtype=np.float64).tolist()
    pidx = 0
    pairs: list[tuple] = []
    for spec in gate_specs:
        if spec.is_parametric:
            pairs.append((spec, param_list[pidx]))
            pidx += 1
        else:
            pairs.append((spec, None))

    changed = True
    while changed:
        changed = False
        out: list[tuple] = []
        i = 0
        while i < len(pairs):
            spec, angle = pairs[i]
            if spec.is_parametric:
                norm = _normalize_angle(angle)
                if abs(norm) < threshold:
                    changed = True
                    i += 1
                    continue
                if (
                    out
                    and out[-1][0].is_parametric
                    and out[-1][0].gate_type == spec.gate_type
                    and out[-1][0].qubits == spec.qubits
                ):
                    prev_spec, prev_angle = out.pop()
                    merged = _normalize_angle(prev_angle + norm)
                    changed = True
                    if abs(merged) < threshold:
                        i += 1
                        continue
                    out.append((prev_spec, merged))
                    i += 1
                    continue
                out.append((spec, norm))
            else:
                if (
                    spec.gate_type == "CNOT"
                    and out
                    and out[-1][0].gate_type == "CNOT"
                    and out[-1][0].qubits == spec.qubits
                ):
                    out.pop()
                    changed = True
                    i += 1
                    continue
                out.append((spec, None))
            i += 1
        pairs = out

    new_specs = [pair[0] for pair in pairs]
    new_angles = [pair[1] for pair in pairs if pair[0].is_parametric]
    if new_angles:
        new_params = jnp.asarray(new_angles, dtype=jnp.float64)
    else:
        new_params = jnp.zeros((0,), dtype=jnp.float64)
    return new_specs, new_params


class ContinuousOptimizer:
    def __init__(
        self,
        u_target,
        num_qubits,
        steps,
        lr,
        optimizer_type,
        top_k,
        max_gates: int,
        num_restarts: int = 1,
        pool_token_names: list[str] | None = None,
        fast_runtime: bool = False,
    ):
        self.num_qubits = num_qubits
        self.steps = steps
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.top_k = top_k
        self.max_gates = max_gates
        self.num_restarts = num_restarts
        self.fast_runtime = fast_runtime
        self._real_dtype = jnp.float32 if fast_runtime else jnp.float64
        self._complex_dtype = jnp.complex64 if fast_runtime else jnp.complex128
        self._np_real_dtype = np.float32 if fast_runtime else np.float64
        self._np_complex_dtype = np.complex64 if fast_runtime else np.complex128
        self._imag_unit = jnp.asarray(1.0j, dtype=self._complex_dtype)
        self.u_target_jax = jnp.asarray(u_target, dtype=self._complex_dtype)

        self._identity = jnp.eye(2**num_qubits, dtype=self._complex_dtype)
        self._pauli_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=self._complex_dtype)
        self._pauli_y = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=self._complex_dtype)
        self._pauli_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=self._complex_dtype)
        self._cnot_pairs = [
            (ctrl, tgt)
            for ctrl in range(num_qubits)
            for tgt in range(num_qubits)
            if ctrl != tgt
        ]
        self._cnot_pair_to_idx = {pair: i for i, pair in enumerate(self._cnot_pairs)}
        if self._cnot_pairs:
            self._cnot_matrices = jnp.stack(
                [_embed_cnot(ctrl, tgt, num_qubits) for ctrl, tgt in self._cnot_pairs],
                axis=0,
            ).astype(self._complex_dtype)
        else:
            self._cnot_matrices = self._identity[None, ...]
        self._sx_matrices = jnp.stack(
            [_embed_single(_SX, qubit, num_qubits) for qubit in range(num_qubits)],
            axis=0,
        ).astype(self._complex_dtype)
        self._pauli_x_matrices = jnp.stack(
            [_embed_single(self._pauli_x, qubit, num_qubits) for qubit in range(num_qubits)],
            axis=0,
        ).astype(self._complex_dtype)
        self._pauli_y_matrices = jnp.stack(
            [_embed_single(self._pauli_y, qubit, num_qubits) for qubit in range(num_qubits)],
            axis=0,
        ).astype(self._complex_dtype)
        self._pauli_z_matrices = jnp.stack(
            [_embed_single(self._pauli_z, qubit, num_qubits) for qubit in range(num_qubits)],
            axis=0,
        ).astype(self._complex_dtype)
        self._pool_gate_specs: tuple[GateSpec, ...] | None = None
        self._pool_gate_type_ids: jax.Array | None = None
        self._pool_qubit0: jax.Array | None = None
        self._pool_cnot_pair: jax.Array | None = None
        self._pool_initial_angles: jax.Array | None = None
        if pool_token_names is not None:
            self._initialize_pool_encoding(pool_token_names)
        self._lbfgs = optax.lbfgs(learning_rate=lr)
        self._adam = optax.adam(learning_rate=lr)

        def loss_fn(angles, gate_type_ids, qubit0, cnot_pair):
            u = self._build_circuit_from_encoded(angles, gate_type_ids, qubit0, cnot_pair)
            return 1.0 - process_fidelity_jax(self.u_target_jax, u)

        self._loss_fn = loss_fn
        self._loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
        self._fidelity_fn = jax.jit(
            lambda angles, gate_type_ids, qubit0, cnot_pair: 1.0
            - loss_fn(angles, gate_type_ids, qubit0, cnot_pair)
        )

        def optimize_adam(init_angles, gate_type_ids, qubit0, cnot_pair):
            opt_state = self._adam.init(init_angles)

            def body(_, carry):
                angles, state = carry
                value, grads = jax.value_and_grad(loss_fn)(
                    angles, gate_type_ids, qubit0, cnot_pair
                )
                del value
                updates, state = self._adam.update(grads, state, angles)
                angles = optax.apply_updates(angles, updates)
                return angles, state

            return jax.lax.fori_loop(
                0,
                self.steps,
                body,
                (init_angles, opt_state),
            )[0]

        def optimize_lbfgs(init_angles, gate_type_ids, qubit0, cnot_pair):
            value_fn = lambda angles: loss_fn(angles, gate_type_ids, qubit0, cnot_pair)
            value_and_grad = optax.value_and_grad_from_state(value_fn)
            opt_state = self._lbfgs.init(init_angles)

            def body(_, carry):
                angles, state = carry
                value, grads = value_and_grad(angles, state=state)
                updates, state = self._lbfgs.update(
                    grads,
                    state,
                    angles,
                    value=value,
                    grad=grads,
                    value_fn=value_fn,
                )
                angles = optax.apply_updates(angles, updates)
                return angles, state

            return jax.lax.fori_loop(
                0,
                self.steps,
                body,
                (init_angles, opt_state),
            )[0]

        self._optimize_adam = jax.jit(optimize_adam)
        self._optimize_lbfgs = jax.jit(optimize_lbfgs)
        self._optimize_adam_batch = jax.jit(jax.vmap(optimize_adam, in_axes=(0, 0, 0, 0)))
        self._optimize_lbfgs_batch = jax.jit(jax.vmap(optimize_lbfgs, in_axes=(0, 0, 0, 0)))
        self._fidelity_batch_fn = jax.jit(
            jax.vmap(
                lambda angles, gate_type_ids, qubit0, cnot_pair: 1.0
                - loss_fn(angles, gate_type_ids, qubit0, cnot_pair),
                in_axes=(0, 0, 0, 0),
            )
        )

    def _initialize_pool_encoding(self, pool_token_names: list[str]) -> None:
        gate_specs = tuple(parse_gate_spec(name) for name in pool_token_names)
        gate_type_ids = np.zeros((len(gate_specs),), dtype=np.int32)
        qubit0 = np.zeros((len(gate_specs),), dtype=np.int32)
        cnot_pair = np.zeros((len(gate_specs),), dtype=np.int32)
        initial_angles = np.zeros((len(gate_specs),), dtype=self._np_real_dtype)

        for idx, spec in enumerate(gate_specs):
            gate_type_ids[idx] = _GATE_TYPE_TO_ID[spec.gate_type]
            qubit0[idx] = spec.qubits[0]
            initial_angles[idx] = spec.initial_angle if spec.is_parametric else 0.0
            if spec.gate_type == "CNOT":
                cnot_pair[idx] = self._cnot_pair_to_idx[spec.qubits]

        self._pool_gate_specs = gate_specs
        self._pool_gate_type_ids = jnp.asarray(gate_type_ids, dtype=jnp.int32)
        self._pool_qubit0 = jnp.asarray(qubit0, dtype=jnp.int32)
        self._pool_cnot_pair = jnp.asarray(cnot_pair, dtype=jnp.int32)
        self._pool_initial_angles = jnp.asarray(initial_angles, dtype=self._real_dtype)

    def _build_gate_matrix(
        self,
        gate_type_id: jax.Array,
        qubit0: jax.Array,
        cnot_pair: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        def rx_branch(operand):
            theta, qubit0, _ = operand
            half = jnp.asarray(theta, dtype=self._real_dtype) / jnp.asarray(
                2.0, dtype=self._real_dtype
            )
            cos_half = jnp.asarray(jnp.cos(half), dtype=self._complex_dtype)
            sin_half = jnp.asarray(jnp.sin(half), dtype=self._complex_dtype)
            return (
                cos_half * self._identity
                - self._imag_unit * sin_half * self._pauli_x_matrices[qubit0]
            )

        def ry_branch(operand):
            theta, qubit0, _ = operand
            half = jnp.asarray(theta, dtype=self._real_dtype) / jnp.asarray(
                2.0, dtype=self._real_dtype
            )
            cos_half = jnp.asarray(jnp.cos(half), dtype=self._complex_dtype)
            sin_half = jnp.asarray(jnp.sin(half), dtype=self._complex_dtype)
            return (
                cos_half * self._identity
                - self._imag_unit * sin_half * self._pauli_y_matrices[qubit0]
            )

        def rz_branch(operand):
            theta, qubit0, _ = operand
            half = jnp.asarray(theta, dtype=self._real_dtype) / jnp.asarray(
                2.0, dtype=self._real_dtype
            )
            cos_half = jnp.asarray(jnp.cos(half), dtype=self._complex_dtype)
            sin_half = jnp.asarray(jnp.sin(half), dtype=self._complex_dtype)
            return (
                cos_half * self._identity
                - self._imag_unit * sin_half * self._pauli_z_matrices[qubit0]
            )

        def sx_branch(operand):
            _, qubit0, _ = operand
            return self._sx_matrices[qubit0]

        def cnot_branch(operand):
            _, _, cnot_pair = operand
            return self._cnot_matrices[cnot_pair]

        def noop_branch(operand):
            del operand
            return self._identity

        return jax.lax.switch(
            gate_type_id,
            (
                rx_branch,
                ry_branch,
                rz_branch,
                sx_branch,
                cnot_branch,
                noop_branch,
            ),
            (theta, qubit0, cnot_pair),
        )

    def _build_circuit_from_encoded(
        self,
        angles: jax.Array,
        gate_type_ids: jax.Array,
        qubit0: jax.Array,
        cnot_pair: jax.Array,
    ) -> jax.Array:
        def step(u, operands):
            gate_type_id, gate_qubit0, gate_cnot_pair, theta = operands
            gate_full = self._build_gate_matrix(
                gate_type_id,
                gate_qubit0,
                gate_cnot_pair,
                theta,
            )
            return gate_full @ u, None

        return jax.lax.scan(
            step,
            self._identity,
            (gate_type_ids, qubit0, cnot_pair, angles),
        )[0]

    def _encode_gate_specs(self, gate_specs: list) -> tuple[jax.Array, jax.Array, jax.Array]:
        gate_type_ids = np.full((self.max_gates,), _GATE_TYPE_TO_ID["NOOP"], dtype=np.int32)
        qubit0 = np.zeros((self.max_gates,), dtype=np.int32)
        cnot_pair = np.zeros((self.max_gates,), dtype=np.int32)
        for idx, spec in enumerate(gate_specs):
            gate_type_ids[idx] = _GATE_TYPE_TO_ID[spec.gate_type]
            qubit0[idx] = spec.qubits[0]
            if spec.gate_type == "CNOT":
                cnot_pair[idx] = self._cnot_pair_to_idx[spec.qubits]
        return (
            jnp.asarray(gate_type_ids, dtype=jnp.int32),
            jnp.asarray(qubit0, dtype=jnp.int32),
            jnp.asarray(cnot_pair, dtype=jnp.int32),
        )

    def _encode_gate_specs_batch(
        self,
        gate_specs_batch: list[list],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_size = len(gate_specs_batch)
        gate_type_ids = np.full(
            (batch_size, self.max_gates),
            _GATE_TYPE_TO_ID["NOOP"],
            dtype=np.int32,
        )
        qubit0 = np.zeros((batch_size, self.max_gates), dtype=np.int32)
        cnot_pair = np.zeros((batch_size, self.max_gates), dtype=np.int32)
        for batch_idx, gate_specs in enumerate(gate_specs_batch):
            for gate_idx, spec in enumerate(gate_specs):
                gate_type_ids[batch_idx, gate_idx] = _GATE_TYPE_TO_ID[spec.gate_type]
                qubit0[batch_idx, gate_idx] = spec.qubits[0]
                if spec.gate_type == "CNOT":
                    cnot_pair[batch_idx, gate_idx] = self._cnot_pair_to_idx[spec.qubits]
        return (
            jnp.asarray(gate_type_ids, dtype=jnp.int32),
            jnp.asarray(qubit0, dtype=jnp.int32),
            jnp.asarray(cnot_pair, dtype=jnp.int32),
        )

    def _build_full_angle_vector(
        self,
        gate_specs: list,
        compact_params: jax.Array | None = None,
    ) -> jax.Array:
        angles = np.zeros((self.max_gates,), dtype=self._np_real_dtype)
        param_idx = 0
        compact = (
            None
            if compact_params is None
            else np.asarray(compact_params, dtype=self._np_real_dtype)
        )
        for gate_idx, spec in enumerate(gate_specs):
            if spec.is_parametric:
                if compact is None:
                    angles[gate_idx] = spec.initial_angle
                else:
                    angles[gate_idx] = compact[param_idx]
                    param_idx += 1
        return jnp.asarray(angles, dtype=self._real_dtype)

    def _build_full_angle_vector_batch(self, gate_specs_batch: list[list]) -> jax.Array:
        angles = np.zeros((len(gate_specs_batch), self.max_gates), dtype=self._np_real_dtype)
        for batch_idx, gate_specs in enumerate(gate_specs_batch):
            for gate_idx, spec in enumerate(gate_specs):
                if spec.is_parametric:
                    angles[batch_idx, gate_idx] = spec.initial_angle
        return jnp.asarray(angles, dtype=self._real_dtype)

    def _compact_param_angles(self, gate_specs: list, full_angles: jax.Array) -> jax.Array:
        full_angles_np = np.asarray(full_angles, dtype=self._np_real_dtype)
        params = [full_angles_np[idx] for idx, spec in enumerate(gate_specs) if spec.is_parametric]
        if not params:
            return jnp.zeros((0,), dtype=self._real_dtype)
        return jnp.asarray(params, dtype=self._real_dtype)

    def _take_subkeys(
        self,
        rng_key: jax.Array,
        count: int,
    ) -> tuple[jax.Array, jax.Array]:
        if count <= 0:
            return jnp.zeros((0, *rng_key.shape), dtype=rng_key.dtype), rng_key
        keys = jax.random.split(rng_key, count + 1)
        return keys[1:], keys[0]

    def _sample_restart_angles_batch(
        self,
        subkeys: jax.Array,
        parametric_mask: jax.Array,
    ) -> jax.Array:
        sampled = jax.vmap(
            lambda key: jax.random.uniform(
                key,
                shape=(self.max_gates,),
                minval=-jnp.pi,
                maxval=jnp.pi,
                dtype=self._real_dtype,
            )
        )(subkeys)
        return jnp.where(parametric_mask, sampled, 0.0)

    def _optimize_angles(
        self,
        gate_specs: list,
        initial_angles: jax.Array,
        rng_key: jax.Array,
        *,
        num_restarts: int,
    ) -> tuple[float, jax.Array, jax.Array]:
        gate_type_ids, qubit0, cnot_pair = self._encode_gate_specs(gate_specs)
        fidelities, params_batch, key = self._optimize_angles_batch(
            initial_angles[None, :],
            gate_type_ids[None, :],
            qubit0[None, :],
            cnot_pair[None, :],
            rng_key,
            num_restarts=num_restarts,
        )
        return float(np.asarray(fidelities[0], dtype=np.float64)), params_batch[0], key

    def _optimize_angles_batch(
        self,
        initial_angles: jax.Array,
        gate_type_ids: jax.Array,
        qubit0: jax.Array,
        cnot_pair: jax.Array,
        rng_key: jax.Array,
        *,
        num_restarts: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.optimizer_type == "lbfgs":
            optimize_impl = self._optimize_lbfgs_batch
        elif self.optimizer_type == "adam":
            optimize_impl = self._optimize_adam_batch
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type!r}")

        batch_size = initial_angles.shape[0]
        key = rng_key
        parametric_mask = gate_type_ids < _GATE_TYPE_TO_ID["SX"]
        total_restarts = max(1, num_restarts)

        starts = initial_angles[None, :, :]
        if total_restarts > 1:
            subkeys, key = self._take_subkeys(key, batch_size * (total_restarts - 1))
            repeated_mask = jnp.broadcast_to(
                parametric_mask[None, :, :],
                (total_restarts - 1, batch_size, self.max_gates),
            ).reshape((batch_size * (total_restarts - 1), self.max_gates))
            random_starts = self._sample_restart_angles_batch(subkeys, repeated_mask).reshape(
                (total_restarts - 1, batch_size, self.max_gates)
            )
            starts = jnp.concatenate((starts, random_starts), axis=0)

        gate_type_ids_runs = jnp.broadcast_to(
            gate_type_ids[None, :, :],
            (total_restarts, batch_size, self.max_gates),
        )
        qubit0_runs = jnp.broadcast_to(
            qubit0[None, :, :],
            (total_restarts, batch_size, self.max_gates),
        )
        cnot_pair_runs = jnp.broadcast_to(
            cnot_pair[None, :, :],
            (total_restarts, batch_size, self.max_gates),
        )

        flat_starts = starts.reshape((total_restarts * batch_size, self.max_gates))
        flat_gate_type_ids = gate_type_ids_runs.reshape((total_restarts * batch_size, self.max_gates))
        flat_qubit0 = qubit0_runs.reshape((total_restarts * batch_size, self.max_gates))
        flat_cnot_pair = cnot_pair_runs.reshape((total_restarts * batch_size, self.max_gates))

        flat_params = optimize_impl(flat_starts, flat_gate_type_ids, flat_qubit0, flat_cnot_pair)
        flat_fidelities = self._fidelity_batch_fn(
            flat_params,
            flat_gate_type_ids,
            flat_qubit0,
            flat_cnot_pair,
        ).astype(jnp.float64)

        fidelity_runs = flat_fidelities.reshape((total_restarts, batch_size))
        param_runs = flat_params.reshape((total_restarts, batch_size, self.max_gates))
        best_restart_idx = jnp.argmax(fidelity_runs, axis=0)
        best_fidelity = jnp.max(fidelity_runs, axis=0)
        best_params = jnp.take_along_axis(
            param_runs,
            best_restart_idx[None, :, None],
            axis=0,
        ).squeeze(0)

        return best_fidelity, best_params, key

    def _optimize_encoded_batch(
        self,
        gate_specs_batch: list[list[GateSpec]] | None,
        initial_angles_batch: jax.Array,
        gate_type_ids_batch: jax.Array,
        qubit0_batch: jax.Array,
        cnot_pair_batch: jax.Array,
        rng_key: jax.Array,
        *,
        simplify: bool,
    ) -> tuple[np.ndarray, jax.Array]:
        has_params = np.asarray(
            np.any(np.asarray(gate_type_ids_batch) < _GATE_TYPE_TO_ID["SX"], axis=1),
            dtype=bool,
        )
        fidelities = np.zeros((initial_angles_batch.shape[0],), dtype=np.float64)

        no_param_indices = np.flatnonzero(~has_params)
        if no_param_indices.size > 0:
            fidelities[no_param_indices] = np.asarray(
                self._fidelity_batch_fn(
                    jnp.take(initial_angles_batch, no_param_indices, axis=0),
                    jnp.take(gate_type_ids_batch, no_param_indices, axis=0),
                    jnp.take(qubit0_batch, no_param_indices, axis=0),
                    jnp.take(cnot_pair_batch, no_param_indices, axis=0),
                ),
                dtype=np.float64,
            )

        param_indices = np.flatnonzero(has_params)
        if param_indices.size == 0:
            return fidelities, rng_key

        opt_fidelities, full_params_batch, rng_key = self._optimize_angles_batch(
            jnp.take(initial_angles_batch, param_indices, axis=0),
            jnp.take(gate_type_ids_batch, param_indices, axis=0),
            jnp.take(qubit0_batch, param_indices, axis=0),
            jnp.take(cnot_pair_batch, param_indices, axis=0),
            rng_key,
            num_restarts=self.num_restarts,
        )
        fidelities[param_indices] = np.asarray(opt_fidelities, dtype=np.float64)

        if not simplify:
            return fidelities, rng_key
        if gate_specs_batch is None:
            raise ValueError("gate_specs_batch is required when simplify=True")

        full_params_np = np.asarray(full_params_batch, dtype=np.float64)
        simplified_specs_batch: list[list[GateSpec]] = []
        simplified_full_init_batch: list[np.ndarray] = []
        simplified_global_indices: list[int] = []
        for local_idx, global_idx in enumerate(param_indices.tolist()):
            gate_specs = gate_specs_batch[global_idx]
            compact_params = self._compact_param_angles(gate_specs, full_params_np[local_idx])
            simp_specs, simp_params = simplify_gate_sequence(gate_specs, compact_params)
            if len(simp_specs) == len(gate_specs):
                continue

            simp_full_init = self._build_full_angle_vector(simp_specs, simp_params)
            if int(simp_params.size) > 0:
                simplified_specs_batch.append(simp_specs)
                simplified_full_init_batch.append(np.asarray(simp_full_init, dtype=np.float64))
                simplified_global_indices.append(global_idx)
                continue

            gate_type_ids, qubit0, cnot_pair = self._encode_gate_specs(simp_specs)
            simp_fidelity = float(
                self._fidelity_fn(simp_full_init, gate_type_ids, qubit0, cnot_pair).astype(
                    jnp.float64
                )
            )
            if simp_fidelity >= fidelities[global_idx] - 1e-6:
                fidelities[global_idx] = simp_fidelity

        if simplified_specs_batch:
            simp_initial_angles = jnp.asarray(
                np.stack(simplified_full_init_batch, axis=0),
                dtype=jnp.float64,
            )
            simp_gate_type_ids, simp_qubit0, simp_cnot_pair = self._encode_gate_specs_batch(
                simplified_specs_batch
            )
            simp_fidelities, _, rng_key = self._optimize_angles_batch(
                simp_initial_angles,
                simp_gate_type_ids,
                simp_qubit0,
                simp_cnot_pair,
                rng_key,
                num_restarts=1,
            )
            simp_fidelities_np = np.asarray(simp_fidelities, dtype=np.float64)
            for local_idx, global_idx in enumerate(simplified_global_indices):
                simp_fidelity = float(simp_fidelities_np[local_idx])
                if simp_fidelity >= fidelities[global_idx] - 1e-6:
                    fidelities[global_idx] = simp_fidelity

        return fidelities, rng_key

    def optimize_circuit(self, token_names: list[str], rng_key: jax.Array) -> tuple[float, jax.Array]:
        fidelity, _, _, rng_key = self.optimize_circuit_with_params(token_names, rng_key)
        return fidelity, rng_key

    def optimize_batch(
        self,
        token_names_batch: list[list[str]],
        rng_key: jax.Array,
        *,
        simplify: bool = True,
    ) -> tuple[np.ndarray, jax.Array]:
        if not token_names_batch:
            return np.zeros((0,), dtype=np.float64), rng_key

        gate_specs_batch = [
            [parse_gate_spec(name) for name in token_names] for token_names in token_names_batch
        ]
        initial_angles_batch = self._build_full_angle_vector_batch(gate_specs_batch)
        gate_type_ids_batch, qubit0_batch, cnot_pair_batch = self._encode_gate_specs_batch(
            gate_specs_batch
        )
        return self._optimize_encoded_batch(
            gate_specs_batch,
            initial_angles_batch,
            gate_type_ids_batch,
            qubit0_batch,
            cnot_pair_batch,
            rng_key,
            simplify=simplify,
        )

    def optimize_token_index_batch(
        self,
        token_ids_batch: np.ndarray | jax.Array,
        rng_key: jax.Array,
        *,
        simplify: bool = True,
    ) -> tuple[np.ndarray, jax.Array]:
        if self._pool_gate_specs is None:
            raise ValueError("optimize_token_index_batch requires pool_token_names at construction")

        token_ids_batch = np.asarray(token_ids_batch, dtype=np.int32)
        if token_ids_batch.size == 0:
            return np.zeros((0,), dtype=np.float64), rng_key

        token_ids_batch_jax = jnp.asarray(token_ids_batch, dtype=jnp.int32)
        initial_angles_batch = jnp.take(self._pool_initial_angles, token_ids_batch_jax, axis=0)
        gate_type_ids_batch = jnp.take(self._pool_gate_type_ids, token_ids_batch_jax, axis=0)
        qubit0_batch = jnp.take(self._pool_qubit0, token_ids_batch_jax, axis=0)
        cnot_pair_batch = jnp.take(self._pool_cnot_pair, token_ids_batch_jax, axis=0)
        gate_specs_batch = None
        if simplify:
            gate_specs_batch = [
                [self._pool_gate_specs[idx] for idx in row.tolist()] for row in token_ids_batch
            ]
        return self._optimize_encoded_batch(
            gate_specs_batch,
            initial_angles_batch,
            gate_type_ids_batch,
            qubit0_batch,
            cnot_pair_batch,
            rng_key,
            simplify=simplify,
        )

    def optimize_circuit_with_params(
        self,
        token_names: list[str],
        rng_key: jax.Array,
    ) -> tuple[float, list, jax.Array, jax.Array]:
        gate_specs = [parse_gate_spec(name) for name in token_names]
        initial_angles = self._build_full_angle_vector(gate_specs)

        if not any(spec.is_parametric for spec in gate_specs):
            gate_type_ids, qubit0, cnot_pair = self._encode_gate_specs(gate_specs)
            fidelity = float(
                self._fidelity_fn(initial_angles, gate_type_ids, qubit0, cnot_pair).astype(
                    jnp.float64
                )
            )
            return fidelity, gate_specs, jnp.zeros((0,), dtype=jnp.float64), rng_key

        fidelity, full_params, rng_key = self._optimize_angles(
            gate_specs,
            initial_angles,
            rng_key,
            num_restarts=self.num_restarts,
        )
        compact_params = self._compact_param_angles(gate_specs, full_params)

        simp_specs, simp_params = simplify_gate_sequence(gate_specs, compact_params)
        if len(simp_specs) == len(gate_specs):
            return fidelity, gate_specs, compact_params, rng_key

        simp_full_init = self._build_full_angle_vector(simp_specs, simp_params)
        if simp_params.size > 0:
            simp_fidelity, simp_full_params, rng_key = self._optimize_angles(
                simp_specs,
                simp_full_init,
                rng_key,
                num_restarts=1,
            )
            simp_params = self._compact_param_angles(simp_specs, simp_full_params)
        else:
            gate_type_ids, qubit0, cnot_pair = self._encode_gate_specs(simp_specs)
            simp_fidelity = float(
                self._fidelity_fn(simp_full_init, gate_type_ids, qubit0, cnot_pair).astype(
                    jnp.float64
                )
            )

        if simp_fidelity >= fidelity - 1e-6:
            return simp_fidelity, simp_specs, simp_params, rng_key
        return fidelity, gate_specs, compact_params, rng_key
