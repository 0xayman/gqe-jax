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
    ):
        self.num_qubits = num_qubits
        self.steps = steps
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.top_k = top_k
        self.max_gates = max_gates
        self.num_restarts = num_restarts
        self.u_target_jax = jnp.asarray(u_target, dtype=jnp.complex128)

        self._identity = jnp.eye(2**num_qubits, dtype=jnp.complex128)
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
            )
        else:
            self._cnot_matrices = self._identity[None, ...]
        self._single_embed_cases = tuple(
            self._make_single_embed_case(qubit) for qubit in range(num_qubits)
        )
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

    def _make_single_embed_case(self, qubit: int):
        return lambda gate_2x2: _embed_single(gate_2x2, qubit, self.num_qubits)

    def _embed_single_dynamic(self, gate_2x2: jax.Array, qubit: jax.Array) -> jax.Array:
        return jax.lax.switch(qubit, self._single_embed_cases, gate_2x2)

    def _build_gate_matrix(
        self,
        gate_type_id: jax.Array,
        qubit0: jax.Array,
        cnot_pair: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        def rx_branch(operand):
            theta, qubit0, _ = operand
            return self._embed_single_dynamic(_rx(theta), qubit0)

        def ry_branch(operand):
            theta, qubit0, _ = operand
            return self._embed_single_dynamic(_ry(theta), qubit0)

        def rz_branch(operand):
            theta, qubit0, _ = operand
            return self._embed_single_dynamic(_rz(theta), qubit0)

        def sx_branch(operand):
            _, qubit0, _ = operand
            return self._embed_single_dynamic(_SX, qubit0)

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

    def _build_full_angle_vector(
        self,
        gate_specs: list,
        compact_params: jax.Array | None = None,
    ) -> jax.Array:
        angles = np.zeros((self.max_gates,), dtype=np.float64)
        param_idx = 0
        compact = None if compact_params is None else np.asarray(compact_params, dtype=np.float64)
        for gate_idx, spec in enumerate(gate_specs):
            if spec.is_parametric:
                if compact is None:
                    angles[gate_idx] = spec.initial_angle
                else:
                    angles[gate_idx] = compact[param_idx]
                    param_idx += 1
        return jnp.asarray(angles, dtype=jnp.float64)

    def _compact_param_angles(self, gate_specs: list, full_angles: jax.Array) -> jax.Array:
        full_angles_np = np.asarray(full_angles, dtype=np.float64)
        params = [full_angles_np[idx] for idx, spec in enumerate(gate_specs) if spec.is_parametric]
        if not params:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.asarray(params, dtype=jnp.float64)

    def _optimize_angles(
        self,
        gate_specs: list,
        initial_angles: jax.Array,
        rng_key: jax.Array,
        *,
        num_restarts: int,
    ) -> tuple[float, jax.Array, jax.Array]:
        gate_type_ids, qubit0, cnot_pair = self._encode_gate_specs(gate_specs)
        if self.optimizer_type == "lbfgs":
            optimize_impl = self._optimize_lbfgs
        elif self.optimizer_type == "adam":
            optimize_impl = self._optimize_adam
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type!r}")

        best_fidelity = -1.0
        best_params = initial_angles
        key = rng_key
        parametric_mask = gate_type_ids < _GATE_TYPE_TO_ID["SX"]
        for restart in range(max(1, num_restarts)):
            if restart == 0:
                start = initial_angles
            else:
                key, subkey = jax.random.split(key)
                sampled = jax.random.uniform(
                    subkey,
                    shape=initial_angles.shape,
                    minval=-jnp.pi,
                    maxval=jnp.pi,
                    dtype=jnp.float64,
                )
                start = jnp.where(parametric_mask, sampled, 0.0)

            params = optimize_impl(start, gate_type_ids, qubit0, cnot_pair)
            fidelity = float(
                self._fidelity_fn(params, gate_type_ids, qubit0, cnot_pair).astype(jnp.float64)
            )
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_params = params

        return best_fidelity, best_params, key

    def optimize_circuit(self, token_names: list[str], rng_key: jax.Array) -> tuple[float, jax.Array]:
        fidelity, _, _, rng_key = self.optimize_circuit_with_params(token_names, rng_key)
        return fidelity, rng_key

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
