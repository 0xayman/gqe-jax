"""JAX-based GQE pipeline for transformer-guided circuit generation."""

from __future__ import annotations

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from cost import compilation_cost_batch_jax, process_fidelity
from data import BufferDataset, ReplayBuffer


def _unbiased_std(values: jax.Array) -> jax.Array:
    if values.size <= 1:
        return jnp.asarray(0.0, dtype=values.dtype)
    return jnp.std(values, ddof=1)


def _sequence_structure_metrics(
    token_ids: np.ndarray,
    num_qubits: int,
    token_qubit0: np.ndarray,
    token_qubit1: np.ndarray,
) -> tuple[int, int, int]:
    token_ids = np.asarray(token_ids, dtype=np.int32)
    if token_ids.size == 0:
        return 0, 0, 0

    qubit_depths = np.zeros((num_qubits,), dtype=np.int32)
    total_gates = 0
    cnot_count = 0
    for token_id in token_ids.tolist():
        q0 = int(token_qubit0[token_id])
        q1 = int(token_qubit1[token_id])
        total_gates += 1
        if q1 >= 0:
            layer = max(int(qubit_depths[q0]), int(qubit_depths[q1])) + 1
            qubit_depths[q0] = layer
            qubit_depths[q1] = layer
            cnot_count += 1
        else:
            qubit_depths[q0] += 1

    return int(np.max(qubit_depths)), total_gates, cnot_count


class Pipeline:
    def __init__(self, cfg, cost, pool, model, factory, u_target=None):
        self.cfg = cfg
        self.model = model
        self.factory = factory
        self.pool = pool
        self._cost = cost
        self.loss = self.factory.create_loss_fn(cfg)
        self.scheduler = self.factory.create_temperature_scheduler(cfg)
        self.ngates = cfg.model.max_gates_count
        self.num_samples = cfg.training.num_samples
        self.bos_token_id = 0
        self.gate_token_offset = 1
        self.vocab_size = len(pool) + self.gate_token_offset
        model_vocab_size = getattr(self.model, "vocab_size", None)
        if model_vocab_size is not None and int(model_vocab_size) != self.vocab_size:
            raise ValueError(
                f"Model vocab_size={model_vocab_size} does not match "
                f"expected {self.vocab_size} (pool size + BOS token)"
            )
        self.buffer = ReplayBuffer(size=cfg.buffer.max_size)
        self._starting_idx = self._make_starting_idx(self.num_samples)
        self._last_rollout_costs = np.zeros((0,), dtype=np.float32)
        self._last_rollout_fidelities = np.zeros((0,), dtype=np.float32)
        self._last_rollout_cnot_counts = np.zeros((0,), dtype=np.int32)
        self._last_rollout_indices = np.zeros((0, self.ngates + 1), dtype=np.int32)
        self._reference_params = None

        self.pool_names = ["<BOS>", *[name for name, _ in pool]]
        self.two_qubit_token_mask = np.zeros((self.vocab_size,), dtype=bool)
        self.token_qubit0 = np.zeros((self.vocab_size,), dtype=np.int32)
        self.token_qubit1 = np.full((self.vocab_size,), -1, dtype=np.int32)
        for idx, name in enumerate(self.pool_names[1:], start=self.gate_token_offset):
            parts = name.split("_")
            self.token_qubit0[idx] = int(parts[1][1:])
            if name.startswith("CNOT"):
                self.token_qubit1[idx] = int(parts[2][1:])
                self.two_qubit_token_mask[idx] = True
        self.pool_matrices = np.stack([matrix for _, matrix in pool], axis=0)
        self.pool_matrices_jax = jnp.asarray(self.pool_matrices, dtype=jnp.complex128)
        self.u_target_jax = (
            jnp.asarray(u_target, dtype=jnp.complex128) if u_target is not None else None
        )

        self.rng_key = jax.random.PRNGKey(cfg.training.seed)
        self.rng_key, params_key = jax.random.split(self.rng_key)
        dummy_input = jnp.zeros((1, self.ngates + 1), dtype=jnp.int32)
        dummy_mask = jnp.ones_like(dummy_input, dtype=bool)
        variables = self.model.init(
            {"params": params_key},
            dummy_input,
            attention_mask=dummy_mask,
            deterministic=True,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.training.grad_norm_clip),
            optax.adamw(
                learning_rate=cfg.training.lr,
                weight_decay=0.01,
            ),
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=tx,
        )

        self.continuous_optimizer = (
            self.factory.create_continuous_optimizer(cfg, u_target, pool)
            if u_target is not None
            else None
        )

        self._compile_functions()

    def _make_starting_idx(self, batch_size: int) -> np.ndarray:
        return np.full((batch_size, 1), self.bos_token_id, dtype=np.int32)

    def _decode_gate_token_ids(self, token_ids: np.ndarray) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int32)
        if token_ids.size == 0:
            return token_ids
        if np.any(token_ids < self.gate_token_offset) or np.any(token_ids >= self.vocab_size):
            raise ValueError("Gate token ids must be shifted by +1 and exclude the BOS token")
        return token_ids - self.gate_token_offset

    def _compile_functions(self) -> None:
        apply_fn = self.state.apply_fn
        clip_ratio = self.loss.clip_ratio
        batch_size = self.num_samples
        total_len = self.ngates + 1

        def model_logits(params, input_ids, attention_mask):
            return apply_fn(
                {"params": params},
                input_ids,
                attention_mask=attention_mask,
                deterministic=True,
            )

        def mask_invalid_action_logits(logits):
            return logits.at[..., self.bos_token_id].set(jnp.asarray(jnp.inf, dtype=logits.dtype))

        def selected_log_probs(logits, gate_indices, beta):
            aligned_logits = mask_invalid_action_logits(logits[:, : gate_indices.shape[1], :])
            log_probs = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
            return jnp.take_along_axis(log_probs, gate_indices[..., None], axis=-1).squeeze(-1)

        def sequence_log_probs(logits, gate_indices, beta):
            return jnp.sum(selected_log_probs(logits, gate_indices, beta), axis=-1)

        def calc_advantage(costs):
            return (jnp.mean(costs) - costs) / (_unbiased_std(costs) + 1e-8)

        def grpo_loss(params, reference_params, idx, costs, beta):
            attention_mask = jnp.ones_like(idx, dtype=bool)
            gate_indices = idx[:, 1:]
            advantages = jax.lax.stop_gradient(calc_advantage(costs))
            current_logits = model_logits(params, idx, attention_mask)
            reference_logits = model_logits(reference_params, idx, attention_mask)
            current_sequence_log_probs = sequence_log_probs(current_logits, gate_indices, beta)
            old_sequence_log_probs = jax.lax.stop_gradient(
                sequence_log_probs(reference_logits, gate_indices, beta)
            )
            ratio = jnp.exp(current_sequence_log_probs - old_sequence_log_probs)
            clipped_ratio = jnp.clip(
                ratio,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            )
            surrogate = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
            return -jnp.mean(surrogate)

        def grpo_step(state, reference_params, idx, costs, beta):
            loss_value, grads = jax.value_and_grad(grpo_loss)(
                state.params,
                reference_params,
                idx,
                costs,
                beta,
            )
            state = state.apply_gradients(grads=grads)
            return state, loss_value

        def generate_rollout(params, rng_key, beta):
            tokens = jnp.full((batch_size, total_len), self.bos_token_id, dtype=jnp.int32)
            active_mask = jnp.zeros((batch_size, total_len), dtype=bool)
            active_mask = active_mask.at[:, 0].set(True)

            def body(carry, step_idx):
                tokens, active_mask, rng = carry
                rng, sample_key = jax.random.split(rng)
                logits = model_logits(params, tokens, active_mask)
                step_logits = mask_invalid_action_logits(logits[:, step_idx, :])
                idx_next = jax.random.categorical(
                    sample_key,
                    -beta * step_logits,
                    axis=-1,
                ).astype(jnp.int32)
                tokens = tokens.at[:, step_idx + 1].set(idx_next)
                active_mask = active_mask.at[:, step_idx + 1].set(True)
                return (tokens, active_mask, rng), None

            (tokens, _, _), _ = jax.lax.scan(
                body,
                (tokens, active_mask, rng_key),
                jnp.arange(self.ngates, dtype=jnp.int32),
            )
            return tokens

        self._model_logits = jax.jit(model_logits)
        self._grpo_loss = jax.jit(grpo_loss)
        self._grpo_step = jax.jit(grpo_step)
        self._generate_rollout = jax.jit(generate_rollout)
        if self.u_target_jax is not None:
            self._discrete_cost_batch = jax.jit(
                lambda token_ids: compilation_cost_batch_jax(
                    self.u_target_jax,
                    self.pool_matrices_jax[token_ids],
                )
            )
        else:
            self._discrete_cost_batch = None

    def on_fit_start(self):
        self._starting_idx = self._make_starting_idx(self.num_samples)
        while len(self.buffer) < self.cfg.buffer.warmup_size:
            self.collect_rollout()

    def on_train_epoch_start(self):
        self.collect_rollout()

    def collect_rollout(self):
        idx_output = self.generate()
        costs, fidelities, cnot_counts = self.computeCost(idx_output[:, 1:], self.pool)
        for seq, cost_val in zip(idx_output, costs):
            self.buffer.push(seq, cost_val)
        self.scheduler.update(costs=costs)
        self._last_rollout_costs = np.asarray(costs, dtype=np.float32)
        self._last_rollout_fidelities = np.asarray(fidelities, dtype=np.float32)
        self._last_rollout_cnot_counts = np.asarray(cnot_counts, dtype=np.int32)
        self._last_rollout_indices = np.asarray(idx_output, dtype=np.int32)

    def set_cost(self, cost):
        self._cost = cost

    def sequence_structure_metrics(self, token_ids: np.ndarray) -> tuple[int, int, int]:
        return _sequence_structure_metrics(
            token_ids,
            self.cfg.target.num_qubits,
            self.token_qubit0,
            self.token_qubit1,
        )

    def _count_cnot_tokens(self, idx_output: np.ndarray) -> np.ndarray:
        return np.count_nonzero(self.two_qubit_token_mask[idx_output], axis=1).astype(np.int32)

    def _compute_discrete_metrics(
        self,
        pool_idx_output: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._discrete_cost_batch is not None:
            costs = self._discrete_cost_batch(jnp.asarray(pool_idx_output, dtype=jnp.int32))
            fidelities = 1.0 - costs
            return (
                np.asarray(costs, dtype=np.float32),
                np.asarray(fidelities, dtype=np.float32),
                cnot_counts,
            )

        if self.u_target_jax is None:
            raise RuntimeError("Discrete metric evaluation requires a target unitary")

        u_target = np.asarray(self.u_target_jax, dtype=np.complex128)
        d = u_target.shape[0]
        costs = []
        fidelities = []
        for row in pool_idx_output:
            gate_matrices = [self.pool[j][1] for j in row.tolist()]
            u_circuit = np.eye(d, dtype=np.complex128)
            for gate in gate_matrices:
                u_circuit = gate @ u_circuit
            fidelity = process_fidelity(u_target, u_circuit)
            fidelities.append(fidelity)
            costs.append(1.0 - fidelity)
        return (
            np.asarray(costs, dtype=np.float32),
            np.asarray(fidelities, dtype=np.float32),
            cnot_counts,
        )

    def computeCost(self, idx_output, pool, **kwargs):
        del pool, kwargs
        idx_output = np.asarray(idx_output, dtype=np.int32)
        cnot_counts = self._count_cnot_tokens(idx_output)
        pool_idx_output = self._decode_gate_token_ids(idx_output)

        if self.continuous_optimizer is None:
            return self._compute_discrete_metrics(pool_idx_output, cnot_counts)

        if self.continuous_optimizer.top_k > 0:
            discrete_costs, discrete_fidelities, _ = self._compute_discrete_metrics(
                pool_idx_output,
                cnot_counts,
            )
            top_indices = np.argsort(discrete_costs)[: self.continuous_optimizer.top_k]
            results = discrete_costs.astype(np.float64)
            fidelities = discrete_fidelities.astype(np.float64)
            selected_idx = pool_idx_output[top_indices]
            optimized_fidelities, self.rng_key = self.continuous_optimizer.optimize_token_index_batch(
                selected_idx,
                self.rng_key,
                simplify=False,
            )
            for sample_idx, fidelity in zip(top_indices.tolist(), optimized_fidelities.tolist()):
                fidelities[sample_idx] = fidelity
                results[sample_idx] = 1.0 - fidelity
        else:
            fidelities, self.rng_key = self.continuous_optimizer.optimize_token_index_batch(
                pool_idx_output,
                self.rng_key,
                simplify=False,
            )
            fidelities = np.asarray(fidelities, dtype=np.float64)
            results = 1.0 - fidelities

        return (
            np.asarray(results, dtype=np.float32),
            np.asarray(fidelities, dtype=np.float32),
            cnot_counts,
        )

    def train_dataloader(self):
        return BufferDataset(self.buffer, self.cfg.buffer.steps_per_epoch)

    def train_batch(self, idx: np.ndarray, costs: np.ndarray, batch_idx: int) -> float:
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        costs_jax = jnp.asarray(costs, dtype=jnp.float32)
        beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)

        if batch_idx == 0:
            self._reference_params = jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                self.state.params,
            )
        if self._reference_params is None:
            raise RuntimeError("GRPO reference parameters were not initialized on batch 0")
        self.state, loss_value = self._grpo_step(
            self.state,
            self._reference_params,
            idx_jax,
            costs_jax,
            beta,
        )
        return float(np.asarray(loss_value, dtype=np.float32))

    def generate(self, idx=None, ngates=None):
        if idx is None and (ngates is None or ngates == self.ngates):
            beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)
            self.rng_key, rollout_key = jax.random.split(self.rng_key)
            return np.asarray(
                self._generate_rollout(self.state.params, rollout_key, beta),
                dtype=np.int32,
            )

        idx = np.asarray(idx if idx is not None else self._starting_idx, dtype=np.int32)
        ngates = self.ngates if ngates is None else ngates
        beta = float(self.scheduler.get_inverse_temperature())
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)

        for _ in range(ngates):
            attention_mask = jnp.ones_like(idx_jax, dtype=bool)
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            logits = self._model_logits(self.state.params, idx_jax, attention_mask)
            step_logits = logits[:, -1, :].at[:, self.bos_token_id].set(
                jnp.asarray(jnp.inf, dtype=logits.dtype)
            )
            idx_next = jax.random.categorical(sample_key, -beta * step_logits, axis=-1).astype(
                jnp.int32
            )
            idx_jax = jnp.concatenate((idx_jax, idx_next[:, None]), axis=1)
        return np.asarray(idx_jax, dtype=np.int32)

    def logits(self, idx):
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        attention_mask = jnp.ones_like(idx_jax, dtype=bool)
        logits_base = self._model_logits(self.state.params, idx_jax, attention_mask)
        target_idx = idx_jax[:, 1:]
        aligned_logits = logits_base[:, : target_idx.shape[1], :]
        return jnp.take_along_axis(aligned_logits, target_idx[..., None], axis=-1).squeeze(-1)
