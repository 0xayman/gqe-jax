"""JAX-based GQE pipeline for transformer-guided circuit generation."""

from __future__ import annotations

from typing import Optional

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from circuit_grammar import CircuitGrammar
from cost import compilation_cost_batch_jax
from data import BufferDataset, ReplayBuffer


def _unbiased_std(values: jax.Array) -> jax.Array:
    return jnp.std(values, ddof=1)


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
        self.buffer = ReplayBuffer(size=cfg.buffer.max_size)
        self._starting_idx = np.zeros((self.num_samples, 1), dtype=np.int32)
        self._last_rollout_costs = np.zeros((0,), dtype=np.float32)
        self._old_log_probs = None
        self._advantages = None

        self.pool_names = [name for name, _ in pool]
        self.pool_matrices = np.stack([matrix for _, matrix in pool], axis=0)
        self.pool_matrices_jax = jnp.asarray(self.pool_matrices, dtype=jnp.complex128)
        self.u_target_jax = (
            jnp.asarray(u_target, dtype=jnp.complex128) if u_target is not None else None
        )

        self.rng_key = jax.random.PRNGKey(cfg.training.seed)
        self.rng_key, params_key, dropout_key = jax.random.split(self.rng_key, 3)
        dummy_input = jnp.zeros((1, self.ngates + 1), dtype=jnp.int32)
        dummy_mask = jnp.ones_like(dummy_input, dtype=bool)
        variables = self.model.init(
            {"params": params_key, "dropout": dropout_key},
            dummy_input,
            attention_mask=dummy_mask,
            deterministic=False,
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
        grammar_cfg = getattr(cfg, "grammar", None)
        self.grammar: Optional[CircuitGrammar] = (
            CircuitGrammar(cfg.target.num_qubits, pool)
            if grammar_cfg is not None and grammar_cfg.enabled
            else None
        )

        self._compile_functions()

    def _compile_functions(self) -> None:
        apply_fn = self.state.apply_fn
        clip_ratio = self.loss.clip_ratio
        grammar = self.grammar
        batch_size = self.num_samples
        total_len = self.ngates + 1

        def model_logits(params, input_ids, attention_mask, dropout_key):
            return apply_fn(
                {"params": params},
                input_ids,
                attention_mask=attention_mask,
                deterministic=False,
                rngs={"dropout": dropout_key},
            )

        def selected_log_probs(logits, gate_indices, beta):
            aligned_logits = logits[:, : gate_indices.shape[1], :]
            log_probs = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
            return jnp.take_along_axis(log_probs, gate_indices[..., None], axis=-1).squeeze(-1)

        def calc_advantage(costs):
            return (jnp.mean(costs) - costs) / (_unbiased_std(costs) + 1e-8)

        def first_batch_step(state, idx, costs, beta, dropout_key):
            attention_mask = jnp.ones_like(idx, dtype=bool)

            def loss_fn(params):
                logits = model_logits(params, idx, attention_mask, dropout_key)
                gate_indices = idx[:, 1:]
                current_log_probs = selected_log_probs(logits, gate_indices, beta)
                winner_loss = -jnp.mean(current_log_probs[jnp.argmin(costs)])
                advantages = calc_advantage(costs)
                identical = _unbiased_std(costs) == 0
                full_loss = winner_loss - jnp.mean(advantages[:, None])
                final_loss = jnp.where(identical, winner_loss, full_loss)
                return final_loss, (current_log_probs, advantages, identical)

            (loss_value, (current_log_probs, advantages, identical)), grads = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss_value, current_log_probs, advantages, identical

        def later_batch_step(
            state,
            idx,
            costs,
            beta,
            old_log_probs,
            advantages,
            dropout_key,
        ):
            attention_mask = jnp.ones_like(idx, dtype=bool)

            def loss_fn(params):
                logits = model_logits(params, idx, attention_mask, dropout_key)
                gate_indices = idx[:, 1:]
                current_log_probs = selected_log_probs(logits, gate_indices, beta)
                winner_loss = -jnp.mean(current_log_probs[jnp.argmin(costs)])
                ratio = jnp.exp(current_log_probs - old_log_probs)
                clipped_ratio = jnp.clip(
                    ratio,
                    1.0 - clip_ratio,
                    1.0 + clip_ratio,
                )
                ppo_loss = winner_loss - jnp.mean(clipped_ratio * advantages[:, None])
                final_loss = jnp.where(_unbiased_std(costs) == 0, winner_loss, ppo_loss)
                return final_loss

            loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss_value

        if grammar is None:

            def generate_rollout(params, rng_key, beta):
                tokens = jnp.zeros((batch_size, total_len), dtype=jnp.int32)
                active_mask = jnp.zeros((batch_size, total_len), dtype=bool)
                active_mask = active_mask.at[:, 0].set(True)

                def body(carry, step_idx):
                    tokens, active_mask, rng = carry
                    rng, dropout_key, sample_key = jax.random.split(rng, 3)
                    logits = model_logits(params, tokens, active_mask, dropout_key)
                    step_logits = logits[:, step_idx, :]
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

        else:

            def generate_rollout(params, rng_key, beta):
                tokens = jnp.zeros((batch_size, total_len), dtype=jnp.int32)
                active_mask = jnp.zeros((batch_size, total_len), dtype=bool)
                active_mask = active_mask.at[:, 0].set(True)
                grammar_state = grammar.initial_state(batch_size)

                def body(carry, step_idx):
                    tokens, active_mask, grammar_state, rng = carry
                    rng, dropout_key, sample_key = jax.random.split(rng, 3)
                    logits = model_logits(params, tokens, active_mask, dropout_key)
                    step_logits = logits[:, step_idx, :]
                    mask = grammar.forbidden_mask(grammar_state)
                    step_logits = jnp.where(mask, jnp.inf, step_logits)
                    idx_next = jax.random.categorical(
                        sample_key,
                        -beta * step_logits,
                        axis=-1,
                    ).astype(jnp.int32)
                    grammar_state = grammar.update(grammar_state, idx_next)
                    tokens = tokens.at[:, step_idx + 1].set(idx_next)
                    active_mask = active_mask.at[:, step_idx + 1].set(True)
                    return (tokens, active_mask, grammar_state, rng), None

                (tokens, _, _, _), _ = jax.lax.scan(
                    body,
                    (tokens, active_mask, grammar_state, rng_key),
                    jnp.arange(self.ngates, dtype=jnp.int32),
                )
                return tokens

        self._model_logits = jax.jit(model_logits)
        self._first_batch_step = jax.jit(first_batch_step)
        self._later_batch_step = jax.jit(later_batch_step)
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
        self._starting_idx = np.zeros((self.num_samples, 1), dtype=np.int32)
        while len(self.buffer) < self.cfg.buffer.warmup_size:
            self.collect_rollout()

    def on_train_epoch_start(self):
        self.collect_rollout()

    def collect_rollout(self):
        idx_output = self.generate()
        costs = self.computeCost(idx_output[:, 1:], self.pool)
        for seq, cost_val in zip(idx_output, costs):
            self.buffer.push(seq, cost_val)
        self.scheduler.update(costs=costs)
        self._last_rollout_costs = np.asarray(costs, dtype=np.float32)

    def set_cost(self, cost):
        self._cost = cost

    def _compute_discrete_costs(self, idx_output: np.ndarray) -> np.ndarray:
        if self._discrete_cost_batch is not None:
            costs = self._discrete_cost_batch(jnp.asarray(idx_output, dtype=jnp.int32))
            return np.asarray(costs, dtype=np.float32)

        results = []
        for row in idx_output:
            gate_matrices = [self.pool[j][1] for j in row.tolist()]
            results.append(self._cost(gate_matrices))
        return np.asarray(results, dtype=np.float32)

    def computeCost(self, idx_output, pool, **kwargs):
        idx_output = np.asarray(idx_output, dtype=np.int32)
        token_names_batch = [[pool[j][0] for j in row.tolist()] for row in idx_output]

        if self.continuous_optimizer is None:
            return self._compute_discrete_costs(idx_output)

        verbose = self.cfg.logging.verbose
        if self.continuous_optimizer.top_k > 0:
            discrete_costs = self._compute_discrete_costs(idx_output)
            top_indices = sorted(range(len(discrete_costs)), key=lambda i: discrete_costs[i])[
                : self.continuous_optimizer.top_k
            ]
            results = discrete_costs.astype(np.float64).tolist()
            best_f = 0.0
            width = len(str(max(1, self.continuous_optimizer.top_k)))
            for rank, sample_idx in enumerate(top_indices):
                fidelity, self.rng_key = self.continuous_optimizer.optimize_circuit(
                    token_names_batch[sample_idx],
                    self.rng_key,
                )
                results[sample_idx] = 1.0 - fidelity
                best_f = max(best_f, fidelity)
                if verbose:
                    print(
                        f"    angle opt [{rank + 1:{width}d}/{self.continuous_optimizer.top_k}]"
                        f"  f={fidelity:.6f}  best_f={best_f:.6f}"
                    )
        else:
            results = []
            best_f = 0.0
            total = len(token_names_batch)
            width = len(str(max(1, total)))
            for sample_idx, names in enumerate(token_names_batch):
                fidelity, self.rng_key = self.continuous_optimizer.optimize_circuit(
                    names,
                    self.rng_key,
                )
                results.append(1.0 - fidelity)
                best_f = max(best_f, fidelity)
                if verbose:
                    print(
                        f"    angle opt [{sample_idx + 1:{width}d}/{total}]"
                        f"  f={fidelity:.6f}  best_f={best_f:.6f}"
                    )

        return np.asarray(results, dtype=np.float32)

    def train_dataloader(self):
        return BufferDataset(self.buffer, self.cfg.buffer.steps_per_epoch)

    def train_batch(self, idx: np.ndarray, costs: np.ndarray, batch_idx: int) -> float:
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        costs_jax = jnp.asarray(costs, dtype=jnp.float32)
        beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)
        self.rng_key, dropout_key = jax.random.split(self.rng_key)

        if batch_idx == 0:
            self.state, loss_value, old_log_probs, advantages, identical = self._first_batch_step(
                self.state,
                idx_jax,
                costs_jax,
                beta,
                dropout_key,
            )
            if not bool(np.asarray(identical)):
                self._old_log_probs = jax.lax.stop_gradient(old_log_probs)
                self._advantages = jax.lax.stop_gradient(advantages)
        else:
            if self._old_log_probs is None or self._advantages is None:
                raise RuntimeError("GRPO reference tensors were not initialized on batch 0")
            self.state, loss_value = self._later_batch_step(
                self.state,
                idx_jax,
                costs_jax,
                beta,
                self._old_log_probs,
                self._advantages,
                dropout_key,
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
        grammar_state = (
            self.grammar.initial_state(idx_jax.shape[0]) if self.grammar is not None else None
        )

        for _ in range(ngates):
            attention_mask = jnp.ones_like(idx_jax, dtype=bool)
            self.rng_key, dropout_key, sample_key = jax.random.split(self.rng_key, 3)
            logits = self._model_logits(self.state.params, idx_jax, attention_mask, dropout_key)
            step_logits = logits[:, -1, :]
            if grammar_state is not None:
                mask = self.grammar.forbidden_mask(grammar_state)
                step_logits = jnp.where(mask, jnp.inf, step_logits)
            idx_next = jax.random.categorical(sample_key, -beta * step_logits, axis=-1).astype(
                jnp.int32
            )
            if grammar_state is not None:
                grammar_state = self.grammar.update(grammar_state, idx_next)
            idx_jax = jnp.concatenate((idx_jax, idx_next[:, None]), axis=1)
        return np.asarray(idx_jax, dtype=np.int32)

    def logits(self, idx):
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        attention_mask = jnp.ones_like(idx_jax, dtype=bool)
        self.rng_key, dropout_key = jax.random.split(self.rng_key)
        logits_base = self._model_logits(self.state.params, idx_jax, attention_mask, dropout_key)
        target_idx = idx_jax[:, 1:]
        aligned_logits = logits_base[:, : target_idx.shape[1], :]
        return jnp.take_along_axis(aligned_logits, target_idx[..., None], axis=-1).squeeze(-1)
