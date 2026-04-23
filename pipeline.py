"""JAX-based GQE pipeline for transformer-guided circuit generation.

The transformer emits a variable-length circuit by autoregressively sampling
tokens over a vocabulary of ``{BOS, STOP, gate_1, ..., gate_V}``. BOS sits at
sequence position 0 and is never a valid action. STOP ends the circuit and is
forbidden at position 1 (the first real decision), so every sampled circuit has
at least one gate. Positions after STOP are forced-STOP padding and masked out
of the policy gradient.

The reward is deliberately simple:

    R = F - lambda_structure * (C + gamma_depth * D) / max_gates_count

F is process fidelity, C is CNOT count, D is circuit depth. Fidelity dominates
while it is far from 1; structure savings only move the needle as F approaches 1.
The Pareto archive is kept for reporting and best-circuit selection but does not
feed back into the reward.
"""

from __future__ import annotations

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from cost import _unbiased_std, compilation_cost_batch_jax, process_fidelity
from data import BufferDataset, ReplayBuffer
from kv_rollout import build_kv_rollout_fn, compute_lengths_from_tokens
from loss import apply_action_masks, reduce_sequence_log_probs
from model import _SIZES as _MODEL_SIZES
from pareto import ParetoArchive, ParetoPoint
from simplify import build_token_axis, simplify_token_batch


def _sequence_structure_metrics(
    token_ids: np.ndarray,
    num_qubits: int,
    token_qubit0: np.ndarray,
    token_qubit1: np.ndarray,
    token_is_noop: np.ndarray,
) -> tuple[int, int, int]:
    """Return (depth, total_gates, cnot_count) for a single token sequence.

    Tokens marked as NOOP (BOS / STOP / padding) are skipped entirely so the
    caller can pass the full fixed-length sequence without pre-trimming.
    """
    token_ids = np.asarray(token_ids, dtype=np.int32)
    if token_ids.size == 0:
        return 0, 0, 0

    qubit_depths = np.zeros((num_qubits,), dtype=np.int32)
    total_gates = 0
    cnot_count = 0
    for token_id in token_ids:
        token_id = int(token_id)
        if bool(token_is_noop[token_id]):
            continue
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

        # ── Token layout ────────────────────────────────────────────────────
        self.bos_token_id = 0
        self.stop_token_id = 1
        self.gate_token_offset = 2
        self.vocab_size = len(pool) + self.gate_token_offset
        model_vocab_size = getattr(self.model, "vocab_size", None)
        if model_vocab_size is not None and int(model_vocab_size) != self.vocab_size:
            raise ValueError(
                f"Model vocab_size={model_vocab_size} does not match "
                f"expected {self.vocab_size} (BOS + STOP + pool)"
            )

        self.buffer = ReplayBuffer(size=cfg.buffer.max_size)
        self._starting_idx = self._make_starting_idx(self.num_samples)
        self._last_rollout_costs = np.zeros((0,), dtype=np.float32)
        self._last_rollout_fidelities = np.zeros((0,), dtype=np.float32)
        self._last_rollout_cnot_counts = np.zeros((0,), dtype=np.int32)
        self._last_rollout_depths = np.zeros((0,), dtype=np.int32)
        self._last_rollout_total_gates = np.zeros((0,), dtype=np.int32)
        self._last_rollout_lengths = np.zeros((0,), dtype=np.int32)
        self._last_rollout_old_log_probs = np.zeros((0,), dtype=np.float32)
        self._last_rollout_indices = np.zeros((0, self.ngates + 1), dtype=np.int32)
        self._last_rollout_opt_angles = np.zeros((0, self.ngates), dtype=np.float32)
        self.batch_rng = np.random.default_rng(cfg.training.seed)

        # ── Token metadata tables (vocab-sized) ─────────────────────────────
        self.pool_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
        self.two_qubit_token_mask = np.zeros((self.vocab_size,), dtype=bool)
        self.token_qubit0 = np.zeros((self.vocab_size,), dtype=np.int32)
        self.token_qubit1 = np.full((self.vocab_size,), -1, dtype=np.int32)
        self.token_is_noop = np.zeros((self.vocab_size,), dtype=bool)
        self.token_is_noop[self.bos_token_id] = True
        self.token_is_noop[self.stop_token_id] = True
        for idx, name in enumerate(self.pool_names[self.gate_token_offset:], start=self.gate_token_offset):
            parts = name.split("_")
            self.token_qubit0[idx] = int(parts[1][1:])
            if name.startswith("CNOT"):
                self.token_qubit1[idx] = int(parts[2][1:])
                self.two_qubit_token_mask[idx] = True
        self.token_axis = build_token_axis(self.pool_names)

        # ── Gate matrices indexed by token id ───────────────────────────────
        # Index 0 (BOS) and 1 (STOP) map to identity so variable-length circuits
        # can be composed from a fixed-shape ``(B, max_gates, d, d)`` tensor.
        pool_matrices = np.stack([matrix for _, matrix in pool], axis=0)
        d = 2 ** cfg.target.num_qubits
        identity = np.eye(d, dtype=pool_matrices.dtype)
        token_matrices = np.concatenate(
            [
                np.broadcast_to(identity, (self.gate_token_offset, d, d)),
                pool_matrices,
            ],
            axis=0,
        )

        backend = jax.default_backend()
        is_gpu = backend in ('gpu', 'cuda')
        target_dtype = jnp.complex64 if is_gpu else jnp.complex128
        self.token_matrices_jax = jnp.asarray(token_matrices, dtype=target_dtype)
        self.u_target_jax = (
            jnp.asarray(u_target, dtype=target_dtype) if u_target is not None else None
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
            self.factory.create_continuous_optimizer(cfg, u_target, pool, self.pool_names)
            if u_target is not None
            else None
        )

        # ── Pareto archive (reporting only; decoupled from reward) ──────────
        reward_cfg = cfg.reward
        if reward_cfg.enabled:
            self.pareto_archive: ParetoArchive | None = ParetoArchive(
                max_size=reward_cfg.max_archive_size,
                fidelity_floor=reward_cfg.fidelity_floor,
            )
        else:
            self.pareto_archive = None
        self._current_epoch: int = 0

        self._compile_functions()

    def _make_starting_idx(self, batch_size: int) -> np.ndarray:
        return np.full((batch_size, 1), self.bos_token_id, dtype=np.int32)

    @staticmethod
    def _make_batch_structure_fn(
        num_qubits: int,
        token_qubit0: jax.Array,
        token_qubit1: jax.Array,
        token_is_noop: jax.Array,
    ):
        """Build a JIT function computing (depth, total_gates) for a batch.

        Tokens with ``token_is_noop=True`` (BOS, STOP, padding) are skipped: they
        neither advance per-qubit depth nor count toward total_gates.
        """

        def fn(token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
            # token_ids: (B, T) int32 — without BOS; STOP/padding may appear.
            q0 = token_qubit0[token_ids]                          # (B, T)
            q1 = token_qubit1[token_ids]                          # (B, T)
            is_cnot = q1 >= 0
            is_noop = token_is_noop[token_ids]                    # (B, T)
            q1_safe = jnp.where(is_cnot, q1, 0)
            total_gates = jnp.sum(jnp.where(is_noop, 0, 1), axis=1).astype(jnp.int32)

            def step(depths, xs):
                q0_t, q1_t, is_cnot_t, q1_safe_t, is_noop_t = xs      # each (B,)
                batch_idx = jnp.arange(depths.shape[0])
                d0 = depths[batch_idx, q0_t]
                d1 = depths[batch_idx, q1_safe_t]
                new_d = jnp.where(is_cnot_t, jnp.maximum(d0, d1) + 1, d0 + 1)
                update_q0 = jnp.where(is_noop_t, d0, new_d)
                depths = depths.at[batch_idx, q0_t].set(update_q0)
                update_q1 = jnp.where(is_cnot_t & ~is_noop_t, new_d, depths[batch_idx, q1_safe_t])
                depths = depths.at[batch_idx, q1_safe_t].set(update_q1)
                return depths, None

            B = token_ids.shape[0]
            init = jnp.zeros((B, num_qubits), dtype=jnp.int32)
            q0_T = jnp.transpose(q0, (1, 0))
            q1_T = jnp.transpose(q1, (1, 0))
            is_cnot_T = jnp.transpose(is_cnot, (1, 0))
            q1_safe_T = jnp.transpose(q1_safe, (1, 0))
            is_noop_T = jnp.transpose(is_noop, (1, 0))
            final_depths, _ = jax.lax.scan(
                step, init, (q0_T, q1_T, is_cnot_T, q1_safe_T, is_noop_T)
            )
            depths_max = jnp.max(final_depths, axis=1)
            return depths_max, total_gates

        return fn

    def _compile_functions(self) -> None:
        apply_fn = self.state.apply_fn
        clip_ratio = self.loss.clip_ratio
        batch_size = self.num_samples
        total_len = self.ngates + 1
        bos_id = self.bos_token_id
        stop_id = self.stop_token_id

        def model_logits(params, input_ids, attention_mask):
            return apply_fn(
                {"params": params},
                input_ids,
                attention_mask=attention_mask,
                deterministic=True,
            )

        def selected_log_probs(logits, gate_indices, beta):
            aligned_logits = apply_action_masks(
                logits[:, : gate_indices.shape[1], :],
                bos_token_id=bos_id,
                stop_token_id=stop_id,
            )
            log_probs = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
            return jnp.take_along_axis(log_probs, gate_indices[..., None], axis=-1).squeeze(-1)

        def sequence_log_probs(logits, gate_indices, beta, lengths):
            token_log_probs = selected_log_probs(logits, gate_indices, beta)
            return reduce_sequence_log_probs(token_log_probs, lengths)

        def calc_advantage(costs):
            return (jnp.mean(costs) - costs) / (_unbiased_std(costs) + 1e-8)

        def rollout_sequence_log_probs(params, idx, beta):
            attention_mask = jnp.ones_like(idx, dtype=bool)
            gate_indices = idx[:, 1:]
            lengths = compute_lengths_from_tokens(gate_indices, stop_id)
            current_logits = model_logits(params, idx, attention_mask)
            return sequence_log_probs(current_logits, gate_indices, beta, lengths)

        def grpo_loss(params, idx, costs, beta, old_sequence_log_probs):
            attention_mask = jnp.ones_like(idx, dtype=bool)
            gate_indices = idx[:, 1:]
            lengths = compute_lengths_from_tokens(gate_indices, stop_id)
            current_logits = model_logits(params, idx, attention_mask)
            current_sequence_log_probs = sequence_log_probs(
                current_logits, gate_indices, beta, lengths
            )
            advantages = jax.lax.stop_gradient(calc_advantage(costs))
            ratio = jnp.exp(current_sequence_log_probs - old_sequence_log_probs)
            clipped_ratio = jnp.clip(
                ratio,
                1.0 - clip_ratio,
                1.0 + clip_ratio,
            )
            surrogate = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
            return -jnp.mean(surrogate)

        def grpo_step(state, idx, costs, beta, old_sequence_log_probs):
            loss_value, grads = jax.value_and_grad(grpo_loss)(
                state.params,
                idx,
                costs,
                beta,
                old_sequence_log_probs,
            )
            state = state.apply_gradients(grads=grads)
            return state, loss_value

        def generate_rollout(params, rng_key, beta):
            """Fallback O(N^2) sampler with variable-length STOP handling."""
            tokens = jnp.full((batch_size, total_len), bos_id, dtype=jnp.int32)
            active_mask = jnp.zeros((batch_size, total_len), dtype=bool)
            active_mask = active_mask.at[:, 0].set(True)
            stopped = jnp.zeros((batch_size,), dtype=bool)

            def body(carry, step_idx):
                tokens, active_mask, stopped, rng = carry
                rng, sample_key = jax.random.split(rng)
                logits = model_logits(params, tokens, active_mask)
                step_logits = logits[:, step_idx, :]
                step_logits = step_logits.at[:, bos_id].set(
                    jnp.asarray(jnp.inf, dtype=step_logits.dtype)
                )
                stop_bias = jnp.where(
                    step_idx == 0,
                    jnp.asarray(jnp.inf, dtype=step_logits.dtype),
                    jnp.asarray(0.0, dtype=step_logits.dtype),
                )
                step_logits = step_logits.at[:, stop_id].add(stop_bias)
                sampled = jax.random.categorical(
                    sample_key,
                    -beta * step_logits,
                    axis=-1,
                ).astype(jnp.int32)
                idx_next = jnp.where(stopped, stop_id, sampled)
                new_stopped = stopped | (idx_next == stop_id)
                tokens = tokens.at[:, step_idx + 1].set(idx_next)
                active_mask = active_mask.at[:, step_idx + 1].set(True)
                return (tokens, active_mask, new_stopped, rng), None

            (tokens, _, _, _), _ = jax.lax.scan(
                body,
                (tokens, active_mask, stopped, rng_key),
                jnp.arange(self.ngates, dtype=jnp.int32),
            )
            return tokens

        self._model_logits = jax.jit(model_logits)
        self._grpo_loss = jax.jit(grpo_loss)
        self._grpo_step = jax.jit(grpo_step)
        self._rollout_sequence_log_probs = jax.jit(rollout_sequence_log_probs)
        self._generate_rollout = jax.jit(generate_rollout)

        _arch = _MODEL_SIZES[self.cfg.model.size]
        self._generate_rollout_kv = build_kv_rollout_fn(
            n_layer=_arch.n_layer,
            n_head=_arch.n_head,
            n_embd=_arch.n_embd,
            vocab_size=self.vocab_size,
            batch_size=batch_size,
            total_len=total_len,
            bos_token_id=self.bos_token_id,
            stop_token_id=self.stop_token_id,
        )
        self._batch_structure_jit = jax.jit(
            self._make_batch_structure_fn(
                num_qubits=self.cfg.target.num_qubits,
                token_qubit0=jnp.asarray(self.token_qubit0, dtype=jnp.int32),
                token_qubit1=jnp.asarray(self.token_qubit1, dtype=jnp.int32),
                token_is_noop=jnp.asarray(self.token_is_noop, dtype=bool),
            )
        )
        if self.u_target_jax is not None:
            # NOTE on convention: the CNOT matrices stored in
            # ``self.token_matrices_jax`` come from ``operator_pool``, which
            # uses a different embedding convention from
            # ``ContinuousOptimizer._cnot_matrices``. The two paths can't be
            # swapped without changing numerical outputs for CNOT tokens.
            # Keep the original (d,d) scan for the discrete path; it is cold
            # code under the typical config (top_k=0, continuous enabled).
            self._discrete_cost_batch = jax.jit(
                lambda token_ids: compilation_cost_batch_jax(
                    self.u_target_jax,
                    self.token_matrices_jax[token_ids],
                )
            )
        else:
            self._discrete_cost_batch = None
        self._lengths_from_tokens_jit = jax.jit(
            lambda tokens: compute_lengths_from_tokens(tokens, stop_id)
        )

    def on_fit_start(self):
        self._starting_idx = self._make_starting_idx(self.num_samples)

    def on_train_epoch_start(self):
        self.collect_rollout()

    def collect_rollout(self):
        idx_output = self.generate()
        beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)
        old_log_probs = np.asarray(
            self._rollout_sequence_log_probs(
                self.state.params,
                jnp.asarray(idx_output, dtype=jnp.int32),
                beta,
            ),
            dtype=np.float32,
        )
        token_gates = idx_output[:, 1:]  # strip BOS
        # Raw emission length (position of first STOP) feeds the temperature
        # scheduler's length statistics; simplification may insert interior
        # STOPs so we compute this before rewriting.
        lengths = np.asarray(
            self._lengths_from_tokens_jit(jnp.asarray(token_gates, dtype=jnp.int32)),
            dtype=np.int32,
        )
        # Structurally simplify each circuit before the angle optimiser sees
        # it: cancelled / merged positions become STOP (identity downstream).
        # Fidelity is preserved up to reparametrisation; structural metrics
        # reflect the simplified circuit so reward and Pareto archive count
        # only the gates that survive simplification.
        token_gates = simplify_token_batch(
            np.asarray(token_gates, dtype=np.int32),
            self.token_axis,
            self.token_qubit0,
            self.token_qubit1,
            self.stop_token_id,
        )
        fidelity_costs, fidelities, opt_angles = self.computeCost(token_gates)
        cnot_counts = self._count_cnot_tokens(token_gates)
        depths, total_gates = self._batch_compute_structure(token_gates)

        if self.cfg.reward.enabled:
            buffer_costs = self._compute_reward(fidelities, depths, cnot_counts)
        else:
            buffer_costs = fidelity_costs

        for seq, cost_val, old_log_prob in zip(idx_output, buffer_costs, old_log_probs):
            self.buffer.push(seq, cost_val, old_log_prob)

        # Scheduler annealing uses fidelity costs so it is not confused by the
        # changing scale of the scalarised reward.
        self.scheduler.update(costs=fidelity_costs)
        self._last_rollout_costs = np.asarray(fidelity_costs, dtype=np.float32)
        self._last_rollout_fidelities = np.asarray(fidelities, dtype=np.float32)
        self._last_rollout_cnot_counts = np.asarray(cnot_counts, dtype=np.int32)
        self._last_rollout_depths = np.asarray(depths, dtype=np.int32)
        self._last_rollout_total_gates = np.asarray(total_gates, dtype=np.int32)
        self._last_rollout_lengths = lengths
        self._last_rollout_old_log_probs = old_log_probs
        # Store the simplified BOS-prefixed sequence so best-of-rollout
        # reporting agrees with the simplified structural metrics.
        bos_col = np.full(
            (token_gates.shape[0], 1), self.bos_token_id, dtype=np.int32
        )
        simplified_idx = np.concatenate([bos_col, token_gates], axis=1)
        self._last_rollout_indices = simplified_idx
        self._last_rollout_opt_angles = np.asarray(opt_angles, dtype=np.float32)

        if self.pareto_archive is not None:
            # Pre-filter by fidelity floor in numpy so we only wrap survivors
            # in ParetoPoint objects, then insert the whole batch in one
            # vectorised dominance sweep.
            fid_np = np.asarray(fidelities, dtype=np.float32)
            floor = np.float32(self.pareto_archive.fidelity_floor)
            keep = np.flatnonzero(fid_np >= floor)
            if keep.size:
                epoch = self._current_epoch
                opt_angles_np = np.asarray(opt_angles, dtype=np.float32)
                new_points = [
                    ParetoPoint(
                        fidelity=float(fid_np[i]),
                        depth=int(depths[i]),
                        total_gates=int(total_gates[i]),
                        cnot_count=int(cnot_counts[i]),
                        token_sequence=simplified_idx[i].copy(),
                        epoch=epoch,
                        opt_angles=opt_angles_np[i].copy(),
                    )
                    for i in keep.tolist()
                ]
                self.pareto_archive.update_batch(new_points)

    def set_cost(self, cost):
        self._cost = cost

    def sequence_structure_metrics(self, token_ids: np.ndarray) -> tuple[int, int, int]:
        return _sequence_structure_metrics(
            token_ids,
            self.cfg.target.num_qubits,
            self.token_qubit0,
            self.token_qubit1,
            self.token_is_noop,
        )

    def _count_cnot_tokens(self, idx_output: np.ndarray) -> np.ndarray:
        return np.count_nonzero(self.two_qubit_token_mask[idx_output], axis=1).astype(np.int32)

    def _batch_compute_structure(
        self, token_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (depths, total_gates) for each sequence in the batch.

        token_ids: shape (batch, max_gates) — gate tokens without BOS; STOP /
        forced-STOP padding counts as NOOP and is skipped.
        """
        if token_ids.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )
        token_ids_j = jnp.asarray(token_ids, dtype=jnp.int32)
        depths_j, totals_j = self._batch_structure_jit(token_ids_j)
        return (
            np.asarray(depths_j, dtype=np.int32),
            np.asarray(totals_j, dtype=np.int32),
        )

    def _compute_reward(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> np.ndarray:
        """Return costs (= -reward) using R = F - lam * (C + gamma * D) / L_max.

        Two hyperparameters:
          - lambda_structure (lam): fidelity-vs-structure trade-off weight.
          - gamma_depth (gamma): depth-vs-CNOT relative weight.
        """
        r = self.cfg.reward
        lam = np.float32(r.lambda_structure)
        gamma = np.float32(r.gamma_depth)
        L_max = np.float32(max(self.ngates, 1))
        F = np.asarray(fidelities, dtype=np.float32)
        C = np.asarray(cnot_counts, dtype=np.float32)
        D = np.asarray(depths, dtype=np.float32)
        reward = F - lam * (C + gamma * D) / L_max
        return (-reward).astype(np.float32)

    def _compute_discrete_metrics(
        self,
        token_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._discrete_cost_batch is None:
            raise RuntimeError("Discrete metric evaluation requires a target unitary")
        costs = self._discrete_cost_batch(jnp.asarray(token_ids, dtype=jnp.int32))
        fidelities = 1.0 - costs
        return (
            np.asarray(costs, dtype=np.float32),
            np.asarray(fidelities, dtype=np.float32),
        )

    def computeCost(self, token_ids, *, pool=None, **kwargs):
        """Return ``(costs, fidelities, opt_angles)`` for a batch of circuits.

        ``token_ids`` shape ``(B, max_gates)`` excludes BOS and may contain
        STOP / forced-STOP padding; both are treated as identity for the
        purposes of fidelity evaluation. ``opt_angles`` has shape
        ``(B, max_gates)`` — zeros when no continuous optimiser is active,
        otherwise the angles that achieved ``fidelities``.
        """
        del pool, kwargs
        token_ids = np.asarray(token_ids, dtype=np.int32)
        batch_size, max_gates = token_ids.shape

        if self.continuous_optimizer is None:
            costs, fidelities = self._compute_discrete_metrics(token_ids)
            opt_angles = np.zeros((batch_size, max_gates), dtype=np.float32)
            return costs, fidelities, opt_angles

        opt_angles = np.zeros((batch_size, max_gates), dtype=np.float32)
        if self.continuous_optimizer.top_k > 0:
            discrete_costs, discrete_fidelities = self._compute_discrete_metrics(token_ids)
            top_indices = np.argsort(discrete_costs)[: self.continuous_optimizer.top_k]
            results = discrete_costs.astype(np.float64)
            fidelities = discrete_fidelities.astype(np.float64)
            selected_tokens = token_ids[top_indices]
            optimized_fidelities, selected_angles, self.rng_key = (
                self.continuous_optimizer.optimize_token_id_batch(
                    selected_tokens,
                    self.rng_key,
                )
            )
            for local_idx, sample_idx in enumerate(top_indices.tolist()):
                fidelity = float(optimized_fidelities[local_idx])
                fidelities[sample_idx] = fidelity
                results[sample_idx] = 1.0 - fidelity
                opt_angles[sample_idx] = np.asarray(
                    selected_angles[local_idx], dtype=np.float32
                )
        else:
            fidelities, all_angles, self.rng_key = (
                self.continuous_optimizer.optimize_token_id_batch(
                    token_ids,
                    self.rng_key,
                )
            )
            fidelities = np.asarray(fidelities, dtype=np.float64)
            results = 1.0 - fidelities
            opt_angles = np.asarray(all_angles, dtype=np.float32)

        return (
            np.asarray(results, dtype=np.float32),
            np.asarray(fidelities, dtype=np.float32),
            opt_angles,
        )

    def train_dataloader(self):
        return BufferDataset(self.buffer, self.cfg.buffer.steps_per_epoch)

    def train_batch(
        self,
        idx: np.ndarray,
        costs: np.ndarray,
        old_log_probs: np.ndarray,
    ) -> float:
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        costs_jax = jnp.asarray(costs, dtype=jnp.float32)
        old_log_probs_jax = jnp.asarray(old_log_probs, dtype=jnp.float32)
        beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)

        if np.isnan(np.asarray(old_log_probs)).any():
            raise RuntimeError(
                "Replay buffer entry is missing behavior log-probabilities. "
                "Clear legacy buffer contents and collect fresh rollouts."
            )
        self.state, loss_value = self._grpo_step(
            self.state,
            idx_jax,
            costs_jax,
            beta,
            old_log_probs_jax,
        )
        return float(np.asarray(loss_value, dtype=np.float32))

    def generate(self, idx=None, ngates=None):
        if idx is None and (ngates is None or ngates == self.ngates):
            beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)
            self.rng_key, rollout_key = jax.random.split(self.rng_key)
            return np.asarray(
                self._generate_rollout_kv(self.state.params, rollout_key, beta),
                dtype=np.int32,
            )

        # Arbitrary-prefix / arbitrary-length generation (used mainly for tests).
        idx = np.asarray(idx if idx is not None else self._starting_idx, dtype=np.int32)
        ngates = self.ngates if ngates is None else ngates
        beta = float(self.scheduler.get_inverse_temperature())
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        stopped = jnp.zeros((idx_jax.shape[0],), dtype=bool)

        for step_idx in range(ngates):
            attention_mask = jnp.ones_like(idx_jax, dtype=bool)
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            logits = self._model_logits(self.state.params, idx_jax, attention_mask)
            step_logits = logits[:, -1, :].at[:, self.bos_token_id].set(
                jnp.asarray(jnp.inf, dtype=logits.dtype)
            )
            if step_idx == 0 and idx_jax.shape[1] == 1:
                step_logits = step_logits.at[:, self.stop_token_id].set(
                    jnp.asarray(jnp.inf, dtype=logits.dtype)
                )
            sampled = jax.random.categorical(sample_key, -beta * step_logits, axis=-1).astype(
                jnp.int32
            )
            idx_next = jnp.where(stopped, self.stop_token_id, sampled)
            stopped = stopped | (idx_next == self.stop_token_id)
            idx_jax = jnp.concatenate((idx_jax, idx_next[:, None]), axis=1)
        return np.asarray(idx_jax, dtype=np.int32)

    def logits(self, idx):
        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        attention_mask = jnp.ones_like(idx_jax, dtype=bool)
        logits_base = self._model_logits(self.state.params, idx_jax, attention_mask)
        target_idx = idx_jax[:, 1:]
        aligned_logits = logits_base[:, : target_idx.shape[1], :]
        return jnp.take_along_axis(aligned_logits, target_idx[..., None], axis=-1).squeeze(-1)
