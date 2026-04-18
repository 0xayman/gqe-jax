"""JAX-based GQE pipeline for transformer-guided circuit generation."""

from __future__ import annotations

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from cost import _unbiased_std, compilation_cost_batch_jax, process_fidelity
from data import BufferDataset, ReplayBuffer
from loss import reduce_sequence_log_probs
from pareto import ParetoArchive, ParetoPoint


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
    for token_id in token_ids:
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
        self._last_rollout_old_log_probs = np.zeros((0,), dtype=np.float32)
        self._last_rollout_indices = np.zeros((0, self.ngates + 1), dtype=np.int32)
        self.batch_rng = np.random.default_rng(cfg.training.seed)

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
        backend = jax.default_backend()
        is_gpu = backend in ('gpu', 'cuda')
        target_dtype = jnp.complex64 if is_gpu else jnp.complex128
        self.pool_matrices_jax = jnp.asarray(self.pool_matrices, dtype=target_dtype)
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
            self.factory.create_continuous_optimizer(cfg, u_target, pool)
            if u_target is not None
            else None
        )

        # ── Pareto archive ──────────────────────────────────────────────────
        reward_cfg = cfg.reward
        if reward_cfg.enabled:
            self.pareto_archive: ParetoArchive | None = ParetoArchive(
                max_size=reward_cfg.max_archive_size,
                fidelity_floor=reward_cfg.fidelity_floor,
            )
        else:
            self.pareto_archive = None
        self._warmup_mode: bool = True
        self._current_epoch: int = 0
        self._reward_ref_depth: float | None = None
        self._reward_ref_cnot: float | None = None
        self._reward_ref_fidelity: float = float("-inf")

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
            return reduce_sequence_log_probs(selected_log_probs(logits, gate_indices, beta))

        def calc_advantage(costs):
            return (jnp.mean(costs) - costs) / (_unbiased_std(costs) + 1e-8)

        def rollout_sequence_log_probs(params, idx, beta):
            attention_mask = jnp.ones_like(idx, dtype=bool)
            gate_indices = idx[:, 1:]
            current_logits = model_logits(params, idx, attention_mask)
            return sequence_log_probs(current_logits, gate_indices, beta)

        def grpo_loss(params, idx, costs, beta, old_sequence_log_probs):
            current_sequence_log_probs = rollout_sequence_log_probs(params, idx, beta)
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
        self._rollout_sequence_log_probs = jax.jit(rollout_sequence_log_probs)
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
        beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)
        old_log_probs = np.asarray(
            self._rollout_sequence_log_probs(
                self.state.params,
                jnp.asarray(idx_output, dtype=jnp.int32),
                beta,
            ),
            dtype=np.float32,
        )
        # fidelity_costs = 1 − F; used for tracking, best-circuit selection,
        # and temperature scheduling — always fidelity-only regardless of Pareto mode.
        fidelity_costs, fidelities, cnot_counts = self.computeCost(idx_output[:, 1:], self.pool)

        # When the reward archive is active, compute structure metrics and
        # build the full reward cost that goes into the replay buffer.
        if self.pareto_archive is not None:
            depths, total_gates = self._batch_compute_structure(idx_output[:, 1:])
            self._update_reward_references(fidelities, depths, cnot_counts)
            if not self._warmup_mode:
                buffer_costs = self._compute_reward(fidelities, depths, cnot_counts)
            else:
                buffer_costs = fidelity_costs
        else:
            buffer_costs = fidelity_costs

        for seq, cost_val, old_log_prob in zip(idx_output, buffer_costs, old_log_probs):
            self.buffer.push(seq, cost_val, old_log_prob)

        # Scheduler always sees fidelity costs so its annealing is not confused
        # by the changing scale of scalarized costs.
        self.scheduler.update(costs=fidelity_costs)
        self._last_rollout_costs = np.asarray(fidelity_costs, dtype=np.float32)
        self._last_rollout_fidelities = np.asarray(fidelities, dtype=np.float32)
        self._last_rollout_cnot_counts = np.asarray(cnot_counts, dtype=np.int32)
        self._last_rollout_old_log_probs = old_log_probs
        self._last_rollout_indices = np.asarray(idx_output, dtype=np.int32)

        # Update the Pareto archive with every circuit from this rollout.
        if self.pareto_archive is not None:
            for i in range(len(idx_output)):
                self.pareto_archive.update(
                    ParetoPoint(
                        fidelity=float(fidelities[i]),
                        depth=int(depths[i]),
                        total_gates=int(total_gates[i]),
                        cnot_count=int(cnot_counts[i]),
                        token_sequence=np.asarray(idx_output[i], dtype=np.int32).copy(),
                        epoch=self._current_epoch,
                    )
                )

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

    def _batch_compute_structure(
        self, token_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (depths, total_gates) for each sequence in the batch.

        token_ids: shape (batch, ngates) — gate tokens without BOS.
        Both returned arrays have shape (batch,) and dtype int32.
        """
        token_ids = np.asarray(token_ids, dtype=np.int32)
        n = len(token_ids)
        depths = np.zeros(n, dtype=np.int32)
        total_gates = np.zeros(n, dtype=np.int32)

        for i, row in enumerate(token_ids):
            d, t, _ = self.sequence_structure_metrics(row)
            depths[i] = d
            total_gates[i] = t
        return depths, total_gates

    def _select_reward_reference_index(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> int:
        order = np.lexsort(
            (
                np.asarray(depths, dtype=np.int32),
                np.asarray(cnot_counts, dtype=np.int32),
                -np.asarray(fidelities, dtype=np.float64),
            )
        )
        return int(order[0])

    @staticmethod
    def _is_better_structure(
        cand_depth: float,
        cand_cnot: float,
        ref_depth: float,
        ref_cnot: float,
    ) -> bool:
        return cand_cnot < ref_cnot or (
            cand_cnot == ref_cnot and cand_depth < ref_depth
        )

    def _phi(self, F: np.ndarray) -> np.ndarray:
        """Log-infidelity utility Φ(F) = -log(1 - F + ε_φ)."""
        eps = np.float32(self.cfg.reward.eps_phi)
        F_f = np.asarray(F, dtype=np.float32)
        return -np.log(np.maximum(1.0 - F_f + eps, eps))

    def fidelity_gate(self, fidelities) -> np.ndarray:
        """Soft fidelity gate w(F) = σ((F - F_gate) / τ)."""
        F_f = np.asarray(fidelities, dtype=np.float32)
        f_gate = np.float32(self.cfg.reward.f_gate)
        tau = np.float32(self.cfg.reward.tau)
        z = (F_f - f_gate) / tau
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def _update_reward_references(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> None:
        if len(fidelities) == 0:
            return

        idx = self._select_reward_reference_index(fidelities, depths, cnot_counts)
        cand_fidelity = float(np.asarray(fidelities[idx], dtype=np.float64))
        cand_depth = float(np.asarray(depths[idx], dtype=np.float64))
        cand_cnot = float(np.asarray(cnot_counts[idx], dtype=np.float64))
        tol = 1e-6
        same_fidelity = abs(cand_fidelity - self._reward_ref_fidelity) <= tol if np.isfinite(
            self._reward_ref_fidelity
        ) else False
        better_structure = False
        if self._reward_ref_depth is not None and self._reward_ref_cnot is not None:
            better_structure = self._is_better_structure(
                cand_depth,
                cand_cnot,
                self._reward_ref_depth,
                self._reward_ref_cnot,
            )

        if self._reward_ref_depth is None or self._reward_ref_cnot is None:
            self._reward_ref_depth = cand_depth
            self._reward_ref_cnot = cand_cnot
            self._reward_ref_fidelity = cand_fidelity
            return

        if self._warmup_mode:
            if cand_fidelity > self._reward_ref_fidelity + tol or (
                same_fidelity and better_structure
            ):
                self._reward_ref_depth = cand_depth
                self._reward_ref_cnot = cand_cnot
                self._reward_ref_fidelity = cand_fidelity
            return

        if same_fidelity and better_structure:
            self._reward_ref_depth = cand_depth
            self._reward_ref_cnot = cand_cnot
            self._reward_ref_fidelity = max(self._reward_ref_fidelity, cand_fidelity)
            return

        ema = float(self.cfg.reward.reference_ema)
        if cand_fidelity <= self._reward_ref_fidelity + tol:
            return

        if ema <= 0.0:
            self._reward_ref_depth = cand_depth
            self._reward_ref_cnot = cand_cnot
            self._reward_ref_fidelity = cand_fidelity
            return

        self._reward_ref_depth = (1.0 - ema) * self._reward_ref_depth + ema * cand_depth
        self._reward_ref_cnot = (1.0 - ema) * self._reward_ref_cnot + ema * cand_cnot
        self._reward_ref_fidelity = cand_fidelity

    def _rescore_replay_buffer(self) -> None:
        """Rescore all buffer entries with the full reward after warmup ends."""
        if self.pareto_archive is None or len(self.buffer) == 0:
            return
        if self._reward_ref_depth is None or self._reward_ref_cnot is None:
            return

        raw = list(self.buffer.buf)
        idx_batch = np.stack([item[0] for item in raw], axis=0).astype(np.int32)
        old_log_probs = [
            item[2] if len(item) == 3 else np.float32(np.nan)
            for item in raw
        ]
        token_batch = idx_batch[:, 1:]
        _, fidelities, cnot_counts = self.computeCost(token_batch, self.pool)
        depths, _ = self._batch_compute_structure(token_batch)
        scalar_costs = self._compute_reward(fidelities, depths, cnot_counts)

        rebuilt = ReplayBuffer(size=self.buffer.size)
        for seq, cost_val, old_log_prob in zip(idx_batch, scalar_costs, old_log_probs):
            if np.isnan(np.asarray(old_log_prob, dtype=np.float32)):
                rebuilt.push(seq, cost_val)
            else:
                rebuilt.push(seq, cost_val, float(np.asarray(old_log_prob, dtype=np.float32)))
        self.buffer = rebuilt

    def _compute_pareto_bonus(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> np.ndarray:
        """Pareto proxy bonus: β_1 * [non-dominated] + β_2 * N_dom."""
        r = self.cfg.reward
        beta_1 = np.float32(r.beta_1)
        beta_2 = np.float32(r.beta_2)
        archive = self.pareto_archive
        n = len(fidelities)
        bonus = np.zeros(n, dtype=np.float32)

        for i in range(n):
            pt = ParetoPoint(
                fidelity=float(fidelities[i]),
                depth=int(depths[i]),
                total_gates=0,
                cnot_count=int(cnot_counts[i]),
                token_sequence=np.zeros(1, dtype=np.int32),
                epoch=self._current_epoch,
            )
            if any(archive.dominates(e, pt) for e in archive._archive):
                continue
            bonus[i] += beta_1
            bonus[i] += beta_2 * sum(
                1 for e in archive._archive if archive.dominates(pt, e)
            )

        return bonus

    def _compute_reward(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> np.ndarray:
        """Compute cost = -R_total for each circuit (gqe_reward_design.tex).

        Falls back to 1-F cost when no reference circuit has been set yet.
        """
        if self._reward_ref_depth is None or self._reward_ref_cnot is None:
            return (1.0 - np.asarray(fidelities, dtype=np.float32))

        r = self.cfg.reward
        F = np.asarray(fidelities, dtype=np.float32)
        D = np.asarray(depths, dtype=np.float32)
        C = np.asarray(cnot_counts, dtype=np.float32)

        F_ref = np.float32(self._reward_ref_fidelity)
        D_ref = np.float32(self._reward_ref_depth)
        C_ref = np.float32(self._reward_ref_cnot)

        # Log-infidelity difference ΔΦ = Φ(F) - Φ(F_ref)
        phi_F = self._phi(F)
        phi_F_ref = float(self._phi(np.array([F_ref], dtype=np.float32))[0])
        delta_phi = phi_F - np.float32(phi_F_ref)

        # Positive structure gains: G_X = [log((X_ref+1)/(X+1))]+
        G_D = np.maximum(np.float32(0.0), np.log((D_ref + 1.0) / (D + 1.0)))
        G_C = np.maximum(np.float32(0.0), np.log((C_ref + 1.0) / (C + 1.0)))

        # Structure regression penalties: P_X = [log((X+1)/(X_ref+1))]+
        P_D = np.maximum(np.float32(0.0), np.log((D + 1.0) / (D_ref + 1.0)))
        P_C = np.maximum(np.float32(0.0), np.log((C + 1.0) / (C_ref + 1.0)))

        # Soft fidelity gates
        w_F = self.fidelity_gate(F)
        w_F_ref = float(self.fidelity_gate(np.array([F_ref], dtype=np.float32))[0])

        # Exponential structure coupling (shared across regimes 2 & 3)
        _MAX_EXP = np.float32(30.0)
        lam_d = np.float32(r.lambda_d)
        lam_c = np.float32(r.lambda_c)
        struct_arg = np.clip(lam_d * G_D + lam_c * G_C, np.float32(0.0), _MAX_EXP)
        struct_bonus = np.exp(struct_arg) - np.float32(1.0)

        delta_bad = np.float32(r.delta_bad)

        # Regime 1: clearly bad fidelity drop (ΔΦ < -δ_bad)
        excess = np.clip(-delta_phi - delta_bad, np.float32(0.0), _MAX_EXP)
        r_bad = -np.float32(r.alpha_bad) * (
            np.exp(np.float32(r.kappa_bad) * excess) - np.float32(1.0)
        )

        # Regime 2: small fidelity drop (-δ_bad ≤ ΔΦ < 0)
        neg_dp = np.clip(-delta_phi, np.float32(0.0), _MAX_EXP)
        r_soft = (
            -np.float32(r.alpha_soft) * (
                np.exp(np.float32(r.kappa_soft) * neg_dp) - np.float32(1.0)
            )
            + np.float32(r.eta) * np.float32(w_F_ref) * struct_bonus
        )

        # Regime 3: non-worse fidelity (ΔΦ ≥ 0)
        pos_dp = np.clip(delta_phi, np.float32(0.0), _MAX_EXP)
        r_good = (
            np.float32(r.alpha_good) * (
                np.exp(np.float32(r.kappa_good) * pos_dp) - np.float32(1.0)
            )
            + w_F * struct_bonus
            - w_F * (np.float32(r.mu_d) * P_D + np.float32(r.mu_c) * P_C)
        )

        # Select regime
        reward = np.where(
            delta_phi < -delta_bad,
            r_bad,
            np.where(delta_phi < np.float32(0.0), r_soft, r_good),
        ).astype(np.float32)

        # Add Pareto proxy bonus
        if self.pareto_archive is not None and (r.beta_1 > 0.0 or r.beta_2 > 0.0):
            reward = reward + self._compute_pareto_bonus(fidelities, depths, cnot_counts)

        return (-reward).astype(np.float32)

    def _compute_discrete_metrics(
        self,
        pool_idx_output: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._discrete_cost_batch is None:
            raise RuntimeError("Discrete metric evaluation requires a target unitary")

        costs = self._discrete_cost_batch(jnp.asarray(pool_idx_output, dtype=jnp.int32))
        fidelities = 1.0 - costs
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
            )
            for sample_idx, fidelity in zip(top_indices.tolist(), optimized_fidelities.tolist()):
                fidelities[sample_idx] = fidelity
                results[sample_idx] = 1.0 - fidelity
        else:
            fidelities, self.rng_key = self.continuous_optimizer.optimize_token_index_batch(
                pool_idx_output,
                self.rng_key,
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
