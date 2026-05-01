"""Training loop for one target-conditioned hybrid-action synthesis run."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from circuit import CircuitEvaluator
from config import GQEConfig
from data import BufferDataset, BufferEntry, ReplayBuffer
from loss import (
    categorical_entropy,
    gaussian_entropy,
    grpo_advantages,
    ppo_clipped_loss,
)
from pareto import ParetoArchive, ParetoPoint
from policy import (
    DEFAULT_CONTEXT_TOKENS,
    HybridPolicy,
    _SIZES as _POLICY_SIZES,
    apply_discrete_action_masks,
    build_rollout_fn,
    canonicalize_angle,
    compute_lengths_from_tokens,
    periodic_gaussian_log_prob,
)
from refine import AngleRefiner, refine_pareto_archive
from scheduler import CosineScheduler, FixedScheduler, LinearScheduler
from simplify import simplify_pareto_archive, simplify_token_sequence


class _WandbLogger:
    def __init__(self, cfg: GQEConfig):
        import wandb
        self._wandb = wandb
        self._run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "gqe-jax"),
            name=os.getenv("WANDB_NAME", None),
            config={
                "num_qubits": cfg.target.num_qubits,
                "target_type": cfg.target.type,
                "model_size": cfg.model.size,
                "rollout_context_tokens": rollout_context_tokens(
                    cfg.target.num_qubits
                ),
                "max_epochs": cfg.training.max_epochs,
                "entropy_disc": cfg.policy.entropy_disc,
                "entropy_cont": cfg.policy.entropy_cont,
                "angle_supervision_weight": cfg.policy.angle_supervision_weight,
            },
        )

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def finalize(self, status: str) -> None:
        self._run.finish(exit_code=0 if status == "success" else 1)


def build_logger(cfg: GQEConfig):
    return _WandbLogger(cfg) if cfg.logging.wandb else None


def _build_scheduler(cfg: GQEConfig):
    t = cfg.temperature
    if t.scheduler == "fixed":
        return FixedScheduler(value=t.initial_value)
    if t.scheduler == "linear":
        return LinearScheduler(
            start=t.initial_value, delta=t.delta,
            minimum=t.min_value, maximum=t.max_value,
        )
    if t.scheduler == "cosine":
        return CosineScheduler(
            minimum=t.min_value, maximum=t.max_value,
            frequency=max(1, cfg.training.max_epochs // 2),
        )
    raise ValueError(f"Unknown scheduler {t.scheduler!r}")


def rollout_context_tokens(num_qubits: int) -> int:
    """Return the transformer token window used for rollouts."""
    numer = 4 ** int(num_qubits) - 3 * int(num_qubits) - 1
    cnot_lb = max(0, -(-numer // 4))
    action_budget = max(
        32,
        min(DEFAULT_CONTEXT_TOKENS - 1, int(round(7.5 * cnot_lb))),
    )
    return action_budget + 1


def _make_structure_jit(num_qubits: int, token_qubit0, token_qubit1, token_is_noop):
    """Return a batched JIT function for circuit depth and total gate count."""
    q0_jax = jnp.asarray(token_qubit0, dtype=jnp.int32)
    q1_jax = jnp.asarray(token_qubit1, dtype=jnp.int32)
    noop_jax = jnp.asarray(token_is_noop, dtype=bool)

    def fn(token_ids):
        q0 = q0_jax[token_ids]
        q1 = q1_jax[token_ids]
        is_cnot = q1 >= 0
        is_noop = noop_jax[token_ids]
        q1_safe = jnp.where(is_cnot, q1, 0)
        total_gates = jnp.sum(jnp.where(is_noop, 0, 1), axis=1).astype(jnp.int32)

        def step(depths, xs):
            q0_t, q1_t, is_cnot_t, q1_safe_t, is_noop_t = xs
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
        depths_final, _ = jax.lax.scan(
            step, init,
            (q0.T, q1.T, is_cnot.T, q1_safe.T, is_noop.T),
        )
        return jnp.max(depths_final, axis=1), total_gates

    return jax.jit(fn)


def _row_structure_metrics(
    token_ids: np.ndarray,
    num_qubits: int,
    token_qubit0,
    token_qubit1,
    token_is_noop,
) -> tuple[int, int, int]:
    """Return ``(depth, total_gates, cnot_count)`` for one token row."""
    if token_ids.size == 0:
        return 0, 0, 0
    qubit_depths = np.zeros((num_qubits,), dtype=np.int32)
    total = 0
    cnots = 0
    for tok in token_ids:
        tok = int(tok)
        if bool(token_is_noop[tok]):
            continue
        q0 = int(token_qubit0[tok])
        q1 = int(token_qubit1[tok])
        total += 1
        if q1 >= 0:
            layer = max(int(qubit_depths[q0]), int(qubit_depths[q1])) + 1
            qubit_depths[q0] = layer
            qubit_depths[q1] = layer
            cnots += 1
        else:
            qubit_depths[q0] += 1
    return int(np.max(qubit_depths)), total, cnots


def _cnot_pair_max_repetition(
    token_ids: np.ndarray,
    token_qubit0,
    token_qubit1,
    window: int,
) -> int:
    """Return the largest repeated directed CNOT-pair count in any window."""
    if window <= 1 or token_ids.size == 0:
        return 0
    pairs: list[tuple[int, int]] = []
    for tok in token_ids:
        tok = int(tok)
        q1 = int(token_qubit1[tok])
        if q1 >= 0:
            pairs.append((int(token_qubit0[tok]), q1))
    n = len(pairs)
    if n <= 1:
        return 0
    counter: dict[tuple[int, int], int] = {}
    win = min(window, n)
    for i in range(win):
        counter[pairs[i]] = counter.get(pairs[i], 0) + 1
    best = max(counter.values())
    for start in range(1, n - win + 1):
        old = pairs[start - 1]
        counter[old] -= 1
        if counter[old] == 0:
            del counter[old]
        new = pairs[start + win - 1]
        counter[new] = counter.get(new, 0) + 1
        m = max(counter.values())
        if m > best:
            best = m
    return int(best)


@dataclass
class TrainResult:
    best_cost: float
    best_tokens: list[int] | None
    best_angles: np.ndarray | None
    pareto_archive: ParetoArchive | None
    best_raw_fidelity: float = float("nan")
    best_raw_tokens: list[int] | None = None
    best_raw_angles: np.ndarray | None = None
    refined_raw_fidelity: float | None = None
    refined_raw_angles: np.ndarray | None = None
    epoch_logs: list[dict] = None  # type: ignore[assignment]


class Trainer:
    """Owns rollout collection, PPO updates, archive updates, and refinement."""

    def __init__(
        self,
        cfg: GQEConfig,
        u_target: np.ndarray,
        pool: list,
        logger=None,
    ):
        self.cfg = cfg
        self.u_target = u_target
        self.pool = pool
        self.logger = logger

        self.bos_token_id = 0
        self.stop_token_id = 1
        self.pad_token_id = 2
        self.gate_token_offset = 3
        self.pool_token_names = [
            "<BOS>", "<STOP>", "<PAD>", *[name for name, _ in pool]
        ]
        self.vocab_size = len(self.pool_token_names)
        self.context_tokens = rollout_context_tokens(cfg.target.num_qubits)
        self.action_horizon = self.context_tokens - 1
        self.num_samples = cfg.training.num_samples

        self.token_qubit0 = np.zeros((self.vocab_size,), dtype=np.int32)
        self.token_qubit1 = np.full((self.vocab_size,), -1, dtype=np.int32)
        self.token_is_noop = np.zeros((self.vocab_size,), dtype=bool)
        self.token_is_noop[self.bos_token_id] = True
        self.token_is_noop[self.stop_token_id] = True
        self.token_is_noop[self.pad_token_id] = True
        self.two_qubit_token_mask = np.zeros((self.vocab_size,), dtype=bool)
        for tok_id, name in enumerate(self.pool_token_names):
            if tok_id < self.gate_token_offset:
                continue
            parts = name.split("_")
            self.token_qubit0[tok_id] = int(parts[1][1:])
            if name.startswith("CNOT"):
                self.token_qubit1[tok_id] = int(parts[2][1:])
                self.two_qubit_token_mask[tok_id] = True

        self.evaluator = CircuitEvaluator(
            u_target=u_target,
            num_qubits=cfg.target.num_qubits,
            pool_token_names=self.pool_token_names,
        )
        self.token_is_parametric = self.evaluator.token_is_parametric_np
        target_features = np.concatenate(
            [np.real(u_target).reshape(-1), np.imag(u_target).reshape(-1)]
        ).astype(np.float32)
        self.target_features = target_features
        self._target_features_single = jnp.asarray(target_features[None, :], dtype=jnp.float32)
        self._target_features_rollout = jnp.repeat(
            self._target_features_single, self.num_samples, axis=0,
        )

        self.model = HybridPolicy(
            cfg.model.size,
            self.vocab_size,
            n_positions=self.context_tokens,
        )
        self.scheduler = _build_scheduler(cfg)
        self.rng_key = jax.random.PRNGKey(cfg.training.seed)
        self.batch_rng = np.random.default_rng(cfg.training.seed)

        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_input = jnp.zeros((1, self.context_tokens), dtype=jnp.int32)
        dummy_mask = jnp.ones_like(dummy_input, dtype=bool)
        variables = self.model.init(
            {"params": init_key},
            dummy_input,
            attention_mask=dummy_mask,
            target_features=self._target_features_single,
            deterministic=True,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.training.grad_norm_clip),
            optax.adamw(learning_rate=cfg.training.lr, weight_decay=0.01),
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=tx,
        )

        self.buffer = ReplayBuffer(cfg.buffer.max_size)
        self.epoch_logs: list[dict] = []

        if cfg.reward.enabled:
            self.pareto_archive: ParetoArchive | None = ParetoArchive(
                max_size=cfg.reward.max_archive_size,
                fidelity_floor=cfg.reward.fidelity_floor,
            )
        else:
            self.pareto_archive = None
        self._current_epoch = 0

        if cfg.policy.inner_refine_steps > 0:
            self.inner_refiner: AngleRefiner | None = AngleRefiner(
                self.evaluator,
                steps=cfg.policy.inner_refine_steps,
                lr=cfg.policy.inner_refine_lr,
                use_linear_trace_loss=cfg.refinement.use_linear_trace_loss,
                early_stop_patience=cfg.refinement.early_stop_patience,
                early_stop_rel_tol=cfg.refinement.early_stop_rel_tol,
                sweep_passes=0,
            )
        else:
            self.inner_refiner = None

        self._compile_jit_fns()

    def _compile_jit_fns(self) -> None:
        cfg = self.cfg
        bos_id = self.bos_token_id
        stop_id = self.stop_token_id
        pad_id = self.pad_token_id
        ent_disc = float(cfg.policy.entropy_disc)
        ent_cont = float(cfg.policy.entropy_cont)
        angle_weight = float(cfg.policy.angle_supervision_weight)
        clip = float(cfg.training.grpo_clip_ratio)

        is_param_jax = jnp.asarray(self.token_is_parametric, dtype=bool)

        def model_forward(params, input_ids, attention_mask, target_features):
            return self.state.apply_fn(
                {"params": params}, input_ids,
                attention_mask=attention_mask,
                target_features=target_features,
                deterministic=True,
            )

        def ppo_step(
            state,
            tokens,
            angles,
            advantages,
            beta,
            old_log_p_d,
            target_features,
        ):
            attention_mask = jnp.ones_like(tokens, dtype=bool)
            actions_disc = tokens[:, 1:]
            angle_targets = angles
            T = actions_disc.shape[1]
            param_mask = is_param_jax[actions_disc]
            lengths = compute_lengths_from_tokens(actions_disc, stop_id)
            valid = jnp.arange(T)[None, :] < lengths[:, None]
            angle_mask = param_mask & valid

            def loss_fn(params):
                logits, mu, log_sigma = state.apply_fn(
                    {"params": params}, tokens,
                    attention_mask=attention_mask,
                    target_features=target_features,
                    deterministic=True,
                )
                aligned_logits = apply_discrete_action_masks(
                    logits[:, :T, :],
                    bos_token_id=bos_id,
                    stop_token_id=stop_id,
                    pad_token_id=pad_id,
                )
                logp_all = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
                new_log_p_d = jnp.take_along_axis(
                    logp_all, actions_disc[..., None], axis=-1,
                ).squeeze(-1)

                per_tok_new = new_log_p_d
                per_tok_old = old_log_p_d

                denom = jnp.maximum(
                    valid.sum(axis=-1).astype(per_tok_new.dtype), 1.0
                )
                seq_new = jnp.sum(jnp.where(valid, per_tok_new, 0.0), axis=-1) / denom
                seq_old = jnp.sum(jnp.where(valid, per_tok_old, 0.0), axis=-1) / denom

                advantages_sg = jax.lax.stop_gradient(advantages)
                pg_loss = ppo_clipped_loss(seq_new, seq_old, advantages_sg, clip)

                ent_d = categorical_entropy(aligned_logits, beta, valid)
                ent_c = gaussian_entropy(log_sigma[:, :T], angle_mask)
                angle_delta = canonicalize_angle(mu[:, :T] - angle_targets)
                angle_loss_per = 1.0 - jnp.cos(angle_delta)
                angle_denom = jnp.maximum(
                    angle_mask.sum().astype(angle_loss_per.dtype), 1.0
                )
                angle_loss = (
                    jnp.sum(jnp.where(angle_mask, angle_loss_per, 0.0))
                    / angle_denom
                )
                total = (
                    pg_loss
                    - ent_disc * ent_d
                    - ent_cont * ent_c
                    + angle_weight * angle_loss
                )
                return total, (pg_loss, ent_d, ent_c, angle_loss)

            (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss_value, aux

        self._ppo_step = jax.jit(ppo_step)

        def rollout_log_probs(params, tokens, angles, beta, target_features):
            actions_disc = tokens[:, 1:]
            T = actions_disc.shape[1]
            attention_mask = jnp.ones_like(tokens, dtype=bool)
            logits, mu, log_sigma = self.state.apply_fn(
                {"params": params}, tokens,
                attention_mask=attention_mask,
                target_features=target_features,
                deterministic=True,
            )
            aligned_logits = apply_discrete_action_masks(
                logits[:, :T, :],
                bos_token_id=bos_id,
                stop_token_id=stop_id,
                pad_token_id=pad_id,
            )
            logp_all = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
            log_p_d = jnp.take_along_axis(
                logp_all, actions_disc[..., None], axis=-1,
            ).squeeze(-1)
            log_p_c = periodic_gaussian_log_prob(angles, mu[:, :T], log_sigma[:, :T])
            return log_p_d, log_p_c

        self._rollout_log_probs = jax.jit(rollout_log_probs)

        arch = _POLICY_SIZES[self.cfg.model.size]
        self._rollout_fn = build_rollout_fn(
            n_layer=arch.n_layer,
            n_head=arch.n_head,
            n_embd=arch.n_embd,
            vocab_size=self.vocab_size,
            batch_size=self.num_samples,
            total_len=self.context_tokens,
            bos_token_id=self.bos_token_id,
            stop_token_id=self.stop_token_id,
            pad_token_id=self.pad_token_id,
            use_target_conditioning=True,
        )

        self._structure_jit = _make_structure_jit(
            num_qubits=cfg.target.num_qubits,
            token_qubit0=self.token_qubit0,
            token_qubit1=self.token_qubit1,
            token_is_noop=self.token_is_noop,
        )

    def _simplify_rollout_sequences(
        self,
        tokens: np.ndarray,
        angles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Simplify rollout rows and return canonical tokens, angles, and metrics."""
        n = int(tokens.shape[0])
        simplified_tokens = np.empty_like(tokens)
        simplified_angles = np.zeros_like(angles)
        depths = np.zeros((n,), dtype=np.int32)
        totals = np.zeros((n,), dtype=np.int32)
        cnot_counts = np.zeros((n,), dtype=np.int32)
        canonical_hashes: list[str] = []

        for i in range(n):
            st, sa, depth, total, cnot, h = simplify_token_sequence(
                tokens[i], angles[i], self.pool, self.cfg.target.num_qubits,
            )
            simplified_tokens[i] = st
            simplified_angles[i] = sa.astype(angles.dtype, copy=False)
            depths[i] = int(depth)
            totals[i] = int(total)
            cnot_counts[i] = int(cnot)
            canonical_hashes.append(h)

        return (
            simplified_tokens,
            simplified_angles,
            depths,
            totals,
            cnot_counts,
            canonical_hashes,
        )

    def collect_rollout(self) -> dict:
        beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)
        self.rng_key, sub = jax.random.split(self.rng_key)
        out = self._rollout_fn(
            self.state.params, sub, beta, self._target_features_rollout,
        )

        tokens = np.asarray(out["tokens"], dtype=np.int32)
        init_angles = np.asarray(out["angles"], dtype=np.float32)
        log_p_d = np.asarray(out["log_p_d"], dtype=np.float32)

        action_tokens = tokens[:, 1:]
        raw_fidelities = self.evaluator.fidelity_batch(action_tokens, init_angles)

        if self.inner_refiner is not None:
            refined_fidelities, refined_angles = self.inner_refiner.refine_batch(
                action_tokens, init_angles,
            )
            improved = refined_fidelities >= raw_fidelities
            fidelities = np.where(improved, refined_fidelities, raw_fidelities)
            pareto_angles = np.where(
                improved[:, None], refined_angles, init_angles,
            ).astype(np.float32)
        else:
            fidelities = raw_fidelities
            pareto_angles = init_angles
            refined_fidelities = raw_fidelities

        (
            simplified_tokens,
            simplified_angles,
            depths,
            totals,
            cnot_counts,
            canonical_hashes,
        ) = self._simplify_rollout_sequences(tokens, pareto_angles)
        simplified_action_tokens = simplified_tokens[:, 1:]
        lengths = np.asarray(
            compute_lengths_from_tokens(
                jnp.asarray(simplified_action_tokens), self.stop_token_id,
            ),
            dtype=np.int32,
        )
        no_stop = ~np.any(action_tokens == self.stop_token_id, axis=1)

        if self.cfg.reward.enabled:
            costs = self._compute_reward(
                fidelities, depths, cnot_counts, lengths=lengths, no_stop=no_stop,
            )
        else:
            discounts = self._sequence_discounts(lengths)
            costs = (1.0 - discounts * fidelities).astype(np.float32)
        advantages = np.asarray(
            grpo_advantages(jnp.asarray(costs, dtype=jnp.float32)), dtype=np.float32,
        )

        for i in range(tokens.shape[0]):
            self.buffer.push(BufferEntry(
                tokens=tokens[i].copy(),
                angles=pareto_angles[i].copy(),
                cost=float(costs[i]),
                advantage=float(advantages[i]),
                log_p_disc=log_p_d[i].copy(),
            ))

        if self.pareto_archive is not None:
            keep = np.flatnonzero(fidelities >= self.pareto_archive.fidelity_floor)
            window = int(self.cfg.reward.pair_repeat_window)
            max_rep = int(self.cfg.reward.pair_repeat_max)
            if window > 1 and keep.size:
                survivors: list[int] = []
                for i in keep.tolist():
                    rep = _cnot_pair_max_repetition(
                        simplified_action_tokens[i], self.token_qubit0,
                        self.token_qubit1, window,
                    )
                    if rep <= max_rep:
                        survivors.append(i)
                keep = np.asarray(survivors, dtype=np.int64)
            if keep.size:
                new_points = [
                    ParetoPoint(
                        fidelity=float(fidelities[i]),
                        depth=int(depths[i]),
                        total_gates=int(totals[i]),
                        cnot_count=int(cnot_counts[i]),
                        token_sequence=simplified_tokens[i].copy(),
                        epoch=self._current_epoch,
                        opt_angles=simplified_angles[i].copy(),
                        canonical_hash=canonical_hashes[i],
                    )
                    for i in keep.tolist()
                ]
                self.pareto_archive.update_batch(new_points)

        return {
            "tokens": tokens,
            "angles": pareto_angles,
            "simplified_tokens": simplified_tokens,
            "simplified_angles": simplified_angles,
            "init_angles": init_angles,
            "fidelities": fidelities,
            "raw_fidelities": raw_fidelities,
            "depths": depths,
            "totals": totals,
            "cnots": cnot_counts,
            "lengths": lengths,
            "no_stop": no_stop,
            "costs": costs,
            "advantages": advantages,
            "beta": float(np.asarray(beta, dtype=np.float32)),
        }

    def _sequence_discounts(self, lengths: np.ndarray | None) -> np.ndarray | np.float32:
        if lengths is None:
            return np.float32(1.0)
        steps = np.maximum(lengths.astype(np.float32) - 1.0, 0.0)
        return np.power(
            np.float32(self.cfg.reward.sequence_discount), steps,
        ).astype(np.float32)

    def _compute_reward(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnots: np.ndarray,
        lengths: np.ndarray | None = None,
        no_stop: np.ndarray | None = None,
    ) -> np.ndarray:
        r = self.cfg.reward
        F = fidelities.astype(np.float32)
        C = cnots.astype(np.float32)
        D = depths.astype(np.float32)
        discounts = self._sequence_discounts(lengths)

        eps = np.float32(r.lex_infidelity_eps)
        infidelity = np.clip(1.0 - F + eps, eps, 1.0).astype(np.float32)
        reward = -np.float32(r.lex_fidelity_weight) * np.log(infidelity)
        active = (F >= np.float32(r.lex_structure_fidelity_threshold)).astype(
            np.float32
        )
        structure_cost = active * (
            np.float32(r.lex_cnot_weight) * C
            + np.float32(r.lex_depth_weight) * D
        )
        cost = -(discounts * reward) + structure_cost
        if no_stop is not None:
            cost = cost + np.asarray(no_stop, dtype=np.float32) * np.float32(
                r.lex_no_stop_penalty
            )
        return cost.astype(np.float32)

    def train_epoch(self, beta_val: float) -> list[float]:
        """Run ``buffer.steps_per_epoch`` PPO updates and return per-batch losses."""
        dataset = BufferDataset(self.buffer, self.cfg.buffer.steps_per_epoch)
        beta = jnp.asarray(beta_val, dtype=jnp.float32)
        losses: list[float] = []
        for batch in dataset.iter_batches(
            self.cfg.training.batch_size,
            drop_last=True, shuffle=True, rng=self.batch_rng,
        ):
            target_features = jnp.repeat(
                self._target_features_single, batch["tokens"].shape[0], axis=0,
            )
            self.state, loss_v, _aux = self._ppo_step(
                self.state,
                jnp.asarray(batch["tokens"], dtype=jnp.int32),
                jnp.asarray(batch["angles"], dtype=jnp.float32),
                jnp.asarray(batch["advantage"], dtype=jnp.float32),
                beta,
                jnp.asarray(batch["log_p_disc"], dtype=jnp.float32),
                target_features,
            )
            losses.append(float(np.asarray(loss_v, dtype=np.float32)))
        return losses

    def run(self) -> TrainResult:
        cfg = self.cfg
        best_cost = float("inf")
        best_tokens = None
        best_angles = None
        best_fidelity = float("nan")
        best_raw_fidelity = -1.0
        best_raw_tokens = None
        best_raw_angles = None
        best_seen_fidelity = -1.0
        run_start = time.perf_counter()

        for epoch in range(cfg.training.max_epochs):
            self._current_epoch = epoch
            epoch_start = time.perf_counter()

            roll = self.collect_rollout()
            costs = roll["costs"]
            fids = roll["fidelities"]
            raw_fids = roll["raw_fidelities"]
            beta_val = float(roll["beta"])
            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_cost = float(costs[best_idx])
                if cfg.reward.enabled:
                    best_tokens = roll["simplified_tokens"][best_idx].copy()
                    best_angles = roll["simplified_angles"][best_idx].copy()
                else:
                    best_tokens = roll["tokens"][best_idx].copy()
                    best_angles = roll["angles"][best_idx].copy()
                best_fidelity = float(fids[best_idx])

            best_seen_fidelity = max(best_seen_fidelity, float(np.max(fids)))

            raw_idx = int(np.argmax(raw_fids))
            if float(raw_fids[raw_idx]) > best_raw_fidelity:
                best_raw_fidelity = float(raw_fids[raw_idx])
                best_raw_tokens = roll["tokens"][raw_idx].copy()
                best_raw_angles = roll["init_angles"][raw_idx].copy()

            losses = self.train_epoch(beta_val)
            avg_loss = float(np.mean(losses)) if losses else float("nan")

            self.scheduler.update(costs=jnp.asarray(1.0 - fids, dtype=jnp.float32))

            elapsed = time.perf_counter() - run_start
            epoch_dt = time.perf_counter() - epoch_start

            self._log_epoch(
                epoch=epoch, avg_loss=avg_loss, beta_val=beta_val,
                roll=roll, best_cost=best_cost, best_fidelity=best_fidelity,
                best_raw_fidelity=best_raw_fidelity,
                best_seen_fidelity=best_seen_fidelity,
                best_tokens=best_tokens, epoch_time=epoch_dt, elapsed=elapsed,
            )

            if cfg.training.early_stop and best_seen_fidelity >= 1.0 - 1e-8:
                if cfg.logging.verbose:
                    print(f"Early stop: fidelity 1.0 reached at epoch {epoch + 1:03d}")
                break

        refined_raw_fidelity: float | None = None
        refined_raw_angles: np.ndarray | None = None
        if cfg.refinement.enabled:
            refiner = AngleRefiner(
                self.evaluator,
                steps=cfg.refinement.steps,
                lr=cfg.refinement.lr,
                use_linear_trace_loss=cfg.refinement.use_linear_trace_loss,
                early_stop_patience=cfg.refinement.early_stop_patience,
                early_stop_rel_tol=cfg.refinement.early_stop_rel_tol,
                sweep_passes=cfg.refinement.sweep_passes,
            )

            if best_raw_tokens is not None:
                if cfg.logging.verbose:
                    print(
                        f"\nRefining best raw rollout (F={best_raw_fidelity:.4f}) "
                        f"with {cfg.refinement.steps} adam steps..."
                    )
                tok_no_bos = np.asarray(best_raw_tokens, dtype=np.int32)[1:][None, :]
                ang = np.asarray(best_raw_angles, dtype=np.float32)[None, :]
                rfids, rangles = refiner.refine_batch(tok_no_bos, ang)
                refined_raw_fidelity = float(rfids[0])
                refined_raw_angles = rangles[0]
                if cfg.logging.verbose:
                    print(
                        f"  Best raw: F {best_raw_fidelity:.4f} → "
                        f"{refined_raw_fidelity:.4f}"
                    )

            if self.pareto_archive is not None and len(self.pareto_archive) > 0:
                if cfg.logging.verbose:
                    print(
                        f"\nRefining {len(self.pareto_archive)} Pareto entries "
                        f"with {cfg.refinement.steps} adam steps..."
                    )
                self.pareto_archive = refine_pareto_archive(
                    self.pareto_archive, refiner,
                    structure_metrics_fn=lambda toks: _row_structure_metrics(
                        toks, cfg.target.num_qubits,
                        self.token_qubit0, self.token_qubit1, self.token_is_noop,
                    ),
                    bos_token_id=self.bos_token_id,
                    pad_token_id=self.pad_token_id,
                    verbose=cfg.logging.verbose,
                )
                self.pareto_archive = simplify_pareto_archive(
                    self.pareto_archive, self.pool, cfg.target.num_qubits,
                )
            elif cfg.logging.verbose:
                print(
                    f"\nPareto archive empty (fidelity_floor="
                    f"{cfg.reward.fidelity_floor:.3f} not reached) — "
                    f"skipping archive refinement."
                )

        if self.logger:
            self.logger.finalize("success")

        return TrainResult(
            best_cost=best_cost,
            best_tokens=best_tokens.tolist() if best_tokens is not None else None,
            best_angles=best_angles,
            pareto_archive=self.pareto_archive,
            best_raw_fidelity=best_raw_fidelity,
            best_raw_tokens=best_raw_tokens.tolist() if best_raw_tokens is not None else None,
            best_raw_angles=best_raw_angles,
            refined_raw_fidelity=refined_raw_fidelity,
            refined_raw_angles=refined_raw_angles,
            epoch_logs=list(self.epoch_logs),
        )

    def _log_epoch(
        self, *, epoch, avg_loss, beta_val, roll, best_cost, best_fidelity,
        best_raw_fidelity, best_seen_fidelity, best_tokens, epoch_time, elapsed,
    ):
        cfg = self.cfg
        costs = roll["costs"]
        fids = roll["fidelities"]
        raw_fids = roll["raw_fidelities"]
        epoch_best = int(np.argmin(costs))
        epoch_best_F = float(fids[epoch_best])
        epoch_best_seen_F = float(np.max(fids)) if fids.size else float("nan")
        epoch_best_raw_F = float(np.max(raw_fids)) if raw_fids.size else float("nan")
        epoch_best_cnot = int(roll["cnots"][epoch_best])
        epoch_best_depth = int(roll["depths"][epoch_best])
        epoch_best_total = int(roll["totals"][epoch_best])
        mean_len = float(roll["lengths"].mean()) if roll["lengths"].size else float("nan")
        mean_inner_lift = (
            float((fids - raw_fids).mean()) if self.inner_refiner is not None else 0.0
        )

        if best_tokens is not None:
            tok_no_bos = np.asarray(best_tokens, dtype=np.int32)[1:]
            best_depth, best_total, best_cnot = _row_structure_metrics(
                tok_no_bos, cfg.target.num_qubits,
                self.token_qubit0, self.token_qubit1, self.token_is_noop,
            )
        else:
            best_depth = best_total = best_cnot = -1

        pareto_size = 0
        pareto_hv = 0.0
        pareto_best_F = float("nan")
        pareto_min_cnot = -1
        pareto_min_depth = -1
        pareto_min_gates = -1
        thresh = float(cfg.reward.fidelity_threshold)
        if self.pareto_archive is not None:
            arc = self.pareto_archive
            pareto_size = len(arc)
            pareto_hv = arc.hypervolume_2d()
            top = arc.best_by_fidelity()
            pareto_best_F = top.fidelity if top is not None else float("nan")
            best_c = arc.best_by_cnot(min_fidelity=thresh)
            pareto_min_cnot = best_c.cnot_count if best_c is not None else -1
            best_d = arc.best_by_depth(min_fidelity=thresh)
            pareto_min_depth = best_d.depth if best_d is not None else -1
            best_g = arc.best_by_total_gates(min_fidelity=thresh)
            pareto_min_gates = best_g.total_gates if best_g is not None else -1

        if cfg.logging.verbose:
            pstr = (
                f" | pareto={pareto_size}"
                f" | hv={pareto_hv:.4f}"
                f" | F_pareto_best={pareto_best_F:.4f}"
                f" | best_cnot@{thresh:.2f}={pareto_min_cnot}"
                if self.pareto_archive is not None else ""
            )
            inner_str = (
                f" | inner_lift={mean_inner_lift:.4f}"
                if self.inner_refiner is not None else ""
            )
            print(
                f"Epoch {epoch + 1:03d}/{cfg.training.max_epochs:03d}"
                f" | loss={avg_loss:.4f}"
                f" | beta={beta_val:.3f}"
                f" | F_epoch_best={epoch_best_F:.4f}"
                f" | F_seen={best_seen_fidelity:.4f}"
                f" | cnot_epoch={epoch_best_cnot}"
                f" | depth_epoch={epoch_best_depth}"
                f" | mean_len={mean_len:.1f}"
                f" | F_run_best={best_fidelity:.4f}"
                f" | cnot_run={best_cnot}"
                f"{pstr}"
                f"{inner_str}"
                f" | dt={epoch_time:.1f}s | elapsed={elapsed:.0f}s"
            )

        metrics: dict = {
            "epoch": int(epoch),
            "loss": float(avg_loss) if np.isfinite(avg_loss) else None,
            "inverse_temperature": float(beta_val),
            "cost_best": float(best_cost) if np.isfinite(best_cost) else None,
            "best_cost_fidelity": (
                float(best_fidelity) if np.isfinite(best_fidelity) else None
            ),
            "best_seen_fidelity": (
                float(best_seen_fidelity) if np.isfinite(best_seen_fidelity) else None
            ),
            "raw_fidelity_best": (
                float(best_raw_fidelity) if np.isfinite(best_raw_fidelity) else None
            ),
            "rollout_fidelity_epoch_best": float(epoch_best_seen_F),
            "raw_fidelity_epoch_best": float(epoch_best_raw_F),
            "depth_best": int(best_depth),
            "depth_epoch_best": int(epoch_best_depth),
            "total_gates_best": int(best_total),
            "total_gates_epoch_best": int(epoch_best_total),
            "cnot_count_best": int(best_cnot),
            "cnot_count_epoch_best": int(epoch_best_cnot),
            "mean_sampled_length": float(mean_len) if np.isfinite(mean_len) else None,
            "epoch_time_sec": float(epoch_time),
            "elapsed_time_sec": float(elapsed),
            "inner_refine_lift_mean": float(mean_inner_lift),
        }
        if self.pareto_archive is not None:
            metrics.update({
                "pareto_archive_size": int(pareto_size),
                "pareto_hypervolume": float(pareto_hv),
                "pareto_best_fidelity": (
                    float(pareto_best_F) if np.isfinite(pareto_best_F) else None
                ),
                "pareto_min_cnot_at_threshold": int(pareto_min_cnot),
                "pareto_min_depth_at_threshold": int(pareto_min_depth),
                "pareto_min_gates_at_threshold": int(pareto_min_gates),
            })
        self.epoch_logs.append(metrics)

        if self.logger:
            self.logger.log_metrics(metrics, step=epoch)

    def sequence_structure_metrics(self, tok_no_bos: np.ndarray) -> tuple[int, int, int]:
        return _row_structure_metrics(
            tok_no_bos, self.cfg.target.num_qubits,
            self.token_qubit0, self.token_qubit1, self.token_is_noop,
        )


def gqe(cfg: GQEConfig, u_target: np.ndarray, pool, logger=None) -> TrainResult:
    trainer = Trainer(cfg, u_target, pool, logger=logger)
    return trainer.run()
