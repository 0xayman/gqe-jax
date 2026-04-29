"""Hybrid-action GQE training loop.

Per-epoch:
  1. Roll out ``num_samples`` circuits with the current policy. Each circuit
     comes with both discrete tokens and per-position angle samples plus the
     per-position log-probabilities under the current (behavior) policy.
  2. Optionally refine the sampled angles with a few Adam steps
     (``policy.inner_refine_steps``); evaluate process fidelity from the
     resulting (token, angle) pairs and compute structural metrics.
  3. Score each sample with the scalarised reward and push it into the
     replay buffer.
  4. For ``buffer.steps_per_epoch`` passes, sample mini-batches from the
     buffer and step PPO on the joint discrete + continuous action space.

After the final epoch, refine the Pareto archive with multi-restart Adam.
"""

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
    HybridPolicy,
    _SIZES as _POLICY_SIZES,
    apply_discrete_action_masks,
    build_rollout_fn,
    compute_lengths_from_tokens,
    gaussian_log_prob,
)
from refine import AngleRefiner, refine_pareto_archive
from scheduler import CosineScheduler, FixedScheduler, LinearScheduler


# ── Logger ───────────────────────────────────────────────────────────────────

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
                "max_gates_count": cfg.model.max_gates_count,
                "max_epochs": cfg.training.max_epochs,
                "entropy_disc": cfg.policy.entropy_disc,
                "entropy_cont": cfg.policy.entropy_cont,
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


# ── Structural metrics ──────────────────────────────────────────────────────

def _make_structure_jit(num_qubits: int, token_qubit0, token_qubit1, token_is_noop):
    """JIT'd batched (depth, total_gates) for token-id sequences."""
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
    """Pure-numpy (depth, total_gates, cnot_count) for one row."""
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
    """Max count of any (control, target) CNOT pair in any sliding window.

    Per arXiv:2601.03123 Sec 4.1: when the same CNOT pair recurs ``k`` times
    inside a window of width comparable to the qubit count, layer symmetries
    collapse parameters and the resulting "effectively underparameterised"
    skeleton plateaus far above the desired precision regardless of how long
    you optimise. This metric returns that ``k`` so callers can reject the
    structure cheaply.
    """
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


# ── Trainer ──────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    best_cost: float
    best_tokens: list[int] | None
    best_angles: np.ndarray | None
    pareto_archive: ParetoArchive | None
    # Best raw rollout, kept independently of Pareto archive: the agent's
    # highest-fidelity sample regardless of structure penalty.
    best_raw_fidelity: float = float("nan")
    best_raw_tokens: list[int] | None = None
    best_raw_angles: np.ndarray | None = None
    # Post-refinement view of ``best_raw_*`` (None when refinement disabled).
    refined_raw_fidelity: float | None = None
    refined_raw_angles: np.ndarray | None = None
    # Per-epoch metric dictionaries (the same payloads logged to W&B), in
    # epoch order. Captured even when W&B logging is off.
    epoch_logs: list[dict] = None  # type: ignore[assignment]


class Trainer:
    """Encapsulates the full training run for a single target unitary."""

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

        # ── Vocabulary layout ────────────────────────────────────────────────
        self.bos_token_id = 0
        self.stop_token_id = 1
        self.gate_token_offset = 2
        self.pool_token_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
        self.vocab_size = len(self.pool_token_names)
        self.ngates = cfg.model.max_gates_count
        self.num_samples = cfg.training.num_samples

        # ── Token metadata (vocab-sized) ─────────────────────────────────────
        self.token_qubit0 = np.zeros((self.vocab_size,), dtype=np.int32)
        self.token_qubit1 = np.full((self.vocab_size,), -1, dtype=np.int32)
        self.token_is_noop = np.zeros((self.vocab_size,), dtype=bool)
        self.token_is_noop[self.bos_token_id] = True
        self.token_is_noop[self.stop_token_id] = True
        self.two_qubit_token_mask = np.zeros((self.vocab_size,), dtype=bool)
        for tok_id, name in enumerate(self.pool_token_names):
            if tok_id < self.gate_token_offset:
                continue
            parts = name.split("_")
            self.token_qubit0[tok_id] = int(parts[1][1:])
            if name.startswith("CNOT"):
                self.token_qubit1[tok_id] = int(parts[2][1:])
                self.two_qubit_token_mask[tok_id] = True

        # ── Circuit evaluator (used for fidelity at sampled angles) ──────────
        self.evaluator = CircuitEvaluator(
            u_target=u_target,
            num_qubits=cfg.target.num_qubits,
            pool_token_names=self.pool_token_names,
            max_gates=self.ngates,
        )
        self.token_is_parametric = self.evaluator.token_is_parametric_np

        # ── Policy / model + optimiser ───────────────────────────────────────
        self.model = HybridPolicy(cfg.model.size, self.vocab_size)
        self.scheduler = _build_scheduler(cfg)
        self.rng_key = jax.random.PRNGKey(cfg.training.seed)
        self.batch_rng = np.random.default_rng(cfg.training.seed)

        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_input = jnp.zeros((1, self.ngates + 1), dtype=jnp.int32)
        dummy_mask = jnp.ones_like(dummy_input, dtype=bool)
        variables = self.model.init(
            {"params": init_key},
            dummy_input,
            attention_mask=dummy_mask,
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
        # Captured per-epoch metric dicts (also fed to W&B if enabled).
        self.epoch_logs: list[dict] = []

        # ── Pareto archive ───────────────────────────────────────────────────
        if cfg.reward.enabled:
            self.pareto_archive: ParetoArchive | None = ParetoArchive(
                max_size=cfg.reward.max_archive_size,
                fidelity_floor=cfg.reward.fidelity_floor,
            )
        else:
            self.pareto_archive = None
        self._current_epoch = 0

        # ── QASER reward running maxima ─────────────────────────────────────
        # Maintained across rollouts. Values <= 0 in config mean "auto-init
        # from the first rollout"; until then we use 1.0 as a safe fallback.
        self._qaser_max_d = max(float(cfg.reward.qaser_init_max_depth), 1.0)
        self._qaser_max_c = max(float(cfg.reward.qaser_init_max_cnot), 1.0)
        self._qaser_max_g = max(float(cfg.reward.qaser_init_max_gates), 1.0)
        self._qaser_initialized = (
            cfg.reward.qaser_init_max_depth > 0
            and cfg.reward.qaser_init_max_cnot > 0
            and cfg.reward.qaser_init_max_gates > 0
        )

        # ── Per-rollout angle refiner (HyRLQAS-style hybrid loop) ───────────
        # When enabled, every rollout sample runs ``inner_refine_steps``
        # iterations of Adam initialised at the RL-sampled angles.
        # The PPO buffer stores the *initial* (RL-sampled) angles so the
        # importance ratio is computed against the actually sampled action;
        # cost / Pareto archive use the refined fidelity / refined angles so
        # the reward signal reflects the well-conditioned init the agent found.
        if cfg.policy.inner_refine_steps > 0:
            # Inner (per-rollout) refinement uses the same upgraded loss
            # / early-stop machinery as the post-training refiner. Sweep
            # polish stays off here — the budget per rollout is too small to
            # benefit and Adam alone covers the warm-start regime.
            self.inner_refiner: AngleRefiner | None = AngleRefiner(
                self.evaluator,
                steps=cfg.policy.inner_refine_steps,
                lr=cfg.policy.inner_refine_lr,
                use_linear_trace_loss=cfg.refinement.use_linear_trace_loss,
                early_stop_patience=cfg.refinement.early_stop_patience,
                early_stop_rel_tol=cfg.refinement.early_stop_rel_tol,
                adaptive_restarts=False,
                sweep_passes=0,
            )
        else:
            self.inner_refiner = None

        self._compile_jit_fns()

    # ── JIT-compiled hot functions ───────────────────────────────────────────

    def _compile_jit_fns(self) -> None:
        cfg = self.cfg
        bos_id = self.bos_token_id
        stop_id = self.stop_token_id
        ent_disc = float(cfg.policy.entropy_disc)
        ent_cont = float(cfg.policy.entropy_cont)
        clip = float(cfg.training.grpo_clip_ratio)

        is_param_jax = jnp.asarray(self.token_is_parametric, dtype=bool)

        def model_forward(params, input_ids, attention_mask):
            return self.state.apply_fn(
                {"params": params}, input_ids,
                attention_mask=attention_mask, deterministic=True,
            )

        def ppo_step(state, tokens, angles, costs, beta, old_log_p_d, old_log_p_c):
            # tokens shape (B, T+1) BOS-prefixed; angles/log-probs (B, T).
            attention_mask = jnp.ones_like(tokens, dtype=bool)
            actions_disc = tokens[:, 1:]
            actions_cont = angles
            T = actions_disc.shape[1]
            param_mask = is_param_jax[actions_disc]
            lengths = compute_lengths_from_tokens(actions_disc, stop_id)
            valid = jnp.arange(T)[None, :] < lengths[:, None]

            def loss_fn(params):
                logits, mu, log_sigma = state.apply_fn(
                    {"params": params}, tokens,
                    attention_mask=attention_mask, deterministic=True,
                )
                aligned_logits = apply_discrete_action_masks(
                    logits[:, :T, :], bos_token_id=bos_id, stop_token_id=stop_id,
                )
                logp_all = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
                new_log_p_d = jnp.take_along_axis(
                    logp_all, actions_disc[..., None], axis=-1,
                ).squeeze(-1)
                new_log_p_c = gaussian_log_prob(
                    actions_cont, mu[:, :T], log_sigma[:, :T],
                )

                cont_contrib_new = jnp.where(
                    param_mask & valid, new_log_p_c,
                    jnp.asarray(0.0, dtype=new_log_p_c.dtype),
                )
                cont_contrib_old = jnp.where(
                    param_mask & valid, old_log_p_c,
                    jnp.asarray(0.0, dtype=old_log_p_c.dtype),
                )
                per_tok_new = new_log_p_d + cont_contrib_new
                per_tok_old = old_log_p_d + cont_contrib_old

                denom = jnp.maximum(valid.sum(axis=-1).astype(per_tok_new.dtype), 1.0)
                seq_new = jnp.sum(jnp.where(valid, per_tok_new, 0.0), axis=-1) / denom
                seq_old = jnp.sum(jnp.where(valid, per_tok_old, 0.0), axis=-1) / denom

                advantages = jax.lax.stop_gradient(grpo_advantages(costs))
                pg_loss = ppo_clipped_loss(seq_new, seq_old, advantages, clip)

                ent_d = categorical_entropy(aligned_logits, beta, valid)
                ent_c = gaussian_entropy(log_sigma[:, :T], valid)
                total = pg_loss - ent_disc * ent_d - ent_cont * ent_c
                return total, (pg_loss, ent_d, ent_c)

            (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss_value, aux

        self._ppo_step = jax.jit(ppo_step)

        # Behaviour-policy log-probs at rollout time (used to recompute the old
        # log-probs when the buffer entries were stored under the same params).
        def rollout_log_probs(params, tokens, angles, beta):
            actions_disc = tokens[:, 1:]
            T = actions_disc.shape[1]
            attention_mask = jnp.ones_like(tokens, dtype=bool)
            logits, mu, log_sigma = self.state.apply_fn(
                {"params": params}, tokens,
                attention_mask=attention_mask, deterministic=True,
            )
            aligned_logits = apply_discrete_action_masks(
                logits[:, :T, :], bos_token_id=bos_id, stop_token_id=stop_id,
            )
            logp_all = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
            log_p_d = jnp.take_along_axis(
                logp_all, actions_disc[..., None], axis=-1,
            ).squeeze(-1)
            log_p_c = gaussian_log_prob(angles, mu[:, :T], log_sigma[:, :T])
            return log_p_d, log_p_c

        self._rollout_log_probs = jax.jit(rollout_log_probs)

        # KV-cached rollout sampler.
        arch = _POLICY_SIZES[self.cfg.model.size]
        self._rollout_fn = build_rollout_fn(
            n_layer=arch.n_layer,
            n_head=arch.n_head,
            n_embd=arch.n_embd,
            vocab_size=self.vocab_size,
            batch_size=self.num_samples,
            total_len=self.ngates + 1,
            bos_token_id=self.bos_token_id,
            stop_token_id=self.stop_token_id,
        )

        self._structure_jit = _make_structure_jit(
            num_qubits=cfg.target.num_qubits,
            token_qubit0=self.token_qubit0,
            token_qubit1=self.token_qubit1,
            token_is_noop=self.token_is_noop,
        )

    # ── Per-epoch primitives ─────────────────────────────────────────────────

    def collect_rollout(self) -> dict:
        beta = jnp.asarray(self.scheduler.get_inverse_temperature(), dtype=jnp.float32)
        self.rng_key, sub = jax.random.split(self.rng_key)
        out = self._rollout_fn(self.state.params, sub, beta)

        tokens = np.asarray(out["tokens"], dtype=np.int32)              # (B, T+1)
        init_angles = np.asarray(out["angles"], dtype=np.float32)       # (B, T)
        log_p_d = np.asarray(out["log_p_d"], dtype=np.float32)          # (B, T)
        log_p_c = np.asarray(out["log_p_c"], dtype=np.float32)          # (B, T)

        action_tokens = tokens[:, 1:]                                   # (B, T)
        raw_fidelities = self.evaluator.fidelity_batch(action_tokens, init_angles)

        # ── Inner refinement (HyRLQAS-style) ─────────────────────────────────
        # If enabled, each sample's angles get a few classical-optimiser steps
        # initialised at the RL-sampled angles. Reward and Pareto archive use
        # the refined values; the PPO buffer stores the RL-sampled (initial)
        # angles so the importance ratio is computed against the actual action.
        if self.inner_refiner is not None:
            refined_fidelities, refined_angles = self.inner_refiner.refine_batch(
                action_tokens, init_angles,
            )
            # Refinement shouldn't decrease fidelity, but Adam can briefly
            # overshoot; clip to monotone non-decreasing.
            improved = refined_fidelities >= raw_fidelities
            fidelities = np.where(improved, refined_fidelities, raw_fidelities)
            pareto_angles = np.where(
                improved[:, None], refined_angles, init_angles,
            ).astype(np.float32)
        else:
            fidelities = raw_fidelities
            pareto_angles = init_angles
            refined_fidelities = raw_fidelities

        cnot_counts = np.count_nonzero(
            self.two_qubit_token_mask[action_tokens], axis=1,
        ).astype(np.int32)
        depths_j, totals_j = self._structure_jit(
            jnp.asarray(action_tokens, dtype=jnp.int32),
        )
        depths = np.asarray(depths_j, dtype=np.int32)
        totals = np.asarray(totals_j, dtype=np.int32)
        lengths = np.asarray(
            compute_lengths_from_tokens(jnp.asarray(action_tokens), self.stop_token_id),
            dtype=np.int32,
        )

        if self.cfg.reward.enabled:
            costs = self._compute_reward(fidelities, depths, cnot_counts, totals)
        else:
            costs = (1.0 - fidelities).astype(np.float32)

        # Buffer stores INITIAL (RL-sampled) angles — that is the action whose
        # log-prob the PPO ratio refers to.
        for i in range(tokens.shape[0]):
            self.buffer.push(BufferEntry(
                tokens=tokens[i].copy(),
                angles=init_angles[i].copy(),
                cost=float(costs[i]),
                log_p_disc=log_p_d[i].copy(),
                log_p_cont=log_p_c[i].copy(),
            ))

        # Anneal scheduler with raw-fidelity costs (so its scale doesn't drift
        # with the structure-penalty term).
        self.scheduler.update(costs=jnp.asarray(1.0 - fidelities, dtype=jnp.float32))

        if self.pareto_archive is not None:
            keep = np.flatnonzero(fidelities >= self.pareto_archive.fidelity_floor)
            # Optional CNOT-pair-repetition admission filter — drop rollouts
            # whose CNOT layout is "effectively underparameterised" (the same
            # pair recurring densely, per arXiv:2601.03123 Sec 4.1).
            window = int(self.cfg.reward.pair_repeat_window)
            max_rep = int(self.cfg.reward.pair_repeat_max)
            if window > 1 and keep.size:
                survivors: list[int] = []
                for i in keep.tolist():
                    rep = _cnot_pair_max_repetition(
                        action_tokens[i], self.token_qubit0,
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
                        token_sequence=tokens[i].copy(),
                        epoch=self._current_epoch,
                        opt_angles=pareto_angles[i].copy(),
                    )
                    for i in keep.tolist()
                ]
                self.pareto_archive.update_batch(new_points)

        return {
            "tokens": tokens,
            "angles": pareto_angles,        # what the eventual best circuit uses
            "init_angles": init_angles,     # what the policy sampled (PPO action)
            "fidelities": fidelities,       # post-refinement (or raw if disabled)
            "raw_fidelities": raw_fidelities,
            "depths": depths,
            "totals": totals,
            "cnots": cnot_counts,
            "lengths": lengths,
            "costs": costs,
        }

    def _compute_reward(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnots: np.ndarray,
        total_gates: np.ndarray | None = None,
    ) -> np.ndarray:
        r = self.cfg.reward
        F = fidelities.astype(np.float32)
        C = cnots.astype(np.float32)
        D = depths.astype(np.float32)

        G = (
            total_gates.astype(np.float32)
            if total_gates is not None
            else D
        )
        batch_max_d = float(D.max()) if D.size else 0.0
        batch_max_c = float(C.max()) if C.size else 0.0
        batch_max_g = float(G.max()) if G.size else 0.0
        if not self._qaser_initialized:
            self._qaser_max_d = max(batch_max_d, 1.0)
            self._qaser_max_c = max(batch_max_c, 1.0)
            self._qaser_max_g = max(batch_max_g, 1.0)
            self._qaser_initialized = True
        else:
            self._qaser_max_d = max(self._qaser_max_d, batch_max_d)
            self._qaser_max_c = max(self._qaser_max_c, batch_max_c)
            self._qaser_max_g = max(self._qaser_max_g, batch_max_g)
        w_d = np.float32(r.qaser_w_depth)
        w_c = np.float32(r.qaser_w_cnot)
        w_g = np.float32(r.qaser_w_gates)
        base = (
            w_d * np.float32(self._qaser_max_d) / (D + 1.0)
            + w_c * np.float32(self._qaser_max_c) / (C + 1.0)
            + w_g * np.float32(self._qaser_max_g) / (G + 1.0)
        )
        log_inf_eps = float(r.qaser_log_infidelity_eps)
        if log_inf_eps > 0.0:
            eps = np.float32(log_inf_eps)
            log_inf_mult = (1.0 - np.log(1.0 - F + eps)).astype(np.float32)
            reward = base ** F * log_inf_mult - 1.0
        else:
            reward = base ** F - 1.0
        return (-reward).astype(np.float32)

    def train_epoch(self, beta_val: float) -> list[float]:
        """Run ``buffer.steps_per_epoch`` PPO updates and return per-batch losses."""
        dataset = BufferDataset(self.buffer, self.cfg.buffer.steps_per_epoch)
        beta = jnp.asarray(beta_val, dtype=jnp.float32)
        losses: list[float] = []
        for batch in dataset.iter_batches(
            self.cfg.training.batch_size,
            drop_last=True, shuffle=True, rng=self.batch_rng,
        ):
            self.state, loss_v, _aux = self._ppo_step(
                self.state,
                jnp.asarray(batch["tokens"], dtype=jnp.int32),
                jnp.asarray(batch["angles"], dtype=jnp.float32),
                jnp.asarray(batch["cost"], dtype=jnp.float32),
                beta,
                jnp.asarray(batch["log_p_disc"], dtype=jnp.float32),
                jnp.asarray(batch["log_p_cont"], dtype=jnp.float32),
            )
            losses.append(float(np.asarray(loss_v, dtype=np.float32)))
        return losses

    # ── Top-level run() ──────────────────────────────────────────────────────

    def run(self) -> TrainResult:
        cfg = self.cfg
        best_cost = float("inf")
        best_tokens = None
        best_angles = None
        best_fidelity = float("nan")
        # Track the highest-fidelity raw sample independently — the
        # structure-penalised cost can prefer shorter / lower-fidelity circuits
        # so "best by cost" and "best by fidelity" diverge.
        best_raw_fidelity = -1.0
        best_raw_tokens = None
        best_raw_angles = None
        run_start = time.perf_counter()

        for epoch in range(cfg.training.max_epochs):
            self._current_epoch = epoch
            epoch_start = time.perf_counter()

            roll = self.collect_rollout()
            costs = roll["costs"]
            fids = roll["fidelities"]
            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_cost = float(costs[best_idx])
                best_tokens = roll["tokens"][best_idx].copy()
                best_angles = roll["angles"][best_idx].copy()
                best_fidelity = float(fids[best_idx])

            raw_idx = int(np.argmax(fids))
            if float(fids[raw_idx]) > best_raw_fidelity:
                best_raw_fidelity = float(fids[raw_idx])
                best_raw_tokens = roll["tokens"][raw_idx].copy()
                best_raw_angles = roll["angles"][raw_idx].copy()

            beta_val = float(self.scheduler.get_inverse_temperature())
            losses = self.train_epoch(beta_val)
            avg_loss = float(np.mean(losses)) if losses else float("nan")

            elapsed = time.perf_counter() - run_start
            epoch_dt = time.perf_counter() - epoch_start

            self._log_epoch(
                epoch=epoch, avg_loss=avg_loss, beta_val=beta_val,
                roll=roll, best_cost=best_cost, best_fidelity=best_fidelity,
                best_tokens=best_tokens, epoch_time=epoch_dt, elapsed=elapsed,
            )

            if cfg.training.early_stop and best_fidelity >= 1.0 - 1e-8:
                if cfg.logging.verbose:
                    print(f"Early stop: fidelity 1.0 reached at epoch {epoch + 1:03d}")
                break

        # Post-training refinement.
        refined_raw_fidelity: float | None = None
        refined_raw_angles: np.ndarray | None = None
        if cfg.refinement.enabled:
            refiner = AngleRefiner(
                self.evaluator,
                steps=cfg.refinement.steps,
                lr=cfg.refinement.lr,
                num_restarts=cfg.refinement.num_restarts,
                use_linear_trace_loss=cfg.refinement.use_linear_trace_loss,
                early_stop_patience=cfg.refinement.early_stop_patience,
                early_stop_rel_tol=cfg.refinement.early_stop_rel_tol,
                adaptive_restarts=cfg.refinement.adaptive_restarts,
                restart_fidelity_threshold=cfg.refinement.restart_fidelity_threshold,
                sweep_passes=cfg.refinement.sweep_passes,
            )

            # 1. Always refine the best raw rollout, even if the Pareto archive
            #    is empty (the most useful single circuit to report).
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

            # 2. Refine the Pareto archive (no-op if empty).
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
                    pool_token_names=self.pool_token_names,
                    bos_token_id=self.bos_token_id,
                    stop_token_id=self.stop_token_id,
                    apply_simplify=cfg.refinement.apply_simplify,
                    simplify_max_passes=cfg.refinement.simplify_max_passes,
                    verbose=cfg.logging.verbose,
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

    # ── Logging helpers ──────────────────────────────────────────────────────

    def _log_epoch(
        self, *, epoch, avg_loss, beta_val, roll, best_cost, best_fidelity,
        best_tokens, epoch_time, elapsed,
    ):
        cfg = self.cfg
        costs = roll["costs"]
        fids = roll["fidelities"]
        raw_fids = roll["raw_fidelities"]
        epoch_best = int(np.argmin(costs))
        epoch_best_F = float(fids[epoch_best])
        epoch_best_cnot = int(roll["cnots"][epoch_best])
        epoch_best_depth = int(roll["depths"][epoch_best])
        epoch_best_total = int(roll["totals"][epoch_best])
        mean_len = float(roll["lengths"].mean()) if roll["lengths"].size else float("nan")
        # How much the inner refiner is lifting fidelity (0 if disabled).
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
                f" | cnot_epoch={epoch_best_cnot}"
                f" | depth_epoch={epoch_best_depth}"
                f" | mean_len={mean_len:.1f}"
                f" | F_run_best={best_fidelity:.4f}"
                f" | cnot_run={best_cnot}"
                f"{pstr}"
                f"{inner_str}"
                f" | dt={epoch_time:.1f}s | elapsed={elapsed:.0f}s"
            )

        # Build the per-epoch metric record. Captured regardless of whether
        # W&B is enabled so it can be persisted into the run-artifact JSON.
        metrics: dict = {
            "epoch": int(epoch),
            "loss": float(avg_loss) if np.isfinite(avg_loss) else None,
            "inverse_temperature": float(beta_val),
            "cost_best": float(best_cost) if np.isfinite(best_cost) else None,
            "raw_fidelity_best": float(best_fidelity) if np.isfinite(best_fidelity) else None,
            "raw_fidelity_epoch_best": float(epoch_best_F),
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


# ── Convenience entrypoint ───────────────────────────────────────────────────

def gqe(cfg: GQEConfig, u_target: np.ndarray, pool, logger=None) -> TrainResult:
    trainer = Trainer(cfg, u_target, pool, logger=logger)
    return trainer.run()
