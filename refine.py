"""Angle refinement and Pareto-archive rebuilding after policy rollouts."""

from __future__ import annotations

import time

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import optax

from circuit import CircuitEvaluator
from pareto import ParetoArchive, ParetoPoint
from simplify import build_token_axis, simplify_token_sequence


class AngleRefiner:
    """Batched Adam refinement with optional restarts and sweep polish."""

    def __init__(
        self,
        evaluator: CircuitEvaluator,
        *,
        steps: int = 50,
        lr: float = 0.1,
        num_restarts: int = 1,
        use_linear_trace_loss: bool = True,
        early_stop_patience: int = 30,
        early_stop_rel_tol: float = 1.0e-5,
        adaptive_restarts: bool = True,
        restart_fidelity_threshold: float = 0.999,
        sweep_passes: int = 0,
    ):
        if num_restarts < 1:
            raise ValueError("num_restarts must be >= 1")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive")
        if early_stop_rel_tol < 0:
            raise ValueError("early_stop_rel_tol must be >= 0")
        if not (0.0 < restart_fidelity_threshold <= 1.0):
            raise ValueError(
                "restart_fidelity_threshold must be in (0, 1]"
            )
        if sweep_passes < 0:
            raise ValueError("sweep_passes must be >= 0")
        self.evaluator = evaluator
        self.steps = int(steps)
        self.lr = float(lr)
        self.num_restarts = int(num_restarts)
        self.use_linear_trace_loss = bool(use_linear_trace_loss)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_rel_tol = float(early_stop_rel_tol)
        self.adaptive_restarts = bool(adaptive_restarts)
        self.restart_fidelity_threshold = float(restart_fidelity_threshold)
        self.sweep_passes = int(sweep_passes)
        self._adam = optax.adam(learning_rate=lr)

        if self.use_linear_trace_loss:
            value_and_grad_one = jax.value_and_grad(
                lambda angles, token_ids: evaluator._linear_loss_one(
                    angles, token_ids
                )
            )
        else:
            value_and_grad_one = jax.value_and_grad(
                lambda angles, token_ids: 1.0 - evaluator._fidelity_one(
                    token_ids, angles,
                )
            )

        max_steps = jnp.int32(self.steps)
        patience = jnp.int32(self.early_stop_patience)
        real_dtype = evaluator._real_dtype
        rel_tol = jnp.asarray(self.early_stop_rel_tol, dtype=real_dtype)

        def adam_refine_one(angles, token_ids):
            angles = angles.astype(real_dtype)
            opt_state = self._adam.init(angles)
            init_loss, _ = value_and_grad_one(angles, token_ids)
            init_carry = (
                angles,
                opt_state,
                jnp.int32(0),
                init_loss,
                angles,
                jnp.int32(0),
            )

            def cond(c):
                _a, _st, step, _best, _ba, ss = c
                return jnp.logical_and(step < max_steps, ss < patience)

            def body(c):
                a, st, step, best, ba, ss = c
                v, g = value_and_grad_one(a, token_ids)
                u, st_new = self._adam.update(g, st, a)
                a_next = optax.apply_updates(a, u)
                active = jnp.logical_and(step < max_steps, ss < patience)
                rel_thresh = rel_tol * jnp.maximum(
                    jnp.abs(best), jnp.asarray(1e-12, dtype=best.dtype),
                )
                strictly_improved = v < best
                improved_rel = (best - v) > rel_thresh
                update_best = jnp.logical_and(active, strictly_improved)
                new_best = jnp.where(update_best, v, best)
                new_ba = jnp.where(update_best, a, ba)
                new_ss = jnp.where(
                    jnp.logical_and(active, improved_rel),
                    jnp.int32(0),
                    jnp.where(active, ss + jnp.int32(1), ss),
                )
                new_step = jnp.where(active, step + jnp.int32(1), step)
                new_a = jnp.where(active, a_next, a)
                return (new_a, st_new, new_step, new_best, new_ba, new_ss)

            final_carry = jax.lax.while_loop(cond, body, init_carry)
            return final_carry[4]

        self._refine_batch = jax.jit(jax.vmap(adam_refine_one, in_axes=(0, 0)))
        self._rng_seed = 0

    def _run_adam(
        self,
        token_ids_batch: np.ndarray,
        init_angles_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        real_dtype = self.evaluator._real_dtype
        token_jax = jnp.asarray(token_ids_batch, dtype=jnp.int32)
        out = np.array(
            self._refine_batch(
                jnp.asarray(init_angles_batch, dtype=real_dtype),
                token_jax,
            ),
            dtype=np.float32,
            copy=True,
        )
        fids = np.array(
            self.evaluator.fidelity_batch(token_ids_batch, out),
            dtype=np.float32,
            copy=True,
        )
        return fids, out

    def refine_batch(
        self,
        token_ids_batch: np.ndarray,
        init_angles_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the best refined fidelity and angles for each circuit row."""
        if token_ids_batch.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, self.evaluator.max_gates), dtype=np.float32),
            )
        B, T = init_angles_batch.shape

        best_fids, best_angles = self._run_adam(
            token_ids_batch, init_angles_batch.astype(np.float32)
        )

        rng = np.random.default_rng(self._rng_seed)
        self._rng_seed += 1
        for _ in range(self.num_restarts - 1):
            if self.adaptive_restarts:
                needs = best_fids < self.restart_fidelity_threshold
                if not bool(np.any(needs)):
                    break
                keep_idx = np.flatnonzero(needs)
                sub_tokens = token_ids_batch[keep_idx]
                sub_init = rng.uniform(
                    -np.pi, np.pi, size=(keep_idx.size, T)
                ).astype(np.float32)
                fids_sub, refined_sub = self._run_adam(sub_tokens, sub_init)
                cur_sub = best_fids[keep_idx]
                improved_sub = fids_sub > cur_sub
                upd_local = np.flatnonzero(improved_sub)
                if upd_local.size:
                    upd_global = keep_idx[upd_local]
                    best_fids[upd_global] = fids_sub[upd_local]
                    best_angles[upd_global] = refined_sub[upd_local]
            else:
                random_init = rng.uniform(
                    -np.pi, np.pi, size=(B, T)
                ).astype(np.float32)
                fids, refined = self._run_adam(token_ids_batch, random_init)
                improved = fids > best_fids
                best_fids = np.where(improved, fids, best_fids)
                best_angles = np.where(improved[:, None], refined, best_angles)

        if self.sweep_passes > 0:
            from polish import sweep_refine_batch
            polished_angles, polished_fids = sweep_refine_batch(
                self.evaluator,
                token_ids_batch,
                best_angles,
                num_sweeps=self.sweep_passes,
            )
            improved = polished_fids > best_fids
            best_fids = np.where(improved, polished_fids, best_fids)
            best_angles = np.where(
                improved[:, None], polished_angles, best_angles
            )

        return best_fids, best_angles


def refine_pareto_archive(
    archive: ParetoArchive,
    refiner: AngleRefiner,
    *,
    structure_metrics_fn,
    pool_token_names: list[str],
    bos_token_id: int,
    stop_token_id: int,
    apply_simplify: bool = True,
    simplify_max_passes: int = 3,
    verbose: bool = False,
) -> ParetoArchive:
    """Refine archive entries and return a fresh non-dominated archive."""
    if archive is None or len(archive) == 0:
        return archive

    entries = list(archive._archive)
    n = len(entries)
    max_gates = refiner.evaluator.max_gates

    token_axis = build_token_axis(pool_token_names) if apply_simplify else None
    if apply_simplify:
        from circuit import parse_gate_name
        token_q0 = np.zeros((len(pool_token_names),), dtype=np.int32)
        token_q1 = np.full((len(pool_token_names),), -1, dtype=np.int32)
        for i, name in enumerate(pool_token_names):
            spec = parse_gate_name(name)
            token_q0[i] = spec.qubits[0]
            if spec.gate_type == "CNOT":
                token_q1[i] = spec.qubits[1]

    tokens_batch = np.zeros((n, max_gates), dtype=np.int32)
    angles_batch = np.zeros((n, max_gates), dtype=np.float32)
    for i, p in enumerate(entries):
        tok_no_bos = np.asarray(p.token_sequence, dtype=np.int32)[1:].copy()
        if apply_simplify:
            tok_no_bos = simplify_token_sequence(
                tok_no_bos, token_axis, token_q0, token_q1, stop_token_id,
                max_passes=simplify_max_passes,
            )
        tokens_batch[i, :tok_no_bos.size] = tok_no_bos
        if p.opt_angles is not None:
            init = np.asarray(p.opt_angles, dtype=np.float32)
            angles_batch[i, :init.size] = init

    if verbose:
        t0 = time.perf_counter()

    refined_fids, refined_angles = refiner.refine_batch(tokens_batch, angles_batch)

    refined_archive = ParetoArchive(
        max_size=archive.max_size,
        fidelity_floor=archive.fidelity_floor,
        fidelity_tol=archive.fidelity_tol,
    )
    for i, p in enumerate(entries):
        new_f = float(refined_fids[i])
        if new_f < float(p.fidelity):
            new_f = float(p.fidelity)
            new_angles = (
                np.asarray(p.opt_angles, dtype=np.float32)
                if p.opt_angles is not None
                else np.zeros((max_gates,), dtype=np.float32)
            )
        else:
            new_angles = refined_angles[i]

        tok_no_bos = tokens_batch[i]
        depth, total, cnot = structure_metrics_fn(tok_no_bos)
        full_seq = np.concatenate(
            [np.asarray([bos_token_id], dtype=np.int32), tok_no_bos],
            axis=0,
        )
        refined_archive.update(ParetoPoint(
            fidelity=new_f,
            depth=int(depth),
            total_gates=int(total),
            cnot_count=int(cnot),
            token_sequence=full_seq,
            epoch=int(p.epoch),
            opt_angles=np.asarray(new_angles, dtype=np.float32),
        ))

    if verbose:
        dt = time.perf_counter() - t0
        print(
            f"Refinement: {n} entries → {len(refined_archive)} survivors "
            f"in {dt:.1f}s"
        )

    return refined_archive
