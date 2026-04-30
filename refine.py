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


class AngleRefiner:
    """Batched Adam refinement with optional sweep polish."""

    def __init__(
        self,
        evaluator: CircuitEvaluator,
        *,
        steps: int = 50,
        lr: float = 0.1,
        use_linear_trace_loss: bool = True,
        early_stop_patience: int = 30,
        early_stop_rel_tol: float = 1.0e-5,
        sweep_passes: int = 0,
    ):
        if steps <= 0:
            raise ValueError("steps must be positive")
        if early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be positive")
        if early_stop_rel_tol < 0:
            raise ValueError("early_stop_rel_tol must be >= 0")
        if sweep_passes < 0:
            raise ValueError("sweep_passes must be >= 0")
        self.evaluator = evaluator
        self.steps = int(steps)
        self.lr = float(lr)
        self.use_linear_trace_loss = bool(use_linear_trace_loss)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_rel_tol = float(early_stop_rel_tol)
        self.sweep_passes = int(sweep_passes)
        self._adam = optax.adam(learning_rate=lr)

        if self.use_linear_trace_loss:
            value_and_grad_one = jax.value_and_grad(
                lambda angles, token_ids, u_target: evaluator._linear_loss_one(
                    angles, token_ids, u_target,
                )
            )
        else:
            value_and_grad_one = jax.value_and_grad(
                lambda angles, token_ids, u_target: 1.0 - evaluator._fidelity_one(
                    token_ids, angles, u_target,
                )
            )

        max_steps = jnp.int32(self.steps)
        patience = jnp.int32(self.early_stop_patience)
        real_dtype = evaluator._real_dtype
        rel_tol = jnp.asarray(self.early_stop_rel_tol, dtype=real_dtype)

        def adam_refine_one(angles, token_ids, u_target):
            angles = angles.astype(real_dtype)
            opt_state = self._adam.init(angles)
            init_loss, _ = value_and_grad_one(angles, token_ids, u_target)
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
                v, g = value_and_grad_one(a, token_ids, u_target)
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

        self._refine_batch = jax.jit(
            jax.vmap(adam_refine_one, in_axes=(0, 0, None))
        )

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
                self.evaluator.u_target,
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
                np.zeros(init_angles_batch.shape, dtype=np.float32),
            )

        best_fids, best_angles = self._run_adam(
            token_ids_batch, init_angles_batch.astype(np.float32)
        )

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
    bos_token_id: int,
    pad_token_id: int,
    verbose: bool = False,
) -> ParetoArchive:
    """Refine archive entries and return a fresh verified non-dominated archive."""
    if archive is None or len(archive) == 0:
        return archive

    entries = list(archive._archive)
    n = len(entries)
    action_width = max(
        max(
            np.asarray(p.token_sequence, dtype=np.int32).size - 1,
            0 if p.opt_angles is None else np.asarray(p.opt_angles).size,
        )
        for p in entries
    )

    tokens_batch = np.full((n, action_width), pad_token_id, dtype=np.int32)
    original_angles = np.zeros((n, action_width), dtype=np.float32)
    for i, p in enumerate(entries):
        tok_no_bos = np.asarray(p.token_sequence, dtype=np.int32)[1:].copy()
        tokens_batch[i, :tok_no_bos.size] = tok_no_bos
        if p.opt_angles is not None:
            init = np.asarray(p.opt_angles, dtype=np.float32)
            original_angles[i, :init.size] = init

    if verbose:
        t0 = time.perf_counter()

    original_fids = refiner.evaluator.fidelity_batch(tokens_batch, original_angles)
    refined_fids, refined_angles = refiner.refine_batch(tokens_batch, original_angles)
    improved = refined_fids >= original_fids
    stored_angles = np.where(improved[:, None], refined_angles, original_angles)
    stored_fids = refiner.evaluator.fidelity_batch(tokens_batch, stored_angles)

    refined_archive = ParetoArchive(
        max_size=archive.max_size,
        fidelity_floor=archive.fidelity_floor,
        fidelity_tol=archive.fidelity_tol,
    )
    for i, p in enumerate(entries):
        tok_no_bos = tokens_batch[i]
        depth, total, cnot = structure_metrics_fn(tok_no_bos)
        full_seq = np.concatenate(
            [np.asarray([bos_token_id], dtype=np.int32), tok_no_bos],
            axis=0,
        )
        refined_archive.update(ParetoPoint(
            fidelity=float(stored_fids[i]),
            depth=int(depth),
            total_gates=int(total),
            cnot_count=int(cnot),
            token_sequence=full_seq,
            epoch=int(p.epoch),
            opt_angles=np.asarray(stored_angles[i], dtype=np.float32),
        ))

    if verbose:
        dt = time.perf_counter() - t0
        print(
            f"Refinement: {n} entries → {len(refined_archive)} survivors "
            f"in {dt:.1f}s"
        )

    return refined_archive
