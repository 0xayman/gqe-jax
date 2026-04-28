"""Post-training continuous-parameter refinement.

After RL training has converged we still want to squeeze the last bit of
fidelity out of the best discovered circuits. The RL policy supplied initial
angles; here we run a fixed number of Adam steps initialised at those angles.
Adam is cheap (one ``value_and_grad`` per step, no line search) and converges
fast from a warm RL-suggested init, which is what the policy is trained to
provide.

Operates on an entire ``ParetoArchive``: each surviving entry is re-optimised
in a single batched JAX call, then re-inserted into a fresh archive (so any
entries dominated after refinement are evicted).
"""

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
    """Adam refinement of angle vectors for a batch of circuits.

    PQC loss landscapes are notoriously non-convex, so single-shot Adam from
    one starting point routinely gets stuck in local minima. Set
    ``num_restarts > 1`` to also run Adam from ``num_restarts - 1`` random
    Uniform[-π, π] initialisations alongside the RL-suggested angles, then
    keep the highest-fidelity outcome per circuit.
    """

    def __init__(
        self,
        evaluator: CircuitEvaluator,
        *,
        steps: int = 50,
        lr: float = 0.1,
        num_restarts: int = 1,
    ):
        if num_restarts < 1:
            raise ValueError("num_restarts must be >= 1")
        self.evaluator = evaluator
        self.steps = int(steps)
        self.lr = float(lr)
        self.num_restarts = int(num_restarts)
        self._adam = optax.adam(learning_rate=lr)

        value_and_grad_one = jax.value_and_grad(
            lambda angles, token_ids: 1.0 - evaluator._fidelity_one(
                token_ids, angles,
            )
        )

        def adam_step(angles, token_ids):
            opt_state = self._adam.init(angles)

            def body(_, carry):
                a, st = carry
                _v, g = value_and_grad_one(a, token_ids)
                u, st = self._adam.update(g, st, a)
                return optax.apply_updates(a, u), st

            return jax.lax.fori_loop(0, self.steps, body, (angles, opt_state))[0]

        self._refine_batch = jax.jit(jax.vmap(adam_step, in_axes=(0, 0)))
        self._rng_seed = 0

    def refine_batch(
        self,
        token_ids_batch: np.ndarray,
        init_angles_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(refined_fidelities, refined_angles)`` for a batch.

        With ``num_restarts > 1`` runs Adam from the supplied initialisation
        AND from (num_restarts − 1) random Uniform[-π, π] vectors per circuit,
        then keeps the best refined fidelity per row.
        """
        if token_ids_batch.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, self.evaluator.max_gates), dtype=np.float32),
            )
        B, T = init_angles_batch.shape
        token_jax = jnp.asarray(token_ids_batch, dtype=jnp.int32)
        real_dtype = self.evaluator._real_dtype

        # First run: from the supplied RL-suggested angles.
        best_angles = np.asarray(
            self._refine_batch(
                jnp.asarray(init_angles_batch, dtype=real_dtype),
                token_jax,
            ),
            dtype=np.float32,
        )
        best_fids = self.evaluator.fidelity_batch(token_ids_batch, best_angles)

        # Additional random restarts. Use a NumPy RNG seeded once per refiner
        # instance so reruns are reproducible without complicating the API.
        rng = np.random.default_rng(self._rng_seed)
        self._rng_seed += 1
        for _ in range(self.num_restarts - 1):
            random_init = rng.uniform(-np.pi, np.pi, size=(B, T)).astype(np.float32)
            refined = np.asarray(
                self._refine_batch(
                    jnp.asarray(random_init, dtype=real_dtype),
                    token_jax,
                ),
                dtype=np.float32,
            )
            fids = self.evaluator.fidelity_batch(token_ids_batch, refined)
            improved = fids > best_fids
            best_fids = np.where(improved, fids, best_fids)
            best_angles = np.where(improved[:, None], refined, best_angles)

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
    verbose: bool = False,
) -> ParetoArchive:
    """Refine every entry in ``archive`` in-place; return the rebuilt archive.

    ``structure_metrics_fn(tokens_no_bos) -> (depth, total_gates, cnot_count)``
    so the refiner can recompute structural objectives if simplification
    changes the gate count.

    When ``apply_simplify`` is True, each entry's token sequence is first run
    through the simplifier (which marks merged/cancelled positions as STOP).
    Because we re-optimise the angles afterwards, simplification is lossless
    even when the merge rule pools angle contributions.
    """
    if archive is None or len(archive) == 0:
        return archive

    entries = list(archive._archive)
    n = len(entries)
    max_gates = refiner.evaluator.max_gates

    token_axis = build_token_axis(pool_token_names) if apply_simplify else None
    # Build per-token (q0, q1) lookup from pool names — q1 = -1 for non-CNOT.
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
        # Refinement should not regress fidelity, but Adam can occasionally
        # overshoot — fall back to the stored fidelity in that case.
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
