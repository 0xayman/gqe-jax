"""JAX entrypoint for the Generative Quantum Eigensolver."""

from __future__ import annotations

import os
import time

import numpy as np

from config import GQEConfig
from factory import Factory
from model import GPT2
from pareto import ParetoArchive, ParetoPoint
from pipeline import Pipeline


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
                "lambda_structure": cfg.reward.lambda_structure,
                "gamma_depth": cfg.reward.gamma_depth,
            },
        )

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def finalize(self, status: str) -> None:
        exit_code = 0 if status == "success" else 1
        self._run.finish(exit_code=exit_code)


def _build_logger(cfg: GQEConfig):
    if not cfg.logging.wandb:
        return None
    return _WandbLogger(cfg)


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _update_best_from_latest_rollout(
    pipeline: Pipeline,
    best_cost: float,
    best_indices,
    best_raw_fidelity: float | None,
    best_cnot_count: int | None,
    best_angles,
):
    if pipeline._last_rollout_costs.size == 0:
        return best_cost, best_indices, best_raw_fidelity, best_cnot_count, best_angles

    best_idx = int(np.argmin(pipeline._last_rollout_costs))
    rollout_best_cost = float(pipeline._last_rollout_costs[best_idx])
    if rollout_best_cost < best_cost:
        angles = (
            np.asarray(pipeline._last_rollout_opt_angles[best_idx], dtype=np.float32)
            if pipeline._last_rollout_opt_angles.size
            else None
        )
        return (
            rollout_best_cost,
            np.asarray(pipeline._last_rollout_indices[best_idx], dtype=np.int32),
            float(pipeline._last_rollout_fidelities[best_idx]),
            int(pipeline._last_rollout_cnot_counts[best_idx]),
            angles,
        )
    return best_cost, best_indices, best_raw_fidelity, best_cnot_count, best_angles


def _greedy_gate_drop(
    pipeline: Pipeline,
    tokens_no_bos: np.ndarray,
    base_fidelity: float,
    base_angles: np.ndarray | None,
    fid_floor: float,
    fid_tol: float,
) -> tuple[np.ndarray, float, np.ndarray | None]:
    """Greedily delete gates while preserving fidelity.

    For the current token sequence, build one drop candidate per non-NOOP
    position (that position overwritten with STOP) and re-optimise angles for
    all candidates as one batch. Accept the drop that yields the highest
    fidelity as long as it stays within ``fid_tol`` of the pre-drop fidelity
    and above ``fid_floor``. Prefer dropping CNOTs first — only try rotation
    drops when no CNOT can be removed in the current round. Repeat until no
    drop succeeds.
    """
    optimizer = pipeline.continuous_optimizer
    stop_id = int(pipeline.stop_token_id)
    is_noop = pipeline.token_is_noop
    two_q_mask = pipeline.two_qubit_token_mask

    current = np.asarray(tokens_no_bos, dtype=np.int32).copy()
    current_fid = float(base_fidelity)
    current_angles = (
        np.asarray(base_angles, dtype=np.float32).copy()
        if base_angles is not None
        else None
    )

    while True:
        non_noop = [i for i, t in enumerate(current) if not bool(is_noop[int(t)])]
        if not non_noop:
            break
        cnot_pos = [p for p in non_noop if bool(two_q_mask[int(current[p])])]
        rot_pos = [p for p in non_noop if not bool(two_q_mask[int(current[p])])]

        def evaluate(positions: list[int]):
            if not positions:
                return None
            batch = np.tile(current, (len(positions), 1))
            for row, pos in enumerate(positions):
                batch[row, pos] = stop_id
            fids, angs, pipeline.rng_key = optimizer.optimize_token_id_batch(
                batch, pipeline.rng_key
            )
            fids = np.asarray(fids, dtype=np.float64)
            angs = np.asarray(angs)
            keep = (fids >= current_fid - fid_tol) & (fids >= fid_floor)
            if not keep.any():
                return None
            scored = np.where(keep, fids, -np.inf)
            best = int(np.argmax(scored))
            return (
                batch[best].copy(),
                float(fids[best]),
                np.asarray(angs[best], dtype=np.float32).copy(),
            )

        chosen = evaluate(cnot_pos)
        if chosen is None:
            chosen = evaluate(rot_pos)
        if chosen is None:
            break
        current, current_fid, current_angles = chosen

    return current, current_fid, current_angles


def _refine_pareto_archive(
    pipeline: Pipeline,
    refine_restarts: int = 16,
    drop_fid_floor: float | None = None,
    drop_fid_tol: float = 1e-4,
) -> None:
    """Re-optimise angles then greedily drop redundant gates.

    Phase 1 re-runs L-BFGS on each archive entry with more restarts to tighten
    the angle fit; phase 2 attempts to delete gates while keeping fidelity
    within ``drop_fid_tol`` of the post-phase-1 value and above ``drop_fid_floor``
    (defaults to ``cfg.reward.fidelity_threshold``). Refined points are
    re-inserted through a fresh :class:`ParetoArchive` so dominance filtering
    evicts any entries that the reduced circuits now supersede.
    """
    archive = pipeline.pareto_archive
    optimizer = pipeline.continuous_optimizer
    if archive is None or optimizer is None or len(archive) == 0:
        return

    if drop_fid_floor is None:
        drop_fid_floor = float(pipeline.cfg.reward.fidelity_threshold)

    entries = list(archive._archive)
    # token_sequence is BOS-prefixed; the optimiser consumes (B, max_gates).
    token_batch = np.stack(
        [np.asarray(p.token_sequence, dtype=np.int32)[1:] for p in entries],
        axis=0,
    )

    original_restarts = optimizer.num_restarts
    optimizer.num_restarts = max(refine_restarts, original_restarts)
    try:
        new_fidelities, new_angles, pipeline.rng_key = (
            optimizer.optimize_token_id_batch(token_batch, pipeline.rng_key)
        )
    finally:
        optimizer.num_restarts = original_restarts

    refined = ParetoArchive(
        max_size=archive.max_size,
        fidelity_floor=archive.fidelity_floor,
        fidelity_tol=archive.fidelity_tol,
    )
    ngates = pipeline.ngates
    bos_id = int(pipeline.bos_token_id)
    for p, new_f, new_ang in zip(entries, new_fidelities, new_angles):
        tokens_no_bos = np.asarray(p.token_sequence, dtype=np.int32)[1:].copy()
        improved = float(new_f) >= float(p.fidelity)
        fidelity = max(float(new_f), float(p.fidelity))
        if improved:
            angles = np.asarray(new_ang, dtype=np.float32)
        else:
            angles = (
                np.asarray(p.opt_angles, dtype=np.float32)
                if p.opt_angles is not None
                else None
            )

        depth = p.depth
        total_gates = p.total_gates
        cnot_count = p.cnot_count

        # Phase 2: structural reduction — only worth attempting once the
        # circuit already meets the reporting fidelity threshold.
        if fidelity >= drop_fid_floor:
            reduced_tokens, reduced_fid, reduced_angles = _greedy_gate_drop(
                pipeline,
                tokens_no_bos,
                fidelity,
                angles,
                fid_floor=drop_fid_floor,
                fid_tol=drop_fid_tol,
            )
            if reduced_angles is not None and not np.array_equal(
                reduced_tokens, tokens_no_bos
            ):
                tokens_no_bos = reduced_tokens
                fidelity = reduced_fid
                angles = reduced_angles
                depth, total_gates, cnot_count = pipeline.sequence_structure_metrics(
                    tokens_no_bos
                )

        token_sequence_full = np.empty((ngates + 1,), dtype=np.int32)
        token_sequence_full[0] = bos_id
        token_sequence_full[1:] = tokens_no_bos

        refined.update(
            ParetoPoint(
                fidelity=fidelity,
                depth=int(depth),
                total_gates=int(total_gates),
                cnot_count=int(cnot_count),
                token_sequence=token_sequence_full,
                epoch=p.epoch,
                opt_angles=angles,
            )
        )
    pipeline.pareto_archive = refined


def _run_training(cfg: GQEConfig, pipeline: Pipeline, logger=None, u_target=None, pool=None):
    if logger is None:
        logger = _build_logger(cfg)

    best_cost = float("inf")
    best_indices = None
    best_raw_fidelity = None
    best_cnot_count = None
    best_angles = None
    pipeline._starting_idx = pipeline._make_starting_idx(pipeline.num_samples)
    run_start = time.perf_counter()

    for epoch in range(cfg.training.max_epochs):
        pipeline._current_epoch = epoch
        epoch_start = time.perf_counter()

        pipeline.collect_rollout()
        (
            best_cost,
            best_indices,
            best_raw_fidelity,
            best_cnot_count,
            best_angles,
        ) = _update_best_from_latest_rollout(
            pipeline,
            best_cost,
            best_indices,
            best_raw_fidelity,
            best_cnot_count,
            best_angles,
        )
        epoch_best_idx = int(np.argmin(pipeline._last_rollout_costs))
        epoch_best_cost = float(pipeline._last_rollout_costs[epoch_best_idx])
        epoch_best_fidelity = float(pipeline._last_rollout_fidelities[epoch_best_idx])
        epoch_best_depth = int(pipeline._last_rollout_depths[epoch_best_idx])
        epoch_best_total_gates = int(pipeline._last_rollout_total_gates[epoch_best_idx])
        epoch_best_cnot_count = int(pipeline._last_rollout_cnot_counts[epoch_best_idx])
        epoch_mean_length = float(pipeline._last_rollout_lengths.mean()) if pipeline._last_rollout_lengths.size else float("nan")

        epoch_losses = []
        dataloader = pipeline.train_dataloader()
        for batch_idx, batch in enumerate(
            dataloader.iter_batches(
                cfg.training.batch_size,
                drop_last=True,
                shuffle=True,
                rng=pipeline.batch_rng,
            )
        ):
            loss = pipeline.train_batch(
                batch["idx"],
                batch["cost"],
                batch["old_log_prob"],
            )
            epoch_losses.append(loss)

        epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        )
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - run_start
        run_best_fidelity = (
            float(best_raw_fidelity) if best_raw_fidelity is not None else float("nan")
        )
        run_best_depth = -1
        run_best_total_gates = -1
        run_best_cnot_count = -1
        if best_indices is not None:
            run_best_depth, run_best_total_gates, run_best_cnot_count = (
                pipeline.sequence_structure_metrics(np.asarray(best_indices, dtype=np.int32)[1:])
            )

        pareto_size = 0
        pareto_hv = 0.0
        pareto_best_f = float("nan")
        pareto_min_cnot = -1
        pareto_min_depth = -1
        pareto_min_gates = -1
        pareto_threshold = float(cfg.reward.fidelity_threshold)
        if pipeline.pareto_archive is not None:
            archive = pipeline.pareto_archive
            pareto_size = len(archive)
            pareto_hv = archive.hypervolume_2d()
            top = archive.best_by_fidelity()
            pareto_best_f = top.fidelity if top is not None else float("nan")
            efficient = archive.best_by_cnot(min_fidelity=pareto_threshold)
            pareto_min_cnot = efficient.cnot_count if efficient is not None else -1
            shallow = archive.best_by_depth(min_fidelity=pareto_threshold)
            pareto_min_depth = shallow.depth if shallow is not None else -1
            fewest = archive.best_by_total_gates(min_fidelity=pareto_threshold)
            pareto_min_gates = fewest.total_gates if fewest is not None else -1

        if cfg.logging.verbose:
            pareto_str = (
                f" | pareto_size={pareto_size}"
                f" | hv={pareto_hv:.4f}"
                f" | pareto_best_F={pareto_best_f:.4f}"
                f" | best_cnot@{pareto_threshold:.2f}={pareto_min_cnot}"
                f" | best_depth@{pareto_threshold:.2f}={pareto_min_depth}"
                f" | best_gates@{pareto_threshold:.2f}={pareto_min_gates}"
                if pipeline.pareto_archive is not None
                else ""
            )
            print(
                f"Epoch {epoch + 1:03d}/{cfg.training.max_epochs:03d}"
                f" | cost_epoch_best={epoch_best_cost:.6f}"
                f" | F_epoch_best={epoch_best_fidelity:.6f}"
                f" | depth_epoch_best={epoch_best_depth}"
                f" | gates_epoch_best={epoch_best_total_gates}"
                f" | cnot_epoch_best={epoch_best_cnot_count}"
                f" | mean_len={epoch_mean_length:.1f}"
                f" | cost_run_best={best_cost:.6f}"
                f" | F_run_best={run_best_fidelity:.6f}"
                f" | depth_run_best={run_best_depth}"
                f" | cnot_run_best={run_best_cnot_count}"
                f"{pareto_str}"
                f" | epoch_time={_format_elapsed(epoch_time)}"
                f" | elapsed={_format_elapsed(elapsed)}"
            )

        if logger:
            pareto_metrics: dict[str, float] = {}
            if pipeline.pareto_archive is not None:
                pareto_metrics = {
                    "pareto_archive_size": float(pareto_size),
                    "pareto_hypervolume": pareto_hv,
                    "pareto_best_fidelity": pareto_best_f,
                    "pareto_fidelity_threshold": pareto_threshold,
                    "pareto_min_cnot_at_threshold": float(pareto_min_cnot),
                    "pareto_min_depth_at_threshold": float(pareto_min_depth),
                    "pareto_min_gates_at_threshold": float(pareto_min_gates),
                }
            logger.log_metrics(
                {
                    "loss": epoch_loss,
                    "cost_best": best_cost,
                    "cost_epoch_best": epoch_best_cost,
                    "raw_fidelity_best": run_best_fidelity,
                    "raw_fidelity_epoch_best": epoch_best_fidelity,
                    "depth_best": run_best_depth,
                    "depth_epoch_best": epoch_best_depth,
                    "total_gates_best": run_best_total_gates,
                    "total_gates_epoch_best": epoch_best_total_gates,
                    "cnot_count_best": run_best_cnot_count,
                    "cnot_count_epoch_best": epoch_best_cnot_count,
                    "mean_sampled_length": epoch_mean_length,
                    "inverse_temperature": pipeline.scheduler.get_inverse_temperature(),
                    "epoch_time_sec": epoch_time,
                    "elapsed_time_sec": elapsed,
                    **pareto_metrics,
                },
                step=epoch,
            )

        if cfg.training.early_stop and run_best_fidelity >= 1.0 - 1e-8:
            if cfg.logging.verbose:
                print(
                    f"Fidelity 1.0 reached at epoch {epoch + 1:03d}"
                    f" | elapsed={_format_elapsed(elapsed)}"
                )
            break

    if best_indices is not None:
        best_indices = np.asarray(best_indices, dtype=np.int32).tolist()

    if cfg.logging.verbose and pipeline.pareto_archive is not None and len(pipeline.pareto_archive) > 0:
        print(
            f"\nRefining {len(pipeline.pareto_archive)} Pareto entries "
            f"with extra L-BFGS restarts..."
        )
    refine_start = time.perf_counter()
    _refine_pareto_archive(pipeline)
    if cfg.logging.verbose and pipeline.pareto_archive is not None:
        print(
            f"Pareto refinement done in "
            f"{_format_elapsed(time.perf_counter() - refine_start)}"
            f" | surviving entries: {len(pipeline.pareto_archive)}"
        )

    pipeline.set_cost(None)
    if logger:
        logger.finalize("success")
    return best_cost, best_indices, best_angles, pipeline.pareto_archive


def gqe(cost_fn, pool, cfg: GQEConfig, model=None, u_target=None, logger=None):
    factory = Factory()
    if model is None:
        # BOS (0) + STOP (1) + len(pool) gate tokens
        model = GPT2(cfg.model.size, len(pool) + 2)
    pipeline = Pipeline(cfg, cost_fn, pool, model, factory, u_target=u_target)
    return _run_training(cfg, pipeline, logger=logger, u_target=u_target, pool=pool)
