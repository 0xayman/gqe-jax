"""JAX entrypoint for the Generative Quantum Eigensolver."""

from __future__ import annotations

import os
import time

import numpy as np

from config import GQEConfig
from factory import Factory
from model import GPT2
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
):
    if pipeline._last_rollout_costs.size == 0:
        return best_cost, best_indices, best_raw_fidelity, best_cnot_count

    best_idx = int(np.argmin(pipeline._last_rollout_costs))
    rollout_best_cost = float(pipeline._last_rollout_costs[best_idx])
    if rollout_best_cost < best_cost:
        return (
            rollout_best_cost,
            np.asarray(pipeline._last_rollout_indices[best_idx], dtype=np.int32),
            float(pipeline._last_rollout_fidelities[best_idx]),
            int(pipeline._last_rollout_cnot_counts[best_idx]),
        )
    return best_cost, best_indices, best_raw_fidelity, best_cnot_count


def _run_training(cfg: GQEConfig, pipeline: Pipeline, logger=None, u_target=None, pool=None):
    if logger is None:
        logger = _build_logger(cfg)

    best_cost = float("inf")
    best_indices = None
    best_raw_fidelity = None
    best_cnot_count = None
    pipeline._starting_idx = pipeline._make_starting_idx(pipeline.num_samples)
    run_start = time.perf_counter()

    if cfg.logging.verbose:
        print("Warming up buffer...")
    while len(pipeline.buffer) < cfg.buffer.warmup_size:
        pipeline.collect_rollout()
        best_cost, best_indices, best_raw_fidelity, best_cnot_count = _update_best_from_latest_rollout(
            pipeline,
            best_cost,
            best_indices,
            best_raw_fidelity,
            best_cnot_count,
        )

    # Warmup is fidelity-only. After it ends, replay is rescored with the
    # stationary QASER-style reward using the frozen warmup references.
    pipeline._warmup_mode = False
    if pipeline.pareto_archive is not None:
        pipeline._rescore_replay_buffer_with_stationary_reward()

    for epoch in range(cfg.training.max_epochs):
        pipeline._current_epoch = epoch

        # Raise the Pareto fidelity floor once training has matured
        if (
            pipeline.pareto_archive is not None
            and epoch == cfg.pareto.floor_ramp_epoch
        ):
            pipeline.pareto_archive.set_fidelity_floor(cfg.pareto.fidelity_floor_late)
            if cfg.logging.verbose:
                print(
                    f"  [Pareto] Epoch {epoch + 1}: "
                    f"fidelity floor raised to {cfg.pareto.fidelity_floor_late}"
                )
        epoch_start = time.perf_counter()

        pipeline.collect_rollout()
        best_cost, best_indices, best_raw_fidelity, best_cnot_count = _update_best_from_latest_rollout(
            pipeline,
            best_cost,
            best_indices,
            best_raw_fidelity,
            best_cnot_count,
        )
        epoch_best_idx = int(np.argmin(pipeline._last_rollout_costs))
        epoch_best_cost = float(pipeline._last_rollout_costs[epoch_best_idx])
        epoch_best_fidelity = float(pipeline._last_rollout_fidelities[epoch_best_idx])
        (
            epoch_best_depth,
            epoch_best_total_gates,
            epoch_best_cnot_count,
        ) = pipeline.sequence_structure_metrics(
            pipeline._last_rollout_indices[epoch_best_idx, 1:]
        )

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

        # ── Pareto metrics for this epoch ───────────────────────────────────
        pareto_size = 0
        pareto_hv = 0.0
        pareto_best_f = float("nan")
        pareto_min_cnot = -1
        pareto_min_depth = -1
        pareto_threshold = float(cfg.pareto.fidelity_threshold)
        epoch_structure_scale = (
            float(
                pipeline.structure_scale_for_fidelity(
                    np.asarray([epoch_best_fidelity], dtype=np.float32)
                )[0]
            )
            if pipeline.pareto_archive is not None
            else float("nan")
        )
        run_structure_scale = (
            float(
                pipeline.structure_scale_for_fidelity(
                    np.asarray([run_best_fidelity], dtype=np.float32)
                )[0]
            )
            if pipeline.pareto_archive is not None and np.isfinite(run_best_fidelity)
            else float("nan")
        )
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

        if cfg.logging.verbose:
            ref_depth = (
                float(pipeline._reward_ref_depth)
                if pipeline._reward_ref_depth is not None
                else float("nan")
            )
            ref_cnot = (
                float(pipeline._reward_ref_cnot)
                if pipeline._reward_ref_cnot is not None
                else float("nan")
            )
            ref_fidelity = float(pipeline._reward_ref_fidelity)
            lambda_depth, lambda_cnot = pipeline._reward_lambdas
            pareto_str = (
                f" | pareto_size={pareto_size}"
                f" | hv={pareto_hv:.4f}"
                f" | lambda=[{lambda_depth:.2f},{lambda_cnot:.2f}]"
                f" | scale=[{epoch_structure_scale:.2f},{run_structure_scale:.2f}]"
                f" | ref=[{ref_depth:.2f},{ref_cnot:.2f},{ref_fidelity:.6f}]"
                if pipeline.pareto_archive is not None
                else ""
            )
            print(
                f"Epoch {epoch + 1:03d}/{cfg.training.max_epochs:03d}"
                f" | cost_epoch_best={epoch_best_cost:.6f}"
                f" | raw_fidelity_epoch_best={epoch_best_fidelity:.6f}"
                f" | depth_epoch_best={epoch_best_depth}"
                f" | total_gates_epoch_best={epoch_best_total_gates}"
                f" | cnot_epoch_best={epoch_best_cnot_count}"
                f" | cost_run_best={best_cost:.6f}"
                f" | raw_fidelity_run_best={run_best_fidelity:.6f}"
                f" | depth_run_best={run_best_depth}"
                f" | total_gates_run_best={run_best_total_gates}"
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
                    "reward_lambda_depth": float(pipeline._reward_lambdas[0]),
                    "reward_lambda_cnot": float(pipeline._reward_lambdas[1]),
                    "reward_structure_scale_epoch_best": epoch_structure_scale,
                    "reward_structure_scale_run_best": run_structure_scale,
                    "reward_ref_depth": (
                        float(pipeline._reward_ref_depth)
                        if pipeline._reward_ref_depth is not None
                        else float("nan")
                    ),
                    "reward_ref_cnot": (
                        float(pipeline._reward_ref_cnot)
                        if pipeline._reward_ref_cnot is not None
                        else float("nan")
                    ),
                    "reward_ref_fidelity": float(pipeline._reward_ref_fidelity),
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

    # ── Post-training Pareto GD optimization ────────────────────────────────
    if (
        cfg.pareto_gd.enabled
        and pipeline.pareto_archive is not None
        and len(pipeline.pareto_archive) > 0
        and u_target is not None
        and pool is not None
    ):
        from pareto_gd_optimizer import ParetoGDOptimizer

        if cfg.logging.verbose:
            print(
                f"\n[ParetoGD] Starting post-training gradient-descent "
                f"optimization on {len(pipeline.pareto_archive)} Pareto circuits "
                f"(optimizer={cfg.pareto_gd.optimizer}, "
                f"steps={cfg.pareto_gd.steps}, "
                f"restarts={cfg.pareto_gd.num_restarts})..."
            )

        gd_opt = ParetoGDOptimizer(
            u_target=u_target,
            num_qubits=cfg.target.num_qubits,
            max_gates=cfg.model.max_gates_count,
            steps=cfg.pareto_gd.steps,
            lr=cfg.pareto_gd.lr,
            optimizer_type=cfg.pareto_gd.optimizer,
            num_restarts=cfg.pareto_gd.num_restarts,
            fidelity_eps=cfg.pareto_gd.fidelity_eps,
        )
        # pool_names_no_bos: gate name strings indexed from 0, BOS excluded
        pool_names_no_bos = [name for name, _ in pool]
        pipeline.pareto_archive, pipeline.rng_key = gd_opt.optimize_archive(
            pipeline.pareto_archive,
            pool_names_no_bos,
            pipeline.rng_key,
            verbose=cfg.logging.verbose,
        )

        if cfg.logging.verbose:
            archive = pipeline.pareto_archive
            top = archive.best_by_fidelity()
            hv = archive.hypervolume_2d()
            best_f = top.fidelity if top is not None else float("nan")
            print(
                f"[ParetoGD] Done — archive size={len(archive)} | "
                f"hv={hv:.6f} | "
                f"best_fidelity={best_f:.6f}"
            )

    pipeline.set_cost(None)
    if logger:
        logger.finalize("success")
    return best_cost, best_indices, pipeline.pareto_archive


def gqe(cost_fn, pool, cfg: GQEConfig, model=None, u_target=None, logger=None):
    factory = Factory()
    if model is None:
        model = GPT2(cfg.model.size, len(pool) + 1)
    pipeline = Pipeline(cfg, cost_fn, pool, model, factory, u_target=u_target)
    return _run_training(cfg, pipeline, logger=logger, u_target=u_target, pool=pool)
