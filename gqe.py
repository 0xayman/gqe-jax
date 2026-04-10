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
        return False
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
    best_two_qubit_count: int | None,
):
    if len(pipeline.buffer) == 0:
        return best_cost, best_indices, best_two_qubit_count

    tie_tolerance = 1.0e-7
    start = max(0, len(pipeline.buffer) - pipeline.num_samples)
    for i in range(start, len(pipeline.buffer)):
        seq, cost_val = pipeline.buffer.buf[i]
        cost_value = float(np.asarray(cost_val))
        seq_arr = np.asarray(seq, dtype=np.int32)
        two_qubit_count = int(
            np.count_nonzero(pipeline.two_qubit_token_mask[seq_arr[1:]])
        )
        if cost_value < best_cost - tie_tolerance:
            best_cost = cost_value
            best_indices = seq
            best_two_qubit_count = two_qubit_count
            continue
        if abs(cost_value - best_cost) <= tie_tolerance and (
            best_two_qubit_count is None or two_qubit_count < best_two_qubit_count
        ):
            best_indices = seq
            best_two_qubit_count = two_qubit_count
    return best_cost, best_indices, best_two_qubit_count


def _run_training(cfg: GQEConfig, pipeline: Pipeline, logger=None):
    if logger is None:
        logger = _build_logger(cfg)

    best_cost = float("inf")
    best_indices = None
    best_two_qubit_count = None
    pipeline._starting_idx = np.zeros((pipeline.num_samples, 1), dtype=np.int32)
    run_start = time.perf_counter()

    if cfg.logging.verbose:
        print("Warming up buffer...")
    while len(pipeline.buffer) < cfg.buffer.warmup_size:
        pipeline.collect_rollout()
        best_cost, best_indices, best_two_qubit_count = _update_best_from_latest_rollout(
            pipeline, best_cost, best_indices, best_two_qubit_count
        )

    for epoch in range(cfg.training.max_epochs):
        epoch_start = time.perf_counter()

        pipeline.collect_rollout()
        best_cost, best_indices, best_two_qubit_count = _update_best_from_latest_rollout(
            pipeline, best_cost, best_indices, best_two_qubit_count
        )
        epoch_best_fidelity = 1.0 - float(np.min(pipeline._last_rollout_costs))

        epoch_losses = []
        dataloader = pipeline.train_dataloader()
        for batch_idx, batch in enumerate(
            dataloader.iter_batches(
                cfg.training.batch_size,
                drop_last=True,
            )
        ):
            loss = pipeline.train_batch(batch["idx"], batch["cost"], batch_idx)
            epoch_losses.append(loss)

        epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        )
        fidelity_run_best = 1.0 - best_cost
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - run_start

        if cfg.logging.verbose:
            print(
                f"Epoch {epoch + 1:03d}/{cfg.training.max_epochs:03d}"
                f" | fidelity_epoch_best={epoch_best_fidelity:.6f}"
                f" | fidelity_run_best={fidelity_run_best:.6f}"
                f" | epoch_time={_format_elapsed(epoch_time)}"
                f" | elapsed={_format_elapsed(elapsed)}"
            )

        if logger:
            logger.log_metrics(
                {
                    "loss": epoch_loss,
                    "fidelity_best": fidelity_run_best,
                    "fidelity_epoch_best": epoch_best_fidelity,
                    "inverse_temperature": pipeline.scheduler.get_inverse_temperature(),
                    "epoch_time_sec": epoch_time,
                    "elapsed_time_sec": elapsed,
                },
                step=epoch,
            )

        if cfg.training.early_stop and fidelity_run_best >= 1.0:
            if cfg.logging.verbose:
                print(
                    f"Fidelity 1.0 reached at epoch {epoch + 1:03d}"
                    f" | elapsed={_format_elapsed(elapsed)}"
                )
            break

    if best_indices is not None:
        best_indices = np.asarray(best_indices, dtype=np.int32).tolist()

    pipeline.set_cost(None)
    if logger:
        logger.finalize("success")
    return best_cost, best_indices


def gqe(cost_fn, pool, cfg: GQEConfig, model=None, u_target=None, logger=None):
    factory = Factory()
    if model is None:
        model = GPT2(cfg.model.size, len(pool))
    pipeline = Pipeline(cfg, cost_fn, pool, model, factory, u_target=u_target)
    return _run_training(cfg, pipeline, logger=logger)
