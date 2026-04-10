"""JAX entrypoint for the Generative Quantum Eigensolver."""

from __future__ import annotations

import os

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


def _update_best_from_latest_rollout(
    pipeline: Pipeline,
    best_cost: float,
    best_indices,
):
    if len(pipeline.buffer) == 0:
        return best_cost, best_indices

    start = max(0, len(pipeline.buffer) - pipeline.num_samples)
    for i in range(start, len(pipeline.buffer)):
        seq, cost_val = pipeline.buffer.buf[i]
        cost_value = float(np.asarray(cost_val))
        if cost_value < best_cost:
            best_cost = cost_value
            best_indices = seq
    return best_cost, best_indices


def _run_training(cfg: GQEConfig, pipeline: Pipeline, logger=None):
    if logger is None:
        logger = _build_logger(cfg)

    best_cost = float("inf")
    best_indices = None
    pipeline._starting_idx = np.zeros((pipeline.num_samples, 1), dtype=np.int32)

    if cfg.logging.verbose:
        print("Warming up buffer...")
    while len(pipeline.buffer) < cfg.buffer.warmup_size:
        pipeline.collect_rollout()
        best_cost, best_indices = _update_best_from_latest_rollout(
            pipeline, best_cost, best_indices
        )

    for epoch in range(cfg.training.max_epochs):
        if cfg.logging.verbose:
            print(f"\nEpoch {epoch + 1:03d}/{cfg.training.max_epochs:03d}")

        pipeline.collect_rollout()
        best_cost, best_indices = _update_best_from_latest_rollout(
            pipeline, best_cost, best_indices
        )

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
        fidelity_best = 1.0 - best_cost

        if cfg.logging.verbose:
            print(f"  loss={epoch_loss:.6f} | fidelity_best={fidelity_best:.6f}")

        if logger:
            logger.log_metrics(
                {
                    "loss": epoch_loss,
                    "fidelity_best": fidelity_best,
                    "inverse_temperature": pipeline.scheduler.get_inverse_temperature(),
                },
                step=epoch,
            )

        if cfg.training.early_stop and fidelity_best >= 1.0:
            if cfg.logging.verbose:
                print("  Fidelity 1.0 reached. Stopping.")
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
