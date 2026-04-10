"""Framework-agnostic callback helpers for JAX-based GQE training."""

from __future__ import annotations

import json
import os

import numpy as np


class BestCostCallback:
    def __init__(self):
        self.best_cost = float("inf")
        self.best_indices = None
        self.best_cost_history = []

    def on_train_epoch_end(self, pipeline):
        if len(pipeline.buffer) == 0:
            return
        start = max(0, len(pipeline.buffer) - pipeline.num_samples)
        for i in range(start, len(pipeline.buffer)):
            seq, cost_val = pipeline.buffer.buf[i]
            cost_val = float(np.asarray(cost_val))
            if cost_val < self.best_cost:
                self.best_cost = cost_val
                self.best_indices = seq
        self.best_cost_history.append(self.best_cost)

    def get_results(self):
        return self.best_cost, self.best_indices


class DiagnosticsCallback:
    def on_train_epoch_end(self, pipeline, *, step: int | None = None):
        costs = getattr(pipeline, "_last_rollout_costs", None)
        if costs is None or len(costs) == 0:
            return {}

        costs = np.asarray(costs, dtype=np.float32)
        metrics = {
            "rollout/cost_mean": float(costs.mean()),
            "rollout/cost_min": float(costs.min()),
            "rollout/cost_max": float(costs.max()),
            "rollout/cost_std": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        }
        metrics["rollout/fidelity_mean"] = 1.0 - metrics["rollout/cost_mean"]
        metrics["rollout/fidelity_max"] = 1.0 - metrics["rollout/cost_min"]
        metrics["rollout/gap"] = metrics["rollout/cost_mean"] - metrics["rollout/cost_min"]

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {"rollout/cost_histogram": wandb.Histogram(costs)},
                    step=step,
                )
        except ImportError:
            pass
        return metrics


class TrajectoryCallback:
    def __init__(self, trajectory_file_path: str):
        self.trajectory_file_path = trajectory_file_path
        self.trajectory_data = []

    def on_train_batch_end(self, *, epoch: int, batch_idx: int, loss, batch):
        indices = batch.get("idx")
        costs = batch.get("cost")
        if indices is None or costs is None:
            return
        self.trajectory_data.append(
            {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "loss": float(np.asarray(loss)),
                "indices": np.asarray(indices).tolist(),
                "costs": np.asarray(costs).tolist(),
            }
        )

    def on_train_end(self):
        os.makedirs(os.path.dirname(self.trajectory_file_path), exist_ok=True)
        with open(self.trajectory_file_path, "w") as f:
            for entry in self.trajectory_data:
                f.write(json.dumps(entry) + "\n")
