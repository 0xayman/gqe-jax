"""Factory helpers for GQE component construction."""

from continuous_optimizer import ContinuousOptimizer
from loss import GRPOLoss
from scheduler import CosineScheduler, DefaultScheduler, VarBasedScheduler


class Factory:
    def create_loss_fn(self, cfg):
        return GRPOLoss(clip_ratio=cfg.training.grpo_clip_ratio)

    def create_temperature_scheduler(self, cfg):
        scheduler_type = cfg.temperature.scheduler
        scheduler_builders = {
            "fixed": lambda: DefaultScheduler(
                start=cfg.temperature.initial_value,
                delta=cfg.temperature.delta,
                minimum=cfg.temperature.min_value,
                maximum=cfg.temperature.max_value,
            ),
            "cosine": lambda: CosineScheduler(
                minimum=cfg.temperature.min_value,
                maximum=cfg.temperature.max_value,
                frequency=max(1, cfg.training.max_epochs // 2),
            ),
            "variance": lambda: VarBasedScheduler(
                initial=cfg.temperature.initial_value,
                delta=cfg.temperature.delta,
                target_var=1e-5,
            ),
        }
        if scheduler_type not in scheduler_builders:
            available = sorted(scheduler_builders.keys())
            raise ValueError(
                f"Unknown scheduler: {scheduler_type!r}. Available: {available}."
            )
        return scheduler_builders[scheduler_type]()

    def create_continuous_optimizer(self, cfg, u_target, pool) -> ContinuousOptimizer | None:
        del pool
        co_cfg = getattr(cfg, "continuous_opt", None)
        if co_cfg is None or not co_cfg.enabled:
            return None
        return ContinuousOptimizer(
            u_target=u_target,
            num_qubits=cfg.target.num_qubits,
            steps=co_cfg.steps,
            lr=co_cfg.lr,
            optimizer_type=co_cfg.optimizer,
            top_k=co_cfg.top_k,
            max_gates=cfg.model.max_gates_count,
            num_restarts=co_cfg.num_restarts,
        )
