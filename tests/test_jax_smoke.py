from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from config import (
    BufferConfig,
    ContinuousOptConfig,
    GQEConfig,
    GrammarConfig,
    LoggingConfig,
    ModelConfig,
    TargetConfig,
    TemperatureConfig,
    TrainingConfig,
)
from continuous_optimizer import ContinuousOptimizer
from cost import build_cost_fn
from gqe import gqe
from operator_pool import build_operator_pool
from target import build_target


def _tiny_cfg(*, continuous_opt_enabled: bool, batch_size: int = 2) -> GQEConfig:
    return GQEConfig(
        target=TargetConfig(num_qubits=2, type="random_reachable", path=None),
        model=ModelConfig(size="tiny", max_gates_count=4),
        training=TrainingConfig(
            max_epochs=1,
            num_samples=4,
            batch_size=batch_size,
            lr=1.0e-4,
            grad_norm_clip=1.0,
            seed=0,
            grpo_clip_ratio=0.2,
            early_stop=False,
        ),
        temperature=TemperatureConfig(
            scheduler="fixed",
            initial_value=0.5,
            delta=0.01,
            min_value=0.1,
            max_value=1.0,
        ),
        buffer=BufferConfig(max_size=16, warmup_size=4, steps_per_epoch=1),
        logging=LoggingConfig(verbose=False, wandb=False),
        continuous_opt=ContinuousOptConfig(
            enabled=continuous_opt_enabled,
            optimizer="lbfgs",
            steps=2,
            lr=0.1,
            top_k=1,
            num_restarts=2,
        ),
        grammar=GrammarConfig(enabled=True),
    )


def test_discrete_gqe_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=False)
    pool = build_operator_pool(cfg.target.num_qubits)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)

    best_cost, best_indices = gqe(cost_fn, pool, cfg, u_target=u_target)

    assert 0.0 <= best_cost <= 1.0
    assert isinstance(best_indices, list)
    assert len(best_indices) == cfg.model.max_gates_count + 1


def test_continuous_optimizer_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits)
    u_target, _ = build_target(pool, cfg)
    optimizer = ContinuousOptimizer(
        u_target=u_target,
        num_qubits=cfg.target.num_qubits,
        steps=2,
        lr=0.1,
        optimizer_type="lbfgs",
        top_k=0,
        max_gates=cfg.model.max_gates_count,
        num_restarts=2,
    )

    fidelity, gate_specs, params, _ = optimizer.optimize_circuit_with_params(
        ["RZ_q0", "SX_q1"],
        jax.random.PRNGKey(cfg.training.seed),
    )

    assert 0.0 <= fidelity <= 1.0
    assert len(gate_specs) >= 1
    assert params.ndim == 1
