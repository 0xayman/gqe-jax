from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from config import (
    BufferConfig,
    ContinuousOptConfig,
    GQEConfig,
    LoggingConfig,
    ModelConfig,
    TargetConfig,
    TemperatureConfig,
    TrainingConfig,
)
from continuous_optimizer import ContinuousOptimizer
from cost import build_cost_fn
from factory import Factory
from gqe import gqe
from model import GPT2
from operator_pool import build_operator_pool
from pipeline import Pipeline, _sequence_structure_metrics
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
    )


def _build_pipeline(*, continuous_opt_enabled: bool, batch_size: int = 2) -> tuple[Pipeline, list]:
    cfg = _tiny_cfg(continuous_opt_enabled=continuous_opt_enabled, batch_size=batch_size)
    pool = build_operator_pool(cfg.target.num_qubits)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)
    model = GPT2(cfg.model.size, len(pool) + 1)
    pipeline = Pipeline(cfg, cost_fn, pool, model, Factory(), u_target=u_target)
    return pipeline, pool


def _tree_equal(left, right) -> bool:
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
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


def test_continuous_gqe_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=True)
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


def test_continuous_optimizer_batch_matches_single():
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
        num_restarts=1,
    )
    circuits = [
        ["RZ_q0", "SX_q1"],
        ["SX_q0", "CNOT_q0_q1"],
        ["RX_q1", "RY_q0"],
    ]

    batch_fidelities, _ = optimizer.optimize_batch(circuits, jax.random.PRNGKey(cfg.training.seed))

    expected = []
    rng_key = jax.random.PRNGKey(cfg.training.seed)
    for circuit in circuits:
        fidelity, rng_key = optimizer.optimize_circuit(circuit, rng_key)
        expected.append(fidelity)

    np.testing.assert_allclose(
        batch_fidelities,
        np.asarray(expected, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_continuous_optimizer_index_batch_matches_name_batch():
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
        pool_token_names=[name for name, _ in pool],
    )
    circuits = [
        ["RZ_q0", "SX_q1", "CNOT_q0_q1", "RZ_q1"],
        ["SX_q0", "CNOT_q1_q0", "RZ_q1", "SX_q1"],
    ]
    name_to_idx = {name: idx for idx, (name, _) in enumerate(pool)}
    token_ids = np.asarray(
        [[name_to_idx[name] for name in circuit] for circuit in circuits],
        dtype=np.int32,
    )

    by_name, _ = optimizer.optimize_batch(
        circuits,
        jax.random.PRNGKey(cfg.training.seed),
        simplify=False,
    )
    by_index, _ = optimizer.optimize_token_index_batch(
        token_ids,
        jax.random.PRNGKey(cfg.training.seed),
        simplify=False,
    )

    np.testing.assert_allclose(
        by_index,
        by_name,
        rtol=1e-6,
        atol=1e-6,
    )


def test_sequence_structure_metrics_reports_depth_and_gate_counts():
    depth, total_gates, cnot_count = _sequence_structure_metrics(
        np.asarray([0, 1, 2, 3], dtype=np.int32),
        num_qubits=2,
        token_qubit0=np.asarray([0, 1, 0, 1], dtype=np.int32),
        token_qubit1=np.asarray([-1, -1, 1, -1], dtype=np.int32),
    )

    assert depth == 3
    assert total_gates == 4
    assert cnot_count == 1


def test_pipeline_rollout_uses_distinct_bos_token():
    pipeline, pool = _build_pipeline(continuous_opt_enabled=False)

    rollout = pipeline.generate()

    assert rollout.shape == (pipeline.num_samples, pipeline.ngates + 1)
    assert np.all(rollout[:, 0] == pipeline.bos_token_id)
    assert np.all(rollout[:, 1:] >= pipeline.gate_token_offset)
    assert np.all(rollout[:, 1:] < len(pool) + pipeline.gate_token_offset)


def test_grpo_loss_is_zero_when_batch_costs_are_identical():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False)
    beta = jnp.asarray(pipeline.scheduler.get_inverse_temperature(), dtype=jnp.float32)
    idx = jnp.asarray(
        [
            [0, 1, 2, 3, 4],
            [0, 4, 3, 2, 1],
        ],
        dtype=jnp.int32,
    )
    identical_costs = jnp.asarray([0.25, 0.25], dtype=jnp.float32)

    loss = pipeline._grpo_loss(
        pipeline.state.params,
        pipeline.state.params,
        idx,
        identical_costs,
        beta,
    )

    assert float(np.asarray(loss)) == 0.0


def test_train_batch_freezes_reference_params_for_epoch():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False)
    idx = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [0, 4, 3, 2, 1],
        ],
        dtype=np.int32,
    )
    initial_params = pipeline.state.params

    pipeline.train_batch(idx, np.asarray([0.1, 0.9], dtype=np.float32), batch_idx=0)
    reference_after_batch0 = pipeline._reference_params
    params_after_batch0 = pipeline.state.params

    assert _tree_equal(reference_after_batch0, initial_params)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(reference_after_batch0),
            jax.tree_util.tree_leaves(params_after_batch0),
        )
    )

    pipeline.train_batch(idx[::-1], np.asarray([0.8, 0.2], dtype=np.float32), batch_idx=1)

    assert _tree_equal(pipeline._reference_params, reference_after_batch0)


def test_continuous_optimizer_simplify_batch_with_remaining_params():
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
        num_restarts=1,
        pool_token_names=[name for name, _ in pool],
    )
    name_to_idx = {name: idx for idx, (name, _) in enumerate(pool)}
    token_ids = np.asarray(
        [
            [
                name_to_idx["RZ_q0"],
                name_to_idx["RZ_q0"],
                name_to_idx["RZ_q1"],
                name_to_idx["SX_q1"],
            ]
        ],
        dtype=np.int32,
    )

    fidelities, _ = optimizer.optimize_token_index_batch(
        token_ids,
        jax.random.PRNGKey(cfg.training.seed),
        simplify=True,
    )

    assert fidelities.shape == (1,)
    assert 0.0 <= float(fidelities[0]) <= 1.0
