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
from cost import build_cost_fn, process_fidelity
from data import BufferDataset, ReplayBuffer
from factory import Factory
from gqe import gqe
from main import _build_qiskit_circuit
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

def test_discrete_gqe_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=False)
    pool = build_operator_pool(cfg.target.num_qubits)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)

    best_cost, best_indices, pareto_archive = gqe(cost_fn, pool, cfg, u_target=u_target)

    assert 0.0 <= best_cost <= 1.0
    assert isinstance(best_indices, list)
    assert len(best_indices) == cfg.model.max_gates_count + 1
    # Pareto disabled in tiny cfg (default ParetoConfig has enabled=False)
    assert pareto_archive is None


def test_continuous_gqe_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)

    best_cost, best_indices, pareto_archive = gqe(cost_fn, pool, cfg, u_target=u_target)

    assert 0.0 <= best_cost <= 1.0
    assert isinstance(best_indices, list)
    assert len(best_indices) == cfg.model.max_gates_count + 1
    assert pareto_archive is None


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


def test_replay_buffer_batches_include_behavior_log_probs():
    buffer = ReplayBuffer(size=8)
    for idx, old_log_prob in enumerate([-1.5, -0.5, -2.0, -1.0]):
        buffer.push([0, idx + 1], 0.1 * idx, old_log_prob)

    dataset = BufferDataset(buffer, repetition=1)
    batch = next(
        dataset.iter_batches(
            2,
            shuffle=True,
            rng=np.random.default_rng(0),
        )
    )

    assert set(batch) == {"idx", "cost", "old_log_prob"}
    assert batch["idx"].shape == (2, 2)
    assert batch["cost"].shape == (2,)
    assert batch["old_log_prob"].shape == (2,)
    assert np.isfinite(batch["old_log_prob"]).all()


def test_replay_buffer_legacy_entries_get_nan_old_log_prob():
    buffer = ReplayBuffer(size=1)
    buffer.buf.append((np.asarray([0, 1], dtype=np.int32), np.float32(0.25)))

    item = buffer[0]

    assert np.isnan(item["old_log_prob"])


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
    old_log_probs = pipeline._rollout_sequence_log_probs(
        pipeline.state.params,
        idx,
        beta,
    )

    loss = pipeline._grpo_loss(
        pipeline.state.params,
        idx,
        identical_costs,
        beta,
        old_log_probs,
    )

    assert float(np.asarray(loss)) == 0.0


def test_train_batch_updates_params_with_behavior_log_probs():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False)
    idx = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [0, 4, 3, 2, 1],
        ],
        dtype=np.int32,
    )
    initial_params = pipeline.state.params
    beta = jnp.asarray(pipeline.scheduler.get_inverse_temperature(), dtype=jnp.float32)
    old_log_probs = np.asarray(
        pipeline._rollout_sequence_log_probs(
            pipeline.state.params,
            jnp.asarray(idx, dtype=jnp.int32),
            beta,
        ),
        dtype=np.float32,
    )

    pipeline.train_batch(
        idx,
        np.asarray([0.1, 0.9], dtype=np.float32),
        old_log_probs,
    )
    params_after_batch = pipeline.state.params

    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree_util.tree_leaves(initial_params),
            jax.tree_util.tree_leaves(params_after_batch),
        )
    )


def test_qiskit_reconstruction_matches_internal_optimizer_unitary():
    from qiskit.quantum_info import Operator

    root = Path(__file__).resolve().parents[1]
    num_qubits = 2
    u_target = np.load(root / "targets" / "haar_random_q2.npy").astype(np.complex128)
    gate_names = [
        "RZ_q0", "SX_q0", "RZ_q0", "RZ_q1", "CNOT_q0_q1", "SX_q0", "RZ_q0",
        "RZ_q1", "SX_q1", "RZ_q1", "RZ_q1", "CNOT_q0_q1", "SX_q0", "RZ_q1",
        "CNOT_q0_q1", "RZ_q0", "RZ_q1", "SX_q1", "CNOT_q1_q0", "RZ_q0",
        "RZ_q1", "SX_q1", "RZ_q1", "CNOT_q1_q0", "RZ_q0", "SX_q0", "RZ_q0",
        "RZ_q1",
    ]
    verifier = ContinuousOptimizer(
        u_target=u_target,
        num_qubits=num_qubits,
        steps=100,
        lr=0.1,
        optimizer_type="lbfgs",
        top_k=0,
        max_gates=28,
        num_restarts=3,
    )

    verified_f, gate_specs, opt_params, _ = verifier.optimize_circuit_with_params(
        gate_names,
        jax.random.PRNGKey(42),
    )
    qc = _build_qiskit_circuit(gate_specs, opt_params, num_qubits)
    qc_fidelity = process_fidelity(
        u_target,
        np.asarray(Operator(qc).data, dtype=np.complex128),
    )

    assert verified_f > 1.0 - 1e-10
    assert qc_fidelity > 1.0 - 1e-10


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
