from pathlib import Path
import sys
import textwrap

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
    RewardConfig,
    TargetConfig,
    TemperatureConfig,
    TrainingConfig,
    load_config,
)
from continuous_optimizer import ContinuousOptimizer
from cost import build_cost_fn, process_fidelity
from data import BufferDataset, ReplayBuffer
from factory import Factory
from gqe import gqe
from kv_rollout import compute_lengths_from_tokens
from loss import reduce_sequence_log_probs
from main import _basis_gates, _build_qiskit_circuit, _select_report_token_sequence
from model import GPT2
from operator_pool import build_operator_pool
from pareto import ParetoArchive, ParetoPoint
from pipeline import Pipeline, _sequence_structure_metrics
from scheduler import FixedScheduler, LinearScheduler
from target import build_target


def _tiny_cfg(
    *,
    continuous_opt_enabled: bool,
    batch_size: int = 2,
    pareto_enabled: bool = False,
) -> GQEConfig:
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
        buffer=BufferConfig(max_size=16, steps_per_epoch=1),
        logging=LoggingConfig(verbose=False, wandb=False),
        continuous_opt=ContinuousOptConfig(
            enabled=continuous_opt_enabled,
            optimizer="lbfgs",
            steps=2,
            lr=0.1,
            top_k=1,
            num_restarts=2,
        ),
        reward=RewardConfig(
            enabled=pareto_enabled,
            lambda_structure=0.3,
            gamma_depth=1.0,
            fidelity_floor=0.0,
            max_archive_size=16,
            fidelity_threshold=0.99,
        ),
    )


def _build_pipeline(
    *,
    continuous_opt_enabled: bool,
    batch_size: int = 2,
    pareto_enabled: bool = False,
) -> tuple[Pipeline, list]:
    cfg = _tiny_cfg(
        continuous_opt_enabled=continuous_opt_enabled,
        batch_size=batch_size,
        pareto_enabled=pareto_enabled,
    )
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)
    model = GPT2(cfg.model.size, len(pool) + 2)
    pipeline = Pipeline(cfg, cost_fn, pool, model, Factory(), u_target=u_target)
    return pipeline, pool


def test_discrete_gqe_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=False)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)

    best_cost, best_indices, _best_angles, pareto_archive = gqe(cost_fn, pool, cfg, u_target=u_target)

    assert 0.0 <= best_cost <= 1.0
    assert isinstance(best_indices, list)
    assert len(best_indices) == cfg.model.max_gates_count + 1
    assert pareto_archive is None


def test_continuous_gqe_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)

    best_cost, best_indices, _best_angles, pareto_archive = gqe(cost_fn, pool, cfg, u_target=u_target)

    assert 0.0 <= best_cost <= 1.0
    assert isinstance(best_indices, list)
    assert len(best_indices) == cfg.model.max_gates_count + 1
    assert pareto_archive is None


def test_pareto_gqe_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=False, pareto_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    cost_fn = build_cost_fn(u_target)

    best_cost, best_indices, _best_angles, pareto_archive = gqe(cost_fn, pool, cfg, u_target=u_target)

    assert 0.0 <= best_cost <= 1.0
    assert isinstance(best_indices, list)
    assert len(best_indices) == cfg.model.max_gates_count + 1
    assert pareto_archive is not None


def test_report_selection_prefers_pareto_depth_threshold():
    raw_best = np.asarray([0, 2, 2, 2], dtype=np.int32)
    shallower = np.asarray([0, 3, 3, 3], dtype=np.int32)
    archive = ParetoArchive(max_size=8, fidelity_floor=0.0)
    archive.update(
        ParetoPoint(
            fidelity=1.0,
            depth=18,
            total_gates=3,
            cnot_count=3,
            token_sequence=raw_best,
            epoch=0,
        )
    )
    archive.update(
        ParetoPoint(
            fidelity=0.995,
            depth=16,
            total_gates=3,
            cnot_count=3,
            token_sequence=shallower,
            epoch=1,
        )
    )

    selected, _angles, _stored_f, source = _select_report_token_sequence(
        raw_best,
        None,
        archive,
        min_fidelity=0.99,
    )

    np.testing.assert_array_equal(selected, shallower)
    assert source.startswith("Pareto best depth")
    assert "F≥0.990" in source


def test_continuous_optimizer_smoke():
    cfg = _tiny_cfg(continuous_opt_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
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
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
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

    batch_fidelities, _, _ = optimizer.optimize_batch(circuits, jax.random.PRNGKey(cfg.training.seed))

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


def test_continuous_optimizer_token_id_batch_matches_name_batch():
    """When padded with STOP, the token-id interface matches the name interface."""
    cfg = _tiny_cfg(continuous_opt_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    vocab_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
    optimizer = ContinuousOptimizer(
        u_target=u_target,
        num_qubits=cfg.target.num_qubits,
        steps=2,
        lr=0.1,
        optimizer_type="lbfgs",
        top_k=0,
        max_gates=cfg.model.max_gates_count,
        num_restarts=2,
        pool_token_names=vocab_names,
    )
    circuits = [
        ["RZ_q0", "SX_q1", "CNOT_q0_q1", "RZ_q1"],
        ["SX_q0", "CNOT_q1_q0", "RZ_q1", "SX_q1"],
    ]
    name_to_token = {name: idx for idx, name in enumerate(vocab_names)}
    token_ids = np.asarray(
        [[name_to_token[name] for name in circuit] for circuit in circuits],
        dtype=np.int32,
    )

    by_name, _, _ = optimizer.optimize_batch(circuits, jax.random.PRNGKey(cfg.training.seed))
    by_index, _, _ = optimizer.optimize_token_id_batch(
        token_ids,
        jax.random.PRNGKey(cfg.training.seed),
    )

    np.testing.assert_allclose(
        by_index,
        by_name,
        rtol=1e-6,
        atol=1e-6,
    )


def test_continuous_optimizer_token_id_batch_treats_stop_as_identity():
    """STOP-padded positions should not affect the fidelity of the real prefix."""
    cfg = _tiny_cfg(continuous_opt_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    vocab_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
    optimizer = ContinuousOptimizer(
        u_target=u_target,
        num_qubits=cfg.target.num_qubits,
        steps=2,
        lr=0.1,
        optimizer_type="lbfgs",
        top_k=0,
        max_gates=cfg.model.max_gates_count,
        num_restarts=1,
        pool_token_names=vocab_names,
    )
    name_to_token = {name: idx for idx, name in enumerate(vocab_names)}
    short_circuit = ["RZ_q0", "SX_q1"]
    short_tokens = [name_to_token[n] for n in short_circuit] + [1, 1]  # 1 == STOP
    padded_tokens = np.asarray([short_tokens], dtype=np.int32)

    by_padded, _, _ = optimizer.optimize_token_id_batch(
        padded_tokens,
        jax.random.PRNGKey(cfg.training.seed),
    )
    by_name, _, _ = optimizer.optimize_batch(
        [short_circuit],
        jax.random.PRNGKey(cfg.training.seed),
    )

    np.testing.assert_allclose(by_padded, by_name, rtol=1e-6, atol=1e-6)


def test_sequence_structure_metrics_skips_noop_tokens():
    # vocab: BOS=0, STOP=1, gates start at 2
    token_qubit0 = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int32)
    token_qubit1 = np.asarray([-1, -1, -1, -1, 1, -1], dtype=np.int32)
    token_is_noop = np.asarray([True, True, False, False, False, False], dtype=bool)
    # sequence (without BOS): gate@q0, gate@q1, CNOT(0,1), gate@q1, STOP
    depth, total_gates, cnot_count = _sequence_structure_metrics(
        np.asarray([2, 3, 4, 5, 1], dtype=np.int32),
        num_qubits=2,
        token_qubit0=token_qubit0,
        token_qubit1=token_qubit1,
        token_is_noop=token_is_noop,
    )

    assert depth == 3
    assert total_gates == 4
    assert cnot_count == 1


def test_compute_lengths_from_tokens_handles_stop_and_no_stop():
    # row 0: STOP at position 2 → length 3
    # row 1: no STOP → length = seq_len = 4
    tokens = jnp.asarray(
        [
            [2, 3, 1, 1],
            [2, 3, 4, 5],
        ],
        dtype=jnp.int32,
    )
    lengths = compute_lengths_from_tokens(tokens, stop_token_id=1)
    np.testing.assert_array_equal(np.asarray(lengths), np.asarray([3, 4], dtype=np.int32))


def test_replay_buffer_batches_include_behavior_log_probs():
    buffer = ReplayBuffer(size=8)
    for idx, old_log_prob in enumerate([-1.5, -0.5, -2.0, -1.0]):
        buffer.push([0, idx + 2], 0.1 * idx, old_log_prob)

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


def test_pipeline_rollout_uses_distinct_bos_and_stop_tokens():
    pipeline, pool = _build_pipeline(continuous_opt_enabled=False)

    rollout = pipeline.generate()

    assert rollout.shape == (pipeline.num_samples, pipeline.ngates + 1)
    # Every sequence starts with BOS.
    assert np.all(rollout[:, 0] == pipeline.bos_token_id)
    # Position 1 (first real decision) must not be STOP — min 1 gate.
    assert np.all(rollout[:, 1] != pipeline.stop_token_id)
    # Once STOP appears, every subsequent token is also STOP (forced padding).
    for row in rollout:
        sub = row[1:]
        stop_positions = np.where(sub == pipeline.stop_token_id)[0]
        if stop_positions.size > 0:
            first = stop_positions[0]
            assert np.all(sub[first:] == pipeline.stop_token_id)


def test_grpo_loss_is_zero_when_batch_costs_are_identical():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False)
    beta = jnp.asarray(pipeline.scheduler.get_inverse_temperature(), dtype=jnp.float32)
    # Use token ids >= gate_token_offset so STOP doesn't end the sequence early.
    offset = pipeline.gate_token_offset
    idx = jnp.asarray(
        [
            [0, offset + 0, offset + 1, offset + 2, offset + 0],
            [0, offset + 2, offset + 1, offset + 0, offset + 0],
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


def test_reduce_sequence_log_probs_length_masks_padding():
    # Two samples: first has length 2 (padding beyond), second has length 4.
    lp = jnp.asarray(
        [
            [-0.2, -0.4, -5.0, -5.0],  # padding should not count
            [-0.2, -0.4, -0.6, -0.8],
        ],
        dtype=jnp.float32,
    )
    lengths = jnp.asarray([2, 4], dtype=jnp.int32)

    scores = reduce_sequence_log_probs(lp, lengths)
    expected = np.asarray(
        [
            (-0.2 + -0.4) / 2,
            (-0.2 + -0.4 + -0.6 + -0.8) / 4,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(scores), expected, rtol=0.0, atol=1e-6)


def test_reduce_sequence_log_probs_without_lengths_is_plain_mean():
    lp = jnp.asarray([[-0.2, -0.4, -0.6, -0.8]], dtype=jnp.float32)
    score = reduce_sequence_log_probs(lp)
    np.testing.assert_allclose(np.asarray(score), np.asarray([-0.5]), rtol=0.0, atol=1e-6)


def test_fixed_scheduler_keeps_inverse_temperature_constant():
    scheduler = FixedScheduler(0.5)
    scheduler.update(costs=np.asarray([0.1, 0.2], dtype=np.float32))
    assert scheduler.get_inverse_temperature() == 0.5


def test_linear_scheduler_matches_previous_annealing_behavior():
    scheduler = LinearScheduler(start=0.5, delta=0.2, minimum=0.1, maximum=0.8)
    scheduler.update()
    scheduler.update()
    assert scheduler.get_inverse_temperature() == 0.8


def test_train_batch_updates_params_with_behavior_log_probs():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False)
    offset = pipeline.gate_token_offset
    idx = np.asarray(
        [
            [0, offset + 0, offset + 1, offset + 2, offset + 0],
            [0, offset + 2, offset + 1, offset + 0, offset + 0],
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


def test_reward_prefers_higher_fidelity_at_same_structure():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False, pareto_enabled=True)

    costs = pipeline._compute_reward(
        np.asarray([0.95, 0.99], dtype=np.float32),
        np.asarray([10, 10], dtype=np.int32),
        np.asarray([5, 5], dtype=np.int32),
    )

    # Cost = -reward; higher-fidelity circuit has lower cost.
    assert costs[1] < costs[0]


def test_reward_prefers_better_structure_at_same_fidelity():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False, pareto_enabled=True)

    costs = pipeline._compute_reward(
        np.asarray([0.99, 0.99], dtype=np.float32),
        np.asarray([12, 8], dtype=np.int32),
        np.asarray([6, 4], dtype=np.int32),
    )

    # Cost = -reward; lower (C+gamma*D) has lower cost.
    assert costs[1] < costs[0]


def test_reward_is_simple_linear_formula():
    pipeline, _ = _build_pipeline(continuous_opt_enabled=False, pareto_enabled=True)

    lam = pipeline.cfg.reward.lambda_structure
    gamma = pipeline.cfg.reward.gamma_depth
    L_max = pipeline.ngates

    F = np.asarray([0.5, 0.9], dtype=np.float32)
    D = np.asarray([2, 4], dtype=np.int32)
    C = np.asarray([1, 3], dtype=np.int32)
    costs = pipeline._compute_reward(F, D, C)
    expected = -(F - lam * (C.astype(np.float32) + gamma * D.astype(np.float32)) / L_max)
    np.testing.assert_allclose(costs, expected, rtol=1e-6, atol=1e-6)


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
    backend = jax.default_backend()
    is_gpu = backend in ('gpu', 'cuda')
    verifier = ContinuousOptimizer(
        u_target=u_target,
        num_qubits=num_qubits,
        steps=100,
        lr=0.1,
        optimizer_type="lbfgs",
        top_k=0,
        max_gates=28,
        num_restarts=3,
        fast_runtime=None,
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

    if is_gpu:
        fidelity_threshold = 1.0 - 2e-3
    else:
        fidelity_threshold = 1.0 - 1e-10

    assert verified_f > fidelity_threshold
    assert qc_fidelity > fidelity_threshold


def test_build_operator_pool_uses_configured_rotation_gate_order():
    pool = build_operator_pool(num_qubits=1, rotation_gates=("ry", "rx"))

    assert [name for name, _ in pool] == ["RY_q0", "RX_q0", "SX_q0"]


def test_basis_gates_include_configured_rotations():
    assert _basis_gates(("ry", "rz")) == ["ry", "rz", "sx", "cx"]


def test_load_config_reads_pool_rotation_gates(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        textwrap.dedent(
            """
            target:
              num_qubits: 1
              type: "haar_random"

            pool:
              rotation_gates: ["ry", "rx"]

            model:
              size: "tiny"
              max_gates_count: 4

            training:
              max_epochs: 1
              num_samples: 2
              batch_size: 1
              lr: 1.0e-4
              grad_norm_clip: 1.0
              seed: 0
              grpo_clip_ratio: 0.2
              early_stop: false

            temperature:
              scheduler: "fixed"
              initial_value: 0.5
              delta: 0.01
              min_value: 0.1
              max_value: 1.0

            buffer:
              max_size: 4
              steps_per_epoch: 1

            logging:
              verbose: false
              wandb: false
            """
        )
    )

    cfg = load_config(str(config_path))

    assert cfg.pool.rotation_gates == ("ry", "rx")
