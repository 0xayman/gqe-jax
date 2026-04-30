from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from circuit import CircuitEvaluator, parse_gate_name
from config import (
    BufferConfig,
    GQEConfig,
    LoggingConfig,
    ModelConfig,
    PolicyConfig,
    RefinementConfig,
    RewardConfig,
    TargetConfig,
    TemperatureConfig,
    TrainingConfig,
    load_config,
)
from cost import build_cost_fn, process_fidelity
from data import BufferDataset, BufferEntry, ReplayBuffer
from loss import (
    grpo_advantages,
    joint_sequence_log_prob,
    ppo_clipped_loss,
    reduce_per_token,
)
from main import _basis_gates
from operator_pool import build_operator_pool
from pareto import ParetoArchive, ParetoPoint
from policy import (
    HybridPolicy,
    apply_discrete_action_masks,
    compute_lengths_from_tokens,
    gaussian_log_prob,
)
from refine import AngleRefiner, refine_pareto_archive
from reporting import select_report_token_sequence
from scheduler import FixedScheduler, LinearScheduler
from target import build_target
from trainer import Trainer, _row_structure_metrics, gqe, rollout_context_tokens


def _tiny_cfg(
    *,
    pareto_enabled: bool = False,
    refinement_enabled: bool = False,
) -> GQEConfig:
    return GQEConfig(
        target=TargetConfig(num_qubits=2, type="random_reachable", path=None),
        model=ModelConfig(size="tiny"),
        training=TrainingConfig(
            max_epochs=1,
            num_samples=4,
            batch_size=2,
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
        policy=PolicyConfig(
            entropy_disc=0.0, entropy_cont=0.0,
            angle_supervision_weight=1.0,
            inner_refine_steps=0,
        ),
        refinement=RefinementConfig(
            enabled=refinement_enabled,
            steps=2, lr=0.1,
        ),
        reward=RewardConfig(
            enabled=pareto_enabled,
            fidelity_floor=0.0, max_archive_size=16,
            fidelity_threshold=0.99,
        ),
    )


def _build_trainer(*, pareto_enabled: bool = False) -> tuple[Trainer, list]:
    cfg = _tiny_cfg(pareto_enabled=pareto_enabled)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    return Trainer(cfg, u_target, pool, logger=None), pool


def dataclasses_replace_policy(cfg: GQEConfig, **kwargs) -> GQEConfig:
    """Return ``cfg`` with selected policy fields replaced."""
    import dataclasses
    return dataclasses.replace(
        cfg, policy=dataclasses.replace(cfg.policy, **kwargs),
    )


def test_gqe_smoke_no_pareto():
    cfg = _tiny_cfg(pareto_enabled=False)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)

    result = gqe(cfg, u_target, pool, logger=None)

    assert 0.0 <= result.best_cost <= 1.0
    assert isinstance(result.best_tokens, list)
    assert len(result.best_tokens) == rollout_context_tokens(cfg.target.num_qubits)
    assert result.pareto_archive is None


def test_gqe_smoke_with_pareto_and_refinement():
    cfg = _tiny_cfg(pareto_enabled=True, refinement_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)

    result = gqe(cfg, u_target, pool, logger=None)

    assert np.isfinite(result.best_cost)
    assert result.pareto_archive is not None


def test_circuit_evaluator_recovers_self_built_target():
    """Verify that a unitary built by the evaluator is scored as exact."""
    import jax.numpy as jnp

    num_qubits = 2
    pool = build_operator_pool(num_qubits=num_qubits, rotation_gates=("rz",))
    pool_token_names = ["<BOS>", "<STOP>", "<PAD>", *[name for name, _ in pool]]
    name_to_id = {n: i for i, n in enumerate(pool_token_names)}

    seed_evaluator = CircuitEvaluator(
        u_target=np.eye(2 ** num_qubits, dtype=np.complex128),
        num_qubits=num_qubits, pool_token_names=pool_token_names,
    )
    tokens_build = np.zeros((8,), dtype=np.int32)
    angles_build = np.zeros((8,), dtype=np.float32)
    chosen = [("RZ_q0", 0.7), ("SX_q1", 0.0), ("CNOT_q0_q1", 0.0)]
    for i, (name, theta) in enumerate(chosen):
        tokens_build[i] = name_to_id[name]
        angles_build[i] = theta
    tokens_build[len(chosen):] = 1
    u_target = np.asarray(
        seed_evaluator._build_unitary(jnp.asarray(tokens_build), jnp.asarray(angles_build)),
        dtype=np.complex128,
    )

    evaluator = CircuitEvaluator(
        u_target=u_target, num_qubits=num_qubits,
        pool_token_names=pool_token_names,
    )
    fids = evaluator.fidelity_batch(tokens_build[None, :], angles_build[None, :])
    assert fids.shape == (1,)
    assert fids[0] > 0.999


def test_circuit_evaluator_handles_empty_batch():
    pool = build_operator_pool(num_qubits=2, rotation_gates=("rz",))
    pool_token_names = ["<BOS>", "<STOP>", "<PAD>", *[name for name, _ in pool]]
    evaluator = CircuitEvaluator(
        u_target=np.eye(4, dtype=np.complex128),
        num_qubits=2, pool_token_names=pool_token_names,
    )
    fids = evaluator.fidelity_batch(
        np.zeros((0, 4), dtype=np.int32),
        np.zeros((0, 4), dtype=np.float32),
    )
    assert fids.shape == (0,)


def test_parametric_mask_marks_only_rotations():
    pool = build_operator_pool(num_qubits=2, rotation_gates=("rz", "ry"))
    pool_token_names = ["<BOS>", "<STOP>", "<PAD>", *[name for name, _ in pool]]
    evaluator = CircuitEvaluator(
        u_target=np.eye(4, dtype=np.complex128),
        num_qubits=2, pool_token_names=pool_token_names,
    )
    name_to_id = {n: i for i, n in enumerate(pool_token_names)}
    rz = name_to_id["RZ_q0"]
    sx = name_to_id["SX_q0"]
    cnot = name_to_id["CNOT_q0_q1"]
    bos = 0
    stop = 1
    tokens = np.asarray([bos, rz, sx, cnot, stop], dtype=np.int32)
    mask = evaluator.parametric_mask(tokens)
    assert mask.tolist() == [False, True, False, False, False]


def test_gaussian_log_prob_matches_scipy():
    from math import sqrt, log, pi
    mu = jnp.asarray([0.0, 1.0])
    log_sigma = jnp.asarray([0.0, jnp.log(0.5)])
    x = jnp.asarray([0.5, 0.7])
    got = np.asarray(gaussian_log_prob(x, mu, log_sigma))
    sigma = np.exp(np.asarray(log_sigma))
    expected = -0.5 * (np.log(2 * np.pi) + 2 * np.asarray(log_sigma)
                       + (np.asarray(x) - np.asarray(mu)) ** 2 / sigma ** 2)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_joint_sequence_log_prob_masks_padding_and_non_parametric():
    log_p_d = jnp.asarray([[-0.5, -1.0, -2.0, -2.0]], dtype=jnp.float32)
    log_p_c = jnp.asarray([[-0.1, -0.2, -10.0, -10.0]], dtype=jnp.float32)
    lengths = jnp.asarray([2], dtype=jnp.int32)
    parametric = jnp.asarray([[True, False, True, True]], dtype=bool)
    score = joint_sequence_log_prob(log_p_d, log_p_c, lengths, parametric)
    expected = (-0.5 + -0.1 + -1.0) / 2
    np.testing.assert_allclose(np.asarray(score), [expected], rtol=1e-6, atol=1e-6)


def test_grpo_advantages_normalises_to_zero_mean():
    costs = jnp.asarray([0.1, 0.5, 0.9], dtype=jnp.float32)
    advs = grpo_advantages(costs)
    np.testing.assert_allclose(float(jnp.sum(advs)), 0.0, atol=1e-6)


def test_ppo_clipped_loss_zero_when_advantages_are_zero():
    new = jnp.asarray([-0.5, -0.6])
    old = jnp.asarray([-0.4, -0.7])
    advs = jnp.zeros((2,), dtype=jnp.float32)
    loss = ppo_clipped_loss(new, old, advs, clip_ratio=0.2)
    np.testing.assert_allclose(float(loss), 0.0, atol=1e-6)


def test_reduce_per_token_mean_over_valid_positions():
    per_tok = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
    mask = jnp.asarray([[True, True, False, False]])
    np.testing.assert_allclose(float(reduce_per_token(per_tok, mask)[0]), 1.5, atol=1e-6)


def test_apply_discrete_action_masks_blocks_bos_and_first_stop():
    """BOS is never sampled and STOP is blocked for the first action."""
    logits = jnp.zeros((1, 3, 5), dtype=jnp.float32)
    masked = apply_discrete_action_masks(
        logits, bos_token_id=0, stop_token_id=1, pad_token_id=2,
    )
    arr = np.asarray(masked)
    assert np.isposinf(arr[0, :, 0]).all()
    assert np.isposinf(arr[0, :, 2]).all()
    assert np.isposinf(arr[0, 0, 1])
    assert arr[0, 1, 1] == 0.0
    assert arr[0, 2, 1] == 0.0


def test_compute_lengths_from_tokens_handles_stop_and_no_stop():
    tokens = jnp.asarray([
        [3, 4, 1, 2],
        [3, 4, 5, 6],
    ], dtype=jnp.int32)
    lengths = compute_lengths_from_tokens(tokens, stop_token_id=1)
    np.testing.assert_array_equal(np.asarray(lengths), [3, 4])


def test_trainer_rollout_emits_valid_token_layout():
    trainer, _ = _build_trainer()
    roll = trainer.collect_rollout()
    tokens = roll["tokens"]
    angles = roll["angles"]
    assert tokens.shape == (trainer.num_samples, trainer.context_tokens)
    assert angles.shape == (trainer.num_samples, trainer.action_horizon)
    assert np.all(tokens[:, 0] == trainer.bos_token_id)
    assert np.all(tokens[:, 1] != trainer.stop_token_id)
    for row in tokens:
        sub = row[1:]
        stop_pos = np.where(sub == trainer.stop_token_id)[0]
        if stop_pos.size > 0:
            first = stop_pos[0]
            assert sub[first] == trainer.stop_token_id
            assert np.all(sub[first + 1:] == trainer.pad_token_id)


def test_trainer_train_epoch_changes_params():
    trainer, _ = _build_trainer()
    trainer.collect_rollout()
    initial = trainer.state.params
    trainer.train_epoch(beta_val=0.5)
    after = trainer.state.params
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(initial), jax.tree_util.tree_leaves(after))
    )


def test_fixed_scheduler_is_constant():
    s = FixedScheduler(0.5)
    s.update(costs=np.asarray([0.1, 0.2], dtype=np.float32))
    assert s.get_inverse_temperature() == 0.5


def test_linear_scheduler_clamps_to_max():
    s = LinearScheduler(start=0.5, delta=0.2, minimum=0.1, maximum=0.8)
    s.update()
    s.update()
    assert s.get_inverse_temperature() == 0.8


def test_reward_prefers_higher_fidelity_at_same_structure():
    trainer, _ = _build_trainer(pareto_enabled=True)
    costs = trainer._compute_reward(
        np.asarray([0.95, 0.99], dtype=np.float32),
        np.asarray([10, 10], dtype=np.int32),
        np.asarray([5, 5], dtype=np.int32),
    )
    assert costs[1] < costs[0]


def test_reward_prefers_better_structure_at_same_fidelity():
    trainer, _ = _build_trainer(pareto_enabled=True)
    costs = trainer._compute_reward(
        np.asarray([0.99, 0.99], dtype=np.float32),
        np.asarray([12, 8], dtype=np.int32),
        np.asarray([6, 4], dtype=np.int32),
    )
    assert costs[1] < costs[0]


def test_reward_discount_prefers_shorter_same_quality():
    trainer, _ = _build_trainer(pareto_enabled=True)
    costs = trainer._compute_reward(
        np.asarray([0.99, 0.99], dtype=np.float32),
        np.asarray([8, 8], dtype=np.int32),
        np.asarray([4, 4], dtype=np.int32),
        lengths=np.asarray([3, 12], dtype=np.int32),
    )
    assert costs[0] < costs[1]




def test_replay_buffer_iterates_full_batches_with_log_probs():
    buffer = ReplayBuffer(size=8)
    for i in range(4):
        buffer.push(BufferEntry(
            tokens=np.asarray([0, 2, 3], dtype=np.int32),
            angles=np.asarray([0.1, 0.2], dtype=np.float32),
            cost=0.1 * i,
            advantage=float(i),
            log_p_disc=np.asarray([-1.5, -0.5], dtype=np.float32),
        ))

    dataset = BufferDataset(buffer, repetition=1)
    batch = next(dataset.iter_batches(2, shuffle=True, rng=np.random.default_rng(0)))
    assert set(batch) == {
        "tokens", "angles", "cost", "advantage", "log_p_disc",
    }
    assert batch["tokens"].shape == (2, 3)
    assert batch["angles"].shape == (2, 2)
    assert batch["cost"].shape == (2,)
    assert batch["advantage"].shape == (2,)
    assert batch["log_p_disc"].shape == (2, 2)


def test_select_report_prefers_pareto_best_depth():
    raw_best = np.asarray([0, 3, 3, 3], dtype=np.int32)
    shallower = np.asarray([0, 4, 4, 4], dtype=np.int32)
    archive = ParetoArchive(max_size=8, fidelity_floor=0.0)
    archive.update(ParetoPoint(
        fidelity=1.0, depth=18, total_gates=3, cnot_count=3,
        token_sequence=raw_best, epoch=0,
    ))
    archive.update(ParetoPoint(
        fidelity=0.995, depth=16, total_gates=3, cnot_count=3,
        token_sequence=shallower, epoch=1,
    ))

    selected, _angles, _f, source = select_report_token_sequence(
        raw_best, None, archive, min_fidelity=0.99,
    )
    np.testing.assert_array_equal(selected, shallower)
    assert source.startswith("Pareto best depth")


def test_pareto_dominance_includes_total_gates():
    bloated = np.asarray([0, 3, 4, 1], dtype=np.int32)
    compact = np.asarray([0, 3, 1, 2], dtype=np.int32)
    archive = ParetoArchive(max_size=8, fidelity_floor=0.0)
    assert archive.update(ParetoPoint(
        fidelity=0.99, depth=5, total_gates=3, cnot_count=1,
        token_sequence=bloated, epoch=0,
    ))
    assert archive.update(ParetoPoint(
        fidelity=0.99, depth=5, total_gates=2, cnot_count=1,
        token_sequence=compact, epoch=1,
    ))
    points = archive.to_sorted_list()
    assert len(points) == 1
    assert points[0].total_gates == 2


def test_best_fidelity_tie_breaks_by_compactness():
    high_cnot = np.asarray([0, 3, 4, 5], dtype=np.int32)
    low_cnot = np.asarray([0, 3, 4, 6], dtype=np.int32)
    archive = ParetoArchive(max_size=8, fidelity_floor=0.0)
    assert archive.update(ParetoPoint(
        fidelity=1.0, depth=18, total_gates=24, cnot_count=5,
        token_sequence=high_cnot, epoch=0,
    ))
    assert archive.update(ParetoPoint(
        fidelity=1.0, depth=16, total_gates=25, cnot_count=3,
        token_sequence=low_cnot, epoch=1,
    ))

    best = archive.best_by_fidelity()
    assert best is not None
    assert best.cnot_count == 3
    assert best.depth == 16
    np.testing.assert_array_equal(archive.to_sorted_list()[0].token_sequence, low_cnot)


def test_archive_refinement_revalidates_stored_pair():
    class FakeEvaluator:
        def fidelity_batch(self, token_ids_batch, angles_batch):
            del token_ids_batch
            angles = np.asarray(angles_batch, dtype=np.float32)
            return np.where(angles[:, 0] > 0.5, 0.2, 0.4).astype(np.float32)

    class FakeRefiner:
        evaluator = FakeEvaluator()

        def refine_batch(self, token_ids_batch, init_angles_batch):
            worse_angles = np.asarray(init_angles_batch, dtype=np.float32).copy()
            worse_angles[:, 0] = 1.0
            return (
                self.evaluator.fidelity_batch(token_ids_batch, worse_angles),
                worse_angles,
            )

    archive = ParetoArchive(max_size=8, fidelity_floor=0.0)
    archive.update(ParetoPoint(
        fidelity=1.0,
        depth=1,
        total_gates=1,
        cnot_count=0,
        token_sequence=np.asarray([0, 3, 1, 2], dtype=np.int32),
        epoch=0,
        opt_angles=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
    ))

    refined = refine_pareto_archive(
        archive,
        FakeRefiner(),
        structure_metrics_fn=lambda _toks: (1, 1, 0),
        bos_token_id=0,
        pad_token_id=2,
    )
    point = refined.to_sorted_list()[0]
    np.testing.assert_allclose(point.fidelity, 0.4, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(point.opt_angles, [0.1, 0.0, 0.0])


def test_basis_gates_include_configured_rotations():
    assert _basis_gates(("ry", "rz")) == ["ry", "rz", "sx", "cx"]


def test_build_operator_pool_uses_configured_rotation_gate_order():
    pool = build_operator_pool(num_qubits=1, rotation_gates=("ry", "rx"))
    assert [name for name, _ in pool] == ["RY_q0", "RX_q0", "SX_q0"]


def test_load_config_reads_pool_rotation_gates(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(textwrap.dedent("""
        target:
          num_qubits: 1
          type: "haar_random"

        pool:
          rotation_gates: ["ry", "rx"]

        model:
          size: "tiny"

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
    """))
    cfg = load_config(str(path))
    assert cfg.pool.rotation_gates == ("ry", "rx")


def test_row_structure_metrics_skips_noop_tokens():
    token_qubit0 = np.asarray([0, 0, 0, 0, 1, 0, 1], dtype=np.int32)
    token_qubit1 = np.asarray([-1, -1, -1, -1, -1, 1, -1], dtype=np.int32)
    token_is_noop = np.asarray(
        [True, True, True, False, False, False, False], dtype=bool,
    )
    depth, total, cnots = _row_structure_metrics(
        np.asarray([3, 4, 5, 6, 1, 2], dtype=np.int32),
        num_qubits=2,
        token_qubit0=token_qubit0,
        token_qubit1=token_qubit1,
        token_is_noop=token_is_noop,
    )
    assert depth == 3
    assert total == 4
    assert cnots == 1


def test_inner_refinement_lifts_rollout_fidelity():
    """Inner refinement should not lower rollout fidelity."""
    cfg = _tiny_cfg(pareto_enabled=True)
    cfg = dataclasses_replace_policy(cfg, inner_refine_steps=4)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    trainer = Trainer(cfg, u_target, pool, logger=None)
    roll = trainer.collect_rollout()
    assert (roll["fidelities"] >= roll["raw_fidelities"] - 1e-6).all()


def test_angle_refiner_does_not_decrease_fidelity():
    cfg = _tiny_cfg(pareto_enabled=False)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    pool_token_names = ["<BOS>", "<STOP>", "<PAD>", *[name for name, _ in pool]]
    name_to_id = {n: i for i, n in enumerate(pool_token_names)}

    evaluator = CircuitEvaluator(
        u_target=u_target,
        num_qubits=cfg.target.num_qubits,
        pool_token_names=pool_token_names,
    )
    refiner = AngleRefiner(evaluator, steps=4, lr=0.1)

    tokens = np.zeros((1, 4), dtype=np.int32)
    angles = np.zeros((1, 4), dtype=np.float32)
    tokens[0, 0] = name_to_id["RZ_q0"]
    tokens[0, 1] = name_to_id["SX_q1"]
    tokens[0, 2] = name_to_id["<STOP>"]
    tokens[0, 3] = name_to_id["<PAD>"]
    angles[0, 0] = 0.0

    init_fids = evaluator.fidelity_batch(tokens, angles)
    refined_fids, _ = refiner.refine_batch(tokens, angles)
    assert refined_fids[0] >= init_fids[0] - 1e-6
