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
from refine import AngleRefiner
from reporting import select_report_token_sequence
from scheduler import FixedScheduler, LinearScheduler
from target import build_target
from trainer import Trainer, _row_structure_metrics, gqe


def _tiny_cfg(
    *,
    pareto_enabled: bool = False,
    refinement_enabled: bool = False,
) -> GQEConfig:
    return GQEConfig(
        target=TargetConfig(num_qubits=2, type="random_reachable", path=None),
        model=ModelConfig(size="tiny", max_gates_count=4),
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
            inner_refine_steps=0,
        ),
        refinement=RefinementConfig(
            enabled=refinement_enabled,
            steps=2, lr=0.1,
            apply_simplify=True,
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
    """Frozen-dataclass helper: return ``cfg`` with ``policy`` fields overridden."""
    import dataclasses
    return dataclasses.replace(
        cfg, policy=dataclasses.replace(cfg.policy, **kwargs),
    )


# ── End-to-end smoke tests ──────────────────────────────────────────────────

def test_gqe_smoke_no_pareto():
    cfg = _tiny_cfg(pareto_enabled=False)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)

    result = gqe(cfg, u_target, pool, logger=None)

    assert 0.0 <= result.best_cost <= 1.0
    assert isinstance(result.best_tokens, list)
    assert len(result.best_tokens) == cfg.model.max_gates_count + 1
    assert result.pareto_archive is None


def test_gqe_smoke_with_pareto_and_refinement():
    cfg = _tiny_cfg(pareto_enabled=True, refinement_enabled=True)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)

    result = gqe(cfg, u_target, pool, logger=None)

    # With reward enabled, cost = -reward = -(F - lambda * structure / L_max),
    # so it can be negative if fidelity outweighs the structural penalty.
    assert np.isfinite(result.best_cost)
    assert result.pareto_archive is not None


# ── Circuit / fidelity ──────────────────────────────────────────────────────

def test_circuit_evaluator_recovers_self_built_target():
    """Build a target with the evaluator itself, then verify F == 1.

    The evaluator and Qiskit-reconstruction path agree on conventions; the
    operator_pool stored matrices use a different (LSB-first) convention and
    are not used in the new training pipeline.
    """
    import jax.numpy as jnp

    num_qubits = 2
    pool = build_operator_pool(num_qubits=num_qubits, rotation_gates=("rz",))
    pool_token_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
    name_to_id = {n: i for i, n in enumerate(pool_token_names)}

    # Use the evaluator with an identity target just to access the unitary
    # builder; then re-instantiate with that built unitary as the actual target.
    seed_evaluator = CircuitEvaluator(
        u_target=np.eye(2 ** num_qubits, dtype=np.complex128),
        num_qubits=num_qubits, pool_token_names=pool_token_names, max_gates=8,
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
        pool_token_names=pool_token_names, max_gates=8,
    )
    fids = evaluator.fidelity_batch(tokens_build[None, :], angles_build[None, :])
    assert fids.shape == (1,)
    assert fids[0] > 0.999


def test_circuit_evaluator_handles_empty_batch():
    pool = build_operator_pool(num_qubits=2, rotation_gates=("rz",))
    pool_token_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
    evaluator = CircuitEvaluator(
        u_target=np.eye(4, dtype=np.complex128),
        num_qubits=2, pool_token_names=pool_token_names, max_gates=4,
    )
    fids = evaluator.fidelity_batch(
        np.zeros((0, 4), dtype=np.int32),
        np.zeros((0, 4), dtype=np.float32),
    )
    assert fids.shape == (0,)


def test_parametric_mask_marks_only_rotations():
    pool = build_operator_pool(num_qubits=2, rotation_gates=("rz", "ry"))
    pool_token_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
    evaluator = CircuitEvaluator(
        u_target=np.eye(4, dtype=np.complex128),
        num_qubits=2, pool_token_names=pool_token_names, max_gates=4,
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


# ── Gaussian helpers ────────────────────────────────────────────────────────

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


# ── Loss helpers ────────────────────────────────────────────────────────────

def test_joint_sequence_log_prob_masks_padding_and_non_parametric():
    log_p_d = jnp.asarray([[-0.5, -1.0, -2.0, -2.0]], dtype=jnp.float32)
    log_p_c = jnp.asarray([[-0.1, -0.2, -10.0, -10.0]], dtype=jnp.float32)
    lengths = jnp.asarray([2], dtype=jnp.int32)
    parametric = jnp.asarray([[True, False, True, True]], dtype=bool)
    score = joint_sequence_log_prob(log_p_d, log_p_c, lengths, parametric)
    # Position 0: log_p_d + log_p_c (parametric, valid)
    # Position 1: log_p_d (non-parametric, valid)
    # Positions 2-3: ignored (beyond length).
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


# ── Policy / rollout ────────────────────────────────────────────────────────

def test_apply_discrete_action_masks_blocks_bos_and_first_stop():
    """Sampling uses softmax(-beta * logits): +inf in a slot suppresses it."""
    logits = jnp.zeros((1, 3, 4), dtype=jnp.float32)
    masked = apply_discrete_action_masks(logits, bos_token_id=0, stop_token_id=1)
    arr = np.asarray(masked)
    # BOS at every position should be +inf (suppressed under softmax(-beta * x)).
    assert np.isposinf(arr[0, :, 0]).all()
    # STOP at position 0 should be +inf (suppressed at first decision).
    assert np.isposinf(arr[0, 0, 1])
    # STOP at later positions should be unaffected.
    assert arr[0, 1, 1] == 0.0
    assert arr[0, 2, 1] == 0.0


def test_compute_lengths_from_tokens_handles_stop_and_no_stop():
    tokens = jnp.asarray([
        [2, 3, 1, 1],
        [2, 3, 4, 5],
    ], dtype=jnp.int32)
    lengths = compute_lengths_from_tokens(tokens, stop_token_id=1)
    np.testing.assert_array_equal(np.asarray(lengths), [3, 4])


def test_trainer_rollout_emits_valid_token_layout():
    trainer, _ = _build_trainer()
    roll = trainer.collect_rollout()
    tokens = roll["tokens"]
    angles = roll["angles"]
    assert tokens.shape == (trainer.num_samples, trainer.ngates + 1)
    assert angles.shape == (trainer.num_samples, trainer.ngates)
    assert np.all(tokens[:, 0] == trainer.bos_token_id)
    assert np.all(tokens[:, 1] != trainer.stop_token_id)
    for row in tokens:
        sub = row[1:]
        stop_pos = np.where(sub == trainer.stop_token_id)[0]
        if stop_pos.size > 0:
            first = stop_pos[0]
            assert np.all(sub[first:] == trainer.stop_token_id)


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


# ── Schedulers ──────────────────────────────────────────────────────────────

def test_fixed_scheduler_is_constant():
    s = FixedScheduler(0.5)
    s.update(costs=np.asarray([0.1, 0.2], dtype=np.float32))
    assert s.get_inverse_temperature() == 0.5


def test_linear_scheduler_clamps_to_max():
    s = LinearScheduler(start=0.5, delta=0.2, minimum=0.1, maximum=0.8)
    s.update()
    s.update()
    assert s.get_inverse_temperature() == 0.8


# ── Reward ──────────────────────────────────────────────────────────────────

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


def test_qaser_log_infidelity_strengthens_high_fidelity_gradient():
    """With qaser_log_infidelity_eps > 0, the marginal reward at high F is
    much larger than with the paper-faithful (eps=0) form.
    """
    import dataclasses
    cfg = _tiny_cfg(pareto_enabled=True)
    cfg = dataclasses.replace(
        cfg, reward=dataclasses.replace(cfg.reward,
                                        qaser_init_max_depth=20,
                                        qaser_init_max_cnot=10,
                                        qaser_init_max_gates=40),
    )
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)

    # Paper-faithful (eps=0): cost gap between F=0.99 and F=0.999 is small.
    cfg_off = dataclasses.replace(
        cfg, reward=dataclasses.replace(cfg.reward, qaser_log_infidelity_eps=0.0),
    )
    trainer_off = Trainer(cfg_off, u_target, pool, logger=None)
    c_off = trainer_off._compute_reward(
        np.asarray([0.99, 0.999], dtype=np.float32),
        np.asarray([5, 5], dtype=np.int32),
        np.asarray([2, 2], dtype=np.int32),
        np.asarray([10, 10], dtype=np.int32),
    )
    gap_off = abs(float(c_off[0] - c_off[1]))

    # Log-inf enabled: the gradient near F=1 should be much larger.
    cfg_on = dataclasses.replace(
        cfg, reward=dataclasses.replace(cfg.reward, qaser_log_infidelity_eps=1e-3),
    )
    trainer_on = Trainer(cfg_on, u_target, pool, logger=None)
    c_on = trainer_on._compute_reward(
        np.asarray([0.99, 0.999], dtype=np.float32),
        np.asarray([5, 5], dtype=np.int32),
        np.asarray([2, 2], dtype=np.int32),
        np.asarray([10, 10], dtype=np.int32),
    )
    gap_on = abs(float(c_on[0] - c_on[1]))
    assert gap_on > gap_off * 5  # at least ~5× steeper near F=1
    # F=0 should give a tiny reward (≈ -log(1+eps) ≈ +eps cost), not collapse
    # to a huge magnitude that would zero-out GRPO advantages by dominating
    # the batch.
    c_zero = trainer_on._compute_reward(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([5], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([10], dtype=np.int32),
    )
    assert abs(float(c_zero[0])) < 1e-2


def test_qaser_reward_strongly_rewards_compact_high_fidelity():
    """QASER reward: structure savings only matter once F is high."""
    import dataclasses
    cfg = _tiny_cfg(pareto_enabled=True)
    cfg = dataclasses.replace(
        cfg, reward=dataclasses.replace(cfg.reward,
                                        qaser_init_max_depth=20,
                                        qaser_init_max_cnot=10),
    )
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    trainer = Trainer(cfg, u_target, pool, logger=None)

    # At high F, compact circuit should beat bloated circuit.
    costs_high_F = trainer._compute_reward(
        np.asarray([0.99, 0.99], dtype=np.float32),
        np.asarray([20, 5], dtype=np.int32),    # bloated vs compact depth
        np.asarray([10, 2], dtype=np.int32),    # bloated vs compact cnots
        np.asarray([40, 8], dtype=np.int32),    # bloated vs compact total gates
    )
    assert costs_high_F[1] < costs_high_F[0]

    # At low F, the reward collapses (≈ 0 for both); structure barely matters.
    costs_low_F = trainer._compute_reward(
        np.asarray([0.05, 0.05], dtype=np.float32),
        np.asarray([20, 5], dtype=np.int32),
        np.asarray([10, 2], dtype=np.int32),
        np.asarray([40, 8], dtype=np.int32),
    )
    spread_low = abs(float(costs_low_F[0] - costs_low_F[1]))
    spread_high = abs(float(costs_high_F[0] - costs_high_F[1]))
    assert spread_high > spread_low * 5


# ── Replay buffer ───────────────────────────────────────────────────────────

def test_replay_buffer_iterates_full_batches_with_log_probs():
    buffer = ReplayBuffer(size=8)
    for i in range(4):
        buffer.push(BufferEntry(
            tokens=np.asarray([0, 2, 3], dtype=np.int32),
            angles=np.asarray([0.1, 0.2], dtype=np.float32),
            cost=0.1 * i,
            log_p_disc=np.asarray([-1.5, -0.5], dtype=np.float32),
            log_p_cont=np.asarray([-0.2, -0.3], dtype=np.float32),
        ))

    dataset = BufferDataset(buffer, repetition=1)
    batch = next(dataset.iter_batches(2, shuffle=True, rng=np.random.default_rng(0)))
    assert set(batch) == {"tokens", "angles", "cost", "log_p_disc", "log_p_cont"}
    assert batch["tokens"].shape == (2, 3)
    assert batch["angles"].shape == (2, 2)
    assert batch["cost"].shape == (2,)
    assert batch["log_p_disc"].shape == (2, 2)
    assert batch["log_p_cont"].shape == (2, 2)


# ── Pareto ──────────────────────────────────────────────────────────────────

def test_select_report_prefers_pareto_best_depth():
    raw_best = np.asarray([0, 2, 2, 2], dtype=np.int32)
    shallower = np.asarray([0, 3, 3, 3], dtype=np.int32)
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


# ── Misc ────────────────────────────────────────────────────────────────────

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
    """))
    cfg = load_config(str(path))
    assert cfg.pool.rotation_gates == ("ry", "rx")


def test_row_structure_metrics_skips_noop_tokens():
    token_qubit0 = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int32)
    token_qubit1 = np.asarray([-1, -1, -1, -1, 1, -1], dtype=np.int32)
    token_is_noop = np.asarray([True, True, False, False, False, False], dtype=bool)
    depth, total, cnots = _row_structure_metrics(
        np.asarray([2, 3, 4, 5, 1], dtype=np.int32),
        num_qubits=2,
        token_qubit0=token_qubit0,
        token_qubit1=token_qubit1,
        token_is_noop=token_is_noop,
    )
    assert depth == 3
    assert total == 4
    assert cnots == 1


# ── Refinement ──────────────────────────────────────────────────────────────

def test_inner_refinement_lifts_rollout_fidelity():
    """With inner_refine_steps>0 the post-refinement fidelity should be at
    least as good as the raw RL fidelity for every rollout sample."""
    cfg = _tiny_cfg(pareto_enabled=True)
    cfg = dataclasses_replace_policy(cfg, inner_refine_steps=4)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    trainer = Trainer(cfg, u_target, pool, logger=None)
    roll = trainer.collect_rollout()
    # post-refinement F is always >= raw F (we clip overshoots).
    assert (roll["fidelities"] >= roll["raw_fidelities"] - 1e-6).all()


def test_angle_refiner_does_not_decrease_fidelity():
    cfg = _tiny_cfg(pareto_enabled=False)
    pool = build_operator_pool(cfg.target.num_qubits, cfg.pool.rotation_gates)
    u_target, _ = build_target(pool, cfg)
    pool_token_names = ["<BOS>", "<STOP>", *[name for name, _ in pool]]
    name_to_id = {n: i for i, n in enumerate(pool_token_names)}

    evaluator = CircuitEvaluator(
        u_target=u_target,
        num_qubits=cfg.target.num_qubits,
        pool_token_names=pool_token_names,
        max_gates=cfg.model.max_gates_count,
    )
    refiner = AngleRefiner(evaluator, steps=4, lr=0.1)

    tokens = np.zeros((1, cfg.model.max_gates_count), dtype=np.int32)
    angles = np.zeros((1, cfg.model.max_gates_count), dtype=np.float32)
    tokens[0, 0] = name_to_id["RZ_q0"]
    tokens[0, 1] = name_to_id["SX_q1"]
    tokens[0, 2:] = 1  # STOP
    angles[0, 0] = 0.0  # bad initial guess

    init_fids = evaluator.fidelity_batch(tokens, angles)
    refined_fids, _ = refiner.refine_batch(tokens, angles)
    assert refined_fids[0] >= init_fids[0] - 1e-6
