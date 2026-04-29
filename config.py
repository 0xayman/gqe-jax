"""Configuration schema and loader for the hybrid-action GQE training run."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


def _require_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


# ── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TargetConfig:
    num_qubits: int
    type: str
    path: str | None = None
    # Brickwork-target depth (number of brick layers). Each layer applies
    # Haar-random 2-qubit gates on adjacent pairs, alternating between even
    # offsets (0,1)(2,3)... and odd offsets (1,2)(3,4).... If None, defaults
    # to ``2 * num_qubits`` — enough scrambling to make the resulting unitary
    # behave Haar-like on small systems without becoming gratuitously deep.
    brickwork_depth: int | None = None


@dataclass(frozen=True)
class PoolConfig:
    rotation_gates: tuple[str, ...] = ("rz",)


@dataclass(frozen=True)
class ModelConfig:
    size: str
    max_gates_count: int


def default_max_gates_count(num_qubits: int) -> int:
    """Heuristic gate budget per circuit, scaling with qubit count.

    Anchored on the Shende-Markov-Bullock CNOT lower bound for an arbitrary
    n-qubit unitary::

        cnot_lb(n) = ceil((4^n - 3n - 1) / 4)

    A Qiskit-style decomposition typically wraps each CNOT in ~4 single-qubit
    basis rotations (Euler angles + basis change), so a near-optimal circuit
    needs roughly ``5 * cnot_lb`` total gates. We multiply by 1.5 for agent
    exploration slack — tight enough to discourage bloat but loose enough to
    let the policy still find compact decompositions.

    Floored at 32 (small problems still need a few extra slots beyond the
    minimum) and capped at 1023 (transformer positional-embedding limit
    minus the BOS slot).

    Approximate values produced:

    +------+-------------------+-----------------+
    |  n   |  default budget   |  Qiskit ref     |
    +------+-------------------+-----------------+
    |  1   |        32         |       3         |
    |  2   |        32         |      ~24        |
    |  3   |       105         |      ~96        |
    |  4   |       458         |    ~400-700     |
    +------+-------------------+-----------------+

    Override via ``model.max_gates_count`` in YAML for non-Haar workloads.
    """
    # Ceiling division: cnot_lb(n) = ⌈(4^n - 3n - 1) / 4⌉
    numer = 4 ** num_qubits - 3 * num_qubits - 1
    cnot_lb = max(0, -(-numer // 4))
    raw = int(round(1.5 * 5 * cnot_lb))
    return max(32, min(1023, raw))


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    num_samples: int
    batch_size: int
    lr: float
    grad_norm_clip: float
    seed: int
    grpo_clip_ratio: float
    early_stop: bool = False


@dataclass(frozen=True)
class TemperatureConfig:
    scheduler: str
    initial_value: float
    delta: float
    min_value: float
    max_value: float


@dataclass(frozen=True)
class BufferConfig:
    max_size: int
    steps_per_epoch: int


@dataclass(frozen=True)
class LoggingConfig:
    verbose: bool
    wandb: bool


@dataclass(frozen=True)
class PolicyConfig:
    """Hybrid-action policy hyperparameters.

    - ``entropy_disc`` / ``entropy_cont``: entropy-regularisation weights for
      the categorical and Gaussian heads.
    - ``inner_refine_steps``: per-rollout Adam iterations on the sampled angles
      before scoring (0 = pure RL). The buffer stores the RL-sampled angles so
      the PPO ratio is computed against the actually sampled actions; the
      Pareto archive stores the refined angles.
    - ``inner_refine_lr``: passed to AngleRefiner.
    """

    entropy_disc: float = 0.0
    entropy_cont: float = 0.0
    inner_refine_steps: int = 0
    inner_refine_lr: float = 0.1


@dataclass(frozen=True)
class RefinementConfig:
    """Post-training continuous-parameter refinement of the Pareto archive.

    Each entry's angles get up to ``steps`` iterations of Adam, run once from
    the RL-suggested angles and ``num_restarts - 1`` more times from
    Uniform[-π, π] initialisations; the best per circuit is kept.
    ``enabled=False`` reports the archive as-is.

    Knobs added per arXiv:2601.03123 (Meiburg & Gomathi):

    - ``use_linear_trace_loss``: minimise ``1 − |Tr|/d`` instead of HS
      infidelity ``1 − |Tr|²/d²``. Linear in trace deviation → stronger
      gradient near the optimum.
    - ``early_stop_patience`` / ``early_stop_rel_tol``: bail out of Adam when
      the loss has not relatively-improved by ``rel_tol`` for ``patience``
      consecutive steps. Critical because underparameterised skeletons plateau
      near 1e-3 forever (Sec 4 / Appendix B).
    - ``adaptive_restarts`` / ``restart_fidelity_threshold``: skip random
      restarts on circuits that already cleared the threshold; the paper
      reports random init reliably converges first try for adequate skeletons.
    - ``sweep_passes``: per-parameter closed-form sweep refinement (Sec 3.2.1
      "DMRG-style" updates) applied after Adam. 0 disables.
    - ``simplify_max_passes``: number of fixed-point iterations of the
      structural simplifier; cascades may unlock further merges.
    """

    enabled: bool = True
    steps: int = 50
    lr: float = 0.1
    num_restarts: int = 1
    apply_simplify: bool = True
    use_linear_trace_loss: bool = True
    early_stop_patience: int = 30
    early_stop_rel_tol: float = 1.0e-5
    adaptive_restarts: bool = True
    restart_fidelity_threshold: float = 0.999
    sweep_passes: int = 0
    simplify_max_passes: int = 3


@dataclass(frozen=True)
class RewardConfig:
    """QASER reward (inspired by `arXiv:2511.16272`_)::

        base = w_d * M_D/(D+1) + w_c * M_C/(C+1) + w_g * M_G/(G+1)
        R    = base ** F * (1 - log(1 - F + eps)) - 1     # if eps > 0
        R    = base ** F - 1                              # if eps == 0

    M_D, M_C, M_G are running maxima of depth, CNOT count, and total gate
    count. ``qaser_log_infidelity_eps > 0`` adds an unbounded F → 1 gradient
    on top of the otherwise-saturating ``base ** F`` term. The Pareto archive
    is reporting-only and does not feed back into the reward.

    .. _arXiv\\:2511.16272: https://arxiv.org/abs/2511.16272
    """

    enabled: bool = True
    # If <= 0, running maxima are auto-initialised from the first rollout.
    qaser_init_max_depth: float = 0.0
    qaser_init_max_cnot: float = 0.0
    qaser_init_max_gates: float = 0.0
    qaser_w_depth: float = 1.0
    qaser_w_cnot: float = 1.0
    qaser_w_gates: float = 1.0
    qaser_log_infidelity_eps: float = 0.0
    fidelity_floor: float = 0.0
    max_archive_size: int = 500
    fidelity_threshold: float = 0.99
    # CNOT-pair-repetition admission filter (arXiv:2601.03123, Sec 4.1).
    # When the same (control, target) pair recurs ``> pair_repeat_max`` times
    # within any sliding window of ``pair_repeat_window`` consecutive CNOTs,
    # the structure is "effectively underparameterised" — symmetries collapse
    # parameters in that block and refinement plateaus around 10⁻³. Such
    # rollouts are skipped at Pareto admission to save refinement compute.
    # ``pair_repeat_window <= 0`` disables the filter.
    pair_repeat_window: int = 0
    pair_repeat_max: int = 1


@dataclass(frozen=True)
class GQEConfig:
    target: TargetConfig
    model: ModelConfig
    training: TrainingConfig
    temperature: TemperatureConfig
    buffer: BufferConfig
    logging: LoggingConfig
    pool: PoolConfig = field(default_factory=PoolConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


# ── Validation ──────────────────────────────────────────────────────────────

VALID_TARGET_TYPES = {
    "random", "random_reachable", "haar_random", "brickwork", "file",
}
VALID_MODEL_SIZES = {"tiny", "small", "medium", "large"}
VALID_SCHEDULERS = {"fixed", "linear", "cosine"}
VALID_ROTATION_GATES = {"rz", "rx", "ry"}


def normalize_rotation_gates(value) -> tuple[str, ...]:
    if value is None:
        return PoolConfig().rotation_gates
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        raise ValueError("pool.rotation_gates must be a string or a list of strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for gate in candidates:
        if not isinstance(gate, str):
            raise ValueError("pool.rotation_gates entries must be strings")
        gate_name = gate.strip().lower()
        if gate_name not in VALID_ROTATION_GATES:
            raise ValueError(
                f"Invalid rotation gate {gate!r}. Allowed: {sorted(VALID_ROTATION_GATES)}"
            )
        if gate_name in seen:
            raise ValueError(f"Duplicate rotation gate: {gate_name!r}")
        seen.add(gate_name)
        normalized.append(gate_name)

    if not normalized:
        raise ValueError("pool.rotation_gates must contain at least one gate")
    return tuple(normalized)


def validate_config(raw: dict) -> None:
    if raw["target"]["num_qubits"] <= 0:
        raise ValueError("num_qubits must be positive")
    if raw["target"]["type"] not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target type: {raw['target']['type']}")
    if raw["target"]["type"] == "file" and not raw["target"].get("path"):
        raise ValueError("target.path is required when target.type='file'")
    if raw["target"].get("brickwork_depth") is not None and raw["target"]["brickwork_depth"] <= 0:
        raise ValueError("target.brickwork_depth must be positive when provided")

    if raw["model"]["size"] not in VALID_MODEL_SIZES:
        raise ValueError(f"Invalid model size: {raw['model']['size']}")
    if "max_gates_count" in raw["model"] and raw["model"]["max_gates_count"] <= 0:
        raise ValueError("max_gates_count must be positive when provided")

    normalize_rotation_gates(raw.get("pool", {}).get("rotation_gates"))

    t = raw["training"]
    if t["max_epochs"] <= 0:
        raise ValueError("max_epochs must be positive")
    if t["num_samples"] <= 0:
        raise ValueError("num_samples must be positive")
    if t["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if t["lr"] <= 0:
        raise ValueError("learning rate must be positive")
    if t["grad_norm_clip"] <= 0:
        raise ValueError("grad_norm_clip must be positive")
    if t["grpo_clip_ratio"] <= 0:
        raise ValueError("grpo_clip_ratio must be positive")

    temp = raw["temperature"]
    if temp["scheduler"] not in VALID_SCHEDULERS:
        raise ValueError(f"Invalid scheduler: {temp['scheduler']}")
    if temp["min_value"] > temp["max_value"]:
        raise ValueError("temperature.min_value must be <= temperature.max_value")
    if not (temp["min_value"] <= temp["initial_value"] <= temp["max_value"]):
        raise ValueError("initial_value must lie within [min_value, max_value]")

    b = raw["buffer"]
    if b["max_size"] <= 0:
        raise ValueError("buffer.max_size must be positive")
    if b["steps_per_epoch"] <= 0:
        raise ValueError("buffer.steps_per_epoch must be positive")

    _require_bool("logging.verbose", raw["logging"]["verbose"])
    _require_bool("logging.wandb", raw["logging"]["wandb"])

    p = raw.get("policy", {})
    if p:
        if p.get("entropy_disc", 0.0) < 0:
            raise ValueError("policy.entropy_disc must be >= 0")
        if p.get("entropy_cont", 0.0) < 0:
            raise ValueError("policy.entropy_cont must be >= 0")
        if p.get("inner_refine_steps", 0) < 0:
            raise ValueError("policy.inner_refine_steps must be >= 0")
        if p.get("inner_refine_lr", 0.1) <= 0:
            raise ValueError("policy.inner_refine_lr must be positive")

    r = raw.get("refinement", {})
    if r:
        _require_bool("refinement.enabled", r.get("enabled", True))
        if r.get("steps", 1) <= 0:
            raise ValueError("refinement.steps must be positive")
        if r.get("lr", 1.0) <= 0:
            raise ValueError("refinement.lr must be positive")
        if r.get("num_restarts", 1) < 1:
            raise ValueError("refinement.num_restarts must be >= 1")
        _require_bool("refinement.apply_simplify", r.get("apply_simplify", True))
        if "use_linear_trace_loss" in r:
            _require_bool(
                "refinement.use_linear_trace_loss", r["use_linear_trace_loss"]
            )
        if r.get("early_stop_patience", 1) <= 0:
            raise ValueError("refinement.early_stop_patience must be positive")
        if r.get("early_stop_rel_tol", 1.0) < 0:
            raise ValueError("refinement.early_stop_rel_tol must be >= 0")
        if "adaptive_restarts" in r:
            _require_bool("refinement.adaptive_restarts", r["adaptive_restarts"])
        if not (0.0 < r.get("restart_fidelity_threshold", 0.999) <= 1.0):
            raise ValueError(
                "refinement.restart_fidelity_threshold must be in (0, 1]"
            )
        if r.get("sweep_passes", 0) < 0:
            raise ValueError("refinement.sweep_passes must be >= 0")
        if r.get("simplify_max_passes", 1) <= 0:
            raise ValueError("refinement.simplify_max_passes must be positive")

    rw = raw.get("reward", {})
    if rw:
        _require_bool("reward.enabled", rw.get("enabled", True))
        if rw.get("qaser_init_max_depth", 0.0) < 0.0:
            raise ValueError("reward.qaser_init_max_depth must be >= 0")
        if rw.get("qaser_init_max_cnot", 0.0) < 0.0:
            raise ValueError("reward.qaser_init_max_cnot must be >= 0")
        if rw.get("qaser_init_max_gates", 0.0) < 0.0:
            raise ValueError("reward.qaser_init_max_gates must be >= 0")
        if rw.get("qaser_w_depth", 1.0) < 0.0:
            raise ValueError("reward.qaser_w_depth must be >= 0")
        if rw.get("qaser_w_cnot", 1.0) < 0.0:
            raise ValueError("reward.qaser_w_cnot must be >= 0")
        if rw.get("qaser_w_gates", 1.0) < 0.0:
            raise ValueError("reward.qaser_w_gates must be >= 0")
        if rw.get("qaser_log_infidelity_eps", 0.0) < 0.0:
            raise ValueError("reward.qaser_log_infidelity_eps must be >= 0")
        if not (0.0 <= rw.get("fidelity_floor", 0.0) <= 1.0):
            raise ValueError("reward.fidelity_floor must be in [0, 1]")
        if rw.get("max_archive_size", 500) <= 0:
            raise ValueError("reward.max_archive_size must be positive")
        if not (0.0 < rw.get("fidelity_threshold", 0.99) <= 1.0):
            raise ValueError("reward.fidelity_threshold must be in (0, 1]")
        if rw.get("pair_repeat_window", 0) < 0:
            raise ValueError("reward.pair_repeat_window must be >= 0")
        if rw.get("pair_repeat_max", 1) < 1:
            raise ValueError("reward.pair_repeat_max must be >= 1")


def load_config(path: str) -> GQEConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    validate_config(raw)
    pool_raw = raw.get("pool", {})
    policy_raw = raw.get("policy", {})
    refinement_raw = raw.get("refinement", {})
    reward_raw = raw.get("reward", {})

    num_qubits = int(raw["target"]["num_qubits"])
    model_raw = dict(raw["model"])
    if model_raw.get("max_gates_count") is None:
        model_raw["max_gates_count"] = default_max_gates_count(num_qubits)

    return GQEConfig(
        target=TargetConfig(**raw["target"]),
        pool=PoolConfig(
            rotation_gates=normalize_rotation_gates(pool_raw.get("rotation_gates")),
        ),
        model=ModelConfig(**model_raw),
        training=TrainingConfig(**raw["training"]),
        temperature=TemperatureConfig(**raw["temperature"]),
        buffer=BufferConfig(**raw["buffer"]),
        logging=LoggingConfig(**raw["logging"]),
        policy=PolicyConfig(**policy_raw) if policy_raw else PolicyConfig(),
        refinement=RefinementConfig(**refinement_raw) if refinement_raw else RefinementConfig(),
        reward=RewardConfig(**reward_raw) if reward_raw else RewardConfig(),
    )
