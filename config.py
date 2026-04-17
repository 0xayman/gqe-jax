from dataclasses import dataclass, field
from typing import Optional

import yaml


def _require_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class TargetConfig:
    num_qubits: int
    type: str
    path: str | None = None


@dataclass(frozen=True)
class PoolConfig:
    rotation_gates: tuple[str, ...] = ("rz",)


@dataclass(frozen=True)
class ModelConfig:
    size: str
    max_gates_count: int


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    num_samples: int
    batch_size: int
    lr: float
    grad_norm_clip: float
    seed: int
    grpo_clip_ratio: float
    early_stop: bool = False  # Stop training once fidelity reaches 1.0


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
    warmup_size: int
    steps_per_epoch: int


@dataclass(frozen=True)
class LoggingConfig:
    verbose: bool
    wandb: bool


@dataclass(frozen=True)
class ContinuousOptConfig:
    enabled: bool = False
    steps: int = 50  # gradient steps per circuit; 50 is enough for L-BFGS
    lr: float = 0.1  # learning rate; higher is fine for L-BFGS
    optimizer: str = "lbfgs"  # "lbfgs" (recommended) or "adam"
    top_k: int = 0  # 0 = optimize all circuits; N = optimize only top-N per rollout
    num_restarts: int = 1  # independent random restarts for angle optimization (≥1)


@dataclass(frozen=True)
class ParetoGDConfig:
    """Configuration for post-training gradient-descent optimization of
    Pareto-optimal circuits.

    After the generative training loop finishes, gradient descent is run on
    the rotation-gate angles of every Pareto-archived circuit whose fidelity
    is below ``1.0 - fidelity_eps``.  The gate structure is kept fixed; only
    the continuous angles are optimised.

    Attributes
    ----------
    enabled:
        Set to ``true`` to activate post-training GD optimization.
    steps:
        Number of optimizer iterations per circuit.
        For L-BFGS 200 steps is usually sufficient; Adam may need more.
    lr:
        Learning rate / step size for the optimizer.
    optimizer:
        ``"lbfgs"`` (recommended) or ``"adam"``.
    num_restarts:
        Number of independent random angle initialisations.  The restart
        with the highest fidelity is kept.  Higher values improve the chance
        of escaping local minima at the cost of more compute.
    fidelity_eps:
        Circuits with fidelity >= ``1 - fidelity_eps`` are considered perfect
        and are skipped.  Default ``1e-6`` treats anything above 0.999999 as
        already optimal.
    """

    enabled: bool = False
    steps: int = 200
    lr: float = 0.1
    optimizer: str = "lbfgs"
    num_restarts: int = 3
    fidelity_eps: float = 1e-6


@dataclass(frozen=True)
class ParetoConfig:
    """Configuration for Pareto-front tracking plus QASER-style reward shaping.

    When enabled, the training pipeline still maintains a Pareto archive over:
      - fidelity (maximize)
      - depth (minimize)
      - cnot_count (minimize)

    Replay training, however, uses a stationary QASER-inspired scalar cost:

      phi(F) = clip(-log(1 - F + eps), 0, phi_max)
      B(D, C) = lambda_depth * (D_ref + 1) / (D + 1)
              + lambda_cnot  * (C_ref + 1) / (C + 1)
      p(F) = clip(
               (F - structure_ramp_start)
               / max(fidelity_threshold - structure_ramp_start, eps),
               0, 1
             )
      s(F) = structure_weight_min
           + (structure_weight_max - structure_weight_min) * p(F) ** structure_ramp_power
      score = phi(F) * max(structure_score_offset + s(F) * (B - 1), eps)
      cost = -score

    The structure references (D_ref, C_ref) are initialized during the
    fidelity-only warmup from the best warmup circuit by fidelity, then kept
    fixed or updated slowly via an EMA when higher-fidelity circuits appear.
    Equal-fidelity but structurally better circuits are allowed to refresh the
    references after warmup so the reward does not freeze once fidelity saturates.

    Legacy Pareto scalarization fields are retained for config compatibility.
    The fidelity threshold is also used as the point where structure pressure
    reaches full strength, in addition to Pareto summary tags plus
    depth-focused final-circuit selection.
    """

    enabled: bool = False          # set to true in config.yml to activate
    fidelity_floor: float = 0.5    # circuits below this fidelity are not archived
    fidelity_floor_late: float = 0.9  # floor raised to this after floor_ramp_epoch
    floor_ramp_epoch: int = 100    # epoch at which the floor is raised
    max_archive_size: int = 500    # cap on archive size (pruned by crowding distance)
    lambda_depth: float = 0.35     # structure bonus weight for depth
    lambda_cnot: float = 0.65      # structure bonus weight for CNOT count
    structure_score_offset: float = 1.0  # additive offset inside phi(F) * (...)
    fidelity_log_eps: float = 1.0e-8  # numerical epsilon in -log(1 - F + eps)
    fidelity_log_cap: float = 12.0  # upper clip for the fidelity shaping term
    reference_ema: float = 0.0     # 0 = freeze refs after warmup; >0 updates slowly
    structure_weight_min: float = 0.15  # nonzero structure pressure even at low fidelity
    structure_weight_max: float = 2.0   # full structure pressure once fidelity is high
    structure_ramp_start: float = 0.90  # keep structure pressure near-min below this fidelity
    structure_ramp_power: float = 4.0   # higher = stay fidelity-first for longer
    alpha_F: float = 3.0           # legacy/unused by reward; kept for compatibility
    alpha_d: float = 1.0           # legacy/unused by reward; kept for compatibility
    alpha_c: float = 1.0           # legacy/unused by reward; kept for compatibility
    w_F_min: float = 0.6           # legacy/unused by reward; kept for compatibility
    fidelity_threshold: float = 0.99  # full structure pressure + Pareto depth/CNOT threshold


@dataclass(frozen=True)
class GQEConfig:
    target: TargetConfig
    model: ModelConfig
    training: TrainingConfig
    temperature: TemperatureConfig
    buffer: BufferConfig
    logging: LoggingConfig
    pool: PoolConfig = field(default_factory=PoolConfig)
    continuous_opt: ContinuousOptConfig = field(default_factory=ContinuousOptConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    pareto_gd: ParetoGDConfig = field(default_factory=ParetoGDConfig)


VALID_TARGET_TYPES = {"random", "random_reachable", "haar_random", "file"}
VALID_MODEL_SIZES = {"tiny", "small", "medium", "large"}
VALID_SCHEDULERS = {"fixed", "linear", "cosine"}
VALID_ROTATION_GATES = {"rz", "rx", "ry"}


def normalize_rotation_gates(value) -> tuple[str, ...]:
    """Normalize configured rotation gates to a lowercase, duplicate-free tuple."""
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
                f"Invalid rotation gate: {gate!r}. "
                f"Allowed values: {sorted(VALID_ROTATION_GATES)}"
            )
        if gate_name in seen:
            raise ValueError(f"Duplicate rotation gate in pool.rotation_gates: {gate_name!r}")
        seen.add(gate_name)
        normalized.append(gate_name)

    if not normalized:
        raise ValueError("pool.rotation_gates must contain at least one gate")
    return tuple(normalized)


def validate_config(raw: dict) -> None:
    """Validate raw config dictionary. Raises ValueError on invalid values."""
    if raw["target"]["num_qubits"] <= 0:
        raise ValueError("num_qubits must be positive")
    if raw["target"]["type"] not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target type: {raw['target']['type']}")
    if raw["target"]["type"] == "file" and not raw["target"].get("path"):
        raise ValueError("target.path is required when target.type='file'")

    if raw["model"]["size"] not in VALID_MODEL_SIZES:
        raise ValueError(f"Invalid model size: {raw['model']['size']}")
    if raw["model"]["max_gates_count"] <= 0:
        raise ValueError("max_gates_count must be positive")

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
    if b["warmup_size"] <= 0:
        raise ValueError("buffer.warmup_size must be positive")
    if b["steps_per_epoch"] <= 0:
        raise ValueError("buffer.steps_per_epoch must be positive")

    _require_bool("logging.verbose", raw["logging"]["verbose"])
    _require_bool("logging.wandb", raw["logging"]["wandb"])

    co = raw.get("continuous_opt", {})
    if co:
        _require_bool("continuous_opt.enabled", co.get("enabled", False))
        if co.get("steps", 1) <= 0:
            raise ValueError("continuous_opt.steps must be positive")
        if co.get("lr", 1.0) <= 0:
            raise ValueError("continuous_opt.lr must be positive")
        if co.get("optimizer", "lbfgs") not in {"lbfgs", "adam"}:
            raise ValueError(
                f"Invalid continuous_opt.optimizer: {co['optimizer']!r}. "
                "Must be 'lbfgs' or 'adam'."
            )
        if co.get("top_k", 0) < 0:
            raise ValueError("continuous_opt.top_k must be >= 0")
        if co.get("num_restarts", 1) < 1:
            raise ValueError("continuous_opt.num_restarts must be >= 1")

    pgd = raw.get("pareto_gd", {})
    if pgd:
        _require_bool("pareto_gd.enabled", pgd.get("enabled", False))
        if pgd.get("steps", 1) <= 0:
            raise ValueError("pareto_gd.steps must be positive")
        if pgd.get("lr", 0.1) <= 0:
            raise ValueError("pareto_gd.lr must be positive")
        if pgd.get("optimizer", "lbfgs") not in {"lbfgs", "adam"}:
            raise ValueError(
                f"Invalid pareto_gd.optimizer: {pgd['optimizer']!r}. "
                "Must be 'lbfgs' or 'adam'."
            )
        if pgd.get("num_restarts", 1) < 1:
            raise ValueError("pareto_gd.num_restarts must be >= 1")
        if not (0.0 < pgd.get("fidelity_eps", 1e-6) < 1.0):
            raise ValueError("pareto_gd.fidelity_eps must be in (0, 1)")

    p = raw.get("pareto", {})
    if p:
        _require_bool("pareto.enabled", p.get("enabled", False))
        if not (0.0 <= p.get("fidelity_floor", 0.5) <= 1.0):
            raise ValueError("pareto.fidelity_floor must be in [0, 1]")
        if not (0.0 <= p.get("fidelity_floor_late", 0.9) <= 1.0):
            raise ValueError("pareto.fidelity_floor_late must be in [0, 1]")
        if p.get("fidelity_floor", 0.5) > p.get("fidelity_floor_late", 0.9):
            raise ValueError("pareto.fidelity_floor must be <= fidelity_floor_late")
        if p.get("floor_ramp_epoch", 100) < 0:
            raise ValueError("pareto.floor_ramp_epoch must be >= 0")
        if p.get("max_archive_size", 500) <= 0:
            raise ValueError("pareto.max_archive_size must be positive")
        if p.get("lambda_depth", 0.35) < 0.0:
            raise ValueError("pareto.lambda_depth must be >= 0")
        if p.get("lambda_cnot", 0.65) < 0.0:
            raise ValueError("pareto.lambda_cnot must be >= 0")
        if p.get("lambda_depth", 0.35) + p.get("lambda_cnot", 0.65) <= 0.0:
            raise ValueError("pareto.lambda_depth + pareto.lambda_cnot must be > 0")
        if p.get("structure_score_offset", 1.0) <= 0.0:
            raise ValueError("pareto.structure_score_offset must be positive")
        if p.get("fidelity_log_eps", 1e-8) <= 0.0:
            raise ValueError("pareto.fidelity_log_eps must be positive")
        if p.get("fidelity_log_cap", 12.0) <= 0.0:
            raise ValueError("pareto.fidelity_log_cap must be positive")
        if not (0.0 <= p.get("reference_ema", 0.0) <= 1.0):
            raise ValueError("pareto.reference_ema must be in [0, 1]")
        if p.get("structure_weight_min", 0.15) < 0.0:
            raise ValueError("pareto.structure_weight_min must be >= 0")
        if p.get("structure_weight_max", 2.0) <= 0.0:
            raise ValueError("pareto.structure_weight_max must be positive")
        if p.get("structure_weight_max", 2.0) < p.get("structure_weight_min", 0.15):
            raise ValueError(
                "pareto.structure_weight_max must be >= pareto.structure_weight_min"
            )
        if not (0.0 <= p.get("structure_ramp_start", 0.90) < p.get("fidelity_threshold", 0.99)):
            raise ValueError(
                "pareto.structure_ramp_start must be in [0, pareto.fidelity_threshold)"
            )
        if p.get("structure_ramp_power", 4.0) <= 0.0:
            raise ValueError("pareto.structure_ramp_power must be positive")
        if p.get("alpha_F", 3.0) <= 0:
            raise ValueError("pareto.alpha_F must be positive")
        if p.get("alpha_d", 1.0) <= 0:
            raise ValueError("pareto.alpha_d must be positive")
        if p.get("alpha_c", 1.0) <= 0:
            raise ValueError("pareto.alpha_c must be positive")
        if not (0.0 <= p.get("w_F_min", 0.6) < 1.0):
            raise ValueError("pareto.w_F_min must be in [0, 1)")
        if not (0.0 < p.get("fidelity_threshold", 0.99) <= 1.0):
            raise ValueError("pareto.fidelity_threshold must be in (0, 1]")


def load_config(path: str) -> GQEConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        GQEConfig: Validated, immutable configuration object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If any config value is invalid.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    validate_config(raw)
    pool_raw = raw.get("pool", {})
    co_raw = raw.get("continuous_opt", {})
    pareto_raw = raw.get("pareto", {})
    pareto_gd_raw = raw.get("pareto_gd", {})
    return GQEConfig(
        target=TargetConfig(**raw["target"]),
        pool=PoolConfig(
            rotation_gates=normalize_rotation_gates(pool_raw.get("rotation_gates")),
        ),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        temperature=TemperatureConfig(**raw["temperature"]),
        buffer=BufferConfig(**raw["buffer"]),
        logging=LoggingConfig(**raw["logging"]),
        continuous_opt=ContinuousOptConfig(**co_raw) if co_raw else ContinuousOptConfig(),
        pareto=ParetoConfig(**pareto_raw) if pareto_raw else ParetoConfig(),
        pareto_gd=ParetoGDConfig(**pareto_gd_raw) if pareto_gd_raw else ParetoGDConfig(),
    )
