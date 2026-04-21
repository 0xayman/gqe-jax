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


def default_max_gates_count(num_qubits: int) -> int:
    """Upper bound on circuit length, scaling ~2 * 4^n with qubit count.

    Slightly over-estimates a Qiskit-level compilation (~4^n gates in the
    worst case) so the STOP token has room to terminate naturally. Floored
    at 16 for trivial problems; capped at 1023 to stay within the
    transformer's positional-embedding limit
    (``model._N_POSITIONS = 1024``, minus the BOS slot).
    """
    return min(1023, max(16, 2 * (4 ** num_qubits)))


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
    steps_per_epoch: int


@dataclass(frozen=True)
class LoggingConfig:
    verbose: bool
    wandb: bool


@dataclass(frozen=True)
class ContinuousOptConfig:
    enabled: bool = False
    steps: int = 50
    lr: float = 0.1
    optimizer: str = "lbfgs"
    top_k: int = 0
    num_restarts: int = 1


@dataclass(frozen=True)
class RewardConfig:
    """Clean, two-hyperparameter reward for variable-length circuit synthesis.

    R(F, C, D) = F - lambda_structure * (C + gamma_depth * D) / max_gates_count

    - F: process fidelity (after continuous angle optimisation, if enabled).
    - C: CNOT count. D: circuit depth. max_gates_count: model.max_gates_count.
    - lambda_structure: trade-off weight; structure penalty only matters once F
      is close enough to 1 that the fidelity term stops dominating.
    - gamma_depth: relative weight of depth vs CNOT; set to 1.0 by default.

    Pareto archive is retained for reporting and best-circuit selection only;
    it does not feed back into the reward.
    """

    enabled: bool = True
    lambda_structure: float = 0.3
    gamma_depth: float = 1.0

    # Pareto archive (reporting-only)
    fidelity_floor: float = 0.0
    max_archive_size: int = 500
    fidelity_threshold: float = 0.99


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
    reward: RewardConfig = field(default_factory=RewardConfig)


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

    r = raw.get("reward", {})
    if r:
        _require_bool("reward.enabled", r.get("enabled", True))
        if r.get("lambda_structure", 0.3) < 0.0:
            raise ValueError("reward.lambda_structure must be >= 0")
        if r.get("gamma_depth", 1.0) < 0.0:
            raise ValueError("reward.gamma_depth must be >= 0")
        if not (0.0 <= r.get("fidelity_floor", 0.0) <= 1.0):
            raise ValueError("reward.fidelity_floor must be in [0, 1]")
        if r.get("max_archive_size", 500) <= 0:
            raise ValueError("reward.max_archive_size must be positive")
        if not (0.0 < r.get("fidelity_threshold", 0.99) <= 1.0):
            raise ValueError("reward.fidelity_threshold must be in (0, 1]")


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
        continuous_opt=ContinuousOptConfig(**co_raw) if co_raw else ContinuousOptConfig(),
        reward=RewardConfig(**reward_raw) if reward_raw else RewardConfig(),
    )
