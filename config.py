from dataclasses import dataclass, field

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
class GrammarConfig:
    enabled: bool = False  # Set True to enforce structural sampling constraints


@dataclass(frozen=True)
class ContinuousOptConfig:
    enabled: bool = False
    steps: int = 50  # gradient steps per circuit; 50 is enough for L-BFGS
    lr: float = 0.1  # learning rate; higher is fine for L-BFGS
    optimizer: str = "lbfgs"  # "lbfgs" (recommended) or "adam"
    top_k: int = 0  # 0 = optimize all circuits; N = optimize only top-N per rollout
    num_restarts: int = 1  # independent random restarts for angle optimization (≥1)


@dataclass(frozen=True)
class GQEConfig:
    target: TargetConfig
    model: ModelConfig
    training: TrainingConfig
    temperature: TemperatureConfig
    buffer: BufferConfig
    logging: LoggingConfig
    continuous_opt: ContinuousOptConfig = field(default_factory=ContinuousOptConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)


VALID_TARGET_TYPES = {"random_reachable", "haar_random", "file"}
VALID_MODEL_SIZES = {"tiny", "small", "medium", "large"}
VALID_SCHEDULERS = {"fixed", "cosine", "variance"}


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
    co_raw = raw.get("continuous_opt", {})
    grammar_raw = raw.get("grammar", {})

    return GQEConfig(
        target=TargetConfig(**raw["target"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        temperature=TemperatureConfig(**raw["temperature"]),
        buffer=BufferConfig(**raw["buffer"]),
        logging=LoggingConfig(**raw["logging"]),
        continuous_opt=ContinuousOptConfig(**co_raw)
        if co_raw
        else ContinuousOptConfig(),
        grammar=GrammarConfig(**grammar_raw) if grammar_raw else GrammarConfig(),
    )
