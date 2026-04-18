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
class RewardConfig:
    """Configuration for the GQE reward function (see docs/gqe_reward_design.tex).

    Total reward:
        R_total(x) = R_local(x | x_ref) + R_pareto(x; A_t)

    Local comparison reward (piecewise in ΔΦ = Φ(F) - Φ(F_ref)):
        Φ(F) = -log(1 - F + ε_φ)    [log-infidelity utility]
        w(F) = σ((F - F_gate) / τ)   [soft fidelity gate]
        G_D  = [log((D_ref+1)/(D+1))]+    [positive depth gain]
        G_C  = [log((C_ref+1)/(C+1))]+    [positive CNOT gain]
        P_D  = [log((D+1)/(D_ref+1))]+    [depth regression]
        P_C  = [log((C+1)/(C_ref+1))]+    [CNOT regression]

        Regime 1 (ΔΦ < -δ_bad):
            R_local = -α_bad * (exp(κ_bad * (-ΔΦ - δ_bad)) - 1)
        Regime 2 (-δ_bad ≤ ΔΦ < 0):
            R_local = -α_soft * (exp(κ_soft * (-ΔΦ)) - 1)
                    + η * w(F_ref) * (exp(λ_D*G_D + λ_C*G_C) - 1)
        Regime 3 (ΔΦ ≥ 0):
            R_local = α_good * (exp(κ_good * ΔΦ) - 1)
                    + w(F) * (exp(λ_D*G_D + λ_C*G_C) - 1)
                    - w(F) * (μ_D*P_D + μ_C*P_C)

    Pareto proxy bonus:
        R_pareto = β_1 * [non-dominated w.r.t. archive] + β_2 * N_dom

    The archive tracks non-dominated circuits over (fidelity↑, depth↓, cnot↓).
    """

    # ── Pareto archive ────────────────────────────────────────────────────────
    enabled: bool = True
    fidelity_floor: float = 0.5
    fidelity_floor_late: float = 0.9
    floor_ramp_epoch: int = 100
    max_archive_size: int = 500
    fidelity_threshold: float = 0.99   # used for archive queries and logging

    # ── Log-infidelity utility ────────────────────────────────────────────────
    eps_phi: float = 1.0e-8            # ε_φ in Φ(F) = -log(1 - F + ε_φ)

    # ── Soft fidelity gate ────────────────────────────────────────────────────
    f_gate: float = 0.90               # F_gate: centre of sigmoid gate
    tau: float = 0.05                  # τ: gate temperature (smaller = sharper)

    # ── Bad-drop threshold ────────────────────────────────────────────────────
    delta_bad: float = 0.5             # δ_bad: separates mild from clearly bad drops

    # ── Penalty / reward magnitudes and steepnesses ───────────────────────────
    alpha_bad: float = 2.0             # α_bad
    kappa_bad: float = 1.0             # κ_bad
    alpha_soft: float = 0.5            # α_soft
    kappa_soft: float = 1.0            # κ_soft
    alpha_good: float = 1.0            # α_good
    kappa_good: float = 1.0            # κ_good

    # ── Structure terms ───────────────────────────────────────────────────────
    lambda_d: float = 0.4              # λ_D: depth-improvement bonus weight
    lambda_c: float = 0.6              # λ_C: CNOT-improvement bonus weight
    mu_d: float = 0.1                  # μ_D: depth-regression penalty (0 = off)
    mu_c: float = 0.2                  # μ_C: CNOT-regression penalty (0 = off)
    eta: float = 0.3                   # η ∈ [0,1]: structure-reward fraction in mild-drop regime

    # ── Pareto proxy bonus ────────────────────────────────────────────────────
    beta_1: float = 0.5                # β_1: bonus for entering the non-dominated front
    beta_2: float = 0.05               # β_2: bonus per archive point dominated

    # ── Reference update ──────────────────────────────────────────────────────
    reference_ema: float = 0.0         # 0 = hard update; >0 = EMA blend


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

    r = raw.get("reward", {})
    if r:
        _require_bool("reward.enabled", r.get("enabled", True))
        if not (0.0 <= r.get("fidelity_floor", 0.5) <= 1.0):
            raise ValueError("reward.fidelity_floor must be in [0, 1]")
        if not (0.0 <= r.get("fidelity_floor_late", 0.9) <= 1.0):
            raise ValueError("reward.fidelity_floor_late must be in [0, 1]")
        if r.get("fidelity_floor", 0.5) > r.get("fidelity_floor_late", 0.9):
            raise ValueError("reward.fidelity_floor must be <= reward.fidelity_floor_late")
        if r.get("floor_ramp_epoch", 100) < 0:
            raise ValueError("reward.floor_ramp_epoch must be >= 0")
        if r.get("max_archive_size", 500) <= 0:
            raise ValueError("reward.max_archive_size must be positive")
        if not (0.0 < r.get("fidelity_threshold", 0.99) <= 1.0):
            raise ValueError("reward.fidelity_threshold must be in (0, 1]")
        if r.get("eps_phi", 1e-8) <= 0.0:
            raise ValueError("reward.eps_phi must be positive")
        if not (0.0 <= r.get("f_gate", 0.90) < 1.0):
            raise ValueError("reward.f_gate must be in [0, 1)")
        if r.get("tau", 0.05) <= 0.0:
            raise ValueError("reward.tau must be positive")
        if r.get("delta_bad", 0.5) < 0.0:
            raise ValueError("reward.delta_bad must be >= 0")
        for _field in ("alpha_bad", "kappa_bad", "alpha_soft", "kappa_soft",
                       "alpha_good", "kappa_good"):
            if r.get(_field, 1.0) <= 0.0:
                raise ValueError(f"reward.{_field} must be positive")
        if r.get("lambda_d", 0.4) < 0.0:
            raise ValueError("reward.lambda_d must be >= 0")
        if r.get("lambda_c", 0.6) < 0.0:
            raise ValueError("reward.lambda_c must be >= 0")
        if r.get("lambda_d", 0.4) + r.get("lambda_c", 0.6) <= 0.0:
            raise ValueError("reward.lambda_d + reward.lambda_c must be > 0")
        if r.get("mu_d", 0.1) < 0.0:
            raise ValueError("reward.mu_d must be >= 0")
        if r.get("mu_c", 0.2) < 0.0:
            raise ValueError("reward.mu_c must be >= 0")
        if not (0.0 <= r.get("eta", 0.3) <= 1.0):
            raise ValueError("reward.eta must be in [0, 1]")
        if r.get("beta_1", 0.5) < 0.0:
            raise ValueError("reward.beta_1 must be >= 0")
        if r.get("beta_2", 0.05) < 0.0:
            raise ValueError("reward.beta_2 must be >= 0")
        if not (0.0 <= r.get("reference_ema", 0.0) <= 1.0):
            raise ValueError("reward.reference_ema must be in [0, 1]")


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
        reward=RewardConfig(**reward_raw) if reward_raw else RewardConfig(),
        pareto_gd=ParetoGDConfig(**pareto_gd_raw) if pareto_gd_raw else ParetoGDConfig(),
    )
