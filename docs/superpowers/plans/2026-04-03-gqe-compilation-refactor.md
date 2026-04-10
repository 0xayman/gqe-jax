# GQE Quantum Compilation Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the GQE codebase to remove CUDA-Q/MPI dependencies, replace the chemistry operator pool and Hamiltonian cost with a Qiskit-based hardware-gate pool and process-fidelity cost for unitary compilation, externalize all configuration into `config.yml`, and provide a working `main.py` demo.

**Architecture:** The transformer (GPT-2) autoregressively generates sequences of gate indices from a discrete operator pool. Each gate in the pool is a concrete unitary matrix (NumPy). The pool consists of discretized `RX/RY/RZ` rotations plus fixed `SX` and `CNOT` gates. Given a sequence, we left-multiply the matrices to get `U_circuit`, then compute process fidelity against `U_target`; the cost is `1 - F_process`. Configuration is loaded from `config.yml` via PyYAML into frozen dataclasses.

**Plug-and-play extension points:** Two registries make the system easy to extend without modifying core logic: (1) `factory.py` contains a `SCHEDULER_BUILDERS` dict mapping `temperature.scheduler` YAML values to constructor lambdas — adding a new scheduler is 1 class + 1 dict entry; (2) `target.py` contains a `TARGET_REGISTRY` dict mapping `target.type` YAML values to generator functions — adding a new target unitary strategy is 1 function + 1 dict entry. The project uses a **flat module structure** (no package `__init__.py`); all imports are **absolute**.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Lightning, HuggingFace Transformers (GPT-2), Qiskit (gate matrices only — no simulation), NumPy, PyYAML, pytest.

---

## Background: The GQE Algorithm (from the paper)

### Paper Mapping to Code

The paper (Nakaji et al., 2024) defines GQE with:

- **Operator pool** `G = {U_j}^L_{j=1}` — L unitary matrices (vocabulary). The code calls this `pool`, a list of `(name, matrix)` tuples.
- **Circuit** `U_N(j) = U_{j_N} · ... · U_{j_1}` — the gate sequence is written right-to-left so **j_1 (first in the list) is applied first** to the state, but appears **rightmost** in the matrix product. In code: `U_circuit = gate[N-1] @ ... @ gate[0]`, i.e. later gates are leftmost.
- **Current cost** `E(H, j) = Tr(H · U_N(j) · ρ₀ · U_N(j)†)` — expectation of the Hamiltonian.
- **New cost** `C(j) = 1 - F_process(U_target, U_circuit)` where `F_process = |Tr(U_target† @ U_circuit)|² / d²`.
- **Why it's a drop-in replacement:** The training loop only uses that the cost function returns a float. Nothing in the loss functions, replay buffer, or sampling logic references the physics.

### Training Loop (paper Sec. 2.2.2)

Each epoch in the paper:
1. **Collect** N_sample new circuits via autoregressive sampling from GPT.
2. **Evaluate** each circuit's cost (paper: QPU; new code: matrix multiplication).
3. **Store** `(sequence, cost)` pairs in the replay buffer (FIFO, max N_buffer entries).
4. **Train** the model for N_iter gradient steps using batched data from the buffer.

In Lightning, steps 1-3 run in `on_train_epoch_start → collect_rollout()`. Step 4 happens via `train_dataloader() + training_step()`. The `steps_per_epoch` config key controls how many times the buffer is repeated, so total gradient steps per epoch ≈ `len(buffer) * steps_per_epoch / num_samples`.

### Sampling (paper Eq. 3)

The GPT model outputs logits `w`. Token j is sampled with probability ∝ `exp(-β·w_j)` where β is the inverse temperature. In code: `Categorical(logits=-β·w)`. Low logit → high probability. The model is trained so that gate sequences producing low cost get high logits selected by the loss.

### Process Fidelity — The New Cost

For an n-qubit system with dimension `d = 2^n`:

```
F_process(U_target, U_circuit) = |Tr(U_target† @ U_circuit)|² / d²
```

- Value is 1.0 when U_circuit == U_target (up to global phase).
- Value is 0 when unitaries are completely unrelated.
- **Cost** = `1 - F_process` (minimize cost = maximize fidelity).

---

## Pre-existing State (read before starting)

Before implementing, understand what already exists:

| File | State | Action |
|------|-------|--------|
| `config.yml` | Exists with correct YAML schema | **Keep as-is** |
| `config.py` | Partially exists — dataclasses only, missing `validate_config` and `load_config` | **Complete it** |
| `pool.py` | Stub file (`print("HI")`) | **Delete** in Task 9 |
| `utils.py` | Full cudaq/cudaq_solvers code | **Gut** in Task 9 |
| `pipeline.py` | Uses relative imports + cudaq | **Rewrite** in Task 4 |
| `gqe.py` | Uses relative imports + cudaq | **Rewrite** in Task 8 |
| `factory.py` | Uses relative imports + old cfg keys | **Rewrite** in Task 7 |
| `callbacks.py` | `MinEnergyCallback` + `TrajectoryCallback` | **Rename/update** in Task 5 |
| `loss.py` | `energies` parameter name | **Rename** in Task 6 |
| `scheduler.py` | `DefaultScheduler` without clamping | **Update** in Task 7 |
| `data.py` | `energy` key in buffer | **Rename** in Task 4 |
| `requirements.txt` | Has cudaq, mpi4py, ml-collections | **Update** in Task 9 |

**Critical import rule:** This project is a **flat module** (no `__init__.py`). Never use relative imports (`from .data import ...`). Always use absolute imports (`from data import ...`).

---

## File Map — What This Plan Creates/Modifies

### Files to CREATE

| File | Responsibility |
|------|----------------|
| `operator_pool.py` | Build the hardware gate pool from Qiskit gate definitions. Returns `List[Tuple[str, np.ndarray]]`. |
| `cost.py` | Process fidelity cost function: takes gate matrices + target unitary → returns `1 - F`. |
| `target.py` | **Target unitary generator registry.** Defines the `TargetGenerator` protocol, implements `random_reachable` and `haar_random` generators, and exposes `build_target(pool, cfg)` as the single public API. Adding a new strategy = 1 function + 1 registry entry. |
| `main.py` | Entry point: load config, build pool, call `build_target()`, run GQE, print results. |
| `tests/__init__.py` | Empty file to make `tests/` a package so pytest discovers it. |
| `tests/test_config.py` | Tests for config loading and validation. |
| `tests/test_operator_pool.py` | Tests for pool construction: gate matrices, unitarity, pool size. |
| `tests/test_cost.py` | Tests for process fidelity and compilation cost. |
| `tests/test_target.py` | Tests for both built-in target generators: unitarity, correct dimension, reproducibility. |
| `tests/test_pipeline_integration.py` | Integration tests: Pipeline initializes and runs without cudaq. |

### Files to MODIFY

| File | What changes |
|------|-------------|
| `config.py` | Add `validate_config()` and `load_config()` functions (dataclasses already exist). |
| `pipeline.py` | Remove cudaq/mpi4py; rewrite `computeCost()`; rename `energies`→`costs`; use `cfg.training.*` and `cfg.buffer.*` keys; absolute imports. |
| `data.py` | Rename `energy` key to `cost` in `ReplayBuffer` and `BufferDataset`. |
| `callbacks.py` | Rename `MinEnergyCallback` → `BestCostCallback`; remove `benchmark_energy` references; rename logging keys to `best_cost`/`best_fidelity`. |
| `loss.py` | Rename `energies` parameter to `costs` in all `compute()` methods and `calc_advantage()`. |
| `scheduler.py` | Add `minimum`/`maximum` clamping to `DefaultScheduler`; update `VarBasedScheduler` to accept `costs` kwarg. |
| `factory.py` | Rewrite to use `GQEConfig` nested structure; use a **`SCHEDULER_BUILDERS` registry dict** to resolve all 3 scheduler types (`fixed`, `cosine`, `variance`); absolute imports. |
| `gqe.py` | Remove cudaq; remove `ConfigDict`; use `GQEConfig`; absolute imports; rename internal function. |
| `utils.py` | Replace all content with redirect comment. |
| `requirements.txt` | Remove cudaq, cudaq-solvers, mpi4py, ml-collections; add pyyaml. |

### Dependency Flow

```
config.yml
    │
    ▼
config.py ──► main.py
                │
    ┌───────────┼─────────────────┐
    ▼           ▼                 ▼
operator_pool.py  cost.py      target.py ← TARGET_REGISTRY (plug-and-play)
    │                │             │
    └────────────────┘             │
             │                    │
             ▼                    ▼
           gqe.py ◄──────────── main.py
             │
             ▼
         pipeline.py
             │
    ┌────────┼────────────────────┐
    ▼        ▼                    ▼
model.py  data.py    loss.py / factory.py / callbacks.py
                                  │
                             scheduler.py
                          SCHEDULER_BUILDERS (plug-and-play)
```

---

## Task 0: Setup

**Why first:** The `tests/` directory and dependencies must exist before any code can be tested.

- [ ] **Step 0.1: Create the tests directory**

```bash
mkdir -p /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch/tests
touch /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch/tests/__init__.py
```

- [ ] **Step 0.2: Install pyyaml if not present**

```bash
cd /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch
pip install pyyaml --quiet
```

- [ ] **Step 0.3: Confirm qiskit is importable**

```bash
python -c "from qiskit.circuit.library import RXGate, RYGate, RZGate, SXGate, CXGate; print('Qiskit OK')"
```

Expected: `Qiskit OK`

- [ ] **Step 0.4: Commit the tests directory**

```bash
git add tests/__init__.py
git commit -m "chore: add tests directory"
```

---

## Task 1: Complete `config.py`

**Why first:** Every other module reads from this config. The dataclasses already exist in `config.py`. This task adds `validate_config()` and `load_config()`.

**Files:**
- Modify: `config.py` (add two functions at the bottom)
- Modify: `config.yml` (update to good demo defaults)
- Create: `tests/test_config.py`

- [ ] **Step 1.1: Write failing tests for config**

Create `tests/test_config.py`:

```python
import pytest
import yaml

from config import GQEConfig, load_config, validate_config


class TestLoadConfig:

    def test_load_default_config(self, tmp_path):
        """Loading a valid YAML produces a GQEConfig with correct values."""
        raw = {
            "target": {"num_qubits": 2, "type": "random"},
            "pool": {"discretization_factor": 64},
            "model": {"size": "small", "max_gates_count": 20},
            "training": {
                "max_epochs": 100, "num_samples": 20, "batch_size": 16, "lr": 5e-7,
                "grad_norm_clip": 1.0, "seed": 3047, "grpo_clip_ratio": 0.2,
            },
            "temperature": {
                "scheduler": "fixed", "initial_value": 0.5, "delta": 0.02,
                "min_value": 0.1, "max_value": 1.5,
            },
            "buffer": {"max_size": 1000, "warmup_size": 100, "steps_per_epoch": 10},
            "logging": {"verbose": True, "wandb": True},
        }
        cfg_path = tmp_path / "config.yml"
        cfg_path.write_text(yaml.dump(raw))

        cfg = load_config(str(cfg_path))

        assert isinstance(cfg, GQEConfig)
        assert cfg.target.num_qubits == 2
        assert cfg.target.type == "random"
        assert cfg.pool.discretization_factor == 64
        assert cfg.model.size == "small"
        assert cfg.model.small is True     # property
        assert cfg.model.max_gates_count == 20
        assert cfg.training.lr == pytest.approx(5e-7)
        assert cfg.training.max_epochs == 100
        assert cfg.training.grpo_clip_ratio == pytest.approx(0.2)
        assert cfg.buffer.max_size == 1000
        assert cfg.logging.wandb is True

    def test_missing_file_raises(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yml")


class TestValidateConfig:

    def _base_raw(self):
        return {
            "target": {"num_qubits": 2, "type": "random"},
            "pool": {"discretization_factor": 64},
            "model": {"size": "small", "max_gates_count": 20},
            "training": {
                "max_epochs": 100, "num_samples": 20, "batch_size": 16, "lr": 5e-7,
                "grad_norm_clip": 1.0, "seed": 3047, "grpo_clip_ratio": 0.2,
            },
            "temperature": {
                "scheduler": "fixed", "initial_value": 0.5, "delta": 0.02,
                "min_value": 0.1, "max_value": 1.5,
            },
            "buffer": {"max_size": 1000, "warmup_size": 100, "steps_per_epoch": 10},
            "logging": {"verbose": True, "wandb": True},
        }

    def test_valid_config_passes(self):
        validate_config(self._base_raw())  # must not raise

    def test_zero_qubits_raises(self):
        raw = self._base_raw()
        raw["target"]["num_qubits"] = 0
        with pytest.raises(ValueError, match="num_qubits must be positive"):
            validate_config(raw)

    def test_zero_discretization_raises(self):
        raw = self._base_raw()
        raw["pool"]["discretization_factor"] = 0
        with pytest.raises(ValueError, match="discretization_factor must be positive"):
            validate_config(raw)

    def test_unknown_model_size_raises(self):
        raw = self._base_raw()
        raw["model"]["size"] = "giant"
        with pytest.raises(ValueError, match="Invalid model size"):
            validate_config(raw)

    def test_initial_temp_out_of_bounds_raises(self):
        raw = self._base_raw()
        raw["temperature"]["initial_value"] = 2.0  # > max_value 1.5
        with pytest.raises(ValueError, match="initial_value must lie within"):
            validate_config(raw)

    def test_wandb_not_bool_raises(self):
        raw = self._base_raw()
        raw["logging"]["wandb"] = "yes"
        with pytest.raises(ValueError, match="logging.wandb must be a boolean"):
            validate_config(raw)
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch
python -m pytest tests/test_config.py -v 2>&1 | head -30
```

Expected: `ImportError` — `load_config` and `validate_config` do not exist yet.

- [ ] **Step 1.3: Add `validate_config` and `load_config` to `config.py`**

Open `config.py` and append the following to the bottom (after the `GQEConfig` dataclass):

```python
VALID_TARGET_TYPES = {"random", "random_reachable", "haar_random"}  # extend as new generators are added to target.py
VALID_MODEL_SIZES = {"small", "base"}
VALID_SCHEDULERS = {"fixed", "cosine", "variance"}  # extend as new schedulers are added to factory.py


def _require_bool(name: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def validate_config(raw: dict) -> None:
    """Validate raw config dictionary. Raises ValueError on invalid values."""
    if raw["target"]["num_qubits"] <= 0:
        raise ValueError("num_qubits must be positive")
    if raw["target"]["type"] not in VALID_TARGET_TYPES:
        raise ValueError(f"Invalid target type: {raw['target']['type']}")

    if raw["pool"]["discretization_factor"] <= 0:
        raise ValueError("discretization_factor must be positive")

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

    return GQEConfig(
        target=TargetConfig(**raw["target"]),
        pool=PoolConfig(**raw["pool"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        temperature=TemperatureConfig(**raw["temperature"]),
        buffer=BufferConfig(**raw["buffer"]),
        logging=LoggingConfig(**raw["logging"]),
    )
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_config.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 1.5: Update `config.yml` to sensible demo defaults**

The existing `config.yml` uses `discretization_factor: 64` (slow for demos) and `wandb: True` (requires W&B login). Replace it with demo-friendly values:

```yaml
# config.yml — GQE Quantum Compilation Configuration

target:
  num_qubits: 2
  type: "random"            # Build a random reachable target from the pool

pool:
  discretization_factor: 32 # Fewer angle bins for faster demo (32 × 2qubits × 3axes = 192 rotations)

model:
  size: "small"              # 6 attention layers, 6 heads (vs default 12/12)
  max_gates_count: 10        # Sequence length N

training:
  max_epochs: 50
  num_samples: 30            # Circuits generated per epoch (rollout size)
  batch_size: 16             # Gradient-step batch size (drawn from replay buffer)
  lr: 1.0e-4
  grad_norm_clip: 1.0
  seed: 3047
  grpo_clip_ratio: 0.2

temperature:
  scheduler: "fixed"
  initial_value: 0.5
  delta: 0.02
  min_value: 0.1
  max_value: 1.5

buffer:
  max_size: 100
  warmup_size: 30            # Collect 30 circuits before training starts
  steps_per_epoch: 5

logging:
  verbose: true
  wandb: false               # Set true when you want W&B metric logging
```

- [ ] **Step 1.6: Verify config loads from the updated file**

```bash
python -c "from config import load_config; cfg = load_config('config.yml'); print('OK:', cfg.target.num_qubits, 'qubits,', cfg.pool.discretization_factor, 'bins')"
```

Expected: `OK: 2 qubits, 32 bins`

- [ ] **Step 1.7: Commit**

```bash
git add config.py config.yml tests/test_config.py
git commit -m "feat: complete config.py with validate_config and load_config"
```

---

## Task 2: Operator Pool (`operator_pool.py`)

**Why second:** The pool is the vocabulary. Both the cost function (Task 3) and the pipeline (Task 4) depend on it. Building it first means each can be tested independently.

**Design:** Each gate is a `(name: str, matrix: np.ndarray)` tuple. For an n-qubit system, every matrix is `2^n × 2^n`. The pool contains:
- `RX(θ)`, `RY(θ)`, `RZ(θ)` on every qubit, where θ ∈ `{0, 2π·k/D : k=0,...,D-1}` with `D = discretization_factor`
- `SX` on every qubit
- `CNOT` on every ordered qubit pair (control, target)

Pool size formula for n qubits, D bins: `3·D·n + n + n·(n-1)`

**Single-qubit embedding:** Use Kronecker product. `gate_on_qubit_q = I ⊗ ... ⊗ gate ⊗ ... ⊗ I` where `gate` appears at position `q` (qubit 0 is leftmost = most significant bit in Qiskit convention).

**Two-qubit embedding:** Map each column of the full Hilbert space through the 2-qubit gate using bit manipulation. This handles non-adjacent qubits correctly for n > 2.

**Files:**
- Create: `operator_pool.py`
- Create: `tests/test_operator_pool.py`

- [ ] **Step 2.1: Write failing tests for operator pool**

Create `tests/test_operator_pool.py`:

```python
import numpy as np
import pytest

from operator_pool import (
    build_operator_pool,
    get_discretized_angles,
    get_fixed_gate_matrix,
    get_rotation_matrix,
)


class TestAngleDiscretization:

    def test_four_bins_cover_quarter_turns(self):
        """D=4 should produce angles [0, π/2, π, 3π/2]."""
        angles = get_discretized_angles(4)
        np.testing.assert_allclose(
            angles,
            np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]),
            atol=1e-10,
        )

    def test_first_angle_is_zero(self):
        """linspace with endpoint=False always starts at 0."""
        assert get_discretized_angles(8)[0] == pytest.approx(0.0)

    def test_zero_factor_raises(self):
        with pytest.raises(ValueError, match="discretization_factor must be positive"):
            get_discretized_angles(0)


class TestGateMatrices:

    def test_rx_pi_equals_minus_i_x(self):
        """RX(π) = -iX (up to global phase conventions)."""
        rx_pi = get_rotation_matrix("RX", np.pi)
        expected = -1j * np.array([[0, 1], [1, 0]], dtype=np.complex128)
        np.testing.assert_allclose(rx_pi, expected, atol=1e-10)

    def test_sx_squared_is_x(self):
        """SX·SX = X (the square-root of X property)."""
        sx = get_fixed_gate_matrix("SX")
        x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        np.testing.assert_allclose(sx @ sx, x, atol=1e-10)

    def test_cnot_shape(self):
        """CXGate returns a 4×4 matrix."""
        assert get_fixed_gate_matrix("CNOT").shape == (4, 4)

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError):
            get_rotation_matrix("RQ", 0.5)

    def test_unknown_gate_raises(self):
        with pytest.raises(ValueError):
            get_fixed_gate_matrix("TOFFOLI")


class TestBuildOperatorPool:

    def test_names_include_expected_entries(self):
        """Pool names must encode axis, qubit, and bin index."""
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        names = [name for name, _ in pool]
        assert "RX_q0_k0" in names
        assert "RY_q1_k3" in names
        assert "RZ_q0_k2" in names
        assert "SX_q1" in names
        assert "CNOT_q0_q1" in names
        assert "CNOT_q1_q0" in names

    def test_pool_size_formula(self):
        """Pool size = 3·D·n + n + n·(n-1)."""
        # n=2, D=4: 3*4*2 + 2 + 2*1 = 24 + 2 + 2 = 28
        pool_d4 = build_operator_pool(num_qubits=2, discretization_factor=4)
        assert len(pool_d4) == (3 * 4 * 2) + 2 + 2

        # n=2, D=8: 3*8*2 + 2 + 2 = 52
        pool_d8 = build_operator_pool(num_qubits=2, discretization_factor=8)
        assert len(pool_d8) == (3 * 8 * 2) + 2 + 2

    def test_all_matrices_are_unitary(self):
        """Every matrix must satisfy M·M† = I."""
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        d = 2 ** 2
        for name, matrix in pool:
            assert matrix.shape == (d, d), f"{name}: wrong shape {matrix.shape}"
            np.testing.assert_allclose(
                matrix @ matrix.conj().T,
                np.eye(d, dtype=np.complex128),
                atol=1e-10,
                err_msg=f"{name} is not unitary",
            )

    def test_single_qubit_pool_no_cnot(self):
        """1-qubit pool has only rotations and SX (no CNOT — no qubit pairs)."""
        pool = build_operator_pool(num_qubits=1, discretization_factor=4)
        names = [name for name, _ in pool]
        assert all("CNOT" not in n for n in names)
        # size = 3*4*1 + 1 + 0 = 13
        assert len(pool) == (3 * 4 * 1) + 1
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_operator_pool.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'operator_pool'`

- [ ] **Step 2.3: Implement `operator_pool.py`**

Create `operator_pool.py`:

```python
"""Operator pool for GQE quantum compilation.

The pool is fixed in code and consists of:
  - Discretized RX / RY / RZ rotations on every qubit
  - SX (√X gate) on every qubit
  - CNOT on every ordered qubit pair

The user-configured knob is `discretization_factor` (D): how many evenly
spaced angles in [0, 2π) are used for the Pauli rotation families.

Pool size: 3·D·n + n + n·(n−1)
  where n = num_qubits, D = discretization_factor.

Each entry is a (name, matrix) tuple. The matrix is a (2^n × 2^n) complex128
unitary embedded into the full n-qubit Hilbert space.

Import convention: absolute imports only (flat module structure).
"""

from itertools import permutations
from typing import List, Tuple

import numpy as np
from qiskit.circuit.library import CXGate, RXGate, RYGate, RZGate, SXGate


def get_discretized_angles(discretization_factor: int) -> np.ndarray:
    """Return D evenly spaced angles in [0, 2π), endpoint excluded."""
    if discretization_factor <= 0:
        raise ValueError("discretization_factor must be positive")
    return np.linspace(
        0.0,
        2 * np.pi,
        num=discretization_factor,
        endpoint=False,
        dtype=np.float64,
    )


def get_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Return the 2×2 unitary for RX, RY, or RZ at the given angle."""
    if axis == "RX":
        gate = RXGate(angle)
    elif axis == "RY":
        gate = RYGate(angle)
    elif axis == "RZ":
        gate = RZGate(angle)
    else:
        raise ValueError(f"Unknown rotation axis: {axis!r}")
    return np.array(gate.to_matrix(), dtype=np.complex128)


def get_fixed_gate_matrix(name: str) -> np.ndarray:
    """Return the matrix for a non-parameterized gate: SX or CNOT."""
    if name == "SX":
        return np.array(SXGate().to_matrix(), dtype=np.complex128)
    if name == "CNOT":
        return np.array(CXGate().to_matrix(), dtype=np.complex128)
    raise ValueError(f"Unknown fixed gate: {name!r}")


def _embed_single_qubit(
    gate_2x2: np.ndarray,
    qubit: int,
    num_qubits: int,
) -> np.ndarray:
    """Embed a 2×2 gate into the full n-qubit Hilbert space via Kronecker product.

    Qiskit convention: qubit 0 is the most significant bit (leftmost).
    """
    factors = [
        gate_2x2 if q == qubit else np.eye(2, dtype=np.complex128)
        for q in range(num_qubits)
    ]
    result = factors[0]
    for f in factors[1:]:
        result = np.kron(result, f)
    return result


def _embed_two_qubit(
    gate_4x4: np.ndarray,
    control: int,
    target: int,
    num_qubits: int,
) -> np.ndarray:
    """Embed a 4×4 two-qubit gate into the full n-qubit Hilbert space.

    Handles non-adjacent qubits correctly for num_qubits > 2.
    Qiskit convention: qubit 0 is the most significant bit.
    """
    d = 2 ** num_qubits
    full = np.zeros((d, d), dtype=np.complex128)

    for col in range(d):
        # Decompose column index into individual qubit bits (bit 0 = qubit 0 = MSB)
        bits = [(col >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        # Pack the two relevant qubits into a 2-bit index (control = MSB)
        two_q_in = (bits[control] << 1) | bits[target]

        for two_q_out in range(4):
            amp = gate_4x4[two_q_out, two_q_in]
            if abs(amp) < 1e-15:
                continue
            # Reconstruct full-space row from two_q_out
            new_bits = bits.copy()
            new_bits[control] = (two_q_out >> 1) & 1
            new_bits[target] = two_q_out & 1
            row = sum(new_bits[q] << (num_qubits - 1 - q) for q in range(num_qubits))
            full[row, col] += amp

    return full


def build_operator_pool(
    num_qubits: int,
    discretization_factor: int,
) -> List[Tuple[str, np.ndarray]]:
    """Build the compilation operator pool.

    Returns a list of (name, matrix) tuples ordered as:
      1. All RX/RY/RZ rotations for each qubit and each angle bin
      2. SX for each qubit
      3. CNOT for each ordered (control, target) qubit pair

    Args:
        num_qubits: Number of qubits in the system (determines matrix dimension 2^n).
        discretization_factor: Number of evenly-spaced angle bins D for rotations.

    Returns:
        List of (name, 2^n × 2^n complex128 unitary matrix) tuples.
    """
    pool: List[Tuple[str, np.ndarray]] = []
    angles = get_discretized_angles(discretization_factor)

    for qubit in range(num_qubits):
        for axis in ("RX", "RY", "RZ"):
            for k, angle in enumerate(angles):
                base = get_rotation_matrix(axis, float(angle))
                full = _embed_single_qubit(base, qubit, num_qubits)
                pool.append((f"{axis}_q{qubit}_k{k}", full))

        sx = get_fixed_gate_matrix("SX")
        pool.append((f"SX_q{qubit}", _embed_single_qubit(sx, qubit, num_qubits)))

    cnot = get_fixed_gate_matrix("CNOT")
    for ctrl, tgt in permutations(range(num_qubits), 2):
        pool.append((f"CNOT_q{ctrl}_q{tgt}", _embed_two_qubit(cnot, ctrl, tgt, num_qubits)))

    return pool
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_operator_pool.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add operator_pool.py tests/test_operator_pool.py
git commit -m "feat: add discretized Qiskit operator pool for compilation"
```

---

## Task 3: Cost Function (`cost.py`)

**Why third:** With the pool in place, we can implement and independently test the cost function before touching the pipeline.

**The formula:**
```
U_circuit = gate_matrices[N-1] @ ... @ gate_matrices[0]   # gate[0] applied first
F_process  = |Tr(U_target† @ U_circuit)|² / d²
cost       = 1 − F_process   ∈ [0, 1]
```

**Why this ordering?** The paper writes `U_N(j) = U_{j_N} · ... · U_{j_1}`, meaning j_1 (index 0 in the list) is applied first to the quantum state, so it appears rightmost in the matrix product.

**Files:**
- Create: `cost.py`
- Create: `tests/test_cost.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_cost.py`:

```python
import numpy as np
import pytest

from cost import build_cost_fn, compilation_cost, process_fidelity
from operator_pool import build_operator_pool, get_rotation_matrix


class TestProcessFidelity:

    def test_identical_unitaries_give_one(self):
        """F(U, U) = 1.0 for any unitary U."""
        U = get_rotation_matrix("RX", np.pi / 3)
        assert process_fidelity(U, U) == pytest.approx(1.0, abs=1e-10)

    def test_global_phase_invariance(self):
        """F(U, e^{iφ}·U) = 1.0 — global phase doesn't affect fidelity."""
        U = get_rotation_matrix("RY", np.pi / 4)
        U_phased = np.exp(1j * 1.23) * U
        assert process_fidelity(U, U_phased) == pytest.approx(1.0, abs=1e-10)

    def test_fidelity_in_unit_interval(self):
        """Fidelity is always in [0, 1]."""
        rx = get_rotation_matrix("RX", np.pi)
        rz = get_rotation_matrix("RZ", np.pi)
        f = process_fidelity(rx, rz)
        assert 0.0 <= f <= 1.0

    def test_2qubit_identity(self):
        """F(I₄, I₄) = 1.0 for the 2-qubit identity."""
        I4 = np.eye(4, dtype=np.complex128)
        assert process_fidelity(I4, I4) == pytest.approx(1.0, abs=1e-10)


class TestCompilationCost:

    def test_perfect_match_gives_zero_cost(self):
        """If the circuit exactly produces U_target, cost = 0."""
        # 1-qubit pool, D=4 → angles [0, π/2, π, 3π/2]; k=2 → angle = π
        pool = dict(build_operator_pool(num_qubits=1, discretization_factor=4))
        target = get_rotation_matrix("RX", np.pi)
        cost = compilation_cost([pool["RX_q0_k2"]], target)
        assert cost == pytest.approx(0.0, abs=1e-10)

    def test_empty_circuit_against_nontrivial_target_has_positive_cost(self):
        """Identity circuit (no gates) against RX(π) is not zero cost."""
        target = get_rotation_matrix("RX", np.pi)
        cost = compilation_cost([], target)
        assert cost > 0.1

    def test_two_rx_half_pi_compose_to_rx_pi(self):
        """Two RX(π/2) compose to RX(π), reducing cost vs a single gate."""
        pool = dict(build_operator_pool(num_qubits=1, discretization_factor=4))
        target = get_rotation_matrix("RX", np.pi)
        # k=1 → angle = π/2
        one_gate_cost = compilation_cost([pool["RX_q0_k1"]], target)
        two_gate_cost = compilation_cost([pool["RX_q0_k1"], pool["RX_q0_k1"]], target)
        assert two_gate_cost < one_gate_cost

    def test_cost_in_unit_interval(self):
        """Cost is always in [0, 1]."""
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        target = np.eye(4, dtype=np.complex128)
        matrices = [m for _, m in pool[:5]]
        cost = compilation_cost(matrices, target)
        assert 0.0 <= cost <= 1.0


class TestBuildCostFn:

    def test_returns_float(self):
        """build_cost_fn returns a callable that returns a Python float."""
        target = np.eye(2, dtype=np.complex128)
        cost_fn = build_cost_fn(target)
        result = cost_fn([np.eye(2, dtype=np.complex128)])
        assert isinstance(result, float)

    def test_identity_target_and_identity_circuit(self):
        """Cost of identity circuit against identity target = 0."""
        target = np.eye(4, dtype=np.complex128)
        cost_fn = build_cost_fn(target)
        assert cost_fn([np.eye(4, dtype=np.complex128)]) == pytest.approx(0.0, abs=1e-10)
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_cost.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'cost'`

- [ ] **Step 3.3: Implement `cost.py`**

Create `cost.py`:

```python
"""Process fidelity cost function for quantum unitary compilation.

Given a target unitary U_target and an ordered sequence of gate matrices,
computes cost = 1 − F_process where:

    F_process = |Tr(U_target† @ U_circuit)|² / d²

The circuit unitary is:
    U_circuit = gate_matrices[-1] @ ... @ gate_matrices[0]

i.e. gate_matrices[0] is applied first to the quantum state (rightmost
in matrix product), matching the paper's notation U_N(j) = U_{j_N}·...·U_{j_1}.

Import convention: absolute imports only (flat module structure).
"""

from typing import Callable, List

import numpy as np


def process_fidelity(u_target: np.ndarray, u_circuit: np.ndarray) -> float:
    """Compute process fidelity between two unitary matrices.

    F = |Tr(U_target† @ U_circuit)|² / d²

    This measure is invariant under global phase and equals 1.0 iff the
    two unitaries are identical up to a global phase factor.

    Args:
        u_target: Target unitary (d × d complex matrix).
        u_circuit: Circuit unitary (d × d complex matrix).

    Returns:
        Process fidelity in [0, 1].
    """
    d = u_target.shape[0]
    trace = np.trace(u_target.conj().T @ u_circuit)
    return float((abs(trace) ** 2) / (d ** 2))


def compilation_cost(gate_matrices: List[np.ndarray], u_target: np.ndarray) -> float:
    """Compute compilation cost for an ordered gate sequence.

    U_circuit = gate_matrices[-1] @ ... @ gate_matrices[0]
    cost      = 1 − F_process(u_target, U_circuit)

    Empty gate list → U_circuit = identity.

    Args:
        gate_matrices: Ordered list of unitaries. gate_matrices[0] is applied
            first to the state (rightmost in the matrix product).
        u_target: Target unitary to compile.

    Returns:
        Cost in [0, 1]. Lower is better.
    """
    d = u_target.shape[0]
    u_circuit = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u_circuit = gate @ u_circuit
    return 1.0 - process_fidelity(u_target, u_circuit)


def build_cost_fn(u_target: np.ndarray) -> Callable[[List[np.ndarray]], float]:
    """Build a cost function closed over the target unitary.

    The returned callable has the signature expected by the GQE pipeline:
    it accepts a list of gate matrices and returns a float cost.

    Args:
        u_target: Target unitary (d × d). Captured by closure.

    Returns:
        Callable: (List[np.ndarray]) → float
    """
    def cost_fn(gate_matrices: List[np.ndarray]) -> float:
        return compilation_cost(gate_matrices, u_target)

    return cost_fn
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_cost.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add cost.py tests/test_cost.py
git commit -m "feat: add process fidelity cost function for unitary compilation"
```

---

## Task 4: Update `data.py` and Rewrite `pipeline.py`

**Why now:** With cost.py ready, we can rewire the pipeline to use it. The `data.py` rename must happen first because `pipeline.py` depends on the buffer key name.

**Key changes in `pipeline.py`:**
- Remove `import cudaq`, `from mpi4py import MPI`
- Use **absolute imports** (not `from .data import ...`)
- Use `GQEConfig` attribute paths: `cfg.training.seed`, `cfg.training.num_samples`, etc.
- Rewrite `computeCost()` to call `self._cost(gate_matrices)` directly
- Rename `energies` → `costs` throughout
- Remove `numQPUs` parameter and `benchmark_energy` attribute

**Files:**
- Modify: `data.py` (rename `energy` → `cost`)
- Modify: `pipeline.py` (full rewrite)
- Create: `tests/test_pipeline_integration.py`

- [ ] **Step 4.1: Write failing integration tests**

Create `tests/test_pipeline_integration.py`:

```python
import numpy as np
import pytest
import torch

from config import (
    BufferConfig,
    GQEConfig,
    LoggingConfig,
    ModelConfig,
    PoolConfig,
    TargetConfig,
    TemperatureConfig,
    TrainingConfig,
)
from cost import build_cost_fn
from factory import Factory
from model import GPT2
from operator_pool import build_operator_pool
from pipeline import Pipeline


def make_test_config() -> GQEConfig:
    """Minimal config for fast integration tests."""
    return GQEConfig(
        target=TargetConfig(num_qubits=1, type="random"),
        pool=PoolConfig(discretization_factor=4),
        model=ModelConfig(size="small", max_gates_count=5),
        training=TrainingConfig(
            max_epochs=2, num_samples=4, batch_size=4, lr=1e-4,
            grad_norm_clip=1.0, seed=42, grpo_clip_ratio=0.2,
        ),
        temperature=TemperatureConfig(
            scheduler="fixed", initial_value=0.5, delta=0.02,
            min_value=0.1, max_value=1.5,
        ),
        buffer=BufferConfig(max_size=10, warmup_size=4, steps_per_epoch=2),
        logging=LoggingConfig(verbose=False, wandb=False),
    )


def make_pipeline(cfg: GQEConfig, num_qubits: int = 1) -> Pipeline:
    pool = build_operator_pool(
        num_qubits=num_qubits,
        discretization_factor=cfg.pool.discretization_factor,
    )
    target = np.eye(2 ** num_qubits, dtype=np.complex128)
    cost_fn = build_cost_fn(target)
    model = GPT2(cfg.model.small, len(pool))
    factory = Factory()
    return Pipeline(cfg, cost_fn, pool, model, factory), pool


class TestPipelineInitialization:

    def test_pipeline_initializes_without_cudaq(self):
        """Pipeline must construct without any cudaq import."""
        cfg = make_test_config()
        pipeline, _ = make_pipeline(cfg)
        assert pipeline is not None

    def test_no_benchmark_energy_attribute(self):
        """Old benchmark_energy attribute must not exist."""
        cfg = make_test_config()
        pipeline, _ = make_pipeline(cfg)
        assert not hasattr(pipeline, "benchmark_energy")


class TestGenerate:

    def test_generate_shape(self):
        """generate() returns (num_samples, max_gates_count + 1) tensor."""
        cfg = make_test_config()
        pipeline, _ = make_pipeline(cfg)
        with torch.no_grad():
            sequences = pipeline.generate()
        assert sequences.shape == (
            cfg.training.num_samples,
            cfg.model.max_gates_count + 1,  # +1 for start token
        )

    def test_generate_indices_in_range(self):
        """All generated indices (excluding start token) must be valid pool indices."""
        cfg = make_test_config()
        pipeline, pool = make_pipeline(cfg)
        with torch.no_grad():
            sequences = pipeline.generate()
        gate_indices = sequences[:, 1:]
        assert (gate_indices >= 0).all()
        assert (gate_indices < len(pool)).all()


class TestComputeCost:

    def test_compute_cost_shape(self):
        """computeCost returns a 1-D tensor with one cost per sequence."""
        cfg = make_test_config()
        pipeline, pool = make_pipeline(cfg)
        with torch.no_grad():
            sequences = pipeline.generate()
        costs = pipeline.computeCost(sequences[:, 1:], pool)
        assert costs.shape == (cfg.training.num_samples,)

    def test_compute_cost_dtype(self):
        """computeCost returns float32 tensor."""
        cfg = make_test_config()
        pipeline, pool = make_pipeline(cfg)
        with torch.no_grad():
            sequences = pipeline.generate()
        costs = pipeline.computeCost(sequences[:, 1:], pool)
        assert costs.dtype == torch.float32

    def test_costs_in_unit_interval(self):
        """Process fidelity costs must be in [0, 1]."""
        cfg = make_test_config()
        pipeline, pool = make_pipeline(cfg)
        with torch.no_grad():
            sequences = pipeline.generate()
        costs = pipeline.computeCost(sequences[:, 1:], pool)
        assert (costs >= -0.01).all()
        assert (costs <= 1.01).all()
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_pipeline_integration.py -v 2>&1 | head -20
```

Expected: `ImportError` from cudaq or mpi4py inside the current `pipeline.py`.

- [ ] **Step 4.3: Rewrite `data.py` — rename `energy` → `cost`**

Replace the entire content of `data.py`:

```python
"""Replay buffer and dataset for GQE training.

Stores (sequence, cost) pairs in a FIFO queue. The BufferDataset wraps
the buffer to support repeating data for multiple gradient steps per epoch.

Import convention: absolute imports only (flat module structure).
"""

import pickle
import sys
from collections import deque

from torch.utils.data import Dataset


class ReplayBuffer:
    """FIFO replay buffer storing (sequence, cost) pairs.

    Args:
        size: Maximum number of entries in the buffer. Oldest entries
            are dropped when this limit is exceeded.
    """

    def __init__(self, size: int = sys.maxsize):
        self.size = size
        self.buf = deque(maxlen=size)

    def push(self, seq, cost) -> None:
        self.buf.append((seq, cost))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.buf, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.buf = pickle.load(f)

    def __getitem__(self, idx: int):
        seq, cost = self.buf[idx]
        return {"idx": seq, "cost": cost}

    def __len__(self) -> int:
        return len(self.buf)


class BufferDataset(Dataset):
    """Dataset that repeats the replay buffer `repetition` times.

    Args:
        buffer: The replay buffer to wrap.
        repetition: How many times to repeat the buffer data per epoch.
            This controls how many gradient steps happen per epoch.
    """

    def __init__(self, buffer: ReplayBuffer, repetition: int):
        self.buffer = buffer
        self.repetition = repetition

    def __getitem__(self, idx: int):
        item = self.buffer[idx % len(self.buffer)]
        return {"idx": item["idx"], "cost": item["cost"]}

    def __len__(self) -> int:
        return len(self.buffer) * self.repetition
```

- [ ] **Step 4.4: Rewrite `pipeline.py` — remove cudaq, use absolute imports**

Replace the entire content of `pipeline.py`:

```python
"""GQE Pipeline — Lightning module for transformer-guided circuit generation.

Autoregressively generates gate sequences using GPT-2, evaluates them via
a cost function (process fidelity for unitary compilation), and trains the
model to generate circuits with lower cost.

Import convention: absolute imports only (flat module structure).
"""

import torch
import lightning as L
from lightning import LightningModule
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from data import ReplayBuffer, BufferDataset


class Pipeline(LightningModule):
    """GPT-2-based transformer for quantum gate sequence generation.

    The pipeline implements the GQE training loop (paper Sec. 2.2.2):
      1. generate() — sample N_sample circuits autoregressively
      2. computeCost() — evaluate each circuit's cost using the cost function
      3. Replay buffer stores (sequence, cost) pairs (FIFO)
      4. training_step() — gradient update using GRPO loss on buffer data

    Args:
        cfg: GQEConfig — the full configuration object.
        cost: Callable (List[np.ndarray]) → float. Called once per circuit.
        pool: List of (name, matrix) tuples — the operator vocabulary.
        model: GPT2 model instance.
        factory: Factory for creating loss and scheduler.
    """

    def __init__(self, cfg, cost, pool, model, factory):
        super().__init__()

        L.seed_everything(cfg.training.seed)

        self.cfg = cfg
        self.model = model.to(self.device)
        self.factory = factory
        self.pool = pool
        self._cost = cost
        self.loss = self.factory.create_loss_fn(cfg).to(self.device)
        self.scheduler = self.factory.create_temperature_scheduler(cfg)
        self.ngates = cfg.model.max_gates_count
        self.num_samples = cfg.training.num_samples
        self.buffer = ReplayBuffer(size=cfg.buffer.max_size)
        self.save_hyperparameters(ignore=["cost", "pool", "model", "factory"])
        self._starting_idx = torch.zeros(
            self.num_samples, 1, dtype=torch.long, device=self.device
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.training.lr
        )

    def on_fit_start(self):
        self._starting_idx = torch.zeros(
            self.num_samples, 1, dtype=torch.long, device=self.device
        )
        while len(self.buffer) < self.cfg.buffer.warmup_size:
            self.collect_rollout()
        super().on_fit_start()

    def on_train_epoch_start(self):
        self.collect_rollout()

    def collect_rollout(self):
        """Generate circuits, evaluate costs, push to replay buffer, update scheduler."""
        idx_output = self.generate()
        costs = self.computeCost(idx_output[:, 1:], self.pool)
        for seq, cost_val in zip(idx_output, costs):
            self.buffer.push(seq, cost_val)
        self.scheduler.update(costs=costs)

    def set_cost(self, cost):
        """Replace the cost function (e.g. to set None at end of training)."""
        self._cost = cost

    @torch.no_grad()
    def computeCost(self, idx_output, pool, **kwargs):
        """Compute cost for each generated gate sequence.

        For each row of gate indices, looks up the corresponding matrices
        from the pool and passes them to the cost function. Gate index j
        maps to pool[j] = (name, matrix).

        Args:
            idx_output: LongTensor of shape (batch, seq_len) — gate indices.
                idx_output[i, k] is the (k+1)-th gate in circuit i.
            pool: List of (name, matrix) tuples.

        Returns:
            torch.Tensor of shape (batch,), dtype=float32, values in [0, 1].
        """
        results = []
        for row in idx_output:
            gate_matrices = [pool[j][1] for j in row.tolist()]
            cost_val = self._cost(gate_matrices)
            if not isinstance(cost_val, float):
                raise RuntimeError(
                    f"Cost function must return float, got {type(cost_val)}"
                )
            results.append(cost_val)
        return torch.tensor(results, dtype=torch.float32)

    def training_step(self, batch, batch_idx):
        idx = batch["idx"].to(self.device)
        costs = batch["cost"].to(self.device)

        log_values = {}
        logits = self.model(idx).logits
        loss = self.loss.compute(
            costs,
            logits,
            idx[:, 1:],
            log_values,
            inverse_temperature=self.scheduler.get_inverse_temperature(),
            current_step=batch_idx,
        )

        self.log_dict(log_values, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("cost_mean", costs.mean(), prog_bar=False, on_epoch=True, on_step=False)
        self.log("cost_min", costs.min(), prog_bar=False, on_epoch=True, on_step=False)
        self.log(
            "inverse_temperature",
            self.scheduler.get_inverse_temperature(),
            prog_bar=True, on_epoch=True, on_step=False,
        )
        return loss

    def train_dataloader(self):
        return DataLoader(
            BufferDataset(self.buffer, self.cfg.buffer.steps_per_epoch),
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )

    @torch.no_grad()
    def generate(self, idx=None, ngates=None):
        """Autoregressively generate gate sequences.

        Starts from the zero token (start-of-sequence) and samples
        one gate index at a time using Categorical(logits=-β·w).

        Args:
            idx: Optional seed tensor of shape (batch, t). If None,
                uses the internal start token (all zeros).
            ngates: Number of gates to generate. Defaults to max_gates_count.

        Returns:
            LongTensor of shape (num_samples, ngates + 1) including the
            start token as the first column.
        """
        if idx is None:
            idx = self._starting_idx.clone()
        if ngates is None:
            ngates = self.ngates
        beta = self.scheduler.get_inverse_temperature()
        for _ in range(ngates):
            logits = self.model(idx).logits[:, -1, :]
            probs = Categorical(logits=-beta * logits)
            idx_next = probs.sample()
            idx = torch.cat((idx, idx_next.unsqueeze(1)), dim=1)
        return idx

    def logits(self, idx):
        """Return per-token log-probabilities for a given sequence."""
        logits_base = self.model(idx)
        idx = idx[:, 1:]
        return torch.gather(logits_base.logits, 2, idx.unsqueeze(-1)).squeeze(-1)
```

- [ ] **Step 4.5: Run integration tests**

```bash
python -m pytest tests/test_pipeline_integration.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 4.6: Commit**

```bash
git add pipeline.py data.py tests/test_pipeline_integration.py
git commit -m "refactor: remove cudaq/mpi4py from pipeline, use process fidelity cost"
```

---

## Task 5: Update `callbacks.py`

**What changes:**
- Rename `MinEnergyCallback` → `BestCostCallback`
- Remove `benchmark_energy` reference (that field no longer exists on Pipeline)
- Change logging keys `"best energy"` → `"best_cost"` and add `"best_fidelity"`
- Update `TrajectoryCallback` to use `"cost"` key (matches new `data.py`)

**Files:**
- Modify: `callbacks.py` (full rewrite)

- [ ] **Step 5.1: Rewrite `callbacks.py`**

Replace the entire content:

```python
"""Lightning callbacks for GQE training.

Import convention: absolute imports only (flat module structure).
"""

import sys
import torch
from lightning.pytorch.callbacks import Callback


class BestCostCallback(Callback):
    """Track the best (lowest) cost found across all training epochs.

    For unitary compilation, cost = 1 − process_fidelity, so lower cost
    means higher fidelity. After training, call get_results() to retrieve
    the best cost and corresponding gate sequence.
    """

    def __init__(self):
        super().__init__()
        self.best_cost = float("inf")
        self.best_indices = None
        self.best_cost_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        if len(pl_module.buffer) == 0:
            return
        # Scan the most recently added entries (last num_samples slots)
        start = max(0, len(pl_module.buffer) - pl_module.num_samples)
        for i in range(start, len(pl_module.buffer)):
            seq, cost_val = pl_module.buffer.buf[i]
            if isinstance(cost_val, torch.Tensor):
                cost_val = cost_val.item()
            if cost_val < self.best_cost:
                self.best_cost = cost_val
                self.best_indices = seq

        self.best_cost_history.append(self.best_cost)
        pl_module.log(
            "best_cost", self.best_cost, prog_bar=True, on_epoch=True, on_step=False
        )
        pl_module.log(
            "best_fidelity", 1.0 - self.best_cost,
            prog_bar=True, on_epoch=True, on_step=False,
        )

    def get_results(self):
        """Return (best_cost, best_indices) after training completes."""
        return self.best_cost, self.best_indices


class TrajectoryCallback(Callback):
    """Save training trajectory to a JSONL file for post-hoc analysis.

    Each line is a JSON object with epoch, batch_idx, loss, indices, costs.
    """

    def __init__(self, trajectory_file_path: str):
        super().__init__()
        self.trajectory_file_path = trajectory_file_path
        self.trajectory_data = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None or "loss" not in outputs:
            return
        loss = outputs["loss"]
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        indices = batch.get("idx")
        costs = batch.get("cost")
        if indices is None or costs is None:
            return

        self.trajectory_data.append({
            "epoch": trainer.current_epoch,
            "batch_idx": batch_idx,
            "loss": loss,
            "indices": indices.cpu().numpy().tolist() if isinstance(indices, torch.Tensor) else indices,
            "costs": costs.cpu().numpy().tolist() if isinstance(costs, torch.Tensor) else costs,
        })

    def on_train_end(self, trainer, pl_module):
        import json
        import os

        os.makedirs(os.path.dirname(self.trajectory_file_path), exist_ok=True)
        with open(self.trajectory_file_path, "w") as f:
            for entry in self.trajectory_data:
                f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 5.2: Commit**

```bash
git add callbacks.py
git commit -m "refactor: rename MinEnergyCallback to BestCostCallback, energy->cost"
```

---

## Task 6: Update `loss.py` — rename `energies` → `costs`

**What changes:** Pure rename of the `energies` parameter to `costs` in all `compute()` signatures and `calc_advantage()`. The mathematical formulas are **identical** — the advantage `(mean − value) / std` is already correct for minimization (sequences with below-average cost get positive advantage).

**Files:**
- Modify: `loss.py`

- [ ] **Step 6.1: Apply the rename in `loss.py`**

Make the following targeted edits (the rest of the file is unchanged):

In the abstract `Loss` class, line ~23:
```python
# Before:
def compute(self, energies, gate_logits, gate_indices, log_values, **kwargs):
# After:
def compute(self, costs, gate_logits, gate_indices, log_values, **kwargs):
```

In `ExpLogitMatching.compute()`:
```python
# Before:
def compute(self, energies, gate_logits, gate_indices, log_values, **kwargs):
    ...
    return self.loss_fn(
        torch.exp(-mean_logits),
        torch.exp(-energies.to(device) - self.energy_offset))
# After:
def compute(self, costs, gate_logits, gate_indices, log_values, **kwargs):
    ...
    return self.loss_fn(
        torch.exp(-mean_logits),
        torch.exp(-costs.to(device) - self.energy_offset))
```

In `GFlowLogitMatching.compute()`:
```python
# Before:
def compute(self, energies, gate_logits, gate_indices, log_values, **kwargs):
    ...
    loss = self.loss_fn(
        torch.exp(-mean_logits),
        torch.exp(-(energies.to(device) + energy_offset.to(device))))
# After:
def compute(self, costs, gate_logits, gate_indices, log_values, **kwargs):
    ...
    loss = self.loss_fn(
        torch.exp(-mean_logits),
        torch.exp(-(costs.to(device) + energy_offset.to(device))))
```

In `GRPOLoss.compute()`:
```python
# Before:
def compute(self, energies, gate_logits, gate_indices, log_values=None, **kwargs):
    ...
    win_id = torch.argmin(energies)
    ...
    if torch.std(energies) == 0:
    ...
        self.advantages = self.calc_advantage(energies)
# After:
def compute(self, costs, gate_logits, gate_indices, log_values=None, **kwargs):
    ...
    win_id = torch.argmin(costs)
    ...
    if torch.std(costs) == 0:
    ...
        self.advantages = self.calc_advantage(costs)
```

In `GRPOLoss.calc_advantage()`:
```python
# Before:
def calc_advantage(self, energies):
    return (energies.mean() - energies) / (energies.std() + 1e-8)
# After:
def calc_advantage(self, costs):
    return (costs.mean() - costs) / (costs.std() + 1e-8)
```

- [ ] **Step 6.2: Verify the import chain still works**

```bash
python -c "from loss import GRPOLoss; print('OK')"
```

Expected: `OK`

- [ ] **Step 6.3: Commit**

```bash
git add loss.py
git commit -m "refactor: rename energies parameter to costs in all loss functions"
```

---

## Task 7: Update `scheduler.py` and `factory.py`

**What changes in `scheduler.py`:**
- `DefaultScheduler`: add `minimum` and `maximum` clamping parameters ("fixed" means bounded linear ramp, not frozen)
- `VarBasedScheduler`: accept `costs` kwarg (or legacy `energies`) since `pipeline.py` calls `scheduler.update(costs=costs)`
- `CosineScheduler`: no changes needed — already correct

**What changes in `factory.py`:**
- Use `GQEConfig` nested paths (`cfg.training.grpo_clip_ratio`, `cfg.temperature.*`)
- Use **absolute imports** (not `from .loss import ...`)
- Replace the single `if scheduler_type == "fixed"` branch with a **`SCHEDULER_BUILDERS` dict registry** that maps all 3 scheduler names to constructor lambdas. Adding a new scheduler = 1 new class in `scheduler.py` + 1 new lambda in `SCHEDULER_BUILDERS`. No other files need to change.

**Files:**
- Modify: `scheduler.py`
- Modify: `factory.py`

- [ ] **Step 7.1: Rewrite `scheduler.py`**

Replace the entire content:

```python
"""Temperature schedulers for GQE training.

The inverse temperature β controls the sharpness of the sampling
distribution: Categorical(logits=-β·w). Higher β → sharper (more
greedy) sampling.

Import convention: absolute imports only (flat module structure).
"""

import math
from abc import ABC, abstractmethod


class TemperatureScheduler(ABC):

    @abstractmethod
    def get_inverse_temperature(self) -> float:
        """Return current inverse temperature β."""

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update the scheduler state after each rollout."""


class DefaultScheduler(TemperatureScheduler):
    """Linear scheduler that increments β by `delta` each update.

    Optional clamping keeps β within [minimum, maximum]. When both
    bounds are set and delta > 0, β gradually increases (warms up)
    until it hits maximum.

    Args:
        start: Initial inverse temperature.
        delta: Amount added to β each update call.
        minimum: Lower clamp bound (optional).
        maximum: Upper clamp bound (optional).
    """

    def __init__(self, start: float, delta: float, minimum=None, maximum=None):
        self.start = start
        self.delta = delta
        self.minimum = minimum
        self.maximum = maximum
        self.current_temperature = start

    def get_inverse_temperature(self) -> float:
        return self.current_temperature

    def update(self, **kwargs) -> None:
        self.current_temperature += self.delta
        if self.minimum is not None:
            self.current_temperature = max(self.minimum, self.current_temperature)
        if self.maximum is not None:
            self.current_temperature = min(self.maximum, self.current_temperature)


class CosineScheduler(TemperatureScheduler):
    """Cosine-oscillating scheduler between minimum and maximum.

    Args:
        minimum: Minimum β.
        maximum: Maximum β.
        frequency: Number of updates per full oscillation cycle.
    """

    def __init__(self, minimum: float, maximum: float, frequency: int):
        self.minimum = minimum
        self.maximum = maximum
        self.frequency = frequency
        self.current_iter = 0
        self.current_temperature = (maximum + minimum) / 2

    def get_inverse_temperature(self) -> float:
        return self.current_temperature

    def update(self, **kwargs) -> None:
        self.current_iter += 1
        self.current_temperature = (
            (self.maximum + self.minimum) / 2
            - (self.maximum - self.minimum) / 2
            * math.cos(2 * math.pi * self.current_iter / self.frequency)
        )


class VarBasedScheduler(TemperatureScheduler):
    """Variance-based adaptive scheduler.

    Increases β when cost variance exceeds target (diverse costs →
    explore more); decreases β when variance is low (converged → exploit).

    Args:
        initial: Initial inverse temperature.
        delta: Step size for each adjustment.
        target_var: Variance threshold that controls direction of adjustment.
    """

    def __init__(self, initial: float, delta: float, target_var: float):
        self.delta = delta
        self.current_temperature = initial
        self.target_var = target_var

    def get_inverse_temperature(self) -> float:
        return self.current_temperature

    def update(self, **kwargs) -> None:
        # Accept either 'costs' (new name) or 'energies' (legacy name)
        costs = kwargs.get("costs", kwargs.get("energies"))
        if costs is None:
            return
        current_var = costs.var().item()
        if current_var > self.target_var:
            self.current_temperature += self.delta
        else:
            self.current_temperature -= self.delta
        self.current_temperature = max(self.current_temperature, 0.01)
```

- [ ] **Step 7.2: Rewrite `factory.py`**

Replace the entire content:

```python
"""Factory for creating loss functions and temperature schedulers.

Reads the GQEConfig nested structure (cfg.training.*, cfg.temperature.*)
to instantiate the correct components.

## How to add a new scheduler (plug-and-play)

1. Implement a `TemperatureScheduler` subclass in `scheduler.py`.
2. Import it here.
3. Add one entry to `SCHEDULER_BUILDERS` in `create_temperature_scheduler()`.

That's it — no other files need to change. The YAML key
`temperature.scheduler` selects which builder runs.

Import convention: absolute imports only (flat module structure).
"""

from loss import GRPOLoss, ExpLogitMatching, GFlowLogitMatching
from scheduler import DefaultScheduler, CosineScheduler, VarBasedScheduler


class Factory:

    def create_loss_fn(self, cfg):
        """Return the GRPO loss (the only loss supported by the current config schema).

        GRPOLoss is the recommended loss from the paper (Appendix B).
        The clip_ratio comes from cfg.training.grpo_clip_ratio.
        """
        return GRPOLoss(clip_ratio=cfg.training.grpo_clip_ratio)

    def create_temperature_scheduler(self, cfg):
        """Return a temperature scheduler based on cfg.temperature.scheduler.

        Uses a registry dict so adding a new scheduler requires only:
          1. A new class in scheduler.py
          2. One new lambda entry in SCHEDULER_BUILDERS below

        Supported values for temperature.scheduler in config.yml:
          "fixed"    — linear ramp clamped within [min_value, max_value]
          "cosine"   — cosine oscillation between min_value and max_value
          "variance" — adaptive: increases β when cost variance is high
        """
        scheduler_type = cfg.temperature.scheduler

        # ── Scheduler registry ──────────────────────────────────────────────
        # To add a new scheduler:
        #   1. Implement TemperatureScheduler subclass in scheduler.py
        #   2. Import it above
        #   3. Add one entry here — no other files need changing
        SCHEDULER_BUILDERS = {
            "fixed": lambda: DefaultScheduler(
                start=cfg.temperature.initial_value,
                delta=cfg.temperature.delta,
                minimum=cfg.temperature.min_value,
                maximum=cfg.temperature.max_value,
            ),
            "cosine": lambda: CosineScheduler(
                minimum=cfg.temperature.min_value,
                maximum=cfg.temperature.max_value,
                frequency=max(1, cfg.training.max_epochs // 2),
            ),
            "variance": lambda: VarBasedScheduler(
                initial=cfg.temperature.initial_value,
                delta=cfg.temperature.delta,
                target_var=1e-5,
            ),
        }

        if scheduler_type not in SCHEDULER_BUILDERS:
            available = sorted(SCHEDULER_BUILDERS.keys())
            raise ValueError(
                f"Unknown scheduler: {scheduler_type!r}. "
                f"Available: {available}. "
                f"To add a new scheduler, implement TemperatureScheduler in "
                f"scheduler.py and add one lambda to SCHEDULER_BUILDERS in factory.py."
            )
        return SCHEDULER_BUILDERS[scheduler_type]()
```

- [ ] **Step 7.3: Verify the factory resolves all 3 scheduler types**

```bash
python -c "
import yaml, io
from config import load_config, GQEConfig, TargetConfig, PoolConfig, ModelConfig, TrainingConfig, TemperatureConfig, BufferConfig, LoggingConfig
from factory import Factory

# Helper to build a config with a given scheduler name
def make_cfg(scheduler_name):
    return GQEConfig(
        target=TargetConfig(num_qubits=2, type='random'),
        pool=PoolConfig(discretization_factor=32),
        model=ModelConfig(size='small', max_gates_count=10),
        training=TrainingConfig(max_epochs=50, num_samples=30, batch_size=16, lr=1e-4,
                                grad_norm_clip=1.0, seed=42, grpo_clip_ratio=0.2),
        temperature=TemperatureConfig(scheduler=scheduler_name, initial_value=0.5,
                                      delta=0.02, min_value=0.1, max_value=1.5),
        buffer=BufferConfig(max_size=100, warmup_size=30, steps_per_epoch=5),
        logging=LoggingConfig(verbose=False, wandb=False),
    )

f = Factory()
for name in ['fixed', 'cosine', 'variance']:
    cfg = make_cfg(name)
    sched = f.create_temperature_scheduler(cfg)
    print(f'{name:10s} -> {type(sched).__name__}, beta={sched.get_inverse_temperature()}')
"
```

Expected output:
```
fixed      -> DefaultScheduler,   beta=0.5
cosine     -> CosineScheduler,    beta=0.8
variance   -> VarBasedScheduler,  beta=0.5
```

- [ ] **Step 7.4: Commit**

```bash
git add scheduler.py factory.py
git commit -m "refactor: wire factory and scheduler to GQEConfig structure"
```

---

## Task 8: Rewrite `gqe.py`

**What changes:**
- Remove `import cudaq`, `from ml_collections import ConfigDict`
- Remove `TrajectoryData`, `FileMonitor`, `get_default_config()`, old `validate_config()`
- Use `GQEConfig` from `config.py`
- Use **absolute imports** (not `from .pipeline import ...`)
- Replace `MinEnergyCallback` with `BestCostCallback`
- Control W&B via `cfg.logging.wandb`; control verbosity via `cfg.logging.verbose`
- Keep the public `gqe()` signature compatible with `main.py`

**Files:**
- Modify: `gqe.py` (full rewrite)

- [ ] **Step 8.1: Rewrite `gqe.py`**

Replace the entire content:

```python
"""GQE — Generative Quantum Eigensolver for unitary compilation.

Public API: gqe(cost_fn, pool, cfg) → (best_cost, best_indices)

Import convention: absolute imports only (flat module structure).
"""

import os

import torch
import lightning as L

from callbacks import BestCostCallback
from config import GQEConfig
from factory import Factory
from model import GPT2
from pipeline import Pipeline

torch.set_float32_matmul_precision("high")


def _build_logger(cfg: GQEConfig):
    """Create a W&B logger when cfg.logging.wandb is True, else return False."""
    if not cfg.logging.wandb:
        return False

    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(
        project=os.getenv("WANDB_PROJECT", "gqe-torch"),
        name=os.getenv("WANDB_NAME", None),
        config={
            "num_qubits": cfg.target.num_qubits,
            "target_type": cfg.target.type,
            "discretization_factor": cfg.pool.discretization_factor,
            "model_size": cfg.model.size,
            "max_gates_count": cfg.model.max_gates_count,
            "max_epochs": cfg.training.max_epochs,
        },
    )


def _run_training(cfg: GQEConfig, pipeline: Pipeline):
    """Run the GQE training loop using Lightning Trainer.

    Args:
        cfg: Configuration object.
        pipeline: Constructed Pipeline (Lightning module).

    Returns:
        tuple: (best_cost: float, best_indices: list[int] | None)
    """
    best_cost_cb = BestCostCallback()

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.grad_norm_clip,
        enable_progress_bar=cfg.logging.verbose,
        enable_model_summary=cfg.logging.verbose,
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        logger=_build_logger(cfg),
        callbacks=[best_cost_cb],
    )

    trainer.fit(pipeline)

    best_cost, best_indices = best_cost_cb.get_results()
    if best_indices is not None and isinstance(best_indices, torch.Tensor):
        best_indices = best_indices.cpu().numpy().tolist()

    pipeline.set_cost(None)
    return best_cost, best_indices


def gqe(cost_fn, pool, cfg: GQEConfig, model=None):
    """Run the Generative Quantum Eigensolver for unitary compilation.

    Args:
        cost_fn: Callable (List[np.ndarray]) → float. Returns cost in [0, 1].
            Use cost.build_cost_fn(u_target) to construct this.
        pool: List of (name, matrix) tuples — the operator pool / vocabulary.
            Use operator_pool.build_operator_pool(...) to construct this.
        cfg: GQEConfig loaded via config.load_config('config.yml').
        model: Optional pre-built GPT2 model. If None, one is created from cfg.

    Returns:
        tuple: (best_cost: float, best_indices: list[int] | None)
            best_cost: Lowest cost (1 − fidelity) found during training.
            best_indices: Gate index sequence for that circuit, including
                the start token at position 0.
    """
    factory = Factory()
    if model is None:
        model = GPT2(cfg.model.small, len(pool))
    pipeline = Pipeline(cfg, cost_fn, pool, model, factory)
    return _run_training(cfg, pipeline)
```

- [ ] **Step 8.2: Verify gqe.py imports work**

```bash
python -c "from gqe import gqe; print('gqe import OK')"
```

Expected: `gqe import OK`

- [ ] **Step 8.3: Commit**

```bash
git add gqe.py
git commit -m "refactor: remove cudaq from gqe.py, use GQEConfig and absolute imports"
```

---

## Task 9: Clean Up `utils.py`, `pool.py`, `requirements.txt`

**Files:**
- Modify: `utils.py` (gut all cudaq code)
- Delete: `pool.py` (replace with redirect comment)
- Modify: `requirements.txt` (remove cudaq/mpi4py/ml-collections, add pyyaml)

- [ ] **Step 9.1: Replace `utils.py` content**

Replace the entire content of `utils.py`:

```python
"""Utilities module.

The operator pool functionality previously in this file has moved to
operator_pool.py.
"""
```

- [ ] **Step 9.2: Replace `pool.py` content**

Replace the entire content of `pool.py`:

```python
"""pool.py — superseded.

Operator pool construction has moved to operator_pool.py.
"""
```

- [ ] **Step 9.3: Update `requirements.txt`**

Replace the entire content of `requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cu130
torch>=2.9.0
lightning
qiskit
qiskit[visualization]
matplotlib
seaborn
transformers
wandb
python-dotenv
pyyaml
pytest
```

Notes:
- Removed: `cudaq`, `cudaq-solvers`, `mpi4py`, `ml-collections`
- Added: `pyyaml` (config loading), `pytest` (test runner)

- [ ] **Step 9.4: Verify no remaining cudaq references**

```bash
grep -r "cudaq\|mpi4py\|ml_collections" \
    /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch/*.py \
    --include="*.py" | grep -v ".pyc"
```

Expected: No output (zero matches).

- [ ] **Step 9.5: Commit**

```bash
git add utils.py pool.py requirements.txt
git commit -m "chore: remove cudaq/mpi4py/ml-collections deps, clean up utils and pool"
```

---

## Task 10: Target Generator Registry (`target.py`)

**Why here:** The target unitary strategy is a first-class extension point. Isolating it in `target.py` with a registry means `main.py` never needs to change when a new target type is added — only `target.py` grows.

**Design:** A `TargetGenerator` is any callable matching `(pool, cfg) → (np.ndarray, str)`. The return values are the target unitary and a human-readable description. The `TARGET_REGISTRY` dict maps `cfg.target.type` strings to generator functions. `build_target(pool, cfg)` is the sole public API used by `main.py`.

**Built-in generators:**
- `"random"` / `"random_reachable"`: sample a short random circuit from the pool, compose its unitary. **Guaranteed reachable** — a perfect solution exists in the vocabulary.
- `"haar_random"`: sample a Haar-uniform random unitary using QR decomposition of a complex Gaussian matrix. **Not guaranteed reachable** — useful for harder benchmarks where the pool may not express the target exactly.

**Files:**
- Create: `target.py`
- Create: `tests/test_target.py`

- [ ] **Step 10.1: Write failing tests for target generators**

Create `tests/test_target.py`:

```python
import numpy as np
import pytest

from config import (
    BufferConfig, GQEConfig, LoggingConfig, ModelConfig,
    PoolConfig, TargetConfig, TemperatureConfig, TrainingConfig,
)
from operator_pool import build_operator_pool
from target import TARGET_REGISTRY, build_target


def make_cfg(target_type: str, num_qubits: int = 2) -> GQEConfig:
    return GQEConfig(
        target=TargetConfig(num_qubits=num_qubits, type=target_type),
        pool=PoolConfig(discretization_factor=4),
        model=ModelConfig(size="small", max_gates_count=5),
        training=TrainingConfig(
            max_epochs=10, num_samples=4, batch_size=4, lr=1e-4,
            grad_norm_clip=1.0, seed=42, grpo_clip_ratio=0.2,
        ),
        temperature=TemperatureConfig(
            scheduler="fixed", initial_value=0.5, delta=0.02,
            min_value=0.1, max_value=1.5,
        ),
        buffer=BufferConfig(max_size=10, warmup_size=4, steps_per_epoch=2),
        logging=LoggingConfig(verbose=False, wandb=False),
    )


class TestTargetRegistry:

    def test_registry_contains_all_builtin_types(self):
        """Registry must have entries for all documented target types."""
        for name in ["random", "random_reachable", "haar_random"]:
            assert name in TARGET_REGISTRY, f"{name!r} missing from TARGET_REGISTRY"

    def test_unknown_type_raises_value_error(self):
        """build_target raises ValueError for unknown target type."""
        cfg = make_cfg("nonexistent_type")
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        with pytest.raises(ValueError, match="Unknown target type"):
            build_target(pool, cfg)


class TestRandomReachableGenerator:

    def test_returns_unitary(self):
        """random_reachable target must be a unitary matrix."""
        cfg = make_cfg("random_reachable")
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        u, _ = build_target(pool, cfg)
        d = 2 ** cfg.target.num_qubits
        assert u.shape == (d, d)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(d), atol=1e-10)

    def test_reproducible_with_same_seed(self):
        """Same seed must produce the same target."""
        cfg = make_cfg("random_reachable")
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        u1, _ = build_target(pool, cfg)
        u2, _ = build_target(pool, cfg)
        np.testing.assert_array_equal(u1, u2)

    def test_random_alias_matches_random_reachable(self):
        """'random' must produce identical output to 'random_reachable'."""
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        u_r, _ = build_target(pool, make_cfg("random"))
        u_rr, _ = build_target(pool, make_cfg("random_reachable"))
        np.testing.assert_array_equal(u_r, u_rr)

    def test_description_is_string(self):
        cfg = make_cfg("random_reachable")
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        _, desc = build_target(pool, cfg)
        assert isinstance(desc, str) and len(desc) > 0


class TestHaarRandomGenerator:

    def test_returns_unitary(self):
        """Haar-random target must be a unitary matrix."""
        cfg = make_cfg("haar_random")
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        u, _ = build_target(pool, cfg)
        d = 2 ** cfg.target.num_qubits
        assert u.shape == (d, d)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(d), atol=1e-10)

    def test_reproducible_with_same_seed(self):
        """Same seed must produce the same Haar-random unitary."""
        cfg = make_cfg("haar_random")
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        u1, _ = build_target(pool, cfg)
        u2, _ = build_target(pool, cfg)
        np.testing.assert_array_equal(u1, u2)

    def test_different_seeds_produce_different_unitaries(self):
        """Different seeds must (almost certainly) produce different targets."""
        pool = build_operator_pool(num_qubits=2, discretization_factor=4)
        u1, _ = build_target(pool, make_cfg("haar_random"))  # seed=42

        # Manually build a config with a different seed
        from config import TrainingConfig
        cfg2 = make_cfg("haar_random")
        cfg2 = GQEConfig(
            target=cfg2.target, pool=cfg2.pool, model=cfg2.model,
            training=TrainingConfig(
                max_epochs=cfg2.training.max_epochs,
                num_samples=cfg2.training.num_samples,
                batch_size=cfg2.training.batch_size,
                lr=cfg2.training.lr,
                grad_norm_clip=cfg2.training.grad_norm_clip,
                seed=99,  # different seed
                grpo_clip_ratio=cfg2.training.grpo_clip_ratio,
            ),
            temperature=cfg2.temperature, buffer=cfg2.buffer, logging=cfg2.logging,
        )
        u2, _ = build_target(pool, cfg2)
        assert not np.allclose(u1, u2), "Different seeds produced identical targets"

    def test_1qubit_haar(self):
        """Haar generator works for 1-qubit systems (2×2 unitary)."""
        cfg = make_cfg("haar_random", num_qubits=1)
        pool = build_operator_pool(num_qubits=1, discretization_factor=4)
        u, _ = build_target(pool, cfg)
        assert u.shape == (2, 2)
        np.testing.assert_allclose(u @ u.conj().T, np.eye(2), atol=1e-10)
```

- [ ] **Step 10.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_target.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'target'`

- [ ] **Step 10.3: Implement `target.py`**

Create `target.py`:

```python
"""Target unitary generators for GQE quantum compilation.

## Extension point — how to add a new target type

A TargetGenerator is any callable with this signature:

    def my_generator(
        pool: List[Tuple[str, np.ndarray]],
        cfg: GQEConfig,
    ) -> Tuple[np.ndarray, str]:
        ...
        return u_target, "human-readable description"

To register it so cfg.target.type = "my_type" selects it:

    TARGET_REGISTRY["my_type"] = my_generator

That's the entire extension. No other files need to change.

## Built-in generators

| cfg.target.type    | Strategy                                  | Fidelity 1.0 possible? |
|--------------------|-------------------------------------------|------------------------|
| "random"           | alias for "random_reachable"              | Yes                    |
| "random_reachable" | random circuit from the pool              | Yes (by construction)  |
| "haar_random"      | Haar-uniform random unitary               | Unlikely for small N   |

Import convention: absolute imports only (flat module structure).
"""

from typing import Callable, Dict, List, Tuple

import numpy as np

from config import GQEConfig


# Type alias for generator callables
TargetGenerator = Callable[
    [List[Tuple[str, np.ndarray]], GQEConfig],
    Tuple[np.ndarray, str],
]


def _compose_unitary(gate_matrices: List[np.ndarray], d: int) -> np.ndarray:
    """Compose an ordered gate list into a single unitary.

    U = gate_matrices[-1] @ ... @ gate_matrices[0]
    (gate_matrices[0] is applied first to the quantum state.)
    """
    u = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u = gate @ u
    return u


def random_reachable_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Generate a random reachable target from the operator pool.

    Samples `depth` random gates from the pool using cfg.training.seed and
    composes them into U_target. Because the target is built from pool gates,
    it is guaranteed to be expressible by the GQE vocabulary — i.e., a
    perfect fidelity solution exists in principle.

    Depth is capped at cfg.model.max_gates_count to ensure the target is
    not deeper than what the model can generate.
    """
    depth = min(4, cfg.model.max_gates_count)
    rng = np.random.default_rng(cfg.training.seed)
    indices = rng.integers(0, len(pool), size=depth)
    recipe = [pool[i][0] for i in indices]
    matrices = [pool[i][1] for i in indices]
    d = 2 ** cfg.target.num_qubits
    u_target = _compose_unitary(matrices, d)
    desc = f"random_reachable depth={depth}, recipe=[{', '.join(recipe)}]"
    return u_target, desc


def haar_random_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Generate a Haar-uniform random unitary.

    Uses QR decomposition of a random complex Gaussian matrix to produce
    a unitary drawn uniformly from the Haar measure on U(d). The target
    may not be exactly expressible by the operator pool, so perfect fidelity
    is not guaranteed. Useful for harder benchmarks or stress-testing the
    model's generalization.

    The `pool` argument is accepted but unused (kept for interface uniformity).
    """
    d = 2 ** cfg.target.num_qubits
    rng = np.random.default_rng(cfg.training.seed)
    # Random complex Gaussian matrix
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    # QR decomposition gives a Haar-uniform unitary
    q, r = np.linalg.qr(z)
    # Correct phases to ensure uniform Haar measure
    phase = np.diag(r) / np.abs(np.diag(r))
    u_target = q * phase[np.newaxis, :]
    desc = f"haar_random d={d}, seed={cfg.training.seed}"
    return u_target, desc


# ── Registry ──────────────────────────────────────────────────────────────────
# Maps cfg.target.type strings to generator callables.
#
# To add a new target type:
#   1. Write a function matching the TargetGenerator signature above.
#   2. Add one entry here.
#   3. Add the new name to VALID_TARGET_TYPES in config.py.
#   No other files need to change.
TARGET_REGISTRY: Dict[str, TargetGenerator] = {
    "random": random_reachable_generator,         # backward-compatible alias
    "random_reachable": random_reachable_generator,
    "haar_random": haar_random_generator,
}


def build_target(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Look up and run the appropriate target generator for cfg.target.type.

    This is the single public API used by main.py.

    Args:
        pool: The operator pool (list of (name, matrix) tuples).
        cfg: Full GQEConfig. cfg.target.type selects the generator.

    Returns:
        tuple: (u_target: np.ndarray, description: str)

    Raises:
        ValueError: If cfg.target.type is not in TARGET_REGISTRY.
    """
    target_type = cfg.target.type
    if target_type not in TARGET_REGISTRY:
        available = sorted(TARGET_REGISTRY.keys())
        raise ValueError(
            f"Unknown target type: {target_type!r}. "
            f"Available: {available}. "
            f"To add a new type, implement a TargetGenerator function and "
            f"register it in target.TARGET_REGISTRY."
        )
    return TARGET_REGISTRY[target_type](pool, cfg)
```

- [ ] **Step 10.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_target.py -v
```

Expected: All 10 tests PASS.

- [ ] **Step 10.5: Commit**

```bash
git add target.py tests/test_target.py
git commit -m "feat: add target unitary generator registry with random_reachable and haar_random"
```

---

## Task 11: Write `main.py` — Working Demo

**Why last:** All pieces are now in place. This task wires them together into a runnable demonstration. `main.py` is intentionally thin — it delegates all strategy decisions to `build_target()` and `gqe()`, so adding new target types or schedulers never requires editing `main.py`.

**Files:**
- Create: `main.py`

- [ ] **Step 11.1: Create `main.py`**

`main.py` is intentionally thin. It has **no strategy logic** — it just orchestrates. Changing `target.type` in `config.yml` selects a different target generator automatically; changing `temperature.scheduler` selects a different scheduler automatically. `main.py` never needs to change for new strategies.

```python
"""GQE Quantum Compilation — Demo Script.

Trains a GPT-2 model to find a gate sequence that approximates a target
unitary using process fidelity as the cost. Reads configuration from
config.yml (or a path passed via --config).

The target unitary strategy is selected by cfg.target.type in config.yml:
  "random" / "random_reachable"  — random circuit from the pool (default)
  "haar_random"                   — Haar-uniform random unitary

The temperature scheduler is selected by cfg.temperature.scheduler:
  "fixed"    — bounded linear ramp (default)
  "cosine"   — cosine oscillation
  "variance" — adaptive based on cost variance

To add new strategies, see target.py and factory.py — main.py never changes.

Usage:
    python main.py
    python main.py --config path/to/config.yml

Import convention: absolute imports only (flat module structure).
"""

import argparse

import numpy as np

from config import load_config
from cost import build_cost_fn, process_fidelity
from gqe import gqe
from operator_pool import build_operator_pool
from target import build_target


def _compose_unitary(gate_matrices, d: int) -> np.ndarray:
    """Compose an ordered gate list into a single unitary (for result decoding)."""
    u = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u = gate @ u
    return u


def main():
    parser = argparse.ArgumentParser(description="GQE Quantum Compilation Demo")
    parser.add_argument("--config", default="config.yml", help="Path to config.yml")
    args = parser.parse_args()

    # ── Load configuration ──────────────────────────────────────────────────
    cfg = load_config(args.config)
    print(f"\nLoaded config: {args.config}")
    print(f"  Target type:          {cfg.target.type}")
    print(f"  Qubits:               {cfg.target.num_qubits}")
    print(f"  Discretization bins:  {cfg.pool.discretization_factor}")
    print(f"  Scheduler:            {cfg.temperature.scheduler}")
    print(f"  Model size:           {cfg.model.size}")
    print(f"  Max gates/circuit:    {cfg.model.max_gates_count}")
    print(f"  Training epochs:      {cfg.training.max_epochs}")
    print(f"  W&B logging:          {cfg.logging.wandb}")

    # ── Build operator pool ─────────────────────────────────────────────────
    pool = build_operator_pool(
        num_qubits=cfg.target.num_qubits,
        discretization_factor=cfg.pool.discretization_factor,
    )
    print(f"  Pool size (vocab):    {len(pool)}")

    # ── Build target unitary ────────────────────────────────────────────────
    # Delegated entirely to target.py — changing cfg.target.type selects
    # a different generator with no changes here.
    u_target, target_desc = build_target(pool, cfg)
    print(f"\nTarget unitary: {target_desc}")

    # Sanity-check: every generator must return a unitary
    d = 2 ** cfg.target.num_qubits
    deviation = np.max(np.abs(u_target @ u_target.conj().T - np.eye(d)))
    assert deviation < 1e-10, f"Target is not unitary! Max deviation: {deviation}"
    print(f"  Unitarity verified ✓  (d={d})")

    # ── Build cost function ─────────────────────────────────────────────────
    cost_fn = build_cost_fn(u_target)

    # ── Run GQE training ────────────────────────────────────────────────────
    print("\nStarting GQE training...\n")
    best_cost, best_indices = gqe(cost_fn, pool, cfg)

    # ── Report results ───────────────────────────────────────────────────────
    best_fidelity = 1.0 - best_cost
    print(f"\n{'='*55}")
    print(f"Training complete!")
    print(f"  Best cost  (1 − F):  {best_cost:.6f}")
    print(f"  Best fidelity (F):   {best_fidelity:.6f}")

    if best_indices is not None:
        # best_indices includes start token at position 0 — skip it
        gate_indices = (
            best_indices[1:]
            if isinstance(best_indices, list)
            else best_indices[1:].tolist()
        )
        gate_names = [pool[i][0] for i in gate_indices]
        print(f"  Best circuit ({len(gate_names)} gates):")
        for k, name in enumerate(gate_names):
            print(f"    [{k}] {name}")

        # Independently verify by recomputing fidelity from the raw gate sequence
        gate_matrices = [pool[i][1] for i in gate_indices]
        u_circuit = _compose_unitary(gate_matrices, d)
        verified_f = process_fidelity(u_target, u_circuit)
        print(f"  Verified fidelity:   {verified_f:.6f}")
        # "Verified fidelity" must match "Best fidelity" (up to float rounding)


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Run the demo end-to-end with the default config**

```bash
cd /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch
python main.py
```

Expected output structure (exact values vary with training):
```
Loaded config: config.yml
  Target type:          random
  Qubits:               2
  Discretization bins:  32
  Scheduler:            fixed
  Model size:           small
  Max gates/circuit:    10
  Training epochs:      50
  W&B logging:          False
  Pool size (vocab):    ...

Target unitary: random_reachable depth=4, recipe=[..., ..., ..., ...]
  Unitarity verified ✓  (d=4)

Starting GQE training...
  [Lightning progress bar]

=======================================================
Training complete!
  Best cost  (1 − F):  0.XXXXXX
  Best fidelity (F):   0.XXXXXX
  Best circuit (10 gates):
    [0] ...
    ...
  Verified fidelity:   0.XXXXXX
```

The "Verified fidelity" must match "Best fidelity" exactly. The best fidelity should be meaningfully above the random baseline (`≈ 1/d² = 0.0625` for 2 qubits) after 50 epochs.

- [ ] **Step 11.3: Test the Haar-random target type by editing config.yml temporarily**

In `config.yml`, change `target.type` to `"haar_random"` and re-run:

```bash
python main.py
```

Expected: script runs without error, prints `"Target unitary: haar_random d=4, seed=3047"`, and trains normally. The best fidelity will likely be lower than with `random_reachable` (since Haar targets may not be in the pool's span), but training should still converge.

Restore `target.type: "random"` in `config.yml` after the test.

- [ ] **Step 11.4: Test the cosine scheduler by editing config.yml temporarily**

In `config.yml`, change `temperature.scheduler` to `"cosine"` and re-run:

```bash
python main.py
```

Expected: script runs without error. Restore `temperature.scheduler: "fixed"` after the test.

- [ ] **Step 11.5: Commit**

```bash
git add main.py
git commit -m "feat: add main.py demo, delegates to build_target and factory registries"
```

---

## Task 12: Final Validation

- [ ] **Step 11.1: Run the full test suite**

```bash
cd /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch
python -m pytest tests/ -v --tb=short
```

Expected: All tests PASS. If any fail, diagnose using the error message:
- `ImportError`: check that the failing file uses absolute (not relative) imports
- `AttributeError` on `cfg.*`: check that the attribute path matches the `GQEConfig` dataclass
- Shape mismatch in pipeline test: check that `computeCost` calls `pool[j][1]` (matrix) not `pool[j]`

- [ ] **Step 11.2: Confirm no stray cudaq references remain**

```bash
grep -rn "cudaq\|mpi4py\|ml_collections\|ConfigDict\|MinEnergyCallback\|benchmark_energy" \
    /home/aymantarig/Downloads/AIMS/Quantum/project/gqe-torch/*.py
```

Expected: No output.

- [ ] **Step 11.3: Final end-to-end run**

```bash
python main.py
```

Verify the script completes without error and prints a fidelity value greater than 0.

- [ ] **Step 11.4: Final commit**

```bash
git add tests/ main.py
git commit -m "test: verify full test suite and e2e demo pass"
```

---

## Summary of All Changes

| Change | Task |
|--------|------|
| Complete `config.py` with `validate_config` / `load_config` | 1 |
| Add `operator_pool.py` with discretized Qiskit gates | 2 |
| Add `cost.py` with process fidelity | 3 |
| Rewrite `data.py`: `energy` → `cost` | 4 |
| Rewrite `pipeline.py`: remove cudaq, absolute imports, use `GQEConfig` paths | 4 |
| Rename `MinEnergyCallback` → `BestCostCallback` in `callbacks.py` | 5 |
| Rename `energies` → `costs` in `loss.py` | 6 |
| Add clamping to `DefaultScheduler`; fix `VarBasedScheduler` kwarg | 7 |
| Rewrite `factory.py`: `SCHEDULER_BUILDERS` registry, absolute imports | 7 |
| Rewrite `gqe.py`: remove cudaq, absolute imports | 8 |
| Gut `utils.py` and `pool.py`; update `requirements.txt` | 9 |
| Add `target.py` with `TARGET_REGISTRY` (random_reachable + haar_random) | 10 |
| Add `main.py` demo (uses `build_target`, no if/elif) | 11 |

---

## Key Design Decisions

1. **Flat module structure (absolute imports):** The project has no `__init__.py` — it's a flat collection of Python files. All imports use `from data import ...` not `from .data import ...`. Relative imports would break when `main.py` and test files import modules directly.

2. **Frozen dataclasses for config:** `@dataclass(frozen=True)` prevents accidental mutation during training. Once loaded from YAML, the config cannot be changed.

3. **`(name, matrix)` tuples for the pool:** The name is needed for human-readable output (decoding the best circuit). The matrix is needed for cost computation. A tuple is the simplest structure with no overhead.

4. **Direct NumPy matrix multiplication instead of Qiskit simulation:** For 1-3 qubit systems, `np.matmul` on small matrices is orders of magnitude faster than building a Qiskit `QuantumCircuit` and running it through Aer. GQE evaluates thousands of circuits per epoch — speed matters. Qiskit is used only to obtain the canonical gate matrix definitions.

5. **Separate `num_samples` and `batch_size`:** `training.num_samples` is the rollout size — how many circuits are generated and evaluated each epoch before being stored in the replay buffer. `training.batch_size` is the DataLoader batch size used for gradient steps. These are independent: a large `num_samples` fills the buffer faster; `batch_size` controls gradient noise. The `train_dataloader()` uses `cfg.training.batch_size`.

6. **Process fidelity is global-phase invariant:** `F = |Tr(U†V)|²/d²` equals 1.0 if V = e^{iφ}U for any phase φ. This means the GQE only needs to find a circuit matching the target up to global phase — a weaker and more achievable condition than exact matrix equality.

7. **Random reachable target:** A target sampled as a short random circuit from the same pool is guaranteed to be expressible by the GQE vocabulary. This makes the demo tractable and ensures fidelity 1.0 is achievable in principle.

8. **`VarBasedScheduler` accepts both `costs` and `energies`:** `kwargs.get("costs", kwargs.get("energies"))` makes it forward-compatible with the new naming while remaining backward-compatible with any existing code that passes `energies=`.

---

## Future Extensions

This section describes how to extend the system after the initial implementation. Both extension points require **no changes to core training code** — only add a class/function and register it.

---

### Adding a New Temperature Scheduler

**Where to look:** `scheduler.py` (implementation) + `factory.py` (registration)

**Protocol:** Any scheduler must implement:

```python
class MyScheduler:
    def update(self, **kwargs) -> None:
        """Called at end of each epoch. kwargs always contains 'costs' (np.ndarray)."""
        ...

    @property
    def beta(self) -> float:
        """Current inverse temperature β. Sampling uses Categorical(logits=-β·w)."""
        ...
```

**Step 1 — Implement the class in `scheduler.py`:**

```python
class AnnealingScheduler:
    """Geometric annealing: β doubles every `doubling_epochs` epochs."""

    def __init__(self, initial: float, doubling_epochs: int, maximum: float) -> None:
        self._beta = initial
        self._doubling_epochs = doubling_epochs
        self._maximum = maximum
        self._epoch = 0

    def update(self, **kwargs) -> None:
        self._epoch += 1
        if self._epoch % self._doubling_epochs == 0:
            self._beta = min(self._beta * 2.0, self._maximum)

    @property
    def beta(self) -> float:
        return self._beta
```

**Step 2 — Register it in `factory.py`:**

```python
from scheduler import DefaultScheduler, CosineScheduler, VarBasedScheduler, AnnealingScheduler

SCHEDULER_BUILDERS = {
    "fixed": lambda: DefaultScheduler(...),
    "cosine": lambda: CosineScheduler(...),
    "variance": lambda: VarBasedScheduler(...),
    # New entry — one line:
    "annealing": lambda: AnnealingScheduler(
        initial=cfg.temperature.initial_value,
        doubling_epochs=max(1, cfg.training.max_epochs // 4),
        maximum=cfg.temperature.max_value,
    ),
}
```

**Step 3 — Update `config.py` validation:**

```python
VALID_SCHEDULERS = {"fixed", "cosine", "variance", "annealing"}
```

**Step 4 — Use it in `config.yml`:**

```yaml
temperature:
  scheduler: "annealing"
  initial_value: 0.1
  max_value: 10.0
```

**That's it.** `factory.py → build_scheduler(cfg)` already looks up `SCHEDULER_BUILDERS[cfg.temperature.scheduler]`. No other files change.

---

**Catalog of future scheduler ideas:**

| Name | Description | When to use |
|------|-------------|-------------|
| `"annealing"` | Geometric (doubling) warmup | Long runs; start exploring, then exploit |
| `"cyclic"` | Cosine restart every K epochs | Avoid local optima; promote diversity |
| `"adaptive_variance"` | Increase β when cost variance drops below threshold | Automatic convergence detection |
| `"step"` | Fixed multiplier every N epochs | Simple monotonic schedules |
| `"warmup_linear"` | Linear warmup then linear decay | Stable early training |

---

### Adding a New Target Unitary Generator

**Where to look:** `target.py` (implementation + registration)

**Protocol:** A `TargetGenerator` is any callable with this exact signature:

```python
def my_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """
    Args:
        pool: The operator pool — list of (name, gate_matrix) tuples.
              gate_matrix.shape == (d, d), dtype=complex128.
        cfg:  Full GQEConfig (frozen dataclass). Use cfg.target.num_qubits,
              cfg.training.seed, and any custom fields from TargetConfig.

    Returns:
        u_target: np.ndarray, shape (d, d), dtype=complex128, must be unitary.
        description: human-readable string, e.g. "haar_random d=4, seed=42".
    """
    ...
    return u_target, description
```

**Step 1 — Implement the function in `target.py`:**

```python
def identity_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Target is the identity — useful for debugging (optimal cost = 0 trivially)."""
    d = 2 ** cfg.target.num_qubits
    u_target = np.eye(d, dtype=np.complex128)
    return u_target, f"identity d={d}"
```

**Step 2 — Register it in `TARGET_REGISTRY` (same file, `target.py`):**

```python
TARGET_REGISTRY: Dict[str, TargetGenerator] = {
    "random":           random_reachable_generator,
    "random_reachable": random_reachable_generator,
    "haar_random":      haar_random_generator,
    # New entry — one line:
    "identity":         identity_generator,
}
```

**Step 3 — Update `config.py` validation:**

```python
VALID_TARGET_TYPES = {"random", "random_reachable", "haar_random", "identity"}
```

**Step 4 — Use it in `config.yml`:**

```yaml
target:
  num_qubits: 2
  type: "identity"
```

**That's it.** `build_target(pool, cfg)` already looks up `TARGET_REGISTRY[cfg.target.type]`. `main.py` never changes.

---

**Catalog of future target generator ideas:**

| `cfg.target.type` | Description | Notes |
|-------------------|-------------|-------|
| `"identity"` | `I_d` — trivial baseline | Optimal cost = 0 for any circuit |
| `"pauli_x"` | `X ⊗ I ⊗ ... ⊗ I` (flip qubit 0) | Simple 1-gate solution exists |
| `"file"` | Load from `.npy` file path in config | Allows replaying known hard targets |
| `"fixed_circuit"` | Build from a user-specified gate sequence string | Reproducible non-trivial targets |
| `"random_clifford"` | Haar-random over the Clifford group | Structured, potentially easier |
| `"qft"` | Quantum Fourier Transform for n qubits | Closed-form target, increasing difficulty |

**Example `"file"` generator** (loads a saved unitary):

```python
def file_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Load target unitary from a .npy file. Expects cfg.target.path to be set."""
    path = getattr(cfg.target, "path", None)
    if path is None:
        raise ValueError("target.type='file' requires target.path in config.yml")
    u_target = np.load(path).astype(np.complex128)
    d = 2 ** cfg.target.num_qubits
    if u_target.shape != (d, d):
        raise ValueError(f"Loaded unitary shape {u_target.shape} != ({d},{d})")
    return u_target, f"file:{path}"
```

> Note: to add `path` to `TargetConfig`, add `path: str | None = None` as an optional field in the `TargetConfig` dataclass in `config.py`.

---

### Extension Checklist (both types)

When adding any new scheduler or target type, always:

- [ ] Implement the class/function with the correct protocol signature
- [ ] Add one entry to the registry dict (`SCHEDULER_BUILDERS` or `TARGET_REGISTRY`)
- [ ] Add the new name to `VALID_SCHEDULERS` or `VALID_TARGET_TYPES` in `config.py`
- [ ] Write at least one pytest test (unitary check for targets; beta-range check for schedulers)
- [ ] Update `config.yml` if you want it as the default demo type
