# Implementation Plan: Hybrid Discrete-Continuous Optimization

## Problem Being Solved

The current GQE pipeline treats rotation angles as discrete tokens (e.g. `RZ_q0_k17` maps to
a fixed pre-computed matrix at angle `17 × 2π/32`). This has two consequences:

1. **Quantization floor**: the best possible fidelity is bounded by the angle grid resolution.
   With 32 bins, each step is `2π/32 ≈ 11.25°` — far too coarse for a generic SU(4) target.
2. **No gradient signal through angles**: GRPO can only swap tokens, not nudge angles.
   The model cannot learn "RZ_q0_k17 is almost right, try k18" — it sees only whole-sequence rewards.

The fix (supported by *Gradient Descent Synthesis* paper): use GQE to search the **structure**
(gate types + qubit assignments), then use gradient descent to find the **optimal continuous
angles** for that structure. GRPO then receives the post-optimization fidelity as the reward,
which is both higher and more informative.

---

## Architecture Overview

```
GQE rollout
    │
    ▼
token sequence [RZ_q0_k3, CNOT_q0_q1, RZ_q1_k11, SX_q0, ...]
    │
    ▼  parse_gate_specs()
list of GateSpec objects   ←── parametric (rotation) or fixed (SX, CNOT)
    │
    ▼  extract initial angles from k-indices (warm start)
θ_init = [angles[k3], angles[k11], ...]
    │
    ▼  run_optimization(gate_specs, θ_init, u_target)
θ_opt  (L-BFGS or Adam, N steps)
    │
    ▼
optimized_fidelity  ──► GRPO buffer   (replaces naive discrete cost)
```

The key insight: the token sequence still determines **structure** (which gates, which qubits,
what order). Only the continuous angles are optimized after the fact. GQE learns to propose
good structures; gradient descent fills in the exact angles.

---

## New and Modified Files

| File | Change |
|------|--------|
| `continuous_optimizer.py` | **New** — all optimization logic |
| `config.py` | Add `ContinuousOptConfig` dataclass + field in `GQEConfig` |
| `config.yml` | Add `continuous_opt` section |
| `factory.py` | Add `create_continuous_optimizer` method |
| `pipeline.py` | Call optimizer inside `computeCost` when enabled |

---

## Step 1 — Create `continuous_optimizer.py`

This file contains all the differentiable circuit machinery and the optimization loop.

### 1a — GateSpec dataclass

Parsed representation of a single gate in a token sequence.

```python
from dataclasses import dataclass

@dataclass
class GateSpec:
    gate_type: str          # "RX", "RY", "RZ", "SX", "CNOT"
    qubits: tuple           # (q,) for single-qubit; (ctrl, tgt) for two-qubit
    initial_angle: float    # initial angle (from discretization); 0.0 for fixed gates
    is_parametric: bool     # True for RX/RY/RZ; False for SX/CNOT
```

### 1b — Token name parser

Converts a pool token name string into a `GateSpec`. Uses the discretization angles
array for warm-start initialization.

```python
def parse_gate_spec(token_name: str, angles: np.ndarray) -> GateSpec:
    """
    Parse a pool token name into a GateSpec.

    Token name formats:
        "RZ_q0_k17"     → axis=RZ, qubit=0, initial_angle=angles[17]
        "RX_q1_k3"      → axis=RX, qubit=1, initial_angle=angles[3]
        "SX_q0"         → gate=SX, qubit=0, no angle
        "CNOT_q0_q1"    → gate=CNOT, ctrl=0, tgt=1, no angle
    """
    if token_name.startswith(("RX", "RY", "RZ")):
        parts = token_name.split("_")
        axis = parts[0]
        qubit = int(parts[1][1:])
        k = int(parts[2][1:])
        return GateSpec(axis, (qubit,), float(angles[k]), is_parametric=True)
    elif token_name.startswith("SX"):
        qubit = int(token_name.split("_")[1][1:])
        return GateSpec("SX", (qubit,), 0.0, is_parametric=False)
    elif token_name.startswith("CNOT"):
        parts = token_name.split("_")
        ctrl = int(parts[1][1:])
        tgt = int(parts[2][1:])
        return GateSpec("CNOT", (ctrl, tgt), 0.0, is_parametric=False)
    else:
        raise ValueError(f"Unknown token name: {token_name!r}")
```

### 1c — Differentiable gate matrices

All functions return `torch.complex128` tensors so that gradients flow through angles.

```python
import torch

def _rz(theta: torch.Tensor) -> torch.Tensor:
    """2×2 RZ(θ) = diag(exp(-iθ/2), exp(iθ/2))"""
    h = theta / 2
    zero = torch.zeros_like(h)
    top    = torch.complex(torch.cos(-h), torch.sin(-h))
    bottom = torch.complex(torch.cos(h),  torch.sin(h))
    return torch.stack([
        torch.stack([top,  torch.zeros((), dtype=torch.complex128, device=theta.device)]),
        torch.stack([torch.zeros((), dtype=torch.complex128, device=theta.device), bottom])
    ])

def _rx(theta: torch.Tensor) -> torch.Tensor:
    """2×2 RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]"""
    h = theta / 2
    c = torch.cos(h).to(torch.complex128)
    s = (torch.sin(h) * (-1j)).to(torch.complex128)  # -i·sin
    return torch.stack([torch.stack([c, s]), torch.stack([s, c])])

def _ry(theta: torch.Tensor) -> torch.Tensor:
    """2×2 RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]"""
    h = theta / 2
    c = torch.cos(h).to(torch.complex128)
    s = torch.sin(h).to(torch.complex128)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

_ROTATION_BUILDERS = {"RX": _rx, "RY": _ry, "RZ": _rz}
```

### 1d — Fixed gate constants

Pre-computed as constant tensors. No gradient needed through these.

```python
# SX = (1/2) * [[1+i, 1-i], [1-i, 1+i]]
_SX = torch.tensor(
    [[0.5 + 0.5j, 0.5 - 0.5j],
     [0.5 - 0.5j, 0.5 + 0.5j]],
    dtype=torch.complex128
)

# CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
_CNOT = torch.tensor(
    [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
    dtype=torch.complex128
)
```

### 1e — N-qubit embedding

```python
def _embed_single(gate_2x2: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    """Embed a 2×2 gate into the n-qubit space. Qubit 0 = MSB (Qiskit convention)."""
    eye = torch.eye(2, dtype=torch.complex128, device=gate_2x2.device)
    factors = [gate_2x2 if q == qubit else eye for q in range(num_qubits)]
    result = factors[0]
    for f in factors[1:]:
        result = torch.kron(result, f)
    return result


def _embed_cnot(ctrl: int, tgt: int, num_qubits: int, device) -> torch.Tensor:
    """
    Embed CNOT into n-qubit space.
    For 2 qubits this is just the standard 4×4 CNOT (with possible qubit ordering swap).
    For n > 2, use the same column-by-column construction as operator_pool.py,
    but with pre-built torch tensors.
    """
    d = 2 ** num_qubits
    cnot = _CNOT.to(device)

    if num_qubits == 2 and ctrl == 0 and tgt == 1:
        return cnot
    if num_qubits == 2 and ctrl == 1 and tgt == 0:
        # Swap qubit ordering: apply Hadamard-like basis change
        # CNOT_{1→0} = SWAP · CNOT_{0→1} · SWAP
        swap = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],
                             dtype=torch.complex128, device=device)
        return swap @ cnot @ swap

    # General n-qubit case: mirror operator_pool._embed_two_qubit logic in torch
    full = torch.zeros((d, d), dtype=torch.complex128, device=device)
    for col in range(d):
        bits = [(col >> (num_qubits - 1 - q)) & 1 for q in range(num_qubits)]
        two_q_in = (bits[ctrl] << 1) | bits[tgt]
        for two_q_out in range(4):
            amp = cnot[two_q_out, two_q_in]
            if amp.abs() < 1e-15:
                continue
            new_bits = bits.copy()
            new_bits[ctrl] = (two_q_out >> 1) & 1
            new_bits[tgt] = two_q_out & 1
            row = sum(new_bits[q] << (num_qubits - 1 - q) for q in range(num_qubits))
            full[row, col] += amp
    return full
```

### 1f — Differentiable circuit builder

Composes all gates into the full circuit unitary. Parametric gates use `params`;
fixed gates use their constant tensors.

```python
def build_circuit_unitary(
    gate_specs: list,
    params: torch.Tensor,     # shape: (num_parametric,), real float64
    num_qubits: int,
) -> torch.Tensor:
    """
    Build the full circuit unitary as a differentiable torch tensor.

    gate_specs: list of GateSpec (from parse_gate_spec)
    params:     1-D tensor, one entry per parametric gate (in order of appearance)
    num_qubits: system size

    Returns: (2^n × 2^n) complex128 tensor
    """
    d = 2 ** num_qubits
    device = params.device
    u = torch.eye(d, dtype=torch.complex128, device=device)

    param_idx = 0
    for spec in gate_specs:
        if spec.is_parametric:
            theta = params[param_idx]
            param_idx += 1
            gate_2x2 = _ROTATION_BUILDERS[spec.gate_type](theta)
            gate_full = _embed_single(gate_2x2, spec.qubits[0], num_qubits)
        elif spec.gate_type == "SX":
            gate_full = _embed_single(
                _SX.to(device), spec.qubits[0], num_qubits
            )
        elif spec.gate_type == "CNOT":
            gate_full = _embed_cnot(
                spec.qubits[0], spec.qubits[1], num_qubits, device
            )
        else:
            raise ValueError(f"Unknown gate type: {spec.gate_type!r}")

        u = gate_full @ u   # gate applied left-to-right (same as cost.py convention)

    return u
```

### 1g — Differentiable process fidelity

```python
def process_fidelity_torch(u_target: torch.Tensor, u_circuit: torch.Tensor) -> torch.Tensor:
    """F = |Tr(U†_target · U_circuit)|² / d²  (phase-invariant)"""
    d = u_target.shape[0]
    overlap = torch.trace(u_target.conj().T @ u_circuit)
    return overlap.abs() ** 2 / (d ** 2)
```

### 1h — The optimization loop

Runs L-BFGS (preferred for small param count) or Adam. Returns `(optimized_fidelity, optimized_angles)`.

```python
def optimize_angles(
    gate_specs: list,
    u_target_torch: torch.Tensor,
    initial_angles: torch.Tensor,
    num_qubits: int,
    steps: int,
    lr: float,
    optimizer_type: str = "lbfgs",   # "lbfgs" or "adam"
) -> tuple[float, torch.Tensor]:
    """
    Optimize the continuous rotation angles for a fixed gate structure.

    Returns:
        (optimized_fidelity: float, optimized_params: torch.Tensor)
    """
    params = initial_angles.clone().detach().to(torch.float64).requires_grad_(True)

    if optimizer_type == "lbfgs":
        opt = torch.optim.LBFGS(
            [params], lr=lr, max_iter=steps,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            u = build_circuit_unitary(gate_specs, params, num_qubits)
            loss = 1.0 - process_fidelity_torch(u_target_torch, u)
            loss.real.backward()
            return loss

        opt.step(closure)

    elif optimizer_type == "adam":
        opt = torch.optim.Adam([params], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            u = build_circuit_unitary(gate_specs, params, num_qubits)
            loss = 1.0 - process_fidelity_torch(u_target_torch, u)
            loss.real.backward()
            opt.step()
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type!r}")

    with torch.no_grad():
        u_final = build_circuit_unitary(gate_specs, params, num_qubits)
        fidelity = float(process_fidelity_torch(u_target_torch, u_final).real)

    return fidelity, params.detach()
```

### 1i — ContinuousOptimizer class (public API)

The main class that the pipeline calls. Holds all configuration and state.

```python
class ContinuousOptimizer:
    """
    Wraps continuous angle optimization for GQE-generated circuit skeletons.

    Args:
        u_target:        Target unitary as numpy array (captured at construction).
        angles:          Discretization angles array from get_discretized_angles().
        num_qubits:      Number of qubits.
        steps:           Gradient steps per optimization call.
        lr:              Learning rate.
        optimizer_type:  "lbfgs" (recommended) or "adam".
        top_k:           Only optimize the top-k circuits by initial discrete fidelity
                         per call to optimize_batch(). 0 = optimize all.
    """

    def __init__(self, u_target, angles, num_qubits, steps, lr, optimizer_type, top_k):
        self.angles = angles
        self.num_qubits = num_qubits
        self.steps = steps
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.top_k = top_k
        self.u_target_torch = torch.tensor(u_target, dtype=torch.complex128)

    def optimize_circuit(self, token_names: list[str]) -> float:
        """
        Parse a token name list, run angle optimization, return optimized fidelity.

        Args:
            token_names: List of pool token name strings (e.g. ["RZ_q0_k3", "CNOT_q0_q1", ...])
        Returns:
            Optimized fidelity (float in [0, 1])
        """
        gate_specs = [parse_gate_spec(name, self.angles) for name in token_names]
        initial_angles = torch.tensor(
            [s.initial_angle for s in gate_specs if s.is_parametric],
            dtype=torch.float64,
        )

        if len(initial_angles) == 0:
            # No parametric gates — nothing to optimize, return discrete fidelity
            u = build_circuit_unitary(gate_specs, initial_angles, self.num_qubits)
            return float(process_fidelity_torch(self.u_target_torch, u).real)

        fidelity, _ = optimize_angles(
            gate_specs,
            self.u_target_torch,
            initial_angles,
            self.num_qubits,
            self.steps,
            self.lr,
            self.optimizer_type,
        )
        return fidelity
```

---

## Step 2 — Add `ContinuousOptConfig` to `config.py`

```python
@dataclass(frozen=True)
class ContinuousOptConfig:
    enabled: bool = False
    steps: int = 50             # gradient steps per circuit; 50 is enough for L-BFGS
    lr: float = 0.1             # learning rate; higher is fine for L-BFGS
    optimizer: str = "lbfgs"    # "lbfgs" (recommended) or "adam"
    top_k: int = 0              # 0 = optimize all circuits; N = optimize only top-N per rollout
```

Add the field to `GQEConfig`:

```python
@dataclass(frozen=True)
class GQEConfig:
    ...
    continuous_opt: ContinuousOptConfig = field(default_factory=ContinuousOptConfig)
```

Update `load_config` to parse the optional section:

```python
co_raw = raw.get("continuous_opt", {})
return GQEConfig(
    ...
    continuous_opt=ContinuousOptConfig(**co_raw) if co_raw else ContinuousOptConfig(),
)
```

---

## Step 3 — Add `continuous_opt` section to `config.yml`

```yaml
continuous_opt:
  enabled: true
  steps: 50         # L-BFGS converges in ~20-50 steps for circuits with ≤28 params
  lr: 0.1
  optimizer: lbfgs  # lbfgs is ~10x faster than adam for small param counts
  top_k: 0          # optimize all circuits; set to e.g. 20 to only optimize top-20% by discrete fidelity
```

---

## Step 4 — Add `create_continuous_optimizer` to `factory.py`

```python
from continuous_optimizer import ContinuousOptimizer
from operator_pool import get_discretized_angles

def create_continuous_optimizer(self, cfg, u_target, pool) -> ContinuousOptimizer | None:
    co_cfg = getattr(cfg, "continuous_opt", None)
    if co_cfg is None or not co_cfg.enabled:
        return None
    angles = get_discretized_angles(cfg.pool.discretization_factor)
    return ContinuousOptimizer(
        u_target=u_target,
        angles=angles,
        num_qubits=cfg.target.num_qubits,
        steps=co_cfg.steps,
        lr=co_cfg.lr,
        optimizer_type=co_cfg.optimizer,
        top_k=co_cfg.top_k,
    )
```

Note: `u_target` is not currently available inside `Pipeline`. It needs to be threaded from
`main.py → gqe() → Pipeline.__init__`. See Step 5 for how this is handled.

---

## Step 5 — Wire into `pipeline.py`

### 5a — Thread `u_target` through `gqe()`

`ContinuousOptimizer` needs the target unitary at construction time.
The cleanest way is to pass it to `Pipeline.__init__` alongside the cost function:

In `gqe.py`, update the `gqe()` call:
```python
pipeline = Pipeline(cfg, cost_fn, pool, model, factory, u_target=u_target)
```

In `Pipeline.__init__`, add the parameter and construct the optimizer:
```python
def __init__(self, cfg, cost, pool, model, factory, u_target=None):
    ...
    self.continuous_optimizer = self.factory.create_continuous_optimizer(
        cfg, u_target, pool
    ) if u_target is not None else None
```

### 5b — Modify `computeCost`

Replace the discrete cost with the continuous-optimized cost when enabled.
If `top_k > 0`, first rank circuits by their discrete fidelity and only optimize the top-k.

```python
@torch.no_grad()  # outer no_grad; optimizer runs its own grad context internally
def computeCost(self, idx_output, pool, **kwargs):
    token_names_batch = [
        [pool[j][0] for j in row.tolist()]
        for row in idx_output
    ]

    if self.continuous_optimizer is None:
        # Original behaviour: use pre-computed discrete matrices
        results = []
        for token_names in token_names_batch:
            gate_matrices = [pool[int(pool_index)][1]
                             for pool_index, name in enumerate(pool)
                             if name in token_names]  # keep original path
            ...
        return torch.tensor(results, dtype=torch.float32)

    # Continuous optimization path
    if self.continuous_optimizer.top_k > 0:
        # Compute cheap discrete fidelity first to rank circuits
        discrete_costs = [
            self._cost([pool[j][1] for j in row.tolist()])
            for row in idx_output
        ]
        k = self.continuous_optimizer.top_k
        top_indices = sorted(range(len(discrete_costs)), key=lambda i: discrete_costs[i])[:k]
        results = list(discrete_costs)
        for i in top_indices:
            results[i] = 1.0 - self.continuous_optimizer.optimize_circuit(token_names_batch[i])
    else:
        results = [
            1.0 - self.continuous_optimizer.optimize_circuit(names)
            for names in token_names_batch
        ]

    return torch.tensor(results, dtype=torch.float32)
```

> **Note on `@torch.no_grad()`**: The outer decorator disables grad tracking for
> the pipeline's generation machinery. The `optimize_angles()` function internally
> calls `.requires_grad_(True)` on its own parameter tensor, so gradients still
> flow correctly inside the optimizer. These two grad contexts are independent.

---

## Step 6 — Log continuous opt activity

In `gqe.py`, add a log metric to make it easy to see whether optimization is
running and what the speedup in fidelity looks like:

```python
if logger:
    logger.log_metrics(
        {
            ...
            "continuous_opt_enabled": int(pipeline.continuous_optimizer is not None),
        },
        step=epoch,
    )
```

---

## Expected Behavior After Implementation

| Phase | What happens |
|-------|-------------|
| Rollout | GQE samples 100 circuit structures (token sequences) |
| Optimization | For each circuit, gradient descent finds the best angles for that structure |
| Buffer | Stores (token_sequence, optimized_fidelity) — fidelity is now meaningful |
| GRPO | Computes advantages over optimized fidelities — variance is non-zero |
| Model update | Model learns which **structures** lead to high fidelity, not which angle bins |

The gradient does not flow from optimized fidelity back into the transformer weights —
the transformer is trained purely through GRPO. The continuous optimizer is only a
**reward shaper**: it converts a discrete circuit skeleton into a realistic upper-bound
fidelity estimate for that skeleton.

---

## Tuning Guidance

| Parameter | Recommended start | Notes |
|-----------|------------------|-------|
| `optimizer` | `lbfgs` | 10–50 steps typically enough; much faster than Adam per step for ≤30 params |
| `steps` | 50 | Increase to 100 if fidelity still has room to improve |
| `lr` | 0.1 | For L-BFGS, `lr` sets the initial step size; strong Wolfe line search adjusts it |
| `top_k` | 0 (all) | If speed is a concern, set to 20–30 (optimize top 20–30% per rollout) |

## Known Limitations

1. **Speed**: optimizing 100 circuits × 50 L-BFGS steps per epoch adds wall-clock time.
   Mitigate with `top_k` or by reducing `num_samples`.
2. **Structure vs angle entanglement**: GRPO rewards a structure by its **best possible** angles.
   If two structures have the same optimized fidelity, GRPO provides no signal between them.
   This is correct behavior — it means both structures are equally good.
3. **No angle feedback to the model**: the transformer never sees the optimized angle values.
   This is by design — the transformer only needs to learn structures. If angle feedback
   is desired in the future, one option is to include the optimized angles as context tokens
   in a conditional architecture.
