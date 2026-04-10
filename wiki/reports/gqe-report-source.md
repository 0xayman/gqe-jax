# GQE-Torch: Quantum Circuit Compilation with Generative Models

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [The GQE Approach](#2-the-gqe-approach)
   - 2.1 [Architecture Overview](#21-architecture-overview)
   - 2.2 [Operator Pool](#22-operator-pool)
   - 2.3 [Cost Function](#23-cost-function)
   - 2.4 [Training Loop](#24-training-loop)
   - 2.5 [Temperature Scheduling](#25-temperature-scheduling)
   - 2.6 [Replay Buffer](#26-replay-buffer)
3. [Continuous Angle Optimization](#3-continuous-angle-optimization)
   - 3.1 [Why Continuous Optimization is Needed](#31-why-continuous-optimization-is-needed)
   - 3.2 [L-BFGS](#32-l-bfgs)
   - 3.3 [Adam](#33-adam)
   - 3.4 [Multiple Random Restarts](#34-multiple-random-restarts)
4. [Post-Processing: Gate Simplification](#4-post-processing-gate-simplification)
5. [Training Signal: GRPO Loss](#5-training-signal-grpo-loss)
6. [Circuit Grammar: Structural Sampling Constraints](#6-circuit-grammar-structural-sampling-constraints)
   - 6.1 [Motivation](#61-motivation)
   - 6.2 [Exact Algebraic Rules](#62-exact-algebraic-rules)
   - 6.3 [Canonical Ordering](#63-canonical-ordering)
   - 6.4 [Implementation](#64-implementation)
7. [Baseline Comparison: Qiskit Compiler](#7-baseline-comparison-qiskit-compiler)
8. [Known Limitations and Open Problems](#8-known-limitations-and-open-problems)
9. [Configuration Reference](#9-configuration-reference)

---

## 1. Problem Definition

**Quantum circuit compilation** is the problem of finding a sequence of hardware-native gates that implements a given target unitary transformation $U_\text{target} \in SU(d)$, where $d = 2^n$ and $n$ is the number of qubits.

Given a fixed **operator pool** (the set of available hardware gates), the goal is to find:

$$j^* = \arg\min_{j \in \text{sequences}} \left(1 - F_\text{process}(U_\text{target},\, U_j)\right)$$

where the **process fidelity** is:

$$F_\text{process}(U_\text{target}, U_\text{circuit}) = \frac{|\text{Tr}(U_\text{target}^\dagger \cdot U_\text{circuit})|^2}{d^2}$$

This metric is phase-invariant (global phase does not affect it) and equals 1.0 if and only if $U_\text{circuit} = e^{i\phi} U_\text{target}$ for some real $\phi$.

The search space is **combinatorial and exponential**: for a pool of $V$ gates and sequences of length $N$, there are $V^N$ possible circuits. For the current configuration ($V = 6$, $N = 28$), this is approximately $6^{28} \approx 10^{21}$ possible circuits — completely intractable by exhaustive search.

Additionally, circuits with rotation gates (RX, RY, RZ) have **continuous parameters** (rotation angles), making the space a hybrid discrete-continuous optimization problem. This dual nature is what makes the problem particularly challenging:

- **Discrete part**: which gates appear and in which order.
- **Continuous part**: what rotation angle each parametric gate uses.

The secondary objective is **circuit efficiency**: minimize depth and total gate count (especially two-qubit CNOT gates, which are slow and error-prone on real hardware) while maintaining high fidelity.

---

## 2. The GQE Approach

The Generative Quantum Eigensolver (GQE) addresses the discrete part of the problem by training a **transformer language model** to autoregressively generate good gate sequences, guided by a reinforcement-learning-style training signal.

### 2.1 Architecture Overview

The model is a **GPT-2 transformer** (`model.py`) that treats gate selection as token prediction. The vocabulary consists of the operator pool (each gate is one token). Given a partial sequence of gates, the model predicts the next gate token.

Available size configurations:

| Size   | Layers | Heads | Embedding dim |
|--------|--------|-------|---------------|
| tiny   | 2      | 2     | 128           |
| small  | 6      | 6     | 384           |
| medium | 12     | 12    | 768           |
| large  | 24     | 16    | 1024          |

**Why a language model?** Gate sequences share structural properties with language: context matters (a CNOT after RZ has a different effect than RZ after CNOT), and the model must learn which structural patterns lead to high-fidelity circuits — analogous to learning grammar and semantics.

**Pros:**
- Handles variable-length sequences natively.
- The attention mechanism can learn long-range gate dependencies.
- Pre-training on large corpora of circuits is possible (transfer learning).

**Cons:**
- Training requires many circuit evaluations (costly for large qubit counts).
- The model has no built-in notion of unitarity or quantum physics.
- No guarantee of finding the optimal circuit.

### 2.2 Operator Pool

The pool (`operator_pool.py`) is the vocabulary of available gates. For $n$ qubits it contains:

| Gate | Type | Parameters |
|------|------|------------|
| `RZ_qi` | Rotation (Z-axis) on qubit $i$ | 1 continuous angle $\theta$ |
| `SX_qi` | Square root of X on qubit $i$ | None (fixed) |
| `CNOT_qi_qj` | Controlled-NOT, ctrl $i$, target $j$ | None (fixed) |

For 2 qubits: 2 RZ + 2 SX + 2 CNOT = **6 tokens**.

This matches IBM Quantum's native gate set `{rz, sx, cx}`, making compiled circuits directly executable on IBM hardware without further transpilation.

Each gate is stored as a pre-computed $2^n \times 2^n$ unitary matrix. Rotation gates use a placeholder angle ($\pi/4$) for the discrete evaluation path; the actual angle is found by continuous optimization when `continuous_opt.enabled = true`.

**Note:** Only RZ is included in `ROTATION_AXIS` (not RX/RY). RX and RY can be synthesized from RZ and SX (`RX(θ) = SX · RZ(θ) · SX†`), so the gate set is still universal.

### 2.3 Cost Function

The cost function (`cost.py`) wraps process fidelity:

$$\text{cost}(j) = 1 - F_\text{process}(U_\text{target}, U_j) \in [0, 1]$$

Lower cost = better circuit. Cost = 0 means perfect fidelity.

The circuit unitary is computed as:

$$U_j = G_{j_N} \cdots G_{j_2} \cdot G_{j_1}$$

where $G_{j_1}$ is applied first to the quantum state (rightmost in the product).

### 2.4 Training Loop

The training loop (`gqe.py`, `pipeline.py`) follows the GQE paper's procedure:

```
1. Warm-up: collect rollouts until buffer has warmup_size entries
2. For each epoch:
   a. Rollout: generate num_samples circuits autoregressively
   b. Evaluate: compute cost for each circuit (via continuous optimizer if enabled)
   c. Buffer: push (sequence, cost) pairs into the replay buffer
   d. Train: take steps_per_epoch gradient steps on batches from the buffer
   e. Track: update best_cost and best_indices if improved
   f. Early stop: if early_stop=true and best fidelity ≥ 1.0, halt immediately
```

**Generation** (`pipeline.py:generate`): The model samples token by token using:
$$p(\text{token}) \propto \text{Categorical}(\text{logits} = -\beta \cdot w)$$
where $\beta$ is the inverse temperature and $w$ are the model's output logits. Higher $\beta$ makes sampling more deterministic (greedy); lower $\beta$ increases exploration. When `grammar.enabled = true`, a `CircuitGrammar` mask is applied before each sampling step, setting forbidden gate logits to $+\infty$ so their probability is exactly zero (see Section 6).

**Best circuit tracking**: The globally best circuit (lowest cost ever seen) is tracked across all epochs and returned at the end of training. This means the model's final output is the best circuit found during the entire training run, not just the last epoch.

### 2.5 Temperature Scheduling

The inverse temperature $\beta$ controls the exploration-exploitation tradeoff during sampling. Three schedulers are available:

| Scheduler  | Behavior | When to use |
|------------|----------|-------------|
| `fixed`    | Linear ramp: $\beta_{t+1} = \beta_t + \delta$, clamped to $[\beta_\min, \beta_\max]$ | Default; simple and reliable |
| `cosine`   | Cosine oscillation between $\beta_\min$ and $\beta_\max$ | Periodic exploration-exploitation cycling |
| `variance` | Adaptive: increases $\beta$ when cost variance is high | Self-tuning; handles convergence automatically |

**Current setting**: `fixed` with $\beta_0 = 0.5$, $\delta = 0.02$. The temperature gradually increases, shifting from exploration toward exploitation over training.

### 2.6 Replay Buffer

The replay buffer (`data.py`) is a FIFO queue of (sequence, cost) pairs with configurable maximum size. Its role is to decouple data collection from gradient updates:

- **Size 500** (current): retains circuits from up to 5 recent epochs, providing diverse training data.
- `steps_per_epoch = 4`: 4 gradient steps are taken per epoch, reusing buffer data.
- Larger buffers improve data diversity but may include stale circuits from when the model was less trained.

---

## 3. Continuous Angle Optimization

### 3.1 Why Continuous Optimization is Needed

Without continuous optimization, every rotation gate in a sampled circuit uses the pool's placeholder angle ($\pi/4$). This means two structurally identical circuits with different optimal angles look the same to the model, and the fidelity signal is severely degraded.

With `continuous_opt.enabled = true`, **for every sampled circuit**, the rotation angles are independently optimized to maximize fidelity for that structure. This transforms the problem from:

> "Find the gate structure AND angles simultaneously"

to:

> "Find the gate structure; angles are always optimal given the structure"

This separation makes the transformer's task much simpler and allows it to focus on structural patterns.

The cost function seen by the transformer becomes:

$$\text{cost}(j) = 1 - \max_\theta F_\text{process}(U_\text{target},\, U_j(\theta))$$

### 3.2 L-BFGS

**L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a quasi-Newton gradient-based optimizer. It approximates the inverse Hessian using a rolling window of recent gradient information, enabling fast convergence on smooth landscapes.

**How it works here**: PyTorch's autograd differentiates through `build_circuit_unitary` (the full matrix product) to compute $\nabla_\theta (1 - F)$. L-BFGS uses these gradients plus curvature estimates to take large, well-directed steps.

**Configuration**: `steps` = maximum number of L-BFGS line-search iterations (200 is sufficient for shallow circuits).

| Pros | Cons |
|------|------|
| Very fast convergence on smooth landscapes (often < 50 steps) | Local optimizer — can get stuck in basins of attraction |
| Exploits curvature; far fewer steps than Adam for same precision | Requires full gradient computation (forward + backward pass) |
| Strong Wolfe line search ensures reliable progress | Scales poorly with number of parameters |
| Works well for small parameter counts (< 50) | Sensitive to starting point |

### 3.3 Adam

**Adam** is a first-order gradient-based optimizer with adaptive per-parameter learning rates.

**Configuration**: `steps` = number of gradient steps.

| Pros | Cons |
|------|------|
| More robust than L-BFGS on noisy or discontinuous landscapes | Much slower convergence than L-BFGS for smooth problems |
| Good default for unknown landscapes | Still a local optimizer |
| Simple hyperparameters | Requires careful learning rate tuning |
| Handles sparse gradients well | Needs many more steps (500+) to match L-BFGS quality |

### 3.4 Multiple Random Restarts

All optimizers are susceptible to local minima in the multi-parameter landscape. To mitigate this, **`num_restarts`** independent optimization runs are performed per circuit:

- **Restart 0**: Uses the structured default starting point ($\pi/4$ for all rotation gates).
- **Restarts 1 … N−1**: Start from angles sampled uniformly at random from $(-\pi, \pi]$.

The best result across all restarts is kept.

**Configuration**: `num_restarts: 5` (current).

| Pros | Cons |
|------|------|
| Directly addresses local minima — higher $N$ increases the probability of finding the global optimum | Linear cost multiplier: $N$ times more optimizer calls per circuit |
| Simple and parallelizable | Random restarts do not communicate — no shared information between runs |
| Compatible with all optimizer types | Diminishing returns: most improvement from restart 1→2; smaller gains beyond 5 |

---

## 4. Post-Processing: Gate Simplification

After continuous optimization finds the best angles for a given circuit structure, the **gate simplification pass** (`simplify_gate_sequence` in `continuous_optimizer.py`) applies three algebraic rules to reduce circuit depth without changing the unitary:

### Rule 1: Near-Identity Rotation Removal

If a rotation angle satisfies $|\theta \bmod 2\pi| < \varepsilon$ (default $\varepsilon = 10^{-3}$ rad), the gate is effectively the identity and is removed:

$$R(\theta \approx 0) \approx I \quad \Rightarrow \quad \text{remove}$$

The infidelity introduced is at most $\sim \varepsilon^2/4 \approx 2.5 \times 10^{-7}$, which is recovered by re-optimizing the remaining gates.

### Rule 2: Consecutive Rotation Merging

Two adjacent rotation gates of the same type on the same qubit can be merged into one:

$$R_Z(\theta_1) \cdot R_Z(\theta_2) = R_Z(\theta_1 + \theta_2)$$

The merged angle is renormalized to $(-\pi, \pi]$. If the merged angle is also near-identity, both gates are dropped.

### Rule 3: CNOT Cancellation

Two adjacent identical CNOT gates compose to the identity:

$$\text{CNOT} \cdot \text{CNOT} = I \quad \Rightarrow \quad \text{remove both}$$

### Simplification Pipeline

All three rules are applied in a **loop until stable** (no further changes). After simplification, if any gates were removed, the shorter circuit is **re-optimized** (one warm-started L-BFGS/Adam pass) to recover any residual fidelity loss. The result is accepted only if its fidelity is not worse than the original by more than $10^{-6}$.

| Pros | Cons |
|------|------|
| Rules 2 and 3 are **exact** (zero fidelity loss) | Only handles adjacent gates — non-adjacent redundancies require a full circuit DAG |
| Directly reduces depth and gate count | Near-identity removal introduces a small error (mitigated by re-optimization) |
| No additional hyperparameters | SX cancellation not implemented (SX² = X ≠ I; requires 4 consecutive SX gates) |
| Runs after every circuit optimization, applies to training and evaluation | |

---

## 5. Training Signal: GRPO Loss

The transformer is trained using a **GRPO loss** (Generalized Reward Policy Optimization, `loss.py`), a clipped PPO-style policy gradient loss adapted for circuit generation.

For a batch of circuits sampled from the replay buffer, the loss has two components:

**1. Negative log-likelihood of the best circuit** (always active):

$$\mathcal{L}_\text{NLL} = -\log p_\theta(\text{sequence}_\text{best})$$

This directly increases the probability of generating the best-seen circuit.

**2. Clipped advantage-weighted policy gradient** (active after the first batch step):

$$\mathcal{L}_\text{GRPO} = -\left(\text{clip}\!\left(\frac{p_\theta}{p_{\theta_\text{old}}},\, 1-\epsilon,\, 1+\epsilon\right) \cdot A\right)_\text{mean}$$

where the advantage is:

$$A_i = \frac{\bar{c} - c_i}{\sigma_c + \epsilon}, \quad \bar{c} = \text{mean cost},\; \sigma_c = \text{std cost}$$

Circuits with lower cost (higher fidelity) than average get positive advantage (probability increased); circuits with higher cost get negative advantage (probability decreased). The clip ratio $\epsilon$ (default 0.2) prevents large policy updates that destabilize training.

| Pros | Cons |
|------|------|
| Handles non-differentiable circuit evaluation gracefully | Reward signal is noisy — high variance in cost estimates |
| Advantage normalization stabilizes gradients | Clipping can slow convergence if the policy is far from optimal |
| Directly optimizes the distribution toward low-cost circuits | Requires warm-up period to fill the replay buffer |

---

## 6. Circuit Grammar: Structural Sampling Constraints

### 6.1 Motivation

Without constraints, the autoregressive sampler generates many circuits that are **structurally equivalent** but appear as different token sequences. For example:

$$RZ_i(\theta_1),\, RZ_i(\theta_2) \;\equiv\; RZ_i(\theta_1 + \theta_2)$$

The sampler wastes budget generating, evaluating, and storing duplicates of the same unitary. These duplicates also corrupt the GRPO advantage signal: if the buffer is filled with structurally redundant circuits achieving the same fidelity, the cost variance collapses and the learning signal vanishes.

The `CircuitGrammar` class (`circuit_grammar.py`) eliminates this by masking forbidden gate choices at each autoregressive step, using only **exact algebraic identities and exact commutation relations** — no approximations.

### 6.2 Exact Algebraic Rules

The following identities define which gate sequences are structurally redundant:

**Rule 6 — RZ merge prevention**

$$RZ_i(\theta_1) \cdot RZ_i(\theta_2) = RZ_i(\theta_1 + \theta_2)$$

A new $RZ_i$ is forbidden if an unresolved $RZ_i$ already exists in the circuit — i.e., no gate that fails to commute with $RZ_i$ has appeared since the previous $RZ_i$.

Gates that do **not** commute with $RZ_i$ (and therefore resolve it):
- $SX_i$ (same qubit)
- $CNOT_{c,i}$ for any control $c$ (where qubit $i$ is the target)

**Rule 7 — CNOT cancellation prevention**

$$CX_{c,t}^2 = I$$

A new $CX_{c,t}$ is forbidden when an unresolved $CX_{c,t}$ exists.

Gates that resolve a pending $CX_{c,t}$:
- $RZ_t$ (target rotation)
- $SX_c$, $SX_t$ (SX on control or target)
- Any other CNOT sharing qubit $c$ or $t$

**Rule 8 — SX fourth-power prevention**

$$SX_i^4 = I$$

A fourth $SX_i$ is forbidden when three unresolved $SX_i$ gates exist. The SX count resets whenever a non-commuting gate is added ($RZ_i$ or any CNOT involving qubit $i$).

**Commutation relations used** (exact; no approximations):

| Pair | Commutes? | Source |
|------|-----------|--------|
| Gates on disjoint qubits | Yes | Rule 4 (general) |
| $RZ_c$ and $CX_{c,t}$ | Yes | Rule 5 (control qubit) |
| $RZ_i$ and $RZ_j$ (any qubits) | Yes | Both diagonal |
| $SX_i$ and $SX_i$ | Yes | $SX^2 = X$ (fixed) |
| $SX_i$ and $RZ_i$ | **No** | Non-commuting pair |
| $SX_c$ or $SX_t$ and $CX_{c,t}$ | **No** | Non-commuting pair |
| $RZ_t$ and $CX_{c,t}$ | **No** | Non-commuting pair (target) |

### 6.3 Canonical Ordering

Two circuits that differ only by commuting adjacent gates are unitarily identical. To keep a single canonical representative, the grammar enforces:

> If a candidate gate $G_\text{new}$ commutes with the immediately preceding gate $G_\text{prev}$, and $\text{pool\_index}(G_\text{new}) < \text{pool\_index}(G_\text{prev})$, then $G_\text{new}$ is forbidden.

This ensures that among all equivalent orderings of commuting adjacent gates, only the one with monotonically non-decreasing pool indices is generated.

**Rule 9 — Disjoint commuting gates.** For example, after sampling $SX_{q1}$ (pool index 3), sampling $RZ_{q0}$ (pool index 0) is forbidden: the canonical circuit always places $RZ_{q0}$ first.

**Rule 10 — RZ(control) and CNOT.** Since $RZ_c$ commutes with $CX_{c,t}$ (Rule 5), and $\text{pool\_index}(RZ_c) < \text{pool\_index}(CX_{c,t})$ by construction, the canonical order is always $RZ_c$ before $CX_{c,t}$. After sampling $CX_{c,t}$, sampling $RZ_c$ next is forbidden.

The check is applied only against the **immediately preceding gate**. This correctly eliminates all adjacent-commuting-pair duplicates; non-adjacent equivalent orderings are only partially reduced, which is a known conservative trade-off.

### 6.4 Implementation

**State representation.** The grammar tracks only three small arrays per partial circuit — independent of circuit length:

| Field | Type | Meaning |
|-------|------|---------|
| `rz_active[q]` | `bool` per qubit | True if a mergeable $RZ_q$ is pending |
| `sx_count[q]` | `int` (0–3) per qubit | Number of unresolved $SX_q$ gates |
| `cnot_active[(c,t)]` | `bool` per pair | True if a cancellable $CX_{c,t}$ is pending |
| `last_gate_idx` | `int \| None` | Pool index of the last sampled gate |

**Masking.** At each autoregressive step, `CircuitGrammar.forbidden_mask(state)` returns a boolean list of length `len(pool)`. Forbidden gates are masked in the logit tensor by setting their value to $+\infty$ before sampling:

$$\text{logit}_\text{forbidden} \leftarrow +\infty \;\Rightarrow\; -\beta \cdot \text{logit} = -\infty \;\Rightarrow\; p = 0$$

A safety fallback allows all gates if the mask would forbid the entire pool (cannot occur with a well-formed gate pool, but prevents deadlock).

**State update.** After sampling, `CircuitGrammar.update(state, gate_idx)` updates the three arrays in $O(|\text{CNOT pairs}|)$ time, which is negligible.

| Pros | Cons |
|------|------|
| Eliminates structurally redundant circuits exactly | Only adjacent-gate canonical ordering is enforced |
| Preserves GRPO advantage variance | The transformer's learned distribution is corrected extrinsically; forbidden logits are never trained away |
| No approximations — based on exact algebraic identities | Canonical ordering makes the effective vocabulary non-stationary (context-dependent) |
| Negligible runtime cost per step | |

**Configuration**: set `grammar.enabled: true` in `config.yml` to activate.

---

## 7. Baseline Comparison: Qiskit Compiler

The GQE results are benchmarked against **Qiskit's transpiler** at optimization level 3, using the same basis gate set `{rz, sx, cx}`.

Qiskit uses **KAK decomposition** (Cartan decomposition / Weyl chamber analysis): any 2-qubit unitary can be written as:

$$U = (A_1 \otimes A_2) \cdot \exp\!\left(i(c_1 XX + c_2 YY + c_3 ZZ)\right) \cdot (A_3 \otimes A_4)$$

The three interaction parameters $c_1, c_2, c_3$ are read analytically from $U_\text{target}$'s eigenvalues. The single-qubit parts $A_1, \ldots, A_4$ are found via ZYZ Euler decomposition. This gives:

- **Fidelity = 1.0** (guaranteed exact synthesis)
- **At most 3 CNOT gates** (provably optimal for generic 2-qubit unitaries)
- **Depth ≈ 16** for a generic Haar-random unitary

Qiskit's approach is non-learnable (pure analytical math) and only applies to 2-qubit unitaries or small sub-blocks. GQE is learnable and in principle generalizes to $n$ qubits, which is the primary motivation for the approach.

### Current Results (2-qubit Haar-random target)

| Metric | GQE | Qiskit |
|--------|-----|--------|
| Fidelity | 0.999007 | 1.000000 |
| Depth | 19 | 16 |
| Total gates | 26 | 28 |
| 2-qubit gates | 5 | 3 |

---

## 8. Known Limitations and Open Problems

### 8.1 Fixed-Length Generation

The model always generates exactly `max_gates_count` gates (currently 28). There is no end-of-sequence (EOS) token, so the model cannot learn to produce shorter circuits. Qiskit's circuits are compact by construction; GQE pads with potentially redundant gates that simplification partially removes.

**Proposed fix**: Add an EOS token and penalize circuit length in the cost function with a small weight $\lambda$:

$$\text{cost}(j, \theta) = (1 - F) + \lambda \cdot \frac{|j|}{N_\text{max}}$$

### 8.2 Scalability of Continuous Optimization

All continuous optimizers require building or evaluating the full $2^n \times 2^n$ circuit unitary at each step. This becomes intractable for $n \gtrsim 4$:

| $n$ qubits | Matrix size | Memory (complex128) | Matrix multiply cost |
|------------|-------------|---------------------|----------------------|
| 2 | 4×4 | negligible | negligible |
| 3 | 8×8 | negligible | negligible |
| 4 | 16×16 | ~4 KB | fast |
| 10 | 1024×1024 | ~16 MB | slow |
| 20 | 1M×1M | ~16 TB | impossible |

**Proposed fixes**:
- **Statevector simulation**: Apply gates to state vectors of size $2^n$ instead of building the full unitary. Memory: $O(2^n)$; cost per gate: $O(2^n)$ instead of $O(4^n)$ for embedding.
- **Randomized trace estimation**: Estimate $\text{Tr}(U^\dagger_\text{target} U_\text{circuit})$ using $K \ll 2^n$ random state vectors, reducing cost to $O(K \cdot \text{depth} \cdot 2^n)$.

### 8.3 Local Minima in Multi-Parameter Optimization

Even with 5 random restarts, the optimizers may miss the global optimum when the circuit has many parameters and the fidelity landscape is highly non-convex. For the 2-qubit case, the fidelity gap (0.999 vs. 1.0) likely originates here.

### 8.4 No Depth Awareness During Training

The transformer receives no signal about circuit depth or gate count during training — only fidelity. It therefore has no incentive to prefer a 15-gate circuit over a 28-gate circuit if both achieve the same fidelity.

---

## 9. Configuration Reference

```yaml
# config.yml — key parameters and their roles

model:
  size: "small"          # GPT-2 size: tiny | small | medium | large
  max_gates_count: 28    # Fixed length of every generated circuit

training:
  max_epochs: 100        # Total training epochs
  num_samples: 100       # Circuits generated per rollout
  batch_size: 16         # Gradient-step batch size from replay buffer
  lr: 1.0e-4             # AdamW learning rate for the transformer
  grpo_clip_ratio: 0.2   # PPO clip epsilon for GRPO loss
  early_stop: true       # Stop training immediately once fidelity reaches 1.0

temperature:
  scheduler: "fixed"     # fixed | cosine | variance
  initial_value: 0.5     # Starting inverse temperature β₀
  delta: 0.02            # β increment per epoch (fixed scheduler only)
  min_value: 0.1         # Lower clamp on β
  max_value: 1.5         # Upper clamp on β

buffer:
  max_size: 500          # Replay buffer capacity
  warmup_size: 50        # Circuits collected before training starts
  steps_per_epoch: 4     # Gradient steps taken per epoch

continuous_opt:
  enabled: true          # Whether to run angle optimization per circuit
  optimizer: lbfgs       # lbfgs | adam
  steps: 100             # lbfgs: max L-BFGS iterations | adam: gradient steps
  lr: 0.1                # Learning rate (lbfgs and adam)
  top_k: 0               # 0 = optimize all circuits; N = optimize top-N per rollout
  num_restarts: 5        # Independent random restarts (restart 0 uses π/4 defaults)

grammar:
  enabled: true          # Enforce structural constraints during sampling (see Section 6)
```
