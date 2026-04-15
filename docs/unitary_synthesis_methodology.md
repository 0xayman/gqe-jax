# Unitary Synthesis Methodology in the Current Codebase

This document describes the methodology currently implemented in this repository for quantum unitary synthesis. It is grounded in the active code paths in `main.py`, `gqe.py`, `pipeline.py`, `model.py`, `cost.py`, `continuous_optimizer.py`, `pareto.py`, `pareto_gd_optimizer.py`, `operator_pool.py`, `target.py`, and `config.yml`.

The repository solves unitary synthesis as a hybrid discrete-continuous optimization problem:

- the discrete part is the circuit structure, represented as a sequence of gate tokens sampled by a transformer policy;
- the continuous part is the optimization of rotation angles for parameterized gates;
- the training signal is fidelity-driven, optionally augmented by Pareto-style pressure toward lower depth and fewer CNOT gates.

## 1. Formal Problem Definition

Let:

- `n` be the number of qubits,
- `d = 2^n` be the Hilbert-space dimension,
- `U_target in U(d)` be the target unitary,
- `V` be the discrete gate vocabulary induced by the operator pool,
- `N = max_gates_count` be the fixed sequence length used by the current model.

In the current code, a candidate circuit is represented by a token sequence

`x = (x_1, ..., x_N),  x_t in V`

and, for parameterized gates, a continuous parameter vector

`theta = (theta_1, ..., theta_m)`.

Each token `x_t` maps to a gate matrix `G_t`. For non-parameterized gates, `G_t` is fixed. For parameterized gates, `G_t` depends on the relevant entry of `theta`. The synthesized circuit unitary is

`U(x, theta) = G_N(theta) ... G_2(theta) G_1(theta)`.

The quality measure used by the code is process fidelity:

```math
F(U_target, U(x, theta)) =
\frac{\left|\mathrm{Tr}\left(U_target^\dagger U(x, theta)\right)\right|^2}{d^2}.
```

The baseline unitary-synthesis objective is therefore

```math
\min_{x, theta} C_F(x, theta)
= 1 - F(U_target, U(x, theta)).
```

When Pareto mode is enabled, the training pipeline does not directly optimize only `C_F`. Instead, after warmup it uses a scalarized rollout cost

```math
C_i =
w_F (1 - F_i)
    + \mathbf{1}[F_i \ge \tau]
      \left(
        w_d \hat d_i + w_c \hat c_i
      \right),
```

where:

- `F_i` is the fidelity of circuit `i`,
- `d_i` is its estimated circuit depth,
- `c_i` is its CNOT count,
- `hat d_i` and `hat c_i` are rollout-batch z-scores,
- `tau` is the configured fidelity threshold,
- `w = (w_F, w_d, w_c)` is a Dirichlet-sampled weight vector with a hard lower bound on `w_F`.

So the current implementation should be viewed as solving:

```math
\min_{x, theta} \; 1 - F(U_target, U(x, theta))
```

as the core synthesis objective, while optionally training toward a Pareto front over:

- maximize fidelity,
- minimize depth,
- minimize CNOT count.

## 2. Methodology at a Glance

At a high level, the code follows this pipeline:

1. Load and validate the experiment configuration.
2. Build the operator pool that defines the gate vocabulary.
3. Generate or load the target unitary.
4. Verify target unitarity and compile the target with Qiskit for reference.
5. Initialize a GPT-2 style autoregressive policy over gate tokens.
6. Sample batches of fixed-length gate sequences.
7. Evaluate sampled circuits with either pure discrete fidelity or hybrid discrete-continuous fidelity.
8. Store the sampled sequences, scalar costs, and behavior log-probabilities in a replay buffer.
9. Train the policy with a clipped GRPO objective using replay-buffered data.
10. Maintain an optional Pareto archive of non-dominated circuits.
11. Optionally run a post-training Pareto gradient-descent refinement pass.
12. Decode the best circuit, verify it again, rebuild it as a Qiskit circuit, and compare it against the Qiskit reference compilation.

This is therefore not a pure transpiler, not a pure gradient-based circuit optimizer, and not a pure language model. It is a hybrid search pipeline that combines:

- sequence modeling for structure search,
- matrix-based fidelity evaluation,
- gradient-based angle refinement,
- replay-buffered policy optimization,
- and optional multi-objective Pareto tracking.

## 3. Current Default Configuration

The current `config.yml` describes the following default experiment:

| Component | Current value |
| --- | --- |
| Target qubits | `2` |
| Target type | `file` |
| Target path | `targets/haar_random_q2.npy` |
| Model size | `small` |
| Max generated gates | `28` |
| Training epochs | `200` |
| Rollout samples per epoch | `200` |
| Batch size | `32` |
| Learning rate | `1e-4` |
| Scheduler | `fixed` |
| Initial inverse temperature | `0.5` |
| Scheduler delta | `0.02` |
| Replay buffer size | `500` |
| Replay warmup size | `100` |
| Continuous optimization | enabled |
| Continuous optimizer | `lbfgs` |
| Continuous optimizer steps | `200` |
| Continuous optimizer restarts | `3` |
| `top_k` continuous filtering | `0` meaning optimize all sampled circuits |
| Pareto archive | enabled |
| Pareto post-training GD | enabled |

In other words, the default run today is already the most complex path in the repository: it uses a file-based target, continuous angle optimization during rollout scoring, Pareto scalarization during training, and a post-training Pareto refinement step.

## 4. Code Map

The main responsibilities are distributed as follows:

| File | Responsibility |
| --- | --- |
| `main.py` | Top-level orchestration, Qiskit baseline, final reporting |
| `config.py` | Typed configuration and validation |
| `operator_pool.py` | Gate vocabulary and full embedded matrices |
| `target.py` | Target-unitary generation and loading |
| `model.py` | GPT-2 style Flax transformer |
| `pipeline.py` | Rollout generation, scoring, replay buffering, training steps |
| `cost.py` | Process fidelity and batched JAX cost functions |
| `continuous_optimizer.py` | Angle optimization and algebraic simplification |
| `pareto.py` | Non-dominated archive maintenance |
| `pareto_gd_optimizer.py` | Post-training gradient refinement of Pareto circuits |
| `gqe.py` | Training loop and metric logging |

## 5. Detailed Methodology

### 5.1 Operator Pool and Search Space

The operator pool is the discrete search space used by the policy. In the current implementation, the sampling vocabulary contains:

- one `RZ_qi` token for each qubit `i`,
- one `SX_qi` token for each qubit `i`,
- one `CNOT_qi_qj` token for each ordered control-target pair with `i != j`.

For `n` qubits, the current pool size is

```math
|V| = n + n + n(n - 1) = n^2 + n.
```

Important implementation details:

- Although `continuous_optimizer.py` knows how to parse `RX` and `RY`, the operator pool currently exposes only `RZ`, `SX`, and `CNOT` at generation time.
- Rotation gates are stored in the pool with a placeholder angle of `pi/4`.
- The stored matrices are full dense `2^n x 2^n` unitaries embedded into the whole system, not symbolic gate descriptions.

For the default `n = 2` case:

- `|V| = 6`,
- the model vocabulary size is `7` after adding the BOS token.

### 5.2 Target Unitary Preparation

Target construction is centralized in `target.py`. The current supported target modes are:

- `random_reachable`: build a target by composing a random short circuit from the operator pool, so exact realization is possible in principle;
- `haar_random`: generate a Haar-random unitary using QR decomposition of a complex Gaussian matrix;
- `file`: load a saved unitary from a `.npy` file and validate its dimension.

With the current config, the target path is `targets/haar_random_q2.npy`, so the target is loaded from disk rather than sampled on the fly.

After loading, `main.py` numerically verifies that the matrix is unitary:

```math
U_target U_target^\dagger \approx I_d.
```

This is a sanity check before any training begins.

### 5.3 Reference Compilation with Qiskit

Before running GQE, the target unitary is compiled by Qiskit using:

- basis gates `["rz", "sx", "cx"]`,
- `optimization_level = 3`.

This Qiskit result is not used as a target policy, teacher, or reward source. It is used strictly as a reference baseline for:

- depth,
- total gate count,
- two-qubit gate count,
- and final side-by-side reporting.

This gives the repository a classical baseline under the same native gate set as the learned method.

### 5.4 Sequence Representation

The current model uses fixed-length sequences. There is:

- one beginning-of-sequence token `BOS` with id `0`,
- a gate-token offset of `+1`,
- no end-of-sequence token,
- no variable-length decoding in the main policy.

So each candidate circuit is always represented as a sequence of length

```math
N + 1,
```

where `N = max_gates_count` and the extra element is the BOS token.

For the current config:

- `N = 28`,
- every sampled candidate has exactly `29` tokens including BOS,
- all candidates contain exactly `28` generated gate tokens before any later simplification.

This fixed-length design is a central property of the current methodology. It means circuit shortening is handled indirectly rather than by explicit early termination.

### 5.5 Policy Model

The structure-search model is implemented in `model.py` as a GPT-2 style causal transformer in Flax. It uses:

- learned token embeddings,
- learned positional embeddings,
- causal self-attention,
- pre-layer-normalized transformer blocks,
- an MLP block with GELU,
- and tied output projection through the token embedding matrix.

The supported size presets are:

| Size | Layers | Heads | Embedding size |
| --- | ---: | ---: | ---: |
| `tiny` | 2 | 2 | 128 |
| `small` | 6 | 6 | 384 |
| `medium` | 12 | 12 | 768 |
| `large` | 24 | 16 | 1024 |

The default run uses `small`, so the current default policy has:

- 6 transformer blocks,
- 6 attention heads,
- embedding dimension 384.

One subtle but important detail is that the code treats the model outputs as energies rather than conventional language-model logits. Sampling and log-probabilities use `softmax(-beta * output)` instead of `softmax(output)`.

### 5.6 Rollout Generation

Rollouts are generated autoregressively in `Pipeline.generate()` with a JIT-compiled `jax.lax.scan`.

At each generation step:

1. the transformer receives the current token prefix,
2. the BOS token is masked out so it cannot be sampled again,
3. the next token is sampled with `jax.random.categorical` from `-beta * logits`,
4. the sampled token is appended to the sequence.

The current inverse temperature `beta` is provided by the scheduler.

Formally, if `a_t` is the token chosen at step `t`, the policy used by the code is:

```math
\pi_\beta(a_t = v \mid a_{<t})
\propto
\exp\left(-\beta E_t(v)\right),
```

where `E_t(v)` is the model output for token `v` at position `t`.

The sequence log-probability used later by GRPO is:

```math
\log \pi_\beta(x)
=
\sum_{t=1}^{N}
\log \pi_\beta(x_t \mid x_{<t}).
```

### 5.7 Fidelity Evaluation

The fidelity computation is defined in `cost.py`. Given a composed circuit unitary `U_circuit`, the code evaluates

```math
F(U_target, U_circuit)
=
\frac{\left|\mathrm{Tr}(U_target^\dagger U_circuit)\right|^2}{d^2},
```

and then converts it into a cost:

```math
C_F = 1 - F.
```

The JAX implementation uses:

- `jax.lax.scan` to compose gate matrices across gate positions,
- `jax.vmap` to score a whole batch of candidate circuits,
- `jax.jit` to compile the whole evaluation path.

### 5.8 Discrete and Hybrid Scoring Paths

`Pipeline.computeCost()` supports two evaluation modes.

#### Pure discrete mode

If `continuous_opt.enabled = false`, the code:

1. decodes token ids into operator-pool indices,
2. composes the stored pool matrices,
3. computes process fidelity directly,
4. returns `1 - fidelity`.

In this mode, parameterized rotations are evaluated at the pool placeholder angle `pi/4`.

#### Hybrid discrete-continuous mode

If `continuous_opt.enabled = true`, the code evaluates the circuit structure and then refines its continuous parameters before assigning the final fidelity.

Two sub-modes exist:

- if `top_k = 0`, every sampled circuit is continuously optimized;
- if `top_k > 0`, the pipeline first computes a cheap discrete score for all circuits, then continuously optimizes only the top `k` structures.

With the current config, `top_k = 0`, so every sampled circuit in every rollout is angle-optimized.

### 5.9 Continuous Angle Optimization

The continuous optimizer is implemented in `continuous_optimizer.py`. It fixes a sampled gate structure and optimizes only the parameter vector.

For a given structure `x`, it solves:

```math
\min_\theta \; 1 - F(U_target, U(x, \theta)).
```

The optimizer works by:

1. parsing gate names into `GateSpec` objects,
2. encoding each circuit into fixed-shape arrays of gate type, qubit index, CNOT pair index, and padded angles,
3. building the circuit unitary differentiably in JAX,
4. optimizing the angles with either L-BFGS or Adam,
5. optionally simplifying the circuit algebraically,
6. re-optimizing once if simplification produced a shorter parametric circuit.

Supported optimizers:

- `lbfgs`,
- `adam`.

The default run uses L-BFGS with:

- 200 steps,
- 3 restarts.

The restart strategy is:

- one start from the default initialization,
- remaining starts sampled uniformly from `[-pi, pi]` on the active parameter entries.

The simplification rules currently implemented are:

- drop near-zero parameterized rotations,
- merge consecutive same-axis rotations on the same qubit,
- cancel adjacent identical CNOT gates.

Another important implementation detail is precision. The global JAX runtime enables x64, but `Factory.create_continuous_optimizer()` constructs the training-time optimizer with `fast_runtime=True`, so the rollout-time continuous optimization uses float32/complex64 internally for speed.

### 5.10 Structure Metrics

The code tracks three structure statistics per sampled circuit:

- estimated depth,
- total gate count,
- CNOT count.

Depth is computed by a greedy per-qubit accumulation procedure. For each token:

- a single-qubit gate increments the depth counter of its qubit,
- a two-qubit gate sets both qubit depths to `max(depth_q0, depth_q1) + 1`.

This is an inexpensive structural proxy, not a full scheduling or transpilation pass.

`total_gates` is recorded mostly for reporting. The active multi-objective logic uses:

- fidelity,
- depth,
- CNOT count.

### 5.11 Pareto Scalarization and Archive

If Pareto mode is enabled, the pipeline maintains a non-dominated archive over:

- maximize fidelity,
- minimize depth,
- minimize CNOT count.

A circuit `a` dominates circuit `b` if:

- `a.fidelity >= b.fidelity`,
- `a.depth <= b.depth`,
- `a.cnot_count <= b.cnot_count`,
- and at least one of those inequalities is strict.

During training:

- warmup uses fidelity-only cost;
- after warmup, the pipeline samples a weight vector from a Dirichlet distribution;
- it enforces a minimum fidelity weight `w_F_min`;
- it applies complexity penalties only when `fidelity >= fidelity_threshold`.

The scalarized cost used for replay-buffer training is therefore:

```math
C_i =
w_F (1 - F_i)
    + \mathbf{1}[F_i \ge \tau]
      \left(
        w_d \frac{d_i - \mu_d}{\sigma_d + \varepsilon}
        +
        w_c \frac{c_i - \mu_c}{\sigma_c + \varepsilon}
      \right).
```

The archive:

- rejects points below a fidelity floor,
- removes dominated points,
- prunes by crowding distance if it grows too large,
- reports a 2-D hypervolume over fidelity and CNOT count.

With the current config:

- the fidelity floor starts at `0.5`,
- it is raised to `0.9` at epoch 100.

### 5.12 Replay Buffer and Training Data Reuse

The policy is not trained purely on the latest rollout. Instead, sampled sequences are inserted into a FIFO replay buffer.

Each buffer item stores:

- the token sequence,
- the scalar cost used for learning,
- the sequence log-probability under the behavior policy that produced it.

Warmup is implemented as:

```text
while len(buffer) < warmup_size:
    collect_rollout()
```

Given the current config:

- `num_samples = 200`,
- `warmup_size = 100`,

so one rollout is already enough to fill the warmup requirement.

`BufferDataset` then exposes replay-buffer data for multiple passes per epoch via `steps_per_epoch`.

### 5.13 GRPO Training Objective

The actual policy update is implemented in `pipeline.py` as a clipped GRPO-style surrogate.

For a mini-batch of replayed trajectories:

- let `C_i` be the stored scalar cost,
- let `A_i` be the normalized advantage,
- let `L_i` be the current sequence log-probability,
- let `L_i_old` be the stored behavior sequence log-probability,
- let `r_i = exp(L_i - L_i_old)`.

The code computes:

```math
A_i =
\frac{\mathrm{mean}(C) - C_i}{\mathrm{std}(C) + \varepsilon},
```

so lower-than-average cost yields positive advantage.

The clipped surrogate loss is:

```math
\mathcal{L}_{GRPO}
=
- \frac{1}{B}
  \sum_{i=1}^{B}
  \min\left(
    r_i A_i,\;
    \mathrm{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i
  \right),
```

where `epsilon = grpo_clip_ratio`.

The optimizer used on the policy parameters is:

- global-norm gradient clipping,
- followed by `optax.adamw`.

### 5.14 Scheduler Behavior

The scheduler provides the inverse temperature `beta` used during rollout generation and log-probability evaluation.

The code supports:

- `fixed`,
- `cosine`.

The current config selects `fixed`, but in the current implementation `DefaultScheduler` still updates:

```text
current_temperature <- current_temperature + delta
```

after each rollout, followed by clamping to `[min_value, max_value]`.

So in practice, the default run uses a monotone annealing schedule rather than a truly fixed inverse temperature.

### 5.15 Best-Circuit Tracking and Final Decoding

During training, the code tracks the best circuit from the most recent rollout batch using fidelity-only costs. After training:

1. the best token sequence is decoded back into gate names;
2. if continuous optimization is enabled, the code re-runs the continuous optimizer on that structure for verification;
3. the optimized structure is converted into a Qiskit circuit;
4. the resulting circuit is compared against the Qiskit transpilation baseline.

This final comparison reports:

- fidelity,
- depth,
- total gates,
- two-qubit gates.

### 5.16 Post-Training Pareto Gradient Descent

If `pareto_gd.enabled = true`, the code performs a post-training pass over all archived Pareto circuits.

For each archived circuit below `1 - fidelity_eps`:

1. convert the stored token sequence back to gate names,
2. run continuous optimization again,
3. update the archived fidelity if the optimizer improved it,
4. rebuild the archive from the updated points.

This is intended to improve the final Pareto front without changing the learned policy itself.

## 6. End-to-End Algorithm Summary

The current implementation can be summarized as:

```text
Input:
    config, target specification, operator pool

Build:
    U_target
    Qiskit reference circuit
    transformer policy
    replay buffer
    optional continuous optimizer
    optional Pareto archive

Warmup:
    sample rollouts
    compute fidelity-based costs
    fill replay buffer

Training epoch:
    generate num_samples fixed-length circuits
    compute old sequence log-probabilities
    evaluate fidelity, depth, and CNOT count
    if Pareto enabled and not in warmup:
        sample Pareto weights
        scalarize fidelity/depth/CNOT cost
    push sequences and costs to replay buffer
    train with clipped GRPO on replay-buffer batches
    update best-so-far statistics
    update scheduler
    update Pareto archive

After training:
    optionally run ParetoGD on archive members
    decode best structure
    verify best structure with continuous optimization
    build final Qiskit circuit
    compare against Qiskit baseline
```

## 7. Current Issues, Bugs, and Bottlenecks

The following are the main issues visible in the current implementation.

### 7.1 Fixed-Length Generation Is a Hard Structural Limitation

The policy has no EOS token and no variable-length decoding. Every sample contains exactly `max_gates_count` generated gates before any post hoc simplification.

Consequences:

- the model cannot directly learn to stop early;
- circuit-shortening pressure is only indirect;
- the structural search space is larger than necessary;
- reported complexity is partly shaped by simplification rather than native generation.

### 7.2 `ParetoGDOptimizer` Does Not Write Back Simplified Structures

The post-training Pareto GD pass calls `optimize_circuit_with_params()`, but it only keeps the improved fidelity. It discards the returned optimized gate structure and parameters.

Consequences:

- `token_sequence`, `depth`, `total_gates`, and `cnot_count` in the archive remain unchanged;
- if simplification found a shorter equivalent circuit, the archive does not reflect it;
- the rebuilt archive can still be structurally stale after Pareto GD.

This is a concrete correctness gap between what Pareto GD computes and what the archive stores.

### 7.3 Pareto Weight Sampling Is Not Fully Reproducible

`Pipeline._sample_weights()` uses `np.random.dirichlet(...)` from the global NumPy RNG instead of a seeded generator tied to `training.seed`.

Consequences:

- Pareto runs are not fully reproducible even when `training.seed` is fixed;
- repeated runs can differ for reasons not captured by the configured seed.

### 7.4 The "Fixed" Scheduler Is Not Actually Fixed

The scheduler named `fixed` still increments its value by `delta` after every rollout.

Consequences:

- experiment semantics are easy to misread;
- users may believe they are running with constant inverse temperature when they are not.

### 7.5 Dropout Exists in the Model but Is Disabled in Practice

The transformer defines dropout layers, but the pipeline calls the model with `deterministic=True` during both generation and optimization.

Consequences:

- no dropout regularization is active;
- the documented model capacity and the effective training behavior are slightly different.

### 7.6 Training-Time and Evaluation-Time Precision Differ

The global runtime enables x64, but the training-time continuous optimizer is created with `fast_runtime=True`, which uses float32/complex64 internally. The final verifier in `main.py` is not forced into that same low-precision path.

Consequences:

- rollout-time scoring and final verification do not use exactly the same numerics;
- near-perfect fidelities may be sensitive to this mismatch.

### 7.7 Evaluation Cost Scales Poorly

The current methodology uses full dense `2^n x 2^n` matrices for:

- all operator-pool entries,
- all circuit compositions,
- all fidelity evaluations,
- and all continuous optimization passes.

With `top_k = 0`, every sampled circuit in every rollout is angle-optimized.

Consequences:

- runtime and memory grow exponentially with qubit count;
- the continuous optimizer becomes a dominant bottleneck;
- the current design is practical only for relatively small systems.

### 7.8 Depth Is a Proxy, Not a Hardware-Aware Metric

Depth is estimated by a greedy qubit-depth counter on the raw token sequence. It is not computed from:

- a simplified circuit,
- a scheduled circuit,
- or a transpiled hardware circuit.

Consequences:

- the optimization target for depth is approximate;
- the depth used during Pareto ranking may differ from the true executable depth.

### 7.9 Logging and Naming Become Ambiguous in Pareto Mode

Even when replay training uses scalarized multi-objective costs, `best_cost` and `epoch_best_cost` are still tracked using fidelity-only costs from `_last_rollout_costs`.

Consequences:

- the term "best cost" no longer matches the actual replay objective in Pareto mode;
- some logs are semantically closer to "best fidelity loss" than "best training objective".

### 7.10 The Search Vocabulary Is Narrower Than the Optimizer's Capabilities

The continuous optimizer understands `RX`, `RY`, and `RZ`, but the operator pool currently exposes only `RZ` as a parameterized single-qubit gate type.

Consequences:

- the structure generator cannot directly propose richer one-qubit rotations;
- some expressivity must be synthesized indirectly through longer sequences of `RZ`, `SX`, and `CNOT`.

## 8. Suggestions for Improvement

The most impactful next steps are the following.

### 8.1 Add Variable-Length Generation

Introduce:

- an EOS token,
- a PAD or NOOP token,
- attention masking for padded positions,
- and log-probability accumulation only up to EOS.

This would let the policy learn short circuits directly instead of relying on indirect scalarization and post hoc simplification.

### 8.2 Fix Pareto GD So It Updates Structure as Well as Fidelity

When `ParetoGDOptimizer` receives an improved or simplified circuit, it should write back:

- the new token sequence,
- updated depth,
- updated total gate count,
- updated CNOT count.

Without this, the Pareto archive is only partially refined.

### 8.3 Make Pareto Runs Reproducible

Replace global `np.random.dirichlet(...)` calls with a seeded generator owned by the pipeline.

This is a small change with high experimental value.

### 8.4 Separate Scheduler Names from Scheduler Behavior

Either:

- make `fixed` truly fixed,
- or rename it to something like `linear`.

This will reduce configuration ambiguity.

### 8.5 Unify or Explicitly Manage Precision Policy

Choose one of:

- full x64 throughout,
- explicit mixed precision with logging,
- or fast rollout scoring plus x64 rechecks on shortlisted candidates.

The current mixed behavior is pragmatic, but it should be made explicit in the methodology and logs.

### 8.6 Reduce the Continuous-Optimization Bottleneck

Possible improvements:

- use `top_k > 0` to avoid optimizing every sampled circuit;
- add a cheap prefilter before L-BFGS;
- cache partial products or prefix unitaries;
- reuse optimized angles for repeated structures;
- or move to a two-stage search where only promising candidates are continuously refined.

### 8.7 Improve Complexity Metrics

Instead of ranking circuits by raw token-sequence depth, compute structure metrics from:

- simplified circuits,
- or rebuilt Qiskit circuits under the same basis.

This would make Pareto optimization closer to the actual deployment metric.

### 8.8 Expand the Structural Vocabulary

Consider:

- enabling `RX` and `RY` tokens,
- adding learned macro-gates,
- or using short parameterized gate templates as actions.

This could reduce the burden on sequence length and continuous post-processing.

### 8.9 Reconsider Replay-Buffer Strategy

The current method is replay-buffered and off-policy with behavior log-probabilities, but it does not explicitly prioritize:

- recency,
- novelty,
- or Pareto-critical samples.

Possible improvements:

- prioritized replay,
- separate recent-policy and archive replay pools,
- or a more on-policy training schedule.

### 8.10 Add Regression Tests Around the Current Weak Points

The codebase would benefit from targeted tests for:

- Pareto GD structure updates,
- reproducible Pareto weight sampling,
- fixed-vs-annealed scheduler behavior,
- consistency between simplified and archived structure metrics.

## 9. Closing Assessment

The current codebase implements a coherent and technically interesting hybrid methodology for unitary synthesis:

- a transformer proposes circuit structures,
- a continuous optimizer repairs the continuous degrees of freedom,
- GRPO updates the policy from replayed rollouts,
- and Pareto machinery encourages better fidelity-complexity trade-offs.

Its strongest features are:

- clear modular separation between target generation, structure search, and angle refinement;
- efficient JAX compilation of rollout and fidelity code paths;
- an explicit bridge to Qiskit for baseline comparison;
- and an emerging multi-objective training framework rather than fidelity-only search.

Its main current limitations are equally clear:

- fixed-length generation,
- expensive dense-matrix scaling,
- partial inconsistency between Pareto refinement and stored structure,
- and several experiment-control details that should be tightened for reproducibility and clarity.

That makes the project a strong small-scale research platform for unitary synthesis, with a clear path toward a more rigorous and scalable next version.
