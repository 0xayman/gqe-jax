# Multi-Objective RL for Short, High-Fidelity Quantum Circuits

## 1. Problem Statement

The current GQE pipeline trains a GPT-2 policy with a single objective:
minimize `cost = 1 − process_fidelity(U_target, U_circuit)`.

Every generated circuit has **exactly** `max_gates_count = 28` gates regardless of its
complexity. The result is that the model never learns to prefer shorter circuits, even when
a short circuit would achieve the same or better fidelity.

The full goal is to find circuits that simultaneously:
1. **Maximize** process fidelity (≥ some threshold, e.g., 0.999).
2. **Minimize** circuit depth.
3. **Minimize** CNOT count (closely related to depth since CNOTs dominate the critical path).

This is a true multi-objective optimization problem. Fidelity and circuit complexity are
in tension: longer circuits have more expressibility but hardware error rates scale with
depth, and CNOT gates carry disproportionate hardware cost (~10× the error of single-qubit
gates on current devices).

---

## 2. Background and Literature

### 2.1 Current Approach: GQE with GRPO

The current pipeline is a policy-gradient loop modelled after
*Generative Quantum Eigensolver* (Nakayama & Ohnishi, 2023) and updated with a
**GRPO** (Group Relative Policy Optimization) training signal borrowed from DeepSeek-R1.

- The **policy** is a GPT-2 causal transformer that autoregressively samples discrete gate
  tokens.
- The **reward signal** is `−cost = process_fidelity − 1` (always in [−1, 0]).
- The **advantage** is the z-score of `cost` within each rollout batch.
- The **training update** is a clipped PPO-style ratio update over the full sequence
  log-probability.
- A separate **continuous optimizer** (L-BFGS) refines the rotation angles of the discrete
  structure after sampling.

Key limitation: the model generates **fixed-length** sequences (no EOS, no length signal).

### 2.2 Related Work

#### Quantum Circuit Synthesis and Compilation
- **Kak decomposition** (Shende et al., 2004): optimal decomposition of arbitrary 2-qubit
  gates into at most 3 CNOTs. Provides a tight lower bound for 2-qubit circuits.
- **Solovay-Kitaev theorem**: any single-qubit gate can be approximated to ε using
  O(log^c(1/ε)) gates from a universal set; gives depth lower bounds.
- **QFAST / QSearch** (Younis et al., 2020-2021): numerical search for short circuits using
  block-topology templates.
- **BQSKit** (Lawrence Berkeley): modular compiler that combines numerical synthesis with
  structure search; achieves state-of-art CNOT counts for small unitaries.
- **Qiskit transpiler at level 3**: reference baseline in this codebase. For 2 qubits it
  uses the KAK decomposition and achieves optimal CNOT counts.

#### Reinforcement Learning for Circuit Design
- **Quantum Architecture Search (QAS)**: RL-based search for NISQ ansatz circuits
  (Du et al. 2020, Kuo et al. 2021, Zhang et al. 2022). These frame circuit design as a
  sequential action MDP, same structure as the current pipeline.
- **RLSP / GQE-style**: language-model sampling from a gate vocabulary, then REINFORCE or
  PPO. The key differentiator of the current code is GRPO + continuous angle optimization.
- **MCTS for circuit synthesis** (Fosel et al., 2021 / AlphaZero-inspired): tree search
  combined with neural value functions. Better sample efficiency than pure policy gradient
  when the objective landscape is sparse.

#### Multi-Objective RL (MORL)
- **Linear scalarization**: combine objectives into a single scalar reward with fixed
  weights. Simple, differentiable, but can only recover convex parts of the Pareto front.
- **MORL with Pareto archive** (Roijers et al., 2013; Xu et al., 2020): maintain a
  non-dominated set of solutions. Queries sample weight vectors and run policy gradient
  toward each. Recovers the full Pareto front.
- **Constrained policy gradient** (CPO, Achiam et al., 2017): treat one objective as a
  constraint and optimize the other. Lagrangian relaxation converts the constrained problem
  into an unconstrained one with adaptive penalty coefficient.
- **Reward shaping for length control**: adding a length penalty to the reward is
  equivalent to a discounted MDP where the discount is applied per-gate (Ng et al., 1999).

#### Variable-Length Generation in Discrete RL
- **EOS token**: add a stop-action to the vocabulary; the agent can terminate early. The
  reward for partial sequences must be defined consistently (only assigned at EOS).
- **Length-penalized reward**: reward = fidelity − λ × length. As λ increases, the model
  learns to stop earlier.
- **Curriculum learning** (Bengio et al., 2009): start with small or no length penalty,
  increase it as training progresses. Prevents premature collapse to trivial short circuits
  with zero fidelity.

---

## 3. Proposed Methodology

The methodology is organized into four phases. Each phase builds on the previous one. Only
Phase 1 is strictly necessary to get length-aware learning; the later phases improve
Pareto-coverage and practical circuit quality.

### Phase 0: Establish Baselines (No Code Change Required)

Before any modification, run the current pipeline and record:
- Best fidelity achieved.
- Best raw depth and CNOT count from the comparison table.
- Qiskit reference circuit: depth, total gates, CNOT count.

For 2-qubit Haar-random unitaries, Qiskit at level-3 produces 3 CNOTs and depth ≈ 5.
This is the **theoretical minimum** for a generic 2-qubit unitary (Kak bound). Any GQE
circuit that achieves fidelity ≥ 0.999 with ≤ 3 CNOTs is a success.

---

### Phase 1: Variable-Length Generation via EOS Token

**Goal**: allow the model to generate circuits shorter than `max_gates_count`.

#### 1.1 Add EOS to the Vocabulary
- Token id 1 → EOS (shifting all gate tokens by +2; BOS remains 0).
- During `generate()`, stop appending tokens when EOS is sampled, and pad remaining
  positions with a new NOOP/PAD token.
- Track the effective sequence length per circuit (number of gate tokens before EOS).

#### 1.2 Define Reward for Variable-Length Sequences
At EOS (or at `max_gates_count` if no EOS was sampled), assign:

```
reward = fidelity(circuit_up_to_EOS) - λ_len × (effective_length / max_gates_count)
```

where `λ_len` is a small positive constant (start at 0.0, increase over training).

This is equivalent to the circuit having an additional step-level cost:
each gate token incurs a penalty `λ_len / max_gates_count`, so the model prefers to
stop as soon as fidelity stops improving.

#### 1.3 Attention Mask and Padding
The transformer already uses an attention mask. Pad positions after EOS must be:
- masked out in attention
- excluded from log-probability computation (sum only over non-PAD positions)

This is already structurally supported by the existing `attention_mask` pathway in
`pipeline.py`.

#### 1.4 Implementation Points in the Codebase
| File | Change |
|---|---|
| `pipeline.py` | Add EOS token id; modify `generate()` to stop at EOS; modify log-prob sum to exclude PAD |
| `cost.py` | `compilation_cost_batch_jax` must receive variable-length gate sequences; already works if you truncate before computing |
| `config.py` | Add `lambda_len` scalar to `TrainingConfig` |
| `config.yml` | Add `lambda_len: 0.0` initially |

---

### Phase 2: Multi-Objective Reward Shaping

**Goal**: explicitly penalize CNOT count and circuit depth in the training signal.

The current cost is:
```
cost = 1 − fidelity
```

Replace it with a scalarized multi-objective cost:

```
cost = (1 − fidelity)
     + α × (cnot_count / cnot_budget)
     + β × (depth / depth_budget)
```

where `cnot_budget` and `depth_budget` are reference values (e.g., the Qiskit baseline
counts, or fixed constants such as 3 and 5 for 2-qubit circuits).

#### 2.1 Choice of Weights

| Stage | α (CNOT) | β (depth) | Rationale |
|---|---|---|---|
| Warm-up epochs | 0.00 | 0.00 | Learn to achieve high fidelity first |
| Mid training | 0.05 | 0.02 | Light pressure to reduce gates |
| Late training | 0.15 | 0.05 | Strong pressure; fidelity should already be high |

The weight annealing can be driven by a simple schedule: ramp α and β linearly from
zero after a fixed number of epochs (e.g., epoch 50 out of 200).

**Why this ordering matters**: if you start with large α and β, the model collapses to
trivially short circuits with zero fidelity (all-NOOP behavior). The curriculum from
zero weights → increasing weights avoids this.

#### 2.2 Threshold Clipping (Recommended Alternative)

A safer alternative to additive weighting is a **threshold-gated** approach:

```python
if fidelity >= threshold:
    cost = (1 − fidelity) + α * cnot_term + β * depth_term
else:
    cost = 1 − fidelity   # ignore complexity until fidelity is good enough
```

Set `threshold = 0.90` initially. Circuits below 90% fidelity are penalized only on
fidelity; circuits above 90% fidelity additionally face the complexity penalty.
This prevents the model from gaming complexity at the expense of fidelity.

#### 2.3 Implementation Points
| File | Change |
|---|---|
| `cost.py` | Add `compilation_cost_multi_objective(fidelity, cnot_count, depth, α, β, budgets)` |
| `pipeline.py` | Pass CNOT counts and depth from `_sequence_structure_metrics` into cost computation |
| `config.py` | Add `alpha_cnot`, `beta_depth`, `fidelity_threshold`, `cnot_budget`, `depth_budget` to config |

The CNOT count per sample is already tracked in `pipeline._last_rollout_cnot_counts`.
The depth per sample needs `_sequence_structure_metrics` called per circuit, which is
currently only called for the epoch-best circuit. Add a batch version.

---

### Phase 3: Pareto Archive and Diverse Replay

**Goal**: instead of tracking only the single best circuit by cost, maintain a
**Pareto archive** of non-dominated circuits across (fidelity, depth, cnot_count).

#### 3.1 What Is a Pareto Archive?

A circuit A **dominates** circuit B if:
- A.fidelity ≥ B.fidelity AND A.depth ≤ B.depth AND A.cnot_count ≤ B.cnot_count
- AND at least one inequality is strict.

The **Pareto front** is the set of circuits that are not dominated by any other.
Tracking this front gives you the full trade-off surface between fidelity and
complexity, not just a single weighted-sum optimum.

#### 3.2 Using the Archive During Training

1. After each rollout, update the archive with new circuits.
2. Sample batches for GRPO training from the **archive** (in addition to the
   replay buffer), giving the policy a diverse set of "good" examples to learn from.
3. For evaluation, report the Pareto front rather than a single best fidelity.

#### 3.3 Archive-Weighted Replay

Each archived circuit has a **quality score** used for prioritized replay:

```
quality(c) = fidelity(c) / (1 + cnot_count(c) / cnot_budget)
```

Higher-quality circuits are sampled more often from the replay buffer, which biases
the GRPO advantage toward short, high-fidelity circuits.

---

### Phase 4: Post-Training Compression and Verification

Even after training, there are deterministic post-processing steps that can reduce
circuit length without touching fidelity.

#### 4.1 Rule-Based Simplification (Already in the Codebase)

The existing `ContinuousOptimizer` already has simplification rules (when
`simplify=True`):
- Cancel adjacent CNOT pairs: `CNOT · CNOT = I`
- Merge adjacent same-axis rotations: `RZ(θ₁) · RZ(θ₂) = RZ(θ₁+θ₂)`
- Drop near-zero rotations: `RZ(ε) ≈ I`

Currently `simplify=False` during rollout scoring. Enable it selectively for the
best circuits found during training (not during rollout, as it's expensive).

#### 4.2 KAK Decomposition for Final Output

For 2-qubit circuits, apply Qiskit's `TwoQubitBasisDecomposer` (which implements the
KAK algorithm) to any 2-qubit sub-block. This gives the globally optimal CNOT count
for that block. Qiskit already does this at `optimization_level=3`; you can apply it
to GQE's output circuit as a final step.

For n > 2 qubits, use `UniformSynthesis` or BQSKit's synthesis pass.

#### 4.3 Redundancy Detection via Commutation

Two gates commute if they act on disjoint qubits. Reordering commuting gates can
expose cancellation opportunities missed by left-to-right greedy simplification.
BQSKit's `CommutationSlicePass` implements this efficiently.

---

## 4. Recommended Implementation Order

The following is a practical step-by-step implementation sequence for this codebase:

### Step 1: Add EOS Token and Variable-Length Generation
Estimated complexity: **medium**. Core change is to `pipeline.py` and the
`generate()` / `computeCost()` methods. GRPO log-prob sum must be updated to
exclude PAD positions.

**Definition of done**: the model can sample circuits shorter than `max_gates_count`.
Verify with `λ_len = 0.1` that mean circuit length decreases over training.

### Step 2: Compute Per-Sample Depth (Batch Version)
Estimated complexity: **low**. `_sequence_structure_metrics` already works per-sample;
wrap it in a loop over the rollout batch. Required for Step 3.

### Step 3: Add Multi-Objective Cost with Curriculum Scheduling
Estimated complexity: **low**. Add `α`, `β`, `threshold` to config and implement the
threshold-gated cost formula in `cost.py`. Add schedule logic in `gqe._run_training()`:
ramp `α` and `β` from 0 after epoch 50.

**Definition of done**: logged `cnot_count_best` decreases over training relative to
a no-penalty run, while fidelity stays above threshold.

### Step 4: Pareto Archive
Estimated complexity: **medium**. Implement a `ParetoArchive` class that:
- stores (fidelity, depth, cnot_count, token_sequence) tuples
- updates the non-dominated set after each rollout
- provides a quality-weighted sampler for replay

Integrate with `_run_training` to log Pareto front size and the best-quality circuit
per epoch.

### Step 5: Post-Training Simplification Pass
Estimated complexity: **low** (already mostly in the codebase).
Enable `simplify=True` in the post-training verification path.
Optionally add BQSKit/Qiskit transpiler pass on the final GQE circuit.

---

## 5. Key Hyperparameters and Suggested Starting Values

| Parameter | Symbol | Suggested Start | Notes |
|---|---|---|---|
| Length penalty | `λ_len` | 0.0 → 0.05 | Ramp up after epoch 50 |
| CNOT weight | `α` | 0.0 → 0.10 | Ramp up after epoch 50 |
| Depth weight | `β` | 0.0 → 0.03 | Ramp up after epoch 50 |
| Fidelity threshold | `threshold` | 0.90 | Below this, no complexity penalty |
| CNOT budget | `cnot_budget` | 3 (2-qubit) | Qiskit KAK lower bound |
| Depth budget | `depth_budget` | 5 (2-qubit) | Qiskit reference depth |
| Max gates count | `max_gates_count` | 28 → 20 | Can reduce as training improves |

---

## 6. Evaluation Protocol

### 6.1 Per-Run Metrics to Log (All Already in WandB Logger)
- `cnot_count_best` and `depth_best`: track convergence on complexity
- `raw_fidelity_best`: track fidelity does not degrade under complexity pressure
- `pareto_front_size`: new metric, shows coverage of trade-off surface

### 6.2 Benchmark Against Qiskit Baseline
The final GQE circuit should be compared against Qiskit's `optimization_level=3`
output on the same target. For 2-qubit Haar-random unitaries:
- **Target**: CNOT count ≤ 3 (Kak bound), fidelity ≥ 0.999
- **Stretch**: depth ≤ Qiskit depth, fidelity = 1.000

### 6.3 Multi-Target Evaluation
Run the full training pipeline on a **suite** of 10–50 different target unitaries
(Haar-random, structured, etc.) and report the **distribution** of fidelity and CNOT
count, not just a single run. Single-target results are highly sensitive to random
seed and initial conditions.

---

## 7. Anticipated Failure Modes and Mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Premature short-circuit collapse | Fidelity drops to zero early in training | Keep `λ_len = 0` until fidelity > threshold |
| Reward hacking: model learns trivial identity | Fidelity ≈ 0 but very short circuits | Add fidelity floor: `cost = max(1-fidelity, 0.5)` |
| Advantage collapse: all costs identical | GRPO gradient vanishes | Monitor `std(costs)` per rollout; increase `num_samples` |
| CNOT oscillation: no convergence | CNOT count noisy, no downtrend | Increase `cnot_budget` smoothly, don't hard-penalize |
| Pareto front degeneracy | All circuits at same (fidelity, cnot) | Increase `num_samples` and replay buffer diversity |

---

## 8. Long-Term Extensions

These are not required for the core goal but represent natural next steps:

1. **Conditioned generation**: embed the target unitary (or its spectral features) into
   the transformer context. The model can then learn across targets, not just one fixed
   target per run. This is the "amortized circuit synthesis" research direction.

2. **MCTS guidance**: replace pure policy-gradient with Monte Carlo Tree Search guided
   by the trained policy as a prior. MCTS is more sample-efficient in sparse-reward
   settings and can find shorter circuits with fewer rollouts.

3. **Noise-aware compilation**: replace process fidelity with a hardware-noise-aware
   fidelity (e.g., gate-error-weighted fidelity). CNOT penalty becomes implicit in the
   noise model rather than a separate term.

4. **Hierarchical decomposition**: for n > 3 qubits, decompose the target into blocks
   and apply GQE per block. Block boundaries are determined by a structure-learning outer
   loop.

---

## 9. Summary

The recommended path to multi-objective GQE is:

1. **Enable variable-length generation** by adding EOS to the vocabulary and updating
   the generation / scoring loop. This is the single highest-leverage change.
2. **Add a curriculum multi-objective cost** that starts fidelity-only and gradually
   introduces CNOT and depth penalties. Gate the complexity penalty on a fidelity
   threshold to prevent collapse.
3. **Track a Pareto archive** to expose the full trade-off surface and improve replay
   diversity.
4. **Apply deterministic simplification** on the best circuits found, including Qiskit's
   KAK-based compression as a post-processing pass.

The goal — circuits with fidelity ≥ 0.999 and CNOT count ≤ 3 for generic 2-qubit
unitaries — is achievable with these changes, since the theoretical lower bound (KAK)
already tells us the answer is 3 CNOTs and Qiskit already reaches it. The RL pipeline's
job is to find circuits that match or beat Qiskit's CNOT count while achieving the same
fidelity, and to scale this to larger unitaries where Qiskit's synthesis becomes
intractable.
