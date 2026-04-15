# Pareto-Front GQE: Plan and Implementation Guide

## 1. What Is a Pareto Front and Why Does It Fit This Problem

### 1.1 The Multi-Objective Problem

The goal is to find quantum circuits that simultaneously satisfy three objectives:

| Objective | Direction | Symbol |
|---|---|---|
| Process fidelity | Maximize | F ∈ [0, 1] |
| Circuit depth | Minimize | d ∈ {0, 1, 2, …} |
| CNOT count | Minimize | c ∈ {0, 1, 2, …} |

These objectives conflict. A longer circuit has more degrees of freedom and can achieve
higher fidelity, but more gates means more hardware error and longer execution time.
There is no single "correct" trade-off; the right answer depends on the hardware and
the application.

### 1.2 Dominance and the Pareto Front

Circuit A **dominates** circuit B (written A ≻ B) if and only if:

```
F(A) ≥ F(B)  AND  d(A) ≤ d(B)  AND  c(A) ≤ c(B)
with at least one strict inequality
```

A circuit is **Pareto-optimal** if no other known circuit dominates it. The
**Pareto front** is the complete set of Pareto-optimal circuits. It defines the best
achievable trade-off surface across all three objectives.

```
Fidelity
  1.0 ┤    ●          ← high fidelity, high CNOT
  0.9 ┤       ●       ← medium fidelity, medium CNOT
  0.8 ┤          ●    ← lower fidelity, low CNOT
  0.7 ┤             ● ← lowest useful fidelity, very low CNOT
      └────────────────
        many → few   CNOT count
```

The Pareto front is a curve (or surface in 3D). Every point on it is optimal given
some trade-off preference. No circuit can improve on a Pareto-optimal circuit in one
objective without getting worse in another.

### 1.3 Why Not a Fixed Weighted Sum?

A common alternative is to collapse all objectives into one scalar:

```
cost = (1 − F) + α × d_norm + β × c_norm
```

This works for a specific (α, β), but:
1. You must choose (α, β) before training. The wrong weights bias the result toward one
   region of the Pareto front and miss all others.
2. A weighted sum can only recover points on the **convex hull** of the Pareto front.
   Non-convex parts of the front are permanently inaccessible.
3. The output is a single circuit, not a set of trade-off options.

The Pareto-front approach produces the **full trade-off surface** in a single training
run and does not require choosing weights upfront.

### 1.4 Why the Pareto Front Is Especially Natural Here

The current pipeline already computes and logs fidelity, depth, and CNOT count for
every rollout sample (see `pipeline._last_rollout_costs`,
`pipeline._last_rollout_fidelities`, `pipeline._last_rollout_cnot_counts`, and
`pipeline.sequence_structure_metrics`). The raw materials are there; what is missing is:

1. A data structure that stores the non-dominated set.
2. A training signal that covers the full trade-off surface instead of a single point.
3. Logging and reporting of the Pareto front over time.

---

## 2. The Training Algorithm: Scalarized GRPO with Random Weight Sampling

### 2.1 Core Idea

Instead of using a fixed cost `1 − F` for every rollout, we sample a **weight vector**
w = (w_F, w_d, w_c) at the start of each rollout epoch and compute a
**scalarized cost** for that rollout:

```
cost_i(w) = w_F × (1 − F_i) + w_d × d̃_i + w_c × c̃_i
```

where d̃ = d / d_ref and c̃ = c / c_ref are normalized depth and CNOT count,
and d_ref, c_ref are reference values (e.g., from the Qiskit baseline).

The weight vector changes each epoch. Over many epochs, training covers the full
trade-off surface:
- Epochs with w = (1.0, 0, 0): policy is rewarded purely for fidelity.
- Epochs with w = (0.5, 0.3, 0.2): policy is rewarded for balanced fidelity + short circuits.
- Epochs with w = (0.5, 0, 0.5): policy is rewarded for fidelity + low CNOT count.

The GRPO update is mathematically identical to the current one; only the cost value
changes. Advantages are still computed as z-scores within the current rollout batch.
The clipped PPO-style ratio update does not need to change at all.

This approach is known as **PGMORL** (Policy Gradient Multi-Objective Reinforcement
Learning, Xu et al. 2020) and works because: if a circuit is good under many different
weight vectors, it is likely Pareto-optimal or near-Pareto-optimal.

### 2.2 Weight Sampling Distribution

Sample weight vectors from a **Dirichlet distribution with biased parameters**:

```python
raw = np.random.dirichlet([alpha_F, alpha_d, alpha_c])
w_F, w_d, w_c = raw
```

Suggested parameters:
- `alpha_F = 3.0` — strong bias toward fidelity (we always want high-fidelity circuits).
- `alpha_d = 1.0` — moderate weight on depth.
- `alpha_c = 1.0` — moderate weight on CNOT count.

With these parameters, w_F is approximately Beta(3, 2) marginally, so it concentrates
around 0.6 but spans [0.2, 1.0]. Fidelity is never completely ignored.

**Hard minimum for w_F**: clip w_F ≥ w_F_min (e.g., 0.4) after sampling. This
prevents degenerate rollouts where the policy is rewarded purely for minimizing depth
with zero regard for fidelity, which produces useless trivially-short circuits.

### 2.3 Normalization References

The reference values for normalization should come from the Qiskit baseline circuit
already computed in `main.py`:

```python
d_ref = max(qiskit_depth, 1)
c_ref = max(qiskit_two_q, 1)
```

If the Qiskit depth is 5 and CNOT count is 3, then a circuit with depth 5 and 3 CNOTs
contributes `d̃ = 1.0` and `c̃ = 1.0`. A circuit with depth 3 and 1 CNOT contributes
`d̃ = 0.6` and `c̃ = 0.33`. This makes the complexity penalties comparable in scale to
the fidelity term.

### 2.4 The Complete Updated Training Loop

```
initialize:
  - Pareto archive (empty)
  - replay buffer (FIFO, same as current)
  - model, optimizer (same as current)
  - d_ref, c_ref from Qiskit baseline
  - Dirichlet(alpha_F=3, alpha_d=1, alpha_c=1)

warm-up:
  - run existing warmup with w = (1, 0, 0) (fidelity only)
  - this ensures circuits in the early buffer are at least fidelity-useful

for each epoch:
  1. sample w = (w_F, w_d, w_c) from Dirichlet, enforce w_F ≥ 0.4
  2. generate N circuits (identical to current generate())
  3. evaluate each circuit:
       - F_i: process fidelity (current path via continuous_optimizer or discrete)
       - d_i, c_i: from batch call to _sequence_structure_metrics
  4. compute scalarized cost:
       cost_i = w_F*(1−F_i) + w_d*(d_i/d_ref) + w_c*(c_i/c_ref)
  5. push (sequence, scalarized_cost, old_log_prob) to replay buffer
  6. update Pareto archive with all N circuits (see Section 3)
  7. run GRPO training steps (identical to current train_batch loop)
  8. update temperature scheduler
  9. log: epoch cost, fidelity, depth, CNOT, Pareto archive size, hypervolume

after training:
  - report the full Pareto archive (sorted by fidelity)
  - for each Pareto-optimal circuit, run the post-training verifier
    (ContinuousOptimizer with simplify=True)
  - print comparison table for each circuit on the Pareto front
```

---

## 3. The Pareto Archive

### 3.1 Data Structure

```python
@dataclass
class ParetoPoint:
    fidelity: float        # maximize
    depth: int             # minimize
    cnot_count: int        # minimize
    token_sequence: np.ndarray  # shape (max_gates_count + 1,) including BOS
    epoch: int             # when it was found
```

The archive is a list of `ParetoPoint`s with the invariant that no entry dominates
any other entry.

### 3.2 Dominance Check

```python
def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Return True if a dominates b."""
    better_or_equal = (
        a.fidelity >= b.fidelity and
        a.depth <= b.depth and
        a.cnot_count <= b.cnot_count
    )
    strictly_better = (
        a.fidelity > b.fidelity or
        a.depth < b.depth or
        a.cnot_count < b.cnot_count
    )
    return better_or_equal and strictly_better
```

### 3.3 Archive Update Algorithm

After each rollout, add each circuit to the archive:

```python
def update(archive, new_point, fidelity_floor=0.5):
    """Add new_point to the archive if it is not dominated."""
    if new_point.fidelity < fidelity_floor:
        return  # reject trivially useless circuits

    # Check if new_point is dominated by any existing archive member
    for existing in archive:
        if dominates(existing, new_point):
            return  # new_point is dominated; discard it

    # new_point is Pareto-optimal; remove any existing members it dominates
    archive[:] = [p for p in archive if not dominates(new_point, p)]
    archive.append(new_point)
```

This is O(|archive| × N_rollout) per epoch where N_rollout is typically 100-200.
The archive size grows slowly (logarithmically in practice) and is bounded by the
complexity of the true Pareto front.

### 3.4 Archive Size Cap (Optional)

If the archive grows too large (e.g., > 500 entries), prune using **crowding distance**:
keep the entry with the smallest crowding distance. This maintains diversity across the
Pareto front.

Crowding distance for a point p in objective k:
```
CD_k(p) = [objective_k(successor) - objective_k(predecessor)] / range_k
CD(p) = sum over k of CD_k(p)
```
Points with small CD are in dense regions; prune them to favor spread.

### 3.5 Fidelity Floor

Set `fidelity_floor = 0.5` to start. This means a circuit is only eligible for the
Pareto archive if it achieves at least 50% fidelity. The floor can be raised during
training (e.g., to 0.9 after epoch 100) to keep the archive focused on high-quality
circuits once the policy has learned to achieve high fidelity.

---

## 4. The Hypervolume Indicator

The **hypervolume** is a scalar quality metric for the entire Pareto archive. It
measures the volume of objective space dominated by the archive and bounded by a
reference point.

Define the reference (worst-case) point:
```
ref = (F_ref=0.0, d_ref=max_depth, c_ref=max_cnot)
     = (0.0, 28, 28)  with the current config
```

The hypervolume is:
```
HV = volume of the region that is:
     - better than ref in all objectives
     - dominated by at least one point in the archive
```

In 2D (fidelity vs CNOT count with depth ignored for visualization):
```
Fidelity
 1.0 ┤   ╔══╗             ← HV contribution of this point
 0.9 ┤   ║  ╔═══════╗    ← HV contribution of this point
 0.8 ┤   ║  ║       ║
     └───╨──╨───────╨──── CNOTs
```

Tracking hypervolume over training epochs tells you whether the policy is exploring
new parts of the Pareto front or stagnating. It is the primary training progress
metric for multi-objective optimization.

Computing hypervolume in 3D is more involved; use the WFG algorithm
(hypervolume-wfg package or a simple 2D projection for visualization).

---

## 5. What Changes Where in the Codebase

### 5.1 New file: `pareto.py`

Implement `ParetoPoint` (dataclass), `ParetoArchive` (class with `update`, `dominates`,
`hypervolume_2d`, `to_sorted_list` methods). No dependencies on JAX or the training loop.

### 5.2 `config.py` and `config.yml`

Add a `ParetoConfig` dataclass and corresponding YAML block:

```yaml
pareto:
  enabled: true
  fidelity_floor: 0.5       # minimum fidelity to enter archive
  fidelity_floor_late: 0.9  # raised after floor_ramp_epoch
  floor_ramp_epoch: 100     # epoch at which floor is raised
  max_archive_size: 500
  alpha_F: 3.0              # Dirichlet param for fidelity weight
  alpha_d: 1.0              # Dirichlet param for depth weight
  alpha_c: 1.0              # Dirichlet param for CNOT weight
  w_F_min: 0.4              # hard minimum for fidelity weight
```

### 5.3 `pipeline.py`

**`collect_rollout()`**: this is the central method to modify.

Currently:
```python
costs, fidelities, cnot_counts = self.computeCost(idx_output[:, 1:], self.pool)
for seq, cost_val, old_log_prob in zip(idx_output, costs, old_log_probs):
    self.buffer.push(seq, cost_val, old_log_prob)
```

New version:
```python
costs_raw, fidelities, cnot_counts = self.computeCost(idx_output[:, 1:], self.pool)
depths = self._batch_compute_depths(idx_output[:, 1:])  # new helper

# compute scalarized cost with the epoch's weight vector
costs = self._scalarize(fidelities, depths, cnot_counts, self.current_weights)

for seq, cost_val, old_log_prob in zip(idx_output, costs, old_log_probs):
    self.buffer.push(seq, cost_val, old_log_prob)

# update pareto archive
if self.pareto_archive is not None:
    for i in range(len(idx_output)):
        self.pareto_archive.update(ParetoPoint(
            fidelity=float(fidelities[i]),
            depth=int(depths[i]),
            cnot_count=int(cnot_counts[i]),
            token_sequence=np.asarray(idx_output[i]),
            epoch=self._current_epoch,
        ))
```

Add `_batch_compute_depths()`:
```python
def _batch_compute_depths(self, token_ids: np.ndarray) -> np.ndarray:
    depths = np.zeros(len(token_ids), dtype=np.int32)
    for i, row in enumerate(token_ids):
        depths[i], _, _ = self.sequence_structure_metrics(row)
    return depths
```

Add `_scalarize()`:
```python
def _scalarize(self, fidelities, depths, cnot_counts, w):
    w_F, w_d, w_c = w
    return (
        w_F * (1.0 - fidelities)
        + w_d * (depths / self.cfg.pareto.d_ref)
        + w_c * (cnot_counts / self.cfg.pareto.c_ref)
    ).astype(np.float32)
```

Add weight sampling at the start of `collect_rollout()`:
```python
self.current_weights = self._sample_weights()
```

```python
def _sample_weights(self):
    alpha = [self.cfg.pareto.alpha_F, self.cfg.pareto.alpha_d, self.cfg.pareto.alpha_c]
    w = np.random.dirichlet(alpha)
    w[0] = max(w[0], self.cfg.pareto.w_F_min)
    w = w / w.sum()  # renormalize after clipping
    return w
```

### 5.4 `gqe.py` (`_run_training`)

- Pass epoch number into `pipeline` so it can tag Pareto archive entries.
- After training finishes, extract the Pareto archive and return it alongside
  `best_cost` and `best_indices`.
- During warm-up, use w = (1, 0, 0) (fidelity only). Only introduce random weight
  sampling after warm-up is complete.

### 5.5 `main.py`

After training, instead of printing only the single best circuit, iterate over the
Pareto archive and for each Pareto-optimal circuit:
1. Run `ContinuousOptimizer` with `simplify=True`.
2. Build the Qiskit circuit.
3. Print a row in the comparison table.

This gives you the full trade-off surface as the final output.

---

## 6. Warm-Up Strategy

A critical detail: **do not use random weights during warm-up**.

During warm-up, the policy is essentially random. If you sample complex weight vectors
(e.g., w_d = 0.7) during warm-up, the buffer fills with circuits that the policy was
accidentally "good" at optimizing under that random weight — which has nothing to do
with the actual Pareto front. This poisons the buffer with misleading data.

Warm-up strategy:
- All warm-up rollouts use w = (1, 0, 0): fidelity only.
- This matches the current behavior and seeds the buffer with fidelity-useful circuits.
- Random weight sampling begins only at epoch 0 (after warm-up is complete).

Additionally, use a fidelity-weighted Dirichlet schedule:
- Epochs 0–50: alpha_F = 5.0 (strong fidelity bias). The policy first learns to achieve
  high fidelity; complexity is a secondary concern.
- Epochs 50–200: alpha_F = 3.0 (moderate fidelity bias). Complexity penalties become
  more influential as the policy is already capable of achieving high fidelity.

---

## 7. Expected Behavior Over Training

```
Epochs 0–50:
  - Policy mostly sees fidelity-only weights.
  - Pareto archive fills primarily with high-fidelity circuits at various depths.
  - Hypervolume grows rapidly in the fidelity direction.

Epochs 50–150:
  - Complexity weights start influencing training.
  - Policy begins to discover shorter circuits that maintain high fidelity.
  - Pareto front "pushes out" in the low-CNOT direction.
  - Hypervolume continues growing, now in both dimensions.

Epochs 150–200:
  - Policy is near-converged. Most rollouts sample circuits near the Pareto front.
  - New archive entries are rare; the front is mostly stable.
  - Hypervolume plateaus.

Final output:
  - Archive contains ~5–20 Pareto-optimal circuits.
  - At one extreme: high fidelity (F ≈ 0.999) with CNOT count similar to Qiskit.
  - At other extreme: lower fidelity (F ≈ 0.95) with fewer CNOTs.
  - For 2-qubit Haar-random targets: expect to match or beat Qiskit's 3 CNOTs at F > 0.99.
```

---

## 8. New Metrics to Log (WandB)

| Metric | What It Measures |
|---|---|
| `pareto_archive_size` | Number of non-dominated circuits found so far |
| `pareto_hypervolume` | Total hypervolume dominated by archive (higher = better front) |
| `pareto_best_fidelity` | Maximum fidelity in the archive |
| `pareto_min_cnot` | Minimum CNOT count among circuits with F ≥ 0.99 |
| `pareto_min_depth` | Minimum depth among circuits with F ≥ 0.99 |
| `w_F` / `w_d` / `w_c` | Weight vector sampled this epoch (for debugging) |
| `cost_epoch_best` | Scalarized cost of best circuit this epoch (same as current) |

---

## 9. Why Pareto-Front Is the Right Choice for This Project

| Aspect | Single-objective (current) | Fixed weighted sum | Pareto front |
|---|---|---|---|
| Result | One circuit | One circuit | Full trade-off surface |
| Weight choice | N/A (fidelity only) | Must be chosen before training | Automatic |
| Pareto coverage | Only one point | Only convex hull | Full front including non-convex parts |
| Reusability | Retrain for new weights | Retrain for new weights | Query any trade-off post-hoc |
| Complexity overhead | None | None | O(|archive| × N) per epoch; archive stays small |
| Change to GRPO | None | Minimal (add terms) | Minimal (random weights, same GRPO equations) |
| Suitability for hardware deployment | Poor (ignores gate cost) | Partial (fixed penalty) | Best (explicit gate-cost trade-off) |

The Pareto approach adds essentially no computational overhead to the existing GRPO loop.
The only additions are: one weight sample per epoch (negligible), one dominance-check pass
over the archive per rollout (O(archive_size × rollout_size)), and depth computation
per sample (same complexity as CNOT counting which is already done).

The payoff is that at the end of a single training run, you have a map of the full
achievable trade-off surface for that target unitary — not just a single circuit.
