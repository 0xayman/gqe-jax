# Current GQE Training Pipeline

This document describes the current end-to-end GQE pipeline implemented in this repository, from target-unitary generation/loading to the final compiled circuit that `main.py` prints and compares against Qiskit.

It is based on the code paths in `main.py`, `gqe.py`, `pipeline.py`, `target.py`, `operator_pool.py`, and `continuous_optimizer.py`, not on the older design notes.

## Scope

There are two closely related "compiled circuits" in the current workflow:

1. A Qiskit reference circuit obtained by transpiling the target unitary into the native basis `["rz", "sx", "cx"]`.
2. A GQE-learned circuit produced by training a transformer over gate tokens, then re-optimizing the best structure's continuous angles and converting that result into a Qiskit circuit for display and comparison.

## Default Run Today

With the current `config.yml`, the default run is:

- `target.type: "file"`
- `target.path: "targets/haar_random_q2.npy"`
- `target.num_qubits: 2`
- `model.size: "small"`
- `model.max_gates_count: 28`
- `continuous_opt.enabled: true`
- `continuous_opt.optimizer: "lbfgs"`
- `continuous_opt.top_k: 0`
- `continuous_opt.num_restarts: 3`

That means the "current" default path is:

1. A Haar-random unitary has already been saved as a `.npy` file.
2. `main.py` loads that file as the target.
3. GQE trains on 28-gate fixed-length sequences.
4. Every sampled circuit is continuously re-optimized before its final fidelity is used as the training signal.
5. The best learned structure is re-optimized again at the end and rebuilt as a Qiskit circuit.

## End-to-End Flow

```text
Optional offline target generation
    -> config load
    -> operator pool build
    -> target build/load
    -> target unitarity check
    -> Qiskit reference compilation
    -> cost/pipeline/model initialization
    -> replay-buffer warmup
    -> repeated rollout generation
    -> discrete or hybrid scoring
    -> replay-buffer insertion
    -> GRPO training updates
    -> best-sequence tracking
    -> final re-optimization of best structure
    -> GQE circuit reconstruction in Qiskit
    -> side-by-side comparison with Qiskit reference
```

## 1. Target Unitary Generation or Loading

Target preparation is centralized in `target.py` through `build_target(pool, cfg)`.

Supported target modes:

- `random_reachable`: sample a short random circuit from the operator pool and compose it into a reachable target unitary.
- `haar_random`: generate a Haar-random unitary directly in memory with QR decomposition.
- `file`: load a saved complex matrix from disk.

For the current default config, `target.type` is `file`, so `build_target()` resolves `targets/haar_random_q2.npy`, loads it with NumPy, and checks that its shape is `(2**num_qubits, 2**num_qubits)`.

If you want to create that file ahead of time, the repository also includes `generate_haar_unitary.py`, which writes a Haar-random unitary into `targets/`.

Important implementation detail:

- `random_reachable` builds the target from the operator-pool matrices themselves, so any sampled `RZ` gates use the pool's placeholder angle `pi/4`.
- `haar_random` and `file` targets are independent of the operator pool's placeholder angles.

## 2. Configuration and Runtime Setup

`main.py` starts by:

1. Loading environment variables from `.env`.
2. Loading and validating `config.yml` through `config.py`.
3. Printing the selected runtime settings.
4. Optionally initializing Weights & Biases logging.

JAX runtime defaults are defined in `jax_setup.py`:

- `jax_enable_x64 = True`
- `jax_default_matmul_precision = "highest"`

One current exception is continuous angle optimization: `Factory.create_continuous_optimizer()` constructs `ContinuousOptimizer(..., fast_runtime=True)`, so the optimizer internally uses lower-precision float/complex dtypes for speed even though the wider JAX runtime is configured for x64 elsewhere.

## 3. Operator Pool Construction

`build_operator_pool(num_qubits)` creates the discrete gate vocabulary used by the model.

Today the effective sampled basis is:

- `RZ_q{i}` for each qubit
- `SX_q{i}` for each qubit
- `CNOT_q{i}_q{j}` for each ordered control-target pair where `i != j`

Two details matter here:

1. `ROTATION_AXIS = ("RZ",)`, so the training pipeline does not currently sample `RX` or `RY`, even though `continuous_optimizer.py` still knows how to parse and optimize them.
2. Each `RZ` token stores a full unitary matrix at a placeholder angle of `pi/4`. That placeholder is only the discrete default. When continuous optimization is enabled, the angle is re-optimized later.

The pool matrices are full `2^n x 2^n` embedded unitaries, not symbolic gate descriptions.

## 4. Target Validation and Qiskit Baseline

After `build_target()` returns, `main.py` verifies unitarity numerically by checking:

`u_target @ u_target.conj().T ~= I`

It then compiles the target with Qiskit:

- wraps the target matrix in `UnitaryGate`
- inserts it into a `QuantumCircuit`
- transpiles with basis gates `["rz", "sx", "cx"]`
- uses `optimization_level=3`

This Qiskit result is a reference baseline only. It is not used to supervise training. The code records:

- circuit depth
- total gate count
- two-qubit gate count

Those metrics are used only at the end for comparison against the learned GQE circuit.

## 5. Cost Function

`main.py` builds `cost_fn = build_cost_fn(u_target)`, which represents:

`cost = 1 - process_fidelity(U_target, U_circuit)`

where:

`process_fidelity(U_target, U_circuit) = |Tr(U_target^dagger U_circuit)|^2 / d^2`

Current implementation detail:

- `cost_fn` is still passed through the public API into `gqe(...)` and `Pipeline(...)`.
- In the current JAX pipeline, rollout scoring is actually done from `u_target` plus precomputed pool matrices, not by calling the Python `cost_fn` directly.

So `cost_fn` remains part of the public interface, but the active scoring path is the batched JAX implementation in `pipeline.py` and `cost.py`.

## 6. Sequence Representation

The transformer does not generate variable-length circuits with an EOS token. The current representation is fixed-length.

Each sampled sequence has length:

`max_gates_count + 1`

The extra token is a leading BOS token:

- BOS token id is `0`
- actual gate token ids are shifted by `+1`
- decoding back into the operator pool subtracts that offset

With the current config, every rollout sample is always:

- 1 BOS token
- 28 generated gate tokens

There is no early termination during generation. The model always emits exactly `max_gates_count` gates per sample.

## 7. Model Initialization

`gqe.py` creates a `GPT2` model with vocabulary size `len(pool) + 1`, to account for BOS.

`Pipeline.__init__()` then:

1. Stores config, model, pool, and target references.
2. Builds the GRPO loss object.
3. Builds the inverse-temperature scheduler.
4. Creates the replay buffer.
5. Precomputes token metadata such as qubit ownership and which tokens are two-qubit gates.
6. Stacks pool matrices into a batch-friendly array.
7. Initializes model parameters with a dummy `(1, ngates + 1)` input.
8. Creates an `optax` optimizer: global-norm clip followed by `adamw`.
9. Optionally creates the continuous optimizer.
10. JIT-compiles rollout generation, GRPO training, and discrete batched scoring helpers.

The model architecture in `model.py` is a GPT-2 style causal transformer with learned token embeddings, learned positional embeddings, pre-layernorm blocks, causal self-attention, and tied output projection via `wte.attend(...)`.

Current implementation detail:

- The model defines dropout layers.
- The active pipeline calls `apply(..., deterministic=True)` for rollout and training computations.
- In practice, dropout is therefore disabled in the current training path.

## 8. Replay-Buffer Warmup

Training does not start from an empty replay buffer.

`gqe._run_training()` first loops:

`while len(buffer) < warmup_size: pipeline.collect_rollout()`

`collect_rollout()` does four things:

1. Generate a batch of candidate sequences.
2. Score every sequence.
3. Push `(sequence, cost)` pairs into the FIFO replay buffer.
4. Update the scheduler using the rollout costs.

With the current config:

- `num_samples = 100`
- `warmup_size = 50`

So one rollout is already enough to satisfy warmup, because the buffer receives 100 sequences at once.

## 9. Rollout Generation

Rollouts are generated autoregressively in `Pipeline.generate()` through a JIT-compiled `jax.lax.scan`.

At each position:

1. The current partial token matrix is fed through the transformer.
2. The BOS token is masked out so it cannot be sampled again.
3. The next token is sampled with `jax.random.categorical` from `-beta * logits`.

`beta` comes from the scheduler's `get_inverse_temperature()`.

Although some class names still say "temperature", the current code treats that value directly as inverse temperature:

- larger `beta` sharpens the sampling distribution
- smaller `beta` flattens it

The default scheduler named `"fixed"` is not actually fixed: `DefaultScheduler` adds `delta` after each rollout and clamps the result to `[min_value, max_value]`.

## 10. Rollout Scoring

After generation, `Pipeline.computeCost()` scores the sampled circuits.

### 10.1 Discrete path

If continuous optimization is disabled, the code:

1. decodes gate token ids back into operator-pool indices
2. gathers the corresponding precomputed unitary matrices
3. composes each circuit unitary in JAX
4. computes fidelity and cost in batch

In that path, every `RZ` is evaluated at the pool's placeholder angle `pi/4`.

### 10.2 Hybrid discrete-continuous path

If continuous optimization is enabled, the discrete structure still comes from the transformer, but the final fidelity can be reassigned after angle optimization.

Two modes exist:

- `top_k = 0`: optimize every sampled circuit
- `top_k > 0`: score all circuits discretely first, then only refine the best `top_k`

The current config uses `top_k = 0`, so every sampled circuit is sent through the continuous optimizer before its final rollout fidelity is stored.

### 10.3 What is stored

For each rollout, the pipeline records:

- `costs`
- `fidelities`
- `cnot_counts`
- full token sequences including BOS

All sampled sequences are pushed into the replay buffer together with their scalar cost.

## 11. Continuous Angle Optimization

`ContinuousOptimizer` receives a gate structure and optimizes only the continuous angles for parametric gates.

The current optimizer flow is:

1. Convert token names or token ids into fixed-shape encoded arrays.
2. Build an angle vector of length `max_gates`, with non-parametric gates holding zero.
3. Pad unused positions with a `NOOP` gate type internally.
4. Optimize the angle vector against `1 - process_fidelity`.
5. Optionally simplify the sequence after optimization.

Supported optimizers:

- `lbfgs`
- `adam`

The current config uses `lbfgs` with `num_restarts = 3`.

That means each circuit is optimized from:

- one deterministic initial point based on the placeholder angles
- two additional random restarts in `[-pi, pi]`

The optimizer keeps the restart with the best fidelity.

### Simplification rules

When simplification is enabled, the optimizer can:

- remove near-zero rotations
- merge adjacent rotations of the same axis on the same qubit
- cancel adjacent identical `CNOT`s

Important current behavior:

- rollout scoring calls `optimize_token_index_batch(..., simplify=False)`, so sampled circuits are scored without structural simplification during training
- final verification calls `optimize_circuit_with_params(...)`, which does simplify and may re-optimize the simplified sequence

This is why the final displayed GQE circuit can be shorter or cleaner than the raw best rollout sequence.

## 12. Replay Buffer and GRPO Training

Once a rollout has been collected, training uses the replay buffer rather than only the newest rollout.

`BufferDataset(buffer, repetition=steps_per_epoch)` creates a repeated view of the FIFO buffer. It does not shuffle entries.

With the current config:

- buffer max size is `500`
- `steps_per_epoch = 4`
- batch size is `32`

So each epoch trains over repeated buffer contents, in deterministic order, using full batches only.

### GRPO update

For each epoch:

1. A new rollout is collected and inserted into the buffer.
2. The first batch freezes `reference_params` as a stop-gradient copy of the current model weights.
3. Every batch computes a clipped GRPO objective against that frozen reference.

The advantage is:

`(mean(costs) - costs) / (std(costs) + 1e-8)`

So lower-cost circuits get higher advantage.

The objective is sequence-level:

- log-probabilities are computed over the full generated sequence
- the ratio compares current sequence log-probability to the frozen reference policy
- the ratio is clipped using `grpo_clip_ratio`

The replay buffer stores only sequences and costs. Old log-probabilities are not stored; the reference policy for an epoch is recomputed from the frozen parameters taken at batch 0.

## 13. Best-Circuit Tracking and Early Stop

`gqe._run_training()` tracks the best circuit seen so far from rollout results, not from training batches.

After each rollout it updates:

- best cost
- best token sequence
- best raw fidelity
- best CNOT count

If `training.early_stop` is enabled, training stops when the tracked best raw fidelity reaches `1.0`.

## 14. Post-Training Reconstruction of the Final GQE Circuit

After `gqe(...)` returns, `main.py` decodes the best token sequence:

1. Drop the BOS token.
2. Subtract the `+1` gate-token offset.
3. Map token ids back into pool gate names.

Then the code branches:

### If continuous optimization is enabled

`main.py` creates a fresh `ContinuousOptimizer` and re-runs optimization on the best gate-name sequence with the same optimizer settings from the config.

It gets back:

- verified fidelity
- `gate_specs`
- optimized continuous parameters

Those are converted into a Qiskit circuit with actual `rz` angles and the fixed `sx` / `cx` gates.

### If continuous optimization is disabled

`main.py` composes the stored pool matrices directly, computes fidelity once, and rebuilds a Qiskit circuit from gate names using the placeholder angle `pi/4` for rotation gates.

## 15. Final Output and Comparison

At the end of the run, `main.py` prints:

- best training cost
- decoded best gate list
- verified final fidelity
- CNOT count
- the final GQE circuit drawing, when continuous optimization is enabled
- a side-by-side GQE vs Qiskit comparison table

The comparison table reports:

- fidelity
- depth
- total gate count
- two-qubit gate count

So the practical output of the full pipeline is not just a best token sequence. It is:

1. a verified circuit reconstruction for the learned structure
2. a Qiskit representation of that learned circuit
3. a direct structural comparison against Qiskit's compilation of the same target unitary

## 16. Current Behavioral Notes and Limitations

The following points are easy to miss if you only read high-level notes:

- The current model always generates fixed-length circuits. There is no EOS token and no learned early stop.
- The active sampled basis is `RZ`, `SX`, `CNOT` only.
- `RX` and `RY` support exists in the continuous optimizer, but those gates are not currently emitted by `build_operator_pool()`.
- During training, rollout circuits are scored without simplification; simplification only happens in the final verifier path.
- `build_cost_fn()` is constructed in `main.py`, but the active training-time scoring path uses batched JAX evaluation from `u_target` and precomputed pool matrices.
- Dropout layers exist in the model definition, but the current pipeline runs the model with `deterministic=True`.
- The standalone printed `CNOT count` comes from the raw decoded best token sequence, while the comparison table uses the reconstructed final Qiskit circuit after verification and possible simplification.

## Summary

The current GQE pipeline is a hybrid loop:

1. obtain a target unitary
2. compile that target with Qiskit for a reference baseline
3. let a GPT-style policy sample fixed-length gate sequences from the native basis
4. score each sequence by process fidelity, optionally after continuous angle optimization
5. store scored sequences in a FIFO replay buffer
6. train the policy with a clipped GRPO objective against a frozen per-epoch reference policy
7. keep the best sampled structure
8. re-optimize and simplify that best structure after training
9. convert the result into a final Qiskit circuit and compare it against the Qiskit baseline

That is the full current path from unitary generation/loading to the final compiled circuit.
