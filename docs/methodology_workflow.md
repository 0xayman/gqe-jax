# Methodology Workflow Draft

This document summarizes the end-to-end workflow implemented in the repository and is intended as a starting point for the methodology chapter of a thesis. The description is based on the current code path in `main.py`, `gqe.py`, `pipeline.py`, and the supporting modules that define target generation, circuit evaluation, continuous optimization, and result comparison.

## 1. Research Objective

The project addresses quantum circuit compilation as a hybrid discrete-continuous optimization problem. Given a target unitary matrix $U_{\text{target}} \in \mathbb{C}^{d \times d}$ with $d = 2^n$, the goal is to construct a circuit over a hardware-native gate set whose unitary $U_{\text{circuit}}$ maximizes process fidelity with the target:

$$
F_{\text{process}}(U_{\text{target}}, U_{\text{circuit}})
= \frac{\left|\mathrm{Tr}\left(U_{\text{target}}^\dagger U_{\text{circuit}}\right)\right|^2}{d^2}.
$$

The optimization objective used by the code is the equivalent minimization

$$
\mathcal{L}_{\text{compile}} = 1 - F_{\text{process}}.
$$

The discrete part of the problem is the gate sequence itself, while the continuous part is the choice of rotation angles for parameterized gates. The repository solves the structural search with a transformer-based generative model and optionally refines the continuous parameters with a gradient-based optimizer.

## 2. High-Level Workflow

At a high level, the project runs the following stages:

1. Load and validate the experiment configuration from `config.yml`.
2. Build the operator pool that defines the circuit vocabulary.
3. Generate or load the target unitary.
4. Compile the target with Qiskit to obtain a reference baseline.
5. Construct the fidelity-based cost function.
6. Initialize the GPT-style generative model and the training pipeline.
7. Warm up a replay buffer with sampled circuits and their costs.
8. Repeat rollout, evaluation, replay-buffer insertion, and gradient updates for each epoch.
9. Track the best circuit encountered during training.
10. Re-optimize and decode the best circuit, then compare it against the Qiskit baseline.

The implementation therefore combines sequence modeling, reinforcement-learning-style policy updates, continuous angle refinement, and classical circuit synthesis for evaluation.

## 3. Configuration and Runtime Setup

The experiment is controlled by immutable dataclass-based configuration objects loaded through `config.py`. The configuration is split into six main groups:

- `target`: number of qubits, target type, and optional file path.
- `model`: transformer size and maximum circuit length.
- `training`: epochs, rollout size, batch size, learning rate, gradient clipping, random seed, and GRPO clip ratio.
- `temperature`: the sampling scheduler for inverse temperature.
- `buffer`: replay-buffer capacity, warmup size, and repeated training steps per epoch.
- `continuous_opt`: optional continuous angle optimization settings.

The JAX runtime is configured in `jax_setup.py` to use 64-bit floating point arithmetic and highest-precision matrix multiplication. This choice is important because the project repeatedly composes complex-valued unitary matrices, and numerical stability matters for fidelity estimation.

`main.py` also loads environment variables from `.env` before starting optional Weights & Biases logging. Logging is therefore an implementation detail of the experimental pipeline, not a requirement for the core algorithm.

## 4. Operator Pool Construction

The operator pool is defined in `operator_pool.py` and acts as the vocabulary for the generative model. Each token in the vocabulary corresponds to one gate in the allowed compilation basis.

In the current implementation, the pool contains:

- one `RZ_qi` gate for each qubit $i$,
- one `SX_qi` gate for each qubit $i$,
- one `CNOT_qi_qj` gate for each ordered control-target pair with $i \neq j$.

For $n$ qubits, this gives

$$
|V| = n + n + n(n-1) = n^2 + n
$$

tokens in total.

The stored matrix for each `RZ` token uses a placeholder angle of $\pi/4$. This placeholder is only used for pure discrete evaluation. When continuous optimization is enabled, the gate structure is kept but the angle is re-optimized for each sampled circuit before the final fidelity is assigned.

An important implementation detail is that the current operator pool only exposes `RZ`, `SX`, and `CNOT` at the sampling level. Although `continuous_optimizer.py` supports parsing `RX` and `RY`, those gates are not currently emitted by `build_operator_pool()`. The effective basis used by the main workflow is therefore aligned with Qiskit's `["rz", "sx", "cx"]` basis.

## 5. Target Unitary Preparation

Target preparation is handled by `target.py`. The repository supports three target-generation modes:

### 5.1 Random Reachable Target

`random_reachable` composes a short random circuit sampled from the operator pool. Because the target is built from the same gate vocabulary used during training, perfect reconstruction is possible in principle.

### 5.2 Haar-Random Target

`haar_random` draws a complex Gaussian matrix, performs QR decomposition, and corrects the diagonal phases to obtain a Haar-distributed unitary. This produces a more difficult target because exact representation by a finite circuit of bounded length is generally not guaranteed.

### 5.3 File-Based Target

`file` loads a saved NumPy array from disk and validates that its dimension matches the chosen qubit count.

After the target is created, `main.py` verifies unitarity numerically by checking that

$$
U_{\text{target}} U_{\text{target}}^\dagger \approx I.
$$

One practical note for the thesis text is that the `path` field is only used when `target.type = "file"`. If `target.type = "haar_random"`, the target is generated from the random seed rather than loaded from the path.

## 6. Reference Compilation with Qiskit

Before training starts, `main.py` compiles the target unitary with Qiskit using:

- basis gates `["rz", "sx", "cx"]`,
- optimization level `3`.

This Qiskit circuit is not used for training. Instead, it serves as a classical reference point for post-training comparison. The code records at least three structural statistics from the compiled circuit:

- depth,
- total gate count,
- two-qubit gate count.

This makes it possible to compare the learned GQE circuit against a conventional transpiler output under the same native gate basis.

## 7. Cost Function

The cost function is defined in `cost.py`. Given a sequence of gate matrices, the circuit unitary is composed from left to right as successive matrix multiplications:

$$
U_{\text{circuit}} = G_L \cdots G_2 G_1,
$$

where $G_1$ is the first gate applied to the state.

The fidelity is then converted into a loss by

$$
\text{cost} = 1 - F_{\text{process}}.
$$

The repository provides both NumPy and JAX implementations. The JAX version batches circuit composition and fidelity calculation across many sampled circuits using `jax.lax.scan`, `jax.vmap`, and `jax.jit`.

## 8. Generative Model

The structural search model is implemented in `model.py` as a Flax GPT-2 style autoregressive transformer. The available model sizes are:

| Size | Layers | Heads | Embedding dimension |
| --- | ---: | ---: | ---: |
| `tiny` | 2 | 2 | 128 |
| `small` | 6 | 6 | 384 |
| `medium` | 12 | 12 | 768 |
| `large` | 24 | 16 | 1024 |

The model uses:

- learned token embeddings,
- learned positional embeddings,
- causal self-attention,
- pre-layer-normalization transformer blocks,
- tied output projection through the token embedding matrix,
- dropout during both training and rollout generation.

Each circuit is represented as a fixed-length token sequence of length `max_gates_count + 1`. The extra leading position acts as the start token in the implementation. There is no explicit end-of-sequence token, so all generated circuits have the same maximum length from the model's perspective.

## 9. Pipeline Initialization

`gqe.py` builds the transformer and wraps it inside the `Pipeline` class defined in `pipeline.py`. During initialization, the pipeline:

1. Stores the configuration, model, operator pool, cost function, and helper factory.
2. Creates the GRPO-style loss object.
3. Creates the inverse-temperature scheduler.
4. Allocates a FIFO replay buffer.
5. Converts the pool into token-level metadata such as qubit indices and two-qubit masks.
6. Precomputes stacked gate matrices for batched JAX fidelity evaluation.
7. Initializes the model parameters and an `optax.adamw` training state with gradient clipping.
8. Optionally creates the continuous optimizer.
9. JIT-compiles the rollout, training-step, and batched-cost functions.

This stage effectively converts the abstract experiment configuration into an executable JAX training system.

## 10. Rollout Generation

At every rollout, the model samples `num_samples` candidate circuits autoregressively. Sampling is performed token by token.

At step $t$, the model receives the currently generated prefix and predicts logits over the operator pool. The next token is sampled with

$$
\Pr(a_t = v) \propto \exp(-\beta \cdot \ell_{t,v}),
$$

where $\ell_{t,v}$ is the model logit for token $v$ and $\beta$ is the current inverse temperature. Larger $\beta$ increases exploitation, while smaller $\beta$ promotes exploration.

The implementation uses a compiled `jax.lax.scan` loop, which keeps rollout generation inside a single JAX computation rather than a Python loop. This is important for performance because rollouts occur at every epoch and during buffer warmup.

## 11. Circuit Evaluation

After a rollout is generated, every sampled sequence is assigned a cost. The evaluation path depends on whether continuous optimization is enabled.

### 11.1 Discrete Evaluation

If `continuous_opt.enabled = false`, the sampled token IDs are mapped directly to their stored unitary matrices from the operator pool. The circuit unitary is composed, process fidelity is computed, and the resulting loss is stored.

This mode is fast, but the `RZ` gates are evaluated at the placeholder angle $\pi/4$, so the fidelity signal reflects both structure and the placeholder parameterization.

### 11.2 Hybrid Discrete-Continuous Evaluation

If `continuous_opt.enabled = true`, the project refines the continuous angles before assigning the final cost. In this setting, the model is judged primarily on the gate structure, while rotation parameters are optimized separately.

Two modes are supported:

- If `top_k = 0`, all sampled circuits are angle-optimized.
- If `top_k > 0`, all sampled circuits first receive a cheap discrete score, then only the top `k` structures are refined continuously.

The resulting fidelity values are converted into rollout costs and used as the training signal.

## 12. Continuous Angle Optimization

Continuous optimization is implemented in `continuous_optimizer.py`. The optimizer receives a sampled gate sequence, converts each token into a gate specification, pads the circuit to a fixed length, and optimizes the angle vector associated with the parameterized gates.

### 12.1 Encoded Circuit Representation

For JAX efficiency, each circuit is represented by fixed-shape arrays that encode:

- gate type identifiers,
- the primary qubit index,
- the CNOT control-target pair index,
- the full angle vector padded to `max_gates`.

This avoids repeated recompilation when circuits have different numbers of active gates.

### 12.2 Optimization Objective

For a fixed gate structure, the optimizer minimizes

$$
\mathcal{L}_{\text{angle}}(\theta) = 1 - F_{\text{process}}(U_{\text{target}}, U_{\text{circuit}}(\theta)).
$$

Gradients are obtained through JAX automatic differentiation over the full circuit-construction pipeline.

### 12.3 Optimizers

Two optimizers are supported:

- `lbfgs`, implemented with Optax L-BFGS,
- `adam`, implemented with Optax Adam.

The default configuration in `config.yml` uses L-BFGS.

### 12.4 Multi-Restart Strategy

Because the angle landscape is non-convex, the optimizer can run multiple restarts:

- one restart begins from the default pool angle initialization,
- the remaining restarts sample angles uniformly from $[-\pi, \pi]$.

The best fidelity across restarts is retained.

### 12.5 Simplification and Re-Optimization

After optimization, the circuit can be simplified algebraically:

- near-zero rotations are removed,
- consecutive rotations of the same type on the same qubit are merged,
- consecutive identical CNOT gates are canceled.

If simplification shortens the circuit and still preserves fidelity within a small tolerance, the simplified circuit is accepted. If the simplified circuit still contains trainable parameters, the code performs a one-restart refinement pass on the shorter structure.

## 13. Replay Buffer and Training Data Reuse

Every evaluated circuit is inserted into the FIFO replay buffer together with its cost. The replay buffer allows the model to learn from multiple recent rollouts rather than only from the most recent batch.

The training dataset is represented by `BufferDataset`, which repeats the replay buffer `steps_per_epoch` times. Batches are then read sequentially from this repeated view. This means that each epoch consists of:

- one new rollout,
- insertion of all sampled circuits into the buffer,
- multiple gradient batches drawn from the accumulated buffer contents.

The default configuration therefore mixes on-policy and short-horizon replayed experience.

## 14. GRPO-Style Policy Update

The training update is implemented directly in `pipeline.py`, with the helper loss logic preserved in `loss.py`. The update resembles a clipped PPO/GRPO variant:

1. For the first batch in an epoch, the model computes token log-probabilities for the sampled circuits.
2. The lowest-cost circuit in the batch is treated as the winner.
3. Advantages are computed by normalizing costs relative to the batch mean and unbiased standard deviation.
4. The first batch stores reference log-probabilities and advantages.
5. Later batches in the same epoch use a clipped probability ratio against that stored reference.

The clipped ratio is

$$
r_t = \exp\left(\log \pi_{\theta}(a_t) - \log \pi_{\theta_{\text{ref}}}(a_t)\right),
$$

and the update uses a clip interval controlled by `grpo_clip_ratio`.

From an implementation perspective, the optimizer is `optax.adamw` with:

- global gradient clipping,
- weight decay,
- fixed learning rate from the configuration.

Best-circuit tracking is handled outside the gradient step. After each rollout, the pipeline checks whether the current epoch produced a better circuit than any seen before and stores the corresponding token sequence.

## 15. Temperature Scheduling

The inverse temperature schedule is defined in `scheduler.py`. Three policies are available:

- `fixed`: increment by a constant `delta` and clamp to `[min_value, max_value]`,
- `cosine`: oscillate between the minimum and maximum values,
- `variance`: adapt the temperature based on rollout cost variance.

This scheduler is updated after each rollout, not after every gradient batch. It therefore controls the exploration-exploitation balance at the level of sampled circuit populations.

## 16. Epoch-Level Training Procedure

The full training logic in `gqe.py` can be summarized as:

```text
load config
build operator pool
build target unitary
compile target with Qiskit
build cost function
initialize model, pipeline, scheduler, replay buffer, and optional logger

while replay buffer is smaller than warmup_size:
    generate rollout
    evaluate sampled circuits
    push (sequence, cost) pairs into buffer
    update best-so-far circuit

for each epoch:
    generate rollout
    evaluate sampled circuits
    push rollout into replay buffer
    update best-so-far circuit
    update temperature scheduler

    for each replay-buffer batch:
        perform GRPO-style training update

    log epoch metrics
    stop early if enabled and fidelity reaches 1.0

decode best circuit
verify best circuit fidelity
compare against Qiskit baseline
```

This structure is important for a thesis write-up because the model is not trained on a fixed offline dataset. Instead, it continually generates new candidate circuits, evaluates them against the target, and improves its policy from the resulting feedback.

## 17. Post-Training Decoding and Verification

When training ends, `main.py` decodes the best token sequence into gate names and reports the learned circuit.

Two verification paths are used:

- If continuous optimization was enabled during training, the best structure is re-optimized independently using the same continuous optimizer before final reporting.
- If continuous optimization was disabled, the decoded sequence is reconstructed directly from the pool matrices and evaluated as-is.

The final GQE circuit is then converted into a Qiskit `QuantumCircuit` so that structural metrics can be measured in the same representation used for the baseline.

## 18. Final Evaluation Outputs

The final comparison table printed by `main.py` reports, for both GQE and Qiskit:

- process fidelity,
- circuit depth,
- total gate count,
- two-qubit gate count.

This final stage turns the training run into an experimentally interpretable result. The repository therefore evaluates success not only in terms of fidelity, but also in terms of circuit efficiency.

## 19. Default Experimental Setting in the Current Repository

The checked-in `config.yml` currently specifies the following default run:

- `num_qubits = 3`,
- target type `haar_random`,
- model size `small`,
- maximum circuit length `125`,
- `100` training epochs,
- `100` sampled circuits per rollout,
- replay buffer size `500`,
- replay-buffer warmup size `50`,
- `4` training passes over the replay data per epoch,
- continuous optimization enabled,
- L-BFGS angle optimization with `100` steps and `3` restarts,
- early stopping enabled once fidelity reaches `1.0`.

This default setting should be described as an example configuration, not as a hard-coded property of the method. The same workflow can be rerun with different target types, qubit counts, model sizes, and optimization settings.

## 20. Suggested Thesis Framing

For a thesis methodology chapter, the repository can be described as a hybrid compiler-learning system with four interacting components:

1. A hardware-native gate vocabulary that defines the discrete search space.
2. A GPT-style autoregressive policy that proposes candidate circuit structures.
3. A fidelity-based evaluator with optional continuous parameter refinement.
4. A replay-buffer training loop that updates the policy from sampled circuit quality.

A concise chapter-level summary could therefore be phrased as follows:

> The proposed method formulates quantum circuit compilation as an autoregressive sequence-generation problem over a native gate vocabulary. At each epoch, a transformer policy samples candidate circuits, each candidate is scored against a target unitary using process fidelity, and the resulting trajectories are stored in a replay buffer. When parameterized gates are present, a continuous inner-loop optimizer refines the rotation angles for each sampled structure before the fidelity is assigned. The transformer is then updated with a GRPO-style clipped policy objective, and the best circuit found during training is finally re-optimized and compared against a Qiskit compilation baseline in terms of fidelity, depth, total gate count, and two-qubit gate count.

## 21. Notes for Final Thesis Editing

Before using this document verbatim in a thesis, it would be worth deciding:

- whether to describe the policy update as "GRPO-style", "PPO-style", or both,
- whether to include the implementation detail that the start token is encoded as index `0`,
- whether to present the replay buffer as an intentional off-policy component or simply as an experience-reuse mechanism,
- whether to discuss only the active operator pool (`RZ`, `SX`, `CNOT`) or also mention the latent support for `RX` and `RY` inside the continuous optimizer.

Those decisions are editorial rather than technical, but clarifying them will make the final methodology chapter more precise.
