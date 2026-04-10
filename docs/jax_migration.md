# JAX Migration Notes

## Original Architecture

The original codebase was a flat PyTorch/Lightning project with these runtime stages:

1. `main.py` loads config, builds the gate pool and target unitary, builds a process-fidelity cost function, runs `gqe()`, and optionally verifies the best circuit with continuous angle optimization plus Qiskit reconstruction.
2. `gqe.py` constructs a GPT-2 language model, wraps it in `Pipeline`, warms a replay buffer with sampled circuits, then runs a manual epoch/batch training loop.
3. `pipeline.py` generates fixed-length gate sequences autoregressively, optionally applies a circuit grammar mask, evaluates each sequence with either discrete process fidelity or continuous angle optimization, stores `(sequence, cost)` pairs in a FIFO replay buffer, and trains with a GRPO-style loss.
4. `continuous_optimizer.py` performs per-circuit angle optimization and algebraic simplification for parametric gates.

## PyTorch -> JAX Mapping

| PyTorch component | JAX replacement |
|---|---|
| `GPT2LMHeadModel` | Custom Flax GPT-2 LM in `model.py` |
| `torch.optim.AdamW` | `optax.adamw` with gradient clipping |
| `torch.autograd` | `jax.value_and_grad` |
| `torch.distributions.Categorical` | `jax.random.categorical` |
| `torch.no_grad()` rollout | JAX sampling path with explicit PRNG keys |
| `Dataset` / `DataLoader` | Pure Python replay buffer batching in `data.py` |
| Tensor ops / masking | `jax.numpy`, `jax.lax.scan`, `jax.vmap`, `jit` |
| Torch-based continuous optimizer | JAX + Optax `lbfgs` / `adam` |
| Lightning W&B logger | Direct W&B wrapper in `gqe.py` |

## Subtle Translation Choices

- The GPT model still uses GPT-2-style tied embeddings, learned positional embeddings, pre-LN blocks, and dropout during both rollout sampling and training, because the PyTorch code never switched the model to eval mode during generation.
- The start token is still the literal token index `0`, matching the original behavior where the "start" token is not a separate vocabulary item.
- The GRPO loss preserves the original first-batch reference behavior, including the unusual use of batch 0 log-probabilities and advantages as the reference for the rest of the epoch.
- Unbiased standard deviation (`ddof=1`) is kept to stay close to PyTorch's default reduction semantics. This means very small batches can still produce `nan`, just like the original implementation.
- Continuous angle optimization now uses fixed-shape padded circuit encodings so the heavy unitary-building path can stay jitted. The optimization objective and simplification rules are unchanged.
- The largest unavoidable numerical difference is L-BFGS: Optax's implementation is close in spirit but not bit-identical to PyTorch's `LBFGS(..., line_search_fn="strong_wolfe")`.

## Validation Checklist

- Compare sampled sequence shapes and token ranges between the old and new rollout path.
- Compare replay buffer insertion order and batch ordering over one short epoch.
- Compare discrete process fidelity for the same explicit gate sequence.
- Compare grammar masks for a few hand-picked partial circuits.
- Compare GRPO loss on one frozen minibatch using saved logits/costs.
- Compare continuous optimizer output fidelity and simplified gate list on a small fixed circuit.
- Run `main.py` on the same config in both versions and compare best fidelity, decoded gate names, and Qiskit circuit statistics.

## Performance Notes

- The JAX version removes host-side tensor/autograd overhead from the main training loop and keeps the model forward/backward path jitted.
- Discrete cost evaluation is now batched with `lax.scan` and `vmap` instead of per-sequence Python loops.
- Autoregressive rollout generation runs as a compiled scan with explicit PRNG splitting and batched grammar masking.
- The main remaining bottleneck is continuous angle optimization, because each sampled circuit still requires its own iterative optimizer run and occasional simplification/re-optimization pass. The fixed-shape encoded representation reduces recompilation cost, but this stage remains more expensive than pure discrete rollouts.
