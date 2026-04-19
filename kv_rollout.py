"""KV-cache-based autoregressive decoding for GPT-2.

The stock training model (`model.py`) processes the full sequence at every
generation step. For a rollout of length N, that is O(N^2) work on the
transformer. This module implements a decode-only variant that processes one
token at a time and keeps a per-layer (K, V) cache, giving O(N) work overall.

It reads weights directly from the Flax train-state `params` dict — no model
refactor required. Params layout must match `model.GPT2`:

    params = {
        "wte": {"embedding": [V, D]},
        "wpe": {"embedding": [P, D]},
        "h_0":  {"ln_1": {scale, bias}, "attn": {"c_attn": {kernel, bias},
                                                  "c_proj": {kernel, bias}},
                 "ln_2": {scale, bias}, "mlp":  {"c_fc":   {kernel, bias},
                                                  "c_proj": {kernel, bias}}},
        ...
        "h_{L-1}": ...,
        "ln_f": {scale, bias},
    }
"""

from __future__ import annotations

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp


_LAYER_NORM_EPS = 1.0e-5


def _layernorm(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + _LAYER_NORM_EPS)
    return (x - mean) * inv * scale + bias


def _gelu_new(x: jax.Array) -> jax.Array:
    coeff = jnp.sqrt(2.0 / jnp.pi)
    return 0.5 * x * (1.0 + jnp.tanh(coeff * (x + 0.044715 * (x**3))))


def build_kv_rollout_fn(
    n_layer: int,
    n_head: int,
    n_embd: int,
    vocab_size: int,
    batch_size: int,
    total_len: int,
    bos_token_id: int = 0,
):
    """Return a JIT-compiled rollout function using a KV cache.

    The returned function signature is ``(params, rng_key, beta) -> tokens``
    where ``tokens`` has shape ``(batch_size, total_len)`` with the BOS token
    at position 0.
    """
    head_dim = n_embd // n_head
    assert head_dim * n_head == n_embd

    def decode_step(params, k_cache, v_cache, prev_token, step_idx):
        """One autoregressive step.

        Args:
            params: Flax params pytree matching model.GPT2 layout.
            k_cache: (L, B, H, T, Dh) pre-allocated key cache.
            v_cache: (L, B, H, T, Dh) pre-allocated value cache.
            prev_token: (B,) int32 — token at position step_idx.
            step_idx: scalar int32 — current position in the sequence.

        Returns:
            k_cache, v_cache, logits where logits has shape (B, V).
        """
        wte_E = params["wte"]["embedding"]     # (V, D)
        wpe_E = params["wpe"]["embedding"]     # (P, D)

        # x: (B, 1, D)
        x = wte_E[prev_token] + wpe_E[step_idx]
        x = x[:, None, :]

        # Mask: valid keys are positions [0, step_idx].
        positions = jnp.arange(total_len, dtype=jnp.int32)
        key_mask = positions <= step_idx                          # (T,)

        for i in range(n_layer):
            block = params[f"h_{i}"]

            # Pre-attention LayerNorm
            attn_in = _layernorm(
                x,
                block["ln_1"]["scale"],
                block["ln_1"]["bias"],
            )

            # QKV projection: (B, 1, 3D)
            c_attn = block["attn"]["c_attn"]
            qkv = attn_in @ c_attn["kernel"] + c_attn["bias"]
            q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, 1, D)

            # Split heads: (B, 1, H, Dh) -> (B, H, 1, Dh)
            q = q.reshape(batch_size, 1, n_head, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, 1, n_head, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, 1, n_head, head_dim).transpose(0, 2, 1, 3)

            # Update cache: write (B, H, 1, Dh) slice at index step_idx along T.
            k_c = jax.lax.dynamic_update_slice_in_dim(
                k_cache[i], k, step_idx, axis=2
            )
            v_c = jax.lax.dynamic_update_slice_in_dim(
                v_cache[i], v, step_idx, axis=2
            )
            k_cache = k_cache.at[i].set(k_c)
            v_cache = v_cache.at[i].set(v_c)

            # Scaled dot-product attention with cache: scores (B, H, 1, T)
            scale = jnp.asarray(head_dim, dtype=x.dtype) ** -0.5
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_c) * scale

            # Mask out positions beyond step_idx.
            min_val = jnp.finfo(scores.dtype).min
            scores = jnp.where(
                key_mask[None, None, None, :],
                scores,
                jnp.asarray(min_val, dtype=scores.dtype),
            )
            attn = jax.nn.softmax(scores, axis=-1)

            # Combine with V: (B, H, 1, Dh) -> (B, 1, D)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v_c)
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, 1, n_embd)

            # Output projection + residual
            c_proj = block["attn"]["c_proj"]
            out = out @ c_proj["kernel"] + c_proj["bias"]
            x = x + out

            # MLP
            mlp_in = _layernorm(
                x,
                block["ln_2"]["scale"],
                block["ln_2"]["bias"],
            )
            c_fc = block["mlp"]["c_fc"]
            c_mlp_proj = block["mlp"]["c_proj"]
            h = mlp_in @ c_fc["kernel"] + c_fc["bias"]
            h = _gelu_new(h)
            h = h @ c_mlp_proj["kernel"] + c_mlp_proj["bias"]
            x = x + h

        # Final LN and token projection (tied with wte)
        x = _layernorm(
            x,
            params["ln_f"]["scale"],
            params["ln_f"]["bias"],
        )
        logits = x @ wte_E.T                                      # (B, 1, V)
        return k_cache, v_cache, logits[:, 0, :]

    def rollout(params, rng_key, beta):
        """Generate ``total_len`` tokens starting from BOS.

        Uses a scan over ``total_len - 1`` decode steps, each consuming the
        previous token and emitting the next one.
        """
        dtype = params["wte"]["embedding"].dtype
        k_cache = jnp.zeros(
            (n_layer, batch_size, n_head, total_len, head_dim), dtype=dtype
        )
        v_cache = jnp.zeros_like(k_cache)

        init_token = jnp.full(
            (batch_size,), bos_token_id, dtype=jnp.int32
        )

        # Produce (total_len - 1) tokens after the BOS. At step t we consume
        # token at position t and emit token at position t+1.
        n_steps = total_len - 1

        def body(carry, step_idx):
            k_cache, v_cache, prev_token, rng = carry
            rng, sample_key = jax.random.split(rng)
            k_cache, v_cache, logits = decode_step(
                params, k_cache, v_cache, prev_token, step_idx
            )
            # Mask BOS so it cannot be sampled.
            logits = logits.at[:, bos_token_id].set(
                jnp.asarray(jnp.inf, dtype=logits.dtype)
            )
            idx_next = jax.random.categorical(
                sample_key, -beta * logits, axis=-1
            ).astype(jnp.int32)
            return (k_cache, v_cache, idx_next, rng), idx_next

        (_, _, _, _), tokens_out = jax.lax.scan(
            body,
            (k_cache, v_cache, init_token, rng_key),
            jnp.arange(n_steps, dtype=jnp.int32),
        )
        # tokens_out: (n_steps, B) -> (B, n_steps); prepend BOS column.
        tokens_out = tokens_out.T
        bos_col = jnp.full((batch_size, 1), bos_token_id, dtype=jnp.int32)
        return jnp.concatenate([bos_col, tokens_out], axis=1)

    return jax.jit(rollout)
