"""Hybrid-action policy: GPT-2 backbone + discrete (token) and continuous
(angle) heads, plus KV-cached rollout and log-probability evaluation.

At each sequence position the policy emits:
  - a discrete action: a vocab token id (BOS / STOP / one of the pool gates)
  - a continuous action: a real-valued rotation angle (used iff the discrete
    token denotes a parametric rotation, ignored otherwise — but always
    sampled so the action shape stays regular)

The continuous head shares the transformer backbone with the discrete head and
predicts ``(mu, log_sigma)`` per position. The angle is sampled from
``Normal(mu, exp(log_sigma))``. Sampling is unbounded — rotation angles are
2π-periodic so there is no need to squash to a finite interval, and using a
plain Gaussian keeps the log-prob exact (no Tanh-correction) and the
PPO-importance ratio numerically stable.

Vocabulary layout: ``[BOS=0, STOP=1, gate_2, ..., gate_{V-1}]``.

  - BOS sits at position 0 of every sequence and is unsamplable elsewhere.
  - STOP is unsamplable at the very first decision (so circuits have ≥ 1 gate)
    and is forced to fill all positions after a sample's first STOP.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import jax_setup  # noqa: F401

# ── Architecture sizes ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ArchConfig:
    n_layer: int
    n_head: int
    n_embd: int


_SIZES: dict[str, _ArchConfig] = {
    # "tiny":   _ArchConfig(n_layer=2,  n_head=2,  n_embd=128),
    "tiny": _ArchConfig(n_layer=2, n_head=2, n_embd=384),
    "small": _ArchConfig(n_layer=6, n_head=6, n_embd=384),
    "medium": _ArchConfig(n_layer=12, n_head=12, n_embd=768),
    "large": _ArchConfig(n_layer=24, n_head=16, n_embd=1024),
}

VALID_MODEL_SIZES = frozenset(_SIZES)
_INIT_STDDEV = 0.02
_LAYER_NORM_EPS = 1.0e-5
_N_POSITIONS = 1024
_DROPOUT_RATE = 0.1
# Bounds on the continuous-head log-std. Lower bound prevents collapse to
# delta-policy; upper bound prevents log-prob from blowing up early.
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _gelu_new(x: jax.Array) -> jax.Array:
    coeff = jnp.sqrt(2.0 / jnp.pi)
    return 0.5 * x * (1.0 + jnp.tanh(coeff * (x + 0.044715 * (x**3))))


def _init():
    return nn.initializers.normal(stddev=_INIT_STDDEV)


# ── Transformer blocks ───────────────────────────────────────────────────────


class _CausalSelfAttention(nn.Module):
    n_head: int
    n_embd: int
    pdrop: float = _DROPOUT_RATE

    @nn.compact
    def __call__(self, x, mask, deterministic):
        B, T, _ = x.shape
        head_dim = self.n_embd // self.n_head
        qkv = nn.Dense(3 * self.n_embd, kernel_init=_init(), name="c_attn")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        scale = jnp.asarray(head_dim, dtype=x.dtype) ** -0.5
        scores = jnp.einsum("bnth,bnsh->bnts", q, k) * scale

        causal = jnp.tril(jnp.ones((T, T), dtype=bool))
        key_mask = jnp.ones((B, T), dtype=bool) if mask is None else mask.astype(bool)
        full_mask = causal[None, None, :, :] & key_mask[:, None, None, :]
        bias = jnp.where(
            full_mask,
            jnp.asarray(0.0, dtype=x.dtype),
            jnp.asarray(jnp.finfo(x.dtype).min, dtype=x.dtype),
        )
        attn = jax.nn.softmax(scores + bias, axis=-1)
        attn = nn.Dropout(rate=self.pdrop)(attn, deterministic=deterministic)
        out = jnp.einsum("bnts,bnsh->bnth", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_embd)
        out = nn.Dense(self.n_embd, kernel_init=_init(), name="c_proj")(out)
        return nn.Dropout(rate=self.pdrop)(out, deterministic=deterministic)


class _MLP(nn.Module):
    n_embd: int
    pdrop: float = _DROPOUT_RATE

    @nn.compact
    def __call__(self, x, deterministic):
        h = nn.Dense(4 * self.n_embd, kernel_init=_init(), name="c_fc")(x)
        h = _gelu_new(h)
        h = nn.Dense(self.n_embd, kernel_init=_init(), name="c_proj")(h)
        return nn.Dropout(rate=self.pdrop)(h, deterministic=deterministic)


class _Block(nn.Module):
    n_head: int
    n_embd: int

    @nn.compact
    def __call__(self, x, mask, deterministic):
        attn_in = nn.LayerNorm(epsilon=_LAYER_NORM_EPS, name="ln_1")(x)
        x = x + _CausalSelfAttention(
            n_head=self.n_head,
            n_embd=self.n_embd,
            name="attn",
        )(attn_in, mask, deterministic)
        mlp_in = nn.LayerNorm(epsilon=_LAYER_NORM_EPS, name="ln_2")(x)
        x = x + _MLP(n_embd=self.n_embd, name="mlp")(mlp_in, deterministic)
        return x


class HybridPolicy(nn.Module):
    """GPT-2 with a discrete (token) and a continuous (angle) head.

    Returns ``(discrete_logits, mu, log_sigma)``:
      - ``discrete_logits``: ``(B, T, V)``  raw logits over the vocab.
      - ``mu``:              ``(B, T)``     mean of the angle Gaussian.
      - ``log_sigma``:       ``(B, T)``     log-std, clipped to a safe range.
    """

    size: str
    vocab_size: int
    n_positions: int = _N_POSITIONS

    @nn.compact
    def __call__(self, input_ids, attention_mask=None, deterministic=False):
        if self.size not in _SIZES:
            raise ValueError(f"Unknown model size {self.size!r}")
        arch = _SIZES[self.size]
        T = input_ids.shape[1]
        if T > self.n_positions:
            raise ValueError(f"Sequence length {T} exceeds max {self.n_positions}")

        wte = nn.Embed(self.vocab_size, arch.n_embd, embedding_init=_init(), name="wte")
        wpe = nn.Embed(
            self.n_positions, arch.n_embd, embedding_init=_init(), name="wpe"
        )
        x = wte(input_ids) + wpe(jnp.arange(T, dtype=jnp.int32)[None, :])
        x = nn.Dropout(rate=_DROPOUT_RATE)(x, deterministic=deterministic)

        for i in range(arch.n_layer):
            x = _Block(n_head=arch.n_head, n_embd=arch.n_embd, name=f"h_{i}")(
                x,
                attention_mask,
                deterministic,
            )

        x = nn.LayerNorm(epsilon=_LAYER_NORM_EPS, name="ln_f")(x)
        discrete_logits = wte.attend(x)  # tied weights

        # Continuous head: small MLP -> (mu, log_sigma) per position.
        h = nn.Dense(arch.n_embd, kernel_init=_init(), name="cont_fc")(x)
        h = _gelu_new(h)
        params = nn.Dense(2, kernel_init=_init(), name="cont_out")(h)
        mu = params[..., 0]
        log_sigma = jnp.clip(params[..., 1], LOG_STD_MIN, LOG_STD_MAX)
        return discrete_logits, mu, log_sigma


# ── Helpers shared by training-mode and rollout-mode evaluation ──────────────


def gaussian_log_prob(
    value: jax.Array, mu: jax.Array, log_sigma: jax.Array
) -> jax.Array:
    """Per-element log-probability of a univariate Normal."""
    var = jnp.exp(2.0 * log_sigma)
    return -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * log_sigma + (value - mu) ** 2 / var)


def apply_discrete_action_masks(
    logits: jax.Array,
    bos_token_id: int,
    stop_token_id: int,
) -> jax.Array:
    """Make BOS unsamplable everywhere; STOP unsamplable at position 0.

    Sampling uses ``softmax(-beta * logits)``, so a logit of ``+inf`` becomes
    ``-inf`` after the sign flip and is fully suppressed.
    """
    pos_inf = jnp.asarray(jnp.inf, dtype=logits.dtype)
    logits = logits.at[..., bos_token_id].set(pos_inf)
    seq_len = logits.shape[-2]
    pos = jnp.arange(seq_len, dtype=jnp.int32)
    stop_bias = jnp.where(pos == 0, pos_inf, jnp.asarray(0.0, dtype=logits.dtype))
    broadcast_shape = (1,) * (logits.ndim - 2) + (seq_len,)
    logits = logits.at[..., stop_token_id].add(stop_bias.reshape(broadcast_shape))
    return logits


def reduce_sequence_log_probs(
    token_log_probs: jax.Array,
    lengths: jax.Array,
) -> jax.Array:
    """Length-normalised mean of per-token log-probs (zero-mass beyond length)."""
    if token_log_probs.ndim <= 1:
        return token_log_probs
    max_len = token_log_probs.shape[-1]
    pos = jnp.arange(max_len, dtype=lengths.dtype)
    mask = pos[None, :] < lengths[:, None]
    masked = jnp.where(
        mask, token_log_probs, jnp.asarray(0.0, dtype=token_log_probs.dtype)
    )
    denom = jnp.maximum(lengths.astype(token_log_probs.dtype), 1.0)
    return jnp.sum(masked, axis=-1) / denom


def compute_lengths_from_tokens(token_ids: jax.Array, stop_token_id: int) -> jax.Array:
    """Position of first STOP + 1 (or full length if no STOP), for each row."""
    b, n = token_ids.shape
    is_stop = (token_ids == stop_token_id).astype(jnp.int32)
    padded = jnp.concatenate([is_stop, jnp.ones((b, 1), dtype=jnp.int32)], axis=1)
    first_stop = jnp.argmax(padded, axis=1)
    return jnp.minimum(first_stop + 1, n).astype(jnp.int32)


# ── KV-cache rollout ─────────────────────────────────────────────────────────
#
# The training-mode forward (``HybridPolicy.__call__``) processes the full
# sequence in one pass — fine for loss evaluation, O(N^2) for generation. For
# rollout we want O(N) by caching per-layer (K, V). We read weights directly
# from the Flax params dict; layout matches ``HybridPolicy`` (and
# ``model.GPT2`` historically).


def _layernorm(x, scale, bias):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + _LAYER_NORM_EPS)
    return (x - mean) * inv * scale + bias


def build_rollout_fn(
    *,
    n_layer: int,
    n_head: int,
    n_embd: int,
    vocab_size: int,
    batch_size: int,
    total_len: int,
    bos_token_id: int = 0,
    stop_token_id: int = 1,
):
    """Return a JIT'd rollout function for the hybrid policy.

    The returned callable has signature ``(params, rng_key, beta) -> dict`` with
    keys:
        ``tokens``    (B, total_len)              — int32, BOS at position 0,
                                                    STOP padding after first STOP
        ``angles``    (B, max_gates=total_len-1)  — float, sampled angles per step
        ``log_p_d``   (B, max_gates)              — discrete per-step log-prob
        ``log_p_c``   (B, max_gates)              — continuous per-step log-prob

    ``beta`` is the discrete inverse-temperature; the continuous head is
    sampled from its native ``Normal(mu, exp(log_sigma))`` regardless.
    """
    head_dim = n_embd // n_head
    assert head_dim * n_head == n_embd
    n_steps = total_len - 1  # tokens after BOS

    def decode_step(params, k_cache, v_cache, prev_token, step_idx):
        wte = params["wte"]["embedding"]
        wpe = params["wpe"]["embedding"]
        x = (wte[prev_token] + wpe[step_idx])[:, None, :]  # (B, 1, D)
        positions = jnp.arange(total_len, dtype=jnp.int32)
        key_mask = positions <= step_idx  # (T,)

        for i in range(n_layer):
            block = params[f"h_{i}"]
            attn_in = _layernorm(x, block["ln_1"]["scale"], block["ln_1"]["bias"])

            ca = block["attn"]["c_attn"]
            qkv = attn_in @ ca["kernel"] + ca["bias"]
            q, k, v = jnp.split(qkv, 3, axis=-1)
            q = q.reshape(batch_size, 1, n_head, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, 1, n_head, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, 1, n_head, head_dim).transpose(0, 2, 1, 3)

            k_c = jax.lax.dynamic_update_slice_in_dim(k_cache[i], k, step_idx, axis=2)
            v_c = jax.lax.dynamic_update_slice_in_dim(v_cache[i], v, step_idx, axis=2)
            k_cache = k_cache.at[i].set(k_c)
            v_cache = v_cache.at[i].set(v_c)

            scale = jnp.asarray(head_dim, dtype=x.dtype) ** -0.5
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_c) * scale
            min_val = jnp.finfo(scores.dtype).min
            scores = jnp.where(
                key_mask[None, None, None, :],
                scores,
                jnp.asarray(min_val, dtype=scores.dtype),
            )
            attn = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v_c)
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, 1, n_embd)

            cp = block["attn"]["c_proj"]
            x = x + (out @ cp["kernel"] + cp["bias"])

            mlp_in = _layernorm(x, block["ln_2"]["scale"], block["ln_2"]["bias"])
            cf = block["mlp"]["c_fc"]
            cmp_ = block["mlp"]["c_proj"]
            h = mlp_in @ cf["kernel"] + cf["bias"]
            h = _gelu_new(h)
            x = x + (h @ cmp_["kernel"] + cmp_["bias"])

        x = _layernorm(x, params["ln_f"]["scale"], params["ln_f"]["bias"])

        # Discrete head (tied with wte).
        discrete_logits = (x @ wte.T)[:, 0, :]  # (B, V)

        # Continuous head: 2-layer MLP -> (mu, log_sigma).
        cont_fc = params["cont_fc"]
        cont_out = params["cont_out"]
        h = x @ cont_fc["kernel"] + cont_fc["bias"]
        h = _gelu_new(h)
        h = h @ cont_out["kernel"] + cont_out["bias"]  # (B, 1, 2)
        mu = h[:, 0, 0]
        log_sigma = jnp.clip(h[:, 0, 1], LOG_STD_MIN, LOG_STD_MAX)
        return k_cache, v_cache, discrete_logits, mu, log_sigma

    def rollout(params, rng_key, beta):
        dtype = params["wte"]["embedding"].dtype
        k_cache = jnp.zeros(
            (n_layer, batch_size, n_head, total_len, head_dim), dtype=dtype
        )
        v_cache = jnp.zeros_like(k_cache)
        init_token = jnp.full((batch_size,), bos_token_id, dtype=jnp.int32)
        init_stopped = jnp.zeros((batch_size,), dtype=bool)

        def body(carry, step_idx):
            k_c, v_c, prev_token, stopped, rng = carry
            rng, k_disc, k_cont = jax.random.split(rng, 3)
            k_c, v_c, logits, mu, log_sigma = decode_step(
                params,
                k_c,
                v_c,
                prev_token,
                step_idx,
            )
            # Mask BOS and (at first step) STOP. Convention: sampling uses
            # softmax(-beta * logits), so +inf suppresses.
            pos_inf = jnp.asarray(jnp.inf, dtype=logits.dtype)
            logits = logits.at[:, bos_token_id].set(pos_inf)
            stop_bias = jnp.where(
                step_idx == 0, pos_inf, jnp.asarray(0.0, dtype=logits.dtype)
            )
            logits = logits.at[:, stop_token_id].add(stop_bias)

            # Discrete sample under temperature ``beta``: p ∝ softmax(-beta * logits).
            scaled = -beta * logits
            sampled = jax.random.categorical(k_disc, scaled, axis=-1).astype(jnp.int32)
            tok_next = jnp.where(stopped, stop_token_id, sampled)
            new_stopped = stopped | (tok_next == stop_token_id)

            # Discrete log-prob under the same scaled distribution.
            log_p_disc_all = jax.nn.log_softmax(scaled, axis=-1)
            log_p_disc = jnp.take_along_axis(
                log_p_disc_all,
                tok_next[:, None],
                axis=-1,
            ).squeeze(-1)

            # Continuous sample from N(mu, exp(log_sigma)).
            sigma = jnp.exp(log_sigma)
            noise = jax.random.normal(k_cont, shape=mu.shape, dtype=mu.dtype)
            angle_next = mu + sigma * noise
            log_p_cont = gaussian_log_prob(angle_next, mu, log_sigma)

            return (k_c, v_c, tok_next, new_stopped, rng), (
                tok_next,
                angle_next,
                log_p_disc,
                log_p_cont,
            )

        (_, _, _, _, _), (toks, angles, log_p_d, log_p_c) = jax.lax.scan(
            body,
            (k_cache, v_cache, init_token, init_stopped, rng_key),
            jnp.arange(n_steps, dtype=jnp.int32),
        )
        # scan stacks along leading axis -> transpose to (B, T).
        toks = toks.T
        angles = angles.T
        log_p_d = log_p_d.T
        log_p_c = log_p_c.T

        bos_col = jnp.full((batch_size, 1), bos_token_id, dtype=jnp.int32)
        full_tokens = jnp.concatenate([bos_col, toks], axis=1)
        return {
            "tokens": full_tokens,
            "angles": angles,
            "log_p_d": log_p_d,
            "log_p_c": log_p_c,
        }

    return jax.jit(rollout)


# ── Training-mode log-prob recomputation ─────────────────────────────────────
#
# Given a stored (tokens, angles) batch, recompute the *current* policy's
# discrete + continuous per-token log-probs. Used by the PPO loss.


def hybrid_log_probs(
    discrete_logits: jax.Array,
    mu: jax.Array,
    log_sigma: jax.Array,
    actions_disc: jax.Array,
    actions_cont: jax.Array,
    *,
    beta: jax.Array,
    bos_token_id: int,
    stop_token_id: int,
):
    """Per-position discrete + continuous log-probs for the recorded actions.

    Inputs are aligned to the *output* positions (i.e. excluding the BOS prefix).
    Returns ``(log_p_disc, log_p_cont)`` each with shape ``(B, max_gates)``.
    """
    seq_len = actions_disc.shape[1]
    aligned_logits = apply_discrete_action_masks(
        discrete_logits[:, :seq_len, :],
        bos_token_id=bos_token_id,
        stop_token_id=stop_token_id,
    )
    log_p_all = jax.nn.log_softmax(-beta * aligned_logits, axis=-1)
    log_p_disc = jnp.take_along_axis(
        log_p_all,
        actions_disc[..., None],
        axis=-1,
    ).squeeze(-1)

    mu_a = mu[:, :seq_len]
    log_sigma_a = log_sigma[:, :seq_len]
    log_p_cont = gaussian_log_prob(actions_cont, mu_a, log_sigma_a)
    return log_p_disc, log_p_cont
