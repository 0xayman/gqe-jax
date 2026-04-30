"""Target-conditioned hybrid policy and rollout-time sampling utilities.

The model predicts one gate token and one angle distribution per sequence
position. Angle likelihoods use a wrapped normal because circuit rotations are
2*pi-periodic. STOP is a normal action after the first gate; once every row in
the rollout batch has stopped, generation exits before reaching the safety cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import jax_setup  # noqa: F401


@dataclass(frozen=True)
class _ArchConfig:
    n_layer: int
    n_head: int
    n_embd: int


_SIZES: dict[str, _ArchConfig] = {
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
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _gelu_new(x: jax.Array) -> jax.Array:
    coeff = jnp.sqrt(2.0 / jnp.pi)
    return 0.5 * x * (1.0 + jnp.tanh(coeff * (x + 0.044715 * (x**3))))


def _init():
    return nn.initializers.normal(stddev=_INIT_STDDEV)


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
    """Causal transformer with tied token logits and wrapped-angle parameters."""

    size: str
    vocab_size: int
    n_positions: int = _N_POSITIONS

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        target_features=None,
        deterministic=False,
    ):
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
        if target_features is not None:
            target_cond = nn.Dense(
                arch.n_embd,
                kernel_init=_init(),
                name="target_proj",
            )(target_features)
            x = x + target_cond[:, None, :]
        x = nn.Dropout(rate=_DROPOUT_RATE)(x, deterministic=deterministic)

        for i in range(arch.n_layer):
            x = _Block(n_head=arch.n_head, n_embd=arch.n_embd, name=f"h_{i}")(
                x,
                attention_mask,
                deterministic,
            )

        x = nn.LayerNorm(epsilon=_LAYER_NORM_EPS, name="ln_f")(x)
        discrete_logits = wte.attend(x)

        h = nn.Dense(arch.n_embd, kernel_init=_init(), name="cont_fc")(x)
        h = _gelu_new(h)
        params = nn.Dense(2, kernel_init=_init(), name="cont_out")(h)
        mu = params[..., 0]
        log_sigma = jnp.clip(params[..., 1], LOG_STD_MIN, LOG_STD_MAX)
        return discrete_logits, mu, log_sigma


def gaussian_log_prob(
    value: jax.Array, mu: jax.Array, log_sigma: jax.Array
) -> jax.Array:
    """Per-element log-probability of a univariate Normal."""
    var = jnp.exp(2.0 * log_sigma)
    return -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * log_sigma + (value - mu) ** 2 / var)


def canonicalize_angle(value: jax.Array) -> jax.Array:
    """Map angles to the canonical interval [-pi, pi)."""
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=value.dtype)
    pi = jnp.asarray(jnp.pi, dtype=value.dtype)
    return jnp.mod(value + pi, two_pi) - pi


def periodic_gaussian_log_prob(
    value: jax.Array,
    mu: jax.Array,
    log_sigma: jax.Array,
    *,
    wrap_radius: int = 2,
) -> jax.Array:
    """Wrapped-normal log-probability for 2*pi-periodic angle actions."""
    value = canonicalize_angle(value)
    offsets = jnp.arange(
        -wrap_radius, wrap_radius + 1, dtype=value.dtype,
    ) * jnp.asarray(2.0 * jnp.pi, dtype=value.dtype)
    terms = gaussian_log_prob(value[..., None] + offsets, mu[..., None], log_sigma[..., None])
    return jax.nn.logsumexp(terms, axis=-1)


def apply_discrete_action_masks(
    logits: jax.Array,
    bos_token_id: int,
    stop_token_id: int,
) -> jax.Array:
    """Mask token logits before applying the ``softmax(-beta * logits)`` policy."""
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
    """Average per-token log-probs up to each row's STOP-derived length."""
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
    use_target_conditioning: bool = False,
):
    """Build a JIT sampler that shares parameters with ``HybridPolicy``."""
    head_dim = n_embd // n_head
    assert head_dim * n_head == n_embd
    n_steps = total_len - 1

    def decode_step(params, k_cache, v_cache, prev_token, step_idx, target_features):
        wte = params["wte"]["embedding"]
        wpe = params["wpe"]["embedding"]
        x = (wte[prev_token] + wpe[step_idx])[:, None, :]
        if use_target_conditioning:
            tp = params["target_proj"]
            target_cond = target_features @ tp["kernel"] + tp["bias"]
            x = x + target_cond[:, None, :]
        positions = jnp.arange(total_len, dtype=jnp.int32)
        key_mask = positions <= step_idx

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

        discrete_logits = (x @ wte.T)[:, 0, :]

        cont_fc = params["cont_fc"]
        cont_out = params["cont_out"]
        h = x @ cont_fc["kernel"] + cont_fc["bias"]
        h = _gelu_new(h)
        h = h @ cont_out["kernel"] + cont_out["bias"]
        mu = h[:, 0, 0]
        log_sigma = jnp.clip(h[:, 0, 1], LOG_STD_MIN, LOG_STD_MAX)
        return k_cache, v_cache, discrete_logits, mu, log_sigma

    def rollout(params, rng_key, beta, target_features=None):
        dtype = params["wte"]["embedding"].dtype
        k_cache = jnp.zeros(
            (n_layer, batch_size, n_head, total_len, head_dim), dtype=dtype
        )
        v_cache = jnp.zeros_like(k_cache)
        init_token = jnp.full((batch_size,), bos_token_id, dtype=jnp.int32)
        init_stopped = jnp.zeros((batch_size,), dtype=bool)
        toks = jnp.full((batch_size, n_steps), stop_token_id, dtype=jnp.int32)
        angles = jnp.zeros((batch_size, n_steps), dtype=dtype)
        log_p_d = jnp.zeros((batch_size, n_steps), dtype=dtype)
        log_p_c = jnp.zeros((batch_size, n_steps), dtype=dtype)

        def cond_fun(carry):
            step_idx, _k_c, _v_c, _prev_token, stopped, _rng, _t, _a, _lpd, _lpc = carry
            return (step_idx < n_steps) & (~jnp.all(stopped))

        def body(carry):
            step_idx, k_c, v_c, prev_token, stopped, rng, toks_c, angles_c, lpd_c, lpc_c = carry
            rng, k_disc, k_cont = jax.random.split(rng, 3)
            k_c, v_c, logits, mu, log_sigma = decode_step(
                params,
                k_c,
                v_c,
                prev_token,
                step_idx,
                target_features,
            )
            pos_inf = jnp.asarray(jnp.inf, dtype=logits.dtype)
            logits = logits.at[:, bos_token_id].set(pos_inf)
            stop_bias = jnp.where(
                step_idx == 0, pos_inf, jnp.asarray(0.0, dtype=logits.dtype)
            )
            logits = logits.at[:, stop_token_id].add(stop_bias)

            scaled = -beta * logits
            sampled = jax.random.categorical(k_disc, scaled, axis=-1).astype(jnp.int32)
            tok_next = jnp.where(stopped, stop_token_id, sampled)
            new_stopped = stopped | (tok_next == stop_token_id)

            log_p_disc_all = jax.nn.log_softmax(scaled, axis=-1)
            log_p_disc = jnp.take_along_axis(
                log_p_disc_all,
                tok_next[:, None],
                axis=-1,
            ).squeeze(-1)

            sigma = jnp.exp(log_sigma)
            noise = jax.random.normal(k_cont, shape=mu.shape, dtype=mu.dtype)
            angle_next = canonicalize_angle(mu + sigma * noise)
            log_p_cont = periodic_gaussian_log_prob(angle_next, mu, log_sigma)

            toks_c = toks_c.at[:, step_idx].set(tok_next)
            angles_c = angles_c.at[:, step_idx].set(angle_next)
            lpd_c = lpd_c.at[:, step_idx].set(log_p_disc)
            lpc_c = lpc_c.at[:, step_idx].set(log_p_cont)

            return (
                step_idx + 1,
                k_c,
                v_c,
                tok_next,
                new_stopped,
                rng,
                toks_c,
                angles_c,
                lpd_c,
                lpc_c,
            )

        init_carry = (
            jnp.asarray(0, dtype=jnp.int32),
            k_cache,
            v_cache,
            init_token,
            init_stopped,
            rng_key,
            toks,
            angles,
            log_p_d,
            log_p_c,
        )
        (_, _, _, _, _, _, toks, angles, log_p_d, log_p_c) = jax.lax.while_loop(
            cond_fun,
            body,
            init_carry,
        )

        bos_col = jnp.full((batch_size, 1), bos_token_id, dtype=jnp.int32)
        full_tokens = jnp.concatenate([bos_col, toks], axis=1)
        return {
            "tokens": full_tokens,
            "angles": angles,
            "log_p_d": log_p_d,
            "log_p_c": log_p_c,
        }

    return jax.jit(rollout)


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
    """Recompute per-position action log-probs for a stored rollout batch."""
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
    log_p_cont = periodic_gaussian_log_prob(actions_cont, mu_a, log_sigma_a)
    return log_p_disc, log_p_cont
