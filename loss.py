"""Loss helpers for hybrid discrete/continuous PPO updates."""

from __future__ import annotations

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp


def _length_mask(lengths: jax.Array, max_len: int) -> jax.Array:
    pos = jnp.arange(max_len, dtype=lengths.dtype)
    return pos[None, :] < lengths[:, None]


def reduce_per_token(
    per_token: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """Average per-position values over valid sequence positions."""
    masked = jnp.where(valid_mask, per_token, jnp.asarray(0.0, dtype=per_token.dtype))
    denom = jnp.maximum(valid_mask.sum(axis=-1).astype(per_token.dtype), 1.0)
    return jnp.sum(masked, axis=-1) / denom


def joint_sequence_log_prob(
    log_p_disc: jax.Array,
    log_p_cont: jax.Array,
    lengths: jax.Array,
    parametric_mask: jax.Array,
) -> jax.Array:
    """Joint sequence log-prob with angle terms only on rotation tokens."""
    valid = _length_mask(lengths, log_p_disc.shape[-1])
    cont_contrib = jnp.where(
        parametric_mask & valid,
        log_p_cont,
        jnp.asarray(0.0, dtype=log_p_cont.dtype),
    )
    per_token = log_p_disc + cont_contrib
    return reduce_per_token(per_token, valid)


def grpo_advantages(costs: jax.Array) -> jax.Array:
    """Return mean-centered, std-normalized advantages for one rollout group."""
    mean = jnp.mean(costs)
    std = jnp.where(costs.size > 1, jnp.std(costs, ddof=1), jnp.asarray(0.0, dtype=costs.dtype))
    return (mean - costs) / (std + 1e-8)


def ppo_clipped_loss(
    new_seq_log_p: jax.Array,
    old_seq_log_p: jax.Array,
    advantages: jax.Array,
    clip_ratio: float,
) -> jax.Array:
    """Clipped PPO surrogate loss averaged over the minibatch."""
    ratio = jnp.exp(new_seq_log_p - old_seq_log_p)
    clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = jnp.minimum(ratio * advantages, clipped * advantages)
    return -jnp.mean(surrogate)


def gaussian_entropy(log_sigma: jax.Array, valid_mask: jax.Array) -> jax.Array:
    """Mean Gaussian entropy over the supplied mask."""
    h = log_sigma + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    return reduce_per_token(h, valid_mask).mean()


def categorical_entropy(
    discrete_logits: jax.Array,
    beta: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """Mean categorical entropy over valid positions."""
    p = jax.nn.softmax(-beta * discrete_logits, axis=-1)
    log_p = jax.nn.log_softmax(-beta * discrete_logits, axis=-1)
    safe_log_p = jnp.where(p > 0, log_p, jnp.asarray(0.0, dtype=log_p.dtype))
    h_per = -jnp.sum(p * safe_log_p, axis=-1)
    return reduce_per_token(h_per, valid_mask).mean()
