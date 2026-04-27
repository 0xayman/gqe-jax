"""Clipped-surrogate PPO objective for hybrid (discrete + continuous) actions.

Per-position log-prob of the joint action is::

    log p(a_t | s_t) = log p(token_t | s_t) + 1[token_t is parametric] · log p(angle_t | s_t)

That is, the angle log-prob only counts when the sampled token actually uses
its angle (rotation gates RX / RY / RZ). For non-parametric tokens (SX, CNOT,
STOP, padding) the angle is unused so its log-prob would be a free parameter
that the policy could push around without affecting the reward — masking it
out prevents the importance ratio from drifting on irrelevant noise.

Sequence log-prob is the length-normalised mean over valid positions
(positions ``< length``; padding contributes nothing). The PPO ratio and the
clipped surrogate are then a standard PPO step.
"""

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
    """Mean over valid positions (zero-mass on invalid; safe for length-0 rows)."""
    masked = jnp.where(valid_mask, per_token, jnp.asarray(0.0, dtype=per_token.dtype))
    denom = jnp.maximum(valid_mask.sum(axis=-1).astype(per_token.dtype), 1.0)
    return jnp.sum(masked, axis=-1) / denom


def joint_sequence_log_prob(
    log_p_disc: jax.Array,
    log_p_cont: jax.Array,
    lengths: jax.Array,
    parametric_mask: jax.Array,
) -> jax.Array:
    """Length-normalised joint log-prob per sequence (discrete + masked continuous).

    All inputs share shape ``(B, max_gates)`` aside from ``lengths`` which is
    ``(B,)``. ``parametric_mask`` is True at positions whose sampled token is a
    parametric rotation.
    """
    valid = _length_mask(lengths, log_p_disc.shape[-1])
    cont_contrib = jnp.where(
        parametric_mask & valid,
        log_p_cont,
        jnp.asarray(0.0, dtype=log_p_cont.dtype),
    )
    per_token = log_p_disc + cont_contrib
    return reduce_per_token(per_token, valid)


def grpo_advantages(costs: jax.Array) -> jax.Array:
    """Group-relative-policy-optimisation advantage (mean-baseline, std-normalised)."""
    mean = jnp.mean(costs)
    std = jnp.where(costs.size > 1, jnp.std(costs, ddof=1), jnp.asarray(0.0, dtype=costs.dtype))
    return (mean - costs) / (std + 1e-8)


def ppo_clipped_loss(
    new_seq_log_p: jax.Array,
    old_seq_log_p: jax.Array,
    advantages: jax.Array,
    clip_ratio: float,
) -> jax.Array:
    """Standard PPO clipped surrogate loss (mean over the batch)."""
    ratio = jnp.exp(new_seq_log_p - old_seq_log_p)
    clipped = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = jnp.minimum(ratio * advantages, clipped * advantages)
    return -jnp.mean(surrogate)


def gaussian_entropy(log_sigma: jax.Array, valid_mask: jax.Array) -> jax.Array:
    """Mean per-position entropy of the Gaussian over valid positions."""
    # H(Normal(mu, sigma)) = 0.5 * log(2 pi e sigma^2) = log_sigma + 0.5 * log(2 pi e)
    h = log_sigma + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    return reduce_per_token(h, valid_mask).mean()


def categorical_entropy(
    discrete_logits: jax.Array,
    beta: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """Mean entropy of ``softmax(-beta * logits)`` over valid positions.

    The ``0 * log(0) = 0`` convention is needed because action-mask slots have
    probability zero. We make ``log_p`` safe BEFORE the multiplication — JAX
    autograd evaluates both branches of ``where``, so the naive
    ``where(p > 0, p * log_p, 0)`` would still propagate ``nan`` gradients
    through the masked ``p * (-inf)`` branch.
    """
    p = jax.nn.softmax(-beta * discrete_logits, axis=-1)
    log_p = jax.nn.log_softmax(-beta * discrete_logits, axis=-1)
    safe_log_p = jnp.where(p > 0, log_p, jnp.asarray(0.0, dtype=log_p.dtype))
    h_per = -jnp.sum(p * safe_log_p, axis=-1)
    return reduce_per_token(h_per, valid_mask).mean()
