"""GRPO-style loss helpers for JAX-based GQE training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from cost import _unbiased_std


def reduce_sequence_log_probs(
    token_log_probs: jax.Array,
    lengths: jax.Array | None = None,
) -> jax.Array:
    """Return a length-normalized sequence score from per-token log-probabilities.

    Using a mean instead of a sum keeps GRPO importance ratios comparable across
    different circuit lengths. With variable-length sampling, ``lengths`` gives
    the number of real decisions per sample (STOP inclusive); positions beyond
    that are forced-STOP padding and contribute nothing to the gradient.
    """
    if token_log_probs.ndim <= 1:
        return token_log_probs
    if lengths is None:
        return jnp.mean(token_log_probs, axis=-1)
    max_len = token_log_probs.shape[-1]
    positions = jnp.arange(max_len, dtype=lengths.dtype)
    mask = positions[None, :] < lengths[:, None]
    masked = jnp.where(mask, token_log_probs, jnp.asarray(0.0, dtype=token_log_probs.dtype))
    denom = jnp.maximum(lengths.astype(token_log_probs.dtype), 1.0)
    return jnp.sum(masked, axis=-1) / denom


def apply_action_masks(
    logits: jax.Array,
    bos_token_id: int,
    stop_token_id: int | None = None,
) -> jax.Array:
    """Apply action masks to a ``(..., seq_len, vocab)`` logits tensor.

    BOS is unsamplable at every position. STOP is unsamplable only at sequence
    position 0 (the first post-BOS decision), matching the sampling policy that
    forbids 0-gate circuits.

    Convention: sampling uses ``softmax(-beta * logits)``, so ``+inf`` suppresses.
    """
    logits = logits.at[..., bos_token_id].set(
        jnp.asarray(jnp.inf, dtype=logits.dtype)
    )
    if stop_token_id is not None:
        seq_len = logits.shape[-2]
        positions = jnp.arange(seq_len, dtype=jnp.int32)
        stop_bias = jnp.where(
            positions == 0,
            jnp.asarray(jnp.inf, dtype=logits.dtype),
            jnp.asarray(0.0, dtype=logits.dtype),
        )
        # Broadcast across batch/leading axes on the seq axis.
        broadcast_shape = (1,) * (logits.ndim - 2) + (seq_len,)
        stop_bias = stop_bias.reshape(broadcast_shape)
        logits = logits.at[..., stop_token_id].add(stop_bias)
    return logits


class Loss(ABC):
    @abstractmethod
    def compute(
        self,
        costs,
        gate_logits,
        gate_indices,
        log_values=None,
        **kwargs,
    ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array], bool]:
        """Return loss and optional GRPO reference tensors."""


class GRPOLoss(Loss):
    """Clipped GRPO objective using a frozen reference policy."""

    def __init__(self, clip_ratio: float = 0.2):
        self.clip_ratio = clip_ratio

    def compute(
        self,
        costs,
        gate_logits,
        gate_indices,
        log_values=None,
        **kwargs,
    ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array], bool]:
        bos_token_id = kwargs.get("bos_token_id", 0)
        stop_token_id = kwargs.get("stop_token_id")
        lengths = kwargs.get("lengths")
        current_log_probs = self.log_prob(
            gate_indices,
            gate_logits,
            kwargs["inverse_temperature"],
            bos_token_id=bos_token_id,
            stop_token_id=stop_token_id,
        )
        costs_std = _unbiased_std(costs)
        identical_costs = bool(np.asarray(costs_std == 0))
        advantages = jax.lax.stop_gradient(self.calc_advantage(costs))
        current_sequence_log_probs = reduce_sequence_log_probs(current_log_probs, lengths)

        if "reference_gate_logits" in kwargs:
            old_log_probs = self.log_prob(
                gate_indices,
                kwargs["reference_gate_logits"],
                kwargs["inverse_temperature"],
                bos_token_id=bos_token_id,
                stop_token_id=stop_token_id,
            )
        elif "old_log_probs" in kwargs:
            old_log_probs = kwargs["old_log_probs"]
        else:
            raise ValueError(
                "GRPOLoss.compute requires reference_gate_logits or old_log_probs"
            )

        old_sequence_log_probs = (
            old_log_probs
            if old_log_probs.ndim == 1
            else reduce_sequence_log_probs(old_log_probs, lengths)
        )
        ratio = jnp.exp(current_sequence_log_probs - old_sequence_log_probs)
        clipped_ratio = jnp.clip(
            ratio,
            1.0 - self.clip_ratio,
            1.0 + self.clip_ratio,
        )
        surrogate = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
        loss = -jnp.mean(surrogate)
        return loss, None, advantages, identical_costs

    def calc_advantage(self, costs):
        return (jnp.mean(costs) - costs) / (_unbiased_std(costs) + 1e-8)

    def log_prob(
        self,
        gate_seqs,
        gate_logits,
        inverse_temperature,
        *,
        bos_token_id: int = 0,
        stop_token_id: int | None = None,
    ):
        steps = gate_seqs.shape[1]
        aligned_logits = gate_logits[:, :steps, :]
        aligned_logits = apply_action_masks(
            aligned_logits,
            bos_token_id=bos_token_id,
            stop_token_id=stop_token_id,
        )
        log_probs = jax.nn.log_softmax(-inverse_temperature * aligned_logits, axis=-1)
        return jnp.take_along_axis(log_probs, gate_seqs[..., None], axis=-1).squeeze(-1)
