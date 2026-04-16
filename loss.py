"""GRPO-style loss helpers for JAX-based GQE training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from cost import _unbiased_std


def reduce_sequence_log_probs(token_log_probs: jax.Array) -> jax.Array:
    """Return a length-normalized sequence score from per-token log-probabilities.

    Using a mean instead of a sum keeps GRPO importance ratios comparable across
    different circuit lengths. This is critical once `max_gates_count` scales up,
    because otherwise a small per-token policy shift compounds exponentially with
    sequence length and clips away most of the gradient signal.
    """
    if token_log_probs.ndim <= 1:
        return token_log_probs
    return jnp.mean(token_log_probs, axis=-1)


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
        current_log_probs = self.log_prob(
            gate_indices,
            gate_logits,
            kwargs["inverse_temperature"],
            invalid_token_ids=kwargs.get("invalid_token_ids"),
        )
        costs_std = _unbiased_std(costs)
        identical_costs = bool(np.asarray(costs_std == 0))
        advantages = jax.lax.stop_gradient(self.calc_advantage(costs))
        current_sequence_log_probs = reduce_sequence_log_probs(current_log_probs)

        if "reference_gate_logits" in kwargs:
            old_log_probs = self.log_prob(
                gate_indices,
                kwargs["reference_gate_logits"],
                kwargs["inverse_temperature"],
                invalid_token_ids=kwargs.get("invalid_token_ids"),
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
            else reduce_sequence_log_probs(old_log_probs)
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

    def log_prob(self, gate_seqs, gate_logits, inverse_temperature, invalid_token_ids=None):
        steps = gate_seqs.shape[1]
        aligned_logits = gate_logits[:, :steps, :]
        if invalid_token_ids is not None:
            aligned_logits = aligned_logits.at[..., invalid_token_ids].set(
                jnp.asarray(jnp.inf, dtype=aligned_logits.dtype)
            )
        log_probs = jax.nn.log_softmax(-inverse_temperature * aligned_logits, axis=-1)
        return jnp.take_along_axis(log_probs, gate_seqs[..., None], axis=-1).squeeze(-1)
