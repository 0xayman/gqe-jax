"""GRPO-style loss helpers for JAX-based GQE training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


def _unbiased_std(values: jax.Array) -> jax.Array:
    return jnp.std(values, ddof=1)


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
    """Generalized-RPO / clipped-PPO variant used in the original code."""

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
        )

        win_id = jnp.argmin(costs)
        loss = -jnp.mean(current_log_probs[win_id])
        costs_std = _unbiased_std(costs)
        identical_costs = bool(np.asarray(costs_std == 0))
        if identical_costs:
            return loss, None, None, True

        if kwargs["current_step"] == 0:
            advantages = self.calc_advantage(costs)
            loss = loss - jnp.mean(advantages[:, None])
            return (
                loss,
                jax.lax.stop_gradient(current_log_probs),
                jax.lax.stop_gradient(advantages),
                False,
            )

        old_log_probs = kwargs["old_log_probs"]
        advantages = kwargs["advantages"]
        ratio = jnp.exp(current_log_probs - old_log_probs)
        clipped_ratio = jnp.clip(
            ratio,
            1.0 - self.clip_ratio,
            1.0 + self.clip_ratio,
        )
        loss = loss - jnp.mean(clipped_ratio * advantages[:, None])
        return loss, None, None, False

    def calc_advantage(self, costs):
        return (jnp.mean(costs) - costs) / (_unbiased_std(costs) + 1e-8)

    def log_prob(self, gate_seqs, gate_logits, inverse_temperature):
        steps = gate_seqs.shape[1]
        aligned_logits = gate_logits[:, :steps, :]
        log_probs = jax.nn.log_softmax(-inverse_temperature * aligned_logits, axis=-1)
        return jnp.take_along_axis(log_probs, gate_seqs[..., None], axis=-1).squeeze(-1)
