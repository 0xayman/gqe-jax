"""Per-budget success classifier ``z -> P(success at F>=F_thr with <= k CNOTs)``.

The classifier emits one logit per CNOT budget ``k in [k_min, k_max]``.
Training uses the Qiskit baseline labels: for each row in the dataset, the
target is a monotone-non-decreasing 0/1 vector ``y``:

    y[k] = 1   iff  cnot_count <= k

so the classifier learns a CDF of "smallest CNOT count that suffices for
this unitary". At inference, callers pick the smallest ``k`` whose
predicted probability exceeds a chosen confidence threshold.

Why a CDF rather than a scalar regression?
The minimum-CNOT count for a generic unitary is not analytically known
beyond small dimensions; the Qiskit baseline gives an *upper bound*, not a
proof of optimality. Treating the labels as "succeeded with <= k cnots
under this transpiler" turns the brittle scalar into a calibrated
budget-success curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

import encoder as enc


@dataclass(frozen=True)
class BudgetConfig:
    k_min: int
    k_max: int
    hidden_dim: int = 256

    @property
    def num_budgets(self) -> int:
        return int(self.k_max - self.k_min + 1)


class BudgetHead(nn.Module):
    """One-hidden-layer head over the encoder output."""

    cfg: BudgetConfig

    @nn.compact
    def __call__(self, z, *, deterministic: bool = True):
        h = nn.Dense(self.cfg.hidden_dim, name="fc_h")(z)
        h = nn.gelu(h)
        logits = nn.Dense(self.cfg.num_budgets, name="fc_out")(h)
        return logits


class BudgetModel(nn.Module):
    """Encoder + budget head bundled for joint init / forward."""

    enc_cfg: enc.EncoderConfig
    head_cfg: BudgetConfig

    @nn.compact
    def __call__(self, features, *, deterministic: bool = True):
        z = enc.UnitaryEncoder(self.enc_cfg, name="encoder")(
            features, deterministic=deterministic,
        )
        logits = BudgetHead(self.head_cfg, name="head")(z, deterministic=deterministic)
        return logits


def cnot_to_cdf_labels(
    cnot_counts: jax.Array,
    cfg: BudgetConfig,
) -> jax.Array:
    """Convert per-row CNOT counts to ``[N, K]`` 0/1 CDF labels.

    ``y[i, k] = 1`` iff ``cnot_counts[i] <= k_min + k``. Counts above
    ``k_max`` produce a row of all-zeros (the model is asked to admit it
    cannot reach the target within the budget range).
    """
    k_axis = jnp.arange(cfg.num_budgets, dtype=cnot_counts.dtype)
    thresholds = jnp.asarray(cfg.k_min, dtype=cnot_counts.dtype) + k_axis
    return (cnot_counts[:, None] <= thresholds[None, :]).astype(jnp.float32)


def budget_loss(
    logits: jax.Array,
    cnot_counts: jax.Array,
    cfg: BudgetConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Binary cross-entropy over each (row, budget) cell."""
    labels = cnot_to_cdf_labels(cnot_counts, cfg)
    log_p = jax.nn.log_sigmoid(logits)
    log_1mp = jax.nn.log_sigmoid(-logits)
    bce = -(labels * log_p + (1.0 - labels) * log_1mp)
    loss = bce.mean()

    preds = (logits >= 0.0).astype(jnp.float32)
    accuracy = (preds == labels).mean()
    return loss, {"loss": loss, "accuracy": accuracy}


def predict_budget(
    logits: jax.Array,
    cfg: BudgetConfig,
    confidence: float = 0.5,
) -> jax.Array:
    """Return the smallest ``k`` with ``P(success | k) >= confidence`` per row.

    Returns ``cfg.k_max + 1`` for rows where no budget meets the threshold.
    """
    probs = jax.nn.sigmoid(logits)
    accepts = probs >= confidence
    indices = jnp.argmax(accepts.astype(jnp.int32), axis=-1)
    any_accept = accepts.any(axis=-1)
    fallback = jnp.asarray(cfg.num_budgets, dtype=indices.dtype)
    chosen = jnp.where(any_accept, indices, fallback)
    return chosen.astype(jnp.int32) + jnp.asarray(cfg.k_min, dtype=jnp.int32)


class LocalBudgetModel(nn.Module):
    """Qubit-agnostic budget classifier; takes per-qubit Choi features.

    Input ``local_features`` has shape ``[B, n, site_feat_dim]`` where ``n``
    can vary between batches (and between training and inference).
    ``site_mask`` is ``[B, n]`` bool; when provided only real qubit positions
    contribute to the mean-pool, ignoring zero-padding for n < max_qubits.
    """

    enc_cfg: enc.LocalEncoderConfig
    head_cfg: BudgetConfig

    @nn.compact
    def __call__(
        self,
        local_features,
        *,
        deterministic: bool = True,
        site_mask: Optional[jax.Array] = None,
    ):
        z = enc.LocalUnitaryEncoder(self.enc_cfg, name="encoder")(
            local_features, deterministic=deterministic, site_mask=site_mask,
        )
        logits = BudgetHead(self.head_cfg, name="head")(z, deterministic=deterministic)
        return logits
