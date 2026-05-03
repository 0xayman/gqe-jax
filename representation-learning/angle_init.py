"""Angle initializer ``(U, skeleton tokens) -> initial angles``.

The model predicts one angle per skeleton position. Non-parametric tokens
(SX, CNOT, BOS, STOP, PAD) are masked out of the loss; only positions where
``token_is_parametric[token_id]`` is true contribute. The loss is the
periodic ``1 - cos(pred - target)`` so the network can learn 2*pi-equivalent
angles without bias.

Architecture
------------
* a token embedding (over the project vocabulary) + a learned positional
  embedding,
* the unitary latent ``z`` from :class:`encoder.UnitaryEncoder` is broadcast
  and added to every position,
* a small stack of (non-causal) transformer encoder layers,
* a per-position linear head producing the predicted angle (canonicalized
  to ``[-pi, pi)``).

This is an initializer, not a generator: it is fine to peek at the whole
skeleton at every position because the skeleton is fixed at inference time
(it is the gate sequence we want angles for).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

import encoder as enc


@dataclass(frozen=True)
class AngleInitConfig:
    vocab_size: int
    max_skeleton_len: int
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1


def canonicalize_angle(value: jax.Array) -> jax.Array:
    """Map angles to ``[-pi, pi)``."""
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=value.dtype)
    pi = jnp.asarray(jnp.pi, dtype=value.dtype)
    return jnp.mod(value + pi, two_pi) - pi


class _MLPBlock(nn.Module):
    hidden_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, *, deterministic):
        h = nn.Dense(4 * self.hidden_dim, name="fc1")(x)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, name="fc2")(h)
        return nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)


class _EncoderBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x, mask, *, deterministic):
        attn_in = nn.LayerNorm(name="ln_1")(x)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            out_features=self.hidden_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            name="attn",
        )(attn_in, attn_in, mask=mask)
        x = x + attn
        mlp_in = nn.LayerNorm(name="ln_2")(x)
        x = x + _MLPBlock(self.hidden_dim, self.dropout, name="mlp")(
            mlp_in, deterministic=deterministic,
        )
        return x


class AngleInitModel(nn.Module):
    """Encoder + skeleton transformer + per-position angle head."""

    enc_cfg: enc.EncoderConfig
    head_cfg: AngleInitConfig

    @nn.compact
    def __call__(
        self,
        features,        # [B, F]
        tokens,          # [B, L] int32
        attention_mask,  # [B, L] bool — True at valid positions
        *,
        deterministic: bool = True,
    ):
        z = enc.UnitaryEncoder(self.enc_cfg, name="encoder")(
            features, deterministic=deterministic,
        )
        z_proj = nn.Dense(self.head_cfg.hidden_dim, name="z_proj")(z)

        tok_emb = nn.Embed(
            self.head_cfg.vocab_size,
            self.head_cfg.hidden_dim,
            name="tok_emb",
        )(tokens)
        pos_emb = nn.Embed(
            self.head_cfg.max_skeleton_len,
            self.head_cfg.hidden_dim,
            name="pos_emb",
        )(jnp.arange(tokens.shape[-1], dtype=jnp.int32))
        x = tok_emb + pos_emb[None, :, :] + z_proj[:, None, :]
        x = nn.Dropout(rate=self.head_cfg.dropout)(x, deterministic=deterministic)

        # Build a 2-D attention mask suitable for MultiHeadDotProductAttention.
        # Both query and key positions must be valid.
        m_q = attention_mask[:, :, None]
        m_k = attention_mask[:, None, :]
        attn_mask = (m_q & m_k)[:, None, :, :]
        for i in range(self.head_cfg.num_layers):
            x = _EncoderBlock(
                hidden_dim=self.head_cfg.hidden_dim,
                num_heads=self.head_cfg.num_heads,
                dropout=self.head_cfg.dropout,
                name=f"block_{i}",
            )(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(name="ln_f")(x)
        raw = nn.Dense(1, name="angle_out")(x).squeeze(-1)
        return canonicalize_angle(raw)


class LocalAngleInitModel(nn.Module):
    """Qubit-agnostic angle initializer; takes per-qubit Choi features.

    Identical to :class:`AngleInitModel` except the unitary encoder is a
    :class:`encoder.LocalUnitaryEncoder` that accepts ``[B, n, site_feat_dim]``
    input, making the model independent of qubit count.
    ``site_mask`` is ``[B, n]`` bool; when provided only real qubit positions
    contribute to the mean-pool.
    """

    enc_cfg: enc.LocalEncoderConfig
    head_cfg: AngleInitConfig

    @nn.compact
    def __call__(
        self,
        local_features,  # [B, n, site_feat_dim]
        tokens,          # [B, L] int32
        attention_mask,  # [B, L] bool — True at valid positions
        *,
        deterministic: bool = True,
        site_mask: Optional[jax.Array] = None,
    ):
        z = enc.LocalUnitaryEncoder(self.enc_cfg, name="encoder")(
            local_features, deterministic=deterministic, site_mask=site_mask,
        )
        z_proj = nn.Dense(self.head_cfg.hidden_dim, name="z_proj")(z)

        tok_emb = nn.Embed(
            self.head_cfg.vocab_size,
            self.head_cfg.hidden_dim,
            name="tok_emb",
        )(tokens)
        pos_emb = nn.Embed(
            self.head_cfg.max_skeleton_len,
            self.head_cfg.hidden_dim,
            name="pos_emb",
        )(jnp.arange(tokens.shape[-1], dtype=jnp.int32))
        x = tok_emb + pos_emb[None, :, :] + z_proj[:, None, :]
        x = nn.Dropout(rate=self.head_cfg.dropout)(x, deterministic=deterministic)

        m_q = attention_mask[:, :, None]
        m_k = attention_mask[:, None, :]
        attn_mask = (m_q & m_k)[:, None, :, :]
        for i in range(self.head_cfg.num_layers):
            x = _EncoderBlock(
                hidden_dim=self.head_cfg.hidden_dim,
                num_heads=self.head_cfg.num_heads,
                dropout=self.head_cfg.dropout,
                name=f"block_{i}",
            )(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(name="ln_f")(x)
        raw = nn.Dense(1, name="angle_out")(x).squeeze(-1)
        return canonicalize_angle(raw)


def angle_loss(
    pred_angles: jax.Array,
    target_angles: jax.Array,
    parametric_mask: jax.Array,
    valid_mask: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Periodic ``1 - cos(pred - target)`` averaged over parametric+valid positions."""
    delta = canonicalize_angle(pred_angles - target_angles)
    per_loss = 1.0 - jnp.cos(delta)
    mask = parametric_mask & valid_mask
    denom = jnp.maximum(mask.sum().astype(per_loss.dtype), 1.0)
    loss = jnp.where(mask, per_loss, 0.0).sum() / denom
    abs_delta = jnp.where(mask, jnp.abs(delta), 0.0).sum() / denom
    return loss, {"loss": loss, "mean_abs_delta": abs_delta}
