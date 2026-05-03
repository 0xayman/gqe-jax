"""Flax MLP encoder ``U -> z``.

Takes a phase-normalized real feature vector (see :mod:`features`) and maps
it to a latent representation of dimension ``latent_dim``. The encoder is
deliberately small and standard so the budget classifier and angle initializer
can share it cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class EncoderConfig:
    input_dim: int      # 2 * d * d for a phase-normalized n-qubit unitary
    hidden_dim: int = 256
    latent_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


class UnitaryEncoder(nn.Module):
    """Plain residual-MLP encoder; LayerNorm + GELU between Dense layers."""

    cfg: EncoderConfig

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        x = nn.Dense(self.cfg.hidden_dim, name="proj_in")(x)
        x = nn.LayerNorm(name="ln_in")(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=deterministic)

        for i in range(self.cfg.num_layers):
            residual = x
            h = nn.Dense(self.cfg.hidden_dim, name=f"fc_{i}_a")(x)
            h = nn.gelu(h)
            h = nn.Dropout(rate=self.cfg.dropout)(h, deterministic=deterministic)
            h = nn.Dense(self.cfg.hidden_dim, name=f"fc_{i}_b")(h)
            x = nn.LayerNorm(name=f"ln_{i}")(residual + h)

        z = nn.Dense(self.cfg.latent_dim, name="proj_out")(x)
        return z


def init_encoder(
    cfg: EncoderConfig,
    rng: jax.Array,
    *,
    batch_hint: int = 4,
):
    """Initialize parameters by tracing one forward pass with dummy inputs."""
    model = UnitaryEncoder(cfg)
    dummy = jnp.zeros((batch_hint, cfg.input_dim), dtype=jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, deterministic=True)
    return model, variables["params"]


# ---------------------------------------------------------------------------
# Qubit-agnostic set-attention encoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalEncoderConfig:
    """Config for the qubit-agnostic set-attention encoder.

    The encoder receives a variable-length sequence of per-qubit feature
    vectors (one per qubit, each of dimension ``site_feat_dim``) and
    produces a fixed-size latent ``z`` regardless of the number of qubits.
    """

    site_feat_dim: int = 32   # must match features.SITE_FEAT_DIM
    hidden_dim: int = 128
    latent_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


class _FFNBlock(nn.Module):
    """Position-wise feed-forward block used inside the set encoder."""

    hidden_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        h = nn.Dense(self.hidden_dim * 2, name="fc1")(x)
        h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        h = nn.Dense(self.hidden_dim, name="fc2")(h)
        return h


class LocalUnitaryEncoder(nn.Module):
    """Set-attention encoder: ``[B, n, site_feat_dim] → [B, latent_dim]``.

    The number of sites ``n`` can vary between calls (variable qubit count).
    Parameters are independent of ``n``, so a single checkpoint applies to
    any qubit count seen at inference time.
    """

    cfg: LocalEncoderConfig

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool = True,
        site_mask: Optional[jax.Array] = None,
    ):
        # x: [batch, n_sites, site_feat_dim]
        # site_mask: [batch, n_sites] bool, True at real qubit positions; None = all real
        x = nn.Dense(self.cfg.hidden_dim, name="proj_in")(x)
        x = nn.LayerNorm(name="ln_in")(x)
        x = nn.gelu(x)

        for i in range(self.cfg.num_layers):
            # Self-attention over sites (non-causal; all qubits attend to all).
            residual = x
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.cfg.num_heads,
                qkv_features=self.cfg.hidden_dim,
                out_features=self.cfg.hidden_dim,
                dropout_rate=self.cfg.dropout,
                deterministic=deterministic,
                name=f"attn_{i}",
            )(x, x)
            x = nn.LayerNorm(name=f"ln_attn_{i}")(residual + h)

            residual = x
            h = _FFNBlock(self.cfg.hidden_dim, self.cfg.dropout, name=f"ffn_{i}")(
                x, deterministic=deterministic
            )
            x = nn.LayerNorm(name=f"ln_ffn_{i}")(residual + h)

        # Mean-pool over the site dimension → fixed-size representation.
        # When site_mask is provided, exclude padding sites from the average so that
        # zero-padded positions (for n < max_qubits) do not dilute the representation.
        if site_mask is None:
            z = x.mean(axis=-2)  # [batch, hidden_dim]
        else:
            # site_mask: [batch, n_sites] bool → expand to [batch, n_sites, 1]
            mask_f = jnp.asarray(site_mask, dtype=x.dtype)[..., None]
            z = (x * mask_f).sum(axis=-2) / jnp.maximum(mask_f.sum(axis=-2), 1.0)
        z = nn.Dense(self.cfg.latent_dim, name="proj_out")(z)
        return z


def init_local_encoder(
    cfg: LocalEncoderConfig,
    rng: jax.Array,
    *,
    n_sites: int = 2,
    batch_hint: int = 4,
):
    """Initialize a :class:`LocalUnitaryEncoder` with dummy inputs."""
    model = LocalUnitaryEncoder(cfg)
    dummy = jnp.zeros((batch_hint, n_sites, cfg.site_feat_dim), dtype=jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, deterministic=True)
    return model, variables["params"]
