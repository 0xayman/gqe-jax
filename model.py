"""Flax GPT-2 language model used by GQE."""

from __future__ import annotations

from dataclasses import dataclass

import jax_setup  # noqa: F401
import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class _ArchConfig:
    n_layer: int
    n_head: int
    n_embd: int


_SIZES: dict[str, _ArchConfig] = {
    "tiny": _ArchConfig(n_layer=2, n_head=2, n_embd=128),
    "small": _ArchConfig(n_layer=6, n_head=6, n_embd=384),
    "medium": _ArchConfig(n_layer=12, n_head=12, n_embd=768),
    "large": _ArchConfig(n_layer=24, n_head=16, n_embd=1024),
}

VALID_MODEL_SIZES = frozenset(_SIZES)
_INIT_STDDEV = 0.02
_LAYER_NORM_EPS = 1.0e-5
_N_POSITIONS = 1024
_DROPOUT_RATE = 0.1
_CAUSAL_MASK = jnp.tril(jnp.ones((_N_POSITIONS, _N_POSITIONS), dtype=bool))


def _gelu_new(x: jax.Array) -> jax.Array:
    coeff = jnp.sqrt(2.0 / jnp.pi)
    return 0.5 * x * (1.0 + jnp.tanh(coeff * (x + 0.044715 * (x**3))))


def _normal_init():
    return nn.initializers.normal(stddev=_INIT_STDDEV)


class _CausalSelfAttention(nn.Module):
    n_head: int
    n_embd: int
    attn_pdrop: float = _DROPOUT_RATE
    resid_pdrop: float = _DROPOUT_RATE

    @nn.compact
    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = self.n_embd // self.n_head
        if head_dim * self.n_head != self.n_embd:
            raise ValueError("n_embd must be divisible by n_head")

        qkv = nn.Dense(
            3 * self.n_embd,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="c_attn",
        )(hidden_states)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        def _split_heads(x):
            x = x.reshape(batch_size, seq_len, self.n_head, head_dim)
            return jnp.transpose(x, (0, 2, 1, 3))

        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        scale = jnp.asarray(head_dim, dtype=hidden_states.dtype) ** -0.5
        attn_scores = jnp.einsum("bnth,bnsh->bnts", query, key) * scale

        causal_mask = _CAUSAL_MASK[:seq_len, :seq_len]
        if attention_mask is None:
            key_mask = jnp.ones((batch_size, seq_len), dtype=bool)
        else:
            key_mask = attention_mask.astype(bool)
        full_mask = causal_mask[None, None, :, :] & key_mask[:, None, None, :]
        mask_bias = jnp.where(
            full_mask,
            jnp.array(0.0, dtype=hidden_states.dtype),
            jnp.array(jnp.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype),
        )
        attn_probs = jax.nn.softmax(attn_scores + mask_bias, axis=-1)
        attn_probs = nn.Dropout(rate=self.attn_pdrop)(attn_probs, deterministic=deterministic)

        attn_output = jnp.einsum("bnts,bnsh->bnth", attn_probs, value)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, self.n_embd
        )
        attn_output = nn.Dense(
            self.n_embd,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="c_proj",
        )(attn_output)
        return nn.Dropout(rate=self.resid_pdrop)(
            attn_output,
            deterministic=deterministic,
        )


class _MLP(nn.Module):
    n_embd: int
    resid_pdrop: float = _DROPOUT_RATE

    @nn.compact
    def __call__(self, hidden_states: jax.Array, deterministic: bool) -> jax.Array:
        hidden_states = nn.Dense(
            4 * self.n_embd,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="c_fc",
        )(hidden_states)
        hidden_states = _gelu_new(hidden_states)
        hidden_states = nn.Dense(
            self.n_embd,
            kernel_init=_normal_init(),
            bias_init=nn.initializers.zeros,
            name="c_proj",
        )(hidden_states)
        return nn.Dropout(rate=self.resid_pdrop)(
            hidden_states,
            deterministic=deterministic,
        )


class _Block(nn.Module):
    n_head: int
    n_embd: int
    attn_pdrop: float = _DROPOUT_RATE
    resid_pdrop: float = _DROPOUT_RATE

    @nn.compact
    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        attn_input = nn.LayerNorm(epsilon=_LAYER_NORM_EPS, name="ln_1")(hidden_states)
        hidden_states = hidden_states + _CausalSelfAttention(
            n_head=self.n_head,
            n_embd=self.n_embd,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
            name="attn",
        )(attn_input, attention_mask, deterministic)

        mlp_input = nn.LayerNorm(epsilon=_LAYER_NORM_EPS, name="ln_2")(hidden_states)
        hidden_states = hidden_states + _MLP(
            n_embd=self.n_embd,
            resid_pdrop=self.resid_pdrop,
            name="mlp",
        )(mlp_input, deterministic)
        return hidden_states


class GPT2(nn.Module):
    size: str
    vocab_size: int
    n_positions: int = _N_POSITIONS
    embd_pdrop: float = _DROPOUT_RATE
    resid_pdrop: float = _DROPOUT_RATE
    attn_pdrop: float = _DROPOUT_RATE

    @nn.compact
    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        deterministic: bool = False,
    ) -> jax.Array:
        if self.size not in _SIZES:
            raise ValueError(
                f"Unknown model size '{self.size}'. Choose from: {sorted(_SIZES)}"
            )

        arch = _SIZES[self.size]
        seq_len = input_ids.shape[1]
        if seq_len > self.n_positions:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum positions {self.n_positions}"
            )

        wte = nn.Embed(
            num_embeddings=self.vocab_size,
            features=arch.n_embd,
            embedding_init=_normal_init(),
            name="wte",
        )
        wpe = nn.Embed(
            num_embeddings=self.n_positions,
            features=arch.n_embd,
            embedding_init=_normal_init(),
            name="wpe",
        )

        position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        hidden_states = wte(input_ids) + wpe(position_ids)
        hidden_states = nn.Dropout(rate=self.embd_pdrop)(
            hidden_states,
            deterministic=deterministic,
        )

        for layer_idx in range(arch.n_layer):
            hidden_states = _Block(
                n_head=arch.n_head,
                n_embd=arch.n_embd,
                attn_pdrop=self.attn_pdrop,
                resid_pdrop=self.resid_pdrop,
                name=f"h_{layer_idx}",
            )(hidden_states, attention_mask, deterministic)

        hidden_states = nn.LayerNorm(epsilon=_LAYER_NORM_EPS, name="ln_f")(hidden_states)
        return wte.attend(hidden_states)
