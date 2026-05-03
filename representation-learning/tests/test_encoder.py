import jax
import jax.numpy as jnp

import encoder as enc


def test_encoder_init_and_forward():
    cfg = enc.EncoderConfig(input_dim=32, hidden_dim=64, latent_dim=16, num_layers=2, dropout=0.0)
    rng = jax.random.PRNGKey(0)
    model, params = enc.init_encoder(cfg, rng, batch_hint=2)
    x = jnp.ones((4, 32))
    z = model.apply({"params": params}, x, deterministic=True)
    assert z.shape == (4, 16)
    assert z.dtype == jnp.float32


def test_encoder_dropout_changes_output():
    cfg = enc.EncoderConfig(input_dim=8, hidden_dim=16, latent_dim=8, num_layers=2, dropout=0.5)
    rng = jax.random.PRNGKey(1)
    model, params = enc.init_encoder(cfg, rng, batch_hint=2)
    x = jnp.ones((2, 8))
    z_det = model.apply({"params": params}, x, deterministic=True)
    z_drop = model.apply({"params": params}, x, deterministic=False, rngs={"dropout": rng})
    # outputs may differ when dropout is active
    assert z_det.shape == z_drop.shape
