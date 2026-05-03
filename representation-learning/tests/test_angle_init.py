import jax
import jax.numpy as jnp
import numpy as np

import angle_init as ai
import encoder as enc


def test_canonicalize_angle_range():
    x = jnp.asarray([0.0, jnp.pi - 1e-3, jnp.pi + 1e-3, -jnp.pi - 0.5, 4 * jnp.pi])
    y = ai.canonicalize_angle(x)
    y_np = np.asarray(y)
    assert (y_np >= -np.pi - 1e-6).all()
    assert (y_np < np.pi + 1e-6).all()


def test_angle_init_forward_and_loss():
    enc_cfg = enc.EncoderConfig(input_dim=32, hidden_dim=32, latent_dim=8, num_layers=1, dropout=0.0)
    head_cfg = ai.AngleInitConfig(vocab_size=11, max_skeleton_len=12, hidden_dim=32, num_layers=1, num_heads=2, dropout=0.0)
    rng = jax.random.PRNGKey(0)
    model = ai.AngleInitModel(enc_cfg, head_cfg)
    feats = jnp.zeros((3, 32))
    tokens = jnp.zeros((3, 12), dtype=jnp.int32)
    mask = jnp.ones((3, 12), dtype=bool)
    params = model.init({"params": rng, "dropout": rng}, feats, tokens, mask, deterministic=True)["params"]

    pred = model.apply({"params": params}, feats, tokens, mask, deterministic=True)
    assert pred.shape == (3, 12)
    pred_np = np.asarray(pred)
    assert (pred_np >= -np.pi - 1e-6).all() and (pred_np < np.pi + 1e-6).all()

    target = jnp.zeros_like(pred)
    pmask = jnp.ones_like(mask)
    loss, metrics = ai.angle_loss(pred, target, pmask, mask)
    assert metrics["loss"].shape == ()


def test_angle_loss_zero_when_perfect():
    pred = jnp.asarray([[0.1, 0.2, 0.3]])
    targ = jnp.asarray([[0.1, 0.2, 0.3]])
    mask = jnp.ones((1, 3), dtype=bool)
    loss, _ = ai.angle_loss(pred, targ, mask, mask)
    assert float(loss) < 1e-6


def test_angle_loss_periodic():
    pred = jnp.asarray([[0.0]])
    targ = jnp.asarray([[2.0 * jnp.pi]])
    mask = jnp.ones((1, 1), dtype=bool)
    loss, _ = ai.angle_loss(pred, targ, mask, mask)
    assert float(loss) < 1e-6  # 2pi-equivalent
