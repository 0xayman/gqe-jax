import jax
import jax.numpy as jnp
import numpy as np

import budget_model as bm
import encoder as enc


def test_cdf_labels_monotone():
    cfg = bm.BudgetConfig(k_min=0, k_max=4, hidden_dim=8)
    counts = jnp.asarray([0, 2, 4, 5, -1], dtype=jnp.int32)
    y = bm.cnot_to_cdf_labels(counts, cfg)
    expected = np.array([
        [1, 1, 1, 1, 1],   # 0 <= 0..4
        [0, 0, 1, 1, 1],   # 2 <= 2..4
        [0, 0, 0, 0, 1],   # 4 <= 4
        [0, 0, 0, 0, 0],   # 5 above max
        [1, 1, 1, 1, 1],   # negative count: trivially below all
    ], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(y), expected)


def test_predict_budget_picks_smallest_acceptable():
    cfg = bm.BudgetConfig(k_min=2, k_max=5, hidden_dim=4)
    # logits: row 0 first true at k=3, row 1 never true
    logits = jnp.asarray([[-3.0, -2.0, 4.0, 5.0],
                          [-3.0, -3.0, -3.0, -3.0]], dtype=jnp.float32)
    chosen = bm.predict_budget(logits, cfg, confidence=0.5)
    assert int(chosen[0]) == 4   # k_min(2) + idx(2)
    assert int(chosen[1]) == cfg.k_max + 1  # fallback when nothing accepted


def test_budget_model_forward_and_loss():
    enc_cfg = enc.EncoderConfig(input_dim=16, hidden_dim=32, latent_dim=8, num_layers=2, dropout=0.0)
    head_cfg = bm.BudgetConfig(k_min=0, k_max=6, hidden_dim=16)
    rng = jax.random.PRNGKey(0)
    model = bm.BudgetModel(enc_cfg, head_cfg)
    params = model.init({"params": rng, "dropout": rng},
                        jnp.zeros((2, 16)), deterministic=True)["params"]
    feats = jnp.ones((4, 16))
    logits = model.apply({"params": params}, feats, deterministic=True)
    assert logits.shape == (4, head_cfg.num_budgets)
    counts = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    loss, metrics = bm.budget_loss(logits, counts, head_cfg)
    assert metrics["loss"].shape == ()
    assert metrics["accuracy"].shape == ()
