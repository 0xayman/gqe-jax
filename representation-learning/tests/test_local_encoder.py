"""Tests for per-qubit Choi features and the qubit-agnostic set encoder."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import encoder as enc
import features as feats


def _haar(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return q * phase[np.newaxis, :]


class TestPerQubitChoiFeatures:
    def test_output_shape_n2(self):
        u = _haar(4, 0)
        lf = feats.per_qubit_choi_features(u)
        assert lf.shape == (2, feats.SITE_FEAT_DIM)
        assert lf.dtype == np.float32

    def test_output_shape_n3(self):
        u = _haar(8, 1)
        lf = feats.per_qubit_choi_features(u)
        assert lf.shape == (3, feats.SITE_FEAT_DIM)

    def test_output_shape_n1(self):
        u = _haar(2, 2)
        lf = feats.per_qubit_choi_features(u)
        assert lf.shape == (1, feats.SITE_FEAT_DIM)

    def test_phase_invariant(self):
        u = _haar(4, 3)
        lf_a = feats.per_qubit_choi_features(u)
        lf_b = feats.per_qubit_choi_features(np.exp(1j * 1.23) * u)
        np.testing.assert_allclose(lf_a, lf_b, atol=1e-6)

    def test_identity_gives_finite_values(self):
        u = np.eye(4, dtype=np.complex128)
        lf = feats.per_qubit_choi_features(u)
        assert np.all(np.isfinite(lf))

    def test_choi_trace_is_one_per_qubit(self):
        # Reduced Choi tensor J_q[i,j,k,l] has trace = sum_{i,k} J_q[i,i,k,k] = 1
        # because Tr(U†U)/d = d/d = 1.
        u = _haar(4, 5)
        u_norm = feats.phase_normalize(u)
        for q in range(2):
            j_q = feats._reduced_choi_qubit(u_norm, 2, 4, q)
            # j_q has shape (4,4) after internal reshape; use einsum on the raw 4D tensor.
            # But _reduced_choi_qubit returns shape (4,4) already. We verify via the
            # 4D computation: reshape back to (2,2,2,2) and take iikk trace.
            j_4d = j_q.reshape(2, 2, 2, 2)  # (oq, oq', iq, iq')
            trace = np.einsum("iikk->", j_4d)
            assert abs(trace - 1.0) < 1e-6, f"qubit {q}: trace={trace}"

    def test_site_feat_dim_constant(self):
        assert feats.SITE_FEAT_DIM == 32


class TestLocalUnitaryEncoder:
    def _cfg(self, **kw):
        defaults = dict(
            site_feat_dim=32, hidden_dim=32, latent_dim=16,
            num_layers=1, num_heads=2, dropout=0.0,
        )
        return enc.LocalEncoderConfig(**{**defaults, **kw})

    def test_output_shape_fixed_for_n2(self):
        cfg = self._cfg()
        model, params = enc.init_local_encoder(cfg, jax.random.PRNGKey(0), n_sites=2)
        x = jnp.zeros((3, 2, 32))
        z = model.apply({"params": params}, x, deterministic=True)
        assert z.shape == (3, 16)

    def test_output_shape_fixed_for_n3(self):
        """Same weights, variable n — output latent dim stays constant."""
        cfg = self._cfg()
        model, params = enc.init_local_encoder(cfg, jax.random.PRNGKey(0), n_sites=2)
        x = jnp.zeros((3, 3, 32))
        z = model.apply({"params": params}, x, deterministic=True)
        assert z.shape == (3, 16)

    def test_output_shape_fixed_for_n4(self):
        cfg = self._cfg()
        model, params = enc.init_local_encoder(cfg, jax.random.PRNGKey(0), n_sites=2)
        x = jnp.zeros((2, 4, 32))
        z = model.apply({"params": params}, x, deterministic=True)
        assert z.shape == (2, 16)

    def test_float32_output(self):
        cfg = self._cfg()
        model, params = enc.init_local_encoder(cfg, jax.random.PRNGKey(1), n_sites=2)
        x = jnp.zeros((2, 2, 32))
        z = model.apply({"params": params}, x, deterministic=True)
        assert z.dtype == jnp.float32

    def test_same_weights_different_n(self):
        """The model has no n-dependent parameters; only the input sequence length varies."""
        cfg = self._cfg()
        model, params = enc.init_local_encoder(cfg, jax.random.PRNGKey(2), n_sites=2)
        # Init with n=3 must give the same param tree (no new params).
        _, params3 = enc.init_local_encoder(cfg, jax.random.PRNGKey(2), n_sites=3)
        leaves2 = jax.tree_util.tree_leaves(params)
        leaves3 = jax.tree_util.tree_leaves(params3)
        assert len(leaves2) == len(leaves3)
        for a, b in zip(leaves2, leaves3):
            assert a.shape == b.shape

    def test_masked_pooling_ignores_padding(self):
        """Masked mean-pool must not dilute real sites with zero-padding rows.

        We verify that the masked pooling formula correctly excludes masked-out
        (padding) positions from the average. We do this by monkey-patching the
        encoder to expose the pre-projection pooled vector and checking it
        directly: mean over 2 real sites must equal mean over 2 real + 1 zero
        site when the zero site is masked out.

        The test feeds identical real-site values in both cases and uses a
        num_layers=0 encoder so there is no attention mixing — only the pooling
        step is under test.
        """
        # Use a zero-layer encoder (proj_in + LayerNorm + gelu + mean-pool + proj_out)
        # so self-attention does not mix padding into real positions.
        cfg_0layer = enc.LocalEncoderConfig(
            site_feat_dim=32, hidden_dim=8, latent_dim=4,
            num_layers=0, num_heads=2, dropout=0.0,
        )
        model, params = enc.init_local_encoder(cfg_0layer, jax.random.PRNGKey(9), n_sites=2)

        rng = np.random.default_rng(55)
        x2 = jnp.asarray(rng.standard_normal((1, 2, 32)).astype(np.float32))

        # Pad to 3 sites with zeros.
        x3 = jnp.concatenate([x2, jnp.zeros((1, 1, 32), dtype=jnp.float32)], axis=1)

        # site_mask marks only the first 2 sites as real; the zero row is excluded.
        mask = jnp.array([[True, True, False]])

        z_unmasked = model.apply({"params": params}, x2, deterministic=True)
        z_masked = model.apply({"params": params}, x3, deterministic=True, site_mask=mask)
        # With no attention layers the only difference is the pooling step.
        # Masked pooling over 2 real sites must equal unmasked pooling over 2 sites.
        np.testing.assert_allclose(np.asarray(z_unmasked), np.asarray(z_masked), atol=1e-5)
