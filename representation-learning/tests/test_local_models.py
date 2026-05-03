"""Tests for LocalBudgetModel, LocalAngleInitModel, and local inference."""

import tempfile

import jax
import jax.numpy as jnp
import numpy as np

import angle_init as ai
import brickwork
import budget_model as bm
import dataset as ds
import encoder as enc
import infer
import train_angle as ta
import train_budget as tb
import vocab as _vocab


def _local_enc_cfg():
    return enc.LocalEncoderConfig(
        site_feat_dim=32, hidden_dim=32, latent_dim=16,
        num_layers=1, num_heads=2, dropout=0.0,
    )


def _head_cfg_b():
    return bm.BudgetConfig(k_min=0, k_max=6, hidden_dim=16)


def _head_cfg_a(arrays):
    return ai.AngleInitConfig(
        vocab_size=int(arrays["vocab_token_names"].size),
        max_skeleton_len=int(arrays["tokens"].shape[1]),
        hidden_dim=32, num_layers=1, num_heads=2, dropout=0.0,
    )


def _small_arrays(num_qubits=2):
    spec = ds.DatasetSpec(
        num_qubits=num_qubits,
        num_samples=12,
        brickwork_depth_min=1,
        brickwork_depth_max=3,
        fidelity_threshold=0.999,
        rotation_gates=("rz", "ry"),
        qiskit_optimization_levels=(2,),
        seed=0,
    )
    return ds.generate_dataset(spec)


class TestLocalBudgetModel:
    def test_forward_shape(self):
        cfg = _local_enc_cfg()
        head = _head_cfg_b()
        model = bm.LocalBudgetModel(cfg, head)
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((3, 2, 32))
        params = model.init({"params": rng, "dropout": rng}, x, deterministic=True)["params"]
        logits = model.apply({"params": params}, x, deterministic=True)
        assert logits.shape == (3, head.num_budgets)

    def test_forward_shape_with_site_mask(self):
        cfg = _local_enc_cfg()
        head = _head_cfg_b()
        model = bm.LocalBudgetModel(cfg, head)
        rng = jax.random.PRNGKey(0)
        x = jnp.zeros((3, 4, 32))  # max_qubits=4
        mask = jnp.array([[True, True, False, False],
                          [True, True, True, False],
                          [True, True, True, True]])
        params = model.init(
            {"params": rng, "dropout": rng}, x,
            deterministic=True, site_mask=mask,
        )["params"]
        logits = model.apply({"params": params}, x, deterministic=True, site_mask=mask)
        assert logits.shape == (3, head.num_budgets)

    def test_works_with_n3(self):
        cfg = _local_enc_cfg()
        head = _head_cfg_b()
        model = bm.LocalBudgetModel(cfg, head)
        rng = jax.random.PRNGKey(0)
        x2 = jnp.zeros((2, 2, 32))
        params = model.init({"params": rng, "dropout": rng}, x2, deterministic=True)["params"]
        x3 = jnp.zeros((2, 3, 32))
        logits = model.apply({"params": params}, x3, deterministic=True)
        assert logits.shape == (2, head.num_budgets)


class TestLocalAngleInitModel:
    def test_forward_shape(self):
        arrays = _small_arrays()
        cfg = _local_enc_cfg()
        head = _head_cfg_a(arrays)
        model = ai.LocalAngleInitModel(cfg, head)
        rng = jax.random.PRNGKey(0)
        dummy_lf = jnp.zeros((3, 2, 32))
        dummy_tok = jnp.zeros((3, head.max_skeleton_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((3, head.max_skeleton_len), dtype=bool)
        params = model.init(
            {"params": rng, "dropout": rng},
            dummy_lf, dummy_tok, dummy_mask, deterministic=True,
        )["params"]
        out = model.apply({"params": params}, dummy_lf, dummy_tok, dummy_mask, deterministic=True)
        assert out.shape == (3, head.max_skeleton_len)
        out_np = np.asarray(out)
        assert (out_np >= -np.pi - 1e-5).all() and (out_np < np.pi + 1e-5).all()


class TestLocalPipelineSmoke:
    def test_local_n2_pipeline(self):
        arrays = _small_arrays(num_qubits=2)
        assert "local_features" in arrays
        assert arrays["local_features"].shape == (12, 2, 32)

        cfg = _local_enc_cfg()
        head_b = _head_cfg_b()
        head_a = _head_cfg_a(arrays)

        with tempfile.TemporaryDirectory() as tmp:
            b_spec = tb.BudgetTrainSpec(
                epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
                val_split=0.25, seed=0, out_dir=tmp,
            )
            b_res = tb.train_local_budget(
                arrays, spec=b_spec, local_enc_cfg=cfg, head_cfg=head_b, verbose=False,
            )
            loaded_b = infer.load_local_budget(b_res["checkpoint"])

            a_spec = ta.AngleTrainSpec(
                epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
                val_split=0.25, seed=0, out_dir=tmp,
            )
            a_res = ta.train_local_angle(
                arrays, spec=a_spec, local_enc_cfg=cfg, head_cfg=head_a, verbose=False,
            )
            loaded_a = infer.load_local_angle_init(a_res["checkpoint"])

        rng = np.random.default_rng(42)
        u = brickwork.brickwork_haar(2, 3, rng)

        suggestion = infer.suggest_budget_local(loaded_b, u, confidence=0.5)
        assert isinstance(suggestion.suggested_k, int)
        assert suggestion.probs.shape == (head_b.num_budgets,)

        v = _vocab.build_vocab(2, ("rz", "ry"))
        skeleton = arrays["tokens"][0][: int(arrays["lengths"][0])]
        angles = infer.suggest_angles_local(loaded_a, u, skeleton, v)
        assert angles.shape == (skeleton.size,)
        is_param = np.asarray(v.is_parametric, dtype=bool)
        nonparam = ~is_param[skeleton]
        assert np.all(angles[nonparam] == 0.0)

    def test_local_n3_pipeline(self):
        arrays = _small_arrays(num_qubits=3)
        assert arrays["local_features"].shape == (12, 3, 32)

        cfg = _local_enc_cfg()
        head_b = bm.BudgetConfig(k_min=0, k_max=24, hidden_dim=16)
        head_a = _head_cfg_a(arrays)

        with tempfile.TemporaryDirectory() as tmp:
            b_spec = tb.BudgetTrainSpec(
                epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
                val_split=0.25, seed=0, out_dir=tmp,
            )
            b_res = tb.train_local_budget(
                arrays, spec=b_spec, local_enc_cfg=cfg, head_cfg=head_b, verbose=False,
            )
            loaded_b = infer.load_local_budget(b_res["checkpoint"])

            a_spec = ta.AngleTrainSpec(
                epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
                val_split=0.25, seed=0, out_dir=tmp,
            )
            a_res = ta.train_local_angle(
                arrays, spec=a_spec, local_enc_cfg=cfg, head_cfg=head_a, verbose=False,
            )
            loaded_a = infer.load_local_angle_init(a_res["checkpoint"])

        rng = np.random.default_rng(99)
        u = brickwork.brickwork_haar(3, 4, rng)

        suggestion = infer.suggest_budget_local(loaded_b, u, confidence=0.5)
        assert isinstance(suggestion.suggested_k, int)
        assert suggestion.probs.shape == (head_b.num_budgets,)

        v = _vocab.build_vocab(3, ("rz", "ry"))
        skeleton = arrays["tokens"][0][: int(arrays["lengths"][0])]
        angles = infer.suggest_angles_local(loaded_a, u, skeleton, v)
        assert angles.shape == (skeleton.size,)

    def test_multi_n_pipeline(self):
        """Train on mixed 2- and 3-qubit data with canonical vocab size 4."""
        spec = ds.DatasetSpec(
            num_qubits=2,
            num_samples=8,
            brickwork_depth_min=1,
            brickwork_depth_max=3,
            fidelity_threshold=0.999,
            rotation_gates=("rz", "ry"),
            qiskit_optimization_levels=(2,),
            seed=0,
            num_qubits_list=(2, 3),
            max_qubits=4,
        )
        arrays = ds.generate_dataset(spec)
        assert arrays["local_features"].shape[1] == 4  # max_qubits=4
        assert "num_qubits_per_sample" in arrays

        cfg = _local_enc_cfg()
        head_b = bm.BudgetConfig(k_min=0, k_max=24, hidden_dim=16)
        head_a = ai.AngleInitConfig(
            vocab_size=int(arrays["vocab_token_names"].size),
            max_skeleton_len=int(arrays["tokens"].shape[1]),
            hidden_dim=32, num_layers=1, num_heads=2, dropout=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            b_spec = tb.BudgetTrainSpec(
                epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
                val_split=0.25, seed=0, out_dir=tmp,
            )
            b_res = tb.train_local_budget(
                arrays, spec=b_spec, local_enc_cfg=cfg, head_cfg=head_b, verbose=False,
            )
            loaded_b = infer.load_local_budget(b_res["checkpoint"])
            assert loaded_b.max_qubits == 4

            a_spec = ta.AngleTrainSpec(
                epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
                val_split=0.25, seed=0, out_dir=tmp,
            )
            a_res = ta.train_local_angle(
                arrays, spec=a_spec, local_enc_cfg=cfg, head_cfg=head_a, verbose=False,
            )
            loaded_a = infer.load_local_angle_init(a_res["checkpoint"])
            assert loaded_a.max_qubits == 4

        # Inference on a 2-qubit unitary: features padded to max_qubits=4
        rng = np.random.default_rng(77)
        u2 = brickwork.brickwork_haar(2, 2, rng)
        suggestion = infer.suggest_budget_local(loaded_b, u2, confidence=0.5)
        assert isinstance(suggestion.suggested_k, int)

        # Use the canonical (max_qubits=4) vocab for token lookup
        v_canon = ds.vocab_from_dataset(arrays)
        skeleton = arrays["tokens"][0][: int(arrays["lengths"][0])]
        angles = infer.suggest_angles_local(loaded_a, u2, skeleton, v_canon)
        assert angles.shape == (skeleton.size,)
