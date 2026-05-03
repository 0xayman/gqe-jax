"""End-to-end smoke: tiny dataset, train both heads, run inference."""

import os
import tempfile

import numpy as np

import brickwork
import budget_model as bm
import dataset as ds
import encoder as enc
import angle_init as ai
import infer
import train_angle as ta
import train_budget as tb
import vocab as _vocab


def _build_small_arrays():
    spec = ds.DatasetSpec(
        num_qubits=2,
        num_samples=12,
        brickwork_depth_min=1,
        brickwork_depth_max=3,
        fidelity_threshold=0.999,
        rotation_gates=("rz", "ry"),
        qiskit_optimization_levels=(2,),
        seed=0,
    )
    return ds.generate_dataset(spec)


def test_full_pipeline_runs():
    arrays = _build_small_arrays()
    enc_cfg = enc.EncoderConfig(
        input_dim=int(arrays["features"].shape[1]),
        hidden_dim=32, latent_dim=16, num_layers=1, dropout=0.0,
    )
    head_cfg_b = bm.BudgetConfig(k_min=0, k_max=6, hidden_dim=16)
    head_cfg_a = ai.AngleInitConfig(
        vocab_size=int(arrays["vocab_token_names"].size),
        max_skeleton_len=int(arrays["tokens"].shape[1]),
        hidden_dim=32, num_layers=1, num_heads=2, dropout=0.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        b_spec = tb.BudgetTrainSpec(
            epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
            val_split=0.25, seed=0, out_dir=tmp,
        )
        b_res = tb.train_budget(arrays, spec=b_spec, enc_cfg=enc_cfg,
                                head_cfg=head_cfg_b, verbose=False)
        loaded_b = infer.load_budget(b_res["checkpoint"])

        a_spec = ta.AngleTrainSpec(
            epochs=2, batch_size=4, lr=1e-3, weight_decay=0.0,
            val_split=0.25, seed=0, out_dir=tmp,
        )
        a_res = ta.train_angle(arrays, spec=a_spec, enc_cfg=enc_cfg,
                               head_cfg=head_cfg_a, verbose=False)
        loaded_a = infer.load_angle_init(a_res["checkpoint"])

    rng = np.random.default_rng(99)
    u = brickwork.brickwork_haar(2, 3, rng)
    suggestion = infer.suggest_budget(loaded_b, u, confidence=0.5)
    assert isinstance(suggestion.suggested_k, int)
    assert suggestion.probs.shape == (head_cfg_b.num_budgets,)

    v = _vocab.build_vocab(2, ("rz", "ry"))
    skeleton = arrays["tokens"][0][: int(arrays["lengths"][0])]
    angles = infer.suggest_angles(loaded_a, u, skeleton, v)
    assert angles.shape == (skeleton.size,)
    is_param = np.asarray(v.is_parametric, dtype=bool)
    nonparam = ~is_param[skeleton]
    assert np.all(angles[nonparam] == 0.0)
