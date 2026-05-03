import os
import tempfile

import numpy as np

import dataset


def _spec(num_qubits=2, num_samples=4):
    return dataset.DatasetSpec(
        num_qubits=num_qubits,
        num_samples=num_samples,
        brickwork_depth_min=1,
        brickwork_depth_max=3,
        fidelity_threshold=0.999,
        rotation_gates=("rz", "ry"),
        qiskit_optimization_levels=(2,),
        seed=0,
    )


def _multi_n_spec(num_samples=3):
    return dataset.DatasetSpec(
        num_qubits=2,
        num_samples=num_samples,
        brickwork_depth_min=1,
        brickwork_depth_max=3,
        fidelity_threshold=0.999,
        rotation_gates=("rz", "ry"),
        qiskit_optimization_levels=(2,),
        seed=0,
        num_qubits_list=(2, 3),
        max_qubits=4,
    )


def test_generate_shapes_and_keys():
    arrays = dataset.generate_dataset(_spec(num_qubits=2, num_samples=3))
    # Single-n: max_qubits == num_qubits == 2; features padded to 2*4*4=32 (unchanged)
    assert arrays["features"].shape == (3, 32)
    assert arrays["local_features"].shape == (3, 2, 32)  # [N, max_qubits, SITE_FEAT_DIM]
    assert arrays["unitaries"].shape == (3, 4, 4)
    assert arrays["tokens"].shape[0] == 3
    assert arrays["angles"].shape == arrays["tokens"].shape
    assert arrays["lengths"].shape == (3,)
    assert arrays["cnot_count"].shape == (3,)
    assert arrays["fidelity"].shape == (3,)
    assert arrays["vocab_token_names"].size > 3  # specials + gates
    assert "num_qubits_per_sample" in arrays
    assert arrays["num_qubits_per_sample"].shape == (3,)


def test_generate_succeeds_within_threshold():
    arrays = dataset.generate_dataset(_spec(num_qubits=2, num_samples=4))
    assert (arrays["fidelity"] >= 0.999).all()


def test_save_and_load_roundtrip():
    arrays = dataset.generate_dataset(_spec(num_qubits=2, num_samples=3))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.npz")
        dataset.save_dataset(arrays, path)
        loaded = dataset.load_dataset(path)
    for k in ("features", "tokens", "angles", "lengths", "cnot_count"):
        np.testing.assert_array_equal(arrays[k], loaded[k])


def test_vocab_from_dataset():
    arrays = dataset.generate_dataset(_spec(num_qubits=2, num_samples=2))
    v = dataset.vocab_from_dataset(arrays)
    assert v.num_qubits == 2
    assert v.token_names == tuple(arrays["vocab_token_names"].tolist())


def test_multi_n_dataset_shapes():
    arrays = dataset.generate_dataset(_multi_n_spec(num_samples=3))
    N_total = 6  # 3 samples x 2 qubit counts
    assert arrays["local_features"].shape == (N_total, 4, 32)  # max_qubits=4
    assert arrays["features"].shape == (N_total, 2 * 16 * 16)  # padded to 2*d_max^2
    assert arrays["unitaries"].shape == (N_total, 16, 16)
    assert arrays["tokens"].shape[0] == N_total
    assert arrays["num_qubits_per_sample"].shape == (N_total,)
    nq = arrays["num_qubits_per_sample"]
    assert set(nq.tolist()) == {2, 3}
    assert int(arrays["max_qubits"]) == 4
    # Canonical vocab must have num_qubits=4.
    v = dataset.vocab_from_dataset(arrays)
    assert v.num_qubits == 4


def test_multi_n_token_remapping():
    # Tokens for n=2 and n=3 must both live in the canonical (max_qubits=4) vocab space
    # and use the same name-to-id mapping regardless of original n.
    arrays = dataset.generate_dataset(_multi_n_spec(num_samples=2))
    v = dataset.vocab_from_dataset(arrays)
    max_tok = int(v.vocab_size)
    toks = arrays["tokens"]
    assert toks.max() < max_tok, "all token IDs must fit in canonical vocab"
    # PAD token must map to PAD_TOKEN_ID in the canonical vocab.
    import vocab as _vocab
    assert _vocab.PAD_TOKEN_ID < max_tok
