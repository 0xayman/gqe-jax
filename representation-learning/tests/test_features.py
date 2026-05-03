import numpy as np

import features


def _haar(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return q * phase[np.newaxis, :]


def test_phase_normalize_unit_determinant():
    u = _haar(4, 0)
    u_norm = features.phase_normalize(u)
    det = np.linalg.det(u_norm)
    assert abs(abs(det) - 1.0) < 1e-9
    assert abs(np.angle(det)) < 1e-9


def test_phase_normalize_invariant_to_global_phase():
    u = _haar(4, 1)
    phi = 0.7
    u_phase = np.exp(1j * phi) * u
    a = features.phase_normalize(u)
    b = features.phase_normalize(u_phase)
    diff = np.abs(a - b).max()
    assert diff < 1e-9, f"phase-normalized matrices should agree, max diff = {diff}"


def test_unitary_to_features_dim():
    u = _haar(4, 2)
    f = features.unitary_to_features(u)
    assert f.shape == (features.feature_dim(2),)
    assert f.dtype == np.float32


def test_pauli_transfer_matrix_for_identity():
    u = np.eye(4, dtype=np.complex128)
    ptm = features.pauli_transfer_matrix(u)
    assert ptm.shape == (16, 16)
    np.testing.assert_allclose(ptm, np.eye(16), atol=1e-9)


def test_pauli_transfer_matrix_phase_invariant():
    u = _haar(4, 3)
    ptm_a = features.pauli_transfer_matrix(u)
    ptm_b = features.pauli_transfer_matrix(np.exp(1j * 0.3) * u)
    np.testing.assert_allclose(ptm_a, ptm_b, atol=1e-9)
