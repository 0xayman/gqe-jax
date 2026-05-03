import numpy as np

import brickwork


def test_brickwork_unitary_returns_unitary():
    rng = np.random.default_rng(0)
    u = brickwork.brickwork_haar(3, 4, rng)
    assert u.shape == (8, 8)
    eye = u.conj().T @ u
    np.testing.assert_allclose(eye, np.eye(8), atol=1e-9)


def test_brickwork_minimum_qubits():
    rng = np.random.default_rng(0)
    try:
        brickwork.brickwork_haar(1, 2, rng)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n=1 brickwork")


def test_haar_unitary_unitary():
    rng = np.random.default_rng(1)
    u = brickwork.haar_unitary(4, rng)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-9)
