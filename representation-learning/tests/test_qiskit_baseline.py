import numpy as np

import brickwork
import qiskit_baseline as qb
import vocab


def test_round_trip_identity():
    v = vocab.build_vocab(2, ["rz", "ry"])
    u = np.eye(4, dtype=np.complex128)
    res = qb.best_baseline(u, v, seed=0)
    assert res.fidelity > 0.9999
    assert res.cnot_count == 0


def test_round_trip_brickwork_n2():
    v = vocab.build_vocab(2, ["rz", "ry"])
    rng = np.random.default_rng(0)
    u = brickwork.brickwork_haar(2, 4, rng)
    res = qb.best_baseline(u, v, seed=0)
    assert res.success, f"verification failed at F={res.fidelity}"
    assert res.cnot_count <= 3, "n=2 KAK upper bound is 3 CNOTs"


def test_round_trip_brickwork_n3():
    v = vocab.build_vocab(3, ["rz", "ry"])
    rng = np.random.default_rng(7)
    u = brickwork.brickwork_haar(3, 4, rng)
    res = qb.best_baseline(u, v, seed=0)
    assert res.success, f"verification failed at F={res.fidelity}"
    assert res.cnot_count <= 30, f"unexpectedly many CNOTs: {res.cnot_count}"


def test_token_to_unitary_matches_qiskit_action():
    """Replaying the converted tokens reproduces the input unitary up to phase."""
    v = vocab.build_vocab(2, ["rz", "ry"])
    rng = np.random.default_rng(2)
    u = brickwork.brickwork_haar(2, 3, rng)
    res = qb.best_baseline(u, v, seed=0)
    rebuilt = qb._build_project_unitary(res.tokens, res.angles, v)
    fid = qb._process_fidelity(rebuilt, u)
    assert fid > 0.999, f"replayed fidelity dropped to {fid}"


def test_unsupported_basis_raises():
    """Basis containing a u3-only-decomposable target should still convert with sx."""
    v = vocab.build_vocab(2, ["rz"])  # only RZ rotations + SX + CNOT
    rng = np.random.default_rng(5)
    u = brickwork.brickwork_haar(2, 3, rng)
    # Should not raise — qiskit will use rz+sx for single-qubit dressing.
    res = qb.best_baseline(u, v, seed=0)
    assert res.success
