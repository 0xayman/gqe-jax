"""Target-unitary generators selected by ``cfg.target.type``."""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from config import GQEConfig

TargetGenerator = Callable[
    [List[Tuple[str, np.ndarray]], GQEConfig],
    Tuple[np.ndarray, str],
]


def _infer_num_qubits_from_dimension(dim: int) -> int | None:
    if dim <= 0 or dim & (dim - 1):
        return None
    return dim.bit_length() - 1


def _compose_unitary(gate_matrices: List[np.ndarray], d: int) -> np.ndarray:
    """Compose gates in application order using left multiplication."""
    u = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u = gate @ u
    return u


def random_reachable_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Generate a shallow target exactly expressible by the current pool."""
    depth = 4
    rng = np.random.default_rng(cfg.training.seed)
    indices = rng.integers(0, len(pool), size=depth)
    recipe = [pool[i][0] for i in indices]
    matrices = [pool[i][1] for i in indices]
    d = 2**cfg.target.num_qubits
    u_target = _compose_unitary(matrices, d)
    desc = f"random_reachable depth={depth}, recipe=[{', '.join(recipe)}]"
    return u_target, desc


def haar_random_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Generate a dense Haar-random target unitary."""
    d = 2**cfg.target.num_qubits
    rng = np.random.default_rng(cfg.training.seed)
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    u_target = q * phase[np.newaxis, :]
    desc = f"haar_random d={d}, seed={cfg.training.seed}"
    return u_target, desc


def _haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-uniform unitary on dimension ``d`` (QR-on-Ginibre)."""
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return q * phase[np.newaxis, :]


def _embed_adjacent_2q(
    gate_4x4: np.ndarray,
    q0: int,
    num_qubits: int,
) -> np.ndarray:
    """Embed an adjacent two-qubit gate under the MSB-first convention."""
    if not (0 <= q0 < num_qubits - 1):
        raise ValueError(f"adjacent 2q embedding needs 0 ≤ q0 < n-1, got q0={q0}, n={num_qubits}")
    left = np.eye(2 ** q0, dtype=gate_4x4.dtype)
    right = np.eye(2 ** (num_qubits - q0 - 2), dtype=gate_4x4.dtype)
    return np.kron(np.kron(left, gate_4x4), right)


def brickwork_haar_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Generate an alternating nearest-neighbor brickwork target."""
    del pool
    n = cfg.target.num_qubits
    if n < 2:
        raise ValueError("brickwork target requires num_qubits >= 2")
    d = 2 ** n
    depth = cfg.target.brickwork_depth
    if depth is None:
        depth = 2 * n
    rng = np.random.default_rng(cfg.training.seed)

    u = np.eye(d, dtype=np.complex128)
    for layer in range(depth):
        offset = layer % 2
        for q0 in range(offset, n - 1, 2):
            haar_2q = _haar_unitary(4, rng).astype(np.complex128)
            full_gate = _embed_adjacent_2q(haar_2q, q0, n)
            u = full_gate @ u
    desc = f"brickwork_haar n={n} depth={depth} seed={cfg.training.seed}"
    return u, desc


def file_target_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Load a target unitary from a saved NumPy ``.npy`` file."""
    del pool

    if not cfg.target.path:
        raise ValueError("target.type='file' requires cfg.target.path")

    path = Path(cfg.target.path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    u_target = np.load(path).astype(np.complex128, copy=False)
    d = 2**cfg.target.num_qubits
    if u_target.shape != (d, d):
        inferred_qubits = None
        if len(u_target.shape) == 2 and u_target.shape[0] == u_target.shape[1]:
            inferred_qubits = _infer_num_qubits_from_dimension(u_target.shape[0])
        inferred_hint = (
            f" The file looks like a {inferred_qubits}-qubit unitary."
            if inferred_qubits is not None
            else ""
        )
        raise ValueError(
            f"Loaded unitary from {path} has shape {u_target.shape}, but "
            f"target.num_qubits={cfg.target.num_qubits} expects {(d, d)}."
            f"{inferred_hint} Update either target.path or target.num_qubits "
            f"so they refer to the same system size."
        )
    desc = f"file:{path}"
    return u_target, desc


TARGET_REGISTRY: Dict[str, TargetGenerator] = {
    "random": random_reachable_generator,
    "random_reachable": random_reachable_generator,
    "haar_random": haar_random_generator,
    "brickwork": brickwork_haar_generator,
    "file": file_target_generator,
}


def build_target(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Build the configured target unitary and a human-readable description."""
    target_type = cfg.target.type
    if target_type not in TARGET_REGISTRY:
        available = sorted(TARGET_REGISTRY.keys())
        raise ValueError(
            f"Unknown target type: {target_type!r}. "
            f"Available: {available}. "
            f"To add a new type, implement a TargetGenerator function and "
            f"register it in target.TARGET_REGISTRY."
        )
    return TARGET_REGISTRY[target_type](pool, cfg)
