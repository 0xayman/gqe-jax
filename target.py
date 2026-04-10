"""Target unitary generators for GQE quantum compilation.

## Extension point — how to add a new target type

A TargetGenerator is any callable with this signature:

    def my_generator(
        pool: List[Tuple[str, np.ndarray]],
        cfg: GQEConfig,
    ) -> Tuple[np.ndarray, str]:
        ...
        return u_target, "human-readable description"

To register it so cfg.target.type = "my_type" selects it:

    TARGET_REGISTRY["my_type"] = my_generator

That's the entire extension. No other files need to change.

## Built-in generators

| cfg.target.type    | Strategy                                  | Fidelity 1.0 possible? |
|--------------------|-------------------------------------------|------------------------|
| "random_reachable" | random circuit from the pool              | Yes (by construction)  |
| "haar_random"      | Haar-uniform random unitary               | Unlikely for small N   |
| "file"             | load a saved unitary from .npy            | Depends on the file    |

Import convention: absolute imports only (flat module structure).
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from config import GQEConfig

# Type alias for generator callables
TargetGenerator = Callable[
    [List[Tuple[str, np.ndarray]], GQEConfig],
    Tuple[np.ndarray, str],
]


def _compose_unitary(gate_matrices: List[np.ndarray], d: int) -> np.ndarray:
    """Compose an ordered gate list into a single unitary.

    U = gate_matrices[-1] @ ... @ gate_matrices[0]
    (gate_matrices[0] is applied first to the quantum state.)
    """
    u = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u = gate @ u
    return u


def random_reachable_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Generate a random reachable target from the operator pool.

    Samples `depth` random gates from the pool using cfg.training.seed and
    composes them into U_target. Because the target is built from pool gates,
    it is guaranteed to be expressible by the GQE vocabulary — i.e., a
    perfect fidelity solution exists in principle.

    Depth is capped at cfg.model.max_gates_count to ensure the target is
    not deeper than what the model can generate.
    """
    depth = min(4, cfg.model.max_gates_count)
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
    """Generate a Haar-uniform random unitary.

    Uses QR decomposition of a random complex Gaussian matrix to produce
    a unitary drawn uniformly from the Haar measure on U(d). The target
    may not be exactly expressible by the operator pool, so perfect fidelity
    is not guaranteed. Useful for harder benchmarks or stress-testing the
    model's generalization.

    The `pool` argument is accepted but unused (kept for interface uniformity).
    """
    d = 2**cfg.target.num_qubits
    rng = np.random.default_rng(cfg.training.seed)
    # Random complex Gaussian matrix
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    # QR decomposition gives a Haar-uniform unitary
    q, r = np.linalg.qr(z)
    # Correct phases to ensure uniform Haar measure
    phase = np.diag(r) / np.abs(np.diag(r))
    u_target = q * phase[np.newaxis, :]
    desc = f"haar_random d={d}, seed={cfg.training.seed}"
    return u_target, desc


def file_target_generator(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Load a target unitary from a saved NumPy ``.npy`` file."""
    del pool  # Unused; kept for a uniform generator interface.

    if not cfg.target.path:
        raise ValueError("target.type='file' requires cfg.target.path")

    path = Path(cfg.target.path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    u_target = np.load(path).astype(np.complex128, copy=False)
    d = 2**cfg.target.num_qubits
    if u_target.shape != (d, d):
        raise ValueError(
            f"Loaded unitary shape {u_target.shape} does not match expected {(d, d)}"
        )
    desc = f"file:{path}"
    return u_target, desc


# ── Registry ──────────────────────────────────────────────────────────────────
# Maps cfg.target.type strings to generator callables.
#
# To add a new target type:
#   1. Write a function matching the TargetGenerator signature above.
#   2. Add one entry here.
#   3. Add the new name to VALID_TARGET_TYPES in config.py.
#   No other files need to change.
TARGET_REGISTRY: Dict[str, TargetGenerator] = {
    "random": random_reachable_generator,  # backward-compatible alias
    "random_reachable": random_reachable_generator,
    "haar_random": haar_random_generator,
    "file": file_target_generator,
}


def build_target(
    pool: List[Tuple[str, np.ndarray]],
    cfg: GQEConfig,
) -> Tuple[np.ndarray, str]:
    """Look up and run the appropriate target generator for cfg.target.type.

    This is the single public API used by main.py.

    Args:
        pool: The operator pool (list of (name, matrix) tuples).
        cfg: Full GQEConfig. cfg.target.type selects the generator.

    Returns:
        tuple: (u_target: np.ndarray, description: str)

    Raises:
        ValueError: If cfg.target.type is not in TARGET_REGISTRY.
    """
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
