import argparse
from pathlib import Path

import numpy as np


def generate_haar_unitary(
    num_qubits: int,
    seed: int | None = None,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Generate a Haar-random unitary and save it as a NumPy file.

    Args:
        num_qubits: Number of qubits. The generated matrix has shape
            ``(2**num_qubits, 2**num_qubits)``.
        seed: Optional RNG seed for reproducibility.
        output_path: Optional output file path. If omitted, the file is written
            to ``targets/haar_random_q{num_qubits}_seed{seed}.npy`` when a seed
            is provided, or ``targets/haar_random_q{num_qubits}.npy`` otherwise.

    Returns:
        tuple[np.ndarray, Path]: The generated unitary and the saved file path.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    dimension = 2**num_qubits
    rng = np.random.default_rng(seed)

    z = (
        rng.standard_normal((dimension, dimension))
        + 1j * rng.standard_normal((dimension, dimension))
    ) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)

    diagonal = np.diag(r)
    phases = np.ones_like(diagonal)
    nonzero = np.abs(diagonal) > 0
    phases[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    unitary = q * phases[np.newaxis, :]

    identity = np.eye(dimension, dtype=np.complex128)
    if not np.allclose(unitary.conj().T @ unitary, identity, atol=1e-10):
        raise ValueError("Generated matrix is not unitary")

    if output_path is None:
        filename = (
            f"haar_random_q{num_qubits}_seed{seed}.npy"
            if seed is not None
            else f"haar_random_q{num_qubits}.npy"
        )
        output_path = Path(__file__).resolve().parent / "targets" / filename
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, unitary.astype(np.complex128, copy=False))

    return unitary, output_path.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    _, saved_path = generate_haar_unitary(
        num_qubits=args.num_qubits,
        seed=args.seed,
        output_path=args.output,
    )
    print(saved_path)
