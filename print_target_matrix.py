#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a NumPy .npy target and print it as a 2D matrix."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="targets/haar_random_q2.npy",
        help="Path to the .npy file to print.",
    )
    args = parser.parse_args()

    matrix = np.load(Path(args.path))
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {matrix.shape}")

    np.set_printoptions(precision=6, suppress=False)
    print(matrix)


if __name__ == "__main__":
    main()
