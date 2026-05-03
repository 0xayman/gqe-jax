"""Shared bits for the representation-learning training scripts.

Kept tiny: the goal is to avoid copy/pasting Optax + checkpoint plumbing,
not to provide a generic trainer.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass
class TrainSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_split(
    n: int,
    val_split: float,
    seed: int,
) -> TrainSplit:
    if not (0.0 <= val_split < 1.0):
        raise ValueError("val_split must lie in [0, 1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_split * n))
    return TrainSplit(train_idx=perm[n_val:], val_idx=perm[:n_val])


def iter_minibatches(
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator,
    drop_last: bool,
) -> Iterator[np.ndarray]:
    if shuffle:
        order = rng.permutation(indices)
    else:
        order = indices
    n = order.size
    end = (n // batch_size) * batch_size if drop_last else n
    for start in range(0, end, batch_size):
        yield order[start:min(start + batch_size, n)]


def make_optimizer(
    lr: float,
    weight_decay: float,
    grad_clip: float = 1.0,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay),
    )


def save_checkpoint(
    out_dir: str,
    name: str,
    params,
    extras: dict | None = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pkl")
    payload = {"params": params, "extras": extras or {}}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def to_jax(arr: np.ndarray, dtype) -> jax.Array:
    return jnp.asarray(arr, dtype=dtype)


def merge_metrics(metric_list: Iterable[dict[str, jax.Array]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts = 0
    for m in metric_list:
        counts += 1
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + float(v)
    if counts == 0:
        return {}
    return {k: v / counts for k, v in sums.items()}
