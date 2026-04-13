"""Replay buffer utilities for JAX-based GQE training."""

from __future__ import annotations

import math
import pickle
import sys
from collections import deque
from typing import Iterator

import numpy as np


class ReplayBuffer:
    """FIFO replay buffer storing (sequence, cost) pairs."""

    def __init__(self, size: int = sys.maxsize):
        self.size = size
        self.buf = deque(maxlen=size)

    def push(self, seq, cost, old_log_prob: float | None = None) -> None:
        item = [np.asarray(seq, dtype=np.int32), np.float32(cost)]
        if old_log_prob is not None:
            item.append(np.float32(old_log_prob))
        self.buf.append(tuple(item))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.buf, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.buf = pickle.load(f)

    def __getitem__(self, idx: int):
        item = self.buf[idx]
        if len(item) == 2:
            seq, cost = item
            old_log_prob = np.float32(np.nan)
        else:
            seq, cost, old_log_prob = item
        return {"idx": seq, "cost": cost, "old_log_prob": old_log_prob}

    def __len__(self) -> int:
        return len(self.buf)


class BufferDataset:
    """Dataset-like view that repeats the replay buffer `repetition` times."""

    def __init__(self, buffer: ReplayBuffer, repetition: int):
        self.buffer = buffer
        self.repetition = repetition

    def __getitem__(self, idx: int):
        item = self.buffer[idx % len(self.buffer)]
        return {
            "idx": item["idx"],
            "cost": item["cost"],
            "old_log_prob": item["old_log_prob"],
        }

    def __len__(self) -> int:
        return len(self.buffer) * self.repetition

    def iter_batches(
        self,
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        buf_len = len(self.buffer)
        if buf_len == 0:
            return

        # Snapshot the entire buffer into arrays once to avoid per-item deque access.
        raw = list(self.buffer.buf)
        all_idx = np.stack([item[0] for item in raw], axis=0).astype(np.int32)
        all_cost = np.asarray([item[1] for item in raw], dtype=np.float32)
        all_log_prob = np.asarray(
            [item[2] if len(item) == 3 else np.float32(np.nan) for item in raw],
            dtype=np.float32,
        )

        length = buf_len * self.repetition
        indices = np.arange(length, dtype=np.int32)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            indices = rng.permutation(indices)

        total_batches = length // batch_size if drop_last else math.ceil(length / batch_size)
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            stop = min(start + batch_size, length)
            if drop_last and stop - start < batch_size:
                continue

            buf_indices = indices[start:stop] % buf_len
            yield {
                "idx": all_idx[buf_indices],
                "cost": all_cost[buf_indices],
                "old_log_prob": all_log_prob[buf_indices],
            }
