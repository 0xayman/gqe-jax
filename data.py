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

    def push(self, seq, cost) -> None:
        self.buf.append((np.asarray(seq, dtype=np.int32), np.float32(cost)))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.buf, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.buf = pickle.load(f)

    def __getitem__(self, idx: int):
        seq, cost = self.buf[idx]
        return {"idx": seq, "cost": cost}

    def __len__(self) -> int:
        return len(self.buf)


class BufferDataset:
    """Dataset-like view that repeats the replay buffer `repetition` times."""

    def __init__(self, buffer: ReplayBuffer, repetition: int):
        self.buffer = buffer
        self.repetition = repetition

    def __getitem__(self, idx: int):
        item = self.buffer[idx % len(self.buffer)]
        return {"idx": item["idx"], "cost": item["cost"]}

    def __len__(self) -> int:
        return len(self.buffer) * self.repetition

    def iter_batches(
        self,
        batch_size: int,
        *,
        drop_last: bool = True,
    ) -> Iterator[dict[str, np.ndarray]]:
        length = len(self)
        if length == 0:
            return

        total_batches = length // batch_size if drop_last else math.ceil(length / batch_size)
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            stop = min(start + batch_size, length)
            if drop_last and stop - start < batch_size:
                continue

            items = [self[i] for i in range(start, stop)]
            yield {
                "idx": np.stack([item["idx"] for item in items], axis=0).astype(np.int32),
                "cost": np.asarray([item["cost"] for item in items], dtype=np.float32),
            }
