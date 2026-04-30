"""Replay samples used by PPO updates."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class BufferEntry:
    tokens: np.ndarray
    angles: np.ndarray
    cost: float
    advantage: float
    log_p_disc: np.ndarray


class ReplayBuffer:
    """Fixed-size FIFO store for complete rollout samples."""

    def __init__(self, size: int):
        self.size = size
        self.buf: deque[BufferEntry] = deque(maxlen=size)

    def push(self, entry: BufferEntry) -> None:
        self.buf.append(entry)

    def __len__(self) -> int:
        return len(self.buf)

    def __getitem__(self, idx: int) -> BufferEntry:
        return self.buf[idx]


class BufferDataset:
    """Batch iterator over replay rows, with optional repetition and shuffling."""

    def __init__(self, buffer: ReplayBuffer, repetition: int):
        self.buffer = buffer
        self.repetition = repetition

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
        n = len(self.buffer)
        if n == 0:
            return

        rows = list(self.buffer.buf)
        tokens = np.stack([r.tokens for r in rows], axis=0).astype(np.int32)
        angles = np.stack([r.angles for r in rows], axis=0).astype(np.float32)
        costs = np.asarray([r.cost for r in rows], dtype=np.float32)
        advantages = np.asarray([r.advantage for r in rows], dtype=np.float32)
        log_p_d = np.stack([r.log_p_disc for r in rows], axis=0).astype(np.float32)

        length = n * self.repetition
        idx = np.arange(length, dtype=np.int32)
        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            idx = rng.permutation(idx)

        total = length // batch_size if drop_last else math.ceil(length / batch_size)
        for b in range(total):
            start = b * batch_size
            stop = min(start + batch_size, length)
            if drop_last and stop - start < batch_size:
                continue
            sel = idx[start:stop] % n
            yield {
                "tokens": tokens[sel],
                "angles": angles[sel],
                "cost": costs[sel],
                "advantage": advantages[sel],
                "log_p_disc": log_p_d[sel],
            }
