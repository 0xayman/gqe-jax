"""Pareto archive for multi-objective quantum circuit optimization.

Tracks the non-dominated set of circuits across three objectives:
  - fidelity:   maximize
  - depth:      minimize
  - cnot_count: minimize

A circuit A dominates circuit B if A is at least as good on all objectives
and strictly better on at least one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ParetoPoint:
    """A single circuit with its objective values and token sequence."""

    fidelity: float
    depth: int
    total_gates: int
    cnot_count: int
    token_sequence: np.ndarray  # shape (max_gates_count + 1,) including BOS
    epoch: int


class ParetoArchive:
    """Maintains the Pareto-optimal set of circuits.

    Invariant: no entry in the archive dominates any other entry.
    """

    def __init__(self, max_size: int = 500, fidelity_floor: float = 0.5):
        self.max_size = max_size
        self.fidelity_floor = fidelity_floor
        self._archive: List[ParetoPoint] = []

    # ------------------------------------------------------------------
    # Core dominance logic
    # ------------------------------------------------------------------

    @staticmethod
    def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
        """Return True if a dominates b.

        a dominates b iff a is at least as good on every objective
        and strictly better on at least one.
        """
        better_or_equal = (
            a.fidelity >= b.fidelity
            and a.depth <= b.depth
            and a.cnot_count <= b.cnot_count
        )
        strictly_better = (
            a.fidelity > b.fidelity
            or a.depth < b.depth
            or a.cnot_count < b.cnot_count
        )
        return better_or_equal and strictly_better

    # ------------------------------------------------------------------
    # Archive update
    # ------------------------------------------------------------------

    def update(self, point: ParetoPoint) -> bool:
        """Add point to the archive if it is not dominated by any existing member.

        Also removes any existing members that are dominated by the new point.
        Returns True if the point was accepted into the archive.
        """
        if point.fidelity < self.fidelity_floor:
            return False

        # Reject if any existing member dominates the new point
        for existing in self._archive:
            if self.dominates(existing, point):
                return False

        # Remove members that the new point dominates
        self._archive = [p for p in self._archive if not self.dominates(point, p)]
        self._archive.append(point)

        # Prune to max_size if needed
        if self.max_size > 0 and len(self._archive) > self.max_size:
            self._prune_by_crowding()

        return True

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_by_crowding(self) -> None:
        """Remove the entry with the smallest crowding distance.

        Crowding distance measures how isolated a point is in objective
        space. Removing the point in the densest region preserves spread.
        """
        n = len(self._archive)
        cd = np.zeros(n, dtype=np.float64)

        for vals_list in (
            [p.fidelity for p in self._archive],
            [float(p.depth) for p in self._archive],
            [float(p.cnot_count) for p in self._archive],
        ):
            vals = np.array(vals_list)
            order = np.argsort(vals)
            val_range = float(vals[order[-1]] - vals[order[0]])
            if val_range == 0.0:
                continue
            # Boundary points get infinite CD so they are never pruned
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range

        self._archive.pop(int(np.argmin(cd)))

    # ------------------------------------------------------------------
    # Floor management
    # ------------------------------------------------------------------

    def set_fidelity_floor(self, floor: float) -> None:
        """Raise the fidelity floor and evict any entries below it."""
        self.fidelity_floor = floor
        self._archive = [p for p in self._archive if p.fidelity >= floor]

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def hypervolume_2d(
        self,
        ref_fidelity: float = 0.0,
        ref_cnot: Optional[int] = None,
    ) -> float:
        """Compute the 2-D hypervolume in (fidelity × cnot_count) space.

        The reference (worst-case) point is (ref_fidelity, ref_cnot).
        A larger hypervolume means the Pareto front covers more of the
        objective space — i.e., the trade-off surface is better.

        Algorithm (sweep):
          1. Project the archive onto the 2-D (fidelity, cnot_count) space.
          2. Walk from fewest to most CNOTs, keeping only points with strictly
             increasing fidelity (the 2-D Pareto front).
          3. HV = Σ_i (F_i − F_{i−1}) × (c_ref − c_i)
        """
        if not self._archive:
            return 0.0
        if ref_cnot is None:
            ref_cnot = max(p.cnot_count for p in self._archive) + 1

        # Build the 2-D Pareto front: sorted by cnot_count ascending.
        # On the 2-D front, as cnot_count increases fidelity must also
        # increase (otherwise the lower-cnot point would dominate).
        sorted_pts = sorted(self._archive, key=lambda p: p.cnot_count)
        front: List[ParetoPoint] = []
        max_f = ref_fidelity
        for p in sorted_pts:
            if p.fidelity > max_f:
                front.append(p)
                max_f = p.fidelity

        hv = 0.0
        prev_f = ref_fidelity
        for p in front:
            hv += (p.fidelity - prev_f) * (ref_cnot - p.cnot_count)
            prev_f = p.fidelity
        return hv

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def to_sorted_list(self) -> List[ParetoPoint]:
        """Return all archive entries sorted by fidelity descending."""
        return sorted(self._archive, key=lambda p: -p.fidelity)

    def best_by_fidelity(self) -> Optional[ParetoPoint]:
        """Return the entry with the highest fidelity, or None if empty."""
        return max(self._archive, key=lambda p: p.fidelity) if self._archive else None

    def best_by_cnot(self, min_fidelity: float = 0.0) -> Optional[ParetoPoint]:
        """Return the lowest-CNOT entry among circuits with fidelity >= min_fidelity."""
        candidates = [p for p in self._archive if p.fidelity >= min_fidelity]
        return min(candidates, key=lambda p: p.cnot_count) if candidates else None

    def best_by_depth(self, min_fidelity: float = 0.0) -> Optional[ParetoPoint]:
        """Return the shallowest entry among circuits with fidelity >= min_fidelity."""
        candidates = [p for p in self._archive if p.fidelity >= min_fidelity]
        return min(candidates, key=lambda p: p.depth) if candidates else None

    def __len__(self) -> int:
        return len(self._archive)
