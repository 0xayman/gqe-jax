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
    # Optimised rotation angles aligned to the non-BOS portion of
    # ``token_sequence`` (shape ``(max_gates_count,)``). Stored so downstream
    # reporting can rebuild the exact circuit without re-running the angle
    # optimiser from a fresh RNG state.
    opt_angles: Optional[np.ndarray] = None


class ParetoArchive:
    """Maintains the Pareto-optimal set of circuits.

    Invariant: no entry in the archive dominates any other entry.

    Maintains sidecar NumPy arrays of objective values to make dominance
    checks vectorized; this keeps archive updates and batch bonus queries
    out of Python hot loops.
    """

    def __init__(
        self,
        max_size: int = 500,
        fidelity_floor: float = 0.5,
        fidelity_tol: float = 1e-4,
    ):
        """Build an empty Pareto archive.

        ``fidelity_tol`` defines when two circuits count as fidelity-tied for
        the purpose of dominance. With tolerance, a candidate whose fidelity is
        within ``fidelity_tol`` of an archive entry can still dominate it when
        its structure (depth or cnot_count) is strictly better. Without this,
        float-32 jitter produces "non-dominated" near-duplicates that flood the
        archive and inflate the Pareto bonus signal.
        """
        self.max_size = max_size
        self.fidelity_floor = fidelity_floor
        self.fidelity_tol = float(fidelity_tol)
        self._archive: List[ParetoPoint] = []
        self._fid = np.zeros((0,), dtype=np.float32)
        self._depth = np.zeros((0,), dtype=np.int32)
        self._cnot = np.zeros((0,), dtype=np.int32)

    def _rebuild_arrays(self) -> None:
        if not self._archive:
            self._fid = np.zeros((0,), dtype=np.float32)
            self._depth = np.zeros((0,), dtype=np.int32)
            self._cnot = np.zeros((0,), dtype=np.int32)
            return
        self._fid = np.fromiter(
            (p.fidelity for p in self._archive), dtype=np.float32, count=len(self._archive)
        )
        self._depth = np.fromiter(
            (p.depth for p in self._archive), dtype=np.int32, count=len(self._archive)
        )
        self._cnot = np.fromiter(
            (p.cnot_count for p in self._archive), dtype=np.int32, count=len(self._archive)
        )

    # ------------------------------------------------------------------
    # Core dominance logic
    # ------------------------------------------------------------------

    @staticmethod
    def dominates(
        a: ParetoPoint, b: ParetoPoint, fidelity_tol: float = 0.0
    ) -> bool:
        """Return True if a dominates b.

        a dominates b iff a is at least as good on every objective
        and strictly better on at least one. Fidelity comparison uses
        ``fidelity_tol`` as a tolerance: within ±tol counts as tied.
        """
        better_or_equal = (
            a.fidelity + fidelity_tol >= b.fidelity
            and a.depth <= b.depth
            and a.cnot_count <= b.cnot_count
        )
        strictly_better = (
            a.fidelity > b.fidelity + fidelity_tol
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

        if self._fid.size > 0:
            pf = np.float32(point.fidelity)
            pd = np.int32(point.depth)
            pc = np.int32(point.cnot_count)
            tol = np.float32(self.fidelity_tol)
            # Does any archive entry dominate the new point?
            better_or_equal = (
                (self._fid + tol >= pf) & (self._depth <= pd) & (self._cnot <= pc)
            )
            strictly_better = (
                (self._fid > pf + tol) | (self._depth < pd) | (self._cnot < pc)
            )
            if np.any(better_or_equal & strictly_better):
                return False

            # Does the new point dominate each archive entry?
            nd_be = (self._fid <= pf + tol) & (self._depth >= pd) & (self._cnot >= pc)
            nd_sb = (self._fid + tol < pf) | (self._depth > pd) | (self._cnot > pc)
            keep = ~(nd_be & nd_sb)
            if not keep.all():
                self._archive = [p for p, k in zip(self._archive, keep.tolist()) if k]
                self._fid = self._fid[keep]
                self._depth = self._depth[keep]
                self._cnot = self._cnot[keep]

        self._archive.append(point)
        self._fid = np.append(self._fid, np.float32(point.fidelity))
        self._depth = np.append(self._depth, np.int32(point.depth))
        self._cnot = np.append(self._cnot, np.int32(point.cnot_count))

        # Prune to max_size if needed
        if self.max_size > 0 and len(self._archive) > self.max_size:
            self._prune_by_crowding()

        return True

    def update_batch(self, points: List[ParetoPoint]) -> None:
        """Insert many points in one vectorised pass.

        Equivalent to calling :meth:`update` for each point but avoids the
        per-insert ``np.append`` + re-scan that made ``update`` O(B × A) per
        rollout. The combined (archive ∪ accepted-candidates) set is filtered
        in one O((B+A)²) numpy broadcast and then crowding-pruned to ``max_size``.
        """
        if not points:
            return
        floor = self.fidelity_floor
        candidates = [p for p in points if p.fidelity >= floor]
        if not candidates:
            return

        combined = list(self._archive) + candidates
        n = len(combined)
        fids = np.fromiter((p.fidelity for p in combined), dtype=np.float32, count=n)
        depths = np.fromiter((p.depth for p in combined), dtype=np.int32, count=n)
        cnots = np.fromiter((p.cnot_count for p in combined), dtype=np.int32, count=n)
        tol = np.float32(self.fidelity_tol)

        f_i = fids[:, None]
        d_i = depths[:, None]
        c_i = cnots[:, None]
        f_j = fids[None, :]
        d_j = depths[None, :]
        c_j = cnots[None, :]
        be = (f_j + tol >= f_i) & (d_j <= d_i) & (c_j <= c_i)
        sb = (f_j > f_i + tol) | (d_j < d_i) | (c_j < c_i)
        dominates = be & sb
        # A row i is dominated iff some j ≠ i dominates it. The strict-better
        # mask already excludes equal-on-all-axes pairs, so the diagonal is
        # naturally False and no fill_diagonal is needed.
        keep_mask = ~dominates.any(axis=1)

        # Preserve original archive-insertion order among survivors so
        # dominance ties are resolved deterministically.
        keep_idx = np.flatnonzero(keep_mask)
        self._archive = [combined[i] for i in keep_idx]
        self._fid = fids[keep_idx]
        self._depth = depths[keep_idx]
        self._cnot = cnots[keep_idx]

        while self.max_size > 0 and len(self._archive) > self.max_size:
            self._prune_by_crowding()

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

        remove_idx = int(np.argmin(cd))
        self._archive.pop(remove_idx)
        self._fid = np.delete(self._fid, remove_idx)
        self._depth = np.delete(self._depth, remove_idx)
        self._cnot = np.delete(self._cnot, remove_idx)

    # ------------------------------------------------------------------
    # Floor management
    # ------------------------------------------------------------------

    def set_fidelity_floor(self, floor: float) -> None:
        """Raise the fidelity floor and evict any entries below it."""
        self.fidelity_floor = floor
        self._archive = [p for p in self._archive if p.fidelity >= floor]
        self._rebuild_arrays()

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

    def best_by_total_gates(self, min_fidelity: float = 0.0) -> Optional[ParetoPoint]:
        """Return the entry with fewest total gates among circuits with fidelity >= min_fidelity."""
        candidates = [p for p in self._archive if p.fidelity >= min_fidelity]
        return min(candidates, key=lambda p: p.total_gates) if candidates else None

    def __len__(self) -> int:
        return len(self._archive)

    def bulk_nondominated_mask_and_dom_count(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnot_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized Pareto-proxy queries for a batch of candidates.

        For each candidate i, returns:
          - whether candidate i is NOT dominated by any archive member
          - the number of archive members that candidate i dominates
        """
        n_cand = int(fidelities.shape[0])
        if self._fid.size == 0:
            return (
                np.ones((n_cand,), dtype=bool),
                np.zeros((n_cand,), dtype=np.int32),
            )
        f = np.asarray(fidelities, dtype=np.float32)[:, None]
        d = np.asarray(depths, dtype=np.int32)[:, None]
        c = np.asarray(cnot_counts, dtype=np.int32)[:, None]
        af = self._fid[None, :]
        ad = self._depth[None, :]
        ac = self._cnot[None, :]
        tol = np.float32(self.fidelity_tol)

        arc_dom_cand_be = (af + tol >= f) & (ad <= d) & (ac <= c)
        arc_dom_cand_sb = (af > f + tol) | (ad < d) | (ac < c)
        arc_dom_cand = arc_dom_cand_be & arc_dom_cand_sb

        cand_dom_arc_be = (f + tol >= af) & (d <= ad) & (c <= ac)
        cand_dom_arc_sb = (f > af + tol) | (d < ad) | (c < ac)
        cand_dom_arc = cand_dom_arc_be & cand_dom_arc_sb

        is_nondominated = ~arc_dom_cand.any(axis=1)
        n_dom = cand_dom_arc.sum(axis=1).astype(np.int32)
        return is_nondominated, n_dom
