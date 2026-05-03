"""Non-dominated circuit archive over fidelity and structural objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ParetoPoint:
    """Circuit candidate stored with the objective values used for dominance."""

    fidelity: float
    depth: int
    total_gates: int
    cnot_count: int
    token_sequence: np.ndarray
    epoch: int
    opt_angles: Optional[np.ndarray] = None
    canonical_hash: Optional[str] = None


class ParetoArchive:
    """Maintain circuits that are non-dominated under F, depth, gates, and CNOTs."""

    def __init__(
        self,
        max_size: int = 500,
        fidelity_floor: float = 0.5,
        fidelity_tol: float = 0.0,
    ):
        """Build an archive with a fidelity admission floor and tie tolerance."""
        self.max_size = max_size
        self.fidelity_floor = fidelity_floor
        self.fidelity_tol = float(fidelity_tol)
        self._archive: List[ParetoPoint] = []
        self._fid = np.zeros((0,), dtype=np.float32)
        self._depth = np.zeros((0,), dtype=np.int32)
        self._total = np.zeros((0,), dtype=np.int32)
        self._cnot = np.zeros((0,), dtype=np.int32)
        self._hash_to_index: dict[str, int] = {}

    def _rebuild_arrays(self) -> None:
        if not self._archive:
            self._fid = np.zeros((0,), dtype=np.float32)
            self._depth = np.zeros((0,), dtype=np.int32)
            self._total = np.zeros((0,), dtype=np.int32)
            self._cnot = np.zeros((0,), dtype=np.int32)
            self._hash_to_index = {}
            return
        self._fid = np.fromiter(
            (p.fidelity for p in self._archive),
            dtype=np.float32,
            count=len(self._archive),
        )
        self._depth = np.fromiter(
            (p.depth for p in self._archive), dtype=np.int32, count=len(self._archive)
        )
        self._total = np.fromiter(
            (p.total_gates for p in self._archive),
            dtype=np.int32,
            count=len(self._archive),
        )
        self._cnot = np.fromiter(
            (p.cnot_count for p in self._archive),
            dtype=np.int32,
            count=len(self._archive),
        )
        self._hash_to_index = {
            p.canonical_hash: i
            for i, p in enumerate(self._archive)
            if p.canonical_hash is not None
        }

    @staticmethod
    def _prefer_replacement(
        candidate: ParetoPoint,
        incumbent: ParetoPoint,
        fidelity_tol: float,
    ) -> bool:
        """Return whether a same-hash candidate is the better representative."""
        if candidate.fidelity > incumbent.fidelity + fidelity_tol:
            return True
        if incumbent.fidelity > candidate.fidelity + fidelity_tol:
            return False
        return (
            candidate.cnot_count,
            candidate.depth,
            candidate.total_gates,
            -candidate.fidelity,
            candidate.epoch,
        ) < (
            incumbent.cnot_count,
            incumbent.depth,
            incumbent.total_gates,
            -incumbent.fidelity,
            incumbent.epoch,
        )

    @staticmethod
    def dominates(a: ParetoPoint, b: ParetoPoint, fidelity_tol: float = 0.0) -> bool:
        """Return whether ``a`` is no worse on all objectives and better on one."""
        better_or_equal = (
            a.fidelity + fidelity_tol >= b.fidelity
            and a.depth <= b.depth
            and a.total_gates <= b.total_gates
            and a.cnot_count <= b.cnot_count
        )
        strictly_better = (
            a.fidelity > b.fidelity + fidelity_tol
            or a.depth < b.depth
            or a.total_gates < b.total_gates
            or a.cnot_count < b.cnot_count
        )
        return better_or_equal and strictly_better

    def update(self, point: ParetoPoint) -> bool:
        """Insert one point and evict archive entries it dominates."""
        if point.fidelity < self.fidelity_floor:
            return False

        if point.canonical_hash is not None:
            existing_idx = self._hash_to_index.get(point.canonical_hash)
            if existing_idx is not None:
                existing = self._archive[existing_idx]
                if not self._prefer_replacement(point, existing, self.fidelity_tol):
                    return False
                self._archive.pop(existing_idx)
                self._rebuild_arrays()

        if self._fid.size > 0:
            pf = np.float32(point.fidelity)
            pd = np.int32(point.depth)
            pg = np.int32(point.total_gates)
            pc = np.int32(point.cnot_count)
            tol = np.float32(self.fidelity_tol)
            better_or_equal = (
                (self._fid + tol >= pf)
                & (self._depth <= pd)
                & (self._total <= pg)
                & (self._cnot <= pc)
            )
            strictly_better = (
                (self._fid > pf + tol)
                | (self._depth < pd)
                | (self._total < pg)
                | (self._cnot < pc)
            )
            if np.any(better_or_equal & strictly_better):
                return False

            nd_be = (
                (self._fid <= pf + tol)
                & (self._depth >= pd)
                & (self._total >= pg)
                & (self._cnot >= pc)
            )
            nd_sb = (
                (self._fid + tol < pf)
                | (self._depth > pd)
                | (self._total > pg)
                | (self._cnot > pc)
            )
            keep = ~(nd_be & nd_sb)
            if not keep.all():
                self._archive = [p for p, k in zip(self._archive, keep.tolist()) if k]
                self._fid = self._fid[keep]
                self._depth = self._depth[keep]
                self._total = self._total[keep]
                self._cnot = self._cnot[keep]

        self._archive.append(point)
        self._fid = np.append(self._fid, np.float32(point.fidelity))
        self._depth = np.append(self._depth, np.int32(point.depth))
        self._total = np.append(self._total, np.int32(point.total_gates))
        self._cnot = np.append(self._cnot, np.int32(point.cnot_count))
        if point.canonical_hash is not None:
            self._hash_to_index[point.canonical_hash] = len(self._archive) - 1

        if self.max_size > 0 and len(self._archive) > self.max_size:
            self._prune_by_crowding()

        return True

    def update_batch(self, points: List[ParetoPoint]) -> None:
        """Insert a batch by filtering archive plus candidates in one pass."""
        if not points:
            return
        floor = self.fidelity_floor
        candidates = [p for p in points if p.fidelity >= floor]
        if not candidates:
            return

        combined: list[ParetoPoint] = []
        hash_to_idx: dict[str, int] = {}
        for p in [*self._archive, *candidates]:
            h = p.canonical_hash
            if h is None:
                combined.append(p)
                continue
            existing_idx = hash_to_idx.get(h)
            if existing_idx is None:
                hash_to_idx[h] = len(combined)
                combined.append(p)
                continue
            existing = combined[existing_idx]
            if self._prefer_replacement(p, existing, self.fidelity_tol):
                combined[existing_idx] = p

        n = len(combined)
        fids = np.fromiter((p.fidelity for p in combined), dtype=np.float32, count=n)
        depths = np.fromiter((p.depth for p in combined), dtype=np.int32, count=n)
        totals = np.fromiter((p.total_gates for p in combined), dtype=np.int32, count=n)
        cnots = np.fromiter((p.cnot_count for p in combined), dtype=np.int32, count=n)
        tol = np.float32(self.fidelity_tol)

        f_i = fids[:, None]
        d_i = depths[:, None]
        g_i = totals[:, None]
        c_i = cnots[:, None]
        f_j = fids[None, :]
        d_j = depths[None, :]
        g_j = totals[None, :]
        c_j = cnots[None, :]
        be = (f_j + tol >= f_i) & (d_j <= d_i) & (g_j <= g_i) & (c_j <= c_i)
        sb = (f_j > f_i + tol) | (d_j < d_i) | (g_j < g_i) | (c_j < c_i)
        dominates = be & sb
        keep_mask = ~dominates.any(axis=1)

        keep_idx = np.flatnonzero(keep_mask)
        self._archive = [combined[i] for i in keep_idx]
        self._fid = fids[keep_idx]
        self._depth = depths[keep_idx]
        self._total = totals[keep_idx]
        self._cnot = cnots[keep_idx]
        self._hash_to_index = {
            p.canonical_hash: i
            for i, p in enumerate(self._archive)
            if p.canonical_hash is not None
        }

        while self.max_size > 0 and len(self._archive) > self.max_size:
            self._prune_by_crowding()

    def _prune_by_crowding(self) -> None:
        """Remove one entry from the most crowded region of objective space."""
        n = len(self._archive)
        cd = np.zeros(n, dtype=np.float64)

        for vals_list in (
            [p.fidelity for p in self._archive],
            [float(p.depth) for p in self._archive],
            [float(p.total_gates) for p in self._archive],
            [float(p.cnot_count) for p in self._archive],
        ):
            vals = np.array(vals_list)
            order = np.argsort(vals)
            val_range = float(vals[order[-1]] - vals[order[0]])
            if val_range == 0.0:
                continue
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / val_range

        remove_idx = int(np.argmin(cd))
        self._archive.pop(remove_idx)
        self._fid = np.delete(self._fid, remove_idx)
        self._depth = np.delete(self._depth, remove_idx)
        self._total = np.delete(self._total, remove_idx)
        self._cnot = np.delete(self._cnot, remove_idx)
        self._hash_to_index = {
            p.canonical_hash: i
            for i, p in enumerate(self._archive)
            if p.canonical_hash is not None
        }

    def set_fidelity_floor(self, floor: float) -> None:
        """Raise the fidelity floor and evict any entries below it."""
        self.fidelity_floor = floor
        self._archive = [p for p in self._archive if p.fidelity >= floor]
        self._rebuild_arrays()

    def hypervolume_2d(
        self,
        ref_fidelity: float = 0.0,
        ref_cnot: Optional[int] = None,
    ) -> float:
        """Compute a compact 2-D summary over fidelity and CNOT count."""
        if not self._archive:
            return 0.0
        if ref_cnot is None:
            ref_cnot = max(p.cnot_count for p in self._archive) + 1

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

    def to_sorted_list(self) -> List[ParetoPoint]:
        """Return archive entries with best-fidelity ties ordered by compactness."""
        return sorted(
            self._archive,
            key=lambda p: (-p.fidelity, p.cnot_count, p.depth, p.total_gates),
        )

    def best_by_fidelity(self) -> Optional[ParetoPoint]:
        """Return the highest-fidelity entry, tie-breaking by compactness."""
        if not self._archive:
            return None
        best_f = max(p.fidelity for p in self._archive)
        candidates = [
            p
            for p in self._archive
            if np.isclose(p.fidelity, best_f, rtol=0.0, atol=1.0e-7)
        ]
        return min(
            candidates,
            key=lambda p: (p.cnot_count, p.depth, p.total_gates, -p.fidelity),
        )

    def best_by_cnot(self, min_fidelity: float = 0.0) -> Optional[ParetoPoint]:
        """Return the lowest-CNOT entry among circuits with fidelity >= min_fidelity."""
        candidates = [p for p in self._archive if p.fidelity >= min_fidelity]
        return (
            min(
                candidates,
                key=lambda p: (p.cnot_count, p.depth, p.total_gates, -p.fidelity),
            )
            if candidates
            else None
        )

    def best_by_depth(self, min_fidelity: float = 0.0) -> Optional[ParetoPoint]:
        """Return the shallowest entry among circuits with fidelity >= min_fidelity."""
        candidates = [p for p in self._archive if p.fidelity >= min_fidelity]
        return (
            min(
                candidates,
                key=lambda p: (p.depth, p.cnot_count, p.total_gates, -p.fidelity),
            )
            if candidates
            else None
        )

    def best_by_total_gates(self, min_fidelity: float = 0.0) -> Optional[ParetoPoint]:
        """Return the entry with fewest total gates among circuits with fidelity >= min_fidelity."""
        candidates = [p for p in self._archive if p.fidelity >= min_fidelity]
        return (
            min(
                candidates,
                key=lambda p: (p.total_gates, p.cnot_count, p.depth, -p.fidelity),
            )
            if candidates
            else None
        )

    def __len__(self) -> int:
        return len(self._archive)

    def bulk_nondominated_mask_and_dom_count(
        self,
        fidelities: np.ndarray,
        depths: np.ndarray,
        cnot_counts: np.ndarray,
        total_gates: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return non-dominated flags and dominated-archive counts for a batch."""
        n_cand = int(fidelities.shape[0])
        if self._fid.size == 0:
            return (
                np.ones((n_cand,), dtype=bool),
                np.zeros((n_cand,), dtype=np.int32),
            )
        f = np.asarray(fidelities, dtype=np.float32)[:, None]
        d = np.asarray(depths, dtype=np.int32)[:, None]
        g = (
            np.asarray(total_gates, dtype=np.int32)[:, None]
            if total_gates is not None
            else d
        )
        c = np.asarray(cnot_counts, dtype=np.int32)[:, None]
        af = self._fid[None, :]
        ad = self._depth[None, :]
        ag = self._total[None, :]
        ac = self._cnot[None, :]
        tol = np.float32(self.fidelity_tol)

        arc_dom_cand_be = (af + tol >= f) & (ad <= d) & (ag <= g) & (ac <= c)
        arc_dom_cand_sb = (af > f + tol) | (ad < d) | (ag < g) | (ac < c)
        arc_dom_cand = arc_dom_cand_be & arc_dom_cand_sb

        cand_dom_arc_be = (f + tol >= af) & (d <= ad) & (g <= ag) & (c <= ac)
        cand_dom_arc_sb = (f > af + tol) | (d < ad) | (g < ag) | (c < ac)
        cand_dom_arc = cand_dom_arc_be & cand_dom_arc_sb

        is_nondominated = ~arc_dom_cand.any(axis=1)
        n_dom = cand_dom_arc.sum(axis=1).astype(np.int32)
        return is_nondominated, n_dom
