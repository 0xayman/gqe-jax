"""Post-training gradient-descent optimizer for Pareto-optimal circuits.

After the generative training loop identifies the Pareto front, this module
fine-tunes the rotation-gate angles of every non-perfect circuit using
gradient descent.  The gate *structure* (which gates appear, in which order,
on which qubits) is kept fixed; only the continuous rotation angles are
optimized.

Circuits that already achieve fidelity >= 1 - fidelity_eps are skipped.
After all circuits have been optimized, the archive is rebuilt from scratch
so Pareto dominance is re-evaluated with the updated fidelities — a circuit
that gained fidelity may now dominate circuits it previously did not.
"""

from __future__ import annotations

import numpy as np
import jax

from continuous_optimizer import ContinuousOptimizer
from pareto import ParetoArchive, ParetoPoint


class ParetoGDOptimizer:
    """Gradient-descent angle optimizer for Pareto-archived circuits.

    Wraps :class:`~continuous_optimizer.ContinuousOptimizer` and applies it
    circuit-by-circuit to every entry in a :class:`~pareto.ParetoArchive`.

    Parameters
    ----------
    u_target:
        Target unitary matrix (numpy complex128 or JAX complex128 array).
    num_qubits:
        Number of qubits in the circuit.
    max_gates:
        Maximum gates per circuit — must equal ``model.max_gates_count`` in
        the config so the internal angle vectors are sized correctly.
    steps:
        Number of optimizer iterations per circuit.
    lr:
        Learning rate (step size) for the optimizer.
    optimizer_type:
        ``"lbfgs"`` (recommended, ~10× faster for small param counts) or
        ``"adam"``.
    num_restarts:
        Number of independent random angle initialisations.  The restart
        that achieves the highest fidelity is kept.
    fidelity_eps:
        Circuits with fidelity >= ``1.0 - fidelity_eps`` are considered
        perfect and skipped entirely.
    """

    def __init__(
        self,
        u_target,
        num_qubits: int,
        max_gates: int,
        steps: int,
        lr: float,
        optimizer_type: str,
        num_restarts: int,
        fidelity_eps: float = 1e-6,
    ) -> None:
        self.fidelity_eps = fidelity_eps
        self._optimizer = ContinuousOptimizer(
            u_target=np.asarray(u_target, dtype=np.complex128),
            num_qubits=num_qubits,
            steps=steps,
            lr=lr,
            optimizer_type=optimizer_type,
            top_k=0,
            max_gates=max_gates,
            num_restarts=num_restarts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_sequence_to_gate_names(
        token_sequence: np.ndarray,
        pool_names_no_bos: list[str],
    ) -> list[str]:
        """Convert a stored token sequence to a list of gate name strings.

        Parameters
        ----------
        token_sequence:
            Shape ``(max_gates_count + 1,)`` array.  Index 0 is the BOS
            token (``token_id == 0``); indices 1… are gate tokens
            (``token_id`` in ``[1, vocab_size)``).
        pool_names_no_bos:
            Gate-name strings indexed from 0, i.e. the operator pool names
            *without* the leading ``"<BOS>"`` entry.
            ``pool_names_no_bos[i]`` is the name for pool index ``i``,
            which corresponds to ``token_id == i + 1``.
        """
        # token_sequence[0] is always BOS (token_id=0) — skip it.
        # Gate token_ids are 1-based: pool_index = token_id - 1.
        return [
            pool_names_no_bos[int(tid) - 1]
            for tid in token_sequence[1:]
            if int(tid) > 0
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_archive(
        self,
        archive: ParetoArchive,
        pool_names_no_bos: list[str],
        rng_key: jax.Array,
        verbose: bool = False,
    ) -> tuple[ParetoArchive, jax.Array]:
        """Optimize rotation angles for every non-perfect circuit in the archive.

        For each :class:`~pareto.ParetoPoint` whose fidelity is below
        ``1.0 - fidelity_eps``, this method:

        1. Converts the stored token sequence to gate names.
        2. Runs gradient-descent angle optimization (with random restarts).
        3. Updates the fidelity if the optimizer improved it.

        After all circuits are processed, a new :class:`~pareto.ParetoArchive`
        is built from the updated points so that Pareto dominance is
        re-evaluated with the new fidelities.

        Parameters
        ----------
        archive:
            The Pareto archive to optimize.  Not modified in place.
        pool_names_no_bos:
            Gate-name strings without the leading ``"<BOS>"`` entry
            (i.e. ``pipeline.pool_names[1:]``).
        rng_key:
            JAX PRNG key consumed by random-restart angle sampling.
        verbose:
            Print per-circuit progress and a summary when ``True``.

        Returns
        -------
        new_archive:
            Fresh :class:`~pareto.ParetoArchive` with Pareto dominance
            re-evaluated after angle optimization.
        rng_key:
            Updated PRNG key after consuming randomness.
        """
        points = archive.to_sorted_list()
        if not points:
            return archive, rng_key

        n_total = len(points)
        n_skipped = 0
        n_improved = 0
        n_unchanged = 0

        updated: list[ParetoPoint] = []

        for i, point in enumerate(points):
            if point.fidelity >= 1.0 - self.fidelity_eps:
                if verbose:
                    print(
                        f"  [ParetoGD] [{i + 1}/{n_total}] "
                        f"fidelity={point.fidelity:.6f} — skipped (perfect)"
                    )
                updated.append(point)
                n_skipped += 1
                continue

            gate_names = self._token_sequence_to_gate_names(
                point.token_sequence, pool_names_no_bos
            )

            if verbose:
                print(
                    f"  [ParetoGD] [{i + 1}/{n_total}] "
                    f"fidelity={point.fidelity:.6f} | "
                    f"{len(gate_names)} gates | optimizing...",
                    end="",
                    flush=True,
                )

            new_fidelity, _, _, rng_key = self._optimizer.optimize_circuit_with_params(
                gate_names, rng_key
            )
            new_fidelity = float(new_fidelity)
            best_fidelity = max(point.fidelity, new_fidelity)

            if verbose:
                delta = new_fidelity - point.fidelity
                sign = "+" if delta >= 0 else ""
                print(f" -> {new_fidelity:.6f} ({sign}{delta:.6f})")

            if new_fidelity > point.fidelity:
                n_improved += 1
            else:
                n_unchanged += 1

            updated.append(
                ParetoPoint(
                    fidelity=best_fidelity,
                    depth=point.depth,
                    total_gates=point.total_gates,
                    cnot_count=point.cnot_count,
                    token_sequence=point.token_sequence,
                    epoch=point.epoch,
                )
            )

        if verbose:
            print(
                f"\n  [ParetoGD] Summary: "
                f"{n_improved} improved | "
                f"{n_unchanged} unchanged | "
                f"{n_skipped} skipped (perfect) | "
                f"total {n_total}"
            )

        # Rebuild the archive so Pareto dominance is re-evaluated with the
        # new fidelities.  Insert in fidelity-descending order so
        # high-fidelity entries enter first and drive out dominated ones.
        new_archive = ParetoArchive(
            max_size=archive.max_size,
            fidelity_floor=archive.fidelity_floor,
        )
        for point in sorted(updated, key=lambda p: -p.fidelity):
            new_archive.update(point)

        return new_archive, rng_key
