"""Benchmark GQE against Qiskit on many Haar-random 2-qubit unitaries.

For each target unitary we record, in ``results/benchmark_2q.jsonl``:
    - verified GQE fidelity, CNOT count, circuit depth
    - Qiskit reference CNOT count, circuit depth (fidelity assumed 1.0)

The JSONL is appended one line per unitary. Re-running the script resumes:
completed seeds are skipped so long runs can be interrupted safely.

Usage:
    python benchmark.py [--num-unitaries 50] [--base-seed 1000]
                        [--output results/benchmark_2q.jsonl]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from config import TargetConfig, load_config
from cost import build_cost_fn
from gqe import gqe
from main import _compile_target_with_qiskit
from operator_pool import build_operator_pool
from reporting import (
    build_reported_circuit,
    circuit_stats,
    gate_names_from_token_sequence,
    select_best_fidelity_token_sequence,
    select_report_token_sequence,
)


NUM_QUBITS = 2


def haar_unitary(num_qubits: int, seed: int) -> np.ndarray:
    """Haar-uniform unitary on ``num_qubits`` qubits (QR-on-ginibre method)."""
    d = 2 ** num_qubits
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return q * phase[np.newaxis, :]


def build_benchmark_cfg(base_cfg):
    """Return a copy of ``base_cfg`` forced to 2 qubits with W&B disabled.

    The target section is set to a placeholder; we never call ``build_target``
    in the benchmark because the unitary is constructed in-process per seed.
    """
    target = TargetConfig(num_qubits=NUM_QUBITS, type="haar_random", path=None)
    logging = dataclasses.replace(base_cfg.logging, wandb=False)
    return dataclasses.replace(base_cfg, target=target, logging=logging)


def already_completed_seeds(output_path: Path) -> set[int]:
    """Read any existing JSONL output and return the set of completed seeds."""
    if not output_path.exists():
        return set()
    seeds: set[int] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "seed" in row and "gqe_fidelity" in row:
                seeds.add(int(row["seed"]))
    return seeds


def run_one(cfg, pool, u_target: np.ndarray, seed: int) -> dict:
    """Run GQE + Qiskit on one unitary. Returns a row dict ready for JSONL."""
    cost_fn = build_cost_fn(u_target)

    # ── Qiskit reference ───────────────────────────────────────────────────
    qiskit_qc = _compile_target_with_qiskit(u_target, NUM_QUBITS, cfg.pool.rotation_gates)
    q_depth, q_total, q_two_q = circuit_stats(qiskit_qc)

    # ── GQE training ────────────────────────────────────────────────────────
    t0 = time.time()
    _best_cost, best_indices, best_angles, pareto_archive = gqe(
        cost_fn, pool, cfg, u_target=u_target, logger=None
    )
    elapsed = time.time() - t0

    # Report two views of the trained archive:
    #  1. Best-fidelity circuit  → drives the fidelity panel
    #  2. Pareto best-depth at F≥threshold → drives the CNOT/depth panels
    bf_idx, bf_angles, bf_f, bf_src = select_best_fidelity_token_sequence(
        best_indices, best_angles, pareto_archive
    )
    rep_idx, rep_angles, rep_f, rep_src = select_report_token_sequence(
        best_indices, best_angles, pareto_archive, cfg.reward.fidelity_threshold
    )

    row: dict = {
        "seed": seed,
        "num_qubits": NUM_QUBITS,
        "qiskit_cnot": int(qiskit_qc.count_ops().get("cx", 0)),
        "qiskit_depth": q_depth,
        "qiskit_total_gates": q_total,
        "qiskit_two_q_gates": q_two_q,
        "elapsed_sec": elapsed,
    }

    if bf_idx is not None:
        bf = build_reported_circuit(
            cfg=cfg, pool=pool, u_target=u_target,
            report_indices=bf_idx, report_angles=bf_angles, report_fidelity=bf_f,
        )
        bf_depth, bf_total, _ = circuit_stats(bf.qc)
        row.update({
            "gqe_fidelity": bf.qc_fidelity,
            "gqe_cnot": int(bf.qc.count_ops().get("cx", 0)),
            "gqe_depth": bf_depth,
            "gqe_total_gates": bf_total,
            "gqe_gate_count": len(bf.gate_names),
            "gqe_source": bf_src,
        })
    else:
        row.update({
            "gqe_fidelity": float("nan"),
            "gqe_cnot": -1, "gqe_depth": -1,
            "gqe_total_gates": -1, "gqe_gate_count": 0,
            "gqe_source": "unavailable",
        })

    if rep_idx is not None:
        rep = build_reported_circuit(
            cfg=cfg, pool=pool, u_target=u_target,
            report_indices=rep_idx, report_angles=rep_angles, report_fidelity=rep_f,
        )
        rep_depth, rep_total, _ = circuit_stats(rep.qc)
        row.update({
            "gqe_pareto_fidelity": rep.qc_fidelity,
            "gqe_pareto_cnot": int(rep.qc.count_ops().get("cx", 0)),
            "gqe_pareto_depth": rep_depth,
            "gqe_pareto_total_gates": rep_total,
            "gqe_pareto_source": rep_src,
        })
    else:
        row.update({
            "gqe_pareto_fidelity": row["gqe_fidelity"],
            "gqe_pareto_cnot": row["gqe_cnot"],
            "gqe_pareto_depth": row["gqe_depth"],
            "gqe_pareto_total_gates": row["gqe_total_gates"],
            "gqe_pareto_source": row["gqe_source"],
        })

    return row


def format_eta(remaining: int, mean_sec: float) -> str:
    total = int(remaining * mean_sec)
    h, m = divmod(total // 60, 60)
    return f"{h:d}h{m:02d}m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-unitaries", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--output", default="results/benchmark_2q.jsonl")
    args = parser.parse_args()

    load_dotenv()

    base_cfg = load_config(args.config)
    cfg = build_benchmark_cfg(base_cfg)
    pool = build_operator_pool(num_qubits=NUM_QUBITS, rotation_gates=cfg.pool.rotation_gates)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_completed_seeds(output_path)

    all_seeds = [args.base_seed + i for i in range(args.num_unitaries)]
    pending = [s for s in all_seeds if s not in done]

    print(f"Benchmark config: {NUM_QUBITS}q, {args.num_unitaries} unitaries")
    print(f"  Output: {output_path}")
    print(f"  Completed (resume): {len(done)} / {len(all_seeds)}")
    print(f"  Pending: {len(pending)}")
    print(f"  Training epochs per unitary: {cfg.training.max_epochs}")
    print(f"  Rotation gates: {', '.join(cfg.pool.rotation_gates)}")
    print(f"  Pool size: {len(pool)}  |  max_gates: {cfg.model.max_gates_count}")

    durations: list[float] = []
    for i, seed in enumerate(pending, start=1):
        u_target = haar_unitary(NUM_QUBITS, seed=seed)
        print(f"\n[{i}/{len(pending)}] seed={seed}")
        t0 = time.time()
        row = run_one(cfg, pool, u_target, seed)
        dt = time.time() - t0
        durations.append(dt)

        with output_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        mean_sec = sum(durations) / len(durations)
        remaining = len(pending) - i
        print(
            f"  done in {dt:.1f}s  |  "
            f"GQE F={row['gqe_fidelity']:.4f} CX={row['gqe_cnot']} D={row['gqe_depth']}  |  "
            f"Qiskit CX={row['qiskit_cnot']} D={row['qiskit_depth']}  |  "
            f"ETA {format_eta(remaining, mean_sec)}"
        )

    print(f"\nBenchmark complete. Wrote {len(pending)} new rows to {output_path}.")


if __name__ == "__main__":
    main()
