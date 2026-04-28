"""Benchmark hybrid-action GQE against Qiskit on N random target unitaries.

Usage::

    python benchmark.py -n 50 -q 3                   # 50 targets, 3 qubits
    python benchmark.py -n 30 -q 4 --base-seed 2000  # custom seed range
    python benchmark.py -n 50 -q 3 --resume          # skip seeds already in JSONL

The target distribution is read from ``config.yml`` (``target.type`` —
typically ``haar_random`` or ``brickwork``). For brickwork targets the depth
is also read from the config (``target.brickwork_depth``).

For each target unitary the benchmark trains the GQE policy from scratch with
the user's ``config.yml`` (overriding only ``num_qubits`` and
``training.seed``), records the best Pareto-front circuit found, and compiles
the same target with Qiskit ``optimization_level=3`` for reference.

Outputs (written under ``results/benchmarks/`` by default):

* ``benchmark_<N>q_<type>_<stamp>.jsonl`` — one JSON row per circuit
  with both GQE and Qiskit metrics (process fidelity, depth, total gates,
  CNOT count, training time).

* ``benchmark_<N>q_<type>_<stamp>.meta.json`` — sidecar with the full
  config snapshot, CLI args, seed plan, and host info for reproducibility.

* ``benchmark_<N>q_<type>_<stamp>.png`` — three-panel plot:

  - top: per-circuit GQE fidelity (sorted ascending) + Qiskit reference line
    + F=0.99 threshold line + summary statistics
  - middle: paired CNOT-count bars (GQE vs Qiskit)
  - bottom: paired circuit-depth bars (GQE vs Qiskit)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from config import GQEConfig, TargetConfig, load_config
from operator_pool import build_operator_pool
from reporting import circuit_stats
from target import build_target
from trainer import gqe


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark hybrid-action GQE vs Qiskit on N random unitaries. "
                    "Target type and brickwork depth are read from the YAML config.",
    )
    p.add_argument("-n", "--num-circuits", type=int, default=50,
                   help="Number of target unitaries to benchmark (default: 50).")
    p.add_argument("-q", "--num-qubits", type=int, default=3,
                   help="Qubit count for every target (default: 3).")
    p.add_argument("-c", "--config", default="config.yml",
                   help="Base GQE config YAML (default: config.yml). "
                        "Drives target.type, target.brickwork_depth, and all "
                        "training/reward hyperparameters.")
    p.add_argument("--base-seed", type=int, default=1000,
                   help="Seeds used: base_seed, base_seed+1, ... (default: 1000).")
    p.add_argument("-o", "--output-dir", default="results/benchmarks",
                   help="Directory for the JSONL + PNG outputs "
                        "(default: results/benchmarks).")
    p.add_argument("--resume", action="store_true",
                   help="Skip seeds already present in the JSONL output.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation (useful for headless dispatch).")
    return p.parse_args()


# ── Per-target helpers ───────────────────────────────────────────────────────

def _override_cfg(
    base_cfg: GQEConfig,
    num_qubits: int,
    seed: int,
) -> GQEConfig:
    """Return a copy of ``base_cfg`` with per-iteration overrides applied.

    Only ``num_qubits`` (CLI override) and ``training.seed`` (per-circuit) are
    overridden; ``target.type`` and ``target.brickwork_depth`` come from the
    YAML config and are propagated unchanged. W&B logging is disabled (the
    benchmark would otherwise spam one run per circuit). The gate budget is
    recomputed from the heuristic for the new qubit count.
    """
    from config import default_max_gates_count
    target = dataclasses.replace(
        base_cfg.target,
        num_qubits=num_qubits,
        # Drop any file path — for benchmarking we always synthesise the
        # target on the fly from ``training.seed``.
        path=None if base_cfg.target.type != "file" else base_cfg.target.path,
    )
    logging = dataclasses.replace(base_cfg.logging, wandb=False)
    model = dataclasses.replace(
        base_cfg.model, max_gates_count=default_max_gates_count(num_qubits),
    )
    training = dataclasses.replace(base_cfg.training, seed=seed)
    return dataclasses.replace(
        base_cfg, target=target, logging=logging, model=model, training=training,
    )


def _qiskit_compile(u_target, num_qubits: int, rotation_gates):
    """Run Qiskit's optimization_level=3 transpile against ``u_target``.

    Returns ``(qc, depth, total_gates, cnot_count)``. Process fidelity isn't
    reported because Qiskit's transpile is exact up to numerical noise — we
    take F=1.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(u_target), list(range(num_qubits)))
    basis = [*rotation_gates, "sx", "cx"]
    compiled = transpile(qc, basis_gates=basis, optimization_level=3)
    depth, total, _ = circuit_stats(compiled)
    return compiled, depth, total, int(compiled.count_ops().get("cx", 0))


def _gqe_best(result) -> dict:
    """Return ``{F, depth, total_gates, cnots}`` for the best-fidelity entry
    in ``result.pareto_archive`` (after post-training refinement).

    Falls back to the best raw rollout if the archive is empty (the
    fidelity_floor wasn't reached during training).
    """
    arc = result.pareto_archive
    if arc is not None and len(arc) > 0:
        bf = arc.best_by_fidelity()
        return {
            "F": float(bf.fidelity),
            "depth": int(bf.depth),
            "total_gates": int(bf.total_gates),
            "cnots": int(bf.cnot_count),
        }
    F = (
        result.refined_raw_fidelity
        if result.refined_raw_fidelity is not None
        else result.best_raw_fidelity
    )
    return {
        "F": float(F) if F is not None else float("nan"),
        "depth": -1, "total_gates": -1, "cnots": -1,
    }


# ── JSONL persistence ────────────────────────────────────────────────────────

def _completed_seeds(jsonl_path: Path) -> set[int]:
    if not jsonl_path.exists():
        return set()
    seeds: set[int] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "seed" in row:
                seeds.add(int(row["seed"]))
    return seeds


def _load_rows(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ── One target run ───────────────────────────────────────────────────────────

def _run_one(args, base_cfg: GQEConfig, seed: int) -> dict:
    cfg = _override_cfg(base_cfg, args.num_qubits, seed)
    pool = build_operator_pool(args.num_qubits, cfg.pool.rotation_gates)
    u_target, desc = build_target(pool, cfg)

    # Qiskit reference (tiny cost)
    _, q_depth, q_total, q_cnot = _qiskit_compile(
        u_target, args.num_qubits, cfg.pool.rotation_gates,
    )

    # GQE training
    t0 = time.time()
    result = gqe(cfg, u_target, pool, logger=None)
    dt = time.time() - t0
    gqe_b = _gqe_best(result)

    return {
        "seed": seed,
        "num_qubits": args.num_qubits,
        "target_type": cfg.target.type,
        "target_desc": desc,
        "elapsed_sec": dt,
        "gqe_F": gqe_b["F"],
        "gqe_depth": gqe_b["depth"],
        "gqe_total_gates": gqe_b["total_gates"],
        "gqe_cnot": gqe_b["cnots"],
        "qiskit_depth": int(q_depth),
        "qiskit_total_gates": int(q_total),
        "qiskit_cnot": int(q_cnot),
    }


# ── Plot ─────────────────────────────────────────────────────────────────────

def _plot(rows: list[dict], png_path: Path, *, fidelity_threshold: float = 0.99) -> None:
    """Render the three-panel benchmark figure (sorted by GQE F ascending)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("gqe_F") is not None]
    if not rows:
        print("No rows to plot — aborting plot step.")
        return

    N = len(rows)
    F = np.asarray([r["gqe_F"] for r in rows], dtype=np.float64)
    gqe_cnot = np.asarray([r["gqe_cnot"] for r in rows], dtype=np.int64)
    gqe_depth = np.asarray([r["gqe_depth"] for r in rows], dtype=np.int64)
    qk_cnot = np.asarray([r["qiskit_cnot"] for r in rows], dtype=np.int64)
    qk_depth = np.asarray([r["qiskit_depth"] for r in rows], dtype=np.int64)
    n_qubits = int(rows[0]["num_qubits"])
    target_type = rows[0]["target_type"]

    order = np.argsort(F)
    F_s = F[order]
    gqe_cnot_s = gqe_cnot[order]
    gqe_depth_s = gqe_depth[order]
    qk_cnot_s = qk_cnot[order]
    qk_depth_s = qk_depth[order]
    xs = np.arange(N)

    # Summary stats
    n_above = int((F >= fidelity_threshold).sum())
    pct_above = 100.0 * n_above / N
    if n_above > 0:
        mask = F >= fidelity_threshold
        cnot_delta = gqe_cnot[mask] - qk_cnot[mask]
        depth_delta = gqe_depth[mask] - qk_depth[mask]
        med_cnot_delta = float(np.median(cnot_delta))
        med_depth_delta = float(np.median(depth_delta))
        match_pct = 100.0 * float(np.mean(cnot_delta <= 0))
    else:
        med_cnot_delta = float("nan")
        med_depth_delta = float("nan")
        match_pct = float("nan")

    # Khaneja-Glaser CNOT optimum (for the upper bound line)
    numer = 4 ** n_qubits - 3 * n_qubits - 1
    cnot_lb = max(0, -(-numer // 4))

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(11, 8),
        gridspec_kw={"height_ratios": [1.6, 1, 1]},
    )

    # ── Top: fidelity scatter + threshold lines + stats text ────────────────
    ax = axes[0]
    ax.scatter(xs, F_s, color="C0", s=22, zorder=3, label="GQE best fidelity")
    ax.axhline(1.0, color="C1", linestyle="--", linewidth=1.2, label="Qiskit reference (F=1)")
    ax.axhline(fidelity_threshold, color="C2", linestyle=":", linewidth=1.2,
               label=f"F = {fidelity_threshold}")
    ax.set_ylabel("Process fidelity")
    fmin = max(0.0, F.min() - 0.02)
    ax.set_ylim(fmin, 1.005)
    ax.legend(loc="lower right", fontsize=8)
    stats = (
        f"N = {N} unitaries\n"
        f"GQE fidelity: mean {F.mean():.4f}  |  median {np.median(F):.4f}  |  min {F.min():.4f}\n"
        f"F ≥ {fidelity_threshold}: {pct_above:.0f}% of instances\n"
        f"At F ≥ {fidelity_threshold}: GQE ≤ Qiskit (CNOT) in {match_pct:.0f}%\n"
        f"Median delta (GQE - Qiskit) at F ≥ {fidelity_threshold}: "
        f"CNOT {med_cnot_delta:+.1f}, depth {med_depth_delta:+.1f}"
    )
    ax.text(
        0.02, 0.05, stats, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    # ── Middle: CNOT bars ───────────────────────────────────────────────────
    ax = axes[1]
    width = 0.4
    ax.bar(xs - width / 2, gqe_cnot_s, width, color="C0", label="GQE")
    ax.bar(xs + width / 2, qk_cnot_s, width, color="C1", label="Qiskit")
    if cnot_lb > 0:
        ax.axhline(cnot_lb, color="0.4", linestyle="--", linewidth=1.0,
                   label=f"K-G optimum = {cnot_lb}")
    ax.set_ylabel("CNOT count")
    ax.legend(loc="upper right", fontsize=8)

    # ── Bottom: depth bars ──────────────────────────────────────────────────
    ax = axes[2]
    ax.bar(xs - width / 2, gqe_depth_s, width, color="C0", label="GQE")
    ax.bar(xs + width / 2, qk_depth_s, width, color="C1", label="Qiskit")
    ax.set_ylabel("Circuit depth")
    ax.set_xlabel("Unitary index (sorted by GQE fidelity, ascending)")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"GQE vs Qiskit on {N} {target_type} {n_qubits}-qubit unitaries",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"Saved plot → {png_path}")


# ── Metadata sidecar ─────────────────────────────────────────────────────────

def _write_benchmark_metadata(
    *,
    meta_path: Path,
    args: argparse.Namespace,
    base_cfg: GQEConfig,
    all_seeds: list[int],
    already_done: list[int],
    jsonl_path: Path,
    png_path: Path,
) -> None:
    """Write a JSON sidecar describing the benchmark experiment.

    Captures the full config snapshot, CLI args, seed plan, and host info so
    that a JSONL row file is self-contained for later analysis.
    """
    import getpass
    import platform
    import socket
    import sys
    from datetime import datetime, timezone

    # Reuse the same config snapshot used for single runs to keep the schema
    # consistent across results/runs/ and results/benchmarks/.
    from reporting import _config_snapshot

    target_type = base_cfg.target.type
    payload = {
        "kind": "benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "experiment": {
            "num_qubits": int(args.num_qubits),
            "target_type": target_type,
            "brickwork_depth": (
                int(base_cfg.target.brickwork_depth or (2 * args.num_qubits))
                if target_type == "brickwork" else None
            ),
            "num_circuits": int(args.num_circuits),
            "base_seed": int(args.base_seed),
            "seeds": all_seeds,
            "already_completed_seeds": already_done,
            "config_path": args.config,
            "fidelity_threshold": float(base_cfg.reward.fidelity_threshold),
        },
        "outputs": {
            "jsonl": str(jsonl_path),
            "png": str(png_path),
        },
        "config": _config_snapshot(base_cfg),
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote benchmark metadata → {meta_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    load_dotenv()

    base_cfg = load_config(args.config)
    target_type = base_cfg.target.type
    if target_type == "file":
        raise ValueError(
            "Cannot benchmark with target.type='file' — the benchmark generates "
            "fresh targets per seed. Use 'haar_random' or 'brickwork' in config.yml."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / (
        f"benchmark_{args.num_qubits}q_{target_type}_{stamp}.jsonl"
    )
    png_path = jsonl_path.with_suffix(".png")
    meta_path = jsonl_path.with_suffix(".meta.json")

    # Resume support: if --resume, look for an existing JSONL with the same
    # config tag and append to it. Otherwise we always start fresh.
    if args.resume:
        existing = sorted(output_dir.glob(
            f"benchmark_{args.num_qubits}q_{target_type}_*.jsonl"
        ))
        if existing:
            jsonl_path = existing[-1]
            png_path = jsonl_path.with_suffix(".png")
            meta_path = jsonl_path.with_suffix(".meta.json")
            print(f"Resuming into existing file: {jsonl_path}")

    done = _completed_seeds(jsonl_path) if args.resume else set()
    all_seeds = [args.base_seed + i for i in range(args.num_circuits)]
    pending = [s for s in all_seeds if s not in done]

    print(f"Benchmark configuration:")
    print(f"  Qubits:               {args.num_qubits}")
    print(f"  Target type:          {target_type}  (from {args.config})")
    if target_type == "brickwork":
        depth = base_cfg.target.brickwork_depth or (2 * args.num_qubits)
        print(f"  Brickwork depth:      {depth}")
    print(f"  Total circuits:       {args.num_circuits}")
    print(f"  Already completed:    {len(done)}")
    print(f"  Pending:              {len(pending)}")
    print(f"  Output JSONL:         {jsonl_path}")
    print(f"  Output PNG:           {png_path}")
    print(f"  Output meta:          {meta_path}")
    print()

    _write_benchmark_metadata(
        meta_path=meta_path,
        args=args,
        base_cfg=base_cfg,
        all_seeds=all_seeds,
        already_done=sorted(done),
        jsonl_path=jsonl_path,
        png_path=png_path,
    )

    durations: list[float] = []
    for i, seed in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] seed={seed}", flush=True)
        try:
            row = _run_one(args, base_cfg, seed)
        except Exception as e:
            print(f"  FAILED: {e!r}")
            continue
        durations.append(row["elapsed_sec"])

        with jsonl_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        mean_dt = sum(durations) / len(durations)
        remaining = len(pending) - i
        eta_sec = int(mean_dt * remaining)
        eta_h, rem = divmod(eta_sec, 3600)
        eta_m = rem // 60
        print(
            f"  done in {row['elapsed_sec']:.1f}s  |  "
            f"GQE F={row['gqe_F']:.4f} CX={row['gqe_cnot']} D={row['gqe_depth']}  |  "
            f"Qiskit CX={row['qiskit_cnot']} D={row['qiskit_depth']}  |  "
            f"ETA {eta_h:d}h{eta_m:02d}m"
        )

    print(f"\nBenchmark complete. Wrote {len(pending)} new rows to {jsonl_path}.")

    if not args.no_plot:
        rows = _load_rows(jsonl_path)
        _plot(rows, png_path, fidelity_threshold=base_cfg.reward.fidelity_threshold)


if __name__ == "__main__":
    main()
