"""Benchmark GQE and Qiskit on matched target-unitary inputs.

Each row trains GQE from the configured seed, verifies the Qiskit-transpiled
unitary fidelity, records threshold-constrained GQE optima, and writes paired
statistics suitable for later analysis.
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

from config import GQEConfig, load_config
from operator_pool import build_operator_pool
from reporting import circuit_stats
from simplify import simplify_pareto_archive
from target import build_target
from trainer import gqe


class _FileLogger:
    """Mirror log calls to stdout and a log file simultaneously."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", buffering=1)

    def log(self, msg: str = "") -> None:
        print(msg)
        self._file.write(msg + "\n")

    def close(self) -> None:
        self._file.close()


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
                   help="Directory for the PNG / CSV / log outputs "
                        "(default: results/benchmarks).")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation (useful for headless dispatch).")
    p.add_argument("--target-type", choices=["random", "random_reachable", "haar_random", "brickwork"],
                   default=None,
                   help="Override target.type from the YAML config.")
    p.add_argument("--brickwork-depth", type=int, default=None,
                   help="Override target.brickwork_depth for brickwork targets.")
    p.add_argument("--qiskit-seed", type=int, default=1234,
                   help="seed_transpiler passed to Qiskit transpile (default: 1234).")
    p.add_argument("--qiskit-seeds", default=None,
                   help="Comma-separated seed_transpiler sweep. Overrides --qiskit-seed.")
    p.add_argument("--fidelity-thresholds", default="0.99,0.999,0.9999",
                   help="Comma-separated fidelity thresholds for constrained metrics.")
    return p.parse_args()


def _parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        thr = float(raw)
        if not (0.0 < thr <= 1.0):
            raise ValueError("fidelity thresholds must be in (0, 1]")
        thresholds.append(thr)
    if not thresholds:
        raise ValueError("at least one fidelity threshold is required")
    return thresholds


def _parse_int_list(value: str | None, fallback: int) -> list[int]:
    if value is None:
        return [int(fallback)]
    vals = [int(raw.strip()) for raw in value.split(",") if raw.strip()]
    if not vals:
        raise ValueError("seed list cannot be empty")
    return vals


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _override_cfg(
    base_cfg: GQEConfig,
    args: argparse.Namespace,
    seed: int,
) -> GQEConfig:
    """Apply benchmark-specific target, logging, and seed overrides."""
    num_qubits = int(args.num_qubits)
    target_type = args.target_type or base_cfg.target.type
    brickwork_depth = (
        int(args.brickwork_depth)
        if args.brickwork_depth is not None
        else base_cfg.target.brickwork_depth
    )
    target = dataclasses.replace(
        base_cfg.target,
        num_qubits=num_qubits,
        type=target_type,
        brickwork_depth=brickwork_depth,
        path=None if target_type != "file" else base_cfg.target.path,
    )
    logging = dataclasses.replace(base_cfg.logging, wandb=False)
    training = dataclasses.replace(base_cfg.training, seed=seed)
    return dataclasses.replace(
        base_cfg, target=target, logging=logging, training=training,
    )


def _qiskit_compile(u_target, num_qubits: int, rotation_gates, *, seed_transpiler: int):
    """Transpile a dense target unitary and verify the compiled process fidelity."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(u_target), list(range(num_qubits)))
    basis = [*rotation_gates, "sx", "cx"]
    t0 = time.time()
    compiled = transpile(
        qc,
        basis_gates=basis,
        optimization_level=3,
        seed_transpiler=seed_transpiler,
    )
    elapsed = time.time() - t0
    u_compiled = Operator(compiled).data
    dim = u_target.shape[0]
    fidelity = float(abs(np.trace(u_target.conj().T @ u_compiled)) ** 2 / (dim * dim))
    depth, total, _ = circuit_stats(compiled)
    return compiled, depth, total, int(compiled.count_ops().get("cx", 0)), fidelity, elapsed


def _threshold_payload(key: str, best_cnot, best_depth, best_total) -> dict:
    payload = {
        f"min_cnot_at_F_{key}": None,
        f"F_for_min_cnot_at_F_{key}": None,
        f"depth_for_min_cnot_at_F_{key}": None,
        f"total_gates_for_min_cnot_at_F_{key}": None,
        f"min_depth_at_F_{key}": None,
        f"F_for_min_depth_at_F_{key}": None,
        f"cnot_for_min_depth_at_F_{key}": None,
        f"total_gates_for_min_depth_at_F_{key}": None,
        f"min_total_gates_at_F_{key}": None,
        f"F_for_min_total_gates_at_F_{key}": None,
        f"cnot_for_min_total_gates_at_F_{key}": None,
        f"depth_for_min_total_gates_at_F_{key}": None,
    }
    if best_cnot is not None:
        payload.update({
            f"min_cnot_at_F_{key}": int(best_cnot.cnot_count),
            f"F_for_min_cnot_at_F_{key}": float(best_cnot.fidelity),
            f"depth_for_min_cnot_at_F_{key}": int(best_cnot.depth),
            f"total_gates_for_min_cnot_at_F_{key}": int(best_cnot.total_gates),
        })
    if best_depth is not None:
        payload.update({
            f"min_depth_at_F_{key}": int(best_depth.depth),
            f"F_for_min_depth_at_F_{key}": float(best_depth.fidelity),
            f"cnot_for_min_depth_at_F_{key}": int(best_depth.cnot_count),
            f"total_gates_for_min_depth_at_F_{key}": int(best_depth.total_gates),
        })
    if best_total is not None:
        payload.update({
            f"min_total_gates_at_F_{key}": int(best_total.total_gates),
            f"F_for_min_total_gates_at_F_{key}": float(best_total.fidelity),
            f"cnot_for_min_total_gates_at_F_{key}": int(best_total.cnot_count),
            f"depth_for_min_total_gates_at_F_{key}": int(best_total.depth),
        })
    return payload


def _gqe_summary(result, thresholds: list[float]) -> dict:
    """Return best-fidelity and threshold-constrained GQE metrics."""
    arc = result.pareto_archive
    out: dict = {}
    if arc is not None and len(arc) > 0:
        bf = arc.best_by_fidelity()
        assert bf is not None
        points = arc.to_sorted_list()
        fids = np.asarray([p.fidelity for p in points], dtype=np.float64)
        depths = np.asarray([p.depth for p in points], dtype=np.float64)
        totals = np.asarray([p.total_gates for p in points], dtype=np.float64)
        cnots = np.asarray([p.cnot_count for p in points], dtype=np.float64)
        out.update({
            "best_F": float(bf.fidelity),
            "best_depth": int(bf.depth),
            "best_total_gates": int(bf.total_gates),
            "best_cnot": int(bf.cnot_count),
            "pareto_size": int(len(points)),
            "pareto_F_min": float(np.min(fids)),
            "pareto_F_median": float(np.median(fids)),
            "pareto_F_max": float(np.max(fids)),
            "pareto_cnot_min": int(np.min(cnots)),
            "pareto_cnot_median": float(np.median(cnots)),
            "pareto_cnot_max": int(np.max(cnots)),
            "pareto_depth_min": int(np.min(depths)),
            "pareto_depth_median": float(np.median(depths)),
            "pareto_depth_max": int(np.max(depths)),
            "pareto_total_gates_min": int(np.min(totals)),
            "pareto_total_gates_median": float(np.median(totals)),
            "pareto_total_gates_max": int(np.max(totals)),
            "pareto_hv_cnot": float(arc.hypervolume_2d()),
        })
        primary = arc.best_by_cnot(min_fidelity=thresholds[0])
        selected = primary if primary is not None else bf
        out.update({
            "selected_F": float(selected.fidelity),
            "selected_depth": int(selected.depth),
            "selected_total_gates": int(selected.total_gates),
            "selected_cnot": int(selected.cnot_count),
            "selected_rule": (
                f"min_cnot_at_{thresholds[0]:.6g}"
                if primary is not None else "best_fidelity"
            ),
        })
        for thr in thresholds:
            key = _threshold_key(thr)
            best_c = arc.best_by_cnot(min_fidelity=thr)
            best_d = arc.best_by_depth(min_fidelity=thr)
            best_g = arc.best_by_total_gates(min_fidelity=thr)
            out[f"success_at_F_{key}"] = bool(best_c is not None)
            out.update(_threshold_payload(key, best_c, best_d, best_g))
        return out

    F = (
        result.refined_raw_fidelity
        if result.refined_raw_fidelity is not None
        else result.best_raw_fidelity
    )
    F = float(F) if F is not None else float("nan")
    out.update({
        "best_F": F,
        "best_depth": None,
        "best_total_gates": None,
        "best_cnot": None,
        "selected_F": F,
        "selected_depth": None,
        "selected_total_gates": None,
        "selected_cnot": None,
        "selected_rule": "raw_fallback",
        "pareto_size": 0,
        "pareto_F_min": None,
        "pareto_F_median": None,
        "pareto_F_max": None,
        "pareto_cnot_min": None,
        "pareto_cnot_median": None,
        "pareto_cnot_max": None,
        "pareto_depth_min": None,
        "pareto_depth_median": None,
        "pareto_depth_max": None,
        "pareto_total_gates_min": None,
        "pareto_total_gates_median": None,
        "pareto_total_gates_max": None,
        "pareto_hv_cnot": None,
    })
    for thr in thresholds:
        key = _threshold_key(thr)
        out[f"success_at_F_{key}"] = bool(np.isfinite(F) and F >= thr)
        out.update(_threshold_payload(key, None, None, None))
    return out


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


def _bootstrap_median_ci(values: np.ndarray, *, seed: int = 0, n_boot: int = 2000) -> list[float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    meds = np.median(samples, axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return [float(lo), float(hi)]


def _wilcoxon_pvalue(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.all(values == 0.0):
        return None
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    try:
        return float(wilcoxon(values).pvalue)
    except Exception:
        return None


def _success_rate_ci(k: int, n: int) -> list[float | None]:
    if n <= 0:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def _finite_values(rows: list[dict], key: str, fallback: str | None = None) -> np.ndarray:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None and fallback is not None:
            value = row.get(fallback)
        if value is None:
            continue
        vals.append(value)
    arr = np.asarray(vals, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _basic_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "q25": None,
            "q75": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def _paired_delta_summary(rows: list[dict], gqe_col: str, qiskit_col: str) -> dict:
    deltas = []
    for row in rows:
        gqe_val = row.get(gqe_col)
        qiskit_val = row.get(qiskit_col)
        if gqe_val is None or qiskit_val is None:
            continue
        deltas.append(float(gqe_val) - float(qiskit_val))
    arr = np.asarray(deltas, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "paired_count": int(arr.size),
        "delta_gqe_minus_qiskit": _basic_stats(arr),
        "median_delta_ci95": _bootstrap_median_ci(arr),
        "wilcoxon_delta_pvalue": _wilcoxon_pvalue(arr),
        "fraction_no_worse_than_qiskit": (
            float(np.mean(arr <= 0.0)) if arr.size else None
        ),
    }


def _circuit_metric_block(
    rows: list[dict],
    *,
    fidelity_col: str,
    cnot_col: str,
    depth_col: str,
    total_col: str,
) -> dict:
    return {
        "fidelity": _basic_stats(_finite_values(rows, fidelity_col)),
        "cnot": _basic_stats(_finite_values(rows, cnot_col)),
        "depth": _basic_stats(_finite_values(rows, depth_col)),
        "total_gates": _basic_stats(_finite_values(rows, total_col)),
        "vs_qiskit": {
            "cnot": _paired_delta_summary(rows, cnot_col, "qiskit_cnot"),
            "depth": _paired_delta_summary(rows, depth_col, "qiskit_depth"),
            "total_gates": _paired_delta_summary(rows, total_col, "qiskit_total_gates"),
        },
    }


def _summary_stats(rows: list[dict], thresholds: list[float]) -> dict:
    rows = [r for r in rows if r.get("gqe_F") is not None]
    n = len(rows)
    if n == 0:
        return {"num_rows": 0}

    gqe_f = np.asarray([r.get("gqe_F", np.nan) for r in rows], dtype=np.float64)
    qiskit_f = np.asarray([r.get("qiskit_F", 1.0) for r in rows], dtype=np.float64)
    summary: dict = {
        "num_rows": n,
        "gqe_fidelity": _basic_stats(gqe_f),
        "qiskit_fidelity": _basic_stats(qiskit_f),
        "best_fidelity_circuit": _circuit_metric_block(
            rows,
            fidelity_col="gqe_F",
            cnot_col="gqe_best_cnot",
            depth_col="gqe_best_depth",
            total_col="gqe_best_total_gates",
        ),
        "selected_primary_circuit": _circuit_metric_block(
            rows,
            fidelity_col="gqe_selected_F",
            cnot_col="gqe_cnot",
            depth_col="gqe_depth",
            total_col="gqe_total_gates",
        ),
        "qiskit_circuit": {
            "cnot": _basic_stats(_finite_values(rows, "qiskit_cnot")),
            "depth": _basic_stats(_finite_values(rows, "qiskit_depth")),
            "total_gates": _basic_stats(_finite_values(rows, "qiskit_total_gates")),
            "elapsed_sec": _basic_stats(_finite_values(rows, "qiskit_elapsed_sec")),
        },
        "runtime": {
            "gqe_elapsed_sec": _basic_stats(_finite_values(rows, "elapsed_sec")),
        },
        "pareto_archive": {
            "size": _basic_stats(_finite_values(rows, "gqe_pareto_size")),
            "hypervolume_cnot": _basic_stats(_finite_values(rows, "gqe_pareto_hv_cnot")),
            "fidelity_min": _basic_stats(_finite_values(rows, "gqe_pareto_F_min")),
            "fidelity_median": _basic_stats(_finite_values(rows, "gqe_pareto_F_median")),
            "cnot_min": _basic_stats(_finite_values(rows, "gqe_pareto_cnot_min")),
            "depth_min": _basic_stats(_finite_values(rows, "gqe_pareto_depth_min")),
            "total_gates_min": _basic_stats(_finite_values(rows, "gqe_pareto_total_gates_min")),
        },
        "thresholds": {},
    }

    for thr in thresholds:
        key = _threshold_key(thr)
        success_col = f"gqe_success_at_F_{key}"
        cnot_col = f"gqe_min_cnot_at_F_{key}"
        depth_col = f"gqe_min_depth_at_F_{key}"
        total_col = f"gqe_min_total_gates_at_F_{key}"
        successes = np.asarray([bool(r.get(success_col, False)) for r in rows], dtype=bool)
        k = int(successes.sum())
        passed = [r for r in rows if r.get(success_col, False)]
        min_cnot_block = _circuit_metric_block(
            passed,
            fidelity_col=f"gqe_F_for_min_cnot_at_F_{key}",
            cnot_col=cnot_col,
            depth_col=f"gqe_depth_for_min_cnot_at_F_{key}",
            total_col=f"gqe_total_gates_for_min_cnot_at_F_{key}",
        )
        min_depth_block = _circuit_metric_block(
            passed,
            fidelity_col=f"gqe_F_for_min_depth_at_F_{key}",
            cnot_col=f"gqe_cnot_for_min_depth_at_F_{key}",
            depth_col=depth_col,
            total_col=f"gqe_total_gates_for_min_depth_at_F_{key}",
        )
        min_total_block = _circuit_metric_block(
            passed,
            fidelity_col=f"gqe_F_for_min_total_gates_at_F_{key}",
            cnot_col=f"gqe_cnot_for_min_total_gates_at_F_{key}",
            depth_col=f"gqe_depth_for_min_total_gates_at_F_{key}",
            total_col=total_col,
        )
        summary["thresholds"][key] = {
            "threshold": float(thr),
            "success_count": k,
            "success_rate": float(k / n),
            "success_rate_ci95": _success_rate_ci(k, n),
            "paired_count": min_cnot_block["vs_qiskit"]["cnot"]["paired_count"],
            "median_cnot_delta_gqe_minus_qiskit": (
                min_cnot_block["vs_qiskit"]["cnot"]["delta_gqe_minus_qiskit"]["median"]
            ),
            "median_cnot_delta_ci95": min_cnot_block["vs_qiskit"]["cnot"]["median_delta_ci95"],
            "wilcoxon_cnot_delta_pvalue": (
                min_cnot_block["vs_qiskit"]["cnot"]["wilcoxon_delta_pvalue"]
            ),
            "median_depth_delta_gqe_minus_qiskit": (
                min_depth_block["vs_qiskit"]["depth"]["delta_gqe_minus_qiskit"]["median"]
            ),
            "median_depth_delta_ci95": min_depth_block["vs_qiskit"]["depth"]["median_delta_ci95"],
            "wilcoxon_depth_delta_pvalue": (
                min_depth_block["vs_qiskit"]["depth"]["wilcoxon_delta_pvalue"]
            ),
            "median_total_gates_delta_gqe_minus_qiskit": (
                min_total_block["vs_qiskit"]["total_gates"]["delta_gqe_minus_qiskit"]["median"]
            ),
            "median_total_gates_delta_ci95": (
                min_total_block["vs_qiskit"]["total_gates"]["median_delta_ci95"]
            ),
            "wilcoxon_total_gates_delta_pvalue": (
                min_total_block["vs_qiskit"]["total_gates"]["wilcoxon_delta_pvalue"]
            ),
            "fraction_cnot_no_worse_than_qiskit": (
                min_cnot_block["vs_qiskit"]["cnot"]["fraction_no_worse_than_qiskit"]
            ),
            "fraction_depth_no_worse_than_qiskit": (
                min_depth_block["vs_qiskit"]["depth"]["fraction_no_worse_than_qiskit"]
            ),
            "fraction_total_gates_no_worse_than_qiskit": (
                min_total_block["vs_qiskit"]["total_gates"]["fraction_no_worse_than_qiskit"]
            ),
            "min_cnot_circuit": min_cnot_block,
            "min_depth_circuit": min_depth_block,
            "min_total_gates_circuit": min_total_block,
        }
    return summary


def _write_summary(rows: list[dict], thresholds: list[float], summary_path: Path) -> None:
    payload = _summary_stats(rows, thresholds)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote benchmark summary → {summary_path}")


def _run_one(args, base_cfg: GQEConfig, seed: int) -> dict:
    thresholds = _parse_thresholds(args.fidelity_thresholds)
    cfg = _override_cfg(base_cfg, args, seed)
    pool = build_operator_pool(args.num_qubits, cfg.pool.rotation_gates)
    u_target, desc = build_target(pool, cfg)

    qiskit_seeds = _parse_int_list(args.qiskit_seeds, args.qiskit_seed)
    qiskit_runs = []
    for q_seed in qiskit_seeds:
        _, q_depth, q_total, q_cnot, q_fid, q_dt = _qiskit_compile(
            u_target,
            args.num_qubits,
            cfg.pool.rotation_gates,
            seed_transpiler=q_seed,
        )
        qiskit_runs.append({
            "seed_transpiler": int(q_seed),
            "depth": int(q_depth),
            "total_gates": int(q_total),
            "cnot": int(q_cnot),
            "fidelity": float(q_fid),
            "elapsed_sec": float(q_dt),
        })
    q_best = min(
        qiskit_runs,
        key=lambda r: (r["cnot"], r["depth"], r["total_gates"], -r["fidelity"]),
    )

    t0 = time.time()
    result = gqe(cfg, u_target, pool, logger=None)
    dt = time.time() - t0

    if result.pareto_archive is not None and len(result.pareto_archive) > 0:
        result.pareto_archive = simplify_pareto_archive(
            result.pareto_archive, pool, args.num_qubits,
        )

    pareto_rows: list[dict] = []
    arc = result.pareto_archive
    if arc is not None:
        for pt in arc.to_sorted_list():
            pareto_rows.append({
                "seed": seed,
                "num_qubits": args.num_qubits,
                "target_type": cfg.target.type,
                "target_desc": desc,
                "fidelity": float(pt.fidelity),
                "cnot_count": int(pt.cnot_count),
                "depth": int(pt.depth),
                "total_gates": int(pt.total_gates),
            })

    gqe_b = _gqe_summary(result, thresholds)

    row = {
        "seed": seed,
        "num_qubits": args.num_qubits,
        "target_type": cfg.target.type,
        "target_desc": desc,
        "elapsed_sec": dt,
        "gqe_F": gqe_b["best_F"],
        "gqe_best_depth": gqe_b["best_depth"],
        "gqe_best_total_gates": gqe_b["best_total_gates"],
        "gqe_best_cnot": gqe_b["best_cnot"],
        "gqe_selected_F": gqe_b["selected_F"],
        "gqe_depth": gqe_b["selected_depth"],
        "gqe_total_gates": gqe_b["selected_total_gates"],
        "gqe_cnot": gqe_b["selected_cnot"],
        "gqe_selection": gqe_b["selected_rule"],
        "qiskit_depth": int(q_best["depth"]),
        "qiskit_total_gates": int(q_best["total_gates"]),
        "qiskit_cnot": int(q_best["cnot"]),
        "qiskit_F": float(q_best["fidelity"]),
        "qiskit_elapsed_sec": float(sum(r["elapsed_sec"] for r in qiskit_runs)),
        "qiskit_seed_transpiler": int(q_best["seed_transpiler"]),
        "qiskit_seed_sweep": qiskit_runs,
    }
    row.update({f"gqe_{k}": v for k, v in gqe_b.items() if k not in {
        "best_F",
        "best_depth",
        "best_total_gates",
        "best_cnot",
        "selected_F",
        "selected_depth",
        "selected_total_gates",
        "selected_cnot",
        "selected_rule",
    }})
    return row, pareto_rows


def _plot_main(rows: list[dict], png_path: Path) -> None:
    """4-panel figure: fidelity / CNOT / depth / total-gates, sorted by GQE best fidelity.

    All four panels are built from the same sorted list of per-circuit records so that
    column i in every panel always refers to the identical circuit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("gqe_F") is not None]
    if not rows:
        print("No rows to plot.")
        return

    n_qubits = int(rows[0]["num_qubits"])
    target_type = rows[0]["target_type"]

    def _v(row: dict, key: str) -> float:
        val = row.get(key)
        return float(val) if val is not None else np.nan

    # Sort all per-circuit data together as a single unit — this is the guarantee
    # that column i in every panel below shows the same circuit.
    records = sorted(
        [
            {
                "seed":     row.get("seed"),
                "f":        _v(row, "gqe_F"),
                "gqe_cnot": _v(row, "gqe_best_cnot"),
                "gqe_dep":  _v(row, "gqe_best_depth"),
                "gqe_tot":  _v(row, "gqe_best_total_gates"),
                "qk_cnot":  _v(row, "qiskit_cnot"),
                "qk_dep":   _v(row, "qiskit_depth"),
                "qk_tot":   _v(row, "qiskit_total_gates"),
            }
            for row in rows
        ],
        key=lambda d: d["f"],
    )

    n         = len(records)
    seeds     = [d["seed"] for d in records]
    f         = np.array([d["f"]        for d in records])
    gqe_cnot  = np.array([d["gqe_cnot"] for d in records])
    gqe_dep   = np.array([d["gqe_dep"]  for d in records])
    gqe_tot   = np.array([d["gqe_tot"]  for d in records])
    qk_cnot   = np.array([d["qk_cnot"]  for d in records])
    qk_dep    = np.array([d["qk_dep"]   for d in records])
    qk_tot    = np.array([d["qk_tot"]   for d in records])

    xs = np.arange(n)
    width = 0.4

    fig, axes = plt.subplots(
        4, 1, sharex=True,
        figsize=(max(10, n * 0.32 + 2), 12),
        gridspec_kw={"height_ratios": [1.5, 1, 1, 1]},
    )

    # Panel 1 — fidelity (each dot = same circuit as bars directly below it)
    ax = axes[0]
    ax.scatter(xs, f, color="C0", s=30, zorder=3, label="GQE best fidelity")
    ax.axhline(float(np.nanmedian(f)), color="C0", linestyle="--", linewidth=1.0,
               alpha=0.6, label=f"GQE median {np.nanmedian(f):.4f}")
    ax.set_ylabel("Process fidelity")
    ax.set_ylim(max(0.0, float(np.nanmin(f)) - 0.02), 1.005)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        f"GQE best-fidelity circuits vs Qiskit — {n} {target_type} {n_qubits}q targets",
        fontsize=12,
    )
    stats_txt = (
        f"N={n}  mean={np.nanmean(f):.5f}  median={np.nanmedian(f):.5f}  "
        f"min={np.nanmin(f):.5f}"
    )
    ax.text(
        0.02, 0.05, stats_txt, transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="0.7"),
    )

    # Panels 2-4 — metrics of the SAME circuit whose fidelity dot sits above
    panels = [
        (axes[1], gqe_cnot, qk_cnot, "CNOT count"),
        (axes[2], gqe_dep,  qk_dep,  "Circuit depth"),
        (axes[3], gqe_tot,  qk_tot,  "Total gates"),
    ]
    for ax, gqe_v, qk_v, ylabel in panels:
        ax.bar(xs - width / 2, gqe_v, width, color="C0", label="GQE")
        ax.bar(xs + width / 2, qk_v,  width, color="C1", label="Qiskit")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)

    # x-axis: seed labels so each column is traceable to the JSONL / Pareto CSV
    step = max(1, n // 20)
    tick_xs    = xs[::step]
    tick_seeds = seeds[::step]
    axes[-1].set_xticks(tick_xs)
    axes[-1].set_xticklabels([str(s) for s in tick_seeds], rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Seed (sorted by GQE best fidelity, ascending)")

    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot → {png_path}")


def _write_pareto_csv(pareto_rows: list[dict], csv_path: Path) -> None:
    """Write one row per Pareto-front point across all benchmark circuits."""
    import csv
    if not pareto_rows:
        print("No Pareto points recorded — skipping CSV.")
        return
    fields = ["seed", "num_qubits", "target_type", "target_desc",
              "fidelity", "cnot_count", "depth", "total_gates"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pareto_rows)
    print(f"Saved Pareto CSV → {csv_path} ({len(pareto_rows)} points across {len({r['seed'] for r in pareto_rows})} circuits)")


def _write_benchmark_metadata(
    *,
    meta_path: Path,
    args: argparse.Namespace,
    benchmark_cfg: GQEConfig,
    all_seeds: list[int],
    already_done: list[int],
    jsonl_path: Path,
    png_path: Path,
    summary_path: Path,
    thresholds: list[float],
) -> None:
    """Write reproducibility metadata next to the benchmark JSONL."""
    import getpass
    import platform
    import socket
    import subprocess
    import sys
    from datetime import datetime, timezone

    from reporting import _config_snapshot

    def _pkg_version(name: str) -> str | None:
        try:
            import importlib.metadata as importlib_metadata
            return importlib_metadata.version(name)
        except Exception:
            return None

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except Exception:
        git_commit = None

    target_type = benchmark_cfg.target.type
    payload = {
        "kind": "benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "git_commit": git_commit,
        "dependencies": {
            "jax": _pkg_version("jax"),
            "jaxlib": _pkg_version("jaxlib"),
            "flax": _pkg_version("flax"),
            "optax": _pkg_version("optax"),
            "numpy": _pkg_version("numpy"),
            "qiskit": _pkg_version("qiskit"),
            "scipy": _pkg_version("scipy"),
        },
        "experiment": {
            "num_qubits": int(args.num_qubits),
            "target_type": target_type,
            "brickwork_depth": (
                int(benchmark_cfg.target.brickwork_depth or (2 * args.num_qubits))
                if target_type == "brickwork" else None
            ),
            "num_circuits": int(args.num_circuits),
            "base_seed": int(args.base_seed),
            "seeds": all_seeds,
            "already_completed_seeds": already_done,
            "config_path": args.config,
            "fidelity_threshold": float(benchmark_cfg.reward.fidelity_threshold),
            "fidelity_thresholds": thresholds,
            "qiskit_seed_transpiler": int(args.qiskit_seed),
            "qiskit_seed_sweep": _parse_int_list(args.qiskit_seeds, args.qiskit_seed),
        },
        "outputs": {
            "jsonl": str(jsonl_path),
            "png": str(png_path),
            "summary_json": str(summary_path),
        },
        "config": _config_snapshot(benchmark_cfg),
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote benchmark metadata → {meta_path}")


def main():
    args = _parse_args()
    thresholds = _parse_thresholds(args.fidelity_thresholds)
    load_dotenv()

    base_cfg = load_config(args.config)
    benchmark_cfg = _override_cfg(base_cfg, args, args.base_seed)
    target_type = benchmark_cfg.target.type
    if target_type == "file":
        raise ValueError(
            "Cannot benchmark with target.type='file' — the benchmark generates "
            "fresh targets per seed. Use 'haar_random' or 'brickwork' in config.yml."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = output_dir / f"benchmark_{args.num_qubits}q_{target_type}_{stamp}"
    png_path = stem.with_suffix(".png")
    csv_path = Path(str(stem) + ".pareto.csv")
    log_path = stem.with_suffix(".log.txt")

    all_seeds = [args.base_seed + i for i in range(args.num_circuits)]

    logger = _FileLogger(log_path)

    # ── metadata header ───────────────────────────────────────────────────
    logger.log("=" * 70)
    logger.log("BENCHMARK METADATA")
    logger.log("=" * 70)
    logger.log(f"Timestamp (UTC):      {datetime.utcnow().isoformat(timespec='seconds')}")
    logger.log(f"Config:               {args.config}")
    logger.log(f"Qubits:               {args.num_qubits}")
    logger.log(f"Target type:          {target_type}")
    if target_type == "brickwork":
        bd = benchmark_cfg.target.brickwork_depth or (2 * args.num_qubits)
        logger.log(f"Brickwork depth:      {bd}")
    logger.log(f"Fidelity thresholds:  {thresholds}")
    logger.log(f"Base seed:            {args.base_seed}")
    logger.log(f"Total circuits:       {args.num_circuits}")
    logger.log(f"Qiskit seed(s):       {_parse_int_list(args.qiskit_seeds, args.qiskit_seed)}")
    logger.log("")
    logger.log("Output files:")
    logger.log(f"  Plot (PNG):         {png_path}")
    logger.log(f"  Pareto CSV:         {csv_path}")
    logger.log(f"  Log (this file):    {log_path}")
    logger.log("=" * 70)
    logger.log("")

    # ── main training loop ────────────────────────────────────────────────
    all_rows: list[dict] = []
    all_pareto_rows: list[dict] = []
    durations: list[float] = []

    for i, seed in enumerate(all_seeds, start=1):
        logger.log(f"[{i}/{len(all_seeds)}] seed={seed}")
        try:
            row, pareto_rows = _run_one(args, base_cfg, seed)
        except Exception as e:
            logger.log(f"  FAILED: {e!r}")
            continue

        all_rows.append(row)
        durations.append(row["elapsed_sec"])
        all_pareto_rows.extend(pareto_rows)

        mean_dt = sum(durations) / len(durations)
        remaining = len(all_seeds) - i
        eta_sec = int(mean_dt * remaining)
        eta_h, rem = divmod(eta_sec, 3600)
        eta_m = rem // 60
        logger.log(
            f"  done in {row['elapsed_sec']:.1f}s  |  "
            f"GQE F={row['gqe_F']:.4f} CX={row['gqe_cnot']} D={row['gqe_depth']} "
            f"({row['gqe_selection']})  |  "
            f"Qiskit F={row['qiskit_F']:.4f} CX={row['qiskit_cnot']} "
            f"D={row['qiskit_depth']}  |  "
            f"ETA {eta_h:d}h{eta_m:02d}m"
        )

    logger.log("")
    logger.log(f"Benchmark complete. {len(all_rows)} circuits run.")

    # ── write the three output files ──────────────────────────────────────
    _write_pareto_csv(all_pareto_rows, csv_path)

    if not args.no_plot:
        _plot_main(all_rows, png_path)

    logger.log("")
    logger.log("Outputs written:")
    logger.log(f"  {png_path}")
    logger.log(f"  {csv_path}")
    logger.log(f"  {log_path}")
    logger.close()


if __name__ == "__main__":
    main()
