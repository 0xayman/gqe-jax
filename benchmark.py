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
from target import build_target
from trainer import gqe


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
    """Apply benchmark-specific target, logging, seed, and cap overrides."""
    from config import default_max_gates_count
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
    if base_cfg.model.auto_max_gates_count:
        model = dataclasses.replace(
            base_cfg.model,
            max_gates_count=default_max_gates_count(num_qubits),
            auto_max_gates_count=True,
        )
    else:
        model = base_cfg.model
    training = dataclasses.replace(base_cfg.training, seed=seed)
    return dataclasses.replace(
        base_cfg, target=target, logging=logging, model=model, training=training,
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
    gqe_b = _gqe_summary(result, thresholds)

    row = {
        "seed": seed,
        "num_qubits": args.num_qubits,
        "target_type": cfg.target.type,
        "target_desc": desc,
        "max_gates_count": int(cfg.model.max_gates_count),
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
    return row


def _plot_paths(png_path: Path) -> dict[str, Path]:
    return {
        "best_fidelity": png_path,
        "thresholds": png_path.with_name(f"{png_path.stem}.thresholds.png"),
        "pareto": png_path.with_name(f"{png_path.stem}.pareto.png"),
    }


def _plot_value_array(
    rows: list[dict],
    key: str,
    *,
    fallback: str | None = None,
    missing: float = np.nan,
) -> np.ndarray:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None and fallback is not None:
            value = row.get(fallback)
        vals.append(missing if value is None else value)
    return np.asarray(vals, dtype=np.float64)


def _threshold_plot_stats(rows: list[dict], thresholds: list[float]) -> dict[str, list[float]]:
    stats = {
        "success": [],
        "success_lo": [],
        "success_hi": [],
        "cnot_delta": [],
        "depth_delta": [],
        "total_delta": [],
        "cnot_no_worse": [],
        "depth_no_worse": [],
        "total_no_worse": [],
    }
    n = len(rows)
    for thr in thresholds:
        key = _threshold_key(thr)
        passed = [r for r in rows if r.get(f"gqe_success_at_F_{key}", False)]
        k = len(passed)
        lo, hi = _success_rate_ci(k, n)
        stats["success"].append(k / n if n else np.nan)
        stats["success_lo"].append(0.0 if lo is None else k / n - lo)
        stats["success_hi"].append(0.0 if hi is None else hi - k / n)

        triplets = [
            (
                "cnot_delta",
                "cnot_no_worse",
                f"gqe_min_cnot_at_F_{key}",
                "qiskit_cnot",
            ),
            (
                "depth_delta",
                "depth_no_worse",
                f"gqe_min_depth_at_F_{key}",
                "qiskit_depth",
            ),
            (
                "total_delta",
                "total_no_worse",
                f"gqe_min_total_gates_at_F_{key}",
                "qiskit_total_gates",
            ),
        ]
        for delta_key, frac_key, gqe_col, qiskit_col in triplets:
            vals = [
                float(r[gqe_col]) - float(r[qiskit_col])
                for r in passed
                if r.get(gqe_col) is not None and r.get(qiskit_col) is not None
            ]
            arr = np.asarray(vals, dtype=np.float64)
            stats[delta_key].append(float(np.median(arr)) if arr.size else np.nan)
            stats[frac_key].append(float(np.mean(arr <= 0.0)) if arr.size else np.nan)
    return stats


def _plot_best_fidelity(rows: list[dict], png_path: Path, thresholds: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("gqe_F") is not None]
    if not rows:
        print("No rows to plot — aborting best-fidelity plot.")
        return

    n = len(rows)
    fidelity_threshold = thresholds[0]
    f = _plot_value_array(rows, "gqe_F")
    gqe_cnot = _plot_value_array(rows, "gqe_best_cnot", fallback="gqe_cnot")
    gqe_depth = _plot_value_array(rows, "gqe_best_depth", fallback="gqe_depth")
    gqe_total = _plot_value_array(rows, "gqe_best_total_gates", fallback="gqe_total_gates")
    qk_f = _plot_value_array(rows, "qiskit_F")
    qk_cnot = _plot_value_array(rows, "qiskit_cnot")
    qk_depth = _plot_value_array(rows, "qiskit_depth")
    qk_total = _plot_value_array(rows, "qiskit_total_gates")

    order = np.argsort(f)
    xs = np.arange(n)
    n_qubits = int(rows[0]["num_qubits"])
    target_type = rows[0]["target_type"]
    numer = 4 ** n_qubits - 3 * n_qubits - 1
    cnot_lb = max(0, -(-numer // 4))

    fig, axes = plt.subplots(
        4, 1, sharex=True, figsize=(12, 10),
        gridspec_kw={"height_ratios": [1.5, 1, 1, 1]},
    )

    ax = axes[0]
    ax.scatter(xs, f[order], color="C0", s=22, zorder=3, label="GQE best fidelity")
    ax.axhline(float(np.nanmedian(qk_f)), color="C1", linestyle="--", linewidth=1.2,
               label=f"Qiskit median F={np.nanmedian(qk_f):.4f}")
    for thr in thresholds:
        ax.axhline(thr, linestyle=":", linewidth=1.0, label=f"F={thr:g}")
    ax.set_ylabel("Process fidelity")
    ax.set_ylim(max(0.0, float(np.nanmin(f)) - 0.02), 1.005)
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    stats = (
        "Lower panels use the same GQE circuit that achieved best fidelity.\n"
        f"N={n}; best-F mean={np.nanmean(f):.5f}; median={np.nanmedian(f):.5f}; "
        f"min={np.nanmin(f):.5f}\n"
        f"Success at F>={fidelity_threshold:g}: "
        f"{int(np.sum(f >= fidelity_threshold))}/{n}"
    )
    ax.text(
        0.02, 0.05, stats, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="0.7"),
    )

    width = 0.4
    panels = [
        (axes[1], gqe_cnot, qk_cnot, "CNOT count", cnot_lb),
        (axes[2], gqe_depth, qk_depth, "Circuit depth", None),
        (axes[3], gqe_total, qk_total, "Total gates", None),
    ]
    for ax, gqe_vals, qk_vals, ylabel, lower_bound in panels:
        ax.bar(xs - width / 2, gqe_vals[order], width, color="C0", label="GQE best-F circuit")
        ax.bar(xs + width / 2, qk_vals[order], width, color="C1", label="Qiskit")
        if lower_bound:
            ax.axhline(lower_bound, color="0.4", linestyle="--", linewidth=1.0,
                       label=f"K-G optimum = {lower_bound}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Unitary index (sorted by GQE best fidelity, ascending)")
    fig.suptitle(
        f"Best-fidelity GQE circuits vs Qiskit on {n} {target_type} {n_qubits}-qubit targets",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved best-fidelity plot → {png_path}")


def _plot_thresholds(rows: list[dict], png_path: Path, thresholds: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("gqe_F") is not None]
    if not rows:
        print("No rows to plot — aborting threshold plot.")
        return

    stats = _threshold_plot_stats(rows, thresholds)
    labels = [f"{thr:g}" for thr in thresholds]
    xs = np.arange(len(thresholds))
    width = 0.25
    n_qubits = int(rows[0]["num_qubits"])
    target_type = rows[0]["target_type"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].bar(
        xs,
        stats["success"],
        color="C2",
        yerr=[stats["success_lo"], stats["success_hi"]],
        capsize=4,
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Threshold-constrained Pareto-front metrics")

    axes[1].bar(xs - width, stats["cnot_delta"], width, label="min CNOT circuit", color="C0")
    axes[1].bar(xs, stats["depth_delta"], width, label="min depth circuit", color="C3")
    axes[1].bar(xs + width, stats["total_delta"], width, label="min total-gates circuit", color="C4")
    axes[1].axhline(0.0, color="0.35", linewidth=1.0)
    axes[1].set_ylabel("Median delta vs Qiskit")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(xs, stats["cnot_no_worse"], marker="o", label="CNOT no worse")
    axes[2].plot(xs, stats["depth_no_worse"], marker="o", label="Depth no worse")
    axes[2].plot(xs, stats["total_no_worse"], marker="o", label="Total gates no worse")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Fraction <= Qiskit")
    axes[2].set_xlabel("Fidelity threshold")
    axes[2].set_xticks(xs, labels)
    axes[2].legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Threshold tradeoffs on {len(rows)} {target_type} {n_qubits}-qubit targets",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved threshold plot → {png_path}")


def _boxplot_data(rows: list[dict], keys: list[str]) -> list[np.ndarray]:
    data = []
    for key in keys:
        arr = _finite_values(rows, key)
        data.append(arr if arr.size else np.asarray([np.nan]))
    return data


def _plot_pareto(rows: list[dict], png_path: Path, thresholds: list[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("gqe_F") is not None]
    if not rows:
        print("No rows to plot — aborting Pareto plot.")
        return

    labels = ["best-F"] + [f"F>={thr:g}" for thr in thresholds]
    cnot_keys = ["gqe_best_cnot"] + [
        f"gqe_min_cnot_at_F_{_threshold_key(thr)}" for thr in thresholds
    ]
    depth_keys = ["gqe_best_depth"] + [
        f"gqe_min_depth_at_F_{_threshold_key(thr)}" for thr in thresholds
    ]
    total_keys = ["gqe_best_total_gates"] + [
        f"gqe_min_total_gates_at_F_{_threshold_key(thr)}" for thr in thresholds
    ]
    n_qubits = int(rows[0]["num_qubits"])
    target_type = rows[0]["target_type"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    size = _finite_values(rows, "gqe_pareto_size")
    hv = _finite_values(rows, "gqe_pareto_hv_cnot")
    best_f = _finite_values(rows, "gqe_F")
    if size.size:
        axes[0].hist(size, bins=min(20, max(5, int(np.sqrt(size.size)))), color="C0", alpha=0.8)
        axes[0].set_xlabel("Archive size")
        axes[0].set_ylabel("Targets")
    else:
        axes[0].text(0.5, 0.5, "Archive size unavailable", ha="center", va="center")
    axes[0].set_title("Pareto archive size")

    if hv.size == best_f.size and hv.size:
        axes[1].scatter(hv, best_f, color="C2", s=24)
        axes[1].set_xlabel("Fidelity-CNOT hypervolume")
        axes[1].set_ylabel("Best fidelity")
    else:
        axes[1].text(0.5, 0.5, "Hypervolume unavailable", ha="center", va="center")
    axes[1].set_title("Archive quality")

    axes[2].boxplot(_boxplot_data(rows, cnot_keys), labels=labels, showfliers=False)
    axes[2].axhline(np.nanmedian(_plot_value_array(rows, "qiskit_cnot")), color="C1",
                    linestyle="--", linewidth=1.0, label="Qiskit median")
    axes[2].set_ylabel("CNOT count")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].legend(loc="best", fontsize=8)

    axes[3].boxplot(_boxplot_data(rows, depth_keys), labels=labels, showfliers=False)
    axes[3].axhline(np.nanmedian(_plot_value_array(rows, "qiskit_depth")), color="C1",
                    linestyle="--", linewidth=1.0, label="Qiskit median depth")
    axes[3].set_ylabel("Depth")
    axes[3].tick_params(axis="x", rotation=20)
    ax2 = axes[3].twinx()
    med_total = [
        float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        for vals in _boxplot_data(rows, total_keys)
    ]
    ax2.plot(np.arange(1, len(labels) + 1), med_total, color="C4", marker="o",
             label="Median total gates")
    ax2.set_ylabel("Median total gates")
    handles, labs = axes[3].get_legend_handles_labels()
    handles2, labs2 = ax2.get_legend_handles_labels()
    axes[3].legend(handles + handles2, labs + labs2, loc="best", fontsize=8)

    fig.suptitle(
        f"Pareto/archive diagnostics on {len(rows)} {target_type} {n_qubits}-qubit targets",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"Saved Pareto plot → {png_path}")


def _plot(rows: list[dict], png_path: Path, *, thresholds: list[float]) -> dict[str, Path]:
    """Write distinct plots for best-fidelity, threshold, and archive views."""
    plot_paths = _plot_paths(png_path)
    rows = [r for r in rows if r.get("gqe_F") is not None]
    if not rows:
        print("No rows to plot — aborting plot step.")
        return plot_paths
    _plot_best_fidelity(rows, plot_paths["best_fidelity"], thresholds)
    _plot_thresholds(rows, plot_paths["thresholds"], thresholds)
    _plot_pareto(rows, plot_paths["pareto"], thresholds)
    return plot_paths


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
    plot_paths = _plot_paths(png_path)
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
            "plots": {name: str(path) for name, path in plot_paths.items()},
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
    jsonl_path = output_dir / (
        f"benchmark_{args.num_qubits}q_{target_type}_{stamp}.jsonl"
    )
    png_path = jsonl_path.with_suffix(".png")
    meta_path = jsonl_path.with_suffix(".meta.json")
    summary_path = jsonl_path.with_suffix(".summary.json")

    if args.resume:
        existing = sorted(output_dir.glob(
            f"benchmark_{args.num_qubits}q_{target_type}_*.jsonl"
        ))
        if existing:
            jsonl_path = existing[-1]
            png_path = jsonl_path.with_suffix(".png")
            meta_path = jsonl_path.with_suffix(".meta.json")
            summary_path = jsonl_path.with_suffix(".summary.json")
            print(f"Resuming into existing file: {jsonl_path}")

    done = _completed_seeds(jsonl_path) if args.resume else set()
    all_seeds = [args.base_seed + i for i in range(args.num_circuits)]
    pending = [s for s in all_seeds if s not in done]
    plot_paths = _plot_paths(png_path)

    print(f"Benchmark configuration:")
    print(f"  Qubits:               {args.num_qubits}")
    print(f"  Target type:          {target_type}  (from {args.config})")
    if target_type == "brickwork":
        depth = benchmark_cfg.target.brickwork_depth or (2 * args.num_qubits)
        print(f"  Brickwork depth:      {depth}")
    print(f"  Max gate cap:         {benchmark_cfg.model.max_gates_count}")
    print(f"  Fidelity thresholds:  {thresholds}")
    print(f"  Total circuits:       {args.num_circuits}")
    print(f"  Already completed:    {len(done)}")
    print(f"  Pending:              {len(pending)}")
    print(f"  Output JSONL:         {jsonl_path}")
    print(f"  Output best-F PNG:    {plot_paths['best_fidelity']}")
    print(f"  Output threshold PNG: {plot_paths['thresholds']}")
    print(f"  Output Pareto PNG:    {plot_paths['pareto']}")
    print(f"  Output meta:          {meta_path}")
    print(f"  Output summary:       {summary_path}")
    print()

    _write_benchmark_metadata(
        meta_path=meta_path,
        args=args,
        benchmark_cfg=benchmark_cfg,
        all_seeds=all_seeds,
        already_done=sorted(done),
        jsonl_path=jsonl_path,
        png_path=png_path,
        summary_path=summary_path,
        thresholds=thresholds,
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
            f"GQE F={row['gqe_F']:.4f} CX={row['gqe_cnot']} D={row['gqe_depth']} "
            f"({row['gqe_selection']})  |  "
            f"Qiskit F={row['qiskit_F']:.4f} CX={row['qiskit_cnot']} "
            f"D={row['qiskit_depth']}  |  "
            f"ETA {eta_h:d}h{eta_m:02d}m"
        )

    print(f"\nBenchmark complete. Wrote {len(pending)} new rows to {jsonl_path}.")

    if not args.no_plot:
        rows = _load_rows(jsonl_path)
        _plot(rows, png_path, thresholds=thresholds)
    rows = _load_rows(jsonl_path)
    _write_summary(rows, thresholds, summary_path)


if __name__ == "__main__":
    main()
