"""Render the 2-qubit GQE vs Qiskit benchmark as a 3-panel figure.

Reads ``results/benchmark_3q.jsonl`` (produced by ``benchmark.py``) and writes
``results/benchmark_3q.png`` (+ a ``.pdf``). Re-run anytime; does not touch the
training pipeline.

Panels (shared x-axis, sorted by GQE fidelity ascending):
    1. Fidelity scatter: GQE circuits vs Qiskit reference (F = 1 dashed line).
    2. CNOT count bar chart: GQE vs Qiskit.
    3. Depth bar chart: GQE vs Qiskit.

For the CNOT / depth panels we use the Pareto-reported circuit (best-depth at
F >= threshold), which is the fairer structural comparison. The fidelity panel
uses the best-fidelity circuit so each axis answers the right question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KAK_CNOT_UPPER_BOUND = 3  # any 2q unitary decomposes into <= 3 CNOTs (KAK).
F_THRESHOLD = 0.99


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summary_text(rows: list[dict]) -> str:
    fids = np.array([r["gqe_fidelity"] for r in rows])
    gqe_cx_pareto = np.array([r["gqe_pareto_cnot"] for r in rows])
    gqe_d_pareto = np.array([r["gqe_pareto_depth"] for r in rows])
    gqe_pareto_f = np.array([r["gqe_pareto_fidelity"] for r in rows])
    qi_cx = np.array([r["qiskit_cnot"] for r in rows])
    qi_d = np.array([r["qiskit_depth"] for r in rows])

    passed = gqe_pareto_f >= F_THRESHOLD
    pct_pass = 100.0 * passed.mean()

    if passed.any():
        match_or_beat = ((gqe_cx_pareto[passed] <= qi_cx[passed])).mean() * 100.0
        med_cx_delta = float(np.median(gqe_cx_pareto[passed] - qi_cx[passed]))
        med_d_delta = float(np.median(gqe_d_pareto[passed] - qi_d[passed]))
    else:
        match_or_beat = 0.0
        med_cx_delta = float("nan")
        med_d_delta = float("nan")

    return "\n".join([
        f"N = {len(rows)} unitaries",
        f"GQE fidelity: mean {fids.mean():.4f} | median {np.median(fids):.4f} "
        f"| min {fids.min():.4f}",
        f"F >= {F_THRESHOLD}: {pct_pass:.0f}% of instances",
        f"At F >= {F_THRESHOLD}: GQE CNOT <= Qiskit in {match_or_beat:.0f}%",
        f"Median delta (GQE - Qiskit) at F >= {F_THRESHOLD}: "
        f"CNOT {med_cx_delta:+.1f} | depth {med_d_delta:+.1f}",
    ])


def render(rows: list[dict], output: Path) -> None:
    # Sort by GQE best-fidelity ascending so the x-axis reads worst -> best.
    rows = sorted(rows, key=lambda r: r["gqe_fidelity"])

    x = np.arange(len(rows))
    gqe_f = np.array([r["gqe_fidelity"] for r in rows])
    gqe_cx = np.array([r["gqe_pareto_cnot"] for r in rows])
    gqe_d = np.array([r["gqe_pareto_depth"] for r in rows])
    qi_cx = np.array([r["qiskit_cnot"] for r in rows])
    qi_d = np.array([r["qiskit_depth"] for r in rows])

    fig, (ax_f, ax_cx, ax_d) = plt.subplots(
        3, 1, figsize=(max(10, 0.22 * len(rows) + 4), 9), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0, 1.0]},
    )

    # ── Panel 1: fidelity ──────────────────────────────────────────────────
    ax_f.scatter(x, gqe_f, s=28, color="tab:blue", label="GQE best fidelity", zorder=3)
    ax_f.axhline(1.0, color="tab:orange", linestyle="--", linewidth=1.2,
                 label="Qiskit reference (F = 1)")
    ax_f.axhline(F_THRESHOLD, color="grey", linestyle=":", linewidth=1.0,
                 label=f"F = {F_THRESHOLD}")
    ax_f.set_ylabel("Process fidelity")
    ax_f.set_ylim(min(0.9, float(gqe_f.min()) - 0.02), 1.005)
    ax_f.legend(loc="lower right", fontsize=9)
    ax_f.grid(True, axis="y", alpha=0.3)

    # ── Panels 2/3: grouped bars ──────────────────────────────────────────
    w = 0.4

    def _grouped(ax, gqe_vals, qi_vals, ylabel, hline=None):
        ax.bar(x - w / 2, gqe_vals, width=w, color="tab:blue", label="GQE")
        ax.bar(x + w / 2, qi_vals, width=w, color="tab:orange", label="Qiskit")
        if hline is not None:
            ax.axhline(hline, color="grey", linestyle=":", linewidth=1.0,
                       label=f"KAK upper bound = {hline}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    _grouped(ax_cx, gqe_cx, qi_cx, "CNOT count", hline=KAK_CNOT_UPPER_BOUND)
    _grouped(ax_d, gqe_d, qi_d, "Circuit depth")

    ax_d.set_xlabel("Unitary index (sorted by GQE fidelity, ascending)")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([str(i) for i in x],
                         fontsize=7, rotation=90 if len(rows) > 30 else 0)

    fig.suptitle(
        f"GQE vs Qiskit on {len(rows)} Haar-random 2-qubit unitaries",
        fontsize=13,
    )

    # Summary text box in upper-left of top panel
    ax_f.text(
        0.01, 0.02, summary_text(rows),
        transform=ax_f.transAxes, fontsize=8.5, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.85),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    print(f"Wrote {output} and {output.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/benchmark_3q.jsonl")
    parser.add_argument("--output", default="results/benchmark_3q.png")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    if not rows:
        raise SystemExit(f"No rows in {args.input}. Run benchmark.py first.")
    render(rows, Path(args.output))
    print(summary_text(rows))


if __name__ == "__main__":
    main()
