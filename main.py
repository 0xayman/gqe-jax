"""Run one configured target-conditioned GQE synthesis experiment."""

from __future__ import annotations

import argparse
import numpy as np
import jax
from dotenv import load_dotenv

from config import load_config
from operator_pool import build_operator_pool
from reporting import (
    build_reported_circuit,
    circuit_stats,
    gate_names_from_token_sequence,
    save_run_artifact,
    select_report_token_sequence,
)
from simplify import simplify_pareto_archive
from target import build_target
from trainer import build_logger, gqe


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one GQE synthesis experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="config.yml",
        help="GQE config YAML.",
    )
    return p.parse_args()


def _basis_gates(rotation_gates) -> list[str]:
    return [*rotation_gates, "sx", "cx"]


def _compile_target_with_qiskit(u_target, num_qubits, rotation_gates):
    """Synthesize ``u_target`` with Qiskit using the same basis gate set."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate

    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(u_target), list(range(num_qubits)))
    basis_gates = _basis_gates(rotation_gates)
    compiled = transpile(qc, basis_gates=basis_gates, optimization_level=3)
    print(f"\n{'=' * 55}")
    print("Target unitary compiled by Qiskit (reference circuit):")
    print(
        f"  Basis gates: {', '.join(basis_gates)}"
        f"  |  Depth: {compiled.depth()}  |  Gates: {compiled.count_ops()}"
    )
    return compiled


def _print_pareto_summary(
    pareto_archive,
    qiskit_depth: int,
    qiskit_total: int,
    qiskit_two_q: int,
    min_fidelity: float,
) -> None:
    if pareto_archive is None or len(pareto_archive) == 0:
        return

    entries = pareto_archive.to_sorted_list()
    hv = pareto_archive.hypervolume_2d()
    best_f = pareto_archive.best_by_fidelity()
    best_c = pareto_archive.best_by_cnot(min_fidelity=min_fidelity)
    best_d = pareto_archive.best_by_depth(min_fidelity=min_fidelity)

    print(f"\n{'=' * 55}")
    print(f"Pareto Front  ({len(entries)} non-dominated circuits | HV={hv:.6f})")

    col_w = [14, 10, 7, 13, 15, 7]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def fmt(label, fidelity, depth, total, two_q, epoch):
        return (
            f"| {label:<{col_w[0]-2}} "
            f"| {fidelity:>{col_w[1]-2}.6f} "
            f"| {depth:>{col_w[2]-2}} "
            f"| {total:>{col_w[3]-2}} "
            f"| {two_q:>{col_w[4]-2}} "
            f"| {epoch:>{col_w[5]-2}} |"
        )

    print(sep)
    print(
        f"| {'Label':<{col_w[0]-2}} "
        f"| {'Fidelity':>{col_w[1]-2}} "
        f"| {'Depth':>{col_w[2]-2}} "
        f"| {'Total gates':>{col_w[3]-2}} "
        f"| {'2-qubit gates':>{col_w[4]-2}} "
        f"| {'Epoch':>{col_w[5]-2}} |"
    )
    print(sep)

    for i, p in enumerate(entries):
        tags = []
        if best_f is not None and p is best_f:
            tags.append("BF")
        if best_c is not None and p is best_c:
            tags.append("BC")
        if best_d is not None and p is best_d and p is not best_c:
            tags.append("BD")
        label = f"[{i}]{''.join(tags)}"
        print(fmt(label, p.fidelity, p.depth, p.total_gates, p.cnot_count, p.epoch))

    print(sep)
    print(fmt("Qiskit ref", 1.0, qiskit_depth, qiskit_total, qiskit_two_q, -1))
    print(sep)
    print(
        f"Tags: BF=best fidelity  "
        f"BC=best CNOT (F>={min_fidelity:.3f})  "
        f"BD=best depth (F>={min_fidelity:.3f})"
    )


def main():
    args = _parse_args()
    load_dotenv()

    cfg = load_config(args.config)
    print(f"\nLoaded config: {args.config}")
    print(f"  Target type:          {cfg.target.type}")
    print(f"  Qubits:               {cfg.target.num_qubits}")
    print(f"  Scheduler:            {cfg.temperature.scheduler}")
    print(f"  Model size:           {cfg.model.size}")
    print(f"  Training epochs:      {cfg.training.max_epochs}")
    print(f"  Refinement:           {'adam' if cfg.refinement.enabled else 'off'}")
    print(f"  W&B logging:          {cfg.logging.wandb}")
    print(f"  JAX backend:          {jax.default_backend()}")
    print(f"  JAX devices:          {jax.devices()}")

    pool = build_operator_pool(
        num_qubits=cfg.target.num_qubits,
        rotation_gates=cfg.pool.rotation_gates,
    )
    print(f"  Gate pool size:       {len(pool)}")
    print(f"  Model vocab size:     {len(pool) + 3}")
    print(f"  Rotation gates:       {', '.join(cfg.pool.rotation_gates)}")

    u_target, target_desc = build_target(pool, cfg)
    print(f"\nTarget unitary: {target_desc}")
    d = 2 ** cfg.target.num_qubits
    deviation = np.max(np.abs(u_target @ u_target.conj().T - np.eye(d)))
    assert deviation < 1e-10, f"Target is not unitary! Max deviation: {deviation}"
    print(f"  Unitarity verified (d={d})")

    logger = build_logger(cfg)

    qiskit_compiled = _compile_target_with_qiskit(
        u_target, cfg.target.num_qubits, cfg.pool.rotation_gates,
    )
    qiskit_depth, qiskit_total, qiskit_two_q = circuit_stats(qiskit_compiled)

    print("\nStarting hybrid-action GQE training...\n")
    result = gqe(cfg, u_target, pool, logger=logger)

    if result.pareto_archive is not None and len(result.pareto_archive) > 0:
        result.pareto_archive = simplify_pareto_archive(
            result.pareto_archive, pool, cfg.target.num_qubits,
        )

    print(f"\n{'=' * 55}")
    print("Training complete!")
    print(f"  Best cost (during training):       {result.best_cost:.6f}")
    print(f"  Best raw fidelity (during train):  {result.best_raw_fidelity:.6f}")
    if result.refined_raw_fidelity is not None:
        print(
            f"  Best raw fidelity (post-refine):   "
            f"{result.refined_raw_fidelity:.6f}"
        )

    pareto_empty = (
        result.pareto_archive is None or len(result.pareto_archive) == 0
    )
    if pareto_empty and result.best_raw_tokens is not None:
        rep_idx = np.asarray(result.best_raw_tokens, dtype=np.int32)
        rep_angles = (
            result.refined_raw_angles
            if result.refined_raw_angles is not None
            else result.best_raw_angles
        )
        rep_F = (
            result.refined_raw_fidelity
            if result.refined_raw_fidelity is not None
            else result.best_raw_fidelity
        )
        rep_src = (
            "Best raw rollout (refined)"
            if result.refined_raw_fidelity is not None
            else "Best raw rollout"
        )
    else:
        rep_idx, rep_angles, rep_F, rep_src = select_report_token_sequence(
            result.best_tokens, result.best_angles,
            result.pareto_archive, cfg.reward.fidelity_threshold,
        )
    print(f"  Reported circuit source: {rep_src}")

    if rep_idx is not None:
        gate_names = gate_names_from_token_sequence(rep_idx, pool)
        cnot_count = sum(name.startswith("CNOT") for name in gate_names)
        print(f"  Reported circuit ({len(gate_names)} gates):")
        for k, name in enumerate(gate_names):
            print(f"    [{k}] {name}")

        reported = build_reported_circuit(
            cfg=cfg, pool=pool, u_target=u_target,
            report_indices=rep_idx, report_angles=rep_angles, report_fidelity=rep_F,
        )
        print(f"  Reported verified fidelity: {reported.verified_fidelity:.6f}")
        print(f"  Reported QC fidelity:       {reported.qc_fidelity:.6f}")
        print(f"  Reported raw CNOTs:         {cnot_count}")
        print(f"  Reported QC CNOTs:          {int(reported.qc.count_ops().get('cx', 0))}")

    _print_pareto_summary(
        result.pareto_archive, qiskit_depth, qiskit_total, qiskit_two_q,
        cfg.reward.fidelity_threshold,
    )

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = (
        f"results/runs/run_{cfg.target.num_qubits}q_{cfg.target.type}_{stamp}.json"
    )
    n_saved = save_run_artifact(
        cfg=cfg,
        u_target=u_target,
        target_desc=target_desc,
        pareto_archive=result.pareto_archive,
        epoch_logs=result.epoch_logs,
        output_path=artifact_path,
    )
    print(
        f"\nSaved run artifact to {artifact_path} "
        f"(target unitary, {len(result.epoch_logs or [])} epoch logs, "
        f"{n_saved} Pareto entries)"
    )


if __name__ == "__main__":
    main()
