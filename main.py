import numpy as np
import jax
from dotenv import load_dotenv

from config import load_config
from cost import build_cost_fn
from gqe import _build_logger, gqe
from operator_pool import build_operator_pool
from reporting import (
    build_reported_circuit,
    circuit_stats,
    gate_names_from_token_sequence,
    save_pareto_archive_json,
    select_report_token_sequence,
)
from target import build_target


def _basis_gates(rotation_gates: tuple[str, ...] | list[str]) -> list[str]:
    return [*rotation_gates, "sx", "cx"]


def _compile_target_with_qiskit(
    u_target: np.ndarray,
    num_qubits: int,
    rotation_gates: tuple[str, ...] | list[str],
):
    """Synthesize the target unitary with Qiskit and print the resulting circuit.

    Uses Qiskit's transpiler with the same basis gate set as the operator pool
    so the decomposition is directly comparable to what GQE learns.

    Returns the compiled QuantumCircuit for later stats comparison.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate

    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(u_target), list(range(num_qubits)))

    basis_gates = _basis_gates(rotation_gates)
    compiled = transpile(
        qc,
        basis_gates=basis_gates,
        optimization_level=3,
    )

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
    """Print a summary table of all Pareto-optimal circuits found during training."""
    if pareto_archive is None or len(pareto_archive) == 0:
        return

    entries = pareto_archive.to_sorted_list()
    hv = pareto_archive.hypervolume_2d()
    best_f = pareto_archive.best_by_fidelity()
    best_c = pareto_archive.best_by_cnot(min_fidelity=min_fidelity)
    best_d = pareto_archive.best_by_depth(min_fidelity=min_fidelity)

    print(f"\n{'=' * 55}")
    print(f"Pareto Front  ({len(entries)} non-dominated circuits | HV={hv:.6f})")

    # Column widths: label, fidelity, depth, total gates, 2-qubit gates, epoch
    col_w = [14, 10, 7, 13, 15, 7]
    header = ["Label", "Fidelity", "Depth", "Total gates", "2-qubit gates", "Epoch"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def fmt_row(label, fidelity, depth, total, two_q, epoch):
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
            tags.append("BF")   # best fidelity
        if best_c is not None and p is best_c:
            tags.append("BC")   # best CNOT at F≥threshold
        if best_d is not None and p is best_d and p is not best_c:
            tags.append("BD")   # best depth at F≥threshold
        label = f"[{i}]{''.join(tags)}"

        print(fmt_row(label, p.fidelity, p.depth, p.total_gates, p.cnot_count, p.epoch))

    print(sep)

    # Qiskit reference row
    print(fmt_row("Qiskit ref", 1.0, qiskit_depth, qiskit_total, qiskit_two_q, -1))
    print(sep)

    print(
        f"Tags: BF=best fidelity  "
        f"BC=best CNOT (F≥{min_fidelity:.3f})  "
        f"BD=best depth (F≥{min_fidelity:.3f})"
    )


def main():
    load_dotenv()  # Load WANDB_API_KEY / WANDB_PROJECT from .env before W&B init

    # ── Load configuration ──────────────────────────────────────────────────
    cfg = load_config("config.yml")
    print("\nLoaded config: config.yml")
    print(f"  Target type:          {cfg.target.type}")
    print(f"  Qubits:               {cfg.target.num_qubits}")
    print(f"  Scheduler:            {cfg.temperature.scheduler}")
    print(f"  Model size:           {cfg.model.size}")
    print(f"  Max gates/circuit:    {cfg.model.max_gates_count}")
    print(f"  Training epochs:      {cfg.training.max_epochs}")
    print(f"  W&B logging:          {cfg.logging.wandb}")
    print(f"  JAX backend:          {jax.default_backend()}")
    print(f"  JAX devices:          {jax.devices()}")

    # ── Build operator pool ─────────────────────────────────────────────────
    pool = build_operator_pool(
        num_qubits=cfg.target.num_qubits,
        rotation_gates=cfg.pool.rotation_gates,
    )
    print(f"  Gate pool size:       {len(pool)}")
    print(f"  Model vocab size:     {len(pool) + 1}")
    print(f"  Rotation gates:       {', '.join(cfg.pool.rotation_gates)}")

    # ── Build target unitary ────────────────────────────────────────────────
    # Delegated entirely to target.py — changing cfg.target.type selects
    # a different generator with no changes here.
    u_target, target_desc = build_target(pool, cfg)
    print(f"\nTarget unitary: {target_desc}")

    # Sanity-check: every generator must return a unitary
    d = 2**cfg.target.num_qubits
    deviation = np.max(np.abs(u_target @ u_target.conj().T - np.eye(d)))
    assert deviation < 1e-10, f"Target is not unitary! Max deviation: {deviation}"
    print(f"  Unitarity verified (d={d})")

    # ── Initialize W&B logger only after config/target validation succeeds ──
    logger = _build_logger(cfg)

    # ── Compile target with Qiskit (reference circuit) ──────────────────────
    qiskit_compiled = _compile_target_with_qiskit(
        u_target,
        cfg.target.num_qubits,
        cfg.pool.rotation_gates,
    )
    qiskit_depth, qiskit_total, qiskit_two_q = circuit_stats(qiskit_compiled)

    # ── Build cost function ─────────────────────────────────────────────────
    cost_fn = build_cost_fn(u_target)

    # ── Run GQE training ────────────────────────────────────────────────────
    print("\nStarting GQE training...\n")
    best_cost, best_indices, best_angles, pareto_archive = gqe(
        cost_fn, pool, cfg, u_target=u_target, logger=logger
    )

    # ── Report results ───────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Training complete!")
    print(f"  Best raw cost (1 - F): {best_cost:.6f}")

    report_indices, report_angles, report_fidelity, report_source = (
        select_report_token_sequence(
            best_indices,
            best_angles,
            pareto_archive,
            cfg.reward.fidelity_threshold,
        )
    )
    print(f"  Reported circuit source: {report_source}")

    if report_indices is not None:
        gate_names = gate_names_from_token_sequence(report_indices, pool)
        selected_cnot_count = sum(name.startswith("CNOT") for name in gate_names)
        print(f"  Reported circuit ({len(gate_names)} gates):")
        for k, name in enumerate(gate_names):
            print(f"    [{k}] {name}")

        reported = build_reported_circuit(
            cfg=cfg,
            pool=pool,
            u_target=u_target,
            report_indices=report_indices,
            report_angles=report_angles,
            report_fidelity=report_fidelity,
        )

        print(f"  Reported raw fidelity: {reported.verified_fidelity:.6f}")
        print(f"  Reported GQE fidelity: {reported.qc_fidelity:.6f}")
        print(f"  Reported raw CNOTs:    {selected_cnot_count}")
        print(f"  Reported GQE CNOTs:    {int(reported.qc.count_ops().get('cx', 0))}")

    # ── Pareto front summary ─────────────────────────────────────────────────
    _print_pareto_summary(
        pareto_archive,
        qiskit_depth,
        qiskit_total,
        qiskit_two_q,
        cfg.reward.fidelity_threshold,
    )

    # ── Persist every Pareto-front circuit for later inspection ─────────────
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pareto_path = f"results/pareto_front_{cfg.target.num_qubits}q_{cfg.target.type}_{stamp}.json"
    n_saved = save_pareto_archive_json(
        cfg=cfg,
        pool=pool,
        u_target=u_target,
        pareto_archive=pareto_archive,
        target_desc=target_desc,
        output_path=pareto_path,
    )
    if n_saved > 0:
        print(f"\nSaved {n_saved} Pareto-front circuits to {pareto_path}")


if __name__ == "__main__":
    main()
