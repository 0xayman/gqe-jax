import numpy as np
import jax
from dotenv import load_dotenv

from config import load_config
from continuous_optimizer import ContinuousOptimizer
from cost import build_cost_fn, process_fidelity
from gqe import _build_logger, gqe
from operator_pool import build_operator_pool
from target import build_target


def _compose_unitary(gate_matrices, d: int) -> np.ndarray:
    """Compose an ordered gate list into a single unitary (for result decoding)."""
    u = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u = gate @ u
    return u


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
    print(compiled.draw("text", fold=-1))
    return compiled

def _circuit_stats(qc) -> tuple[int, int, int]:
    """Return (depth, total_gates, two_qubit_gates) for a QuantumCircuit."""
    depth = qc.depth()
    total = sum(qc.count_ops().values())
    two_q = sum(1 for inst in qc.data if len(inst.qubits) >= 2)
    return depth, total, two_q


def _build_qiskit_circuit_from_names(gate_names: list[str], num_qubits: int):
    """Build a Qiskit QuantumCircuit from pool gate name strings.

    Used when continuous_opt is disabled and no gate_specs/opt_params are available.
    Angles for rotation gates use the pool default (π/4); only structure matters for
    depth and gate-count comparison.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    default_angle = np.pi / 4
    def _map_qubit(qubit: int) -> int:
        return num_qubits - 1 - qubit
    for name in gate_names:
        parts = name.split("_")
        gate_type = parts[0]
        if gate_type in ("RX", "RY", "RZ"):
            qubit = int(parts[1][1:])
            getattr(qc, gate_type.lower())(default_angle, _map_qubit(qubit))
        elif gate_type == "SX":
            qubit = int(parts[1][1:])
            qc.sx(_map_qubit(qubit))
        elif gate_type == "CNOT":
            ctrl, tgt = int(parts[1][1:]), int(parts[2][1:])
            qc.cx(_map_qubit(ctrl), _map_qubit(tgt))
    return qc


def _gate_names_from_token_sequence(token_sequence, pool) -> list[str]:
    """Decode a BOS-prefixed token sequence into pool gate names."""
    gate_indices = [
        int(token_id) - 1
        for token_id in np.asarray(token_sequence, dtype=np.int32)[1:].tolist()
        if int(token_id) > 0
    ]
    return [pool[i][0] for i in gate_indices]


def _gate_indices_from_token_sequence(token_sequence) -> list[int]:
    """Decode a BOS-prefixed token sequence into zero-based pool indices."""
    return [
        int(token_id) - 1
        for token_id in np.asarray(token_sequence, dtype=np.int32)[1:].tolist()
        if int(token_id) > 0
    ]


def _select_report_token_sequence(best_indices, pareto_archive, min_fidelity: float):
    """Select the circuit to report at the end of training.

    When Pareto mode is enabled, depth-focused reporting should come from the
    archive rather than the raw-fidelity tracker so the selected circuit is
    consistent with the reported Pareto trade-offs.
    """
    if pareto_archive is not None and len(pareto_archive) > 0:
        best_depth = pareto_archive.best_by_depth(min_fidelity=min_fidelity)
        if best_depth is not None:
            return (
                np.asarray(best_depth.token_sequence, dtype=np.int32),
                f"Pareto best depth (F≥{min_fidelity:.3f})",
            )
        best_fidelity = pareto_archive.best_by_fidelity()
        if best_fidelity is not None:
            return (
                np.asarray(best_fidelity.token_sequence, dtype=np.int32),
                f"Pareto best fidelity (no circuit reached F≥{min_fidelity:.3f})",
            )

    if best_indices is None:
        return None, "No circuit available"
    return np.asarray(best_indices, dtype=np.int32), "Raw-fidelity best"


def _print_comparison_table(rows) -> None:
    """Print a side-by-side comparison table for compiled circuits."""
    col_w = [10, 12, 9, 15, 17]
    header = ["", "Fidelity", "Depth", "Total gates", "2-qubit gates"]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"

    def row(label, fidelity, depth, total, two_q):
        fid_str = f"{fidelity:.6f}" if fidelity is not None else "1.000000"
        return (
            f"| {label:<{col_w[0]-2}} "
            f"| {fid_str:>{col_w[1]-2}} "
            f"| {depth:>{col_w[2]-2}} "
            f"| {total:>{col_w[3]-2}} "
            f"| {two_q:>{col_w[4]-2}} |"
        )

    print(f"\n{'=' * 55}")
    print("Compilation Comparison")
    print(sep)
    print(f"| {'':>{col_w[0]-2}} | {header[1]:>{col_w[1]-2}} | {header[2]:>{col_w[2]-2}} | {header[3]:>{col_w[3]-2}} | {header[4]:>{col_w[4]-2}} |")
    print(sep)
    for label, fidelity, depth, total, two_q in rows:
        print(row(label, fidelity, depth, total, two_q))
    print(sep)


def _build_qiskit_circuit(gate_specs, opt_params, num_qubits):
    """Build a Qiskit QuantumCircuit from gate_specs and optimized angle params."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(num_qubits)
    param_idx = 0
    def _map_qubit(qubit: int) -> int:
        return num_qubits - 1 - qubit
    for spec in gate_specs:
        if spec.is_parametric:
            theta = float(opt_params[param_idx])
            param_idx += 1
            if spec.gate_type == "RX":
                qc.rx(theta, _map_qubit(spec.qubits[0]))
            elif spec.gate_type == "RY":
                qc.ry(theta, _map_qubit(spec.qubits[0]))
            elif spec.gate_type == "RZ":
                qc.rz(theta, _map_qubit(spec.qubits[0]))
        elif spec.gate_type == "SX":
            qc.sx(_map_qubit(spec.qubits[0]))
        elif spec.gate_type == "CNOT":
            qc.cx(_map_qubit(spec.qubits[0]), _map_qubit(spec.qubits[1]))
    return qc


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
    qiskit_depth, qiskit_total, qiskit_two_q = _circuit_stats(qiskit_compiled)

    # ── Build cost function ─────────────────────────────────────────────────
    cost_fn = build_cost_fn(u_target)

    # ── Run GQE training ────────────────────────────────────────────────────
    print("\nStarting GQE training...\n")
    best_cost, best_indices, pareto_archive = gqe(cost_fn, pool, cfg, u_target=u_target, logger=logger)

    # ── Report results ───────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Training complete!")
    print(f"  Best raw cost (1 - F): {best_cost:.6f}")

    report_indices, report_source = _select_report_token_sequence(
        best_indices,
        pareto_archive,
        cfg.pareto.fidelity_threshold,
    )
    print(f"  Reported circuit source: {report_source}")

    if report_indices is not None:
        gate_indices = _gate_indices_from_token_sequence(report_indices)
        gate_names = _gate_names_from_token_sequence(report_indices, pool)
        selected_cnot_count = sum(name.startswith("CNOT") for name in gate_names)
        print(f"  Reported circuit ({len(gate_names)} gates):")
        for k, name in enumerate(gate_names):
            print(f"    [{k}] {name}")

        from qiskit.quantum_info import Operator

        # Independently verify by re-running the optimizer on the best gate structure
        if cfg.continuous_opt.enabled:
            verifier = ContinuousOptimizer(
                u_target=u_target,
                num_qubits=cfg.target.num_qubits,
                steps=cfg.continuous_opt.steps,
                lr=cfg.continuous_opt.lr,
                optimizer_type=cfg.continuous_opt.optimizer,
                top_k=0,
                max_gates=cfg.model.max_gates_count,
                num_restarts=cfg.continuous_opt.num_restarts,
            )
            verified_f, gate_specs, opt_params, _ = verifier.optimize_circuit_with_params(
                gate_names,
                jax.random.PRNGKey(cfg.training.seed),
            )
            gqe_qc = _build_qiskit_circuit(gate_specs, opt_params, cfg.target.num_qubits)
        else:
            gate_matrices = [pool[i][1] for i in gate_indices]
            u_circuit = _compose_unitary(gate_matrices, d)
            verified_f = process_fidelity(u_target, u_circuit)
            gate_specs, opt_params = None, None
            gqe_qc = _build_qiskit_circuit_from_names(gate_names, cfg.target.num_qubits)

        gqe_fidelity = process_fidelity(
            u_target,
            np.asarray(Operator(gqe_qc).data, dtype=np.complex128),
        )

        print(f"  Reported raw fidelity: {verified_f:.6f}")
        print(f"  Reported GQE fidelity: {gqe_fidelity:.6f}")
        print(f"  Reported raw CNOTs:    {selected_cnot_count}")
        print(f"  Reported GQE CNOTs:    {int(gqe_qc.count_ops().get('cx', 0))}")

        # ── Draw the optimized circuit ───────────────────────────────────────
        print(f"\n{'=' * 55}")
        print("Reported GQE circuit (Qiskit draw):")
        print(gqe_qc.draw("text", fold=-1))

        # ── Comparison table ─────────────────────────────────────────────────
        gqe_depth, gqe_total, gqe_two_q = _circuit_stats(gqe_qc)
        _print_comparison_table(
            [
                ("GQE", gqe_fidelity, gqe_depth, gqe_total, gqe_two_q),
                ("Qiskit", 1.0, qiskit_depth, qiskit_total, qiskit_two_q),
            ]
        )

    # ── Pareto front summary ─────────────────────────────────────────────────
    _print_pareto_summary(
        pareto_archive,
        qiskit_depth,
        qiskit_total,
        qiskit_two_q,
        cfg.pareto.fidelity_threshold,
    )


if __name__ == "__main__":
    main()
