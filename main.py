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


def _compile_target_with_qiskit(u_target: np.ndarray, num_qubits: int):
    """Synthesize the target unitary with Qiskit and print the resulting circuit.

    Uses Qiskit's transpiler with the same basis gate set as the operator pool
    (rz, sx, cx) so the decomposition is directly comparable to what GQE learns.

    Returns the compiled QuantumCircuit for later stats comparison.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import UnitaryGate

    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(u_target), list(range(num_qubits)))

    compiled = transpile(
        qc,
        basis_gates=["rz", "sx", "cx"],
        optimization_level=3,
    )

    print(f"\n{'=' * 55}")
    print("Target unitary compiled by Qiskit (reference circuit):")
    print(f"  Basis gates: rz, sx, cx  |  Depth: {compiled.depth()}  |  Gates: {compiled.count_ops()}")
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
    pool = build_operator_pool(num_qubits=cfg.target.num_qubits)
    print(f"  Gate pool size:       {len(pool)}")
    print(f"  Model vocab size:     {len(pool) + 1}")

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
    qiskit_compiled = _compile_target_with_qiskit(u_target, cfg.target.num_qubits)
    qiskit_depth, qiskit_total, qiskit_two_q = _circuit_stats(qiskit_compiled)

    # ── Build cost function ─────────────────────────────────────────────────
    cost_fn = build_cost_fn(u_target)

    # ── Run GQE training ────────────────────────────────────────────────────
    print("\nStarting GQE training...\n")
    best_cost, best_indices = gqe(cost_fn, pool, cfg, u_target=u_target, logger=logger)

    # ── Report results ───────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Training complete!")
    print(f"  Best cost (1 - F):    {best_cost:.6f}")

    if best_indices is not None:
        # best_indices includes start token at position 0 — skip it
        gate_indices = (
            best_indices[1:]
            if isinstance(best_indices, list)
            else best_indices[1:].tolist()
        )
        gate_indices = [token_id - 1 for token_id in gate_indices]
        gate_names = [pool[i][0] for i in gate_indices]
        selected_cnot_count = sum(name.startswith("CNOT") for name in gate_names)
        print(f"  Best circuit ({len(gate_names)} gates):")
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

        print(f"  Best raw fidelity:    {verified_f:.6f}")
        print(f"  GQE fidelity:         {gqe_fidelity:.6f}")
        print(f"  Raw decoded CNOTs:    {selected_cnot_count}")
        print(f"  GQE CNOTs:            {int(gqe_qc.count_ops().get('cx', 0))}")

        # ── Draw the optimized circuit ───────────────────────────────────────
        print(f"\n{'=' * 55}")
        print("Best GQE circuit (Qiskit draw):")
        print(gqe_qc.draw("text", fold=-1))

        # ── Comparison table ─────────────────────────────────────────────────
        gqe_depth, gqe_total, gqe_two_q = _circuit_stats(gqe_qc)
        _print_comparison_table(
            [
                ("GQE", gqe_fidelity, gqe_depth, gqe_total, gqe_two_q),
                ("Qiskit", 1.0, qiskit_depth, qiskit_total, qiskit_two_q),
            ]
        )


if __name__ == "__main__":
    main()
