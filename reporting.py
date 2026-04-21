"""Post-training reporting: pick the circuit to report and materialise it as a
Qiskit ``QuantumCircuit`` with a verified process fidelity.

Extracted from ``main.py`` so the benchmark runner can reuse it without
duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax

from config import GQEConfig
from continuous_optimizer import ContinuousOptimizer, parse_gate_spec
from cost import process_fidelity


GATE_TOKEN_OFFSET = 2  # BOS=0, STOP=1, gates start at 2


def compose_unitary(gate_matrices, d: int) -> np.ndarray:
    u = np.eye(d, dtype=np.complex128)
    for gate in gate_matrices:
        u = gate @ u
    return u


def gate_indices_from_token_sequence(token_sequence) -> list[int]:
    return [
        int(tok) - GATE_TOKEN_OFFSET
        for tok in np.asarray(token_sequence, dtype=np.int32)[1:].tolist()
        if int(tok) >= GATE_TOKEN_OFFSET
    ]


def gate_names_from_token_sequence(token_sequence, pool) -> list[str]:
    return [pool[i][0] for i in gate_indices_from_token_sequence(token_sequence)]


def select_report_token_sequence(
    best_indices,
    best_angles,
    pareto_archive,
    min_fidelity: float,
):
    """Pick which circuit to report: prefer Pareto best-depth at F>=threshold."""
    if pareto_archive is not None and len(pareto_archive) > 0:
        entries = pareto_archive.to_sorted_list()
        index_by_id = {id(p): i for i, p in enumerate(entries)}

        best_depth = pareto_archive.best_by_depth(min_fidelity=min_fidelity)
        if best_depth is not None:
            i = index_by_id.get(id(best_depth), -1)
            return (
                np.asarray(best_depth.token_sequence, dtype=np.int32),
                best_depth.opt_angles,
                float(best_depth.fidelity),
                f"Pareto best depth [{i}]BD (F>={min_fidelity:.3f})",
            )
        best_fidelity = pareto_archive.best_by_fidelity()
        if best_fidelity is not None:
            i = index_by_id.get(id(best_fidelity), -1)
            return (
                np.asarray(best_fidelity.token_sequence, dtype=np.int32),
                best_fidelity.opt_angles,
                float(best_fidelity.fidelity),
                f"Pareto best fidelity [{i}]BF (no circuit reached F>={min_fidelity:.3f})",
            )

    if best_indices is None:
        return None, None, None, "No circuit available"
    return (
        np.asarray(best_indices, dtype=np.int32),
        best_angles,
        None,
        "Raw-fidelity best",
    )


def select_best_fidelity_token_sequence(best_indices, best_angles, pareto_archive):
    """Pick the archive entry with the highest verified fidelity (ignore structure)."""
    if pareto_archive is not None and len(pareto_archive) > 0:
        best_fidelity = pareto_archive.best_by_fidelity()
        if best_fidelity is not None:
            return (
                np.asarray(best_fidelity.token_sequence, dtype=np.int32),
                best_fidelity.opt_angles,
                float(best_fidelity.fidelity),
                "Pareto best fidelity",
            )
    if best_indices is None:
        return None, None, None, "No circuit available"
    return (
        np.asarray(best_indices, dtype=np.int32),
        best_angles,
        None,
        "Raw-fidelity best",
    )


def build_qiskit_circuit(gate_specs, opt_params, num_qubits):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    param_idx = 0

    def _map_qubit(q: int) -> int:
        return num_qubits - 1 - q

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


def build_qiskit_circuit_from_names(gate_names, num_qubits):
    """Fallback when continuous_opt is disabled: use default pool angles."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    default_angle = np.pi / 4

    def _map_qubit(q: int) -> int:
        return num_qubits - 1 - q

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


@dataclass
class ReportedCircuit:
    qc: object  # QuantumCircuit
    gate_names: list[str]
    verified_fidelity: float   # from training (or re-optimised)
    qc_fidelity: float         # fidelity of the reconstructed Qiskit circuit
    source: str


def build_reported_circuit(
    *,
    cfg: GQEConfig,
    pool,
    u_target: np.ndarray,
    report_indices,
    report_angles,
    report_fidelity,
) -> ReportedCircuit:
    """Materialise the reported token sequence as a verified Qiskit circuit."""
    from qiskit.quantum_info import Operator

    num_qubits = cfg.target.num_qubits
    d = 2 ** num_qubits

    gate_indices = gate_indices_from_token_sequence(report_indices)
    gate_names = gate_names_from_token_sequence(report_indices, pool)

    if cfg.continuous_opt.enabled and report_angles is not None:
        gate_specs = [parse_gate_spec(name) for name in gate_names]
        tokens_no_bos = np.asarray(report_indices, dtype=np.int32)[1:].tolist()
        angles_full = np.asarray(report_angles, dtype=np.float64)
        compact_params = []
        name_idx = 0
        for pos, tok in enumerate(tokens_no_bos):
            if tok < GATE_TOKEN_OFFSET:
                continue
            spec = gate_specs[name_idx]
            if spec.is_parametric:
                compact_params.append(float(angles_full[pos]))
            name_idx += 1
        opt_params = np.asarray(compact_params, dtype=np.float64)
        verified_f = report_fidelity if report_fidelity is not None else float("nan")
        qc = build_qiskit_circuit(gate_specs, opt_params, num_qubits)
    elif cfg.continuous_opt.enabled:
        verifier = ContinuousOptimizer(
            u_target=u_target,
            num_qubits=num_qubits,
            steps=cfg.continuous_opt.steps,
            lr=cfg.continuous_opt.lr,
            optimizer_type=cfg.continuous_opt.optimizer,
            top_k=0,
            max_gates=cfg.model.max_gates_count,
            num_restarts=cfg.continuous_opt.num_restarts,
        )
        verified_f, gate_specs, opt_params, _ = verifier.optimize_circuit_with_params(
            gate_names, jax.random.PRNGKey(cfg.training.seed)
        )
        qc = build_qiskit_circuit(gate_specs, opt_params, num_qubits)
    else:
        gate_matrices = [pool[i][1] for i in gate_indices]
        u_circuit = compose_unitary(gate_matrices, d)
        verified_f = process_fidelity(u_target, u_circuit)
        qc = build_qiskit_circuit_from_names(gate_names, num_qubits)

    qc_fidelity = float(process_fidelity(
        u_target,
        np.asarray(Operator(qc).data, dtype=np.complex128),
    ))

    return ReportedCircuit(
        qc=qc,
        gate_names=gate_names,
        verified_fidelity=float(verified_f),
        qc_fidelity=qc_fidelity,
        source="",  # caller fills this in
    )


def circuit_stats(qc) -> tuple[int, int, int]:
    """(depth, total_gates, two_qubit_gates)."""
    depth = qc.depth()
    total = sum(qc.count_ops().values())
    two_q = sum(1 for inst in qc.data if len(inst.qubits) >= 2)
    return depth, total, two_q
