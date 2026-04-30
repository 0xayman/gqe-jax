"""Reporting utilities for selected circuits and run artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from circuit import parse_gate_name
from config import GQEConfig
from cost import process_fidelity


GATE_TOKEN_OFFSET = 2

def gate_indices_from_token_sequence(token_sequence) -> list[int]:
    return [
        int(t) - GATE_TOKEN_OFFSET
        for t in np.asarray(token_sequence, dtype=np.int32)[1:].tolist()
        if int(t) >= GATE_TOKEN_OFFSET
    ]


def gate_names_from_token_sequence(token_sequence, pool) -> list[str]:
    return [pool[i][0] for i in gate_indices_from_token_sequence(token_sequence)]


def select_report_token_sequence(
    best_indices,
    best_angles,
    pareto_archive,
    min_fidelity: float,
):
    """Select the shallowest archive circuit above threshold, with fallbacks."""
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
        best_fid = pareto_archive.best_by_fidelity()
        if best_fid is not None:
            i = index_by_id.get(id(best_fid), -1)
            return (
                np.asarray(best_fid.token_sequence, dtype=np.int32),
                best_fid.opt_angles,
                float(best_fid.fidelity),
                f"Pareto best fidelity [{i}]BF (no circuit reached F>={min_fidelity:.3f})",
            )

    if best_indices is None:
        return None, None, None, "No circuit available"
    return (
        np.asarray(best_indices, dtype=np.int32),
        best_angles,
        None,
        "Best raw rollout",
    )


def select_best_fidelity_token_sequence(best_indices, best_angles, pareto_archive):
    """Select the archive circuit with the highest fidelity, with fallback."""
    if pareto_archive is not None and len(pareto_archive) > 0:
        bf = pareto_archive.best_by_fidelity()
        if bf is not None:
            return (
                np.asarray(bf.token_sequence, dtype=np.int32),
                bf.opt_angles,
                float(bf.fidelity),
                "Pareto best fidelity",
            )
    if best_indices is None:
        return None, None, None, "No circuit available"
    return (
        np.asarray(best_indices, dtype=np.int32),
        best_angles,
        None,
        "Best raw rollout",
    )


def _map_qubit(q: int, num_qubits: int) -> int:
    """Convert internal MSB-first qubit index to Qiskit's LSB-first numbering."""
    return num_qubits - 1 - q


def build_qiskit_circuit_from_actions(
    token_sequence: np.ndarray,
    angles_full: np.ndarray | None,
    pool,
    num_qubits: int,
):
    """Render a token sequence and aligned angle vector as a Qiskit circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    tokens_no_bos = np.asarray(token_sequence, dtype=np.int32)[1:]
    angles = (
        np.asarray(angles_full, dtype=np.float64)
        if angles_full is not None
        else np.zeros(tokens_no_bos.shape, dtype=np.float64)
    )

    for pos, tok in enumerate(tokens_no_bos.tolist()):
        if tok < GATE_TOKEN_OFFSET:
            continue
        name = pool[tok - GATE_TOKEN_OFFSET][0]
        spec = parse_gate_name(name)
        if spec.gate_type == "RX":
            qc.rx(float(angles[pos]), _map_qubit(spec.qubits[0], num_qubits))
        elif spec.gate_type == "RY":
            qc.ry(float(angles[pos]), _map_qubit(spec.qubits[0], num_qubits))
        elif spec.gate_type == "RZ":
            qc.rz(float(angles[pos]), _map_qubit(spec.qubits[0], num_qubits))
        elif spec.gate_type == "SX":
            qc.sx(_map_qubit(spec.qubits[0], num_qubits))
        elif spec.gate_type == "CNOT":
            qc.cx(
                _map_qubit(spec.qubits[0], num_qubits),
                _map_qubit(spec.qubits[1], num_qubits),
            )
    return qc


@dataclass
class ReportedCircuit:
    qc: object
    gate_names: list[str]
    verified_fidelity: float
    qc_fidelity: float


def build_reported_circuit(
    *,
    cfg: GQEConfig,
    pool,
    u_target: np.ndarray,
    report_indices,
    report_angles,
    report_fidelity,
) -> ReportedCircuit:
    """Build the selected Qiskit circuit and verify its process fidelity."""
    from qiskit.quantum_info import Operator

    num_qubits = cfg.target.num_qubits
    gate_names = gate_names_from_token_sequence(report_indices, pool)
    qc = build_qiskit_circuit_from_actions(
        np.asarray(report_indices, dtype=np.int32),
        report_angles,
        pool,
        num_qubits,
    )
    qc_fidelity = float(process_fidelity(
        u_target,
        np.asarray(Operator(qc).data, dtype=np.complex128),
    ))
    verified_f = (
        float(report_fidelity) if report_fidelity is not None else qc_fidelity
    )
    return ReportedCircuit(
        qc=qc,
        gate_names=gate_names,
        verified_fidelity=verified_f,
        qc_fidelity=qc_fidelity,
    )


def _run_metadata(cfg: GQEConfig, target_desc: str, n_epochs: int) -> dict:
    """Return host and experiment metadata for a run artifact."""
    import getpass
    import platform
    import socket
    import sys
    from datetime import datetime, timezone

    return {
        "kind": "single_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "experiment": {
            "num_qubits": int(cfg.target.num_qubits),
            "target_type": cfg.target.type,
            "target_description": target_desc,
            "seed": int(cfg.training.seed),
            "max_epochs": int(cfg.training.max_epochs),
            "epochs_completed": int(n_epochs),
            "fidelity_threshold": float(cfg.reward.fidelity_threshold),
        },
    }


def _config_snapshot(cfg: GQEConfig) -> dict:
    """Serialize the full experiment configuration into JSON-safe values."""
    return {
        "target": {
            "type": cfg.target.type,
            "num_qubits": int(cfg.target.num_qubits),
            "path": cfg.target.path,
            "brickwork_depth": (
                int(cfg.target.brickwork_depth)
                if cfg.target.brickwork_depth is not None else None
            ),
        },
        "pool": {"rotation_gates": list(cfg.pool.rotation_gates)},
        "model": {
            "size": cfg.model.size,
            "max_gates_count": int(cfg.model.max_gates_count),
            "auto_max_gates_count": bool(cfg.model.auto_max_gates_count),
        },
        "training": {
            "max_epochs": int(cfg.training.max_epochs),
            "num_samples": int(cfg.training.num_samples),
            "batch_size": int(cfg.training.batch_size),
            "lr": float(cfg.training.lr),
            "grad_norm_clip": float(cfg.training.grad_norm_clip),
            "seed": int(cfg.training.seed),
            "grpo_clip_ratio": float(cfg.training.grpo_clip_ratio),
            "early_stop": bool(cfg.training.early_stop),
        },
        "policy": {
            "entropy_disc": float(cfg.policy.entropy_disc),
            "entropy_cont": float(cfg.policy.entropy_cont),
            "inner_refine_steps": int(cfg.policy.inner_refine_steps),
            "inner_refine_lr": float(cfg.policy.inner_refine_lr),
        },
        "temperature": {
            "scheduler": cfg.temperature.scheduler,
            "initial_value": float(cfg.temperature.initial_value),
            "delta": float(cfg.temperature.delta),
            "min_value": float(cfg.temperature.min_value),
            "max_value": float(cfg.temperature.max_value),
        },
        "buffer": {
            "max_size": int(cfg.buffer.max_size),
            "steps_per_epoch": int(cfg.buffer.steps_per_epoch),
        },
        "refinement": {
            "enabled": bool(cfg.refinement.enabled),
            "steps": int(cfg.refinement.steps),
            "lr": float(cfg.refinement.lr),
            "apply_simplify": bool(cfg.refinement.apply_simplify),
            "use_linear_trace_loss": bool(cfg.refinement.use_linear_trace_loss),
            "early_stop_patience": int(cfg.refinement.early_stop_patience),
            "early_stop_rel_tol": float(cfg.refinement.early_stop_rel_tol),
            "sweep_passes": int(cfg.refinement.sweep_passes),
            "simplify_max_passes": int(cfg.refinement.simplify_max_passes),
        },
        "reward": {
            "enabled": bool(cfg.reward.enabled),
            "mode": cfg.reward.mode,
            "lex_fidelity_weight": float(cfg.reward.lex_fidelity_weight),
            "lex_infidelity_eps": float(cfg.reward.lex_infidelity_eps),
            "lex_cnot_weight": float(cfg.reward.lex_cnot_weight),
            "lex_depth_weight": float(cfg.reward.lex_depth_weight),
            "lex_total_gate_weight": float(cfg.reward.lex_total_gate_weight),
            "lex_structure_fidelity_threshold": float(
                cfg.reward.lex_structure_fidelity_threshold
            ),
            "lex_no_stop_penalty": float(cfg.reward.lex_no_stop_penalty),
            "qaser_init_max_depth": float(cfg.reward.qaser_init_max_depth),
            "qaser_init_max_cnot": float(cfg.reward.qaser_init_max_cnot),
            "qaser_init_max_gates": float(cfg.reward.qaser_init_max_gates),
            "qaser_w_depth": float(cfg.reward.qaser_w_depth),
            "qaser_w_cnot": float(cfg.reward.qaser_w_cnot),
            "qaser_w_gates": float(cfg.reward.qaser_w_gates),
            "qaser_log_infidelity_eps": float(cfg.reward.qaser_log_infidelity_eps),
            "fidelity_floor": float(cfg.reward.fidelity_floor),
            "max_archive_size": int(cfg.reward.max_archive_size),
            "fidelity_threshold": float(cfg.reward.fidelity_threshold),
            "pair_repeat_window": int(cfg.reward.pair_repeat_window),
            "pair_repeat_max": int(cfg.reward.pair_repeat_max),
        },
    }


def save_run_artifact(
    *,
    cfg: GQEConfig,
    u_target: np.ndarray,
    target_desc: str,
    pareto_archive,
    epoch_logs: list[dict] | None,
    output_path: str,
) -> int:
    """Write target, config, logs, and Pareto summaries to one JSON artifact."""
    import json
    import os

    if pareto_archive is None or len(pareto_archive) == 0:
        pareto_payload: dict = {
            "num_circuits": 0,
            "hypervolume_2d": 0.0,
            "fidelity_floor": (
                float(pareto_archive.fidelity_floor)
                if pareto_archive is not None else None
            ),
            "circuits": [],
        }
        n_entries = 0
    else:
        entries = pareto_archive.to_sorted_list()
        circuits = [
            {
                "index": i,
                "fidelity": float(p.fidelity),
                "depth": int(p.depth),
                "total_gates": int(p.total_gates),
                "cnot_count": int(p.cnot_count),
                "epoch": int(p.epoch),
            }
            for i, p in enumerate(entries)
        ]
        pareto_payload = {
            "num_circuits": len(circuits),
            "hypervolume_2d": float(pareto_archive.hypervolume_2d()),
            "fidelity_floor": float(pareto_archive.fidelity_floor),
            "circuits": circuits,
        }
        n_entries = len(circuits)

    u = np.asarray(u_target, dtype=np.complex128)
    target_payload = {
        "type": cfg.target.type,
        "path": cfg.target.path,
        "num_qubits": int(cfg.target.num_qubits),
        "description": target_desc,
        "shape": list(u.shape),
        "unitary_real": np.real(u).tolist(),
        "unitary_imag": np.imag(u).tolist(),
    }

    payload = {
        "metadata": _run_metadata(cfg, target_desc, n_epochs=len(epoch_logs or [])),
        "target": target_payload,
        "config": _config_snapshot(cfg),
        "training_log": list(epoch_logs) if epoch_logs is not None else [],
        "pareto_front": pareto_payload,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    return n_entries


def circuit_stats(qc) -> tuple[int, int, int]:
    """Return ``(depth, total_gates, two_qubit_gate_count)`` for a Qiskit circuit."""
    depth = qc.depth()
    total = sum(qc.count_ops().values())
    two_q = sum(1 for inst in qc.data if len(inst.qubits) >= 2)
    return depth, total, two_q
