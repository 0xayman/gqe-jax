"""Generate (unitary, Qiskit-baseline circuit) datasets for representation learning.

Each row of the saved dataset stores:

* the phase-normalized real feature vector for the target unitary,
* the raw complex unitary (so callers can re-featurize differently later),
* the Qiskit baseline tokens (project vocabulary, no specials, padded),
* the Qiskit baseline rotation angles (aligned with the tokens),
* the verified process fidelity (in the project's convention),
* the brickwork depth used to sample the target.

The dataset is the **only** label source for budget and angle initializers.
Per the user's constraint, only brickwork-Haar targets and Qiskit-transpiler
baselines are used; no other oracles, exact decompositions, or RL outputs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import brickwork  # noqa: E402
import features as feats  # noqa: E402
import qiskit_baseline as qb  # noqa: E402
import vocab as _vocab  # noqa: E402


@dataclass
class DatasetSpec:
    num_qubits: int
    num_samples: int
    brickwork_depth_min: int
    brickwork_depth_max: int
    fidelity_threshold: float
    rotation_gates: tuple[str, ...]
    qiskit_optimization_levels: tuple[int, ...]
    seed: int
    # If non-empty, overrides num_qubits and generates samples for each n in the list.
    num_qubits_list: tuple[int, ...] = field(default_factory=tuple)
    # Canonical vocab size. 0 means infer as max(effective_num_qubits_list).
    max_qubits: int = 0

    @property
    def effective_num_qubits_list(self) -> tuple[int, ...]:
        if self.num_qubits_list:
            return self.num_qubits_list
        return (self.num_qubits,)

    @property
    def effective_max_qubits(self) -> int:
        if self.max_qubits > 0:
            return self.max_qubits
        return max(self.effective_num_qubits_list)

    @classmethod
    def from_yaml(cls, cfg: dict) -> "DatasetSpec":
        ds = cfg["dataset"]
        num_qubits_list: tuple[int, ...] = ()
        if "num_qubits_list" in ds:
            num_qubits_list = tuple(int(x) for x in ds["num_qubits_list"])
        num_qubits = int(ds.get("num_qubits", 2))
        max_qubits = int(ds.get("max_qubits", 0))
        return cls(
            num_qubits=num_qubits,
            num_samples=int(ds["num_samples"]),
            brickwork_depth_min=int(ds["brickwork_depth_min"]),
            brickwork_depth_max=int(ds["brickwork_depth_max"]),
            fidelity_threshold=float(ds["fidelity_threshold"]),
            rotation_gates=tuple(str(g) for g in ds["rotation_gates"]),
            qiskit_optimization_levels=tuple(
                int(x) for x in ds["qiskit_optimization_levels"]
            ),
            seed=int(ds["seed"]),
            num_qubits_list=num_qubits_list,
            max_qubits=max_qubits,
        )


def _pad_tokens_and_angles(
    tokens: np.ndarray,
    angles: np.ndarray,
    target_len: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Right-pad to ``target_len`` and return (padded_tokens, padded_angles, length)."""
    L = int(tokens.shape[0])
    if L > target_len:
        raise ValueError(f"sequence length {L} exceeds buffer width {target_len}")
    out_t = np.full((target_len,), pad_token_id, dtype=np.int32)
    out_a = np.zeros((target_len,), dtype=np.float32)
    if L > 0:
        out_t[:L] = tokens.astype(np.int32, copy=False)
        out_a[:L] = angles.astype(np.float32, copy=False)
    return out_t, out_a, L


def _generate_for_n(
    n: int,
    N: int,
    spec: DatasetSpec,
    canonical_vocab: _vocab.Vocab,
    rng: np.random.Generator,
    *,
    verbose: bool,
    global_offset: int,
    global_total: int,
) -> dict:
    """Generate N samples for a specific qubit count n.

    Token IDs are remapped from the n-specific vocab to canonical_vocab so that
    the same token name maps to the same ID regardless of n.
    """
    local_vocab = _vocab.build_vocab(n, spec.rotation_gates)
    d = 2 ** n
    max_qubits = canonical_vocab.num_qubits
    site_feat_dim = feats.SITE_FEAT_DIM

    raw_unitaries: list[np.ndarray] = []
    raw_features: list[np.ndarray] = []
    raw_local_features: list[np.ndarray] = []  # padded to [max_qubits, site_feat_dim]
    raw_tokens: list[np.ndarray] = []
    raw_angles: list[np.ndarray] = []
    cnot_counts = np.zeros((N,), dtype=np.int32)
    total_gates_arr = np.zeros((N,), dtype=np.int32)
    depths = np.zeros((N,), dtype=np.int32)
    fidelities = np.zeros((N,), dtype=np.float32)
    bw_depths = np.zeros((N,), dtype=np.int32)
    nq_arr = np.full((N,), n, dtype=np.int32)

    t0 = time.perf_counter()
    for i in range(N):
        bw_d = int(rng.integers(spec.brickwork_depth_min, spec.brickwork_depth_max + 1))
        u = brickwork.brickwork_haar(n, bw_d, rng)
        raw_unitaries.append(u.astype(np.complex64))
        raw_features.append(feats.unitary_to_features(u))

        site_feats = feats.per_qubit_choi_features(u)  # [n, 32]
        # Zero-pad to [max_qubits, 32] so all rows have the same shape.
        padded_lf = np.zeros((max_qubits, site_feat_dim), dtype=np.float32)
        padded_lf[:n] = site_feats
        raw_local_features.append(padded_lf)

        result = qb.best_baseline(
            u, local_vocab,
            optimization_levels=spec.qiskit_optimization_levels,
            seed=spec.seed + global_offset + i,
            min_verify_fidelity=spec.fidelity_threshold,
        )
        # Remap token IDs from n-specific vocab to canonical vocab.
        remapped = _vocab.remap_tokens(result.tokens, local_vocab, canonical_vocab)
        raw_tokens.append(remapped)
        raw_angles.append(result.angles)
        cnot_counts[i] = result.cnot_count
        total_gates_arr[i] = result.total_gates
        depths[i] = result.depth
        fidelities[i] = result.fidelity
        bw_depths[i] = bw_d

        if verbose and ((i + 1) % max(1, N // 10) == 0 or i == N - 1):
            elapsed = time.perf_counter() - t0
            print(
                f"[{global_offset + i + 1}/{global_total}] n={n} bw_depth={bw_d} "
                f"cx={result.cnot_count} F={result.fidelity:.4f} elapsed={elapsed:.1f}s"
            )

    return {
        "unitaries": np.stack(raw_unitaries, axis=0),
        "features": np.stack(raw_features, axis=0),
        "local_features": np.stack(raw_local_features, axis=0),
        "raw_tokens": raw_tokens,
        "raw_angles": raw_angles,
        "cnot_count": cnot_counts,
        "total_gates": total_gates_arr,
        "depth": depths,
        "fidelity": fidelities,
        "brickwork_depth": bw_depths,
        "num_qubits_per_sample": nq_arr,
    }


def generate_dataset(
    spec: DatasetSpec,
    *,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Build the dataset arrays in memory and return them as a dict.

    Returned arrays:
      ``unitaries`` (complex64, [N_total, d_max, d_max]) — NOTE: d_max = 2^max_qubits;
          rows for n<max_qubits are padded with zeros
      ``features``  (float32,  [N_total, 2*d*d]) — phase-normalized; d=2^n per sample
          NOTE: features can differ in size for different n; we store per-n segments and
          pad flat features to 2*d_max*d_max
      ``local_features`` (float32, [N_total, max_qubits, 32]) — zero-padded
      ``tokens``    (int32,    [N_total, max_len]) — canonical vocab IDs, padded with PAD
      ``angles``    (float32,  [N_total, max_len])
      ``lengths``   (int32,    [N_total])
      ``cnot_count`` (int32,   [N_total])
      ``total_gates`` (int32,  [N_total])
      ``depth``      (int32,   [N_total])
      ``fidelity``   (float32, [N_total])
      ``brickwork_depth`` (int32, [N_total])
      ``num_qubits_per_sample`` (int32, [N_total]) — actual n for each row
      ``token_is_parametric`` (bool, [vocab_size]) — from canonical vocab
      ``vocab_token_names`` (object, [vocab_size]) — from canonical vocab
      ``num_qubits`` (int32, scalar) — max_qubits (canonical)
      ``max_qubits`` (int32, scalar)
    """
    nq_list = spec.effective_num_qubits_list
    max_qubits = spec.effective_max_qubits
    canonical_vocab = _vocab.build_vocab(max_qubits, spec.rotation_gates)

    rng = np.random.default_rng(spec.seed)
    N_per = int(spec.num_samples)
    N_total = N_per * len(nq_list)

    all_parts: list[dict] = []
    offset = 0
    for n in nq_list:
        part = _generate_for_n(
            n, N_per, spec, canonical_vocab, rng,
            verbose=verbose,
            global_offset=offset,
            global_total=N_total,
        )
        all_parts.append(part)
        offset += N_per

    # Concatenate arrays.
    unitaries_list = []
    d_max = 2 ** max_qubits
    for part, n in zip(all_parts, nq_list):
        d_n = 2 ** n
        padded_u = np.zeros((N_per, d_max, d_max), dtype=np.complex64)
        padded_u[:, :d_n, :d_n] = part["unitaries"]
        unitaries_list.append(padded_u)

    features_list = []
    feat_dim_max = 2 * d_max * d_max
    for part, n in zip(all_parts, nq_list):
        d_n = 2 ** n
        feat_dim_n = 2 * d_n * d_n
        padded_f = np.zeros((N_per, feat_dim_max), dtype=np.float32)
        padded_f[:, :feat_dim_n] = part["features"]
        features_list.append(padded_f)

    pad_id = _vocab.PAD_TOKEN_ID
    all_raw_tokens = [t for part in all_parts for t in part["raw_tokens"]]
    all_raw_angles = [a for part in all_parts for a in part["raw_angles"]]
    max_len = int(max((t.size for t in all_raw_tokens), default=0))

    tokens_arr = np.full((N_total, max_len), pad_id, dtype=np.int32)
    angles_arr = np.zeros((N_total, max_len), dtype=np.float32)
    lengths = np.zeros((N_total,), dtype=np.int32)
    for i, (tok, ang) in enumerate(zip(all_raw_tokens, all_raw_angles)):
        pad_t, pad_a, L = _pad_tokens_and_angles(tok, ang, max_len, pad_id)
        tokens_arr[i] = pad_t
        angles_arr[i] = pad_a
        lengths[i] = L

    return {
        "unitaries": np.concatenate(unitaries_list, axis=0),
        "features": np.concatenate(features_list, axis=0),
        "local_features": np.concatenate(
            [p["local_features"] for p in all_parts], axis=0
        ),
        "tokens": tokens_arr,
        "angles": angles_arr,
        "lengths": lengths,
        "cnot_count": np.concatenate([p["cnot_count"] for p in all_parts]),
        "total_gates": np.concatenate([p["total_gates"] for p in all_parts]),
        "depth": np.concatenate([p["depth"] for p in all_parts]),
        "fidelity": np.concatenate([p["fidelity"] for p in all_parts]),
        "brickwork_depth": np.concatenate([p["brickwork_depth"] for p in all_parts]),
        "num_qubits_per_sample": np.concatenate(
            [p["num_qubits_per_sample"] for p in all_parts]
        ),
        "token_is_parametric": np.asarray(canonical_vocab.is_parametric, dtype=bool),
        "vocab_token_names": np.asarray(canonical_vocab.token_names, dtype=object),
        "num_qubits": np.int32(max_qubits),
        "max_qubits": np.int32(max_qubits),
        "rotation_gates": np.asarray(canonical_vocab.rotation_gates, dtype=object),
        "vocab_qubit0": np.asarray(canonical_vocab.qubit0, dtype=np.int32),
        "vocab_qubit1": np.asarray(canonical_vocab.qubit1, dtype=np.int32),
    }


def save_dataset(arrays: dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_dataset(path: str) -> dict[str, np.ndarray]:
    """Load a dataset saved by :func:`save_dataset` into a dict of arrays."""
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def vocab_from_dataset(d: dict[str, np.ndarray]) -> _vocab.Vocab:
    """Reconstruct the :class:`vocab.Vocab` from a saved dataset dict."""
    rotation_gates = tuple(
        str(g) for g in np.atleast_1d(d["rotation_gates"]).tolist()
    )
    # Use max_qubits if present (multi-n dataset); fall back to num_qubits for old datasets.
    nq = int(d["max_qubits"]) if "max_qubits" in d else int(d["num_qubits"])
    return _vocab.build_vocab(nq, rotation_gates)


def dataset_summary(arrays: dict[str, np.ndarray]) -> str:
    n = int(arrays["features"].shape[0])
    cx = arrays["cnot_count"]
    fid = arrays["fidelity"]
    succ = (fid >= 0.999).sum()
    nq_field = "max_qubits" if "max_qubits" in arrays else "num_qubits"
    return (
        f"N={n} samples, n_qubits_canonical={int(arrays[nq_field])}, "
        f"vocab_size={arrays['vocab_token_names'].size}, "
        f"max_len={arrays['tokens'].shape[1]}, "
        f"cnot_min={int(cx.min())} max={int(cx.max())} "
        f"mean={float(cx.mean()):.2f}, "
        f"verify_success={int(succ)}/{n}"
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate brickwork-Haar dataset.")
    parser.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    parser.add_argument("--out", default=None,
                        help="Override dataset.out_path from the config.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-qubits", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    spec = DatasetSpec.from_yaml(cfg)
    if args.num_samples is not None:
        spec = DatasetSpec(**{**spec.__dict__, "num_samples": args.num_samples})
    if args.num_qubits is not None:
        spec = DatasetSpec(**{**spec.__dict__, "num_qubits": args.num_qubits,
                               "num_qubits_list": ()})

    out_path = args.out or os.path.join(HERE, cfg["dataset"]["out_path"])
    arrays = generate_dataset(spec, verbose=not args.quiet)
    save_dataset(arrays, out_path)
    print(dataset_summary(arrays))
    print(f"saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
