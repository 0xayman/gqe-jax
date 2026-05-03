"""Train the budget classifier on the Qiskit-baseline dataset.

Usage:
    python train_budget.py --config config.yml --dataset data/brickwork_n2.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import budget_model as bm  # noqa: E402
import dataset as ds  # noqa: E402
import encoder as enc  # noqa: E402
import train_utils as tu  # noqa: E402


@dataclass
class BudgetTrainSpec:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_split: float
    seed: int
    out_dir: str

    @classmethod
    def from_yaml(cls, cfg: dict) -> "BudgetTrainSpec":
        b = cfg["training"]["budget"]
        return cls(
            epochs=int(b["epochs"]),
            batch_size=int(b["batch_size"]),
            lr=float(b["lr"]),
            weight_decay=float(b["weight_decay"]),
            val_split=float(b["val_split"]),
            seed=int(b["seed"]),
            out_dir=str(b["out_dir"]),
        )

    @classmethod
    def from_yaml_local(cls, cfg: dict) -> "BudgetTrainSpec":
        b = cfg["training"]["local_budget"]
        return cls(
            epochs=int(b["epochs"]),
            batch_size=int(b["batch_size"]),
            lr=float(b["lr"]),
            weight_decay=float(b["weight_decay"]),
            val_split=float(b["val_split"]),
            seed=int(b["seed"]),
            out_dir=str(b["out_dir"]),
        )


def _build_model(
    enc_cfg: enc.EncoderConfig,
    head_cfg: bm.BudgetConfig,
    rng: jax.Array,
):
    model = bm.BudgetModel(enc_cfg, head_cfg)
    dummy = jnp.zeros((4, enc_cfg.input_dim), dtype=jnp.float32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, deterministic=True)
    return model, variables["params"]


def train_budget(
    arrays: dict[str, np.ndarray],
    *,
    spec: BudgetTrainSpec,
    enc_cfg: enc.EncoderConfig,
    head_cfg: bm.BudgetConfig,
    verbose: bool = True,
) -> dict:
    rng = jax.random.PRNGKey(spec.seed)
    init_rng, dropout_rng = jax.random.split(rng)
    model, params = _build_model(enc_cfg, head_cfg, init_rng)

    optimizer = tu.make_optimizer(spec.lr, spec.weight_decay)
    opt_state = optimizer.init(params)

    features = arrays["features"].astype(np.float32)
    cnots = arrays["cnot_count"].astype(np.int32)
    n = features.shape[0]
    split = tu.make_split(n, spec.val_split, spec.seed)
    rng_np = np.random.default_rng(spec.seed)

    @jax.jit
    def train_step(params, opt_state, feats, cnots_b, drop_rng):
        def loss_fn(p):
            logits = model.apply(
                {"params": p}, feats,
                rngs={"dropout": drop_rng},
                deterministic=False,
            )
            loss, metrics = bm.budget_loss(logits, cnots_b, head_cfg)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, metrics

    @jax.jit
    def eval_step(params, feats, cnots_b):
        logits = model.apply({"params": params}, feats, deterministic=True)
        _, metrics = bm.budget_loss(logits, cnots_b, head_cfg)
        return metrics

    history: list[dict] = []
    for epoch in range(spec.epochs):
        t0 = time.perf_counter()
        train_metrics: list[dict] = []
        for batch_idx in tu.iter_minibatches(
            split.train_idx, spec.batch_size,
            shuffle=True, rng=rng_np, drop_last=True,
        ):
            feats_b = jnp.asarray(features[batch_idx], dtype=jnp.float32)
            cnots_b = jnp.asarray(cnots[batch_idx], dtype=jnp.int32)
            dropout_rng, sub = jax.random.split(dropout_rng)
            params, opt_state, m = train_step(params, opt_state, feats_b, cnots_b, sub)
            train_metrics.append({k: float(v) for k, v in m.items()})

        val_metrics: list[dict] = []
        if split.val_idx.size:
            for batch_idx in tu.iter_minibatches(
                split.val_idx, spec.batch_size,
                shuffle=False, rng=rng_np, drop_last=False,
            ):
                feats_b = jnp.asarray(features[batch_idx], dtype=jnp.float32)
                cnots_b = jnp.asarray(cnots[batch_idx], dtype=jnp.int32)
                m = eval_step(params, feats_b, cnots_b)
                val_metrics.append({k: float(v) for k, v in m.items()})

        train_avg = tu.merge_metrics(train_metrics)
        val_avg = tu.merge_metrics(val_metrics)
        epoch_time = time.perf_counter() - t0
        history.append({"epoch": int(epoch), "time": epoch_time, **{
            f"train/{k}": v for k, v in train_avg.items()
        }, **{
            f"val/{k}": v for k, v in val_avg.items()
        }})
        if verbose:
            tr = " ".join(f"{k}={v:.4f}" for k, v in train_avg.items())
            va = " ".join(f"{k}={v:.4f}" for k, v in val_avg.items())
            print(f"[budget] epoch {epoch+1:03d}/{spec.epochs:03d} "
                  f"train: {tr}  val: {va}  dt={epoch_time:.1f}s")

    out_dir = spec.out_dir if os.path.isabs(spec.out_dir) else os.path.join(HERE, spec.out_dir)
    ckpt_path = tu.save_checkpoint(out_dir, "budget_model", params, extras={
        "enc_cfg": enc_cfg.__dict__,
        "head_cfg": head_cfg.__dict__,
        "history": history,
    })
    return {"params": params, "history": history, "checkpoint": ckpt_path}


def _make_site_mask(local_features: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return [N, max_qubits] bool mask: True at valid qubit positions.

    Uses ``num_qubits_per_sample`` when present (multi-n dataset). Falls back to
    all-True for the first ``num_qubits`` positions when the dataset is single-n.
    """
    max_qubits = local_features.shape[1]
    N = local_features.shape[0]
    if "num_qubits_per_sample" in arrays:
        nq_b = arrays["num_qubits_per_sample"].astype(np.int32)
        q_idx = np.arange(max_qubits)[None, :]
        return q_idx < nq_b[:, None]
    else:
        nq = int(arrays["num_qubits"])
        mask = np.zeros((N, max_qubits), dtype=bool)
        mask[:, :nq] = True
        return mask


def train_local_budget(
    arrays: dict[str, np.ndarray],
    *,
    spec: BudgetTrainSpec,
    local_enc_cfg: enc.LocalEncoderConfig,
    head_cfg: bm.BudgetConfig,
    verbose: bool = True,
) -> dict:
    """Train the qubit-agnostic budget classifier on ``local_features``."""
    rng = jax.random.PRNGKey(spec.seed)
    init_rng, dropout_rng = jax.random.split(rng)

    local_features = arrays["local_features"].astype(np.float32)
    n_sites = local_features.shape[1]
    site_mask_np = _make_site_mask(local_features, arrays)

    model = bm.LocalBudgetModel(local_enc_cfg, head_cfg)
    dummy = jnp.zeros((4, n_sites, local_enc_cfg.site_feat_dim), dtype=jnp.float32)
    dummy_mask = jnp.ones((4, n_sites), dtype=bool)
    variables = model.init(
        {"params": init_rng, "dropout": init_rng}, dummy,
        deterministic=True, site_mask=dummy_mask,
    )
    params = variables["params"]

    optimizer = tu.make_optimizer(spec.lr, spec.weight_decay)
    opt_state = optimizer.init(params)

    cnots = arrays["cnot_count"].astype(np.int32)
    n = local_features.shape[0]
    split = tu.make_split(n, spec.val_split, spec.seed)
    rng_np = np.random.default_rng(spec.seed)

    @jax.jit
    def train_step(params, opt_state, lf_b, cnots_b, smask_b, drop_rng):
        def loss_fn(p):
            logits = model.apply(
                {"params": p}, lf_b,
                rngs={"dropout": drop_rng},
                deterministic=False,
                site_mask=smask_b,
            )
            loss, metrics = bm.budget_loss(logits, cnots_b, head_cfg)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, metrics

    @jax.jit
    def eval_step(params, lf_b, cnots_b, smask_b):
        logits = model.apply(
            {"params": params}, lf_b, deterministic=True, site_mask=smask_b,
        )
        _, metrics = bm.budget_loss(logits, cnots_b, head_cfg)
        return metrics

    history: list[dict] = []
    for epoch in range(spec.epochs):
        t0 = time.perf_counter()
        train_metrics: list[dict] = []
        for batch_idx in tu.iter_minibatches(
            split.train_idx, spec.batch_size,
            shuffle=True, rng=rng_np, drop_last=True,
        ):
            lf_b = jnp.asarray(local_features[batch_idx], dtype=jnp.float32)
            cnots_b = jnp.asarray(cnots[batch_idx], dtype=jnp.int32)
            smask_b = jnp.asarray(site_mask_np[batch_idx], dtype=bool)
            dropout_rng, sub = jax.random.split(dropout_rng)
            params, opt_state, m = train_step(params, opt_state, lf_b, cnots_b, smask_b, sub)
            train_metrics.append({k: float(v) for k, v in m.items()})

        val_metrics: list[dict] = []
        if split.val_idx.size:
            for batch_idx in tu.iter_minibatches(
                split.val_idx, spec.batch_size,
                shuffle=False, rng=rng_np, drop_last=False,
            ):
                lf_b = jnp.asarray(local_features[batch_idx], dtype=jnp.float32)
                cnots_b = jnp.asarray(cnots[batch_idx], dtype=jnp.int32)
                smask_b = jnp.asarray(site_mask_np[batch_idx], dtype=bool)
                m = eval_step(params, lf_b, cnots_b, smask_b)
                val_metrics.append({k: float(v) for k, v in m.items()})

        train_avg = tu.merge_metrics(train_metrics)
        val_avg = tu.merge_metrics(val_metrics)
        epoch_time = time.perf_counter() - t0
        history.append({"epoch": int(epoch), "time": epoch_time, **{
            f"train/{k}": v for k, v in train_avg.items()
        }, **{
            f"val/{k}": v for k, v in val_avg.items()
        }})
        if verbose:
            tr = " ".join(f"{k}={v:.4f}" for k, v in train_avg.items())
            va = " ".join(f"{k}={v:.4f}" for k, v in val_avg.items())
            print(f"[local_budget] epoch {epoch+1:03d}/{spec.epochs:03d} "
                  f"train: {tr}  val: {va}  dt={epoch_time:.1f}s")

    out_dir = spec.out_dir if os.path.isabs(spec.out_dir) else os.path.join(HERE, spec.out_dir)
    ckpt_path = tu.save_checkpoint(out_dir, "local_budget_model", params, extras={
        "local_enc_cfg": local_enc_cfg.__dict__,
        "head_cfg": head_cfg.__dict__,
        "max_qubits": int(local_features.shape[1]),
        "history": history,
    })
    return {"params": params, "history": history, "checkpoint": ckpt_path}


def _build_configs(cfg: dict, arrays: dict[str, np.ndarray]):
    enc_cfg = enc.EncoderConfig(
        input_dim=int(arrays["features"].shape[1]),
        hidden_dim=int(cfg["encoder"]["hidden_dim"]),
        latent_dim=int(cfg["encoder"]["latent_dim"]),
        num_layers=int(cfg["encoder"]["num_layers"]),
        dropout=float(cfg["encoder"]["dropout"]),
    )
    head_cfg = bm.BudgetConfig(
        k_min=int(cfg["budget"]["k_min"]),
        k_max=int(cfg["budget"]["k_max"]),
        hidden_dim=int(cfg["budget"]["hidden_dim"]),
    )
    return enc_cfg, head_cfg


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CNOT-budget classifier.")
    parser.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    parser.add_argument("--dataset", default=None,
                        help="Path to a saved dataset npz. Defaults to dataset.out_path.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    dataset_path = args.dataset or os.path.join(HERE, cfg["dataset"]["out_path"])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"dataset not found at {dataset_path}. Generate it first via "
            f"`python dataset.py --config {args.config}`."
        )
    arrays = ds.load_dataset(dataset_path)
    enc_cfg, head_cfg = _build_configs(cfg, arrays)

    spec = BudgetTrainSpec.from_yaml(cfg)
    res = train_budget(
        arrays, spec=spec, enc_cfg=enc_cfg, head_cfg=head_cfg,
        verbose=not args.quiet,
    )
    print(f"[budget] checkpoint -> {res['checkpoint']}")

    local_enc_cfg = enc.LocalEncoderConfig(
        site_feat_dim=int(cfg["local_encoder"]["site_feat_dim"]),
        hidden_dim=int(cfg["local_encoder"]["hidden_dim"]),
        latent_dim=int(cfg["local_encoder"]["latent_dim"]),
        num_layers=int(cfg["local_encoder"]["num_layers"]),
        num_heads=int(cfg["local_encoder"]["num_heads"]),
        dropout=float(cfg["local_encoder"]["dropout"]),
    )
    local_spec = BudgetTrainSpec.from_yaml_local(cfg)
    local_res = train_local_budget(
        arrays, spec=local_spec, local_enc_cfg=local_enc_cfg, head_cfg=head_cfg,
        verbose=not args.quiet,
    )
    print(f"[local_budget] checkpoint -> {local_res['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
