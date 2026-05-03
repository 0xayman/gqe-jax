"""Inference helpers for the trained budget classifier and angle initializer.

Two encoder variants are supported:

* **Flat encoder** (``load_budget`` / ``load_angle_init``): the original MLP
  encoder that takes a flat ``[2*d*d]`` feature vector.  n-specific; one
  checkpoint per qubit count.

* **Local encoder** (``load_local_budget`` / ``load_local_angle_init``): the
  qubit-agnostic set-attention encoder that takes per-qubit Choi features of
  shape ``[n, 32]``.  A single checkpoint works for any number of qubits.


The two checkpoints are independent. Either can be loaded on its own:

    from infer import load_budget, suggest_budget, load_angle_init, suggest_angles

    bm_loaded = load_budget("checkpoints/budget/budget_model.pkl")
    suggestion = suggest_budget(bm_loaded, U_target, confidence=0.7)

    ai_loaded = load_angle_init("checkpoints/angle/angle_init.pkl")
    angles = suggest_angles(ai_loaded, U_target, skeleton_tokens, vocab)

The lower-CNOT-budget recommendation is meant to seed the GQE search; the
final answer is still produced by GQE plus angle refinement, so callers
should treat ``suggest_budget`` as a *prior*, not a certificate.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import angle_init as ai  # noqa: E402
import budget_model as bm  # noqa: E402
import encoder as enc  # noqa: E402
import features as feats  # noqa: E402
import train_utils as tu  # noqa: E402
import vocab as _vocab  # noqa: E402


@dataclass
class LoadedBudgetModel:
    model: bm.BudgetModel
    params: dict
    enc_cfg: enc.EncoderConfig
    head_cfg: bm.BudgetConfig


@dataclass
class LoadedAngleInit:
    model: ai.AngleInitModel
    params: dict
    enc_cfg: enc.EncoderConfig
    head_cfg: ai.AngleInitConfig


def _restore_enc_cfg(d: dict) -> enc.EncoderConfig:
    return enc.EncoderConfig(
        input_dim=int(d["input_dim"]),
        hidden_dim=int(d["hidden_dim"]),
        latent_dim=int(d["latent_dim"]),
        num_layers=int(d["num_layers"]),
        dropout=float(d["dropout"]),
    )


def load_budget(path: str) -> LoadedBudgetModel:
    payload = tu.load_checkpoint(path)
    enc_cfg = _restore_enc_cfg(payload["extras"]["enc_cfg"])
    head = payload["extras"]["head_cfg"]
    head_cfg = bm.BudgetConfig(
        k_min=int(head["k_min"]),
        k_max=int(head["k_max"]),
        hidden_dim=int(head["hidden_dim"]),
    )
    model = bm.BudgetModel(enc_cfg, head_cfg)
    return LoadedBudgetModel(
        model=model, params=payload["params"],
        enc_cfg=enc_cfg, head_cfg=head_cfg,
    )


def load_angle_init(path: str) -> LoadedAngleInit:
    payload = tu.load_checkpoint(path)
    enc_cfg = _restore_enc_cfg(payload["extras"]["enc_cfg"])
    head = payload["extras"]["head_cfg"]
    head_cfg = ai.AngleInitConfig(
        vocab_size=int(head["vocab_size"]),
        max_skeleton_len=int(head["max_skeleton_len"]),
        hidden_dim=int(head["hidden_dim"]),
        num_layers=int(head["num_layers"]),
        num_heads=int(head["num_heads"]),
        dropout=float(head["dropout"]),
    )
    model = ai.AngleInitModel(enc_cfg, head_cfg)
    return LoadedAngleInit(
        model=model, params=payload["params"],
        enc_cfg=enc_cfg, head_cfg=head_cfg,
    )


@dataclass
class BudgetSuggestion:
    """Result of a budget-classifier query."""

    suggested_k: int                # smallest k with P(success | k) >= confidence
    above_max: bool                  # True if no budget met the threshold
    probs: np.ndarray                # shape [K] — P(success | k) for k in [k_min..k_max]


def suggest_budget(
    loaded: LoadedBudgetModel,
    u_target: np.ndarray,
    *,
    confidence: float = 0.5,
) -> BudgetSuggestion:
    """Run the budget classifier and pick the smallest sufficient ``k``."""
    feat = feats.unitary_to_features(u_target)
    feat_jax = jnp.asarray(feat[None, :], dtype=jnp.float32)
    logits = loaded.model.apply(
        {"params": loaded.params}, feat_jax, deterministic=True,
    )
    probs = np.asarray(jax.nn.sigmoid(logits)[0], dtype=np.float32)
    accepts = probs >= confidence
    if accepts.any():
        idx = int(np.argmax(accepts))
        return BudgetSuggestion(
            suggested_k=int(loaded.head_cfg.k_min + idx),
            above_max=False,
            probs=probs,
        )
    return BudgetSuggestion(
        suggested_k=int(loaded.head_cfg.k_max + 1),
        above_max=True,
        probs=probs,
    )


def suggest_angles(
    loaded: LoadedAngleInit,
    u_target: np.ndarray,
    skeleton_tokens: np.ndarray,
    vocab: _vocab.Vocab,
) -> np.ndarray:
    """Predict initial angles for ``skeleton_tokens`` (raw, no specials).

    Non-parametric tokens (SX/CNOT) get a 0 angle in the output. The
    returned array has the same shape as the input skeleton.
    """
    skeleton_tokens = np.asarray(skeleton_tokens, dtype=np.int32)
    L_in = int(skeleton_tokens.size)
    L_max = int(loaded.head_cfg.max_skeleton_len)
    if L_in > L_max:
        raise ValueError(
            f"skeleton length {L_in} exceeds head max {L_max}; "
            f"retrain with a larger max_skeleton_len"
        )
    padded = np.full((L_max,), _vocab.PAD_TOKEN_ID, dtype=np.int32)
    padded[:L_in] = skeleton_tokens
    valid = np.zeros((L_max,), dtype=bool)
    valid[:L_in] = True

    feat = feats.unitary_to_features(u_target)
    feats_jax = jnp.asarray(feat[None, :], dtype=jnp.float32)
    tokens_jax = jnp.asarray(padded[None, :], dtype=jnp.int32)
    valid_jax = jnp.asarray(valid[None, :], dtype=bool)
    raw = loaded.model.apply(
        {"params": loaded.params}, feats_jax, tokens_jax, valid_jax,
        deterministic=True,
    )
    pred = np.asarray(raw[0], dtype=np.float32)[:L_in]

    is_param = np.asarray(vocab.is_parametric, dtype=bool)
    param_mask = is_param[skeleton_tokens]
    out = np.where(param_mask, pred, 0.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Qubit-agnostic (local encoder) inference
# ---------------------------------------------------------------------------


@dataclass
class LoadedLocalBudgetModel:
    model: bm.LocalBudgetModel
    params: dict
    local_enc_cfg: enc.LocalEncoderConfig
    head_cfg: bm.BudgetConfig
    max_qubits: int = 0  # 0 means unknown; features are not padded at inference


@dataclass
class LoadedLocalAngleInit:
    model: ai.LocalAngleInitModel
    params: dict
    local_enc_cfg: enc.LocalEncoderConfig
    head_cfg: ai.AngleInitConfig
    max_qubits: int = 0  # 0 means unknown; features are not padded at inference


def _restore_local_enc_cfg(d: dict) -> enc.LocalEncoderConfig:
    return enc.LocalEncoderConfig(
        site_feat_dim=int(d["site_feat_dim"]),
        hidden_dim=int(d["hidden_dim"]),
        latent_dim=int(d["latent_dim"]),
        num_layers=int(d["num_layers"]),
        num_heads=int(d["num_heads"]),
        dropout=float(d["dropout"]),
    )


def load_local_budget(path: str) -> LoadedLocalBudgetModel:
    payload = tu.load_checkpoint(path)
    local_enc_cfg = _restore_local_enc_cfg(payload["extras"]["local_enc_cfg"])
    head = payload["extras"]["head_cfg"]
    head_cfg = bm.BudgetConfig(
        k_min=int(head["k_min"]),
        k_max=int(head["k_max"]),
        hidden_dim=int(head["hidden_dim"]),
    )
    model = bm.LocalBudgetModel(local_enc_cfg, head_cfg)
    max_qubits = int(payload["extras"].get("max_qubits", 0))
    return LoadedLocalBudgetModel(
        model=model, params=payload["params"],
        local_enc_cfg=local_enc_cfg, head_cfg=head_cfg,
        max_qubits=max_qubits,
    )


def load_local_angle_init(path: str) -> LoadedLocalAngleInit:
    payload = tu.load_checkpoint(path)
    local_enc_cfg = _restore_local_enc_cfg(payload["extras"]["local_enc_cfg"])
    head = payload["extras"]["head_cfg"]
    head_cfg = ai.AngleInitConfig(
        vocab_size=int(head["vocab_size"]),
        max_skeleton_len=int(head["max_skeleton_len"]),
        hidden_dim=int(head["hidden_dim"]),
        num_layers=int(head["num_layers"]),
        num_heads=int(head["num_heads"]),
        dropout=float(head["dropout"]),
    )
    model = ai.LocalAngleInitModel(local_enc_cfg, head_cfg)
    max_qubits = int(payload["extras"].get("max_qubits", 0))
    return LoadedLocalAngleInit(
        model=model, params=payload["params"],
        local_enc_cfg=local_enc_cfg, head_cfg=head_cfg,
        max_qubits=max_qubits,
    )


def _pad_local_features(
    local_feat: np.ndarray,
    max_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad [n, 32] features to [max_qubits, 32] and return (padded, site_mask [1, max_qubits])."""
    n = local_feat.shape[0]
    site_feat_dim = local_feat.shape[1]
    padded = np.zeros((max_qubits, site_feat_dim), dtype=np.float32)
    padded[:n] = local_feat
    mask = np.zeros((1, max_qubits), dtype=bool)
    mask[0, :n] = True
    return padded, mask


def suggest_budget_local(
    loaded: LoadedLocalBudgetModel,
    u_target: np.ndarray,
    *,
    confidence: float = 0.5,
) -> BudgetSuggestion:
    """Run the qubit-agnostic budget classifier and pick the smallest ``k``."""
    local_feat = feats.per_qubit_choi_features(u_target)
    if loaded.max_qubits > 0 and local_feat.shape[0] < loaded.max_qubits:
        local_feat, site_mask = _pad_local_features(local_feat, loaded.max_qubits)
        site_mask_jax = jnp.asarray(site_mask, dtype=bool)
    else:
        site_mask_jax = None
    local_feat_jax = jnp.asarray(local_feat[None, :, :], dtype=jnp.float32)
    logits = loaded.model.apply(
        {"params": loaded.params}, local_feat_jax,
        deterministic=True, site_mask=site_mask_jax,
    )
    probs = np.asarray(jax.nn.sigmoid(logits)[0], dtype=np.float32)
    accepts = probs >= confidence
    if accepts.any():
        idx = int(np.argmax(accepts))
        return BudgetSuggestion(
            suggested_k=int(loaded.head_cfg.k_min + idx),
            above_max=False,
            probs=probs,
        )
    return BudgetSuggestion(
        suggested_k=int(loaded.head_cfg.k_max + 1),
        above_max=True,
        probs=probs,
    )


def suggest_angles_local(
    loaded: LoadedLocalAngleInit,
    u_target: np.ndarray,
    skeleton_tokens: np.ndarray,
    vocab: _vocab.Vocab,
) -> np.ndarray:
    """Predict initial angles using the qubit-agnostic model."""
    skeleton_tokens = np.asarray(skeleton_tokens, dtype=np.int32)
    L_in = int(skeleton_tokens.size)
    L_max = int(loaded.head_cfg.max_skeleton_len)
    if L_in > L_max:
        raise ValueError(
            f"skeleton length {L_in} exceeds head max {L_max}; "
            f"retrain with a larger max_skeleton_len"
        )
    padded = np.full((L_max,), _vocab.PAD_TOKEN_ID, dtype=np.int32)
    padded[:L_in] = skeleton_tokens
    valid = np.zeros((L_max,), dtype=bool)
    valid[:L_in] = True

    local_feat = feats.per_qubit_choi_features(u_target)
    if loaded.max_qubits > 0 and local_feat.shape[0] < loaded.max_qubits:
        local_feat, site_mask = _pad_local_features(local_feat, loaded.max_qubits)
        site_mask_jax = jnp.asarray(site_mask, dtype=bool)
    else:
        site_mask_jax = None
    local_feat_jax = jnp.asarray(local_feat[None, :, :], dtype=jnp.float32)
    tokens_jax = jnp.asarray(padded[None, :], dtype=jnp.int32)
    valid_jax = jnp.asarray(valid[None, :], dtype=bool)
    raw = loaded.model.apply(
        {"params": loaded.params}, local_feat_jax, tokens_jax, valid_jax,
        deterministic=True, site_mask=site_mask_jax,
    )
    pred = np.asarray(raw[0], dtype=np.float32)[:L_in]

    is_param = np.asarray(vocab.is_parametric, dtype=bool)
    param_mask = is_param[skeleton_tokens]
    out = np.where(param_mask, pred, 0.0).astype(np.float32)
    return out
