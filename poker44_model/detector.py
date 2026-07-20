"""Poker44 bot detector (BEATER) -- a WITHIN-BATCH RANK-FUSED ENSEMBLE of three
decorrelated members over the SAME 180-dim sanitization-invariant C2 feature row,
topped with our reward-fit, FPR-capped floating decision layer.

Why this model (closing the live-rank gap)
-------------------------------------------
Our benchmark GroupKFold-by-date AP (~0.92 on the 180 C2 features) already EQUALS
the steady top tier's, yet our single-LightGBM mains score ~0.49 live while the
steady band scores ~0.55. The whole gap is in the 65% RANK block
(0.35*AP + 0.30*recall@FPR<=0.05); we already max the 30% hard-0.5-threshold block.
Root cause: a single learner is the highest-variance choice out-of-distribution on
the sanitized live feed. Every steady winner runs a within-batch RANK-FUSED
ENSEMBLE of decorrelated members plus sign-stability-gated monotone constraints.
This model ports that recipe over OUR feature pipeline verbatim.

Members (all over the identical 180-dim FEATURE_NAMES row)
---------------------------------------------------------
  1. STACK  -- LGBM + XGB + RF -> logistic OOF stack (the discrimination anchor).
  2. MONO   -- monotone-constrained LightGBM bag; monotone_constraints set to
               +1/-1 ONLY for features whose per-DATE Spearman(feature,label) sign
               is stable across >=70% of dates AND |mean rho| >= 0.05 (else 0).
               The OOD-transfer regularizer.
  3. MLP    -- PCA(56) -> MLP bag on the standardized feature view; architecturally
               decorrelated from the tree members.

Fusion is calibration-free: each member's WITHIN-BATCH rank (argsort/argsort/(n-1))
is averaged with artifact weights (currently 0.40, 0.25, 0.35), so no member's OOD score-scale
can distort the blend. The fused rank is the movable ordering that drives the 65%
RANK block.

Decision layer (STRICTLY MONOTONE; isotonic removed 2026-07-15)
---------------------------------------------------------------
The fused within-batch rank goes straight into the reward-fit per-batch decision
layer (anchor quantile Q + logit margin/temp + FLOOR + CAP), which SHIFTS each side
of the 0.5 line instead of clamping it -> the map fused -> served is strictly
monotone, so the served order IS the fused order and AP / recall@FPR (the 65% block)
are set purely by the fused rank.

Two corrections vs the previous version, both measured on live captures:
  * The isotonic map is GONE. It is monotone but NON-INJECTIVE, so it merged the
    fused rank into ~22 distinct levels per 100-chunk window and put the
    recall@FPR<=0.05 boundary inside a tie group. The old claim that the transform
    was "monotone, so AP/recall are set purely by the fused rank" was FALSE.
  * FLOOR is 0.10, not 0.02. The old claim of "zero hard-zeros" was ALSO FALSE:
    FLOOR guarantees that k chunks CROSS 0.5, not that any of them is a BOT.
    scoring.py zeroes the WHOLE round when no true bot crosses, and with k=2 the
    crossing set was decided by array index inside the isotonic tie plateau
    (index-arbitrary in 17-18 of 18 live windows) -- which produced uid212 R3 =
    0.000, uid236 R2 = 0.000, and uid236's ~0.077 epoch.
k = ceil(FLOOR*n) chunks cross 0.5; at n=100 that is 10, matching the 10% FPR
budget where threshold_sanity_quality is still 1.0.

IMPORTANT -- inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand). Only the
offline training matrix sanitizes raw benchmark hands (train == serve).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import joblib

from poker44_model.features import (
    CHALLENGER_FEATURE_NAMES,
    FEATURE_NAMES,
    challenger_features,
    chunk_features,
)

try:  # keep any torch backend single-threaded (never deadlock batched predict)
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass

_MODEL = None
CHALLENGER_STRATEGY = "rank_input_coherence_v1"


def validate_artifact(artifact):
    """Fail fast when an artifact cannot satisfy the serving contract."""
    if not isinstance(artifact, dict):
        raise TypeError("model artifact must be a dictionary")
    strategy = artifact.get("strategy")
    if strategy == CHALLENGER_STRATEGY:
        required = (
            "stack",
            "mlp",
            "rank_extra",
            "rank_hist",
            "branch_weights",
            "challenger_feature_names",
        )
        missing = [name for name in required if name not in artifact]
        if missing:
            raise ValueError(f"challenger artifact missing {missing}")
        if tuple(artifact["challenger_feature_names"]) != CHALLENGER_FEATURE_NAMES:
            raise ValueError("challenger feature schema mismatch; retrain artifact")
        weights = np.asarray(artifact["branch_weights"], dtype=float)
        if weights.shape != (4,) or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("challenger branch_weights must contain four non-negative values")
    else:
        required = ("stack", "mono", "mlp", "weights")
        missing = [name for name in required if name not in artifact]
        if missing:
            raise ValueError(f"legacy artifact missing {missing}")
        weights = np.asarray(artifact["weights"], dtype=float)
        if weights.shape != (3,) or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("legacy weights must contain three non-negative values")
    for name in ("Q", "MARGIN", "FLOOR"):
        if name not in artifact:
            raise ValueError(f"artifact missing decision parameter {name}")
    return artifact


def _pin_single_thread(est):
    """Best-effort force n_jobs/thread_count=1 so batched predict never deadlocks."""
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            pass
    # dig into containers (StackingClassifier, VotingClassifier, Pipeline)
    for holder in ("estimators_", "estimators"):
        try:
            for sub in getattr(est, holder):
                _pin_single_thread(sub[1] if isinstance(sub, tuple) else sub)
        except Exception:
            pass
    for attr in ("final_estimator_", "final_estimator"):
        try:
            _pin_single_thread(getattr(est, attr))
        except Exception:
            pass
    try:  # sklearn Pipeline
        for _, step in est.steps:
            _pin_single_thread(step)
    except Exception:
        pass


def _model():
    global _MODEL
    if _MODEL is None:
        default_path = os.path.join(os.path.dirname(__file__), "model.joblib")
        selected_path = os.path.expanduser(
            os.environ.get("POKER44_MODEL_PATH", default_path).strip() or default_path
        )
        b = validate_artifact(joblib.load(selected_path))
        for key in ("stack", "mono", "mlp", "rank_extra", "rank_hist"):
            try:
                _pin_single_thread(b[key])
            except Exception:
                pass
        _MODEL = b
    return _MODEL


def _rank01(s):
    """Within-batch rank in [0,1]: argsort/argsort/(n-1). Calibration-free."""
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def columnwise_batch_rank(matrix):
    """Tie-averaged [0,1] feature percentiles inside one request.

    Training applies this independently inside each source date. Serving
    applies it to the current validator request, avoiding frozen raw-feature
    scales while never reading labels or identifiers.
    """
    values = np.nan_to_num(
        np.asarray(matrix, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    if values.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional")
    rows, columns = values.shape
    if rows <= 1:
        return np.full_like(values, 0.5)
    ranked = np.empty_like(values)
    denominator = float(rows - 1)
    for column in range(columns):
        series = values[:, column]
        order = np.argsort(series, kind="mergesort")
        ordered = series[order]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], rows]
        for start, end in zip(starts, ends):
            average = (float(start) + float(end - 1)) / (2.0 * denominator)
            ranked[order[start:end], column] = average
    return ranked


def _strict_rank01(scores, tie_keys=None):
    """Total-order branch scores without index-dependent tie plateaus."""
    values = np.asarray(scores, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values)
    keys = list(tie_keys or [f"{index:012d}" for index in range(values.size)])
    if len(keys) != values.size:
        raise ValueError("tie key count does not match scores")
    order = sorted(
        range(values.size), key=lambda index: (float(values[index]), str(keys[index]))
    )
    ranked = np.empty(values.size, dtype=float)
    ranked[order] = np.arange(values.size, dtype=float) / (values.size - 1)
    return ranked


def _chunk_tie_key(chunk):
    payload = json.dumps(chunk, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rows(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def _challenger_rows(chunks):
    rows = []
    for chunk in chunks:
        features = challenger_features(chunk)
        rows.append([features.get(name, 0.0) for name in CHALLENGER_FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


_T_HI = 0.00040000000000000002   # logit(0.5001): sigmoid(t) >= 0.5001 <=> t >= _T_HI
_T_LO = -0.00040000000000000002  # logit(0.4999): sigmoid(t) <= 0.4999 <=> t <= _T_LO


def _decision(model, v):
    """Reward-fit, FPR-capped per-batch decision layer on the TIE-FREE fused rank.

    Identical to the deployed layer (same Q / MARGIN / TEMP / FLOOR / CAP / EPS /
    train_ref_logit, same k, same crossing count) except for two tie sources that
    were destroying the 65% rank block (0.35*AP + 0.30*recall@FPR<=0.05, both of
    which argsort the served scores and break ties by ARRAY INDEX):

      1. the isotonic map is GONE -- it is monotone but NON-INJECTIVE, so it
         merged the fused rank into ~26 distinct levels per 100-chunk window and
         put the recall@FPR<=0.05 boundary INSIDE a tie group;
      2. FLOOR/CAP now SHIFT each side instead of CLAMPing it to the constants
         0.5001 / 0.4999, which preserves the internal spacing of both groups.

    The result is a STRICTLY MONOTONE map fused -> served score, so the served
    order is exactly the model's order, while k = ceil(FLOOR*n) chunks still
    cross 0.5 (FLOOR lifts the top-k, CAP pins the rest below) -- the 30%
    hard-0.5-threshold block is unchanged.
    """
    eps = float(model["EPS"])
    q = float(model["Q"])
    margin = float(model["MARGIN"])
    temp = float(model.get("TEMP", 1.0))
    floor = float(model["FLOOR"])
    cap = bool(model.get("CAP", False))
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(v, eps)
    if z.size == 0:
        return []
    anchor = np.quantile(z, q)
    t = (z - anchor + tref) / temp
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(floor * len(t))))
    top, rest = order[:k], order[k:]
    # FLOOR (tie-free): shift the top-k as a block so its MINIMUM sits at 0.5001
    # -- never an all-below-0.5 hard zero, but the spacing inside the block (and
    # hence the ordering that AP / bot-recall read) survives.
    d = _T_HI - t[top].min()
    if d > 0.0:
        t[top] = t[top] + d
    if cap and rest.size:
        # CAP (tie-free): shift the rest as a block so its MAXIMUM sits at 0.4999
        # -> deterministic crossing count k, spacing preserved.
        d = t[rest].max() - _T_LO
        if d > 0.0:
            t[rest] = t[rest] - d
    scores = 1.0 / (1.0 + np.exp(-t))
    return [round(float(s), 9) for s in scores]


def score_rows_with_artifact(
    artifact,
    base_rows,
    full_rows=None,
    *,
    tie_keys=None,
    positive_fraction=None,
):
    """Score precomputed rows for reproducible window replay and training.

    The production path calls the same function after extracting rows from raw
    chunks.  Keeping a row-level entry point makes 40/100-chunk replay fast
    enough to evaluate hundreds of windows without changing inference logic.
    """
    model = validate_artifact(artifact)
    base = np.asarray(base_rows, dtype=float)
    if base.ndim != 2 or base.shape[1] != len(FEATURE_NAMES):
        raise ValueError("base feature matrix does not match FEATURE_NAMES")
    branches = branch_rank_matrix_with_artifact(
        model, base, full_rows, tie_keys=tie_keys
    )
    return score_branch_ranks_with_artifact(
        model, branches, tie_keys=tie_keys, positive_fraction=positive_fraction
    )


def branch_rank_matrix_with_artifact(
    artifact, base_rows, full_rows=None, *, tie_keys=None
):
    """Return ranked member outputs so replay can reuse expensive predictions."""
    model = validate_artifact(artifact)
    base = np.asarray(base_rows, dtype=float)
    if base.ndim != 2 or base.shape[1] != len(FEATURE_NAMES):
        raise ValueError("base feature matrix does not match FEATURE_NAMES")
    if model.get("strategy") == CHALLENGER_STRATEGY:
        full = np.asarray(full_rows, dtype=float)
        if full.shape != (len(base), len(CHALLENGER_FEATURE_NAMES)):
            raise ValueError("challenger feature matrix has the wrong shape")
        ranked_input = columnwise_batch_rank(full)
        raw = (
            model["stack"].predict_proba(base)[:, 1],
            model["mlp"].predict_proba(base)[:, 1],
            model["rank_extra"].predict_proba(ranked_input)[:, 1],
            model["rank_hist"].predict_proba(ranked_input)[:, 1],
        )
        return np.column_stack(
            [_strict_rank01(values, tie_keys=tie_keys) for values in raw]
        )
    raw = (
        model["stack"].predict_proba(base)[:, 1],
        model["mono"].predict_proba(base)[:, 1],
        model["mlp"].predict_proba(base)[:, 1],
    )
    return np.column_stack([_rank01(values) for values in raw])


def score_branch_ranks_with_artifact(
    artifact, branch_ranks, *, tie_keys=None, positive_fraction=None
):
    """Apply configured fusion and the production decision layer."""
    model = validate_artifact(artifact)
    branches = np.asarray(branch_ranks, dtype=float)
    weights = np.asarray(
        model["branch_weights"]
        if model.get("strategy") == CHALLENGER_STRATEGY
        else model["weights"],
        dtype=float,
    )
    if branches.shape != (len(branches), len(weights)):
        raise ValueError("branch rank matrix does not match artifact weights")
    fused = branches @ (weights / weights.sum())
    if model.get("strategy") == CHALLENGER_STRATEGY:
        fused = _strict_rank01(fused, tie_keys=tie_keys)
    decision_model = model
    if positive_fraction is not None:
        decision_model = dict(model)
        decision_model["FLOOR"] = float(positive_fraction)
    return _decision(decision_model, fused)


def score_with_artifact(artifact, chunks, *, positive_fraction=None):
    chunks = chunks or []
    if not chunks:
        return []
    if artifact.get("strategy") == CHALLENGER_STRATEGY:
        full = _challenger_rows(chunks)
        # CHALLENGER_FEATURE_NAMES deliberately begins with FEATURE_NAMES, so
        # the incumbent branches can reuse the same extraction pass.
        base = full[:, : len(FEATURE_NAMES)]
    else:
        base = _rows(chunks)
        full = None
    return score_rows_with_artifact(
        artifact,
        base,
        full,
        tie_keys=[_chunk_tie_key(chunk) for chunk in chunks],
        positive_fraction=positive_fraction,
    )


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk using the selected artifact."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return score_with_artifact(_model(), chunks)
    except Exception:
        return [0.5] * len(chunks)


def warmup_model():
    """Load and validate the artifact before the axon accepts its first query."""
    model = _model()
    return {
        "model_name": str(model.get("model_name") or "poker239-rankfuse-ens3"),
        "model_version": str(model.get("model_version") or "1"),
        "strategy": str(model.get("strategy") or "legacy_rank_fusion"),
    }


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        if m.get("strategy") == CHALLENGER_STRATEGY:
            # A one-row request has no useful percentile context.  Use the raw
            # incumbent branches; validators call score_batch with a request.
            X = _rows([chunk])
            weights = np.asarray(m["branch_weights"][:2], dtype=float)
            raw = (
                weights[0] * m["stack"].predict_proba(X)[:, 1]
                + weights[1] * m["mlp"].predict_proba(X)[:, 1]
            ) / max(float(weights.sum()), 1e-12)
            return round(float(raw[0]), 6)
        # No batch context for a lone chunk: return the calibrated member-mean prob.
        X = _rows([chunk])
        s = (m["weights"][0] * m["stack"].predict_proba(X)[:, 1]
             + m["weights"][1] * m["mono"].predict_proba(X)[:, 1]
             + m["weights"][2] * m["mlp"].predict_proba(X)[:, 1]) / sum(m["weights"])
        return round(float(s[0]), 6)
    except Exception:
        return 0.5
