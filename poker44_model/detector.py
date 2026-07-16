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
is averaged with fixed weights (0.35, 0.30, 0.35), so no member's OOD score-scale
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

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

try:  # keep any torch backend single-threaded (never deadlock batched predict)
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass

_MODEL = None


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
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for key in ("stack", "mono", "mlp"):
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


def _rows(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def _fused_rank(model, chunks):
    """Weighted average of each member's WITHIN-BATCH rank (the movable ordering)."""
    X = _rows(chunks)
    s1 = model["stack"].predict_proba(X)[:, 1]
    s2 = model["mono"].predict_proba(X)[:, 1]
    s3 = model["mlp"].predict_proba(X)[:, 1]
    w1, w2, w3 = model["weights"]
    fused = (w1 * _rank01(s1) + w2 * _rank01(s2) + w3 * _rank01(s3)) / (w1 + w2 + w3)
    return fused


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


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (rank-fused, reward-fit floating output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _fused_rank(m, chunks))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        # No batch context for a lone chunk: return the calibrated member-mean prob.
        X = _rows([chunk])
        s = (m["weights"][0] * m["stack"].predict_proba(X)[:, 1]
             + m["weights"][1] * m["mono"].predict_proba(X)[:, 1]
             + m["weights"][2] * m["mlp"].predict_proba(X)[:, 1]) / sum(m["weights"])
        return round(float(s[0]), 6)
    except Exception:
        return 0.5
