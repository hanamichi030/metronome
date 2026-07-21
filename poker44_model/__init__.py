"""Participant-owned model package for the Poker44 miner — poker239-irunion.

Bot detector = an input-rank soft-vote ensemble over a 452-dim UNION
order-statistic behavioral feature surface (base_features.chunk_features +
features_v2 order-stats, magnitude columns dropped; see union_features.py).
Every feature is mapped to its within-served-window percentile rank BEFORE the
trees (the input-rank transform), then scored by an ExtraTrees + RandomForest +
HistGradientBoosting soft-vote (weights .45/.25/.30). A strictly-monotone,
tie-free decision layer lifts exactly the top ~10% of each served window over
0.5 (hard-zero-safe, bounded human FPR) while preserving the within-window rank
order that AP / recall@FPR are computed on. No isotonic calibration. Training
hands pass through the validator's prepare_hand_for_miner (train==serve);
inference does NOT re-sanitize. No capture-fitted domain adaptation and no
query-chunk fitting. See detector.py (inference), union_features.py /
features_v2.py / base_features.py (extraction), model.joblib (et/rf/hgb members,
vote weights, decision constants).
"""

from poker44_model.detector import score_chunk, score_batch  # noqa: F401

__all__ = ["score_batch", "score_chunk"]
