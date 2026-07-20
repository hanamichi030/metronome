from __future__ import annotations

import copy

import numpy as np

from poker44_model.detector import (
    CHALLENGER_STRATEGY,
    columnwise_batch_rank,
    score_rows_with_artifact,
    validate_artifact,
)
from poker44_model.features import (
    CHALLENGER_FEATURE_NAMES,
    COHERENCE_FEATURE_NAMES,
    FEATURE_NAMES,
    coherence_features,
)
from tools.uid239_lab import promotion_decision


class _FirstColumnModel:
    def predict_proba(self, matrix):
        values = np.asarray(matrix, dtype=float)
        probability = np.clip(0.1 + 0.8 * values[:, 0], 0.0, 1.0)
        return np.column_stack((1.0 - probability, probability))


def _hand(action: str, amount: float, hand_id: str) -> dict:
    return {
        "hand_id": hand_id,
        "metadata": {"max_seats": 2, "hero_seat": 1, "button_seat": 2},
        "players": [{"seat": 1, "starting_stack": 2.0}],
        "streets": [],
        "actions": [
            {
                "street": "preflop",
                "actor_seat": 1,
                "action_type": action,
                "normalized_amount_bb": amount,
                "pot_before": 0.02,
                "pot_after": 0.04,
            }
        ],
        "outcome": {"winner": hand_id},
    }


def _artifact() -> dict:
    model = _FirstColumnModel()
    return {
        "strategy": CHALLENGER_STRATEGY,
        "stack": model,
        "mlp": model,
        "rank_extra": model,
        "rank_hist": model,
        "branch_weights": [0.25, 0.25, 0.25, 0.25],
        "challenger_feature_names": list(CHALLENGER_FEATURE_NAMES),
        "Q": 0.7,
        "MARGIN": 3.0,
        "TEMP": 1.0,
        "FLOOR": 0.10,
        "CAP": True,
        "EPS": 0.0001,
        "train_ref_logit": -0.18,
    }


def test_columnwise_rank_averages_ties() -> None:
    ranked = columnwise_batch_rank(np.asarray([[1.0, 2.0], [1.0, 1.0], [3.0, 1.0]]))
    assert np.allclose(ranked[:, 0], [0.25, 0.25, 1.0])
    assert np.allclose(ranked[:, 1], [1.0, 0.25, 0.25])


def test_coherence_features_ignore_hand_order_ids_and_outcomes() -> None:
    first = [_hand("call", 1.0, "a"), _hand("raise", 2.0, "b")]
    second = copy.deepcopy(list(reversed(first)))
    for index, hand in enumerate(second):
        hand["hand_id"] = f"changed-{index}"
        hand["outcome"] = {"different": True}
    left = coherence_features(first)
    right = coherence_features(second)
    assert len(COHERENCE_FEATURE_NAMES) == 132
    assert left == right


def test_challenger_artifact_scores_one_value_per_row() -> None:
    artifact = validate_artifact(_artifact())
    base = np.zeros((40, len(FEATURE_NAMES)), dtype=float)
    full = np.zeros((40, len(CHALLENGER_FEATURE_NAMES)), dtype=float)
    base[:, 0] = np.linspace(0.0, 1.0, 40)
    full[:, 0] = np.linspace(0.0, 1.0, 40)
    scores = score_rows_with_artifact(
        artifact, base, full, tie_keys=[f"key-{index}" for index in range(40)]
    )
    assert len(scores) == 40
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert len(set(scores)) == 40
    assert sum(score >= 0.5 for score in scores) == 4


def test_promotion_requires_every_gate() -> None:
    baseline = {
        "mean_reward": 0.50,
        "p10_reward": 0.40,
        "worst_date_reward": 0.45,
        "zero_reward_windows": 0,
        "unsafe_windows": 0,
        "mean_hard_fpr": 0.0,
        "mean_bot_recall": 0.50,
        "max_latency_seconds": 0.1,
    }
    challenger = {
        **baseline,
        "mean_reward": 0.52,
        "p10_reward": 0.41,
        "worst_date_reward": 0.46,
        "mean_bot_recall": 0.51,
    }
    assert promotion_decision(
        baseline,
        challenger,
        minimum_delta=0.01,
        worst_date_tolerance=0.01,
        latency_limit=15.0,
    )["promoted"]
    challenger["zero_reward_windows"] = 1
    assert not promotion_decision(
        baseline,
        challenger,
        minimum_delta=0.01,
        worst_date_tolerance=0.01,
        latency_limit=15.0,
    )["promoted"]
