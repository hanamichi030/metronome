#!/usr/bin/env python3
"""Train, replay, and gate UID239 model challengers without touching production.

The lab enforces three data boundaries:

* model fitting uses dates before the selection window;
* candidate weights are selected on the newest public dates up to the cutoff;
* promotion is decided only on dates strictly after that cutoff.

All raw public hands are projected through the validator's miner-visible view
before feature extraction.  Replay creates balanced 40- and 100-group requests,
calls the repository's current reward function once per request, and reports
mean, p10, worst-date, threshold safety, recall, and latency.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.parse
import urllib.request

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44_model.detector import (
    CHALLENGER_STRATEGY,
    branch_rank_matrix_with_artifact,
    columnwise_batch_rank,
    score_branch_ranks_with_artifact,
    score_rows_with_artifact,
    validate_artifact,
)
from poker44_model.features import (
    CHALLENGER_FEATURE_NAMES,
    FEATURE_NAMES,
    challenger_features,
)


API_BASE = "https://api.poker44.net/api/v1/benchmark"
LAB_SCHEMA = "poker126-uid239-lab-v2"


class LabError(RuntimeError):
    """Raised when an experiment would be incomplete or irreproducible."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise LabError(f"benchmark request failed: {url}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("success") is False:
        raise LabError(f"benchmark returned an invalid response: {url}")
    return payload


def discover_dates(api_base: str = API_BASE, timeout: float = 90.0) -> list[str]:
    payload = _get_json(f"{api_base.rstrip('/')}/releases?limit=100", timeout)
    releases: Any = payload.get("data", [])
    if isinstance(releases, dict):
        releases = releases.get("releases", [])
    if not isinstance(releases, list):
        raise LabError("benchmark releases response has no release list")
    return sorted(
        {
            str(row["sourceDate"])
            for row in releases
            if isinstance(row, dict) and row.get("sourceDate")
        }
    )


def fetch_date(
    source_date: str,
    *,
    cache_dir: Path,
    api_base: str = API_BASE,
    timeout: float = 90.0,
    offline: bool = False,
) -> Path:
    path = cache_dir / f"chunks_{source_date}.json"
    if path.exists():
        return path
    if offline:
        raise LabError(f"offline cache missing {source_date}: {path}")
    records: list[dict[str, Any]] = []
    cursor: str | None = None
    metadata: dict[str, Any] = {"sourceDate": source_date}
    while True:
        query: dict[str, Any] = {"sourceDate": source_date, "limit": 48}
        if cursor:
            query["cursor"] = cursor
        page = _get_json(
            f"{api_base.rstrip('/')}/chunks?{urllib.parse.urlencode(query)}", timeout
        ).get("data")
        if not isinstance(page, dict) or not isinstance(page.get("chunks"), list):
            raise LabError(f"invalid chunks response for {source_date}")
        records.extend(row for row in page["chunks"] if isinstance(row, dict))
        metadata.update({key: value for key, value in page.items() if key != "chunks"})
        cursor = str(page.get("nextCursor") or "").strip() or None
        if not cursor:
            break
    metadata["nextCursor"] = None
    metadata["chunks"] = records
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def release_paths(data_dirs: Iterable[Path]) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for directory in data_dirs
            for path in Path(directory).glob("chunks_*.json")
            if path.is_file()
        }
    )


def _release_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        raise LabError(f"release file has no chunks list: {path}")
    return payload


def load_examples(
    paths: Sequence[Path],
    *,
    from_date: str | None = None,
    through_date: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, sanitize, and deduplicate public labeled groups."""
    unique: dict[str, dict[str, Any]] = {}
    conflicts = 0
    used_files: list[dict[str, Any]] = []
    for path in paths:
        payload = _release_payload(path)
        payload_date = str(payload.get("sourceDate") or "")
        accepted = 0
        for record_index, record in enumerate(payload["chunks"]):
            if not isinstance(record, dict):
                continue
            source_date = str(record.get("sourceDate") or payload_date)
            if not source_date:
                continue
            if from_date and source_date < from_date:
                continue
            if through_date and source_date > through_date:
                continue
            groups, labels = record.get("chunks"), record.get("groundTruth")
            if not isinstance(groups, list) or not isinstance(labels, list):
                continue
            if len(groups) != len(labels):
                raise LabError(f"group/label mismatch in {path}")
            request_id = str(record.get("chunkId") or f"{source_date}:{record_index}")
            for group_index, (group, raw_label) in enumerate(zip(groups, labels)):
                if not isinstance(group, list) or not group:
                    continue
                label = int(raw_label)
                if label not in (0, 1):
                    raise LabError(f"invalid label {raw_label!r} in {path}")
                visible = [prepare_hand_for_miner(hand) for hand in group]
                key = _canonical_hash(visible)
                previous = unique.get(key)
                if previous is not None:
                    conflicts += int(previous["label"] != label)
                    continue
                unique[key] = {
                    "source_date": source_date,
                    "request_id": request_id,
                    "request_position": group_index,
                    "label": label,
                    "hands": visible,
                    "group_hash": key,
                }
                accepted += 1
        if accepted:
            used_files.append(
                {"path": str(path), "sha256": _sha256(path), "accepted": accepted}
            )
    if conflicts:
        raise LabError(f"found {conflicts} miner-visible groups with conflicting labels")
    examples = sorted(
        unique.values(),
        key=lambda row: (row["source_date"], row["request_id"], row["request_position"]),
    )
    if not examples or {int(row["label"]) for row in examples} != {0, 1}:
        raise LabError("eligible data must contain both human and bot groups")
    dates = sorted({str(row["source_date"]) for row in examples})
    metadata = {
        "files": used_files,
        "group_count": len(examples),
        "human_groups": sum(int(row["label"]) == 0 for row in examples),
        "bot_groups": sum(int(row["label"]) == 1 for row in examples),
        "dates": dates,
        "date_min": dates[0],
        "date_max": dates[-1],
        "projection": "poker44.validator.payload_view.prepare_hand_for_miner",
    }
    return examples, metadata


def feature_matrices(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full_rows: list[list[float]] = []
    for example in examples:
        full = challenger_features(example["hands"])
        full_rows.append(
            [float(full.get(name, 0.0)) for name in CHALLENGER_FEATURE_NAMES]
        )
    full = np.asarray(full_rows, dtype=np.float64)
    base = full[:, : len(FEATURE_NAMES)].copy()
    labels = np.asarray([int(row["label"]) for row in examples], dtype=np.int8)
    dates = np.asarray([str(row["source_date"]) for row in examples])
    keys = np.asarray([str(row["group_hash"]) for row in examples])
    if not np.isfinite(base).all() or not np.isfinite(full).all():
        raise LabError("feature extraction produced NaN or infinity")
    return base, full, labels, dates, keys


def grouped_feature_ranks(matrix: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    group_values = np.asarray(groups, dtype=str)
    output = np.empty_like(values)
    for group in sorted(set(group_values.tolist())):
        indices = np.flatnonzero(group_values == group)
        output[indices] = columnwise_batch_rank(values[indices])
    return output


def _balanced_windows(
    labels: np.ndarray, size: int, repeats: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not positive.size or not negative.size:
        raise LabError("each replay date must contain both labels")
    left = size // 2
    output: list[np.ndarray] = []
    for _ in range(max(1, repeats)):
        pos = rng.choice(positive, left, replace=positive.size < left)
        neg = rng.choice(negative, size - left, replace=negative.size < size - left)
        indices = np.concatenate((pos, neg))
        rng.shuffle(indices)
        output.append(indices)
    return output


def fit_rank_models(
    full: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    seed: int,
    trees: int,
    boosting_iterations: int,
) -> tuple[Any, Any]:
    ranked = grouped_feature_ranks(full, dates)
    extra = ExtraTreesClassifier(
        n_estimators=trees,
        max_depth=9,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    ).fit(ranked, labels)
    counts = np.bincount(labels.astype(int), minlength=2)
    sample_weight = np.asarray(
        [len(labels) / max(2 * counts[int(label)], 1) for label in labels], dtype=float
    )
    hist = HistGradientBoostingClassifier(
        max_iter=boosting_iterations,
        learning_rate=0.04,
        max_depth=7,
        min_samples_leaf=5,
        l2_regularization=1.5,
        random_state=seed + 17,
    ).fit(ranked, labels, sample_weight=sample_weight)
    return extra, hist


def candidate_configs(incumbent: Mapping[str, Any]) -> list[dict[str, Any]]:
    current = [float(value) for value in incumbent["weights"]]
    return [
        {"name": "legacy-current", "strategy": "legacy", "weights": current},
        {"name": "legacy-stack-only", "strategy": "legacy", "weights": [1.0, 0.0, 0.0]},
        {"name": "legacy-no-mono", "strategy": "legacy", "weights": [0.55, 0.0, 0.45]},
        {"name": "legacy-light-mono", "strategy": "legacy", "weights": [0.55, 0.10, 0.35]},
        {
            "name": "rank-balanced",
            "strategy": CHALLENGER_STRATEGY,
            "branch_weights": [0.30, 0.20, 0.35, 0.15],
        },
        {
            "name": "rank-heavy",
            "strategy": CHALLENGER_STRATEGY,
            "branch_weights": [0.25, 0.15, 0.40, 0.20],
        },
        {
            "name": "rank-tree",
            "strategy": CHALLENGER_STRATEGY,
            "branch_weights": [0.30, 0.00, 0.45, 0.25],
        },
        {
            "name": "raw-stack-rank",
            "strategy": CHALLENGER_STRATEGY,
            "branch_weights": [0.45, 0.00, 0.40, 0.15],
        },
    ]


def artifact_for_config(
    incumbent: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    rank_extra: Any,
    rank_hist: Any,
    positive_fraction: float,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = copy.copy(dict(incumbent))
    artifact["FLOOR"] = float(positive_fraction)
    if config["strategy"] == "legacy":
        if config["name"] == "legacy-current":
            artifact.pop("strategy", None)
        else:
            artifact["strategy"] = f"legacy_rank_fusion_v2:{config['name']}"
            artifact["model_name"] = f"poker239-{config['name']}"
            artifact["model_version"] = "2"
        artifact["weights"] = [float(value) for value in config["weights"]]
    else:
        artifact.update(
            {
                "strategy": CHALLENGER_STRATEGY,
                "model_name": "poker239-rank-coherence-v2",
                "model_version": "2",
                "rank_extra": rank_extra,
                "rank_hist": rank_hist,
                "branch_names": ["raw_stack", "raw_mlp", "rank_extra", "rank_hist"],
                "branch_weights": [float(value) for value in config["branch_weights"]],
                "challenger_feature_names": list(CHALLENGER_FEATURE_NAMES),
                "training_metadata": dict(metadata or {}),
            }
        )
    return validate_artifact(artifact)


def _percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile)) if values else 0.0


def evaluate_artifact(
    artifact: Mapping[str, Any],
    base: np.ndarray,
    full: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    keys: np.ndarray,
    *,
    window_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    per_date: list[dict[str, Any]] = []
    for date_index, source_date in enumerate(sorted(set(dates.tolist()))):
        date_indices = np.flatnonzero(dates == source_date)
        date_requests: list[dict[str, Any]] = []
        for size in window_sizes:
            local_windows = _balanced_windows(
                labels[date_indices], size, repeats, seed + date_index * 1000 + size
            )
            for local in local_windows:
                selected = date_indices[local]
                started = time.perf_counter()
                scores = score_rows_with_artifact(
                    artifact,
                    base[selected],
                    full[selected]
                    if artifact.get("strategy") == CHALLENGER_STRATEGY
                    else None,
                    tie_keys=keys[selected].tolist(),
                )
                latency = time.perf_counter() - started
                value, metrics = reward(
                    np.asarray(scores, dtype=float), labels[selected].astype(int)
                )
                row = {
                    "source_date": source_date,
                    "window_size": int(size),
                    "reward": float(value),
                    "latency_seconds": float(latency),
                    **{key: float(value) for key, value in metrics.items()},
                }
                requests.append(row)
                date_requests.append(row)
        per_date.append(
            {
                "source_date": source_date,
                "mean_reward": statistics.fmean(row["reward"] for row in date_requests),
                "p10_reward": _percentile([row["reward"] for row in date_requests], 0.10),
            }
        )
    rewards = [row["reward"] for row in requests]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return {
        "request_count": len(requests),
        "mean_reward": float(mean),
        "reward_std": float(std),
        "robust_objective": float(mean - 0.5 * std),
        "p10_reward": _percentile(rewards, 0.10),
        "worst_date_reward": min(row["mean_reward"] for row in per_date),
        "zero_reward_windows": sum(row["reward"] <= 0.0 for row in requests),
        "mean_ap": statistics.fmean(row["ap_score"] for row in requests),
        "mean_bot_recall": statistics.fmean(row["bot_recall"] for row in requests),
        "mean_hard_fpr": statistics.fmean(row["hard_fpr"] for row in requests),
        "unsafe_windows": sum(row["human_safety_penalty"] < 1.0 for row in requests),
        "mean_latency_seconds": statistics.fmean(row["latency_seconds"] for row in requests),
        "max_latency_seconds": max(row["latency_seconds"] for row in requests),
        "per_date": per_date,
    }


def _summarize_requests(requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_date: list[dict[str, Any]] = []
    for source_date in sorted({str(row["source_date"]) for row in requests}):
        selected = [row for row in requests if row["source_date"] == source_date]
        per_date.append(
            {
                "source_date": source_date,
                "mean_reward": statistics.fmean(float(row["reward"]) for row in selected),
                "p10_reward": _percentile(
                    [float(row["reward"]) for row in selected], 0.10
                ),
            }
        )
    rewards = [float(row["reward"]) for row in requests]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return {
        "request_count": len(requests),
        "mean_reward": float(mean),
        "reward_std": float(std),
        "robust_objective": float(mean - 0.5 * std),
        "p10_reward": _percentile(rewards, 0.10),
        "worst_date_reward": min(row["mean_reward"] for row in per_date),
        "zero_reward_windows": sum(float(row["reward"]) <= 0.0 for row in requests),
        "mean_ap": statistics.fmean(float(row["ap_score"]) for row in requests),
        "mean_bot_recall": statistics.fmean(
            float(row["bot_recall"]) for row in requests
        ),
        "mean_hard_fpr": statistics.fmean(float(row["hard_fpr"]) for row in requests),
        "unsafe_windows": sum(
            float(row["human_safety_penalty"]) < 1.0 for row in requests
        ),
        "mean_latency_seconds": statistics.fmean(
            float(row["latency_seconds"]) for row in requests
        ),
        "max_latency_seconds": max(float(row["latency_seconds"]) for row in requests),
        "per_date": per_date,
    }


def evaluate_grid(
    incumbent: Mapping[str, Any],
    rank_extra: Any,
    rank_hist: Any,
    configs: Sequence[Mapping[str, Any]],
    fractions: Sequence[float],
    base: np.ndarray,
    full: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    keys: np.ndarray,
    *,
    window_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate the full grid with only two expensive model passes per window."""
    artifacts: dict[tuple[str, float], dict[str, Any]] = {}
    requests: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for config in configs:
        for fraction in fractions:
            key = (str(config["name"]), float(fraction))
            artifacts[key] = artifact_for_config(
                incumbent,
                config,
                rank_extra=rank_extra,
                rank_hist=rank_hist,
                positive_fraction=fraction,
            )
            requests[key] = []

    legacy_prototype = artifacts[("legacy-current", float(fractions[0]))]
    challenger_config = next(
        config for config in configs if config["strategy"] == CHALLENGER_STRATEGY
    )
    challenger_prototype = artifacts[
        (str(challenger_config["name"]), float(fractions[0]))
    ]
    for date_index, source_date in enumerate(sorted(set(dates.tolist()))):
        date_indices = np.flatnonzero(dates == source_date)
        for size in window_sizes:
            windows = _balanced_windows(
                labels[date_indices], size, repeats, seed + date_index * 1000 + size
            )
            for local in windows:
                selected = date_indices[local]
                started = time.perf_counter()
                legacy_branches = branch_rank_matrix_with_artifact(
                    legacy_prototype,
                    base[selected],
                    tie_keys=keys[selected].tolist(),
                )
                legacy_latency = time.perf_counter() - started
                started = time.perf_counter()
                challenger_branches = branch_rank_matrix_with_artifact(
                    challenger_prototype,
                    base[selected],
                    full[selected],
                    tie_keys=keys[selected].tolist(),
                )
                challenger_latency = time.perf_counter() - started
                for config in configs:
                    strategy = config["strategy"]
                    branch_values = (
                        challenger_branches
                        if strategy == CHALLENGER_STRATEGY
                        else legacy_branches
                    )
                    inference_latency = (
                        challenger_latency
                        if strategy == CHALLENGER_STRATEGY
                        else legacy_latency
                    )
                    for fraction in fractions:
                        result_key = (str(config["name"]), float(fraction))
                        artifact = artifacts[result_key]
                        fusion_started = time.perf_counter()
                        scores = score_branch_ranks_with_artifact(
                            artifact,
                            branch_values,
                            tie_keys=keys[selected].tolist(),
                            positive_fraction=fraction,
                        )
                        latency = inference_latency + (time.perf_counter() - fusion_started)
                        value, metrics = reward(
                            np.asarray(scores, dtype=float), labels[selected].astype(int)
                        )
                        requests[result_key].append(
                            {
                                "source_date": source_date,
                                "window_size": int(size),
                                "reward": float(value),
                                "latency_seconds": float(latency),
                                **{
                                    metric: float(metric_value)
                                    for metric, metric_value in metrics.items()
                                },
                            }
                        )
    output: list[dict[str, Any]] = []
    for config in configs:
        for fraction in fractions:
            result_key = (str(config["name"]), float(fraction))
            output.append(
                {
                    "name": config["name"],
                    "positive_fraction": float(fraction),
                    "config": dict(config),
                    "metrics": _summarize_requests(requests[result_key]),
                }
            )
    return output


def promotion_decision(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    minimum_delta: float,
    worst_date_tolerance: float,
    latency_limit: float,
) -> dict[str, Any]:
    checks = {
        "mean_delta": float(challenger["mean_reward"])
        >= float(baseline["mean_reward"]) + minimum_delta,
        "p10_not_lower": float(challenger["p10_reward"]) >= float(baseline["p10_reward"]),
        "worst_date": float(challenger["worst_date_reward"])
        >= float(baseline["worst_date_reward"]) - worst_date_tolerance,
        "zero_reward_windows": int(challenger["zero_reward_windows"]) == 0,
        "threshold_safe": (
            int(challenger.get("unsafe_windows", 0)) == 0
            and float(challenger.get("mean_hard_fpr", 1.0)) <= 0.10
        ),
        "recall_not_lower": float(challenger["mean_bot_recall"])
        >= float(baseline["mean_bot_recall"]),
        "latency": float(challenger["max_latency_seconds"]) <= latency_limit,
    }
    return {"promoted": all(checks.values()), "checks": checks}


def _resolve_data(args: argparse.Namespace, *, through_date: str | None) -> list[Path]:
    directories = [Path(value) for value in args.data_dir]
    if args.fetch:
        available = discover_dates(args.api_base, args.request_timeout)
        for source_date in available:
            if args.from_date and source_date < args.from_date:
                continue
            if through_date and source_date > through_date:
                continue
            fetch_date(
                source_date,
                cache_dir=args.cache_dir,
                api_base=args.api_base,
                timeout=args.request_timeout,
                offline=args.offline,
            )
        directories.append(args.cache_dir)
    paths = release_paths(directories)
    if not paths:
        raise LabError("no benchmark files found; provide --data-dir or --fetch")
    return paths


def build_command(args: argparse.Namespace) -> int:
    incumbent = validate_artifact(joblib.load(args.incumbent))
    paths = _resolve_data(args, through_date=args.train_through_date)
    examples, data_metadata = load_examples(
        paths, from_date=args.from_date, through_date=args.train_through_date
    )
    base, full, labels, dates, keys = feature_matrices(examples)
    unique_dates = sorted(set(dates.tolist()))
    if len(unique_dates) <= args.selection_days:
        raise LabError("not enough dates before cutoff for fit and selection windows")
    selection_dates = unique_dates[-args.selection_days :]
    fit_indices = np.flatnonzero(~np.isin(dates, selection_dates))
    selection_indices = np.flatnonzero(np.isin(dates, selection_dates))
    print(
        f"fit dates {unique_dates[0]}..{unique_dates[-args.selection_days-1]} "
        f"({len(fit_indices)} groups); selection dates {selection_dates}",
        flush=True,
    )
    selection_extra, selection_hist = fit_rank_models(
        full[fit_indices],
        labels[fit_indices],
        dates[fit_indices],
        seed=args.seed,
        trees=args.trees,
        boosting_iterations=args.boosting_iterations,
    )
    configs = candidate_configs(incumbent)
    grid = evaluate_grid(
        incumbent,
        selection_extra,
        selection_hist,
        configs,
        [args.positive_fraction],
        base[selection_indices],
        full[selection_indices],
        labels[selection_indices],
        dates[selection_indices],
        keys[selection_indices],
        window_sizes=args.window_sizes,
        repeats=args.repeats,
        seed=args.seed + 10000,
    )
    selection_results: list[dict[str, Any]] = []
    for row in grid:
        config, metrics = row["config"], row["metrics"]
        selection_results.append({"config": config, "metrics": metrics})
        print(
            f"{config['name']:20s} mean={metrics['mean_reward']:.4f} "
            f"p10={metrics['p10_reward']:.4f} worst={metrics['worst_date_reward']:.4f}",
            flush=True,
        )
    selected = max(
        selection_results,
        key=lambda row: (
            row["metrics"]["robust_objective"],
            row["metrics"]["p10_reward"],
            row["metrics"]["worst_date_reward"],
        ),
    )
    print(f"selected configuration: {selected['config']['name']}; fitting through cutoff")
    final_extra, final_hist = fit_rank_models(
        full,
        labels,
        dates,
        seed=args.seed,
        trees=args.trees,
        boosting_iterations=args.boosting_iterations,
    )
    bundle = {
        "schema_version": LAB_SCHEMA,
        "incumbent": incumbent,
        "rank_extra": final_extra,
        "rank_hist": final_hist,
        "configs": configs,
        "selected_config": selected["config"],
        "positive_fraction": args.positive_fraction,
        "training": {
            **data_metadata,
            "train_through_date": args.train_through_date,
            "selection_dates": selection_dates,
            "feature_count": len(CHALLENGER_FEATURE_NAMES),
            "seed": args.seed,
            "trees": args.trees,
            "boosting_iterations": args.boosting_iterations,
        },
        "selection_results": selection_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    report = {
        "schema_version": LAB_SCHEMA,
        "command": "build",
        "bundle": str(args.output.resolve()),
        "bundle_sha256": _sha256(args.output),
        "training": bundle["training"],
        "selected_config": selected["config"],
        "selection_results": selection_results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {args.output} and {args.report}")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    bundle = joblib.load(args.bundle)
    if bundle.get("schema_version") != LAB_SCHEMA:
        raise LabError("challenger bundle schema mismatch")
    paths = _resolve_data(args, through_date=args.through_date)
    examples, metadata = load_examples(
        paths,
        from_date=args.from_date or args.cutoff_date,
        through_date=args.through_date,
    )
    examples = [row for row in examples if row["source_date"] > args.cutoff_date]
    if not examples:
        raise LabError("locked replay needs dates strictly after --cutoff-date")
    base, full, labels, dates, keys = feature_matrices(examples)
    fractions = args.positive_fractions
    results = evaluate_grid(
        bundle["incumbent"],
        bundle["rank_extra"],
        bundle["rank_hist"],
        bundle["configs"],
        fractions,
        base,
        full,
        labels,
        dates,
        keys,
        window_sizes=args.window_sizes,
        repeats=args.repeats,
        seed=args.seed,
    )
    for row in results:
        config, fraction, metrics = (
            row["config"],
            row["positive_fraction"],
            row["metrics"],
        )
        print(
            f"{config['name']:20s} f={fraction:.3f} "
            f"mean={metrics['mean_reward']:.4f} p10={metrics['p10_reward']:.4f} "
            f"worst={metrics['worst_date_reward']:.4f} zero={metrics['zero_reward_windows']}",
            flush=True,
        )
    baseline = next(
        row
        for row in results
        if row["name"] == "legacy-current" and abs(row["positive_fraction"] - 0.10) < 1e-9
    )
    for row in results:
        row["promotion"] = (
            {"promoted": False, "checks": {}, "reason": "baseline"}
            if row is baseline
            else promotion_decision(
                baseline["metrics"],
                row["metrics"],
                minimum_delta=args.minimum_delta,
                worst_date_tolerance=args.worst_date_tolerance,
                latency_limit=args.latency_limit,
            )
        )
    promoted = sorted(
        [row for row in results if row["promotion"]["promoted"]],
        key=lambda row: (
            row["metrics"]["robust_objective"],
            row["metrics"]["p10_reward"],
            -abs(float(row["positive_fraction"]) - 0.10),
        ),
        reverse=True,
    )
    recommended = promoted[0] if promoted else None
    report = {
        "schema_version": LAB_SCHEMA,
        "command": "evaluate",
        "cutoff_date": args.cutoff_date,
        "locked_dates": sorted(set(dates.tolist())),
        "data": metadata,
        "baseline": {"name": baseline["name"], **baseline["metrics"]},
        "recommended": (
            {
                "name": recommended["name"],
                "positive_fraction": recommended["positive_fraction"],
            }
            if recommended
            else None
        ),
        "selection_warning": (
            "The locked dates compared multiple predeclared candidates. The recommended "
            "artifact needs at least one newer untouched release before production replacement."
        ),
        "fresh_holdout_required": bool(recommended),
        "candidate_comparison_count": len(results),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    if recommended and args.promoted_artifact:
        artifact = artifact_for_config(
            bundle["incumbent"],
            recommended["config"],
            rank_extra=bundle["rank_extra"],
            rank_hist=bundle["rank_hist"],
            positive_fraction=recommended["positive_fraction"],
            metadata=bundle["training"],
        )
        args.promoted_artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.promoted_artifact, compress=3)
        print(f"promotion candidate written to {args.promoted_artifact}")
    print(
        f"recommended: {recommended['name'] if recommended else 'none'}; wrote {args.report}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    """Verify one frozen candidate on releases unused by the model grid."""
    incumbent = validate_artifact(joblib.load(args.incumbent))
    candidate = validate_artifact(joblib.load(args.candidate))
    paths = _resolve_data(args, through_date=args.through_date)
    examples, metadata = load_examples(
        paths,
        from_date=args.from_date or args.cutoff_date,
        through_date=args.through_date,
    )
    examples = [row for row in examples if row["source_date"] > args.cutoff_date]
    if not examples:
        raise LabError("verification needs at least one release after --cutoff-date")
    base, full, labels, dates, keys = feature_matrices(examples)
    common = {
        "window_sizes": args.window_sizes,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    baseline = evaluate_artifact(
        incumbent, base, full, labels, dates, keys, **common
    )
    challenger = evaluate_artifact(
        candidate, base, full, labels, dates, keys, **common
    )
    decision = promotion_decision(
        baseline,
        challenger,
        minimum_delta=args.minimum_delta,
        worst_date_tolerance=args.worst_date_tolerance,
        latency_limit=args.latency_limit,
    )
    report = {
        "schema_version": LAB_SCHEMA,
        "command": "verify",
        "cutoff_date": args.cutoff_date,
        "verification_dates": sorted(set(dates.tolist())),
        "data": metadata,
        "incumbent": {
            "path": str(args.incumbent.resolve()),
            "sha256": _sha256(args.incumbent),
            "metrics": baseline,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": _sha256(args.candidate),
            "metrics": challenger,
        },
        "production_ready": bool(decision["promoted"]),
        "promotion": decision,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"baseline={baseline['mean_reward']:.4f} candidate={challenger['mean_reward']:.4f} "
        f"production_ready={decision['promoted']}"
    )
    print(f"wrote {args.report}")
    return 0


def promote_command(args: argparse.Namespace) -> int:
    """Install only an artifact that passed a single-candidate fresh verification."""
    report = json.loads(args.verification_report.read_text(encoding="utf-8"))
    if report.get("schema_version") != LAB_SCHEMA or report.get("command") != "verify":
        raise LabError("promotion requires a UID239 verify report")
    if report.get("production_ready") is not True:
        raise LabError("verification report did not mark the candidate production-ready")
    expected = str(report.get("candidate", {}).get("sha256") or "")
    actual = _sha256(args.candidate)
    if not expected or actual != expected:
        raise LabError("candidate SHA-256 does not match the verification report")
    validate_artifact(joblib.load(args.candidate))
    if args.destination.exists():
        backup = args.backup_dir / f"{args.destination.stem}-{_sha256(args.destination)[:12]}.joblib"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.destination, backup)
        print(f"backed up incumbent to {backup}")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.candidate, args.destination)
    if _sha256(args.destination) != actual:
        raise LabError("destination hash mismatch after promotion copy")
    print(f"promoted verified artifact to {args.destination}")
    return 0


def _csv_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("window sizes must be comma-separated integers >= 2")
    return values


def _csv_floats(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(not 0.0 < value < 0.5 for value in values):
        raise argparse.ArgumentTypeError("positive fractions must be between 0 and 0.5")
    return values


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", action="append", type=Path, default=[])
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".uid239_lab/cache"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--from-date")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Fit and select candidate families")
    _data_arguments(build)
    build.add_argument("--incumbent", type=Path, default=Path("poker44_model/model.joblib"))
    build.add_argument("--train-through-date", required=True)
    build.add_argument("--selection-days", type=int, default=3)
    build.add_argument("--window-sizes", type=_csv_ints, default=[40, 100])
    build.add_argument("--repeats", type=int, default=20)
    build.add_argument("--positive-fraction", type=float, default=0.10)
    build.add_argument("--seed", type=int, default=239126)
    build.add_argument("--trees", type=int, default=600)
    build.add_argument("--boosting-iterations", type=int, default=500)
    build.add_argument("--output", type=Path, default=Path(".uid239_lab/challenger_bundle.joblib"))
    build.add_argument("--report", type=Path, default=Path(".uid239_lab/build-report.json"))
    build.set_defaults(handler=build_command)

    evaluate = commands.add_parser("evaluate", help="Run locked future-date replay")
    _data_arguments(evaluate)
    evaluate.add_argument("--bundle", type=Path, default=Path(".uid239_lab/challenger_bundle.joblib"))
    evaluate.add_argument("--cutoff-date", required=True)
    evaluate.add_argument("--through-date")
    evaluate.add_argument("--window-sizes", type=_csv_ints, default=[40, 100])
    evaluate.add_argument("--repeats", type=int, default=20)
    evaluate.add_argument("--positive-fractions", type=_csv_floats, default=[0.08, 0.10, 0.125, 0.15])
    evaluate.add_argument("--seed", type=int, default=239127)
    evaluate.add_argument("--minimum-delta", type=float, default=0.01)
    evaluate.add_argument("--worst-date-tolerance", type=float, default=0.01)
    evaluate.add_argument("--latency-limit", type=float, default=15.0)
    evaluate.add_argument("--report", type=Path, default=Path(".uid239_lab/evaluation-report.json"))
    evaluate.add_argument("--promoted-artifact", type=Path)
    evaluate.set_defaults(handler=evaluate_command)

    verify = commands.add_parser(
        "verify", help="Verify one frozen candidate on a newer untouched release"
    )
    _data_arguments(verify)
    verify.add_argument("--incumbent", type=Path, default=Path("poker44_model/model.joblib"))
    verify.add_argument("--candidate", type=Path, required=True)
    verify.add_argument("--cutoff-date", required=True)
    verify.add_argument("--through-date")
    verify.add_argument("--window-sizes", type=_csv_ints, default=[40, 100])
    verify.add_argument("--repeats", type=int, default=20)
    verify.add_argument("--seed", type=int, default=239128)
    verify.add_argument("--minimum-delta", type=float, default=0.01)
    verify.add_argument("--worst-date-tolerance", type=float, default=0.01)
    verify.add_argument("--latency-limit", type=float, default=15.0)
    verify.add_argument("--report", type=Path, default=Path(".uid239_lab/verification-report.json"))
    verify.set_defaults(handler=verify_command)

    promote = commands.add_parser(
        "promote", help="Install an artifact that passed fresh single-candidate verification"
    )
    promote.add_argument("--verification-report", type=Path, required=True)
    promote.add_argument("--candidate", type=Path, required=True)
    promote.add_argument("--destination", type=Path, default=Path("poker44_model/model.joblib"))
    promote.add_argument("--backup-dir", type=Path, default=Path(".uid239_lab/backups"))
    promote.set_defaults(handler=promote_command, repeats=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats <= 0:
        raise LabError("--repeats must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LabError as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
