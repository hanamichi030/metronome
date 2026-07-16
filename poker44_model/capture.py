"""Best-effort capture of live eval chunks + this miner's scores, for offline
analysis and future model upgrades.

Layout (deduped by chunk-content hash so a chunk repeated within a daily window
is stored once):

    captures/<model_name>/<YYYY-MM-DD>_<hash8>/chunks.json   # the received model input
    captures/<model_name>/<YYYY-MM-DD>_<hash8>/score.json    # this miner's scores + metadata

Notes:
  - Never raises into the miner — a capture failure must not break scoring.
  - Disabled with POKER44_CAPTURE=0.
  - The captures/ data dir is gitignored (runtime output, not code).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "captures")


def _enabled() -> bool:
    return os.getenv("POKER44_CAPTURE", "1").strip().lower() not in {"0", "false", "no", "off"}


def chunk_fingerprint(chunks) -> str:
    payload = json.dumps(chunks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_capture(chunks, scores, predictions, *, model_name="", model_version="",
                 impl_sha256="", uid=None, hotkey=""):
    """Persist one observed round. Returns the round dir, or None if skipped/failed."""
    if not _enabled() or not chunks:
        return None
    try:
        fp = chunk_fingerprint(chunks)
        day = time.strftime("%Y-%m-%d", time.gmtime())
        version_dir = (model_name or model_version or "model").replace("/", "_")
        round_dir = os.path.join(CAPTURE_DIR, version_dir, f"{day}_{fp[:8]}")
        os.makedirs(round_dir, exist_ok=True)

        chunks_path = os.path.join(round_dir, "chunks.json")
        if not os.path.exists(chunks_path):  # write the input once (dedupe)
            with open(chunks_path, "w") as fh:
                json.dump({
                    "chunk_hash": fp,
                    "n_chunks": len(chunks),
                    "hands_per_chunk": [len(c) for c in chunks],
                    "chunks": chunks,
                }, fh)

        score_path = os.path.join(round_dir, "score.json")
        prev = {}
        if os.path.exists(score_path):
            try:
                prev = json.load(open(score_path))
            except Exception:
                prev = {}
        record = {
            "chunk_hash": fp,
            "uid": uid,
            "hotkey": hotkey,
            "model_name": model_name,
            "model_version": model_version,
            "impl_sha256": impl_sha256,
            "n_chunks": len(chunks),
            "scores": list(scores) if scores is not None else None,
            "predictions": list(predictions) if predictions is not None else None,
            "seen": int(prev.get("seen", 0)) + 1,
            "first_seen_utc": prev.get("first_seen_utc", _now()),
            "last_seen_utc": _now(),
        }
        with open(score_path, "w") as fh:
            json.dump(record, fh, indent=2)
        return round_dir
    except Exception:
        return None
