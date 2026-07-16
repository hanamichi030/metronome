"""Poker44 miner shim.

Keep this file thin. Your actual bot-detection logic lives in `poker44_model/`
so that upstream updates to this reference file merge cleanly. The validator
scores one risk value per chunk returned by `poker44_model.score_chunk`.
"""

# from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse
from poker44_model import score_batch
from poker44_model.capture import save_capture


class Miner(BaseMinerNeuron):
    """Poker44 miner that delegates scoring to the participant-owned model."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 Miner started")
        repo_root = Path(__file__).resolve().parents[1]
        # Hash EVERY file that backs the served model so implementation_sha256
        # actually reflects the running code. Add new model files here.
        implementation_files = [
            Path(__file__).resolve(),
            repo_root / "poker44_model" / "__init__.py",
            repo_root / "poker44_model" / "detector.py",
            repo_root / "poker44_model" / "features.py",
            repo_root / "poker44_model" / "model.joblib",
            repo_root / "poker44_model" / "capture.py",
        ]
        # Identity defaults below are overridden by POKER44_MODEL_* env vars
        # (see miner.env.example). repo_url is intentionally left blank so a
        # missing config fails loudly (opaque) instead of silently pointing at
        # the upstream reference repo.
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "model_name": "poker239-rankfuse-ens3",
                "model_version": "1",
                "framework": "lightgbm+xgboost+sklearn-ensemble",
                "license": "MIT",
                "repo_url": "",
                "notes": "Within-batch rank-fused ensemble (stacked GBDT + sign-stable monotone LGBM + PCA->MLP) over 180 sanitization-invariant behavioral features (poker44_model/); strictly-monotone reward-fit decision layer, deterministic top-10% crossing, no isotonic calibration.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Set POKER44_MODEL_TRAINING_DATA_STATEMENT to describe your training data."
                ),
                "training_data_sources": ["none"],
                "private_data_attestation": (
                    "Set POKER44_MODEL_PRIVATE_DATA_ATTESTATION. "
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        if not self.model_manifest.get("repo_url"):
            bt.logging.warning(
                "POKER44_MODEL_REPO_URL is not set — the served manifest will be 'opaque'. "
                "Set it (and POKER44_MODEL_REPO_COMMIT) before competing. See miner.env.example."
            )
        bt.logging.info(
            f"Miner prep docs available | miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one bot-risk score per chunk using the participant-owned model."""
        chunks = synapse.chunks or []
        scores = score_batch(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Scored {len(chunks)} chunks; predictions={synapse.predictions}")
        try:
            save_capture(
                chunks, scores, synapse.predictions,
                model_name=self.model_manifest.get("model_name", ""),
                model_version=self.model_manifest.get("model_version", ""),
                impl_sha256=self.model_manifest.get("implementation_sha256", ""),
                uid=getattr(self, "uid", None),
                hotkey=getattr(getattr(self, "wallet", None), "hotkey", None)
                and self.wallet.hotkey.ss58_address,
            )
        except Exception:
            pass
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
