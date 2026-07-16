NEW MINER — rank-fused ensemble (same class as uid12/uid13, distinct seed)
Built 2026-07-16. Self-contained. Deploy on the OTHER VPS. Assign the UID / repo when you have them.

WHAT THIS IS
------------
A within-batch rank-fused ensemble of 3 decorrelated members (stacked GBDT
[LightGBM+XGBoost+RandomForest -> logistic OOF] | sign-stability-gated monotone-constrained
LightGBM | PCA(52)->MLP(72)) over 180 sanitization-invariant cross-hand behavioral features,
topped with the strictly-monotone reward-fit decision layer (no isotonic; FLOOR=0.10 so exactly
10 of 100 chunks cross 0.5). This is the WINNING class from the only clean live head-to-head we
have (R2): the ensemble class scored rank_block ~0.18 vs ~0.10 for a single-LGBM.

It is a FRESH SEED (SEED=61129), so its model.joblib sha256 (starts 212f5394) is DISTINCT from
every other miner in the fleet. That matters: identical model hashes across UIDs are a
disqualification (copy-DQ). Do NOT reuse another miner's model.joblib.

VERIFIED (real live captures): n=100, max 0.956, crossing exactly 10, distinct 95, ~2s/window,
no NaN. Cannot hard-zero by construction (10 chunks always cross 0.5).

PLACEHOLDERS YOU FILL IN (marked xxx) — nothing here assumes a UID or repo yet
-----------------------------------------------------------------------------
When you register the UID and create its PUBLIC GitHub repo, set in miner.env:
    export WALLET_NAME="xxx"
    export HOTKEY="xxx"
    export AXON_PORT="xxx"                     # a free port on that box
    export PM2_NAME="poker44_miner_uidxxx"
    export ALLOWED_VALIDATOR_HOTKEYS=""        # empty = allow all validators
    export POKER44_MODEL_REPO_URL="https://github.com/<youruser>/xxx"   # your OWN public fork
    export POKER44_MODEL_NAME="pokerxxx-ens2"  # any name distinct from other UIDs
    export POKER44_MODEL_VERSION="1"
    export POKER44_MODEL_OPEN_SOURCE="true"
    export POKER44_TORCH_THREADS="1"
    export POKER44_MODEL_TRAINING_DATA_STATEMENT="Within-batch rank-fused ensemble of 3
      decorrelated members (a stacked GBDT [LightGBM+XGBoost+RandomForest -> logistic OOF], a
      sign-stability-gated monotone-constrained LightGBM, and a PCA->MLP) over 180
      sanitization-invariant cross-hand behavioral features; members fused by averaging
      within-batch ranks, then a strictly-monotone reward-fit decision layer (no probability
      calibration) places a deterministic top-10 percent of each window above 0.5, so the served
      order is exactly the fused within-batch rank. Trained only on the public Poker44 released
      benchmark; no validator-only evaluation data or labels."
    export POKER44_MODEL_PRIVATE_DATA_ATTESTATION="No validator-only evaluation labels or private
      data are used. Trained only on public Poker44 benchmark labels."
Also set the "model_name" default in neurons/miner.py to match POKER44_MODEL_NAME.
NEVER commit miner.env (it holds wallet config; keep it gitignored).

INSTALL
-------
1. Put this repo on the box (it must become a clone of the Poker44 subnet fork you control, with
   this poker44_model/ on top). Copy poker44_model/{detector.py,features.py,__init__.py,
   capture.py,model.joblib} into the fork's poker44_model/. model.joblib sha256 starts 212f5394.
2. Ensure the fork's neurons/miner.py implementation_files list includes all 5 served files
   (detector.py, features.py, __init__.py, capture.py, model.joblib) plus miner.py itself.
3. Ensure requirements-model.txt has: lightgbm, xgboost, scikit-learn, torch, threadpoolctl.
   (torch is used single-threaded; threadpoolctl is not strictly needed by THIS model -- it has
   no HistGradientBoosting member -- but harmless to include.)
4. Fill in miner.env (above). git add -A && git commit && git push  (repo PUBLIC, open_source=true
   -- the validator clones it at repo_commit and checks the served code's sha256).
5. Start pinned to the pushed HEAD:
     set -a; . ./miner.env; set +a
     export PYTHONPATH=<repo root>
     export POKER44_MODEL_REPO_COMMIT=$(git rev-parse HEAD)
     pm2 restart <name> --update-env    (or `pm2 start neurons/miner.py ...` first time)
6. VERIFY: local HEAD == remote main == pinned commit, and served values:
     max > 0.5  AND  distinct ~= 100  AND  crossing == 10   (the old "max ~0.99" rule is DEAD)

TIMING (important, same as uid13)
---------------------------------
Composite = MEAN of the 5 rounds and resets each epoch (epochs run 12:00->12:00 UTC, 5-day cycle).
A mid-round model swap contaminates that round -- last epoch that cost uid12 a top-10 place.
So: bring this miner up BEFORE a round boundary (12:00 UTC), and once it is live, FREEZE it for
the whole epoch -- one model per UID, no mid-epoch changes.

NOTE: a brand-new UID has an immunity period (~5000 blocks / ~17h on netuid 126) during which it
is queried but not scored. Register it >=17h before you expect a clean scored round.
