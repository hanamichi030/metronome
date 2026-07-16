# Manifest refresh 2026-07-16
Served model: poker239-rankfuse-ens3 — a within-batch rank-fused ensemble of 3
decorrelated members (stacked GBDT [LightGBM+XGBoost+RandomForest -> logistic OOF]
+ sign-stability-gated monotone LightGBM + PCA->MLPClassifier) over 180
sanitization-invariant behavioral features, strictly-monotone reward-fit decision
layer (isotonic removed), n_jobs=1. Repo: https://github.com/hanamichi030/metronome .
This commit requests a fresh backend manifest re-review for the uid239 slot.
