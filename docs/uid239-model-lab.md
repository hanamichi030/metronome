# UID239 model-improvement lab

The lab trains and tests challengers without replacing
`poker44_model/model.joblib`. It implements:

- legacy Stack/Mono/MLP weight ablations;
- 132 additional identifier-free coherence features;
- request-relative input-feature ranks;
- ExtraTrees and HistGradientBoosting rank-input branches;
- chronological model selection;
- locked future-date replay on balanced 40- and 100-group requests;
- promotion gates for mean reward, p10, worst date, bot recall, failures, and latency.

Generated datasets, artifacts, and reports are stored under `.uid239_lab/` and
are ignored by Git.

## 1. Build challengers

Use public releases through a cutoff. The newest three included dates are used
only for candidate selection; earlier dates fit the selection models. After a
configuration is selected, rank-input models are refit through the cutoff.

```bash
python -m tools.uid239_lab build \
  --data-dir /path/to/public/cache \
  --train-through-date 2026-07-16
```

The default output is `.uid239_lab/challenger_bundle.joblib`. Production is not
changed.

## 2. Run locked replay

Only dates strictly after the build cutoff are eligible:

```bash
python -m tools.uid239_lab evaluate \
  --bundle .uid239_lab/challenger_bundle.joblib \
  --data-dir /path/to/future/public/cache \
  --cutoff-date 2026-07-16 \
  --through-date 2026-07-19 \
  --promoted-artifact .uid239_lab/promoted-model.joblib
```

The evaluator compares 8 model configurations and scans positive fractions of
8%, 10%, 12.5%, and 15%. A candidate artifact is written only when every gate
passes. This still does not overwrite the production artifact. Because the
locked window compares multiple predeclared candidates, one newer untouched
release is still required before replacing production.

## 3. Inspect and promote deliberately

Review `.uid239_lab/evaluation-report.json`, then test the candidate directly:

```bash
POKER44_MODEL_PATH=.uid239_lab/promoted-model.joblib python - <<'PY'
from poker44_model.detector import _model
print(_model()["model_name"], _model().get("strategy"))
PY
```

The current miner always loads `poker44_model/model.joblib`; promotion therefore
requires a newer, single-candidate verification:

```bash
python -m tools.uid239_lab verify \
  --candidate .uid239_lab/promoted-model.joblib \
  --data-dir /path/to/newer/public/cache \
  --cutoff-date 2026-07-19
```

Only when `verification-report.json` says `production_ready: true` can the
guarded promotion command install it. The command verifies the candidate hash
and backs up the incumbent first:

```bash
python -m tools.uid239_lab promote \
  --verification-report .uid239_lab/verification-report.json \
  --candidate .uid239_lab/promoted-model.joblib
```

Commit the reviewed code and artifact, deploy, and restart the miner. Never
select a candidate using the locked dates and report those same dates as a
fresh out-of-sample test.
