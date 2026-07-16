# Poker44 miner — uid239 (`hanamichi030/metronome`)

This repo is my Poker44 miner. My model lives in **`poker44_model/`**.
The support team's official repo is the `upstream` remote; I never push to it.

## Run the miner

```bash
cp miner.env.example miner.env     # first time only: fill in wallet/hotkey
./run_p44_miner.sh                 # pins the served commit to git HEAD automatically
```

## When the official (upstream) repo is updated

One command — it fetches upstream, merges, runs tests, and pushes to my repo:

```bash
./sync.sh
./run_p44_miner.sh                 # restart so the miner serves the new code
```

If `sync.sh` reports a conflict (usually only `neurons/miner.py` or `requirements.txt`),
fix the files, then:

```bash
git add -A && git commit
./sync.sh                          # re-run to finish
```

## Build my model

Edit **`poker44_model/detector.py`** — `score_chunk(chunk)` returns one risk score
in `[0, 1]` per chunk (higher = more bot-like). Put extra dependencies in
`requirements-model.txt`, not in `requirements.txt`.

## Fresh clone on a new machine

A plain clone has **no push credential**, so set it before `./sync.sh` (which pushes):

```bash
git clone https://github.com/hanamichi030/metronome.git
cd metronome
# embed your token so pushes work non-interactively (or use SSH / a credential helper):
git remote set-url origin https://<YOUR_TOKEN>@github.com/hanamichi030/metronome.git
cp miner.env.example miner.env       # fill in wallet/hotkey
./sync.sh                            # auto-adds the upstream remote on first run
```

After a server **reboot**, relaunch with `./run_p44_miner.sh` — don't rely on PM2
auto-resurrect, which can replay a stale commit identity.

> Identity rule: `POKER44_MODEL_REPO_URL` must be **this** repo (already set in
> `miner.env.example`), never the `Poker44/Poker44-subnet` reference repo, and the
> repo must stay **public**.
