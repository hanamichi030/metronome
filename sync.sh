#!/usr/bin/env bash
# Sync THIS fork with the official Poker44 upstream and publish to your repo.
# Self-contained: run it from anywhere inside the repo.  Usage:  ./sync.sh
set -euo pipefail

cd "$(dirname "$0")"
UPSTREAM_URL="https://github.com/Poker44/Poker44-subnet.git"
UPSTREAM_BRANCH="main"                        # the official branch to track
BRANCH="$(git rev-parse --abbrev-ref HEAD)"   # your local branch (push target)

# Add the upstream remote on a fresh clone (fetch-only, push blocked).
if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "==> Adding 'upstream' remote ($UPSTREAM_URL)"
  git remote add upstream "$UPSTREAM_URL"
  git remote set-url --push upstream DISABLED_NO_PUSH
fi

# Pre-flight: confirm we can actually push to origin BEFORE merging, so we never
# end up merged-but-unpushable. A fresh clone of the public URL has no push
# credential — fail fast with guidance instead of a raw error after the merge.
if ! GIT_TERMINAL_PROMPT=0 git push --dry-run origin "$BRANCH" >/dev/null 2>&1; then
  echo "!! Can't push to 'origin' (no credentials?). Set them first, e.g.:"
  echo "   git remote set-url origin https://<YOUR_TOKEN>@github.com/Ares90125/<repo>.git"
  echo "   (or configure a credential helper / SSH), then re-run ./sync.sh"
  exit 1
fi

echo "==> Fetching upstream..."
git fetch upstream

echo "==> Incoming upstream commits:"
git log --oneline "HEAD..upstream/$UPSTREAM_BRANCH" || true

echo "==> Merging upstream/$UPSTREAM_BRANCH..."
if ! git merge --no-edit "upstream/$UPSTREAM_BRANCH"; then
  echo
  echo "!! Merge conflicts. Likely: neurons/miner.py, requirements.txt, scripts/miner/run/run_miner.sh"
  echo "   Your model is in poker44_model/, so conflicts here are usually small."
  echo "   Fix the files, then:  git add -A && git commit  &&  ./sync.sh"
  exit 1
fi

echo "==> Running tests..."
PYTHONPATH="$(pwd)" python3 -m unittest discover -s tests || {
  echo "!! Tests failed. Did you run 'pip install -e .' (and 'pip install -r requirements-model.txt')?"
  echo "   Fix the failure before publishing — sync did NOT push."
  exit 1
}

echo "==> Pushing to your repo (origin/$BRANCH)..."
git push origin "$BRANCH"

echo
echo "==> Done. HEAD is now $(git rev-parse --short HEAD)."
echo "    Restart the miner to serve it:  ./run_p44_miner.sh"
