#!/usr/bin/env bash
# Launch the Poker44 miner with a manifest identity that is ALWAYS consistent
# with the code being served. It:
#   1. loads your local config from miner.env,
#   2. pins POKER44_MODEL_REPO_COMMIT to the current git HEAD (no manual SHA),
#   3. warns if that commit is not yet pushed to your public fork,
#   4. hands off to the stock scripts/miner/run/run_miner.sh.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f miner.env ]; then
  echo "Error: miner.env not found. Run:  cp miner.env.example miner.env  and fill it in."
  exit 1
fi

# shellcheck disable=SC1091
source ./miner.env

# Always serve the exact commit that is checked out.
HEAD_SHA="$(git rev-parse HEAD)"
export POKER44_MODEL_REPO_COMMIT="$HEAD_SHA"

# Public verifiability: the served commit must exist on your public fork.
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if git fetch origin "$BRANCH" >/dev/null 2>&1 \
   && git merge-base --is-ancestor HEAD "origin/$BRANCH" 2>/dev/null; then
  echo "OK: HEAD $HEAD_SHA is published on origin/$BRANCH."
else
  echo "WARNING: HEAD $HEAD_SHA is NOT pushed to origin/$BRANCH."
  echo "         Run 'git push origin $BRANCH' so validators can verify your served code."
fi

# Sanity: identity must not point at the upstream reference repo.
case "${POKER44_MODEL_REPO_URL:-}" in
  ""|*"Poker44/Poker44-subnet"*)
    echo "WARNING: POKER44_MODEL_REPO_URL is empty or points at upstream. Manifest will be 'opaque'."
    ;;
esac

# Use `bash` so this works even though run_miner.sh is tracked non-executable (mode 644).
exec bash ./scripts/miner/run/run_miner.sh
