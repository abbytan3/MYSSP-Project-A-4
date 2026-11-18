#!/usr/bin/env bash
# Paradev setup script. Runs once in each new workspace (git worktree) when it is created,
# so every workspace starts with the NDA check turned on and nothing secret inside it.
set -euo pipefail

# where this workspace lives (Paradev sets SUPERSET_WORKSPACE_PATH; otherwise the current folder)
WT="${SUPERSET_WORKSPACE_PATH:-$PWD}"
cd "$WT"
echo "== A4 workspace setup: ${SUPERSET_WORKSPACE_NAME:-$(basename "$WT")}"

# 1. Turn on the NDA check before every commit (shared by all worktrees of this repo)
git config core.hooksPath .githooks
echo "-- NDA check turned on (core.hooksPath = .githooks)"

# 2. Keep local-only files out of git in every worktree, even if .gitignore changes.
#    info/exclude is shared by all worktrees and is never committed.
EXCLUDE="$(git rev-parse --git-common-dir)/info/exclude"
mkdir -p "$(dirname "$EXCLUDE")"
for p in .env .modal.toml .nda_terms .venv; do
  grep -qxF "$p" "$EXCLUDE" 2>/dev/null || echo "$p" >> "$EXCLUDE"
done

# 3. Show the git identity, so the person can check commits will be credited to them
echo "-- git user.name:  $(git config user.name || echo 'NOT SET')"
echo "-- git user.email: $(git config user.email || echo 'NOT SET')"

# 4. Check Modal is installed (the token itself is never touched here)
if command -v modal >/dev/null 2>&1; then
  echo "-- modal found: $(modal --version 2>/dev/null || echo 'version unknown')"
else
  echo "!! modal not found. Follow docs/SETUP.md step 3."
fi

echo "== setup done. Start Claude Code and say who you are and your task ID."
