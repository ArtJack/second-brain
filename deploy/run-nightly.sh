#!/usr/bin/env bash
set -euo pipefail

# launchd runs with a minimal PATH (no Homebrew), so `uv` is not found -> exit 127.
# Prepend the common locations where uv lives so the job resolves it.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

uv run sb project-context --no-ingest
uv run sb overnight
# After ingesting, drop chunks whose source file has gone. `gc` exits 2 when it
# refuses (an unmounted share, or an implausible share of the corpus missing),
# which is a decision to report, not a failure that should abort the night.
uv run sb gc || true
uv run sb task-sync
uv run sb health
uv run sb morning
