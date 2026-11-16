#!/usr/bin/env bash
set -euo pipefail

# launchd runs with a minimal PATH (no Homebrew), so `uv` is not found -> exit 127.
# Prepend the common locations where uv lives so the job resolves it.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

uv run sb project-context --no-ingest --output-dir "$HOME/.claude/artjeck-inbox/project-context"
uv run sb overnight
# After ingesting, drop chunks whose source file has gone. `gc` exits 2 when it
# refuses (an unmounted share, or an implausible share of the corpus missing),
# which is a decision to report, not a failure that should abort the night.
uv run sb gc || true
uv run sb task-sync
# `health` exits 1 when a check fails. Under `set -e` that would skip the
# briefing, so the night carries on and ends with health's status instead:
# a failing check shows as the job's exit status rather than vanishing.
health_status=0
uv run sb health || health_status=$?
uv run sb morning
exit "$health_status"
