#!/usr/bin/env bash
set -euo pipefail

# launchd runs with a minimal PATH (no Homebrew), so `uv` is not found -> exit 127.
# Prepend the common locations where uv lives so the job resolves it.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

uv run sb project-context --no-ingest
uv run sb overnight
uv run sb task-sync
uv run sb health
uv run sb morning
