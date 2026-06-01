#!/usr/bin/env bash
# Install the second-brain MCP HTTP server as an always-on LaunchAgent on the Mac mini.
# Binds to the Tailscale IP (reachable from your other devices) with bearer-token auth.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$(command -v uv || true)"
[ -z "$UV" ] && { echo "error: uv not found in PATH" >&2; exit 1; }

PORT="${SB_MCP_PORT:-8848}"
TS_BIN="$(command -v tailscale || echo /Applications/Tailscale.app/Contents/MacOS/Tailscale)"
HOST="${SB_MCP_HOST:-$("$TS_BIN" ip -4 2>/dev/null | head -1)}"
HOST="${HOST:-127.0.0.1}"

# Ensure a token exists in the (git-ignored) .env
ENV_FILE="$REPO/.env"
touch "$ENV_FILE"
if ! grep -q '^SB_MCP_TOKEN=' "$ENV_FILE" 2>/dev/null; then
  printf '\nSB_MCP_TOKEN=%s\n' "$(openssl rand -hex 24)" >> "$ENV_FILE"
  echo "Generated SB_MCP_TOKEN and appended it to .env"
fi

PLIST_SRC="$REPO/deploy/com.secondbrain.mcp.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.secondbrain.mcp.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s#__UV__#${UV}#g" \
    -e "s#__REPO__#${REPO}#g" \
    -e "s#__HOST__#${HOST}#g" \
    -e "s#__PORT__#${PORT}#g" \
    -e "s#__HOME__#${HOME}#g" \
    "$PLIST_SRC" > "$PLIST_DST"

launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

echo "Installed com.secondbrain.mcp"
echo "  endpoint : http://${HOST}:${PORT}/mcp"
echo "  token    : grep SB_MCP_TOKEN ${ENV_FILE}"
echo "  logs     : ${HOME}/Library/Logs/secondbrain-mcp.log"
echo "  stop     : launchctl unload ${PLIST_DST}"
