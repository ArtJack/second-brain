#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.secondbrain.overnight"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
HOUR="${SB_OVERNIGHT_HOUR:-3}"
MINUTE="${SB_OVERNIGHT_MINUTE:-15}"
LOG_DIR="$PROJECT_DIR/data/overnight/logs"
RUNNER="$PROJECT_DIR/deploy/run-nightly.sh"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>

  <key>WorkingDirectory</key>
  <string>$PROJECT_DIR</string>

  <key>ProgramArguments</key>
  <array>
    <string>$RUNNER</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>$HOUR</integer>
    <key>Minute</key>
    <integer>$MINUTE</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>$LOG_DIR/launchd.out.log</string>

  <key>StandardErrorPath</key>
  <string>$LOG_DIR/launchd.err.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"

echo "Installed $LABEL"
echo "Schedule: every day at $(printf '%02d:%02d' "$HOUR" "$MINUTE")"
echo "Config: $PROJECT_DIR/data/overnight/config.json"
echo "Reports: $PROJECT_DIR/data/overnight/reports"
