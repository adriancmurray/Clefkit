#!/bin/sh
# Stops the Workers AI LaunchAgent and removes its plist. Safe to run when it is not installed.
set -eu

LABEL=com.clefkit.workers-ai
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"
echo "removed $LABEL"
