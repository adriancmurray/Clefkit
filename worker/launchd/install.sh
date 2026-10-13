#!/bin/sh
# Installs the local Workers AI worker as a LaunchAgent that starts at login and restarts if it exits.
set -eu

LABEL=com.clefkit.workers-ai
HERE=$(cd "$(dirname "$0")" && pwd)
WORKER_DIR=$(cd "$HERE/.." && pwd)
TEMPLATE="$HERE/$LABEL.plist.template"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/clefkit-workers-ai.log"

# The values go into the plist and a sh command line unescaped, so refuse characters that would need escaping.
check_value() {
    case "$2" in
        '') echo "install: $1 is empty" >&2; exit 1 ;;
        *'&'*|*'<'*|*'>'*|*'|'*|*'\'*|*'"'*|*"'"*|*'$'*|*'`'*)
            echo "install: $1 contains a character this script does not escape: $2" >&2; exit 1 ;;
    esac
}

WRANGLER=$(command -v wrangler || true)
[ -n "$WRANGLER" ] || { echo "install: wrangler not found on PATH" >&2; exit 1; }
[ -f "$TEMPLATE" ] || { echo "install: missing $TEMPLATE" >&2; exit 1; }
check_value "worker directory" "$WORKER_DIR"
check_value "wrangler path" "$WRANGLER"
check_value "log path" "$LOG"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
sed -e "s|__WORKER_DIR__|$WORKER_DIR|g" \
    -e "s|__WRANGLER__|$WRANGLER|g" \
    -e "s|__LOG__|$LOG|g" "$TEMPLATE" > "$PLIST"

# Not loaded yet on a first install, so a failed bootout is expected.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

i=0
while [ "$i" -lt 60 ]; do
    if curl -fsS --max-time 2 http://127.0.0.1:8788/health 2>/dev/null | grep -q '"ok":true'; then
        echo "ok: $LABEL answers on http://127.0.0.1:8788/health"
        exit 0
    fi
    i=$((i + 1))
    sleep 1
done
echo "install: no answer from /health after 60 s. Last 20 log lines:" >&2
tail -n 20 "$LOG" >&2 2>/dev/null || true
exit 1
