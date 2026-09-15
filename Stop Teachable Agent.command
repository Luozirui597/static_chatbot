#!/bin/bash
#
# macOS double-click entry point: stops the services started by the demo
# launcher.  It only ever signals processes recorded by scripts/start-demo.sh.

set -euo pipefail

SELF_DIR="$(cd -P "$(dirname "$0")" && pwd)"
SCRIPT="$SELF_DIR/scripts/stop-demo.sh"

printf '\n=== Teachable Agent: stopping the demo ===\n'
printf 'Project: %s\n\n' "$SELF_DIR"

if [ ! -f "$SCRIPT" ]; then
    printf 'ERROR: %s is missing.\n' "$SCRIPT" >&2
    printf '\nPress Return to close this window.' >&2
    read -r _ || true
    exit 1
fi

status=0
bash "$SCRIPT" || status=$?

printf '\n=== Teachable Agent: stop finished (status %s) ===\n' "$status"
printf 'Press Return to close this window.'
read -r _ || true
exit "$status"
