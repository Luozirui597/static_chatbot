#!/bin/bash
#
# macOS double-click entry point: starts the Teachable Agent demo.
#
# The launcher resolves the project root from this file's own location, so it
# works from any checkout path (spaces included) and never hardcodes a user
# name or an absolute path.

set -euo pipefail

SELF_DIR="$(cd -P "$(dirname "$0")" && pwd)"
SCRIPT="$SELF_DIR/scripts/start-demo.sh"

printf '\n=== Teachable Agent: starting the demo ===\n'
printf 'Project: %s\n\n' "$SELF_DIR"

if [ ! -f "$SCRIPT" ]; then
    printf 'ERROR: %s is missing.\n' "$SCRIPT" >&2
    printf 'Please re-check out the repository and try again.\n' >&2
    printf '\nPress Return to close this window.' >&2
    read -r _ || true
    exit 1
fi

if [ ! -x "$SCRIPT" ]; then
    printf 'ERROR: %s is not executable.\n' "$SCRIPT" >&2
    printf 'Fix it with: chmod +x "%s"\n' "$SCRIPT" >&2
    printf '\nPress Return to close this window.' >&2
    read -r _ || true
    exit 1
fi

# Run in the foreground so Ctrl+C and the service logs stay in this window.
status=0
bash "$SCRIPT" || status=$?

printf '\n=== Teachable Agent: launcher exited (status %s) ===\n' "$status"
if [ "$status" -ne 0 ]; then
    printf 'Check the messages above, then see README ("One-command start") for help.\n'
fi
printf 'Press Return to close this window.'
read -r _ || true
exit "$status"
