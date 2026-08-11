#!/bin/bash
set -euo pipefail

# Resolve project root from script location
SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd)"
PROJECT="$(cd -P "$SCRIPT_DIR/.." && pwd)"

CLI="$PROJECT/local_llm/Ollama.app/Contents/Resources/ollama"
HOME_DIR="$PROJECT/local_llm/runtime-home"
MODELS_DIR="$PROJECT/local_llm/models"
TMP_DIR="$PROJECT/local_llm/tmp"
PORT=11435
LSOF="/usr/sbin/lsof"

# ---- Pre-flight ----
[ -f "$CLI" ] && [ ! -L "$CLI" ] && [ -x "$CLI" ] || {
    echo "ERROR: CLI missing, is symlink, or not executable: $CLI" >&2
    exit 1
}
for d in "$HOME_DIR" "$MODELS_DIR" "$TMP_DIR"; do
    [ -d "$d" ] && [ ! -L "$d" ] || {
        echo "ERROR: directory missing or is symlink: $d" >&2
        exit 1
    }
done
[ -x "$LSOF" ] || {
    echo "ERROR: lsof not found at $LSOF" >&2
    exit 1
}

# ---- Port check ----
LSOF_ERR=""
LSOF_OUT="$(mktemp /tmp/ollama-lsof-stdout.XXXXXX)"
# Install trap before creating second temp file
cleanup_lsof() {
    rm -f -- "$LSOF_OUT"
    if [ -n "$LSOF_ERR" ]; then
        rm -f -- "$LSOF_ERR"
    fi
}
trap cleanup_lsof EXIT

LSOF_ERR="$(mktemp /tmp/ollama-lsof-stderr.XXXXXX)" || {
    echo "ERROR: cannot create temp file for lsof stderr" >&2
    exit 1
}

set +e
"$LSOF" -nP -a -iTCP:$PORT -sTCP:LISTEN -t > "$LSOF_OUT" 2> "$LSOF_ERR"
LSOF_RC=$?
set -e

if [ "$LSOF_RC" -eq 0 ]; then
    echo "ERROR: port $PORT is already in use:" >&2
    cat "$LSOF_OUT" >&2
    exit 1
elif [ "$LSOF_RC" -eq 1 ] \
    && [ ! -s "$LSOF_OUT" ] \
    && [ ! -s "$LSOF_ERR" ]; then
    :  # port is free
else
    echo "ERROR: cannot determine port $PORT status (lsof rc=$LSOF_RC)" >&2
    [ -s "$LSOF_ERR" ] && cat "$LSOF_ERR" >&2
    exit 1
fi

# Clean up temp files and cancel trap before exec
cleanup_lsof
trap - EXIT

# ---- Start ----
echo "Starting Ollama on 127.0.0.1:$PORT ..."
echo "  CLI:            $CLI"
echo "  HOME:           $HOME_DIR"
echo "  Models:         $MODELS_DIR"
echo "  Stop:           Ctrl+C"
echo ""

exec env \
    HOME="$HOME_DIR" \
    OLLAMA_MODELS="$MODELS_DIR" \
    TMPDIR="$TMP_DIR" \
    OLLAMA_HOST="127.0.0.1:$PORT" \
    OLLAMA_NO_CLOUD=1 \
    "$CLI" serve
