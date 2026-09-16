#!/bin/bash
#
# One-command launcher for the Teachable Agent demo.
#
# Starts the project-local Ollama runtime (or reuses a healthy one that is
# already listening on 11435), starts FastAPI/uvicorn on 8000, waits until
# both are actually ready, then opens the browser.  The script stays in the
# foreground: a single Ctrl+C stops exactly the processes this run created.
#
# Safety model:
#   * a single-instance lock (atomic mkdir) keeps two launchers from starting
#     the same services; a stale lock is removed only when the lock owner's
#     PID is a trusted `gone` result.  An alive or unverifiable (unknown)
#     owner is never removed and fails closed;
#   * every PID record carries the run id and the process start fingerprint;
#     a launcher only ever deletes records whose RUN_ID matches its own;
#   * a PID is only signalled after the live command line AND the live
#     process start time match the record (guards against PID reuse);
#   * lsof results are interpreted strictly; anything but the standard
#     "no listener" answer fails closed;
#   * if a managed process survives SIGTERM+SIGKILL, its record is kept for
#     retry/manual inspection and the launcher exits non-zero.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (environment variables exist so the test suite can run the
# script against fakes; normal use needs none of them)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd)"
PROJECT="${TEACHABLE_PROJECT_ROOT:-$(cd -P "$SCRIPT_DIR/.." && pwd)}"

PYTHON_BIN="${TEACHABLE_PYTHON_BIN:-$PROJECT/.venv/bin/python}"
LINK_PYTHON_BIN="${TEACHABLE_LINK_PYTHON:-$PYTHON_BIN}"
OLLAMA_LAUNCHER="${TEACHABLE_OLLAMA_LAUNCHER:-$PROJECT/scripts/start-local-ollama.sh}"
LOCAL_OLLAMA_CLI="${TEACHABLE_LOCAL_OLLAMA_CLI:-$PROJECT/local_llm/Ollama.app/Contents/Resources/ollama}"
LOCAL_OLLAMA_HOME="${TEACHABLE_LOCAL_OLLAMA_HOME:-$PROJECT/local_llm/runtime-home}"
LOCAL_OLLAMA_MODELS="${TEACHABLE_LOCAL_OLLAMA_MODELS:-$PROJECT/local_llm/models}"
LOCAL_OLLAMA_TMP="${TEACHABLE_LOCAL_OLLAMA_TMP:-$PROJECT/local_llm/tmp}"

RUNTIME_DIR="${TEACHABLE_RUNTIME_DIR:-$PROJECT/local_llm/run}"
LOG_DIR="${TEACHABLE_LOG_DIR:-$PROJECT/local_llm/logs}"

OLLAMA_HOST="${TEACHABLE_OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${TEACHABLE_OLLAMA_PORT-11435}"
APP_HOST="${TEACHABLE_APP_HOST:-127.0.0.1}"
APP_PORT="${TEACHABLE_APP_PORT-8000}"
APP_MODULE="${TEACHABLE_APP_MODULE:-backend.main:app}"

LSOF_BIN="${TEACHABLE_LSOF_BIN:-/usr/sbin/lsof}"
CURL_BIN="${TEACHABLE_CURL_BIN:-curl}"
OPEN_BIN="${TEACHABLE_OPEN_BIN:-open}"
# Process inspection.  The default is the system ps; the environment
# override exists so the test suite can supply a fixture-backed stub
# on machines where spawning ps is not permitted.
PS_BIN="${TEACHABLE_PS_BIN:-ps}"
SLEEP_BIN="${TEACHABLE_SLEEP_BIN:-sleep}"
MKDIR_BIN="${TEACHABLE_MKDIR_BIN:-mkdir}"
DATE_BIN="${TEACHABLE_DATE_BIN:-date}"

LSTART_RETRIES="${TEACHABLE_LSTART_RETRIES-10}"
LSTART_INTERVAL="${TEACHABLE_LSTART_INTERVAL-0.05}"
SELF_LSTART_RETRIES="${TEACHABLE_SELF_LSTART_RETRIES-40}"
SELF_LSTART_INTERVAL="${TEACHABLE_SELF_LSTART_INTERVAL-0.05}"
READY_ATTEMPTS="${TEACHABLE_READY_ATTEMPTS-100}"
READY_INTERVAL="${TEACHABLE_READY_INTERVAL-0.3}"
OLLAMA_STOP_TIMEOUT="${TEACHABLE_OLLAMA_STOP_TIMEOUT-60}"
BACKEND_STOP_TIMEOUT="${TEACHABLE_BACKEND_STOP_TIMEOUT-30}"

OLLAMA_PID_FILE="$RUNTIME_DIR/ollama-demo.pid"
BACKEND_PID_FILE="$RUNTIME_DIR/backend-demo.pid"
OLLAMA_LOG="$LOG_DIR/ollama-demo.log"
BACKEND_LOG="$LOG_DIR/backend-demo.log"
LOCK_FILE="$RUNTIME_DIR/demo.lock"
CLEANUP_FAILED_FILE="$RUNTIME_DIR/cleanup-failed"
BOOTSTRAP_GUARD_DIR="$RUNTIME_DIR/bootstrap.guard"

OLLAMA_URL="http://$OLLAMA_HOST:$OLLAMA_PORT"
APP_URL="http://$APP_HOST:$APP_PORT"
APP_HEALTH_URL="$APP_URL/api/health"

OLLAMA_SERVICE="ollama"
BACKEND_SERVICE="backend"

# Expected process identity: PID + service kind + project root + a command
# signature that must match the live process before any signal is sent.
OLLAMA_CMD_SIGNATURE="$LOCAL_OLLAMA_CLI serve"
# Fixed logical backend argument tail.  The live macOS process may show a
# framework Python executable instead of .venv/bin/python, so live identity
# anchors on this exact tail while the record must still name the expected
# spawn command exactly.
BACKEND_LOGIC_ARGS="-m uvicorn $APP_MODULE --host $APP_HOST --port $APP_PORT"
BACKEND_CMD_SIGNATURE="$PYTHON_BIN $BACKEND_LOGIC_ARGS"

# This run's identity.  Never shared, never printed with secrets.
RUN_ID="$$-${RANDOM}-${RANDOM}-$("$DATE_BIN" +%s)"

# Processes this run actually created.  Empty means "do not touch".
OWN_OLLAMA_PID=""
OWN_BACKEND_PID=""
OLLAMA_RECORD_FP=""
BACKEND_RECORD_FP=""
BACKEND_LIVE_CMD=""
# 0 while the child is still an unrecorded, unreaped direct child of this
# shell; set to 1 as soon as a verified PID record exists for it.
OLLAMA_RECORDED=0
BACKEND_RECORDED=0
BACKEND_CLEANUP_FAILED=0
BACKEND_GUARD_ACTIVE=0
OLLAMA_GUARD_ACTIVE=0
BOOT_DETAIL=""
LOCK_FP_VALUE=""
LOCK_FP=""
LOCK_PID=""
LOCK_RUN_ID=""
LOCK_ROOT=""
EXTERNAL_OLLAMA=0
LOCK_HELD=0

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

say()  { printf '%s\n' "$*"; }
info() { printf '[demo] %s\n' "$*"; }
warn() { printf '[demo] WARNING: %s\n' "$*" >&2; }
fail() { printf '[demo] ERROR: %s\n' "$*" >&2; }

advise_logs() {
    printf '[demo]   Ollama log: %s\n' "$OLLAMA_LOG" >&2
    printf '[demo]   Backend log: %s\n' "$BACKEND_LOG" >&2
}

die() {
    fail "$1"
    shift || true
    for line in "$@"; do
        printf '[demo]   %s\n' "$line" >&2
    done
    exit 1
}

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

is_positive_integer() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$1" -gt 0 ] 2>/dev/null
}

# Strict positive number (integer or decimal): digits with at most one dot,
# at least one non-zero digit, and no sign, exponent, spaces or shell syntax.
# Configuration is checked with this before any arithmetic expansion.
is_positive_number() {
    local value="${1:-}" whole frac digits
    case "$value" in
        ''|*[!0-9.]*|.*|*.) return 1 ;;
        *.*.*) return 1 ;;
    esac
    whole="${value%%.*}"
    frac=""
    case "$value" in
        *.*) frac="${value#*.}" ;;
    esac
    digits="${whole}${frac}"
    case "$digits" in
        *[1-9]*) return 0 ;;
    esac
    return 1
}

is_valid_port() {
    local value="${1:-}"
    is_positive_integer "$value" || return 1
    if [ "$value" -lt 1 ] 2>/dev/null || [ "$value" -gt 65535 ] 2>/dev/null; then
        return 1
    fi
    return 0
}

# A single BSD/macOS ps state token: one documented state letter followed
# only by known state flag characters.  Empty, numeric, multi-token,
# whitespace-bearing or arbitrary output is malformed and must be unknown.
is_valid_ps_state() {
    local value="${1:-}" first rest suffix
    [ -n "$value" ] || return 1
    first="${value%"${value#?}"}"
    case "$first" in
        I|R|S|T|U|Z) : ;;
        *) return 1 ;;
    esac
    rest="${value#?}"
    while [ -n "$rest" ]; do
        suffix="${rest%"${rest#?}"}"
        case "$suffix" in
            +|A|E|L|N|S|s|V|W|X) rest="${rest#?}" ;;
            "<"|">") rest="${rest#?}" ;;
            *) return 1 ;;
        esac
    done
    return 0
}

# Process observation is deliberately three-valued, because an empty ps
# result can mean either "the PID does not exist" or "ps itself failed":
#
#   alive   - ps reported a usable non-zombie state
#   gone    - ps ran successfully and reported nothing for this PID
#             (the platform's documented "no such process" answer)
#   unknown - ps is missing/unusable, the PID is malformed, ps wrote a
#             diagnostic, ps produced an unusable answer, or the state is
#             a zombie that we cannot verify
#
# Only `gone` may ever justify deleting a PID record or a lock, and only
# `alive` may justify signalling a process.  `unknown` always fails closed.
# --- strict ps field reader -------------------------------------------------
#
# Single source of truth for every ps query.  A field is only trusted when
# ps exits 0, prints exactly one usable value on stdout, and writes nothing
# at all on stderr.  A ps diagnostic next to otherwise valid stdout makes
# the answer unusable (unknown), never "trusted identity data".
#
# Status is reported through globals so the probe can run in the caller's
# shell (never in a command substitution):
#   PS_FIELD_OK=1  the field was read cleanly; PS_FIELD_VALUE holds it
#   PS_FIELD_OK=2  ps explicitly reported no such process (rc=1, no output
#                  on stdout or stderr) -> the PID is definitively GONE
#   PS_FIELD_OK=0  unusable in any other way; PS_FIELD_DETAIL explains why
# PS_FIELD_VALUE is always empty unless PS_FIELD_OK is 1.
PS_FIELD_OK=0
PS_FIELD_VALUE=""
PS_FIELD_DETAIL=""
ps_field_raw() {
    local pid="$1" field="$2"
    local ps_prog out_file err_file rc out err first

    PS_FIELD_OK=0
    PS_FIELD_VALUE=""
    PS_FIELD_DETAIL=""

    if [ -z "$field" ]; then
        PS_FIELD_DETAIL="no ps field requested"
        return 0
    fi
    if is_positive_integer "$pid" 2>/dev/null; then
        :
    else
        PS_FIELD_DETAIL="PID is not a positive integer ('${pid:-<empty>}')"
        return 0
    fi

    ps_prog="$(command -v "$PS_BIN" 2>/dev/null || true)"
    if [ -z "$ps_prog" ] && [ -x "$PS_BIN" ]; then
        ps_prog="$PS_BIN"
    fi
    if [ -z "$ps_prog" ]; then
        PS_FIELD_DETAIL="ps is not available: $PS_BIN"
        return 0
    fi

    out_file="$(mktemp "${TMPDIR:-/tmp}/demo-ps-out.XXXXXX")" || {
        PS_FIELD_DETAIL="cannot create a temporary file for ps"
        return 0
    }
    err_file="$(mktemp "${TMPDIR:-/tmp}/demo-ps-err.XXXXXX")" || {
        rm -f -- "$out_file"
        PS_FIELD_DETAIL="cannot create a temporary file for ps"
        return 0
    }

    set +e
    LC_ALL=C "$ps_prog" -p "$pid" -o "$field" >"$out_file" 2>"$err_file"
    rc=$?
    set -e

    out="$(cat "$out_file" 2>/dev/null || true)"
    err="$(cat "$err_file" 2>/dev/null || true)"
    rm -f -- "$out_file" "$err_file"

    # ANY stderr output makes the answer unusable, even with rc=0 and a
    # perfectly valid-looking value on stdout.
    if [ -n "$(printf '%s' "$err" | tr -d '[:space:]')" ]; then
        first="$(printf '%s' "$err" | head -n 1 | tr -d '\r')"
        PS_FIELD_DETAIL="ps wrote a diagnostic for -o $field: $first"
        return 0
    fi

    out="$(printf '%s' "$out" | tr '\n' ' ' | tr -s '[:space:]' ' ' \
        | sed -e 's/^ //' -e 's/ $//')"

    if [ "$rc" -eq 1 ] && [ -z "$out" ]; then
        PS_FIELD_OK=2
        PS_FIELD_DETAIL="ps reported no such process"
        return 0
    fi
    if [ "$rc" -ne 0 ]; then
        PS_FIELD_DETAIL="ps exit $rc for -o $field"
        return 0
    fi
    if [ -z "$out" ]; then
        PS_FIELD_DETAIL="ps exit 0 with empty output for -o $field"
        return 0
    fi

    PS_FIELD_OK=1
    PS_FIELD_VALUE="$out"
    return 0
}

# Always returns 0; the verdict is in PS_FIELD_OK / PS_FIELD_VALUE and the
# diagnostic in PS_FIELD_DETAIL.  Callers must NOT wrap this in a command
# substitution, which would run the probe (and its status assignments) in a
# subshell where the caller could never see them.
ps_field_into() {
    ps_field_raw "$1" "$2"
    return 0
}

# Non-subshell state probe.  Fills PROC_VIEW / PROC_VIEW_ALIVE /
# PROC_STATE_VALUE / PROC_VIEW_DETAIL in the CALLER's shell.
#
#   alive   - rc=0, one non-empty non-zombie state, stderr empty
#   gone    - rc=1, empty stdout, empty stderr
#   unknown - everything else: any stderr, unusual exit code, empty or
#             malformed output, unusable PID, or ps unavailable
proc_view_into() {
    local pid="$1" raw

    PROC_VIEW="unknown"
    PROC_VIEW_ALIVE=0
    PROC_STATE_VALUE=""
    PROC_VIEW_DETAIL=""

    ps_field_into "$pid" state= || true
    if [ "$PS_FIELD_OK" -eq 2 ]; then
        # ps confirmed there is no such process.
        PROC_VIEW="gone"
        return 0
    fi
    if [ "$PS_FIELD_OK" -ne 1 ]; then
        PROC_VIEW_DETAIL="$PS_FIELD_DETAIL"
        return 0
    fi

    raw="$PS_FIELD_VALUE"
    if is_valid_ps_state "$raw"; then
        case "$raw" in
            Z*)
                PROC_STATE_VALUE="$raw"
                PROC_VIEW_DETAIL="process $pid is a zombie"
                return 0
                ;;
        esac
        PROC_STATE_VALUE="$raw"
        PROC_VIEW="alive"
        PROC_VIEW_ALIVE=1
        return 0
    fi

    # Arbitrary, empty, numeric, multi-token or structurally odd output is
    # never a live process: it is an unusable probe.
    PROC_VIEW_DETAIL="ps reported a malformed state for PID $pid: '$raw'"
    return 0
}

# Value-returning variant for callers that only need the verdict.
proc_view() {
    proc_view_into "$1"
    printf '%s' "$PROC_VIEW"
    return 0
}

# True only for a confirmed live process; `gone` and `unknown` are false.
_is_alive() {
    proc_view_into "$1"
    [ "$PROC_VIEW_ALIVE" -eq 1 ]
}

# True only when ps explicitly confirmed that the PID does not exist.
_is_gone() {
    proc_view_into "$1"
    [ "$PROC_VIEW" = "gone" ]
}

# Collapse newlines and repeated spaces in a command line.  Takes the text
# as an argument on purpose: piping into a function would leave $1 unset and
# abort the whole script under `set -u`.
normalize_cmd() {
    printf '%s' "${1:-}" | tr '\n' ' ' | tr -s '[:space:]' ' ' | sed -e 's/^ //' -e 's/ $//'
}

# Full command line of a PID.  Returns an empty string unless ps produced a
# clean, stderr-free answer, so a diagnostic can never be mistaken for a
# trusted identity.
read_cmdline() {
    ps_field_into "$1" command= || true
    if [ "$PS_FIELD_OK" -ne 1 ]; then
        return 0
    fi
    normalize_cmd "$PS_FIELD_VALUE"
    return 0
}

# Raw state letter for a PID; empty unless the probe was clean and usable.
proc_state() {
    proc_view_into "$1" || true
    printf '%s' "$PROC_STATE_VALUE"
    return 0
}

# Process start time ("lstart"), normalized.  Empty unless ps produced a
# clean, stderr-free answer.  Never returns non-zero: an unreadable
# fingerprint is reported as an empty string, not as a status that would
# abort the caller.
read_lstart() {
    ps_field_into "$1" lstart= || true
    if [ "$PS_FIELD_OK" -ne 1 ]; then
        return 0
    fi
    normalize_cmd "$PS_FIELD_VALUE"
    return 0
}

# True only when cmd is an absolute executable followed by exactly the fixed
# backend logical argument tail.  A path containing spaces is accepted only
# when it is exactly the configured spawn executable, so a different command
# that merely embeds the tail is rejected.
# True only for the configured spawn executable, a path that is the same
# file, or the Python.app executable derived from the configured Python's
# own canonical framework/version root.  Arbitrary path suffixes, basenames,
# other framework versions and symlinks to unrelated files are rejected.
backend_live_exec_allowed() {
    local prefix="$1" expected="$2"
    "$LINK_PYTHON_BIN" - "$prefix" "$expected" <<'PY' 2>/dev/null
import os
import re
import stat
import sys

live = sys.argv[1]
expected = sys.argv[2]


def usable(path):
    try:
        info = os.stat(path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and os.access(path, os.X_OK)


if not live or not expected:
    sys.exit(1)
if not usable(expected):
    sys.exit(1)

expected_canon = os.path.realpath(expected)
live_canon = os.path.realpath(live)
if not usable(live):
    sys.exit(1)

if live == expected:
    sys.exit(0)

try:
    if os.path.samefile(expected, live):
        sys.exit(0)
except OSError:
    pass

if live_canon == expected_canon:
    sys.exit(0)

match = re.fullmatch(
    r"(?P<root>.*/Python[.]framework/Versions/[^/]+)/bin/python[^/]*",
    expected_canon,
)
if match is None:
    sys.exit(1)
derived = (
    match.group("root")
    + "/Resources/Python.app/Contents/MacOS/Python"
)
if os.path.realpath(derived) != derived:
    sys.exit(1)
if live_canon != derived:
    sys.exit(1)
if not usable(derived):
    sys.exit(1)
sys.exit(0)
PY
}

backend_live_cmd_shape_ok() {
    local cmd="$1" logic="$2" expected_exec="$3" prefix
    case "$cmd" in
        *" $logic") : ;;
        *) return 1 ;;
    esac
    prefix="${cmd%" $logic"}"
    case "$prefix" in
        /*) : ;;
        *) return 1 ;;
    esac
    backend_live_exec_allowed "$prefix" "$expected_exec" || return 1
    return 0
}

# Service-specific live command check shared by recorded stops and bootstrap
# child cleanup.
cmd_matches_spawn() {
    local service="$1" cmd="$2" want="$3"
    case "$service" in
        "$OLLAMA_SERVICE")
            case "$cmd" in *"$want") return 0 ;; *) return 1 ;; esac
            ;;
        "$BACKEND_SERVICE")
            backend_live_cmd_shape_ok "$cmd" \
                "$(normalize_cmd "$BACKEND_LOGIC_ARGS")" \
                "$(normalize_cmd "$PYTHON_BIN")"
            ;;
        *) return 1 ;;
    esac
}

# ps may not report a just-started process immediately; retry a bounded
# number of times before failing closed.  Used both for freshly spawned
# children and for this launcher's own fingerprint.
_lstart_retry() {
    local pid="$1" tries="$2" interval="$3" i raw
    i=0
    while [ "$i" -lt "$tries" ]; do
        raw="$(read_lstart "$pid")"
        [ -n "$raw" ] && { printf '%s' "$raw"; return 0; }
        "$SLEEP_BIN" "$interval"
        i=$((i + 1))
    done
    return 1
}

read_lstart_retry() {
    _lstart_retry "$1" "$LSTART_RETRIES" "$LSTART_INTERVAL"
}

# Wait for the child to complete exec, then capture its full live command.
# The command must end with the exact fixed backend logical args; the
# captured executable may be the macOS framework Python path rather than
# .venv/bin/python.  Returns the normalized live command or empty.
read_backend_live_cmd_retry() {
    local pid="$1" tries="$LSTART_RETRIES" interval="$LSTART_INTERVAL"
    local i=0 cmd logic expected_exec
    logic="$(normalize_cmd "$BACKEND_LOGIC_ARGS")"
    expected_exec="$(normalize_cmd "$PYTHON_BIN")"
    while [ "$i" -lt "$tries" ]; do
        cmd="$(read_cmdline "$pid")"
        if [ -n "$cmd" ] && \
           backend_live_cmd_shape_ok "$cmd" "$logic" "$expected_exec"; then
            printf '%s' "$cmd"
            return 0
        fi
        "$SLEEP_BIN" "$interval"
        i=$((i + 1))
    done
    return 1
}

# This launcher's own fingerprint, with a retry window wide enough for the
# process to become visible to ps.
own_lstart_retry() {
    _lstart_retry "$$" "$SELF_LSTART_RETRIES" "$SELF_LSTART_INTERVAL"
}


# Verify that a PID still belongs to the expected process: the live command
# line AND the live start-time fingerprint must match the record.
_identity_ok() {
    local pid="$1" service="$2" expected_cmd="$3" expected_live_cmd="$4"
    local expected_fp="$5"
    local state cmd want live_fp logic expected_exec
    is_positive_integer "$pid" || return 1
    proc_view "$pid" >/dev/null
    [ "$PROC_VIEW_ALIVE" -eq 1 ] || return 1
    state="$PROC_STATE_VALUE"
    [ -n "$state" ] || return 1
    cmd="$(read_cmdline "$pid")"
    [ -n "$cmd" ] || return 1
    want="$(normalize_cmd "$expected_cmd")"
    case "$service" in
        "$OLLAMA_SERVICE")
            case "$cmd" in *"$want") : ;; *) return 1 ;; esac
            ;;
        "$BACKEND_SERVICE")
            logic="$(normalize_cmd "$BACKEND_LOGIC_ARGS")"
            expected_exec="$(normalize_cmd "$PYTHON_BIN")"
            [ -n "$expected_live_cmd" ] || return 1
            backend_live_cmd_shape_ok "$cmd" "$logic" "$expected_exec" || return 1
            [ "$cmd" = "$expected_live_cmd" ] || return 1
            ;;
        *) return 1 ;;
    esac
    # Fingerprint: if the platform cannot provide one, fail closed.
    live_fp="$(read_lstart "$pid")"
    [ -n "$live_fp" ] || return 1
    [ -n "$expected_fp" ] || return 1
    [ "$live_fp" = "$expected_fp" ] || return 1
    return 0
}

# ---------------------------------------------------------------------------
# PID records
# ---------------------------------------------------------------------------

# Atomically write a PID record.  The temp file is unique per run and per
# service, and mv is atomic, so two launchers can never corrupt each other's
# file; the RUN_ID field lets a launcher know which records are its own.
write_pid_record() {
    local path="$1" pid="$2" service="$3" signature="$4" fingerprint="$5"
    local live_cmd="${6:-}" tmp
    "$MKDIR_BIN" -p "$RUNTIME_DIR" || return 1
    # A pre-existing record belongs to some other run; silently replacing it
    # would let this launcher delete or hijack another run's record.
    if [ -f "$path" ]; then
        warn "a PID record already exists at $path; refusing to overwrite it."
        warn "Run: bash scripts/stop-demo.sh  to clean up the previous run first."
        return 2
    fi
    tmp="$path.$$"
    {
        printf '# Teachable Agent demo launcher record; safe to delete.\n'
        printf 'PID=%s\n' "$pid"
        printf 'SERVICE=%s\n' "$service"
        printf 'PROJECT_ROOT=%s\n' "$PROJECT"
        printf 'CMD=%s\n' "$signature"
        if [ -n "$live_cmd" ]; then
            printf 'LIVE_CMD=%s\n' "$live_cmd"
        fi
        printf 'FINGERPRINT=%s\n' "$fingerprint"
        printf 'RUN_ID=%s\n' "$RUN_ID"
    } > "$tmp" || return 1
    mv -f -- "$tmp" "$path" || return 1
    return 0
}

# Strictly parse a record: every field must appear exactly once.  Duplicate
# or missing fields make the record unverifiable -> refuse to signal.
parse_pid_record() {
    local path="$1"
    local pid service root cmd fingerprint run_id
    local pid_count service_count root_count cmd_count fp_count run_count
    local line key value

    [ -f "$path" ] || return 1

    pid=""; service=""; root=""; cmd=""; fingerprint=""; run_id=""
    pid_count=0; service_count=0; root_count=0; cmd_count=0; fp_count=0
    run_count=0

    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|'') continue ;;
        esac
        case "$line" in
            *=*) : ;;
            *) continue ;;  # malformed line; the field counters decide
        esac
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            PID)
                pid="$value"; pid_count=$((pid_count + 1)) ;;
            SERVICE)
                service="$value"; service_count=$((service_count + 1)) ;;
            PROJECT_ROOT)
                root="$value"; root_count=$((root_count + 1)) ;;
            CMD)
                cmd="$value"; cmd_count=$((cmd_count + 1)) ;;
            FINGERPRINT)
                fingerprint="$value"; fp_count=$((fp_count + 1)) ;;
            RUN_ID)
                run_id="$value"; run_count=$((run_count + 1)) ;;
        esac
    done < "$path"

    if [ "$pid_count" -ne 1 ] || [ "$service_count" -ne 1 ] || \
       [ "$root_count" -ne 1 ] || [ "$cmd_count" -ne 1 ] || \
       [ "$fp_count" -ne 1 ] || [ "$run_count" -ne 1 ]; then
        return 1
    fi

    REC_PID="$pid"; REC_SERVICE="$service"; REC_ROOT="$root"
    REC_CMD="$cmd"; REC_FP="$fingerprint"; REC_RUN_ID="$run_id"
    return 0
}

# ---------------------------------------------------------------------------
# Single-instance lock
# ---------------------------------------------------------------------------

# Parse the lock metadata strictly: every field must appear exactly once,
# and the PID must be a strict positive integer.  A lock directory whose
# info file is missing, empty, partially written, duplicated or otherwise
# unparsable is NEVER treated as stale - it is simply unusable.
LOCK_PID=""; LOCK_RUN_ID=""; LOCK_FP=""; LOCK_ROOT=""
parse_lock_info() {
    local info="$LOCK_FILE/info"
    local line key value
    local pid_count=0 run_count=0 fp_count=0 root_count=0

    LOCK_PID=""; LOCK_RUN_ID=""; LOCK_FP=""; LOCK_ROOT=""
    [ -f "$info" ] || return 1

    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|'') continue ;;
        esac
        case "$line" in
            *=*) : ;;
            *) continue ;;
        esac
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            PID)
                LOCK_PID="$value"; pid_count=$((pid_count + 1)) ;;
            RUN_ID)
                LOCK_RUN_ID="$value"; run_count=$((run_count + 1)) ;;
            FINGERPRINT)
                LOCK_FP="$value"; fp_count=$((fp_count + 1)) ;;
            PROJECT_ROOT)
                LOCK_ROOT="$value"; root_count=$((root_count + 1)) ;;
        esac
    done < "$info"

    [ "$pid_count" -eq 1 ] || return 1
    [ "$run_count" -eq 1 ] || return 1
    [ "$fp_count" -eq 1 ] || return 1
    [ "$root_count" -eq 1 ] || return 1
    is_positive_integer "$LOCK_PID" || return 1
    [ -n "$LOCK_FP" ] || return 1
    [ -n "$LOCK_RUN_ID" ] || return 1
    [ -n "$LOCK_ROOT" ] || return 1
    return 0
}

# Atomically publish this run's lock metadata: the complete file is written
# to a temporary file first and then moved into place, so a concurrent
# launcher can never observe a partially written info file.
publish_lock_info() {
    local tmp="$LOCK_FILE/info.tmp.$$"
    if ! {
        printf 'PID=%s\n' "$$"
        printf 'RUN_ID=%s\n' "$RUN_ID"
        printf 'FINGERPRINT=%s\n' "$LOCK_FP_VALUE"
        printf 'PROJECT_ROOT=%s\n' "$PROJECT"
    } > "$tmp"; then
        rm -f -- "$tmp"
        return 1
    fi
    if ! mv -f -- "$tmp" "$LOCK_FILE/info"; then
        rm -f -- "$tmp"
        return 1
    fi
    return 0
}

# Create the lock directory and publish its metadata.  mkdir is atomic, and
# ownership is marked (LOCK_HELD=1) the instant it succeeds, so every failure
# path below can release only the lock this process just created.
create_lock() {
    local state
    if ! "$MKDIR_BIN" "$LOCK_FILE" 2>/dev/null; then
        return 1
    fi
    LOCK_HELD=1
    LOCK_RUN_ID="$RUN_ID"
    LOCK_PID="$$"
    LOCK_ROOT="$PROJECT"

    # The launcher's own start time must be readable; otherwise the lock
    # could not be verified later, so fail closed.  Retry briefly: on a busy
    # machine the process may not be visible to ps for a moment.
    LOCK_FP_VALUE="$(own_lstart_retry || true)"
    if [ -z "$LOCK_FP_VALUE" ]; then
        warn "cannot read this launcher's own start fingerprint; refusing to hold a lock."
        rm -rf -- "$LOCK_FILE"
        LOCK_HELD=0
        return 2
    fi
    state="$(proc_state "$$")"
    case "$state" in
        ''|Z*)
            warn "this launcher process is not observable; refusing to hold a lock."
            rm -rf -- "$LOCK_FILE"
            LOCK_HELD=0
            return 2
            ;;
    esac

    if ! publish_lock_info; then
        warn "could not write the lock metadata at $LOCK_FILE/info."
        rm -rf -- "$LOCK_FILE"
        LOCK_HELD=0
        return 2
    fi
    LOCK_FP="$LOCK_FP_VALUE"
    return 0
}

# The lock directory is created atomically (mkdir), so only one launcher can
# ever win it.  A lock that exists but cannot be verified is never removed.
acquire_lock() {
    local rc owner_view

    set +e
    create_lock
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        return 0
    fi
    if [ "$rc" -eq 2 ]; then
        fail "could not create a verifiable launcher lock at $LOCK_FILE."
        exit 1
    fi

    # The lock directory already exists.
    if parse_lock_info; then
        # A lock for a different project root is foreign and must never be
        # deleted, not even when its owner PID is gone.
        if [ "$LOCK_ROOT" != "$PROJECT" ]; then
            fail "a launcher lock exists for another project root ('$LOCK_ROOT'),"
            fail "not this project ('$PROJECT'). Refusing to remove it."
            fail "Inspect it manually: $LOCK_FILE"
            exit 1
        fi
        proc_view_into "$LOCK_PID"
        owner_view="$PROC_VIEW"
        case "$owner_view" in
            alive)
                if [ "$LOCK_FP" = "$(read_lstart "$LOCK_PID" || true)" ]; then
                    fail "another launcher is already running (PID $LOCK_PID)."
                else
                    fail "a launcher lock exists whose owner (PID $LOCK_PID) cannot be verified;"
                    fail "refusing to remove it. Inspect: $LOCK_FILE"
                fi
                fail "Stop it with Ctrl+C in its terminal or: bash scripts/stop-demo.sh"
                exit 1
                ;;
            unknown)
                fail "a launcher lock exists and its owner (PID $LOCK_PID) cannot be checked"
                fail "($PROC_VIEW_DETAIL). Refusing to treat it as stale; nothing was removed."
                fail "Inspect it manually: $LOCK_FILE"
                exit 1
                ;;
            gone)
                info "Removing a stale launcher lock (owner PID $LOCK_PID no longer exists)."
                rm -rf -- "$LOCK_FILE"
                ;;
        esac
    else
        fail "a launcher lock exists at $LOCK_FILE but its owner metadata is missing,"
        fail "incomplete, or unparsable. Refusing to treat it as stale."
        fail "If no launcher is running, remove it manually: rm -rf \"$LOCK_FILE\""
        exit 1
    fi

    set +e
    create_lock
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        fail "could not acquire the launcher lock at $LOCK_FILE after cleaning a stale one."
        exit 1
    fi
    return 0
}

# Release the lock only when every field still identifies this very run.
release_lock() {
    [ "$LOCK_HELD" -eq 1 ] || return 0
    if parse_lock_info; then
        if [ "$LOCK_PID" = "$$" ] && \
           [ "$LOCK_RUN_ID" = "$RUN_ID" ] && \
           [ "$LOCK_ROOT" = "$PROJECT" ] && \
           [ -n "$LOCK_FP" ] && \
           [ "$LOCK_FP" = "$(own_lstart_retry || true)" ]; then
            rm -rf -- "$LOCK_FILE"
        else
            warn "not releasing $LOCK_FILE: it no longer belongs to this run."
        fi
    else
        warn "not releasing $LOCK_FILE: its owner record is unreadable."
    fi
    LOCK_HELD=0
}

# ---------------------------------------------------------------------------
# Port and health helpers
# ---------------------------------------------------------------------------

# Sets PORT_STATE to free|used|unknown and PORT_PID to the listener PID.
#
# Fail closed: only the standard "no matching listener" result may mark a
# port as free:
#   rc=0 + non-empty stdout whose first line is a strict positive integer
#          + empty stderr                                          -> used
#   rc=1 + empty stdout + empty stderr                              -> free
#   anything else (rc=2/126/127, non-empty stderr even with rc=0 and a
#   valid PID, rc=0 with empty or malformed output)                  -> unknown -> abort
PORT_STATE=""
PORT_PID=""
check_port() {
    local port="$1" out_file err_file rc first_pid
    out_file="$(mktemp "${TMPDIR:-/tmp}/demo-lsof-out.XXXXXX")"
    err_file="$(mktemp "${TMPDIR:-/tmp}/demo-lsof-err.XXXXXX")"
    PORT_STATE="unknown"
    PORT_PID=""

    if [ ! -x "$LSOF_BIN" ]; then
        rm -f -- "$out_file" "$err_file"
        die "lsof is not executable: $LSOF_BIN" \
            "Install the macOS command line tools, or set TEACHABLE_LSOF_BIN." \
            "The demo needs lsof to verify that ports $OLLAMA_PORT/$APP_PORT are free."
    fi

    set +e
    "$LSOF_BIN" -nP -a "-iTCP:$port" -sTCP:LISTEN -t >"$out_file" 2>"$err_file"
    rc=$?
    set -e

    # Strict table:
    #   rc=0 + strict positive integer on stdout + empty stderr -> used
    #   rc=1 + empty stdout + empty stderr                      -> free
    #   anything else                                           -> unknown
    if [ "$rc" -eq 0 ] && [ -s "$out_file" ] && [ ! -s "$err_file" ]; then
        first_pid="$(head -n 1 "$out_file" | tr -d '[:space:]')"
        if is_positive_integer "$first_pid"; then
            PORT_STATE="used"
            PORT_PID="$first_pid"
        fi
    elif [ "$rc" -eq 1 ] && [ ! -s "$out_file" ] && [ ! -s "$err_file" ]; then
        PORT_STATE="free"
    fi

    rm -f -- "$out_file" "$err_file"
    case "$PORT_STATE" in
        free|used) return 0 ;;
        *)
            die "cannot determine whether port $port is in use (lsof exit $rc)" \
                "The lsof output did not look like a normal answer, so the launcher refuses" \
                "to guess. Inspect the port manually with:" \
                "  lsof -nP -iTCP:$port -sTCP:LISTEN" \
                "Nothing was started; no process was touched."
            ;;
    esac
}

ollama_version_ok() {
    local body
    body="$("$CURL_BIN" -sS --max-time 3 "$OLLAMA_URL/api/version" 2>/dev/null)" || return 1
    printf '%s' "$body" | grep -q '"version"' || return 1
    return 0
}

app_health_ok() {
    local body
    body="$("$CURL_BIN" -sS --max-time 3 "$APP_HEALTH_URL" 2>/dev/null)" || return 1
    printf '%s' "$body" | grep -q '"ok"' || return 1
    return 0
}

wait_for() {
    local label="$1" attempts="$2" predicate="$3" i=1
    info "Waiting for $label ..."
    while [ "$i" -le "$attempts" ]; do
        if "$predicate"; then
            info "$label is ready."
            return 0
        fi
        "$SLEEP_BIN" "$READY_INTERVAL"
        i=$((i + 1))
    done
    return 1
}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Try to stop one of this run's own processes.
#
# Return values:
#   0   the process is verifiably gone (the record may be removed)
#   1   the process is still alive and could not be stopped, OR the PID is
#       alive but no longer matches the record (never signal it).  In both
#       cases the PID record must be kept for retry and manual inspection.
#
# `record_file` is kept for diagnostics; `stored_fp` is the fingerprint the
# record carries (empty when unknown).
stop_owned_process() {
    local pid="$1" service="$2" signature="$3" timeout="$4" label="$5" \
          record_file="$6" stored_fp="$7" stored_live_cmd="$8"
    local i max_ticks view

    [ -n "$pid" ] || return 0

    proc_view_into "$pid"
    view="$PROC_VIEW"
    if [ "$view" = "gone" ]; then
        # ps explicitly confirmed the PID does not exist.
        return 0
    fi
    if [ "$view" = "unknown" ]; then
        # We cannot even observe the process; never signal and never let the
        # caller delete the record.
        warn "$label (PID $pid): cannot determine the process state ($PROC_VIEW_DETAIL)."
        warn "Refusing to signal it and keeping the PID record."
        return 1
    fi

    if ! _identity_ok "$pid" "$service" "$signature" "$stored_live_cmd" "$stored_fp"; then
        warn "$label (PID $pid) is alive but no longer matches this launcher's record"
        warn "(command line or process start fingerprint differs - the PID may have"
        warn "been reused). Refusing to signal it."
        warn "Keeping the PID record for manual inspection."
        return 1
    fi

    info "Stopping $label (PID $pid) ..."
    kill -TERM "$pid" 2>/dev/null || true

    max_ticks=$((timeout * 10))
    i=0
    while [ "$i" -lt "$max_ticks" ]; do
        proc_view_into "$pid"
        view="$PROC_VIEW"
        if [ "$view" = "gone" ]; then
            return 0
        fi
        if [ "$view" = "unknown" ]; then
            warn "$label (PID $pid): lost track of the process state ($PROC_VIEW_DETAIL)."
            warn "Refusing to assume it stopped; keeping the PID record."
            return 1
        fi
        if ! _identity_ok "$pid" "$service" "$signature" "$stored_live_cmd" "$stored_fp"; then
            # Either the identity genuinely changed (leave it alone) or the
            # process simply exited between the state and identity probes.
            # Re-probe the state to tell those two cases apart.
            proc_view_into "$pid"
            if [ "$PROC_VIEW" = "gone" ]; then
                return 0
            fi
            if [ "$PROC_VIEW" = "unknown" ]; then
                warn "$label (PID $pid): process state is unreadable ($PROC_VIEW_DETAIL)."
                warn "Keeping the PID record for retry and manual inspection."
                return 1
            fi
            break
        fi
        "$SLEEP_BIN" 0.1
        i=$((i + 1))
    done

    proc_view_into "$pid"
    view="$PROC_VIEW"
    if [ "$view" = "gone" ]; then
        return 0
    fi

    # Still alive: only escalate when the live identity still matches.
    if [ "$view" = "alive" ] && \
       _identity_ok "$pid" "$service" "$signature" "$stored_live_cmd" "$stored_fp"; then
        warn "$label (PID $pid) did not stop within ${timeout}s; escalating to SIGKILL."
        kill -KILL "$pid" 2>/dev/null || true
        i=0
        while [ "$i" -lt 50 ]; do
            proc_view_into "$pid"
        view="$PROC_VIEW"
            if [ "$view" = "gone" ]; then
                return 0
            fi
            if [ "$view" = "unknown" ]; then
                warn "$label (PID $pid): lost track of the process state ($PROC_VIEW_DETAIL)."
                warn "Refusing to assume it stopped; keeping the PID record."
                return 1
            fi
            if ! _identity_ok "$pid" "$service" "$signature" "$stored_live_cmd" "$stored_fp"; then
                proc_view_into "$pid"
                if [ "$PROC_VIEW" = "gone" ]; then
                    return 0
                fi
                if [ "$PROC_VIEW" = "unknown" ]; then
                    warn "$label (PID $pid): process state is unreadable ($PROC_VIEW_DETAIL)."
                    warn "Keeping the PID record for retry and manual inspection."
                    return 1
                fi
                break
            fi
            "$SLEEP_BIN" 0.1
            i=$((i + 1))
        done
    fi

    # The TERM/KILL wait loop probes state first, then identity: the process
    # can exit between those two probes, so the final decision must be based
    # on a fresh state probe taken after the kill, not on an identity
    # mismatch caused by the process disappearing mid-check.
    proc_view_into "$pid"
    view="$PROC_VIEW"
    if [ "$view" = "gone" ]; then
        return 0
    fi
    if [ "$view" = "unknown" ]; then
        warn "$label (PID $pid): process state is unreadable ($PROC_VIEW_DETAIL)."
        warn "Keeping the PID record for retry and manual inspection."
        return 1
    fi
    warn "$label (PID $pid) is still running (or its identity no longer matches)."
    warn "Keeping the PID record for retry and manual inspection."
    return 1
}

# ---------------------------------------------------------------------------
# Bootstrap child cleanup
# ---------------------------------------------------------------------------
#
# Strictly limited safety valve for the window between "the shell just forked
# this child and captured its PID from $!" and "the child's PID record with a
# verified start fingerprint exists".
#
# Safety boundaries (all of them must hold):
#   * the PID is the one this shell obtained from $! for a service it started
#     in this run (BOOT_*_PID); never a PID read from a file;
#   * nothing was written to a PID record for it yet (BOOT_*_RECORDED=0), so
#     the ordinary verified path cannot be used;
#   * the shell has not reaped it yet (BOOT_*_REAPED=0), so it is still a
#     direct child of this shell;
#   * before any signal, the caller passed the same service and command
#     signature this run used to spawn it, and the live command line still
#     matches that signature.  A gone or zombie child is only reaped and is
#     never signalled, so it needs no identity proof;
#   * state unknown fails closed with no signal and no wait on any path.
#
# It never uses broad name-based kills or port lookups, and it never signals
# a live process whose command line does not match this shell's spawn.
# ---------------------------------------------------------------------------
# Bounded bootstrap-child collection
# ---------------------------------------------------------------------------
#
# `wait` is the only way to collect a direct child, but it must never be
# entered for a child that is unknown or still running: a blocking wait
# cannot be interrupted by this script's own limits.  Therefore every reap
# follows this order:
#   1. proc_view_into polls until a fresh probe says `gone` or reports a
#      zombie state;
#   2. only that collectable result may call wait, where it cannot block;
#   3. a 127 result is the one and only "not collectable" answer; any other
#      status (including 137/143 from SIGKILL/SIGTERM) means the direct
#      child was actually reaped.
# Every polling path is bounded, and no path calls wait before step 2.

COLLECT_VIEW=""    # gone|zombie when a wait is safe
COLLECT_DETAIL=""  # diagnostic for the last non-collectable probe

# Poll a direct child until a wait on it cannot block.
# Returns:
#   0  gone or zombie (COLLECT_VIEW)
#   1  still running after max_ticks fresh probes
#   2  state unknown (COLLECT_DETAIL); the caller must fail closed
# This function never signals and never calls wait.
await_child_collectable() {
    local pid="$1" max_ticks="$2" i=0

    COLLECT_VIEW=""
    COLLECT_DETAIL=""
    [ "$max_ticks" -gt 0 ] 2>/dev/null || max_ticks=1

    while [ "$i" -lt "$max_ticks" ]; do
        proc_view_into "$pid"
        case "$PROC_STATE_VALUE" in
            Z*)
                COLLECT_VIEW="zombie"
                return 0
                ;;
        esac
        case "$PROC_VIEW" in
            gone)
                COLLECT_VIEW="gone"
                return 0
                ;;
            unknown)
                COLLECT_DETAIL="$PROC_VIEW_DETAIL"
                return 2
                ;;
        esac
        "$SLEEP_BIN" 0.1
        i=$((i + 1))
    done

    COLLECT_DETAIL="state still looked alive after $max_ticks probes"
    return 1
}

REAP_GOT=0       # 1 when the child was collected
REAP_STATUS=""   # wait status when known
REAP_DETAIL=""   # why a reap failed

# Reap one of this shell's direct children with a hard bound.
#
# Returns:
#   0  the child was collected (137/143, i.e. SIGKILL/SIGTERM, are normal)
#   1  the child was not collected: still running, state unknown, or bash
#      answered 127 ("not a collectable child").  In every case wait was
#      only entered after a fresh probe confirmed gone or zombie.
reap_child() {
    local pid="$1" timeout="${2:-1}" max_ticks rc

    REAP_GOT=0
    REAP_STATUS=""
    REAP_DETAIL=""

    max_ticks=$((timeout * 10))
    [ "$max_ticks" -gt 0 ] 2>/dev/null || max_ticks=1

    set +e
    await_child_collectable "$pid" "$max_ticks"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        REAP_DETAIL="${COLLECT_DETAIL:-no collectable state within ${timeout}s}"
        return 1
    fi

    # COLLECT_VIEW is gone or zombie, so wait cannot block: a collectable
    # direct child is reaped immediately, and a PID that is not this shell's
    # child answers 127 immediately as well.
    set +e
    wait "$pid" 2>/dev/null
    rc=$?
    set -e
    if [ "$rc" -eq 127 ]; then
        REAP_DETAIL="wait reports PID $pid is not a collectable direct child"
        return 1
    fi

    REAP_GOT=1
    REAP_STATUS="$rc"
    return 0
}

stop_bootstrap_child() {
    local pid="$1" service="$2" signature="$3" timeout="$4" label="$5"
    local max_ticks view want cmd rc

    BOOT_DETAIL=""
    if [ -z "$pid" ]; then
        BOOT_DETAIL="$label: no bootstrap child PID"
        return 0
    fi
    if is_positive_integer "$pid" 2>/dev/null; then :; else
        BOOT_DETAIL="$label: bootstrap PID '$pid' is not a positive integer"
        return 1
    fi

    max_ticks=$((timeout * 10))
    [ "$max_ticks" -gt 0 ] 2>/dev/null || max_ticks=1

    # Initial state first.  A process that is already gone or a zombie will
    # never be signalled, so it needs no command-line identity proof: it only
    # needs its direct-child reap.  proc_view_into reports a zombie as
    # PROC_VIEW=unknown together with PROC_STATE_VALUE=Z*, so the state
    # letter must be inspected before the unknown verdict.  Everything
    # genuinely unknown fails closed here, before any command read, signal,
    # or wait.
    proc_view_into "$pid"
    view="$PROC_VIEW"
    case "$PROC_STATE_VALUE" in
        Z*)
            if reap_child "$pid" "$timeout"; then
                return 0
            fi
            BOOT_DETAIL="$label (PID $pid): already a zombie, but could not be reaped (${REAP_DETAIL:-unknown})"
            return 1
            ;;
    esac
    case "$view" in
        gone)
            if reap_child "$pid" "$timeout"; then
                return 0
            fi
            BOOT_DETAIL="$label (PID $pid): gone, but could not be reaped (${REAP_DETAIL:-unknown})"
            return 1
            ;;
        unknown)
            BOOT_DETAIL="$label (PID $pid): cannot observe the child's state ($PROC_VIEW_DETAIL); no signal sent and it may still be running"
            return 1
            ;;
    esac

    # The child is still alive, so this run may only signal it after proving
    # the live command line is still the one this shell spawned.
    want="$(normalize_cmd "$signature")"
    cmd="$(read_cmdline "$pid")"
    if [ -z "$cmd" ]; then
        # The child may have exited between the state and command probes.
        # Re-probe the state and reap it only if that fresh state is
        # collectable; otherwise fail closed without a signal.
        proc_view_into "$pid"
        view="$PROC_VIEW"
        case "$PROC_STATE_VALUE" in
            Z*)
                if reap_child "$pid" "$timeout"; then
                    return 0
                fi
                BOOT_DETAIL="$label (PID $pid): became a zombie, but could not be reaped (${REAP_DETAIL:-unknown})"
                return 1
                ;;
        esac
        if [ "$view" = "gone" ]; then
            if reap_child "$pid" "$timeout"; then
                return 0
            fi
            BOOT_DETAIL="$label (PID $pid): exited, but could not be reaped (${REAP_DETAIL:-unknown})"
            return 1
        fi
        BOOT_DETAIL="$label (PID $pid): cannot read the command line; refusing to signal"
        return 1
    fi
    if cmd_matches_spawn "$service" "$cmd" "$want"; then
        :
    else
        BOOT_DETAIL="$label (PID $pid): command line does not match this run's spawn"
        return 1
    fi

    info "Stopping $label (PID $pid, bootstrap child without a PID record) ..."
    kill -TERM "$pid" 2>/dev/null || true

    # Bounded TERM window.  Every iteration uses its own fresh state probe;
    # there is no cached state and no wait here.
    set +e
    await_child_collectable "$pid" "$max_ticks"
    rc=$?
    set -e
    if [ "$rc" -eq 2 ]; then
        BOOT_DETAIL="$label (PID $pid): lost track of the child's state ($COLLECT_DETAIL); no SIGKILL sent and it may still be running"
        return 1
    fi
    if [ "$rc" -eq 0 ]; then
        if reap_child "$pid" "$timeout"; then
            return 0
        fi
        BOOT_DETAIL="$label (PID $pid): stopped, but could not be reaped (${REAP_DETAIL:-unknown})"
        return 1
    fi

    # Still alive after the TERM window.  Re-verify the live command line
    # before the single permitted escalation; if it has changed, do not
    # signal and do not wait.
    cmd="$(read_cmdline "$pid")"
    if [ -z "$cmd" ]; then
        BOOT_DETAIL="$label (PID $pid): cannot re-verify the command line; no SIGKILL sent"
        return 1
    fi
    if cmd_matches_spawn "$service" "$cmd" "$want"; then
        :
    else
        BOOT_DETAIL="$label (PID $pid): command line no longer matches; no SIGKILL sent"
        return 1
    fi

    warn "$label (PID $pid) did not stop within ${timeout}s; escalating to SIGKILL."
    kill -KILL "$pid" 2>/dev/null || true

    # Bounded KILL window.  A fixed probe budget keeps a process that
    # survives SIGKILL from extending the path; again only a fresh
    # gone/zombie answer may lead to wait.
    set +e
    await_child_collectable "$pid" 50
    rc=$?
    set -e
    if [ "$rc" -eq 2 ]; then
        BOOT_DETAIL="$label (PID $pid): lost track of the child's state after SIGKILL ($COLLECT_DETAIL); no wait entered and it may still be running"
        return 1
    fi
    if [ "$rc" -ne 0 ]; then
        BOOT_DETAIL="$label (PID $pid): still running after SIGKILL; no wait entered"
        return 1
    fi
    if reap_child "$pid" "$timeout"; then
        return 0
    fi
    BOOT_DETAIL="$label (PID $pid): kill was delivered, but the child could not be reaped (${REAP_DETAIL:-unknown})"
    return 1
}

# Delete a record only when it still belongs to this run.
remove_own_record() {
    local record_file="$1"
    local rec_run_id
    [ -n "$record_file" ] || return 0
    [ -f "$record_file" ] || return 0
    rec_run_id="$(sed -n 's/^RUN_ID=//p' "$record_file" | head -n 1 || true)"
    [ -n "$rec_run_id" ] || return 0
    [ "$rec_run_id" = "$RUN_ID" ] || return 0
    rm -f -- "$record_file"
}

# ---------------------------------------------------------------------------
# Cleanup-failed quarantine marker
# ---------------------------------------------------------------------------
#
# Strictly limited bootstrap cleanup can honestly fail closed when the child
# state becomes unknown.  The child then has no PID record, so a marker in the
# runtime directory (not a forged fingerprint, and never a stale launcher
# lock) keeps it visible to the next launch.
#
# Lifecycle:
#   * written atomically, and only when no marker already exists;
#   * a valid own-project marker with a trusted `gone` PID is removed before
#     any service starts;
#   * alive, unknown or malformed/foreign markers are preserved and abort the
#     launch before any service is started;
#   * the marker is never used to signal a process.

CLEANUP_FAILED_PID=""
CLEANUP_FAILED_SERVICE=""
CLEANUP_FAILED_ROOT=""
CLEANUP_FAILED_CMD=""
CLEANUP_FAILED_RUN_ID=""
CLEANUP_FAILED_REASON=""
CLEANUP_FAILED_REASON_TEXT="bootstrap child could not be confirmed stopped"

# ---------------------------------------------------------------------------
# Bootstrap guard (pre-child persistent protection)
# ---------------------------------------------------------------------------
#
# Before this launcher forks a service child it atomically claims a guard
# directory.  The guard carries the service, project root, command, run id and
# a fixed reason, but no PID and no fingerprint: the child does not exist yet.
# It is removed only after a formal PID record exists, after a confirmed
# bootstrap cleanup, or after the full quarantine record has been published.
#
# If the guard cannot be created, no child is started.  If a child exists and
# publishing the full quarantine record fails, the guard remains and the next
# launch fails closed before starting any service.

BOOTSTRAP_GUARD_REASON_TEXT="bootstrap child lifecycle in progress; complete quarantine record not published"
GUARD_SERVICE=""
GUARD_ROOT=""
GUARD_CMD=""
GUARD_RUN_ID=""
GUARD_REASON=""

parse_bootstrap_guard() {
    local guard_dir="$1" info line key value
    local service_count=0 root_count=0 cmd_count=0 run_count=0 reason_count=0

    GUARD_SERVICE=""
    GUARD_ROOT=""
    GUARD_CMD=""
    GUARD_RUN_ID=""
    GUARD_REASON=""
    info="$guard_dir/info"
    [ -f "$info" ] || return 1

    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|'') continue ;;
        esac
        case "$line" in
            *=*) : ;;
            *) return 1 ;;
        esac
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            SERVICE)
                GUARD_SERVICE="$value"; service_count=$((service_count + 1)) ;;
            PROJECT_ROOT)
                GUARD_ROOT="$value"; root_count=$((root_count + 1)) ;;
            CMD)
                GUARD_CMD="$value"; cmd_count=$((cmd_count + 1)) ;;
            RUN_ID)
                GUARD_RUN_ID="$value"; run_count=$((run_count + 1)) ;;
            REASON)
                GUARD_REASON="$value"; reason_count=$((reason_count + 1)) ;;
            *)
                return 1 ;;
        esac
    done < "$info"

    [ "$service_count" -eq 1 ] || return 1
    [ "$root_count" -eq 1 ] || return 1
    [ "$cmd_count" -eq 1 ] || return 1
    [ "$run_count" -eq 1 ] || return 1
    [ "$reason_count" -eq 1 ] || return 1
    case "$GUARD_SERVICE" in
        "$OLLAMA_SERVICE"|"$BACKEND_SERVICE") : ;;
        *) return 1 ;;
    esac
    [ -n "$GUARD_ROOT" ] || return 1
    [ -n "$GUARD_CMD" ] || return 1
    [ -n "$GUARD_RUN_ID" ] || return 1
    [ "$GUARD_REASON" = "$BOOTSTRAP_GUARD_REASON_TEXT" ] || return 1
    return 0
}

# Atomically claim the guard directory before forking.  If the directory
# cannot be claimed, the caller must not start the child.  Once the directory
# exists, a failed metadata write leaves it in place as persistent protection.
create_bootstrap_guard() {
    local service="$1" signature="$2" info tmp
    case "$service" in
        "$OLLAMA_SERVICE"|"$BACKEND_SERVICE") : ;;
        *) return 1 ;;
    esac
    [ -n "$signature" ] || return 1

    # Fault injection for the deterministic no-child-on-guard-failure test.
    if [ -n "${TEACHABLE_TEST_FORCE_GUARD_FAIL:-}" ]; then
        warn "injected bootstrap guard creation failure."
        return 1
    fi

    if [ -e "$CLEANUP_FAILED_FILE" ] || [ -L "$CLEANUP_FAILED_FILE" ]; then
        warn "a cleanup-failed marker already exists; refusing to create a bootstrap guard."
        return 1
    fi
    if [ -e "$BOOTSTRAP_GUARD_DIR" ] || [ -L "$BOOTSTRAP_GUARD_DIR" ]; then
        warn "a bootstrap guard already exists at $BOOTSTRAP_GUARD_DIR; refusing to reuse it."
        return 1
    fi
    "$MKDIR_BIN" "$BOOTSTRAP_GUARD_DIR" 2>/dev/null || return 1

    info="$BOOTSTRAP_GUARD_DIR/info"
    tmp="$BOOTSTRAP_GUARD_DIR/info.tmp.$$"
    if {
        printf '# Teachable Agent bootstrap guard; remove only after inspection.\n'
        printf 'SERVICE=%s\n' "$service"
        printf 'PROJECT_ROOT=%s\n' "$PROJECT"
        printf 'CMD=%s\n' "$signature"
        printf 'RUN_ID=%s\n' "$RUN_ID"
        printf 'REASON=%s\n' "$BOOTSTRAP_GUARD_REASON_TEXT"
    } > "$tmp"; then
        :
    else
        rm -f -- "$tmp"
        return 1
    fi
    if mv -f -- "$tmp" "$info"; then
        return 0
    fi
    rm -f -- "$tmp"
    return 1
}

# Remove the guard only when its complete metadata proves it belongs to this
# exact run, service and command.  Otherwise it is left untouched.
release_bootstrap_guard() {
    local service="$1" signature="$2"
    if parse_bootstrap_guard "$BOOTSTRAP_GUARD_DIR"; then
        :
    else
        warn "not releasing $BOOTSTRAP_GUARD_DIR: its metadata is unreadable."
        return 1
    fi
    if [ "$GUARD_SERVICE" = "$service" ] && \
       [ "$GUARD_ROOT" = "$PROJECT" ] && \
       [ "$GUARD_RUN_ID" = "$RUN_ID" ] && \
       [ "$GUARD_CMD" = "$signature" ]; then
        if rm -rf -- "$BOOTSTRAP_GUARD_DIR"; then
            return 0
        fi
        warn "could not remove the bootstrap guard at $BOOTSTRAP_GUARD_DIR."
        return 1
    fi
    warn "not releasing $BOOTSTRAP_GUARD_DIR: it does not belong to this run."
    return 1
}

# Atomically publish source at the exact destination pathname.  Python's
# os.link calls link(2) directly, so it has no shell `ln source directory`
# container semantics, does not follow a destination symlink and never
# replaces an existing file, directory or symlink.
atomic_link_exact() {
    local source="$1" destination="$2"
    "$LINK_PYTHON_BIN" - "$source" "$destination" <<'PY' 2>/dev/null
import os
import sys

try:
    os.link(sys.argv[1], sys.argv[2], follow_symlinks=False)
except OSError:
    sys.exit(1)
PY
}

# Atomically publish one quarantine record.  Refuses to overwrite any
# pre-existing marker, malformed or not.
write_cleanup_failed_marker() {
    local pid="$1" service="$2" signature="$3" reason="$4" tmp
    local hook_rc=0

    [ -n "$pid" ] || return 1
    is_positive_integer "$pid" || return 1
    case "$service" in
        "$OLLAMA_SERVICE"|"$BACKEND_SERVICE") : ;;
        *) return 1 ;;
    esac
    [ -n "$signature" ] || return 1
    [ -n "$reason" ] || return 1

    if [ -e "$CLEANUP_FAILED_FILE" ] || [ -L "$CLEANUP_FAILED_FILE" ]; then
        warn "a cleanup-failed marker already exists at $CLEANUP_FAILED_FILE; refusing to replace it."
        return 1
    fi
    "$MKDIR_BIN" -p "$RUNTIME_DIR" || return 1

    # Fault injection for deterministic tests of the failure paths below.
    if [ -n "${TEACHABLE_TEST_FORCE_MARKER_TEMP_FAIL:-}" ]; then
        warn "injected cleanup-failed marker temporary-file failure."
        return 1
    fi

    if tmp="$(mktemp "$CLEANUP_FAILED_FILE.tmp.XXXXXX")"; then
        :
    else
        return 1
    fi
    if {
        printf '# Teachable Agent demo cleanup-failed marker; do not delete blindly.\n'
        printf 'PID=%s\n' "$pid"
        printf 'SERVICE=%s\n' "$service"
        printf 'PROJECT_ROOT=%s\n' "$PROJECT"
        printf 'CMD=%s\n' "$signature"
        printf 'RUN_ID=%s\n' "$RUN_ID"
        printf 'REASON=%s\n' "$reason"
    } > "$tmp"; then
        :
    else
        rm -f -- "$tmp"
        return 1
    fi

    # Deterministic test hook: runs after the complete temp file exists but
    # before the atomic no-clobber publish.  It may create the final marker
    # path itself; the publish below must then fail without touching it.
    if [ -n "${TEACHABLE_MARKER_PREPUBLISH_HOOK:-}" ]; then
        if "$TEACHABLE_MARKER_PREPUBLISH_HOOK" "$tmp" "$CLEANUP_FAILED_FILE"; then
            hook_rc=0
        else
            hook_rc=$?
        fi
        if [ "$hook_rc" -ne 0 ]; then
            rm -f -- "$tmp"
            return 1
        fi
    fi

    # True atomic no-clobber publication: ln creates a new directory entry
    # only if the final name does not exist.  A marker (or symlink) that
    # appears after the temporary write therefore wins; this process removes
    # only its own temporary file and never replaces the foreign marker.
    if atomic_link_exact "$tmp" "$CLEANUP_FAILED_FILE"; then
        rm -f -- "$tmp"
        return 0
    fi
    rm -f -- "$tmp"
    return 1
}

# Strict parser: every field must appear exactly once, unknown keys and
# malformed lines are rejected, and the PID must be a positive integer.
parse_cleanup_failed_marker() {
    local path="$1" line key value
    local pid_count=0 service_count=0 root_count=0
    local cmd_count=0 run_count=0 reason_count=0

    CLEANUP_FAILED_PID=""
    CLEANUP_FAILED_SERVICE=""
    CLEANUP_FAILED_ROOT=""
    CLEANUP_FAILED_CMD=""
    CLEANUP_FAILED_RUN_ID=""
    CLEANUP_FAILED_REASON=""
    [ -f "$path" ] || return 1

    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            \#*|'') continue ;;
        esac
        case "$line" in
            *=*) : ;;
            *) return 1 ;;
        esac
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            PID)
                CLEANUP_FAILED_PID="$value"; pid_count=$((pid_count + 1)) ;;
            SERVICE)
                CLEANUP_FAILED_SERVICE="$value"; service_count=$((service_count + 1)) ;;
            PROJECT_ROOT)
                CLEANUP_FAILED_ROOT="$value"; root_count=$((root_count + 1)) ;;
            CMD)
                CLEANUP_FAILED_CMD="$value"; cmd_count=$((cmd_count + 1)) ;;
            RUN_ID)
                CLEANUP_FAILED_RUN_ID="$value"; run_count=$((run_count + 1)) ;;
            REASON)
                CLEANUP_FAILED_REASON="$value"; reason_count=$((reason_count + 1)) ;;
            *)
                return 1 ;;
        esac
    done < "$path"

    [ "$pid_count" -eq 1 ] || return 1
    [ "$service_count" -eq 1 ] || return 1
    [ "$root_count" -eq 1 ] || return 1
    [ "$cmd_count" -eq 1 ] || return 1
    [ "$run_count" -eq 1 ] || return 1
    [ "$reason_count" -eq 1 ] || return 1
    is_positive_integer "$CLEANUP_FAILED_PID" || return 1
    case "$CLEANUP_FAILED_SERVICE" in
        "$OLLAMA_SERVICE"|"$BACKEND_SERVICE") : ;;
        *) return 1 ;;
    esac
    [ -n "$CLEANUP_FAILED_ROOT" ] || return 1
    [ -n "$CLEANUP_FAILED_CMD" ] || return 1
    [ -n "$CLEANUP_FAILED_RUN_ID" ] || return 1
    [ "$CLEANUP_FAILED_REASON" = "$CLEANUP_FAILED_REASON_TEXT" ] || return 1
    return 0
}

# Called after preflight and before the launcher lock or any service.  It
# never signals and never deletes an alive, unknown, malformed or foreign
# marker.
check_cleanup_failed_marker() {
    local marker_present=0 guard_present=0

    if [ -e "$CLEANUP_FAILED_FILE" ] || [ -L "$CLEANUP_FAILED_FILE" ]; then
        marker_present=1
    fi
    if [ -e "$BOOTSTRAP_GUARD_DIR" ] || [ -L "$BOOTSTRAP_GUARD_DIR" ]; then
        guard_present=1
    fi
    if [ "$marker_present" -eq 0 ] && [ "$guard_present" -eq 0 ]; then
        return 0
    fi

    # A symlinked marker or guard is never verified, removed or replaced:
    # its target is outside the launcher's ownership proof.
    if [ -L "$CLEANUP_FAILED_FILE" ]; then
        fail "a cleanup-failed marker at $CLEANUP_FAILED_FILE is a symlink."
        fail "Refusing to verify or remove it; refusing to start any service."
        exit 1
    fi
    if [ -L "$BOOTSTRAP_GUARD_DIR" ]; then
        fail "a bootstrap guard at $BOOTSTRAP_GUARD_DIR is a symlink."
        fail "Refusing to verify or remove it; refusing to start any service."
        exit 1
    fi

    # A guard always has to be verifiable and belong to this project before
    # any other transition is considered.
    if [ "$guard_present" -eq 1 ]; then
        if parse_bootstrap_guard "$BOOTSTRAP_GUARD_DIR"; then
            :
        else
            fail "a bootstrap guard exists at $BOOTSTRAP_GUARD_DIR but cannot be verified."
            fail "Refusing to start any service; nothing was removed."
            exit 1
        fi
        if [ "$GUARD_ROOT" != "$PROJECT" ]; then
            fail "a bootstrap guard belongs to another project root ('$GUARD_ROOT')."
            fail "Refusing to start any service; nothing was removed."
            exit 1
        fi
    fi

    # Guard without a complete record: a child may exist but no PID is known.
    # This is a fail-closed manual state; never remove or signal from it.
    if [ "$marker_present" -eq 0 ]; then
        fail "a bootstrap guard exists at $BOOTSTRAP_GUARD_DIR without a complete cleanup-failed record."
        fail "A child may have been started but its PID cannot be identified from the guard alone."
        fail "Refusing to start any service; inspect SERVICE=$GUARD_SERVICE RUN_ID=$GUARD_RUN_ID"
        fail "and remove the guard manually only after verifying its SERVICE/CMD/RUN_ID."
        exit 1
    fi

    if parse_cleanup_failed_marker "$CLEANUP_FAILED_FILE"; then
        :
    else
        fail "a cleanup-failed marker exists at $CLEANUP_FAILED_FILE but cannot be verified"
        fail "(malformed, incomplete, duplicated or with unknown fields)."
        fail "Refusing to start any service; the marker was preserved for inspection."
        exit 1
    fi
    if [ "$CLEANUP_FAILED_ROOT" != "$PROJECT" ]; then
        fail "a cleanup-failed marker belongs to another project root ('$CLEANUP_FAILED_ROOT')."
        fail "Refusing to start any service; the marker was preserved for inspection."
        exit 1
    fi

    # A guard and a marker may coexist after a crash between publishing the
    # complete record and removing the guard.  They must describe the same run
    # before either can be removed.
    if [ "$guard_present" -eq 1 ]; then
        if [ "$GUARD_SERVICE" != "$CLEANUP_FAILED_SERVICE" ] || \
           [ "$GUARD_ROOT" != "$CLEANUP_FAILED_ROOT" ] || \
           [ "$GUARD_CMD" != "$CLEANUP_FAILED_CMD" ] || \
           [ "$GUARD_RUN_ID" != "$CLEANUP_FAILED_RUN_ID" ]; then
            fail "a bootstrap guard and a cleanup-failed marker describe different runs."
            fail "Refusing to start any service; both were preserved for inspection."
            exit 1
        fi
    fi

    proc_view_into "$CLEANUP_FAILED_PID"
    case "$PROC_VIEW" in
        gone)
            info "Removing a stale cleanup-failed marker for PID $CLEANUP_FAILED_PID (trusted gone)."
            if rm -f -- "$CLEANUP_FAILED_FILE"; then
                :
            else
                fail "could not remove the stale cleanup-failed marker at $CLEANUP_FAILED_FILE."
                fail "Refusing to start any service."
                exit 1
            fi
            if [ "$guard_present" -eq 1 ]; then
                info "Removing the matching bootstrap guard left by the same run."
                if rm -rf -- "$BOOTSTRAP_GUARD_DIR"; then
                    :
                else
                    fail "could not remove the matching bootstrap guard at $BOOTSTRAP_GUARD_DIR."
                    fail "Refusing to start any service."
                    exit 1
                fi
            fi
            return 0
            ;;
    esac

    if [ "$PROC_VIEW" = "alive" ]; then
        fail "a cleanup-failed marker records PID $CLEANUP_FAILED_PID, which is still alive."
        fail "The previous run could not verify that this process stopped."
        fail "Refusing to start any service; no signal was sent and the marker was preserved."
        if [ "$guard_present" -eq 1 ]; then
            fail "The matching bootstrap guard was preserved too: $BOOTSTRAP_GUARD_DIR"
        fi
        fail "Inspect PID $CLEANUP_FAILED_PID and the marker: $CLEANUP_FAILED_FILE"
        exit 1
    fi

    fail "a cleanup-failed marker records PID $CLEANUP_FAILED_PID, whose state cannot be verified"
    fail "($PROC_VIEW_DETAIL). Refusing to start any service; no signal was sent and the"
    if [ "$guard_present" -eq 1 ]; then
        fail "marker and matching bootstrap guard were preserved: $BOOTSTRAP_GUARD_DIR"
    else
        fail "marker was preserved."
    fi
    fail "Inspect PID $CLEANUP_FAILED_PID and the marker: $CLEANUP_FAILED_FILE"
    exit 1
}

# Stop one service, choosing between the verified path (PID record exists)
# and the strictly limited bootstrap path (a child of this shell whose record
# was never written).  Returns 0 only when the process is confirmed gone.
stop_one_service() {
    local pid="$1" service="$2" signature="$3" timeout="$4" label="$5" \
          record_file="$6" recorded="$7" stored_fp="$8" stored_live_cmd="$9"
    local result

    if [ "$recorded" -eq 1 ]; then
        stop_owned_process "$pid" "$service" "$signature" "$timeout" "$label" \
            "$record_file" "$stored_fp" "$stored_live_cmd"
        return $?
    fi

    BOOT_DETAIL=""
    result=0
    stop_bootstrap_child "$pid" "$service" "$signature" "$timeout" "$label" \
        || result=$?
    if [ "$result" -ne 0 ]; then
        if [ -n "$BOOT_DETAIL" ]; then
            warn "$BOOT_DETAIL"
        fi
        warn "$label (PID $pid) could not be confirmed stopped; it may still be running."
        return 1
    fi
    return 0
}

cleanup() {
    local rc="${1:-0}" stop_rc backend_final=1 ollama_final=1
    trap - INT TERM EXIT
    stop_rc=0
    CLEANUP_FAILED=0

    if [ -n "$OWN_BACKEND_PID" ]; then
        if stop_one_service "$OWN_BACKEND_PID" "$BACKEND_SERVICE" \
                "$BACKEND_CMD_SIGNATURE" "$BACKEND_STOP_TIMEOUT" "FastAPI backend" \
                "$BACKEND_PID_FILE" "$BACKEND_RECORDED" "$BACKEND_RECORD_FP" \
                "$BACKEND_LIVE_CMD"; then
            BACKEND_CLEANUP_FAILED=0
            backend_final=1
            if [ "$BACKEND_GUARD_ACTIVE" -eq 1 ]; then
                if release_bootstrap_guard "$BACKEND_SERVICE" "$BACKEND_CMD_SIGNATURE"; then
                    BACKEND_GUARD_ACTIVE=0
                else
                    warn "the FastAPI child stopped, but its bootstrap guard remains at $BOOTSTRAP_GUARD_DIR; the next launch will refuse to start."
                    stop_rc=1
                fi
            fi
        else
            BACKEND_CLEANUP_FAILED=1
            backend_final=0
            stop_rc=1
            # No verifiable PID record may exist for a bootstrap child, so
            # persist a quarantine marker before the process can be lost.
            if [ "$BACKEND_RECORDED" -eq 0 ]; then
                if write_cleanup_failed_marker "$OWN_BACKEND_PID" \
                        "$BACKEND_SERVICE" "$BACKEND_CMD_SIGNATURE" \
                        "$CLEANUP_FAILED_REASON_TEXT"; then
                    if [ "$BACKEND_GUARD_ACTIVE" -eq 1 ]; then
                        if release_bootstrap_guard "$BACKEND_SERVICE" "$BACKEND_CMD_SIGNATURE"; then
                            BACKEND_GUARD_ACTIVE=0
                        else
                            warn "the full quarantine record exists, but the FastAPI bootstrap guard remains at $BOOTSTRAP_GUARD_DIR as extra protection."
                        fi
                    fi
                else
                    warn "could not persist the cleanup-failed marker for FastAPI backend (PID $OWN_BACKEND_PID)."
                    warn "the FastAPI bootstrap guard remains at $BOOTSTRAP_GUARD_DIR to block the next launch."
                fi
            fi
        fi
    fi
    if [ -n "$OWN_OLLAMA_PID" ]; then
        if stop_one_service "$OWN_OLLAMA_PID" "$OLLAMA_SERVICE" \
                "$OLLAMA_CMD_SIGNATURE" "$OLLAMA_STOP_TIMEOUT" "project-local Ollama" \
                "$OLLAMA_PID_FILE" "$OLLAMA_RECORDED" "$OLLAMA_RECORD_FP" ""; then
            ollama_final=1
            if [ "$OLLAMA_GUARD_ACTIVE" -eq 1 ]; then
                if release_bootstrap_guard "$OLLAMA_SERVICE" "$OLLAMA_CMD_SIGNATURE"; then
                    OLLAMA_GUARD_ACTIVE=0
                else
                    warn "the Ollama child stopped, but its bootstrap guard remains at $BOOTSTRAP_GUARD_DIR; the next launch will refuse to start."
                    stop_rc=1
                fi
            fi
        else
            ollama_final=0
            stop_rc=1
            if [ "$OLLAMA_RECORDED" -eq 0 ]; then
                if write_cleanup_failed_marker "$OWN_OLLAMA_PID" \
                        "$OLLAMA_SERVICE" "$OLLAMA_CMD_SIGNATURE" \
                        "$CLEANUP_FAILED_REASON_TEXT"; then
                    if [ "$OLLAMA_GUARD_ACTIVE" -eq 1 ]; then
                        if release_bootstrap_guard "$OLLAMA_SERVICE" "$OLLAMA_CMD_SIGNATURE"; then
                            OLLAMA_GUARD_ACTIVE=0
                        else
                            warn "the full quarantine record exists, but the Ollama bootstrap guard remains at $BOOTSTRAP_GUARD_DIR as extra protection."
                        fi
                    fi
                else
                    warn "could not persist the cleanup-failed marker for project-local Ollama (PID $OWN_OLLAMA_PID)."
                    warn "the Ollama bootstrap guard remains at $BOOTSTRAP_GUARD_DIR to block the next launch."
                fi
            fi
        fi
    fi

    # Only records this run created (verified by RUN_ID) are removed.  A
    # service that was confirmed stopped is persisted (its record is
    # removed) before its in-memory ownership is dropped.  A failed service
    # keeps its real PID for diagnostics and its record, if any, for retry.
    if [ "$backend_final" -eq 1 ]; then
        if remove_own_record "$BACKEND_PID_FILE"; then
            OWN_BACKEND_PID=""
        else
            warn "could not remove the PID record for FastAPI backend (PID $OWN_BACKEND_PID)."
            stop_rc=1
        fi
    fi
    if [ "$ollama_final" -eq 1 ]; then
        if remove_own_record "$OLLAMA_PID_FILE"; then
            OWN_OLLAMA_PID=""
        else
            warn "could not remove the PID record for project-local Ollama (PID $OWN_OLLAMA_PID)."
            stop_rc=1
        fi
    fi

    release_lock

    if [ "$stop_rc" -eq 1 ]; then
        CLEANUP_FAILED=1
        warn "At least one service may still be running; check the messages above."
        if [ -n "$OWN_BACKEND_PID" ]; then
            warn "FastAPI backend (PID $OWN_BACKEND_PID) is not confirmed stopped."
        fi
        if [ -n "$OWN_OLLAMA_PID" ]; then
            warn "project-local Ollama (PID $OWN_OLLAMA_PID) is not confirmed stopped."
        fi
        if [ "$rc" -eq 0 ]; then
            rc=1
        fi
    fi
    return "$rc"
}

on_signal() {
    say ""
    info "Stop requested - shutting down the services started by this launcher ..."
    if ! cleanup 0; then :; fi
    if [ "${CLEANUP_FAILED:-0}" = "1" ]; then
        warn "At least one service may still be running (see above)."
        exit 1
    fi
    info "Stopped. The terminal is free again."
    exit 0
}

# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

validate_positive_integer_setting() {
    local name="$1" env_name="$2" value="${3:-}"
    is_positive_integer "$value" || \
        die "invalid configuration: $name ($env_name) must be a strict positive integer (got '${value:-<empty>}')."
}

validate_positive_number_setting() {
    local name="$1" env_name="$2" value="${3:-}"
    is_positive_number "$value" || \
        die "invalid configuration: $name ($env_name) must be a strict positive number (got '${value:-<empty>}')."
}

validate_port_setting() {
    local name="$1" env_name="$2" value="${3:-}"
    is_valid_port "$value" || \
        die "invalid configuration: $name ($env_name) must be an integer between 1 and 65535 (got '${value:-<empty>}')."
}

validate_start_config() {
    validate_positive_integer_setting LSTART_RETRIES TEACHABLE_LSTART_RETRIES "${LSTART_RETRIES:-}"
    validate_positive_integer_setting SELF_LSTART_RETRIES TEACHABLE_SELF_LSTART_RETRIES "${SELF_LSTART_RETRIES:-}"
    validate_positive_integer_setting READY_ATTEMPTS TEACHABLE_READY_ATTEMPTS "${READY_ATTEMPTS:-}"
    validate_positive_integer_setting OLLAMA_STOP_TIMEOUT TEACHABLE_OLLAMA_STOP_TIMEOUT "${OLLAMA_STOP_TIMEOUT:-}"
    validate_positive_integer_setting BACKEND_STOP_TIMEOUT TEACHABLE_BACKEND_STOP_TIMEOUT "${BACKEND_STOP_TIMEOUT:-}"
    validate_positive_number_setting LSTART_INTERVAL TEACHABLE_LSTART_INTERVAL "${LSTART_INTERVAL:-}"
    validate_positive_number_setting SELF_LSTART_INTERVAL TEACHABLE_SELF_LSTART_INTERVAL "${SELF_LSTART_INTERVAL:-}"
    validate_positive_number_setting READY_INTERVAL TEACHABLE_READY_INTERVAL "${READY_INTERVAL:-}"
    validate_port_setting OLLAMA_PORT TEACHABLE_OLLAMA_PORT "${OLLAMA_PORT:-}"
    validate_port_setting APP_PORT TEACHABLE_APP_PORT "${APP_PORT:-}"
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

OLLAMA_STARTED_BY_US=0
BACKEND_STARTED_BY_US=0
BROWSER_OPENED=0

preflight() {
    [ -x "$PYTHON_BIN" ] || die "project virtualenv interpreter not found: $PYTHON_BIN" \
        "Create it with: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
    "$LINK_PYTHON_BIN" -c 'import os, sys' >/dev/null 2>&1 || \
        die "the atomic quarantine publisher interpreter is not usable: $LINK_PYTHON_BIN" \
            "A working Python interpreter is required for exact-path quarantine publication." \
            "Nothing was started; no process was touched."
    [ -f "$OLLAMA_LAUNCHER" ] || die "Ollama launcher not found: $OLLAMA_LAUNCHER" \
        "Expected the repository script scripts/start-local-ollama.sh."
    [ -x "$OLLAMA_LAUNCHER" ] || die "Ollama launcher is not executable: $OLLAMA_LAUNCHER" \
        "Fix it with: chmod +x \"$OLLAMA_LAUNCHER\""
    command -v "$CURL_BIN" >/dev/null 2>&1 || [ -x "$CURL_BIN" ] || \
        die "curl not found: $CURL_BIN" "macOS ships curl at /usr/bin/curl."
    command -v "$OPEN_BIN" >/dev/null 2>&1 || [ -x "$OPEN_BIN" ] || \
        die "the macOS 'open' command not found: $OPEN_BIN" \
            "Set TEACHABLE_OPEN_BIN to a browser opener, or open $APP_URL manually."
    command -v "$PS_BIN" >/dev/null 2>&1 || [ -x "$PS_BIN" ] || \
        die "ps not found: $PS_BIN" \
            "The demo needs ps to fingerprint every process it starts."
    # ps must be able to report this launcher's own start time: without that
    # the lock and the PID records could never be verified, so fail closed
    # here instead of half-starting.
    local self_fp
    self_fp="$(own_lstart_retry || true)"
    if [ -z "$self_fp" ]; then
        proc_view_into "$$"
        die "ps ($PS_BIN) cannot report this process's start time" \
            "A working ps is required to fingerprint the demo processes." \
            "Process inspection said: ${PROC_VIEW_DETAIL:-no diagnostic}" \
            "Nothing was started; no process was touched."
    fi
    "$MKDIR_BIN" -p "$RUNTIME_DIR" || die "cannot create runtime directory: $RUNTIME_DIR"
    "$MKDIR_BIN" -p "$LOG_DIR" || die "cannot create log directory: $LOG_DIR"
}

preflight_local_ollama() {
    local missing=""
    [ -f "$LOCAL_OLLAMA_CLI" ] && [ ! -L "$LOCAL_OLLAMA_CLI" ] && [ -x "$LOCAL_OLLAMA_CLI" ] \
        || missing="$LOCAL_OLLAMA_CLI"
    [ -z "$missing" ] || die "the project-local Ollama CLI is not runnable: $missing" \
        "Prepare the local runtime first (see docs/ollama-step2d-note.md and README)." \
        "Alternatively start a healthy Ollama yourself on $OLLAMA_HOST:$OLLAMA_PORT and rerun." \
        "Nothing was started; no process was touched."
    for d in "$LOCAL_OLLAMA_HOME" "$LOCAL_OLLAMA_MODELS" "$LOCAL_OLLAMA_TMP"; do
        [ -d "$d" ] && [ ! -L "$d" ] || die "required Ollama directory missing or is a symlink: $d" \
            "The launcher refuses to run with an unexpected local runtime layout."
    done
}

is_ollama_constructed() {
    [ "$OLLAMA_STARTED_BY_US" -eq 1 ] || [ "$EXTERNAL_OLLAMA" -eq 1 ]
}

# ---------------------------------------------------------------------------
# Startup steps
# ---------------------------------------------------------------------------

start_ollama() {
    check_port "$OLLAMA_PORT"
    if [ "$PORT_STATE" = "used" ]; then
        info "Port $OLLAMA_PORT is already in use (PID ${PORT_PID:-unknown}); checking whether it is a healthy Ollama ..."
        if ollama_version_ok; then
            EXTERNAL_OLLAMA=1
            info "Reusing the already-running Ollama at $OLLAMA_URL."
            info "It was NOT started by this launcher and will NOT be stopped by it."
            # Never create, overwrite or delete an ollama-demo.pid record for
            # an instance this launcher did not start.
            return 0
        fi
        die "port $OLLAMA_PORT is in use but $OLLAMA_URL/api/version did not answer like an Ollama server" \
            "Another program may be occupying the port." \
            "Inspect it with: lsof -nP -iTCP:$OLLAMA_PORT -sTCP:LISTEN" \
            "Nothing was started; no process was touched."
    fi

    preflight_local_ollama

    # Establish persistent protection before this shell can produce an
    # untracked child.  If the guard cannot be claimed, no child is started.
    if create_bootstrap_guard "$OLLAMA_SERVICE" "$OLLAMA_CMD_SIGNATURE"; then
        OLLAMA_GUARD_ACTIVE=1
    else
        die "could not establish the bootstrap guard for the Ollama process." \
            "No child was started; inspect any guard left at $BOOTSTRAP_GUARD_DIR."
    fi

    info "Starting the project-local Ollama on $OLLAMA_HOST:$OLLAMA_PORT ..."
    : > "$OLLAMA_LOG"
    bash "$OLLAMA_LAUNCHER" >>"$OLLAMA_LOG" 2>&1 &
    OLLAMA_STARTED_BY_US=1
    OWN_OLLAMA_PID=$!
    if [ -z "$OWN_OLLAMA_PID" ]; then
        fail "could not obtain the Ollama child PID after forking."
        if release_bootstrap_guard "$OLLAMA_SERVICE" "$OLLAMA_CMD_SIGNATURE"; then
            OLLAMA_GUARD_ACTIVE=0
        fi
        die "Startup aborted; no unmanaged child was left running."
    fi

    # Fingerprint the child before writing its record.  If the platform
    # cannot read the start time, we fail closed rather than write a record
    # that could not be verified later.
    OLLAMA_RECORD_FP="$(read_lstart_retry "$OWN_OLLAMA_PID" || true)"
    if [ -z "$OLLAMA_RECORD_FP" ]; then
        fail "could not read the process start fingerprint for the Ollama process."
        fail "No PID record was written, so the verified stop path cannot be used;"
        fail "falling back to the bootstrap-child cleanup for the child this shell"
        fail "just started (PID $OWN_OLLAMA_PID, still an unreaped direct child)."
        advise_logs
        if ! cleanup 1; then :; fi
        if [ "${CLEANUP_FAILED:-0}" = "1" ]; then
            die "Startup aborted and the Ollama process could NOT be confirmed stopped;" \
                "it may still be running. Inspect PID $OWN_OLLAMA_PID and the log above."
        fi
        die "Startup aborted; the freshly started Ollama process was stopped and reaped."
    fi

    if ! write_pid_record "$OLLAMA_PID_FILE" "$OWN_OLLAMA_PID" "$OLLAMA_SERVICE" \
            "$OLLAMA_CMD_SIGNATURE" "$OLLAMA_RECORD_FP"; then
        fail "could not write $OLLAMA_PID_FILE; aborting the start."
        advise_logs
        if ! cleanup 1; then :; fi
        if [ "${CLEANUP_FAILED:-0}" = "1" ]; then
            die "Startup aborted and the Ollama process could NOT be confirmed stopped;" \
                "it may still be running. Inspect PID $OWN_OLLAMA_PID and the log above."
        fi
        die "Nothing of this run was left running."
    fi
    OLLAMA_RECORDED=1

    # The child is now tracked by a formal PID record, so the pre-child guard
    # is no longer needed.  If it cannot be released, abort fail-closed.
    if release_bootstrap_guard "$OLLAMA_SERVICE" "$OLLAMA_CMD_SIGNATURE"; then
        OLLAMA_GUARD_ACTIVE=0
    else
        fail "the Ollama PID record exists but its bootstrap guard could not be released."
        advise_logs
        if cleanup 1; then :; fi
        die "Startup aborted; inspect the guard at $BOOTSTRAP_GUARD_DIR and the Ollama log."
    fi

    wait_for "the Ollama API" "$READY_ATTEMPTS" ollama_version_ok || {
        fail "Ollama did not become ready in time."
        advise_logs
        if ! cleanup 1; then :; fi
        die "Startup aborted; the Ollama process started by this launcher was stopped."
    }
}

start_backend() {
    check_port "$APP_PORT"
    if [ "$PORT_STATE" = "used" ]; then
        die "port $APP_PORT is already in use (PID ${PORT_PID:-unknown})" \
            "This launcher never kills an unknown process, and /api/health alone cannot prove the" \
            "listener belongs to this project." \
            "If it is a previous demo run, stop it with: bash scripts/stop-demo.sh" \
            "Otherwise inspect it with: lsof -nP -iTCP:$APP_PORT -sTCP:LISTEN"
    fi

    # Establish persistent protection before forking the backend child.
    if create_bootstrap_guard "$BACKEND_SERVICE" "$BACKEND_CMD_SIGNATURE"; then
        BACKEND_GUARD_ACTIVE=1
    else
        die "could not establish the bootstrap guard for the FastAPI process." \
            "No child was started; inspect any guard left at $BOOTSTRAP_GUARD_DIR."
    fi

    info "Starting FastAPI on $APP_URL ..."
    : > "$BACKEND_LOG"
    (
        cd -P "$PROJECT" || exit 1
        exec "$PYTHON_BIN" -m uvicorn "$APP_MODULE" \
            --host "$APP_HOST" \
            --port "$APP_PORT"
    ) >>"$BACKEND_LOG" 2>&1 &
    BACKEND_STARTED_BY_US=1
    OWN_BACKEND_PID=$!
    if [ -z "$OWN_BACKEND_PID" ]; then
        fail "could not obtain the FastAPI child PID after forking."
        if release_bootstrap_guard "$BACKEND_SERVICE" "$BACKEND_CMD_SIGNATURE"; then
            BACKEND_GUARD_ACTIVE=0
        fi
        die "Startup aborted; no unmanaged child was left running."
    fi

    BACKEND_LIVE_CMD="$(read_backend_live_cmd_retry "$OWN_BACKEND_PID" || true)"
    if [ -z "$BACKEND_LIVE_CMD" ]; then
        fail "could not read the FastAPI live command after exec."
        fail "No PID record was written, so the verified stop path cannot be used;"
        fail "falling back to the bootstrap-child cleanup for the child this shell"
        fail "just started (PID $OWN_BACKEND_PID, still an unreaped direct child)."
        advise_logs
        if cleanup 1; then :; fi
        if [ "${CLEANUP_FAILED:-0}" = "1" ]; then
            die "Startup aborted and the FastAPI process could NOT be confirmed stopped;" \
                "it may still be running. Inspect PID $OWN_BACKEND_PID and the log above."
        fi
        die "Startup aborted; the freshly started FastAPI process was stopped and reaped."
    fi

    BACKEND_RECORD_FP="$(read_lstart_retry "$OWN_BACKEND_PID" || true)"
    if [ -z "$BACKEND_RECORD_FP" ]; then
        fail "could not read the process start fingerprint for the FastAPI process."
        fail "No PID record was written, so the verified stop path cannot be used;"
        fail "falling back to the bootstrap-child cleanup for the child this shell"
        fail "just started (PID $OWN_BACKEND_PID, still an unreaped direct child)."
        advise_logs
        if ! cleanup 1; then :; fi
        if [ "${CLEANUP_FAILED:-0}" = "1" ]; then
            die "Startup aborted and the FastAPI process could NOT be confirmed stopped;" \
                "it may still be running. Inspect PID $OWN_BACKEND_PID and the log above."
        fi
        die "Startup aborted; the freshly started FastAPI process was stopped and reaped."
    fi

    if ! write_pid_record "$BACKEND_PID_FILE" "$OWN_BACKEND_PID" "$BACKEND_SERVICE" \
            "$BACKEND_CMD_SIGNATURE" "$BACKEND_RECORD_FP" "$BACKEND_LIVE_CMD"; then
        fail "could not write $BACKEND_PID_FILE; aborting the start."
        advise_logs
        if ! cleanup 1; then :; fi
        if [ "${CLEANUP_FAILED:-0}" = "1" ]; then
            die "Startup aborted and the FastAPI process could NOT be confirmed stopped;" \
                "it may still be running. Inspect PID $OWN_BACKEND_PID and the log above."
        fi
        die "Nothing of this run was left running."
    fi
    BACKEND_RECORDED=1

    if release_bootstrap_guard "$BACKEND_SERVICE" "$BACKEND_CMD_SIGNATURE"; then
        BACKEND_GUARD_ACTIVE=0
    else
        fail "the FastAPI PID record exists but its bootstrap guard could not be released."
        advise_logs
        if cleanup 1; then :; fi
        die "Startup aborted; inspect the guard at $BOOTSTRAP_GUARD_DIR and the backend log."
    fi

    wait_for "the FastAPI health endpoint" "$READY_ATTEMPTS" app_health_ok || {
        fail "FastAPI did not become healthy in time."
        advise_logs
        if ! cleanup 1; then :; fi
        die "Startup aborted; everything this launcher started was stopped."
    }
}

open_browser() {
    info "Opening $APP_URL in the browser ..."
    if "$OPEN_BIN" "$APP_URL" >/dev/null 2>&1; then
        BROWSER_OPENED=1
        return 0
    fi
    warn "could not open a browser automatically."
    warn "Both services are running - open $APP_URL manually."
    return 0
}

print_banner() {
    local ollama_status backend_status
    if [ "$EXTERNAL_OLLAMA" -eq 1 ]; then
        ollama_status="reused (already running, not owned by this launcher)"
    else
        ollama_status="started by this launcher (PID $OWN_OLLAMA_PID)"
    fi
    backend_status="started by this launcher (PID $OWN_BACKEND_PID)"
    say ""
    say "=============================================================="
    say " Teachable Agent demo is running"
    say "=============================================================="
    say " Ollama : $OLLAMA_URL  - $ollama_status"
    say " FastAPI: $APP_URL  - $backend_status"
    say " Web UI : $APP_URL"
    say " Health : $APP_HEALTH_URL"
    say " Logs   : $OLLAMA_LOG"
    say "          $BACKEND_LOG"
    if [ "$EXTERNAL_OLLAMA" -eq 1 ]; then
        say " Note   : the Ollama instance above was already running before this"
        say "          launcher started and will be left untouched."
    fi
    say " Stop   : press Ctrl+C in this terminal (stops only what this run"
    say "          started), or run: bash scripts/stop-demo.sh"
    say "=============================================================="
    say ""
    say " Logs   : truncated at the start of each run, so they always show"
    say "          the most recent attempt."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    local final_rc=0
    trap on_signal INT TERM
    trap 'cleanup $?' EXIT

    info "Project root: $PROJECT"
    validate_start_config
    preflight
    check_cleanup_failed_marker
    acquire_lock
    start_ollama
    start_backend
    open_browser
    print_banner

    info "Services are running. Press Ctrl+C to stop them."
    if [ "$BACKEND_STARTED_BY_US" -eq 1 ]; then
        wait "$OWN_BACKEND_PID" 2>/dev/null || true
    else
        while :; do "$SLEEP_BIN" 3600; done
    fi
    info "The FastAPI process exited on its own; shutting down."
    if ! cleanup 0; then
        final_rc=1
    fi
    if [ "$final_rc" -ne 0 ] || [ "${CLEANUP_FAILED:-0}" = "1" ]; then
        warn "The FastAPI process exited, but at least one managed service could not be"
        warn "confirmed stopped; see the messages above."
        exit 1
    fi
    info "All services started by this launcher have stopped."
    exit 0
}

main "$@"
