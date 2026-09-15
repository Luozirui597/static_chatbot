#!/bin/bash
#
# Stop the services started by scripts/start-demo.sh.
#
# Safety rules:
#   * only PID records written by the demo launcher are considered;
#   * a record is strictly parsed: PID, SERVICE, PROJECT_ROOT, CMD,
#     FINGERPRINT and RUN_ID must each appear exactly once;
#   * SERVICE must match the record file, PROJECT_ROOT must be exactly this
#     project, CMD must match the expected signature, and the live process
#     start time must equal the recorded FINGERPRINT - otherwise no signal
#     is ever sent and the record is kept for manual inspection;
#   * "command does not match" is NEVER treated as "process already
#     stopped": only a dead PID makes a record stale;
#   * SIGTERM first, SIGKILL only after a bounded wait and only for a
#     process that is still verifiably the same managed process;
#   * an Ollama instance that was already running before the launcher
#     started is never recorded here and is therefore never stopped;
#   * running this script twice is harmless.

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "$0")" && pwd)"
PROJECT="${TEACHABLE_PROJECT_ROOT:-$(cd -P "$SCRIPT_DIR/.." && pwd)}"

PYTHON_BIN="${TEACHABLE_PYTHON_BIN:-$PROJECT/.venv/bin/python}"
LOCAL_OLLAMA_CLI="${TEACHABLE_LOCAL_OLLAMA_CLI:-$PROJECT/local_llm/Ollama.app/Contents/Resources/ollama}"
RUNTIME_DIR="${TEACHABLE_RUNTIME_DIR:-$PROJECT/local_llm/run}"

OLLAMA_HOST="${TEACHABLE_OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${TEACHABLE_OLLAMA_PORT-11435}"
APP_HOST="${TEACHABLE_APP_HOST:-127.0.0.1}"
APP_PORT="${TEACHABLE_APP_PORT-8000}"
APP_MODULE="${TEACHABLE_APP_MODULE:-backend.main:app}"

# Process inspection.  The default is the system ps; the environment
# override exists so the test suite can supply a fixture-backed stub
# on machines where spawning ps is not permitted.
PS_BIN="${TEACHABLE_PS_BIN:-ps}"
SLEEP_BIN="${TEACHABLE_SLEEP_BIN:-sleep}"
CURL_BIN="${TEACHABLE_CURL_BIN:-curl}"
LSOF_BIN="${TEACHABLE_LSOF_BIN:-/usr/sbin/lsof}"

STOP_TIMEOUT="${TEACHABLE_STOP_TIMEOUT-30}"

OLLAMA_PID_FILE="$RUNTIME_DIR/ollama-demo.pid"
BACKEND_PID_FILE="$RUNTIME_DIR/backend-demo.pid"

OLLAMA_SERVICE="ollama"
BACKEND_SERVICE="backend"
OLLAMA_CMD_SIGNATURE="$LOCAL_OLLAMA_CLI serve"
BACKEND_CMD_SIGNATURE="$PYTHON_BIN -m uvicorn $APP_MODULE --host $APP_HOST --port $APP_PORT"

info() { printf '[stop] %s\n' "$*"; }
warn() { printf '[stop] WARNING: %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Utilities shared with scripts/start-demo.sh (kept in sync deliberately:
# stop-demo.sh must stay usable on its own)
# ---------------------------------------------------------------------------

is_positive_integer() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$1" -gt 0 ] 2>/dev/null
}

# Three-valued process observation (see start-demo.sh for the rationale):
#   alive   - ps reported a usable non-zombie state
#   gone    - ps ran successfully and reported nothing for this PID
#   unknown - ps unusable/missing, malformed PID, ps diagnostic, unusable
#             output, or an unverifiable zombie
# Only `gone` may justify deleting a record; `unknown` always fails closed.
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

    out_file="$(mktemp "${TMPDIR:-/tmp}/stop-ps-out.XXXXXX")" || {
        PS_FIELD_DETAIL="cannot create a temporary file for ps"
        return 0
    }
    err_file="$(mktemp "${TMPDIR:-/tmp}/stop-ps-err.XXXXXX")" || {
        rm -f -- "$out_file"
        PS_FIELD_DETAIL="cannot create a temporary file for ps"
        return 0
    }

    set +e
    "$ps_prog" -p "$pid" -o "$field" >"$out_file" 2>"$err_file"
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

is_valid_port() {
    local value="${1:-}"
    is_positive_integer "$value" || return 1
    if [ "$value" -lt 1 ] 2>/dev/null || [ "$value" -gt 65535 ] 2>/dev/null; then
        return 1
    fi
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

# Strictly parse a record: every field must appear exactly once.  Missing or
# duplicate fields make the record unverifiable; such records are never
# signalled and never deleted.
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
            *) continue ;;
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

# True only when the live process still matches the recorded identity:
# command line AND process start fingerprint.  If the platform cannot read
# the fingerprint, this fails closed.
identity_ok() {
    local pid="$1" service="$2" expected_cmd="$3" recorded_fp="$4"
    local state cmd want live_fp
    is_positive_integer "$pid" || return 1
    proc_view "$pid" >/dev/null
    [ "$PROC_VIEW_ALIVE" -eq 1 ] || return 1
    state="$PROC_STATE_VALUE"
    [ -n "$state" ] || return 1
    cmd="$(read_cmdline "$pid")"
    [ -n "$cmd" ] || return 1
    want="$(normalize_cmd "$expected_cmd")"
    case "$service" in
        "$OLLAMA_SERVICE"|"$BACKEND_SERVICE")
            case "$cmd" in *"$want") : ;; *) return 1 ;; esac
            ;;
        *) return 1 ;;
    esac
    live_fp="$(read_lstart "$pid")"
    [ -n "$live_fp" ] || return 1
    [ -n "$recorded_fp" ] || return 1
    [ "$live_fp" = "$recorded_fp" ] || return 1
    return 0
}

stop_managed() {
    local pid_file="$1" service="$2" signature="$3" label="$4"
    local pid recorded_root recorded_cmd recorded_fp i max_ticks view

    if [ ! -f "$pid_file" ]; then
        info "$label: no PID record ($pid_file) - nothing to stop."
        return 0
    fi

    if ! parse_pid_record "$pid_file"; then
        warn "$label: the PID record is malformed (missing, empty, or"
        warn "        duplicated PID/SERVICE/PROJECT_ROOT/CMD/FINGERPRINT/RUN_ID)."
        warn "        Refusing to signal anything; keeping the record for inspection:"
        warn "        $pid_file"
        return 1
    fi

    pid="$REC_PID"

    if ! is_positive_integer "$pid"; then
        # A malformed PID is an *unknown* state, not a stale record: no
        # signal is sent, the record is kept byte-for-byte, and the failure
        # is reported.  Only an explicit ps "gone" may delete a record.
        warn "$label: PID record value is not a positive integer (found '${pid:-<empty>}')."
        warn "        This is an unverifiable record, not a stale one: refusing to"
        warn "        signal anything and keeping it for inspection:"
        warn "        $pid_file"
        return 1
    fi

    if [ "$REC_SERVICE" != "$service" ]; then
        warn "$label: the record at $pid_file declares SERVICE=$REC_SERVICE,"
        warn "        which does not belong to this record file. Never signalling."
        warn "        Keeping the record for inspection."
        return 1
    fi

    if [ -z "$REC_ROOT" ] || [ "$REC_ROOT" != "$PROJECT" ]; then
        warn "$label: PID record belongs to another project root"
        warn "        ('${REC_ROOT:-<empty>}' != $PROJECT)."
        info "$label: leaving the record alone and not touching PID $pid."
        return 0
    fi

    if [ -z "$REC_CMD" ] || [ -z "$REC_FP" ] || [ -z "$REC_RUN_ID" ]; then
        warn "$label: the record has an empty CMD, FINGERPRINT or RUN_ID and"
        warn "        cannot be verified. Never signalling; keeping the record."
        return 1
    fi

    recorded_cmd="$(normalize_cmd "$REC_CMD")"
    if [ "$recorded_cmd" != "$(normalize_cmd "$signature")" ]; then
        warn "$label: the recorded CMD does not match the expected command"
        warn "        signature for this project. Never signalling; keeping the record."
        return 1
    fi

    # Three-state gate: only an explicit "gone" may delete the stale record.
    proc_view_into "$pid"
    view="$PROC_VIEW"
    case "$view" in
        gone)
            info "$label: PID $pid no longer exists; cleaning up the stale record."
            rm -f -- "$pid_file"
            return 0
            ;;
        unknown)
            warn "$label: cannot determine whether PID $pid exists ($PROC_VIEW_DETAIL)."
            warn "        Refusing to delete the record or signal anything."
            return 1
            ;;
    esac

    if ! identity_ok "$pid" "$service" "$signature" "$REC_FP"; then
        warn "$label: PID $pid is alive but is NOT the managed $service process"
        warn "        (command line or process start fingerprint does not match)."
        warn "        Refusing to signal it - the PID was probably reused."
        warn "        Keeping the record for manual inspection."
        return 1
    fi

    info "Stopping $label (PID $pid) with SIGTERM ..."
    if ! kill -TERM "$pid" 2>/dev/null; then
        warn "$label: SIGTERM could not be delivered to PID $pid."
    fi

    max_ticks=$((STOP_TIMEOUT * 10))
    i=0
    while [ "$i" -lt "$max_ticks" ]; do
        proc_view_into "$pid"
    view="$PROC_VIEW"
        case "$view" in
            gone) break ;;
            unknown)
                warn "$label: lost track of PID $pid ($PROC_VIEW_DETAIL)."
                warn "        Refusing to assume it stopped; keeping the PID record."
                return 1
                ;;
        esac
        identity_ok "$pid" "$service" "$signature" "$REC_FP" || break
        "$SLEEP_BIN" 0.1
        i=$((i + 1))
    done

    proc_view_into "$pid"
    view="$PROC_VIEW"
    if [ "$view" = "gone" ]; then
        info "$label: stopped."
        rm -f -- "$pid_file"
        return 0
    fi
    if [ "$view" = "unknown" ]; then
        warn "$label: cannot confirm that PID $pid stopped ($PROC_VIEW_DETAIL)."
        warn "        Keeping the PID record for retry and manual inspection."
        return 1
    fi

    if identity_ok "$pid" "$service" "$signature" "$REC_FP"; then
        warn "$label: still running after ${STOP_TIMEOUT}s; escalating to SIGKILL for PID $pid."
        kill -KILL "$pid" 2>/dev/null || true
        i=0
        while [ "$i" -lt 50 ]; do
            proc_view_into "$pid"
    view="$PROC_VIEW"
            case "$view" in
                gone) break ;;
                unknown)
                    warn "$label: lost track of PID $pid ($PROC_VIEW_DETAIL)."
                    warn "        Refusing to assume it stopped; keeping the PID record."
                    return 1
                    ;;
            esac
            identity_ok "$pid" "$service" "$signature" "$REC_FP" || break
            "$SLEEP_BIN" 0.1
            i=$((i + 1))
        done
    fi

    proc_view_into "$pid"
    view="$PROC_VIEW"
    if [ "$view" = "gone" ]; then
        info "$label: stopped."
        rm -f -- "$pid_file"
        return 0
    fi
    if [ "$view" = "unknown" ]; then
        warn "$label: cannot confirm that PID $pid stopped ($PROC_VIEW_DETAIL)."
        warn "        Keeping the PID record for retry and manual inspection."
        return 1
    fi

    warn "$label: PID $pid is still running and could not be stopped."
    warn "        Keeping the PID record for retry and manual inspection."
    return 1
}

report_leftovers() {
    local port pid
    for port in "$OLLAMA_PORT" "$APP_PORT"; do
        [ -x "$LSOF_BIN" ] || continue
        pid="$("$LSOF_BIN" -nP -a "-iTCP:$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
        [ -n "$pid" ] || continue
        if [ "$port" = "$APP_PORT" ]; then
            warn "port $APP_PORT is still in use by PID $pid."
            warn "If it is a leftover demo backend, inspect it with:"
            warn "  lsof -nP -iTCP:$APP_PORT -sTCP:LISTEN"
        else
            info "port $OLLAMA_PORT is still in use by PID $pid (not started by this launcher - left untouched)."
        fi
    done
}

validate_stop_config() {
    is_positive_integer "${STOP_TIMEOUT:-}" || {
        printf '[stop] ERROR: invalid configuration: STOP_TIMEOUT (TEACHABLE_STOP_TIMEOUT) must be a strict positive integer (got %s).\n' \
            "'${STOP_TIMEOUT:-<empty>}'" >&2
        exit 1
    }
    is_valid_port "${OLLAMA_PORT:-}" || {
        printf '[stop] ERROR: invalid configuration: OLLAMA_PORT (TEACHABLE_OLLAMA_PORT) must be an integer between 1 and 65535 (got %s).\n' \
            "'${OLLAMA_PORT:-<empty>}'" >&2
        exit 1
    }
    is_valid_port "${APP_PORT:-}" || {
        printf '[stop] ERROR: invalid configuration: APP_PORT (TEACHABLE_APP_PORT) must be an integer between 1 and 65535 (got %s).\n' \
            "'${APP_PORT:-<empty>}'" >&2
        exit 1
    }
}

main() {
    local rc=0
    validate_stop_config
    info "Project root: $PROJECT"

    stop_managed "$BACKEND_PID_FILE" "$BACKEND_SERVICE" "$BACKEND_CMD_SIGNATURE" "FastAPI backend" || rc=1
    stop_managed "$OLLAMA_PID_FILE" "$OLLAMA_SERVICE" "$OLLAMA_CMD_SIGNATURE" "project-local Ollama" || rc=1

    report_leftovers

    if [ "$rc" -eq 0 ]; then
        info "Done. Any Ollama that was already running before the demo keeps running."
    else
        warn "Some services could not be stopped; see the messages above."
    fi
    return "$rc"
}

main "$@"
