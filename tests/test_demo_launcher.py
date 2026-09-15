"""Tests for the one-command demo launcher and stopper.

Everything runs inside a throw-away sandbox under the system temp directory:

* no real Ollama runtime, model, or network call;
* no real browser (``open`` is a fake that records its call);
* no real uvicorn (a fake interpreter records readiness and then sleeps);
* ports are simulated through a fake ``lsof`` and a fake ``curl``;
* process inspection uses a fixture-backed ``ps`` because spawning the real
  ``ps`` is not permitted in every environment; the launcher's PID-identity
  logic itself is still the production code under test;
* no process outside the sandbox is ever signalled - the scripts under test
  only ever touch PIDs they recorded themselves.

The suite therefore exercises the real control flow of ``start-demo.sh`` /
``stop-demo.sh`` without depending on the machine's services.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

START_SCRIPT = REPO_ROOT / "scripts" / "start-demo.sh"
STOP_SCRIPT = REPO_ROOT / "scripts" / "stop-demo.sh"
START_COMMAND = REPO_ROOT / "Start Teachable Agent.command"
STOP_COMMAND = REPO_ROOT / "Stop Teachable Agent.command"

# A PID that cannot exist, used for stale-record tests.
DEAD_PID = "999999"

SECRET_TOKEN = "sk-demo-secret-should-never-be-printed-0123456789"

# Must match the fixed safety reasons in scripts/start-demo.sh.
BOOTSTRAP_GUARD_REASON_TEXT = (
    "bootstrap child lifecycle in progress; complete quarantine record not published"
)
CLEANUP_FAILED_REASON_TEXT = "bootstrap child could not be confirmed stopped"


# ---------------------------------------------------------------------------
# fake external commands
# ---------------------------------------------------------------------------

FAKE_CURL = r"""#!/bin/bash
# Fake curl: answers only the two readiness endpoints, driven by flag files.
set -uo pipefail
url=""
for arg in "$@"; do
    case "$arg" in
        http://*|https://*) url="$arg" ;;
    esac
done
[ -n "$url" ] || exit 2
case "$url" in
    */api/version)
        [ -f "$SANDBOX/state/ollama-ready" ] || exit 7
        printf '{"version":"0.0.0-fake"}\n'
        ;;
    */api/health)
        [ -f "$SANDBOX/state/backend-ready" ] || exit 7
        printf '{"status":"ok"}\n'
        ;;
    *)
        exit 7
        ;;
esac
exit 0
"""

FAKE_LSOF = r"""#!/bin/bash
# Fake lsof.  Default: a port is "listening" while a marker file exists for
# it (rc=0 + PID on stdout, empty stderr), otherwise "free" (rc=1, empty).
#
# An override file (state/lsof-override: "<rc>|<stdout>|<stderr>", with
# literal \n allowed as a newline escape) simulates the odd lsof answers
# the launcher must fail closed on.
set -uo pipefail
port=""
for arg in "$@"; do
    case "$arg" in
        -iTCP:*) port="${arg#-iTCP:}" ;;
    esac
done
[ -n "$port" ] || exit 2
override="$SANDBOX/state/lsof-override-$port"
if [ -f "$override" ]; then
    IFS='|' read -r rc out err < "$override"
    printf '%b' "$out"
    printf '%b' "$err" >&2
    exit "$rc"
fi
marker="$SANDBOX/state/listen-$port.pid"
if [ -f "$marker" ]; then
    cat "$marker"
    exit 0
fi
exit 1
"""

FAKE_OPEN = r"""#!/bin/bash
# Fake browser opener: records the URL, honours a forced-failure flag.
set -uo pipefail
printf '%s\n' "${1:-}" >> "$SANDBOX/state/open-calls"
[ -f "$SANDBOX/state/open-fails" ] && exit 1
exit 0
"""

FAKE_PS = r"""#!/bin/bash
# Fixture-backed ps: reports the processes the sandbox declares in
# state/ps-meta, one "PID|state|command|lstart" row per process.
# Unknown PIDs are reported as gone, exactly like the real command would.
set -uo pipefail
[ -f "$SANDBOX/state/ps-meta" ] || exit 1
pid=""
key=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "-p" ]; then
        pid="$arg"
    fi
    case "$arg" in
        command=) key="command=" ;;
        state=) key="state=" ;;
        lstart=) key="lstart=" ;;
    esac
    prev="$arg"
done
[ -n "$pid" ] && [ -n "$key" ] || exit 2
# Optional flags simulate a platform where ps cannot answer.
if [ "$key" = "lstart=" ] && [ -f "$SANDBOX/state/ps-lstart-fails" ]; then
    exit 1
fi
# Break lstart only for one specific pid (leaves the launcher's own
# fingerprint readable so the lock can still be acquired).
if [ "$key" = "lstart=" ] && [ -f "$SANDBOX/state/ps-lstart-fails-for" ]; then
    if [ "$(tr -d '[:space:]' < "$SANDBOX/state/ps-lstart-fails-for")" = "$pid" ]; then
        exit 1
    fi
fi
if [ "$key" = "state=" ] && [ -f "$SANDBOX/state/ps-state-fails" ]; then
    printf 'ps: illegal option or unusable state\n' >&2
    exit 1
fi
# Emit a *valid* value on stdout together with a diagnostic on stderr.
# Any stderr must make the answer unusable, so nothing may be trusted.
if [ -f "$SANDBOX/state/ps-field-stderr" ]; then
    want_stderr="$(tr -d '[:space:]' < "$SANDBOX/state/ps-field-stderr")"
    if [ "$want_stderr" = "all" ] || [ "$want_stderr" = "$key" ]; then
        printf 'ps: warning: something odd on stderr\n' >&2
    fi
fi
# A per-pid scripted state sequence: state/ps-state-seq holds
# "<pid>:<state>,<state>,..." and each state= probe returns the next entry
# (the last one repeats once the sequence is exhausted).  The token "-"
# means "report unknown" (ps diagnostic); the token "gone" means ps exits
# 1 with empty stdout/stderr, i.e. the documented "no such process".
injected_state_for() {
    local want="$1" entry states n item index
    [ -f "$SANDBOX/state/ps-state-seq" ] || return 1
    while IFS= read -r entry; do
        case "$entry" in ''|'#'*) continue ;; esac
        case "${entry%%:*}" in "$want") ;; *) continue ;; esac
        states="${entry#*:}"
        [ -n "$states" ] || return 1
        # The counter is the 1-based index of the item to return now.
        n=1
        if [ -f "$SANDBOX/state/ps-state-seq-count-$want" ]; then
            n="$(tr -d '[:space:]' < "$SANDBOX/state/ps-state-seq-count-$want")"
            case "$n" in ''|*[!0-9]*) n=1 ;; esac
            [ "$n" -ge 1 ] 2>/dev/null || n=1
        fi
        # Advance before returning, so the next call sees the next item.
        printf '%s\n' "$((n + 1))" > "$SANDBOX/state/ps-state-seq-count-$want"
        index=1
        local IFS=','
        for item in $states; do
            if [ "$index" -eq "$n" ]; then
                printf '%s\n' "$item"
                return 0
            fi
            index=$((index + 1))
        done
        # Past the end: repeat the final state (and keep advancing so a
        # later reset of the sequence cannot accidentally go backwards).
        for item in $states; do :; done
        printf '%s\n' "$item"
        return 0
    done < "$SANDBOX/state/ps-state-seq"
    return 1
}

if [ "$key" = "state=" ] && [ -f "$SANDBOX/state/ps-state-seq" ]; then
    injected="$(injected_state_for "$pid" || true)"
    if [ -n "$injected" ]; then
        if [ -f "$SANDBOX/state/ps-probe-log" ]; then
            printf '%s|%s|injected=%s\n' "$pid" "$key" "$injected" \
                >> "$SANDBOX/state/ps-probe-log"
        fi
        if [ "$injected" = "-" ]; then
            printf 'ps: injected unusable state\n' >&2
            exit 1
        fi
        if [ "$injected" = "gone" ]; then
            # rc=1 with no stdout and no stderr is the platform's documented
            # "no such process" answer, i.e. a trusted `gone` verdict.
            exit 1
        fi
        printf '%s\n' "$injected"
        exit 0
    fi
fi
# Ghost processes: always reported alive, even after the real process
# exits - used to exercise the "SIGKILL could not stop it" path.
if [ -f "$SANDBOX/state/ps-ghost" ]; then
    while IFS='|' read -r ghost_pid ghost_cmd ghost_lstart; do
        if [ "$ghost_pid" = "$pid" ]; then
            if [ -f "$SANDBOX/state/ps-probe-log" ]; then
                printf '%s|%s\n' "$ghost_pid" "$key" \
                    >> "$SANDBOX/state/ps-probe-log"
            fi
            if [ "$key" = "command=" ]; then
                printf '%s\n' "$ghost_cmd"
            elif [ "$key" = "lstart=" ]; then
                printf '%s\n' "$ghost_lstart"
            else
                printf 'S\n'
            fi
            exit 0
        fi
    done < "$SANDBOX/state/ps-ghost"
fi
while IFS='|' read -r meta_pid meta_state meta_cmd meta_lstart; do
    case "$meta_pid" in ''|'#'*) continue ;; esac
    if [ "$meta_pid" = "$pid" ]; then
        if [ -f "$SANDBOX/state/ps-probe-log" ]; then
            printf '%s|%s\n' "$meta_pid" "$key" \
                >> "$SANDBOX/state/ps-probe-log"
        fi
        if [ "$key" = "command=" ]; then
            printf '%s\n' "$meta_cmd"
        elif [ "$key" = "lstart=" ]; then
            printf '%s\n' "$meta_lstart"
        else
            printf '%s\n' "$meta_state"
        fi
        exit 0
    fi
done < "$SANDBOX/state/ps-meta"

# Guard-bound synchronous discovery.  The service shim publishes its PID,
# service and exact command immediately after start, and this fallback is
# active only while a matching bootstrap guard exists for the same service,
# command and project root.  It never waits for Python startup, fake backend
# main(), backend-started* files, the watcher or a sleep.
guard_root_dir="${TEACHABLE_RUNTIME_DIR:-$SANDBOX/project/local_llm/run}/bootstrap.guard"
guard_field() {
    local key="$1" count
    count="$(grep -c "^${key}=" "$guard_root_dir/info" 2>/dev/null || true)"
    if [ "$count" = "1" ]; then
        sed -n "s/^${key}=//p" "$guard_root_dir/info" | head -n 1
    fi
}
fallback_kind=""
fallback_pid=""
fallback_cmd=""
if [ -f "$guard_root_dir/info" ]; then
    guard_service="$(guard_field SERVICE)"
    guard_cmd="$(guard_field CMD)"
    guard_project="$(guard_field PROJECT_ROOT)"
    guard_reason="$(guard_field REASON)"
    expected_reason="bootstrap child lifecycle in progress; complete quarantine record not published"
    expected_project="${TEACHABLE_PROJECT_ROOT:-$SANDBOX/project}"
    if [ "$guard_project" = "$expected_project" ] && [ "$guard_reason" = "$expected_reason" ]; then
        case "$guard_service" in
            backend) hs_prefix="$SANDBOX/state/backend-bootstrap" ;;
            ollama) hs_prefix="$SANDBOX/state/ollama-bootstrap" ;;
            *) hs_prefix="" ;;
        esac
        if [ -n "$hs_prefix" ] && [ -f "$hs_prefix-pid" ] && [ -f "$hs_prefix-service" ] && [ -f "$hs_prefix-cmd" ]; then
            hs_pid="$(tr -d '[:space:]' < "$hs_prefix-pid")"
            hs_service="$(tr -d '[:space:]' < "$hs_prefix-service")"
            hs_cmd="$(cat "$hs_prefix-cmd")"
            if [ "$hs_pid" = "$pid" ] && [ "$hs_service" = "$guard_service" ] && [ "$hs_cmd" = "$guard_cmd" ] && kill -0 "$pid" 2>/dev/null; then
                if [ "$key" = "lstart=" ] && { [ -f "$SANDBOX/state/ps-lstart-fails-for-bootstrap-$guard_service" ] || [ -f "$SANDBOX/state/ps-lstart-fails-for-bootstrap" ]; }; then
                    if [ -f "$SANDBOX/state/ps-probe-log" ]; then
                        printf '%s|%s|guard-lstart-fault\n' "$pid" "$key" \
                            >> "$SANDBOX/state/ps-probe-log"
                    fi
                    exit 1
                fi
                fallback_kind="$guard_service"
                fallback_pid="$pid"
                fallback_cmd="$guard_cmd"
            fi
        fi
    fi
fi
if [ -n "$fallback_kind" ]; then
    if [ -f "$SANDBOX/state/ps-probe-log" ]; then
        printf '%s|%s|guard-discovery=%s\n' "$pid" "$key" "$fallback_kind" \
            >> "$SANDBOX/state/ps-probe-log"
    fi
    if [ "$key" = "command=" ]; then
        printf '%s\n' "$fallback_cmd"
    elif [ "$key" = "lstart=" ]; then
        printf 'Thu Sep 14 12:00:%.2d 2026\n' "$((fallback_pid % 60))"
    else
        printf 'S\n'
    fi
    exit 0
fi

exit 1
"""

FAKE_PYTHON = r'''#!/usr/bin/env python3
"""Fake project interpreter.

* ``-m uvicorn ... --port N`` -> acts as the backend: announces readiness,
  then serves until terminated.
* anything else (e.g. ``-c``)   -> delegates to the real interpreter.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SANDBOX = Path(os.environ["SANDBOX"])
STATE = SANDBOX / "state"


def main() -> int:
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "-m" and argv[1] == "uvicorn":
        gate = os.environ.get("TEACHABLE_FAKE_BACKEND_STARTED_GATE", "")
        if gate:
            gate_timeout = float(
                os.environ.get(
                    "TEACHABLE_FAKE_BACKEND_STARTED_GATE_TIMEOUT", "60"
                )
            )
            gate_deadline = time.time() + gate_timeout
            while not os.path.exists(gate) and time.time() < gate_deadline:
                time.sleep(0.02)
        port = None
        for i, item in enumerate(argv):
            if item == "--port" and i + 1 < len(argv):
                port = argv[i + 1]
        if port is None:
            return 2
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "backend-started").write_text(
            " ".join(argv) + "\n", encoding="utf-8"
        )
        (STATE / "backend-started.pid").write_text(
            str(os.getpid()) + "\n", encoding="utf-8"
        )
        print("fake uvicorn serving " + " ".join(argv), flush=True)
        (STATE / "backend-never-ready").exists() or (
            STATE / "backend-ready"
        ).write_text("ready\n", encoding="utf-8")
        while True:
            time.sleep(0.2)
    return subprocess.call([sys.executable] + argv)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
'''


FAKE_PYTHON_SHIM_BODY = r'''# Lightweight process shim: publish the bootstrap handshake immediately,
# before any Python interpreter startup or business code, then exec the
# real fake interpreter helper.
set -u
printf '%s\n' "$$" > "$SANDBOX/state/backend-bootstrap-pid"
printf 'backend\n' > "$SANDBOX/state/backend-bootstrap-service"
printf '%s %s\n' "$0" "$*" > "$SANDBOX/state/backend-bootstrap-cmd"
exec "$SANDBOX/fakebin/python-helper" "$@"
'''


FAKE_OLLAMA_CLI = r"""#!/bin/bash
# Fake project-local Ollama CLI: serves /api/version until terminated.
set -uo pipefail
printf 'fake ollama cli started: %s\n' "$*"
printf '%s\n' "$$" > "$SANDBOX/state/fake-ollama.pid"
printf '%s\n' "$$" > "$SANDBOX/state/ollama-bootstrap-pid"
printf 'ollama\n' > "$SANDBOX/state/ollama-bootstrap-service"
printf '%s serve\n' "$0" > "$SANDBOX/state/ollama-bootstrap-cmd"
if [ ! -f "$SANDBOX/state/ollama-never-ready" ]; then
    printf 'ready\n' > "$SANDBOX/state/ollama-ready"
fi
cleanup() {
    printf '%s\n' "$$" >> "$SANDBOX/state/ollama-term"
    exit 0
}
trap cleanup TERM INT
while :; do
    sleep 0.2
done
"""

FAKE_OLLAMA_LAUNCHER = r"""#!/bin/bash
# Mirrors scripts/start-local-ollama.sh: execs the CLI so the launcher's
# recorded PID is the CLI process itself.
set -euo pipefail
CLI="$TEACHABLE_LOCAL_OLLAMA_CLI"
exec env \
    HOME="$TEACHABLE_LOCAL_OLLAMA_HOME" \
    OLLAMA_MODELS="$TEACHABLE_LOCAL_OLLAMA_MODELS" \
    TMPDIR="$TEACHABLE_LOCAL_OLLAMA_TMP" \
    OLLAMA_HOST="127.0.0.1:${TEACHABLE_OLLAMA_PORT:-11435}" \
    OLLAMA_NO_CLOUD=1 \
    "$CLI" serve
"""
# Bootstrap-path test doubles.  The holding launcher publishes the PID of
# the direct child the production script obtained from its background fork,
# then waits until the test has installed the failing ps fingerprint and
# the scripted state sequence before it execs the service CLI.  That makes
# the bootstrap window deterministic without any real ps, network, or model
# access.
_BASH_SHEBANG = "#" + chr(33) + "/bin/bash" + "\n"
FAKE_HOLDING_OLLAMA_LAUNCHER = _BASH_SHEBANG + r"""# Test double: fork,
# publish the PID, wait for test setup, then become the service CLI.
set -euo pipefail
printf '%s\n' "$$" > "$SANDBOX/state/bootstrap-ollama-shell.pid"
: > "$SANDBOX/state/bootstrap-ollama-shell-held"
until [ -f "$SANDBOX/state/bootstrap-ollama-release" ]; do
    sleep 0.02
done
CLI="$TEACHABLE_LOCAL_OLLAMA_CLI"
exec env \
    HOME="$TEACHABLE_LOCAL_OLLAMA_HOME" \
    OLLAMA_MODELS="$TEACHABLE_LOCAL_OLLAMA_MODELS" \
    TMPDIR="$TEACHABLE_LOCAL_OLLAMA_TMP" \
    OLLAMA_HOST="127.0.0.1:11435" \
    OLLAMA_NO_CLOUD=1 \
    "$CLI" serve
"""
FAKE_EXITING_OLLAMA_CLI = _BASH_SHEBANG + r"""# Exits immediately
# after exec, so the launcher shell holds a real unreaped zombie child.
set -euo pipefail
printf '%s\n' "$$" > "$SANDBOX/state/fake-ollama.pid"
exit 0
"""
FAKE_STUBBORN_OLLAMA_CLI = _BASH_SHEBANG + r"""# Ignores SIGTERM and stays
# alive; only the test's own SIGKILL removes it.  Used to prove that an
# unknown state fails closed without a SIGKILL and without an unbounded wait.
set -uo pipefail
printf '%s\n' "$$" > "$SANDBOX/state/fake-ollama.pid"
trap '' TERM
while :; do
    sleep 0.05
done
"""





def _write_executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


class Sandbox:
    """A disposable project root wired to fake external commands."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / "state"
        self.bin = root / "fakebin"
        self.project = root / "project"
        self.runtime = self.project / "local_llm" / "run"
        self.logs = self.project / "local_llm" / "logs"
        self.declared: dict[int, tuple[str, str, str]] = {}
        # Processes owned by the test harness itself (the launcher bash under
        # test).  They are always visible in the ps fixture because the
        # production launcher verifies its own start fingerprint.
        self.harness_pids: dict[int, str] = {}
        # When true, sync_ps_meta() does not publish the auto-discovered
        # service rows; the fake ps must answer those directly from the
        # service-published metadata for the happy path to pass.
        self.suppress_service_meta = False
        self._lock = threading.Lock()

    # -- construction ------------------------------------------------------

    def build(self) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        self.project.mkdir(parents=True, exist_ok=True)

        (self.project / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(START_SCRIPT, self.project / "scripts" / "start-demo.sh")
        shutil.copy2(STOP_SCRIPT, self.project / "scripts" / "stop-demo.sh")
        os.chmod(self.project / "scripts" / "start-demo.sh", 0o755)
        os.chmod(self.project / "scripts" / "stop-demo.sh", 0o755)

        (self.project / ".env").write_text(
            "LLM_MODE=fake\n"
            f"LLM_API_KEY={SECRET_TOKEN}\n"
            "LLM_API_BASE_URL=https://example.invalid/v1\n",
            encoding="utf-8",
        )

        _write_executable(self.bin / "curl", FAKE_CURL)
        _write_executable(self.bin / "lsof", FAKE_LSOF)
        _write_executable(self.bin / "open", FAKE_OPEN)
        _write_executable(self.bin / "ps", FAKE_PS)
        _write_executable(self.bin / "python-helper", FAKE_PYTHON)
        _write_executable(
            self.project / ".venv" / "bin" / "python",
            _BASH_SHEBANG + FAKE_PYTHON_SHIM_BODY,
        )
        _write_executable(
            self.project / "scripts" / "start-local-ollama.sh",
            FAKE_OLLAMA_LAUNCHER,
        )
        _write_executable(
            self.project / "local_llm" / "Ollama.app" / "Contents" /
            "Resources" / "ollama",
            FAKE_OLLAMA_CLI,
        )
        for name in ("runtime-home", "models", "tmp"):
            (self.project / "local_llm" / name).mkdir(
                parents=True, exist_ok=True
            )
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.state / "ps-meta").write_text("", encoding="utf-8")

    # -- state helpers -----------------------------------------------------

    def flag(self, name: str) -> None:
        (self.state / name).write_text("1\n", encoding="utf-8")

    def unflag(self, name: str) -> None:
        (self.state / name).unlink(missing_ok=True)

    def has_flag(self, name: str) -> bool:
        return (self.state / name).exists()

    def listen_marker(self, port: int, pid: int | str) -> None:
        (self.state / f"listen-{port}.pid").write_text(
            f"{pid}\n", encoding="utf-8"
        )

    def clear_listen_marker(self, port: int) -> None:
        (self.state / f"listen-{port}.pid").unlink(missing_ok=True)

    def write_pid_record(
        self,
        service: str,
        pid: int | str,
        signature: str | None = None,
        project_root: str | None = None,
        fingerprint: str | None = None,
        run_id: str | None = None,
    ) -> Path:
        self.runtime.mkdir(parents=True, exist_ok=True)
        if service == "ollama":
            path = self.runtime / "ollama-demo.pid"
            default_sig = (
                f"{self.project}/local_llm/Ollama.app/Contents/Resources/"
                "ollama serve"
            )
        else:
            path = self.runtime / "backend-demo.pid"
            default_sig = (
                f"{self.project}/.venv/bin/python -m uvicorn "
                "backend.main:app --host 127.0.0.1 --port 8000"
            )
        if fingerprint is None and isinstance(pid, int):
            fingerprint = self._lstart_for(pid)
        if fingerprint is None:
            fingerprint = "unknown-fingerprint"
        lines = [
            "# Teachable Agent demo launcher record; safe to delete.",
            f"PID={pid}",
            f"SERVICE={service}",
            "PROJECT_ROOT="
            + (project_root if project_root is not None else str(self.project)),
            f"CMD={signature if signature is not None else default_sig}",
            f"FINGERPRINT={fingerprint}",
            f"RUN_ID={run_id if run_id is not None else 'test-run-id'}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def break_lstart_for(self, pid: int) -> None:
        """Make the ps fixture unable to report this pid's start time."""
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "ps-lstart-fails-for").write_text(
            f"{pid}\n", encoding="utf-8"
        )

    def clear_break_lstart_for(self) -> None:
        (self.state / "ps-lstart-fails-for").unlink(missing_ok=True)

    def break_bootstrap_lstart(self, service: str) -> None:
        """Make guard-bound discovery fail lstart probes for one service."""
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / f"ps-lstart-fails-for-bootstrap-{service}").write_text(
            "1\n", encoding="utf-8"
        )

    def clear_break_bootstrap_lstart(self, service: str) -> None:
        (self.state / f"ps-lstart-fails-for-bootstrap-{service}").unlink(
            missing_ok=True
        )
        (self.state / "ps-lstart-fails-for-bootstrap").unlink(missing_ok=True)

    def set_ps_field_stderr(self, which: str = "all") -> None:
        """Make the ps stub write a diagnostic on stderr for a field."""
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "ps-field-stderr").write_text(
            f"{which}\n", encoding="utf-8"
        )

    def clear_ps_field_stderr(self) -> None:
        (self.state / "ps-field-stderr").unlink(missing_ok=True)

    def set_ps_state_sequence(self, pid: int | str, states: str) -> None:
        """Install a deterministic state= sequence for one PID.

        ``states`` is a comma-separated list returned by the successive
        state= probes; the last item repeats.  ``-`` reports unknown and
        ``gone`` reports the trusted no-such-process answer.  Installing a
        sequence resets its counter, so the next probe gets the first item.
        """
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "ps-state-seq").write_text(
            f"{pid}:{states}\n", encoding="utf-8"
        )
        (self.state / f"ps-state-seq-count-{pid}").unlink(missing_ok=True)

    def clear_ps_state_sequence(self) -> None:
        """Remove every scripted state sequence and its per-PID counter."""
        for path in self.state.glob("ps-state-seq*"):
            path.unlink(missing_ok=True)

    def set_ps_probe_log(self) -> None:
        """Enable the stub's probe log and truncate it."""
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "ps-probe-log").write_text("", encoding="utf-8")

    def ps_probe_log(self) -> list[str]:
        """Every state/identity probe recorded since set_ps_probe_log()."""
        path = self.state / "ps-probe-log"
        if not path.exists():
            return []
        return [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def ghost_probe_log(self) -> list[str]:
        path = self.state / "ps-probe-log"
        if not path.exists():
            return []
        return [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def reset_ghost_probe_log(self) -> None:
        (self.state / "ps-probe-log").write_text("", encoding="utf-8")
        (self.state / "ps-ghost-cycle-count").unlink(missing_ok=True)

    def set_lsof_override(self, port: int, rc: int, out: str, err: str) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / f"lsof-override-{port}").write_text(
            f"{rc}|{out}|{err}\n", encoding="utf-8"
        )

    def clear_lsof_override(self, port: int) -> None:
        (self.state / f"lsof-override-{port}").unlink(missing_ok=True)

    def raw_pid_record(self, service: str, text: str) -> Path:
        """Write a hand-crafted record for the strict-validation tests."""
        path = self.runtime / f"{service}-demo.pid"
        path.write_text(text, encoding="utf-8")
        return path

    def read_record(self, service: str) -> str:
        path = self.runtime / f"{service}-demo.pid"
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def record_field(self, service: str, key: str) -> str | None:
        for line in self.read_record(service).splitlines():
            if line.startswith(f"{key}="):
                return line[len(key) + 1:]
        return None

    def record_exists(self, service: str) -> bool:
        return (self.runtime / f"{service}-demo.pid").exists()

    def cleanup_failed_path(self) -> Path:
        return self.runtime / "cleanup-failed"

    def bootstrap_guard_path(self) -> Path:
        return self.runtime / "bootstrap.guard"

    def write_bootstrap_guard(
        self,
        service: str = "ollama",
        project_root: str | None = None,
        cmd: str | None = None,
        run_id: str | None = None,
        reason: str | None = None,
        info_text: str | None = None,
    ) -> Path:
        """Create a bootstrap guard directory, optionally with raw info."""
        path = self.bootstrap_guard_path()
        path.mkdir(parents=True, exist_ok=True)
        if info_text is not None:
            (path / "info").write_text(info_text, encoding="utf-8")
            return path
        if cmd is None:
            if service == "ollama":
                cmd = (
                    f"{self.project}/local_llm/Ollama.app/Contents/"
                    "Resources/ollama serve"
                )
            else:
                cmd = (
                    f"{self.project}/.venv/bin/python -m uvicorn "
                    "backend.main:app --host 127.0.0.1 --port 8000"
                )
        lines = [
            "# Teachable Agent bootstrap guard; remove only after inspection.",
            f"SERVICE={service}",
            "PROJECT_ROOT="
            + (project_root if project_root is not None else str(self.project)),
            f"CMD={cmd}",
            f"RUN_ID={run_id if run_id is not None else 'guard-run'}",
            f"REASON={reason if reason is not None else BOOTSTRAP_GUARD_REASON_TEXT}",
        ]
        (path / "info").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_cleanup_failed_marker(
        self,
        pid: int | str,
        service: str = "ollama",
        project_root: str | None = None,
        cmd: str | None = None,
        run_id: str | None = None,
        reason: str | None = None,
        text: str | None = None,
    ) -> Path:
        """Write a quarantine marker; text bypasses the standard shape."""
        self.runtime.mkdir(parents=True, exist_ok=True)
        path = self.cleanup_failed_path()
        if text is not None:
            path.write_text(text, encoding="utf-8")
            return path
        if cmd is None:
            if service == "ollama":
                cmd = (
                    f"{self.project}/local_llm/Ollama.app/Contents/"
                    "Resources/ollama serve"
                )
            else:
                cmd = (
                    f"{self.project}/.venv/bin/python -m uvicorn "
                    "backend.main:app --host 127.0.0.1 --port 8000"
                )
        lines = [
            "# Teachable Agent demo cleanup-failed marker; do not delete blindly.",
            f"PID={pid}",
            f"SERVICE={service}",
            "PROJECT_ROOT="
            + (project_root if project_root is not None else str(self.project)),
            f"CMD={cmd}",
            f"RUN_ID={run_id if run_id is not None else 'quarantine-run'}",
            f"REASON={reason if reason is not None else CLEANUP_FAILED_REASON_TEXT}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def set_ghost_process(self, pid: int, command: str,
                          lstart: str | None = None) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "ps-ghost").write_text(
            f"{pid}|{command}|{lstart or self._lstart_for(pid)}\n",
            encoding="utf-8",
        )

    def clear_ghost_processes(self) -> None:
        (self.state / "ps-ghost").unlink(missing_ok=True)

    def read_open_calls(self) -> list[str]:
        path = self.state / "open-calls"
        if not path.exists():
            return []
        return [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def log_text(self, name: str) -> str:
        path = self.logs / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    # -- process liveness --------------------------------------------------
    #
    # A sandbox process is reported to the fake ps only while it is really
    # running.  Zombies are reaped by a background thread so that "exited"
    # can never be mistaken for "alive" (os.kill(pid, 0) succeeds for a
    # zombie, and the real ps would report it as Z).

    def _read_pid(self, path: Path) -> int | None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("PID="):
                value = line[len("PID="):].strip()
                return int(value) if value.isdigit() else None
            if line.isdigit():
                return int(line)
        return None

    def fake_ollama_pid(self) -> int | None:
        return self._read_pid(self.state / "fake-ollama.pid")

    def backend_pid(self) -> int | None:
        return self._read_pid(self.state / "backend-started.pid")

    @staticmethod
    def _lstart_for(pid: int) -> str:
        # Deterministic, stable, and unique per PID so the launcher can
        # fingerprint a process and match it again later.
        return f"Thu Sep 14 12:00:{pid % 60:02d} 2026"

    def declare_harness_process(self, pid: int, command: str) -> None:
        """Expose a process owned by the test harness to the ps fixture."""
        with self._lock:
            self.harness_pids[pid] = command
        self.sync_ps_meta()

    def forget_harness_process(self, pid: int) -> None:
        with self._lock:
            self.harness_pids.pop(pid, None)
        self.sync_ps_meta()

    def declare_process(self, pid: int, command: str, state: str = "S") -> None:
        with self._lock:
            self.declared[pid] = (state, command, self._lstart_for(pid))
        self.sync_ps_meta()

    def clear_declared(self) -> None:
        with self._lock:
            self.declared.clear()
        self.sync_ps_meta()

    @staticmethod
    def _pid_is_live(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _declared_pid_is_live(pid: int) -> bool:
        """Liveness for test-owned (declared) children: reap them here so a
        zombie is correctly treated as gone."""
        try:
            reaped, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            reaped = 0
        except OSError:
            reaped = 0
        if reaped == pid:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def sync_ps_meta(self) -> None:
        """Publish the currently live sandbox processes to the ps stub."""
        entries: list[str] = []
        if not self.suppress_service_meta:
            ollama_pid = self.fake_ollama_pid()
            if ollama_pid and self._pid_is_live(ollama_pid):
                entries.append(
                    f"{ollama_pid}|S|"
                    f"{self.project}/local_llm/Ollama.app/Contents/Resources/"
                    f"ollama serve|{self._lstart_for(ollama_pid)}"
                )
            if ollama_pid and not self._pid_is_live(ollama_pid):
                (self.state / "ollama-gone").write_text("1\n", encoding="utf-8")

            backend_pid = self.backend_pid()
            if backend_pid and self._pid_is_live(backend_pid):
                started_path = self.state / "backend-started"
                started = ""
                if started_path.exists():
                    started = started_path.read_text(encoding="utf-8").strip()
                if started:
                    entries.append(
                        f"{backend_pid}|S|{self.project}/.venv/bin/python "
                        f"{started}|{self._lstart_for(backend_pid)}"
                    )

        with self._lock:
            declared = dict(self.declared)
            harness = dict(self.harness_pids)
        for pid, command in harness.items():
            entries.append(f"{pid}|S|{command}|{self._lstart_for(pid)}")
        for pid, (proc_state, command, proc_lstart) in declared.items():
            if self._declared_pid_is_live(pid):
                entries.append(f"{pid}|{proc_state}|{command}|{proc_lstart}")
            else:
                with self._lock:
                    self.declared.pop(pid, None)

        self.state.mkdir(parents=True, exist_ok=True)
        # Unique temp name: several watcher threads may sync concurrently.
        tmp = self.state / f"ps-meta.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_text(
            "\n".join(entries) + ("\n" if entries else ""), encoding="utf-8"
        )
        os.replace(tmp, self.state / "ps-meta")

    def ollama_terminated_pids(self, timeout: float = 5.0) -> list[int]:
        """PIDs the fake Ollama recorded before exiting (written by its trap)."""
        path = self.state / "ollama-term"
        deadline = time.time() + timeout
        while True:
            found: list[int] = []
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.isdigit():
                        found.append(int(line))
            if found or time.time() > deadline:
                return found
            time.sleep(0.05)

    def terminate(self, pid: int, sig: int = signal.SIGKILL) -> None:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def wait_for_pid_exit(self, pid: int, timeout: float = 15) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._pid_is_live(pid):
                return True
            time.sleep(0.05)
        return False

    def wait_for_backend(self, timeout: float = 40) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.has_flag("backend-ready") and self.fake_ollama_pid():
                return
            time.sleep(0.05)
        raise AssertionError("launcher never reached the ready state")

    def kill_all(self) -> None:
        for pid in (self.fake_ollama_pid(), self.backend_pid()):
            if pid:
                self.terminate(pid)
        for pid in (self.fake_ollama_pid(), self.backend_pid()):
            if pid:
                self.wait_for_pid_exit(pid, timeout=10)
        self.clear_declared()
        self.sync_ps_meta()

    # -- launching ---------------------------------------------------------

    def env(self, **overrides: str) -> dict[str, str]:
        env = dict(os.environ)
        env.pop("TEACHABLE_PROJECT_ROOT", None)
        env.update(
            {
                "SANDBOX": str(self.root),
                "TEACHABLE_PROJECT_ROOT": str(self.project),
                "TEACHABLE_PYTHON_BIN": str(
                    self.project / ".venv" / "bin" / "python"
                ),
                # The production script publishes quarantine records with
                # Python os.link; tests must use the real interpreter even
                # when they replace the fake project python with a shell
                # stub.
                "TEACHABLE_LINK_PYTHON": sys.executable,
                "TEACHABLE_OLLAMA_LAUNCHER": str(
                    self.project / "scripts" / "start-local-ollama.sh"
                ),
                "TEACHABLE_LOCAL_OLLAMA_CLI": str(
                    self.project / "local_llm" / "Ollama.app" / "Contents" /
                    "Resources" / "ollama"
                ),
                "TEACHABLE_LOCAL_OLLAMA_HOME": str(
                    self.project / "local_llm" / "runtime-home"
                ),
                "TEACHABLE_LOCAL_OLLAMA_MODELS": str(
                    self.project / "local_llm" / "models"
                ),
                "TEACHABLE_LOCAL_OLLAMA_TMP": str(
                    self.project / "local_llm" / "tmp"
                ),
                "TEACHABLE_RUNTIME_DIR": str(self.runtime),
                "TEACHABLE_LOG_DIR": str(self.logs),
                "TEACHABLE_LSOF_BIN": str(self.bin / "lsof"),
                "TEACHABLE_CURL_BIN": str(self.bin / "curl"),
                "TEACHABLE_OPEN_BIN": str(self.bin / "open"),
                "TEACHABLE_PS_BIN": str(self.bin / "ps"),
                "TEACHABLE_SLEEP_BIN": "/bin/sleep",
                "TEACHABLE_READY_ATTEMPTS": "20",
                "TEACHABLE_READY_INTERVAL": "0.1",
                "TEACHABLE_OLLAMA_STOP_TIMEOUT": "5",
                "TEACHABLE_BACKEND_STOP_TIMEOUT": "5",
                "TEACHABLE_STOP_TIMEOUT": "5",
            }
        )
        # NOTE: the ps stub reads every fault-injection switch from marker
        # files under $SANDBOX/state directly; there is deliberately no
        # environment bridging, so there is only one mechanism to keep
        # consistent.
        env.update(overrides)
        return env

    def _run_with_live_view(
        self,
        script: Path,
        extra: tuple[str, ...],
        timeout: float,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        stop_flag = threading.Event()

        def watch() -> None:
            while not stop_flag.is_set():
                try:
                    self.sync_ps_meta()
                except OSError:
                    pass
                stop_flag.wait(0.01)

        self.sync_ps_meta()
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        proc = subprocess.Popen(
            ["/bin/bash", str(script), *extra],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, cwd="/",  # proves the script never relies on caller cwd
        )
        self.declare_harness_process(proc.pid, f"/bin/bash {script}")
        try:
            out, err = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                proc.args, proc.returncode, out, err
            )
        finally:
            self.forget_harness_process(proc.pid)
            stop_flag.set()
            watcher.join(timeout=5)
            self.sync_ps_meta()

    def start(
        self,
        *extra: str,
        timeout: float = 60,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self.env(**(env_overrides or {}))
        return self._run_with_live_view(
            self.project / "scripts" / "start-demo.sh", extra, timeout, env
        )

    def stop(
        self,
        *extra: str,
        timeout: float = 60,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self.env(**(env_overrides or {}))
        return self._run_with_live_view(
            self.project / "scripts" / "stop-demo.sh", extra, timeout, env
        )

    def launch(self, **overrides: str) -> "ManagedLauncher":
        return ManagedLauncher(self, overrides)

    def start_meta_watcher(self, interval: float = 0.01) -> threading.Event:
        """Keep the ps fixture in sync while a test drives the launcher
        directly.  Returns the stop event for the watcher."""
        stop_flag = threading.Event()

        def watch() -> None:
            while not stop_flag.is_set():
                try:
                    self.sync_ps_meta()
                except OSError:
                    pass
                stop_flag.wait(interval)

        self.sync_ps_meta()
        threading.Thread(target=watch, daemon=True).start()
        return stop_flag


class ManagedLauncher:
    """Runs start-demo.sh in its own session and tears it down afterwards.

    Output goes to files rather than pipes so a test can observe progress
    while the launcher is still running.
    """

    def __init__(self, sandbox: Sandbox, overrides: dict[str, str]) -> None:
        self.sandbox = sandbox
        self.sandbox.sync_ps_meta()
        self._stop_sync = threading.Event()
        self.out_path = sandbox.root / "launcher.out"
        self.err_path = sandbox.root / "launcher.err"
        self._out = open(self.out_path, "w", encoding="utf-8")
        self._err = open(self.err_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["/bin/bash",
             str(sandbox.project / "scripts" / "start-demo.sh")],
            stdout=self._out,
            stderr=self._err,
            text=True,
            env=sandbox.env(**overrides),
            cwd="/",
            start_new_session=True,
        )
        sandbox.declare_harness_process(
            self.proc.pid,
            f"/bin/bash {sandbox.project}/scripts/start-demo.sh",
        )
        self._sync_thread = threading.Thread(
            target=self._sync_loop, daemon=True
        )
        self._sync_thread.start()

    def _sync_loop(self) -> None:
        while not self._stop_sync.is_set():
            try:
                self.sandbox.sync_ps_meta()
            except OSError:
                pass
            self._stop_sync.wait(0.01)

    def out_so_far(self) -> str:
        self._out.flush()
        try:
            return self.out_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def err_so_far(self) -> str:
        self._err.flush()
        try:
            return self.err_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def wait_ready(self, timeout: float = 40) -> None:
        """Wait until both services are up AND the banner was printed."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if "Press Ctrl+C" in self.out_so_far():
                return
            if self.proc.poll() is not None:
                raise AssertionError(
                    "launcher exited before it was ready:\n"
                    + self.out_so_far() + self.err_so_far()
                )
            time.sleep(0.05)
        raise AssertionError(
            "launcher never printed its ready banner:\n"
            + self.out_so_far() + self.err_so_far()
        )

    def signal(self, sig: int) -> None:
        os.kill(self.proc.pid, sig)

    def wait_exit(self, timeout: float = 30) -> tuple[str, str, int]:
        self.proc.wait(timeout=timeout)
        self._out.flush()
        self._err.flush()
        out = self.out_path.read_text(encoding="utf-8")
        err = self.err_path.read_text(encoding="utf-8")
        self._out.close()
        self._err.close()
        return out, err, self.proc.returncode

    def __enter__(self) -> "ManagedLauncher":
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop_sync.set()
        self.sandbox.forget_harness_process(self.proc.pid)
        try:
            if self.proc.poll() is None:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=10)
        except (ProcessLookupError, PermissionError,
                subprocess.TimeoutExpired):
            pass
        for handle in (self._out, self._err):
            try:
                handle.close()
            except OSError:
                pass
        self.sandbox.kill_all()


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    box = Sandbox(tmp_path / "demo-sandbox")
    box.build()
    yield box
    box.clear_ps_state_sequence()
    box.clear_ghost_processes()
    box.kill_all()


# ---------------------------------------------------------------------------
# 1. syntax and entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/start-demo.sh",
        "scripts/stop-demo.sh",
        "Start Teachable Agent.command",
        "Stop Teachable Agent.command",
    ],
)
def test_shell_syntax_is_valid(relative: str) -> None:
    result = subprocess.run(
        ["/bin/bash", "-n", str(REPO_ROOT / relative)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_scripts_are_executable_and_have_shebang() -> None:
    for relative in (
        "scripts/start-demo.sh",
        "scripts/stop-demo.sh",
        "Start Teachable Agent.command",
        "Stop Teachable Agent.command",
    ):
        path = REPO_ROOT / relative
        assert path.exists(), relative
        assert os.access(path, os.X_OK), f"{relative} is not executable"
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("#!"), relative


def test_command_files_resolve_their_own_directory() -> None:
    for relative in (
        "Start Teachable Agent.command",
        "Stop Teachable Agent.command",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "$(dirname \"$0\")" in text
        assert "/Users/" not in text
        for banned in ("sudo", "LaunchAgent", "launchctl", "systemctl"):
            assert banned not in text, relative


def test_start_command_invokes_the_start_script() -> None:
    text = (REPO_ROOT / "Start Teachable Agent.command").read_text(
        encoding="utf-8"
    )
    assert "scripts/start-demo.sh" in text
    assert 'bash "$SCRIPT"' in text


def test_stop_command_invokes_the_stop_script() -> None:
    text = (REPO_ROOT / "Stop Teachable Agent.command").read_text(
        encoding="utf-8"
    )
    assert "scripts/stop-demo.sh" in text
    assert 'bash "$SCRIPT"' in text


def test_scripts_do_not_use_broad_process_kills() -> None:
    for relative in ("scripts/start-demo.sh", "scripts/stop-demo.sh"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for banned in ("killall", "pkill", "sudo "):
            assert banned not in text, f"{relative} must not use {banned}"


def test_start_script_enables_strict_mode() -> None:
    text = (REPO_ROOT / "scripts" / "start-demo.sh").read_text(
        encoding="utf-8"
    )
    assert "set -euo pipefail" in text


def test_launcher_never_starts_deepseek_harness() -> None:
    for relative in ("scripts/start-demo.sh", "Start Teachable Agent.command"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "npx" not in text
        assert "dsh" not in text


def test_launcher_does_not_install_system_services() -> None:
    for relative in (
        "scripts/start-demo.sh",
        "scripts/stop-demo.sh",
        "Start Teachable Agent.command",
        "Stop Teachable Agent.command",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for banned in ("launchctl", "LaunchAgents", "LaunchDaemons",
                       "sudo", "systemctl"):
            assert banned not in text, f"{relative} must not use {banned}"


# ---------------------------------------------------------------------------
# 2. happy path
# ---------------------------------------------------------------------------


def test_happy_path_does_not_depend_on_service_meta_sync(
    sandbox: Sandbox,
) -> None:
    "The fake ps answers service metadata synchronously, without watcher rows."
    sandbox.suppress_service_meta = True
    with sandbox.launch() as launcher:
        launcher.wait_ready(timeout=40)
        assert sandbox.fake_ollama_pid() is not None
        assert sandbox.backend_pid() is not None
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    sandbox.kill_all()


def test_fingerprint_discovery_survives_delayed_backend_started_files(
    sandbox: Sandbox,
) -> None:
    "Guard-bound discovery must not wait for backend-started* files."
    sandbox.suppress_service_meta = True
    gate = sandbox.state / "release-backend-started"
    gate.unlink(missing_ok=True)
    try:
        with sandbox.launch(
            TEACHABLE_FAKE_BACKEND_STARTED_GATE=str(gate),
            TEACHABLE_READY_ATTEMPTS="100",
        ) as launcher:
            # The handshake proves the backend bootstrap stage has started.
            _wait_for_path(
                sandbox.state / "backend-bootstrap-pid", timeout=10
            )
            deadline = time.time() + 10.0
            while time.time() < deadline and not sandbox.record_exists("backend"):
                if launcher.proc.poll() is not None:
                    break
                time.sleep(0.02)
            assert sandbox.record_exists("backend"), launcher.err_so_far()
            assert not (sandbox.state / "backend-started.pid").exists()
            assert not (sandbox.state / "backend-started").exists()
            # Release the helper only after the PID record existence above.
            gate.write_text("release\n", encoding="utf-8")
            launcher.wait_ready(timeout=40)
            launcher.signal(signal.SIGINT)
            out, err, rc = launcher.wait_exit()
        assert rc == 0, out + err
    finally:
        # Never leave the helper waiting, even on an assertion failure.
        gate.write_text("release\n", encoding="utf-8")
        sandbox.kill_all()


def test_fake_ps_guard_discovery_rejects_mismatches(
    sandbox: Sandbox,
) -> None:
    "Only the guard-published service/command/pid handshake may be forged."
    live_a = subprocess.Popen(["/bin/sleep", "120"])
    live_b = subprocess.Popen(["/bin/sleep", "120"])
    signature = _bootstrap_expected_signature(sandbox)
    sandbox.write_bootstrap_guard(service="ollama", cmd=signature)
    # For a backend handshake the guard is deliberately for ollama.
    _write_bootstrap_handshake(sandbox, "backend", live_a.pid, signature)
    try:
        # Service mismatch: guard says ollama, handshake says backend.
        result = _probe_fake_ps(sandbox, live_a.pid, "state=")
        assert result.returncode == 1
        assert result.stdout == ""

        # Matching guard and handshake do produce the identity.
        sandbox.write_bootstrap_guard(service="backend", cmd=signature)
        _write_bootstrap_handshake(sandbox, "backend", live_a.pid, signature)
        result = _probe_fake_ps(sandbox, live_a.pid, "state=")
        assert result.returncode == 0
        assert result.stdout.strip() == "S"
        result = _probe_fake_ps(sandbox, live_a.pid, "command=")
        assert result.returncode == 0
        assert result.stdout.strip() == signature

        # PID mismatch: handshake names live_a, query live_b.
        result = _probe_fake_ps(sandbox, live_b.pid, "state=")
        assert result.returncode == 1
        assert result.stdout == ""

        # Command mismatch: handshake command differs from the guard command.
        _write_bootstrap_handshake(
            sandbox, "backend", live_a.pid, "different command line"
        )
        result = _probe_fake_ps(sandbox, live_a.pid, "state=")
        assert result.returncode == 1
        assert result.stdout == ""

        # No guard: the handshake alone must not forge identity.
        shutil.rmtree(sandbox.bootstrap_guard_path(), ignore_errors=True)
        _write_bootstrap_handshake(sandbox, "backend", live_a.pid, signature)
        result = _probe_fake_ps(sandbox, live_a.pid, "state=")
        assert result.returncode == 1
        assert result.stdout == ""

        # Unknown live PID with no guard is gone/unavailable, not backend.
        result = _probe_fake_ps(sandbox, live_b.pid, "state=")
        assert result.returncode == 1
        assert result.stdout == ""
    finally:
        for proc in (live_a, live_b):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)
        _cleanup_bootstrap_case(sandbox, None)


def test_starts_ollama_and_backend_when_ports_are_free(sandbox: Sandbox) -> None:
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        for name in ("ollama-demo.pid", "backend-demo.pid"):
            pid = sandbox._read_pid(sandbox.runtime / name)
            assert pid is not None and pid > 0, name
        assert sandbox.read_open_calls() == ["http://127.0.0.1:8000"]
        assert sandbox.log_text("ollama-demo.log") != ""
        assert sandbox.log_text("backend-demo.log") != ""
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    assert "Teachable Agent demo is running" in out
    assert "Web UI" in out
    assert "http://127.0.0.1:8000" in out
    assert "Ctrl+C" in out
    assert "ollama-demo.log" in out
    assert "backend-demo.log" in out
    assert "started by this launcher" in out


def test_launcher_keeps_running_in_the_foreground(sandbox: Sandbox) -> None:
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        time.sleep(0.8)
        assert launcher.proc.poll() is None
        launcher.signal(signal.SIGINT)
        _, _, rc = launcher.wait_exit()
    assert rc == 0


def test_backend_runs_without_reload(sandbox: Sandbox) -> None:
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        started = (sandbox.state / "backend-started").read_text(
            encoding="utf-8"
        )
        assert "-m uvicorn backend.main:app" in started
        assert "--reload" not in started
        assert "--host 127.0.0.1" in started
        assert "--port 8000" in started
        launcher.signal(signal.SIGINT)
        launcher.wait_exit()


# ---------------------------------------------------------------------------
# 3. reusing an already-running, healthy Ollama
# ---------------------------------------------------------------------------


def test_reuses_healthy_external_ollama_and_leaves_it_running(
    sandbox: Sandbox,
) -> None:
    external = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.listen_marker(11435, external.pid)
    sandbox.flag("ollama-ready")
    sandbox.declare_process(external.pid, "/bin/sleep 120")
    try:
        with sandbox.launch() as launcher:
            launcher.wait_ready()
            # It never started its own Ollama and never wrote a record.
            assert sandbox.fake_ollama_pid() is None
            assert not (sandbox.runtime / "ollama-demo.pid").exists()
            launcher.signal(signal.SIGINT)
            out, err, rc = launcher.wait_exit()
        assert rc == 0, out + err
        assert "Reusing the already-running Ollama" in out
        assert "not owned by this launcher" in out

        stop = sandbox.stop()
        assert stop.returncode == 0, stop.stdout + stop.stderr
        assert "not started by this launcher" in stop.stdout
        assert external.poll() is None, "external Ollama was killed"
        assert sandbox.ollama_terminated_pids(timeout=1) == []
    finally:
        sandbox.clear_declared()
        external.terminate()
        external.wait(timeout=10)


def test_occupied_but_unhealthy_ollama_port_fails_safely(
    sandbox: Sandbox,
) -> None:
    squatter = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.listen_marker(11435, squatter.pid)
    sandbox.unflag("ollama-ready")
    sandbox.declare_process(squatter.pid, "/bin/sleep 120")
    try:
        result = sandbox.start()
        assert result.returncode != 0
        assert "did not answer like an Ollama server" in result.stderr
        assert sandbox.fake_ollama_pid() is None
        assert not sandbox.has_flag("backend-started")
        assert squatter.poll() is None
        assert sandbox.ollama_terminated_pids(timeout=1) == []
    finally:
        sandbox.clear_declared()
        squatter.terminate()
        squatter.wait(timeout=10)


# ---------------------------------------------------------------------------
# 4. port 8000 conflicts
# ---------------------------------------------------------------------------


def test_backend_port_conflict_fails_without_killing_the_owner(
    sandbox: Sandbox,
) -> None:
    squatter = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.listen_marker(8000, squatter.pid)
    sandbox.flag("backend-ready")
    sandbox.declare_process(squatter.pid, "/bin/sleep 120")
    try:
        result = sandbox.start()
        assert result.returncode != 0
        assert "is already in use" in result.stderr
        assert "stop-demo.sh" in result.stderr
        assert squatter.poll() is None
        assert not sandbox.has_flag("backend-started")
    finally:
        sandbox.clear_declared()
        squatter.terminate()
        squatter.wait(timeout=10)


def test_backend_port_conflict_cleans_up_ollama_it_started(
    sandbox: Sandbox,
) -> None:
    squatter = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.listen_marker(8000, squatter.pid)
    sandbox.flag("backend-ready")
    sandbox.declare_process(squatter.pid, "/bin/sleep 120")
    try:
        result = sandbox.start()
        assert result.returncode != 0
        ollama_pid = sandbox.fake_ollama_pid()
        assert ollama_pid is not None
        assert sandbox.wait_for_pid_exit(ollama_pid, timeout=10), (
            "the Ollama started by the launcher was left running"
        )
        assert ollama_pid in sandbox.ollama_terminated_pids()
        assert not (sandbox.runtime / "ollama-demo.pid").exists()
    finally:
        sandbox.clear_declared()
        squatter.terminate()
        squatter.wait(timeout=10)


# ---------------------------------------------------------------------------
# 5. readiness timeouts
# ---------------------------------------------------------------------------


def test_ollama_readiness_timeout_cleans_up_started_process(
    sandbox: Sandbox,
) -> None:
    sandbox.flag("ollama-never-ready")
    result = sandbox.start(
        env_overrides={"TEACHABLE_READY_ATTEMPTS": "5",
                       "TEACHABLE_READY_INTERVAL": "0.1"}
    )
    assert result.returncode != 0
    assert "Ollama did not become ready" in result.stderr
    assert "ollama-demo.log" in result.stderr
    ollama_pid = sandbox.fake_ollama_pid()
    assert ollama_pid is not None
    assert sandbox.wait_for_pid_exit(ollama_pid, timeout=10), (
        "timed-out Ollama was left running"
    )
    assert ollama_pid in sandbox.ollama_terminated_pids()
    assert not (sandbox.runtime / "ollama-demo.pid").exists()
    assert not sandbox.has_flag("backend-started")


def test_backend_health_timeout_cleans_up_both_processes(
    sandbox: Sandbox,
) -> None:
    sandbox.flag("backend-never-ready")
    result = sandbox.start(
        env_overrides={"TEACHABLE_READY_ATTEMPTS": "5",
                       "TEACHABLE_READY_INTERVAL": "0.1"}
    )
    assert result.returncode != 0
    assert "FastAPI did not become healthy" in result.stderr
    assert "backend-demo.log" in result.stderr

    ollama_pid = sandbox.fake_ollama_pid()
    assert ollama_pid is not None
    assert sandbox.wait_for_pid_exit(ollama_pid, timeout=10), (
        "the Ollama started by this launcher was left running"
    )
    assert ollama_pid in sandbox.ollama_terminated_pids()

    backend_pid = sandbox.backend_pid()
    assert backend_pid is not None
    assert sandbox.wait_for_pid_exit(backend_pid, timeout=10), (
        "timed-out backend was left running"
    )
    assert not (sandbox.runtime / "ollama-demo.pid").exists()
    assert not (sandbox.runtime / "backend-demo.pid").exists()


# ---------------------------------------------------------------------------
# 6. Ctrl+C / SIGTERM cleanup
# ---------------------------------------------------------------------------


def test_sigint_stops_both_processes_started_by_the_run(
    sandbox: Sandbox,
) -> None:
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        ollama_pid = sandbox.fake_ollama_pid()
        backend_pid = sandbox.backend_pid()
        assert ollama_pid and backend_pid
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    assert ollama_pid in sandbox.ollama_terminated_pids()
    assert sandbox.wait_for_pid_exit(ollama_pid, timeout=10)
    assert sandbox.wait_for_pid_exit(backend_pid, timeout=10)
    assert not (sandbox.runtime / "ollama-demo.pid").exists()
    assert not (sandbox.runtime / "backend-demo.pid").exists()


def test_sigterm_stops_only_this_run(sandbox: Sandbox) -> None:
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        launcher.signal(signal.SIGTERM)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    assert not (sandbox.runtime / "ollama-demo.pid").exists()
    assert not (sandbox.runtime / "backend-demo.pid").exists()


def test_sigint_does_not_touch_a_reused_external_ollama(
    sandbox: Sandbox,
) -> None:
    external = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.listen_marker(11435, external.pid)
    sandbox.flag("ollama-ready")
    sandbox.declare_process(external.pid, "/bin/sleep 120")
    try:
        with sandbox.launch() as launcher:
            launcher.wait_ready()
            launcher.signal(signal.SIGINT)
            used_out, used_err, rc = launcher.wait_exit()
        assert rc == 0, used_out + used_err
        assert external.poll() is None, "external Ollama was killed"
        assert sandbox.ollama_terminated_pids(timeout=1) == []
    finally:
        sandbox.clear_declared()
        external.terminate()
        external.wait(timeout=10)


# ---------------------------------------------------------------------------
# 7. stop-demo.sh behaviour
# ---------------------------------------------------------------------------


def test_stop_is_idempotent_with_no_services_running(sandbox: Sandbox) -> None:
    first = sandbox.stop()
    second = sandbox.stop()
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert "no PID record" in first.stdout


def test_stop_cleans_stale_pid_record(sandbox: Sandbox) -> None:
    path = sandbox.write_pid_record("backend", DEAD_PID)
    result = sandbox.stop()
    assert result.returncode == 0, result.stdout + result.stderr
    # Only an explicit ps "gone" may delete a record.
    assert "no longer exists" in result.stdout
    assert not path.exists()


def test_stop_rejects_non_numeric_pid_content(sandbox: Sandbox) -> None:
    """A malformed PID record is unknown, not stale: keep it, fail, no signal."""
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    path = sandbox.write_pid_record("backend", "not-a-pid")
    before = path.read_text(encoding="utf-8")
    try:
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "not a positive integer" in result.stderr
        assert "keeping it for inspection" in result.stderr
        # The record survives byte-for-byte.
        assert path.exists()
        assert path.read_text(encoding="utf-8") == before
        # Nothing was signalled.
        assert victim.poll() is None
    finally:
        path.unlink(missing_ok=True)
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)


@pytest.mark.parametrize("value", ["0", "-1", "", "12abc"])
def test_stop_rejects_invalid_pid_values(sandbox: Sandbox, value: str) -> None:
    """Every malformed PID value fails closed and keeps the record intact."""
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    path = sandbox.write_pid_record("backend", value)
    before = path.read_text(encoding="utf-8")
    try:
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "not a positive integer" in result.stderr
        assert path.exists()
        assert path.read_text(encoding="utf-8") == before
        assert victim.poll() is None
    finally:
        path.unlink(missing_ok=True)
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)


def test_stop_refuses_a_reused_pid(sandbox: Sandbox) -> None:
    """A live PID whose command line is not the managed process."""
    innocent = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(innocent.pid, "/bin/sleep 120")
    try:
        path = sandbox.write_pid_record("backend", innocent.pid)
        result = sandbox.stop()
        # Refusing to signal is a failed stop: non-zero, record kept.
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Refusing to signal it" in result.stderr
        assert "Keeping the record" in result.stderr
        assert innocent.poll() is None
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)
        sandbox.clear_declared()
        innocent.terminate()
        innocent.wait(timeout=10)


def test_stop_ignores_records_from_another_project(sandbox: Sandbox) -> None:
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    path = sandbox.write_pid_record(
        "backend", victim.pid, project_root="/some/other/project"
    )
    try:
        result = sandbox.stop()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "another project root" in result.stderr
        assert victim.poll() is None
        # The record is deliberately kept: it is not ours to delete.
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)


def _matching_backend_process(sandbox: Sandbox) -> tuple[subprocess.Popen, str]:
    """Start a real process whose command line matches the managed one."""
    fake_python = sandbox.project / ".venv" / "bin" / "python"
    signature = (
        f"{sandbox.project}/.venv/bin/python -m uvicorn "
        "backend.main:app --host 127.0.0.1 --port 8000"
    )
    proc = subprocess.Popen(
        [str(fake_python), "-m", "uvicorn", "backend.main:app",
         "--host", "127.0.0.1", "--port", "8000"],
        env=sandbox.env(),
    )
    return proc, signature


def test_stop_terminates_a_managed_process(sandbox: Sandbox) -> None:
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        path = sandbox.write_pid_record("backend", proc.pid, signature)
        result = sandbox.stop()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Stopping FastAPI backend" in result.stdout
        assert "stopped." in result.stdout
        assert sandbox.wait_for_pid_exit(proc.pid, timeout=10)
        assert not path.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()


def test_stop_escalates_only_for_the_same_managed_process(
    sandbox: Sandbox,
) -> None:
    """A process that ignores SIGTERM eventually gets a bounded SIGKILL."""
    fake_python = sandbox.bin / "python-helper"
    body = fake_python.read_text(encoding="utf-8").replace(
        "        while True:\n            time.sleep(0.2)",
        "        import signal as _s\n"
        "        _s.signal(_s.SIGTERM, _s.SIG_IGN)\n"
        "        time.sleep(300)",
    )
    fake_python.write_text(body, encoding="utf-8")
    fake_python.chmod(0o755)

    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        sandbox.write_pid_record("backend", proc.pid, signature)
        result = sandbox.stop(
            env_overrides={"TEACHABLE_STOP_TIMEOUT": "1"}
        )
        assert "escalating to SIGKILL" in result.stderr
        assert sandbox.wait_for_pid_exit(proc.pid, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()


# ---------------------------------------------------------------------------
# 8. browser failure, secrets, odd paths, missing prerequisites
# ---------------------------------------------------------------------------


def test_browser_failure_only_warns(sandbox: Sandbox) -> None:
    sandbox.flag("open-fails")
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        assert sandbox.fake_ollama_pid() is not None
        assert (sandbox.runtime / "backend-demo.pid").exists()
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    assert "could not open a browser automatically" in err
    assert "http://127.0.0.1:8000 manually" in err


def test_no_secret_is_printed_or_logged(sandbox: Sandbox) -> None:
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        for name in ("ollama-demo.pid", "backend-demo.pid"):
            text = (sandbox.runtime / name).read_text(encoding="utf-8")
            assert SECRET_TOKEN not in text
            assert "API_KEY" not in text
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    combined = out + err
    assert SECRET_TOKEN not in combined
    assert "API_KEY" not in combined
    for name in ("ollama-demo.log", "backend-demo.log"):
        assert SECRET_TOKEN not in sandbox.log_text(name)


def test_works_from_a_path_containing_spaces(tmp_path: Path) -> None:
    box = Sandbox(tmp_path / "demo sandbox with spaces")
    box.build()
    try:
        with box.launch() as launcher:
            launcher.wait_ready()
            assert box.read_open_calls() == ["http://127.0.0.1:8000"]
            assert box.fake_ollama_pid() is not None
            launcher.signal(signal.SIGINT)
            out, err, rc = launcher.wait_exit()
        assert rc == 0, out + err
        assert "Teachable Agent demo is running" in out
    finally:
        box.kill_all()


def test_missing_local_ollama_cli_fails_with_guidance(
    sandbox: Sandbox,
) -> None:
    cli = (sandbox.project / "local_llm" / "Ollama.app" /
           "Contents" / "Resources" / "ollama")
    cli.unlink()
    result = sandbox.start()
    assert result.returncode != 0
    assert "project-local Ollama CLI is not runnable" in result.stderr
    assert "no process was touched" in result.stderr
    assert not sandbox.has_flag("backend-started")


def test_missing_project_files_fail_before_starting_anything(
    sandbox: Sandbox,
) -> None:
    (sandbox.project / ".venv" / "bin" / "python").unlink()
    result = sandbox.start()
    assert result.returncode != 0
    assert "virtualenv interpreter not found" in result.stderr
    assert sandbox.fake_ollama_pid() is None
    assert not sandbox.has_flag("backend-started")


def test_logs_are_created_and_truncated_per_run(sandbox: Sandbox) -> None:
    """A second run must not append to the previous run's logs."""
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        launcher.signal(signal.SIGINT)
        launcher.wait_exit()
    if not (sandbox.logs / "backend-demo.log").exists():
        pytest.fail("backend log was never created")
    (sandbox.logs / "backend-demo.log").write_text(
        "stale-run-marker\n" * 50, encoding="utf-8"
    )
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        launcher.signal(signal.SIGINT)
        launcher.wait_exit()
    assert "stale-run-marker" not in sandbox.log_text("backend-demo.log")


# ---------------------------------------------------------------------------
# 9. PID record ownership, concurrent launchers, single-instance lock
# ---------------------------------------------------------------------------


def _record_snapshot(sandbox: Sandbox, service: str) -> str:
    return sandbox.read_record(service)


def test_second_launcher_does_not_delete_first_launcher_pid_records(
    sandbox: Sandbox,
) -> None:
    """Simulated first launcher's records; the second run must not delete
    or overwrite them when it fails."""
    sig = (
        f"{sandbox.project}/.venv/bin/python -m uvicorn "
        "backend.main:app --host 127.0.0.1 --port 8000"
    )
    ollama_sig = (
        f"{sandbox.project}/local_llm/Ollama.app/Contents/Resources/"
        "ollama serve"
    )
    # A first launcher's records, written by "run A".
    first_ollama = sandbox.write_pid_record(
        "ollama", "40001", ollama_sig, run_id="run-A"
    )
    first_backend = sandbox.write_pid_record(
        "backend", "40002", sig, run_id="run-A"
    )
    before_ollama = first_ollama.read_text(encoding="utf-8")
    before_backend = first_backend.read_text(encoding="utf-8")

    # The second launcher fails immediately on the pre-existing record:
    # it must neither overwrite nor delete run A's records.
    second = sandbox.start()
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
    assert first_ollama.read_text(encoding="utf-8") == before_ollama
    assert first_backend.read_text(encoding="utf-8") == before_backend


def test_preflight_failure_preserves_existing_pid_records(
    sandbox: Sandbox,
) -> None:
    seed = sandbox.write_pid_record("backend", "424242", run_id="other-run")
    before = seed.read_text(encoding="utf-8")
    (sandbox.project / ".venv" / "bin" / "python").unlink()
    result = sandbox.start()
    assert result.returncode != 0
    assert "virtualenv interpreter not found" in result.stderr
    assert sandbox.read_record("backend") == before


def test_concurrent_launchers_cannot_overwrite_pid_records(
    sandbox: Sandbox,
) -> None:
    """Two launchers for the SAME project root, runtime dir and ports.

    Launcher A is left running; launcher B must fail immediately on the
    single-instance lock without touching any of A's state.
    """
    with sandbox.launch() as launcher_a:
        launcher_a.wait_ready()

        # A's complete state must be visible and consistent before B starts.
        lock_dir = sandbox.runtime / "demo.lock"
        assert lock_dir.is_dir()
        a_lock = (lock_dir / "info").read_text(encoding="utf-8")
        a_ollama = sandbox.read_record("ollama")
        a_backend = sandbox.read_record("backend")
        a_ollama_pid = sandbox.fake_ollama_pid()
        a_backend_pid = sandbox.backend_pid()
        assert a_ollama_pid and a_backend_pid
        assert "RUN_ID=" in a_ollama and "RUN_ID=" in a_backend
        assert "PID=" in a_lock

        # Launcher B: same project root, same runtime dir, same ports.
        result = sandbox.start()
        assert result.returncode != 0
        assert "another launcher is already running" in result.stderr
        # B must not have started a second copy of anything.
        assert sandbox.fake_ollama_pid() == a_ollama_pid
        assert sandbox.backend_pid() == a_backend_pid

        # A's lock, records and services are byte-for-byte unchanged.
        assert (lock_dir / "info").read_text(encoding="utf-8") == a_lock
        assert sandbox.read_record("ollama") == a_ollama
        assert sandbox.read_record("backend") == a_backend
        assert sandbox._pid_is_live(a_ollama_pid)
        assert sandbox._pid_is_live(a_backend_pid)

        # A still shuts down cleanly afterwards.
        launcher_a.signal(signal.SIGINT)
        a_out, a_err, a_rc = launcher_a.wait_exit()
    assert a_rc == 0, a_out + a_err
    assert a_ollama_pid in sandbox.ollama_terminated_pids()
    assert sandbox.wait_for_pid_exit(a_backend_pid, timeout=10)
    assert not (sandbox.runtime / "demo.lock").exists()
    assert not sandbox.record_exists("backend")
    assert not sandbox.record_exists("ollama")


def test_cleanup_removes_only_records_created_by_its_run(
    sandbox: Sandbox,
) -> None:
    """A foreign record at the ollama path blocks startup and is preserved
    byte-for-byte; the failed run never deletes it."""
    foreign = sandbox.write_pid_record(
        "ollama", DEAD_PID, run_id="some-other-run"
    )
    foreign_text = foreign.read_text(encoding="utf-8")
    result = sandbox.start()
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert foreign.read_text(encoding="utf-8") == foreign_text
    # Nothing of this run was started or left behind.
    assert not sandbox.record_exists("backend")
    # Its own Ollama was started and then cleaned up on the failure path.
    ollama_pid = sandbox.fake_ollama_pid()
    if ollama_pid:
        assert sandbox.wait_for_pid_exit(ollama_pid, timeout=10)
    assert not sandbox.record_exists("ollama") or \
        sandbox.read_record("ollama") == foreign_text


def test_single_instance_lock_rejects_a_second_live_launcher(
    sandbox: Sandbox,
) -> None:
    """A live lock owner (matching fingerprint) blocks a new launcher."""
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120")
    lock = sandbox.runtime / "demo.lock"
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "info").write_text(
        f"PID={owner.pid}\n"
        "RUN_ID=owner-run\n"
        f"FINGERPRINT={sandbox._lstart_for(owner.pid)}\n"
        f"PROJECT_ROOT={sandbox.project}\n",
        encoding="utf-8",
    )
    try:
        result = sandbox.start()
        assert result.returncode != 0
        assert "another launcher is already running" in result.stderr
        assert owner.poll() is None
        assert (lock / "info").exists()
    finally:
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


def test_single_instance_lock_removes_a_stale_lock(
    sandbox: Sandbox,
) -> None:
    """A lock whose owner is gone is cleaned up and the run proceeds."""
    lock = sandbox.runtime / "demo.lock"
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "info").write_text(
        f"PID={DEAD_PID}\n"
        "RUN_ID=dead-run\n"
        "FINGERPRINT=Thu Sep 14 12:00:00 2026\n"
        f"PROJECT_ROOT={sandbox.project}\n",
        encoding="utf-8",
    )
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        # The stale lock was replaced by this run's own, valid lock.
        lock_info = (sandbox.runtime / "demo.lock" / "info").read_text(
            encoding="utf-8"
        )
        assert "RUN_ID=" in lock_info
        assert "FINGERPRINT=" in lock_info
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    # A clean exit releases the lock entirely.
    assert not (sandbox.runtime / "demo.lock").exists()


def test_single_instance_lock_keeps_a_lock_with_a_mismatched_owner(
    sandbox: Sandbox,
) -> None:
    """A live PID whose fingerprint differs from the lock's is NOT removed
    by rmdir force; the launcher must not proceed."""
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120")
    lock = sandbox.runtime / "demo.lock"
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "info").write_text(
        f"PID={owner.pid}\n"
        "RUN_ID=owner-run\n"
        "FINGERPRINT=Wed Sep 13 00:00:00 2026\n"
        f"PROJECT_ROOT={sandbox.project}\n",
        encoding="utf-8",
    )
    try:
        result = sandbox.start()
        # The lock owner is alive but does not match the stored fingerprint:
        # fail closed without touching it.
        assert result.returncode != 0
        assert owner.poll() is None
    finally:
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


def _write_lock(sandbox: Sandbox, text: str | None) -> Path:
    """Create the lock directory, optionally with an info file."""
    lock = sandbox.runtime / "demo.lock"
    lock.mkdir(parents=True, exist_ok=True)
    if text is not None:
        (lock / "info").write_text(text, encoding="utf-8")
    return lock


def _valid_lock_text(sandbox: Sandbox, owner_pid: int,
                     fingerprint: str | None = None) -> str:
    return (
        f"PID={owner_pid}\n"
        "RUN_ID=owner-run\n"
        f"FINGERPRINT={fingerprint or sandbox._lstart_for(owner_pid)}\n"
        f"PROJECT_ROOT={sandbox.project}\n"
    )


def test_missing_lock_info_is_never_removed(sandbox: Sandbox) -> None:
    """A lock directory without info is unusable, never stale."""
    lock = _write_lock(sandbox, None)
    result = sandbox.start()
    assert result.returncode != 0
    assert "owner metadata is missing" in result.stderr
    assert lock.is_dir()
    assert not (lock / "info").exists()
    assert sandbox.fake_ollama_pid() is None


def test_partially_written_lock_is_never_removed(sandbox: Sandbox) -> None:
    """A half-written info file must not be treated as stale."""
    lock = _write_lock(sandbox, "PID=12345\nRUN_ID=partial\n")
    result = sandbox.start()
    assert result.returncode != 0
    assert "owner metadata is missing" in result.stderr
    assert (lock / "info").read_text(encoding="utf-8") == (
        "PID=12345\nRUN_ID=partial\n"
    )
    assert sandbox.fake_ollama_pid() is None


def test_incomplete_lock_is_never_removed(sandbox: Sandbox) -> None:
    """Empty info file: fail closed, keep the lock."""
    lock = _write_lock(sandbox, "")
    result = sandbox.start()
    assert result.returncode != 0
    assert "owner metadata is missing" in result.stderr
    assert lock.is_dir()
    assert (lock / "info").read_text(encoding="utf-8") == ""


def test_duplicate_lock_fields_are_rejected(sandbox: Sandbox) -> None:
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120")
    lock = _write_lock(
        sandbox, _valid_lock_text(sandbox, owner.pid) + "PID=999\n"
    )
    before = (lock / "info").read_text(encoding="utf-8")
    try:
        result = sandbox.start()
        assert result.returncode != 0
        assert "owner metadata is missing" in result.stderr
        assert (lock / "info").read_text(encoding="utf-8") == before
        assert owner.poll() is None
    finally:
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


def test_lock_ps_failure_is_not_treated_as_stale(sandbox: Sandbox) -> None:
    """A ps failure while checking the owner must never remove the lock."""
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120")
    lock = _write_lock(sandbox, _valid_lock_text(sandbox, owner.pid))
    before = (lock / "info").read_text(encoding="utf-8")
    sandbox.flag("ps-state-fails")
    try:
        result = sandbox.start()
        assert result.returncode != 0
        assert "cannot be checked" in result.stderr
        assert "ps wrote a diagnostic" in result.stderr
        assert (lock / "info").read_text(encoding="utf-8") == before
    finally:
        sandbox.unflag("ps-state-fails")
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


def test_lock_info_write_failure_releases_only_own_new_lock(
    sandbox: Sandbox,
) -> None:
    """When the launcher's own fingerprint cannot be read, no lock may be
    left behind (the preflight refuses to start at all, so nothing is
    created and nothing is half-started)."""
    sandbox.flag("ps-lstart-fails")
    result = sandbox.start()
    assert result.returncode != 0
    assert "cannot report this process's start time" in result.stderr
    assert "Nothing was started" in result.stderr
    assert not (sandbox.runtime / "demo.lock").exists()
    assert sandbox.fake_ollama_pid() is None
    assert not sandbox.has_flag("backend-started")
    sandbox.unflag("ps-lstart-fails")


def test_ps_error_does_not_delete_live_pid_record(sandbox: Sandbox) -> None:
    """An unreadable ps must never look like "the process is gone"."""
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, signature)
        before = record.read_text(encoding="utf-8")
        sandbox.flag("ps-state-fails")
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "cannot determine whether PID" in result.stderr
        assert record.exists()
        assert record.read_text(encoding="utf-8") == before
        assert proc.poll() is None
    finally:
        sandbox.unflag("ps-state-fails")
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()


def test_ps_error_does_not_remove_launcher_lock(sandbox: Sandbox) -> None:
    """stop-demo.sh never removes a launcher lock (it has no such code) and
    an unreadable ps must not make the launcher drop its own lock either."""
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120")
    lock = _write_lock(sandbox, _valid_lock_text(sandbox, owner.pid))
    before = (lock / "info").read_text(encoding="utf-8")
    sandbox.flag("ps-state-fails")
    try:
        result = sandbox.stop()
        assert result.returncode == 0, result.stdout + result.stderr
        assert lock.is_dir()
        assert (lock / "info").read_text(encoding="utf-8") == before
    finally:
        sandbox.unflag("ps-state-fails")
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


def test_ps_error_after_term_does_not_report_stopped(sandbox: Sandbox) -> None:
    """If ps breaks right after TERM, the stop must not claim success."""
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, signature)
        # Make the pid disappear from the ps fixture as soon as the stop
        # script starts waiting, without the real process exiting.
        sandbox.flag("ps-state-fails")
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "lost track of PID" in result.stderr or \
            "cannot determine whether PID" in result.stderr
        assert "no longer exists" not in result.stdout
        assert record.exists()
        assert proc.poll() is None
    finally:
        sandbox.unflag("ps-state-fails")
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()


def test_ps_error_after_kill_keeps_pid_record(sandbox: Sandbox) -> None:
    """An unreadable ps after the TERM/KILL sequence keeps the record."""
    sig = (
        f"{sandbox.project}/.venv/bin/python -m uvicorn "
        "backend.main:app --host 127.0.0.1 --port 8000"
    )
    proc, _ = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, sig)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, sig)
        sandbox.set_ghost_process(proc.pid, sig)
        # The ghost keeps the pid observable while ps itself is broken.
        sandbox.flag("ps-state-fails")
        result = sandbox.stop(env_overrides={"TEACHABLE_STOP_TIMEOUT": "1"})
        assert result.returncode == 1, result.stdout + result.stderr
        assert record.exists()
        assert sandbox.record_field("backend", "PID") == str(proc.pid)
    finally:
        sandbox.unflag("ps-state-fails")
        sandbox.clear_ghost_processes()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


def test_backend_exits_and_ollama_cleanup_failure_returns_nonzero(
    sandbox: Sandbox,
) -> None:
    """A natural backend exit must still fail the run when the Ollama
    process cannot be confirmed stopped."""
    sig = (
        f"{sandbox.project}/local_llm/Ollama.app/Contents/Resources/"
        "ollama serve"
    )
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        ollama_pid = sandbox.fake_ollama_pid()
        assert ollama_pid
        # Keep the Ollama pid observable via a ghost, then make ps fail so
        # its state can never be confirmed.
        sandbox.set_ghost_process(ollama_pid, sig)
        sandbox.flag("ps-state-fails")
        # The real backend exits on its own -> the natural-exit path runs.
        sandbox.terminate(sandbox.backend_pid(), signal.SIGTERM)
        out, err, rc = launcher.wait_exit()
    assert rc != 0, out + err
    assert "could not be confirmed stopped" in err or \
        "cannot determine" in err
    assert sandbox.record_exists("ollama")
    sandbox.unflag("ps-state-fails")
    sandbox.clear_ghost_processes()
    sandbox.kill_all()


# ---------------------------------------------------------------------------
# 9b. ps stderr must always be treated as unknown
# ---------------------------------------------------------------------------


def test_ps_state_with_stderr_keeps_launcher_lock(sandbox: Sandbox) -> None:
    """rc=0 + a valid state + non-empty stderr is UNKNOWN, not alive.

    The lock owner must therefore be treated as unverifiable: a second
    launcher fails, the lock survives untouched, and no service starts.
    """
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120")
    lock = _write_lock(sandbox, _valid_lock_text(sandbox, owner.pid))
    before = (lock / "info").read_text(encoding="utf-8")
    # The stub still prints a perfectly valid state on stdout.
    sandbox.set_ps_field_stderr("state=")
    try:
        result = sandbox.start()
        assert result.returncode != 0
        assert "cannot be checked" in result.stderr
        assert "ps wrote a diagnostic" in result.stderr
        # The lock was not deleted and was not taken over.
        assert lock.is_dir()
        assert (lock / "info").read_text(encoding="utf-8") == before
        # No service was started.
        assert sandbox.fake_ollama_pid() is None
        assert not sandbox.has_flag("backend-started")
        assert sandbox.backend_pid() is None
        assert owner.poll() is None
    finally:
        sandbox.clear_ps_field_stderr()
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)
        if lock.exists():
            shutil.rmtree(lock, ignore_errors=True)


def test_stop_ps_state_with_stderr_never_signals(sandbox: Sandbox) -> None:
    """stop-demo.sh: valid state + stderr on the same probe is unknown.

    No signal is delivered, the PID record stays byte-for-byte and the
    command reports failure.
    """
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, signature)
        before = record.read_text(encoding="utf-8")
        sandbox.set_ps_field_stderr("state=")
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "cannot determine whether PID" in result.stderr
        assert "ps wrote a diagnostic" in result.stderr
        # No signal: the process is still alive and untouched.
        assert proc.poll() is None
        assert record.exists()
        assert record.read_text(encoding="utf-8") == before
    finally:
        sandbox.clear_ps_field_stderr()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


@pytest.mark.parametrize("field", ["command=", "lstart="])
def test_ps_identity_field_with_stderr_never_signals(
    sandbox: Sandbox, field: str,
) -> None:
    """A command/lstart probe that also writes to stderr must not be
    accepted as trusted identity data."""
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, signature)
        before = record.read_text(encoding="utf-8")
        sandbox.set_ps_field_stderr(field)
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Refusing to signal" in result.stderr
        # The process is alive and the record survived byte-for-byte.
        assert proc.poll() is None
        assert record.exists()
        assert record.read_text(encoding="utf-8") == before
    finally:
        sandbox.clear_ps_field_stderr()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


def _wait_for_path(path: Path, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def _read_positive_pid(path: Path) -> int:
    raw = path.read_text(encoding="utf-8").strip()
    assert raw.isdigit(), raw
    return int(raw)


def _configure_bootstrap_fakes(sandbox: Sandbox, cli_body: str) -> None:
    """Install the holding launcher and the requested service CLI."""
    cli = (
        sandbox.project / "local_llm" / "Ollama.app" / "Contents" /
        "Resources" / "ollama"
    )
    _write_executable(cli, cli_body)
    launcher = sandbox.project / "scripts" / "start-local-ollama.sh"
    _write_executable(launcher, FAKE_HOLDING_OLLAMA_LAUNCHER)


def _bootstrap_expected_signature(sandbox: Sandbox) -> str:
    return (
        f"{sandbox.project}/local_llm/Ollama.app/Contents/Resources/"
        "ollama serve"
    )


def _release_bootstrap_child(
    sandbox: Sandbox, *, sequence: str, timeout: float = 10.0
) -> int:
    """Let the held direct child exec, after installing all ps faults.

    The PID read here is exactly the one the production shell captured from
    its background fork.  lstart is broken before the process declaration is
    installed, so the script can never succeed in writing a formal PID
    record; the scripted state sequence then controls the cleanup probes.
    """
    _wait_for_path(sandbox.state / "bootstrap-ollama-shell-held", timeout)
    pid = _read_positive_pid(sandbox.state / "bootstrap-ollama-shell.pid")
    sandbox.break_lstart_for(pid)
    sandbox.declare_process(pid, _bootstrap_expected_signature(sandbox))
    sandbox.set_ps_state_sequence(pid, sequence)
    sandbox.set_ps_probe_log()
    (sandbox.state / "bootstrap-ollama-release").write_text(
        "release\n", encoding="utf-8"
    )
    return pid


def _wait_for_launcher_exit(
    launcher: ManagedLauncher, timeout: float = 20.0
) -> tuple[str, str, int]:
    """Wait with a Python-level deadline so a regression cannot hang pytest."""
    try:
        return launcher.wait_exit(timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(
            "launcher did not exit within the Python-level timeout; the "
            "bootstrap cleanup regressed to an unbounded wait:\n"
            + launcher.out_so_far() + launcher.err_so_far()
        )


def _injected_state_probes(sandbox: Sandbox, pid: int) -> list[str]:
    prefix = f"{pid}|state="
    return [
        line.rsplit("injected=", 1)[1]
        for line in sandbox.ps_probe_log()
        if line.startswith(prefix) and "injected=" in line
    ]


def _cleanup_bootstrap_case(sandbox: Sandbox, pid: int | None) -> None:
    if pid is not None:
        sandbox.terminate(pid, signal.SIGKILL)
        sandbox.wait_for_pid_exit(pid, timeout=5)
    sandbox.clear_ps_state_sequence()
    sandbox.clear_break_lstart_for()
    sandbox.clear_ps_field_stderr()
    sandbox.clear_ghost_processes()
    for name in (
        "bootstrap-ollama-shell.pid",
        "bootstrap-ollama-shell-held",
        "bootstrap-ollama-release",
        "fake-ollama.pid",
        "ollama-term",
    ):
        (sandbox.state / name).unlink(missing_ok=True)
    marker = sandbox.cleanup_failed_path()
    if marker.is_dir() and not marker.is_symlink():
        shutil.rmtree(marker, ignore_errors=True)
    else:
        marker.unlink(missing_ok=True)
    shutil.rmtree(sandbox.bootstrap_guard_path(), ignore_errors=True)
    sandbox.clear_declared()


def _bootstrap_launch_env() -> dict[str, str]:
    # A wide fingerprint retry window gives the test time to install the ps
    # faults after the holding child has forked; the stop timeouts keep the
    # cleanup bounds cheap.
    return {
        "TEACHABLE_LSTART_RETRIES": "100",
        "TEACHABLE_LSTART_INTERVAL": "0.02",
        "TEACHABLE_OLLAMA_STOP_TIMEOUT": "1",
        "TEACHABLE_BACKEND_STOP_TIMEOUT": "1",
    }


def test_fake_ps_state_sequence_advances_and_repeats(
    sandbox: Sandbox,
) -> None:
    """The sequence fixture itself must be 1-based and must keep advancing."""
    pid = 424242
    env = dict(os.environ)
    env["SANDBOX"] = str(sandbox.root)
    ps = sandbox.bin / "ps"

    def probe() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(ps), "-p", str(pid), "-o", "state="],
            capture_output=True, text=True, env=env, check=False,
        )

    sandbox.set_ps_state_sequence(pid, "S,Z")
    count = 0
    try:
        results = [probe() for _ in range(4)]
        count = int(
            (sandbox.state / f"ps-state-seq-count-{pid}")
            .read_text(encoding="utf-8").strip()
        )
    finally:
        sandbox.clear_ps_state_sequence()
    assert [r.stdout.strip() for r in results] == ["S", "Z", "Z", "Z"]
    assert all(r.returncode == 0 for r in results)
    assert count == 5  # four calls, each advanced the counter

    # "-" is a ps diagnostic: rc=1 with stderr, i.e. unknown, never gone.
    sandbox.set_ps_state_sequence(pid, "S,-")
    try:
        first = probe()
        second = probe()
    finally:
        sandbox.clear_ps_state_sequence()
    assert first.returncode == 0
    assert first.stdout.strip() == "S"
    assert second.returncode == 1
    assert second.stdout == ""
    assert second.stderr.strip()


def test_bootstrap_cleanup_uses_the_latest_probe_state(
    sandbox: Sandbox,
) -> None:
    """The real start-demo.sh bootstrap path walks S -> S -> Z and reaps.

    The service child is started but no formal PID record can be written
    (its lstart is broken), so the script aborts into its bootstrap-child
    cleanup.  The state sequence is scripted per probe: a caller reusing a
    cached state could never reach the final Z, so the assertion on the
    ordered probe log proves each loop iteration probed afresh.
    """
    _configure_bootstrap_fakes(sandbox, FAKE_OLLAMA_CLI)
    pid: int | None = None
    try:
        with sandbox.launch(**_bootstrap_launch_env()) as launcher:
            pid = _release_bootstrap_child(sandbox, sequence="S,S,Z")
            out, err, rc = _wait_for_launcher_exit(launcher)
        assert rc != 0, out + err
        assert "could not read the process start fingerprint" in err
        assert "bootstrap-child cleanup" in err
        assert "was stopped and reaped" in err
        assert "could NOT be confirmed stopped" not in err
        # No formal PID record was ever written for this bootstrap child.
        assert not sandbox.record_exists("ollama")
        assert not sandbox.record_exists("backend")
        # The zombie/terminated child was reaped by the launcher.
        assert sandbox.wait_for_pid_exit(pid, timeout=10)
        probes = _injected_state_probes(sandbox, pid)
        assert probes[:3] == ["S", "S", "Z"], probes
        assert len(probes) >= 3, probes
        # The real child did receive the bounded TERM.
        assert pid in sandbox.ollama_terminated_pids(timeout=5)
    finally:
        _cleanup_bootstrap_case(sandbox, pid)


@pytest.mark.parametrize(
    ("sequence", "expected_probes"),
    [("Z", ["Z", "Z"]), ("gone", ["gone", "gone"])],
)
def test_bootstrap_cleanup_reaps_initial_collectable_state(
    sandbox: Sandbox, sequence: str, expected_probes: list[str],
) -> None:
    """Initial zombie/gone must be handed to wait and actually reaped.

    The fake CLI exits by itself, so the launcher shell really holds an
    unreaped direct child (a zombie).  The sequence makes the first state
    probe report the terminal state; the production code must call wait
    directly and report a successful reap without sending SIGTERM.
    """
    _configure_bootstrap_fakes(sandbox, FAKE_EXITING_OLLAMA_CLI)
    pid: int | None = None
    try:
        with sandbox.launch(**_bootstrap_launch_env()) as launcher:
            pid = _release_bootstrap_child(sandbox, sequence=sequence)
            out, err, rc = _wait_for_launcher_exit(launcher)
        assert rc != 0, out + err
        assert "was stopped and reaped" in err
        assert "could NOT be confirmed stopped" not in err
        assert not sandbox.record_exists("ollama")
        assert sandbox.wait_for_pid_exit(pid, timeout=10)
        probes = _injected_state_probes(sandbox, pid)
        assert probes[:len(expected_probes)] == expected_probes, probes
        # No TERM was needed: the child was already collectable.
        assert sandbox.ollama_terminated_pids(timeout=1) == []
    finally:
        _cleanup_bootstrap_case(sandbox, pid)


def test_bootstrap_cleanup_unknown_fails_closed_without_sigkill(
    sandbox: Sandbox,
) -> None:
    """An unknown state after TERM aborts without SIGKILL or any wait.

    The service CLI ignores SIGTERM and stays alive, so if the production
    path escalated to SIGKILL or entered an unbounded wait, this test would
    see the child dead or wait for the Python-level timeout instead of
    observing the honest fail-closed result.
    """
    _configure_bootstrap_fakes(sandbox, FAKE_STUBBORN_OLLAMA_CLI)
    pid: int | None = None
    try:
        with sandbox.launch(**_bootstrap_launch_env()) as launcher:
            pid = _release_bootstrap_child(sandbox, sequence="S,-")
            out, err, rc = _wait_for_launcher_exit(launcher)
            assert rc != 0, out + err
            assert "lost track of the child's state" in err
            assert "no SIGKILL sent" in err
            assert "could NOT be confirmed stopped" in err
            assert not sandbox.record_exists("ollama")
            # The real child ignores TERM and is still alive: neither
            # SIGKILL nor a blocking wait can have run.
            alive = True
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                alive = False
            assert alive, "child was killed despite the unknown state"
            probes = _injected_state_probes(sandbox, pid)
            assert probes[:2] == ["S", "-"], probes
    finally:
        _cleanup_bootstrap_case(sandbox, pid)


def _marker_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = value
    return fields


def _suppress_auto_backend_row(sandbox: Sandbox) -> None:
    """Wait for the fake backend process, then remove its auto ps row.

    This makes a test-declared state token the row the fake ps actually
    reports instead of the fixture's hard-coded discovery row.
    """
    deadline = time.time() + 5.0
    while sandbox.backend_pid() is None and time.time() < deadline:
        time.sleep(0.01)
    (sandbox.state / "backend-started.pid").unlink(missing_ok=True)
    sandbox.sync_ps_meta()


def _dir_snapshot(path: Path) -> dict[str, tuple[str, bytes | str]]:
    result: dict[str, tuple[str, bytes | str]] = {}
    for entry in sorted(path.rglob("*")):
        relative = str(entry.relative_to(path))
        if entry.is_symlink():
            result[relative] = ("symlink", os.readlink(entry))
        elif entry.is_dir():
            result[relative] = ("dir", "")
        else:
            result[relative] = ("file", entry.read_bytes())
    return result


def _make_marker_publish_directory_hook(
    sandbox: Sandbox,
) -> tuple[Path, bytes]:
    payload = b"external directory payload\n"
    source = sandbox.state / "external-dir-payload"
    source.write_bytes(payload)
    hook = sandbox.bin / "marker-publish-directory-hook"
    body = (
        _BASH_SHEBANG
        + "set -euo pipefail\n"
        + "mkdir \"$2\"\n"
        + "cp " + shlex.quote(str(source)) + " \"$2/payload\"\n"
    )
    _write_executable(hook, body)
    return hook, payload


def _make_marker_publish_symlink_hook(
    sandbox: Sandbox, external_dir: Path,
) -> Path:
    hook = sandbox.bin / "marker-publish-symlink-hook"
    body = (
        _BASH_SHEBANG
        + "set -euo pipefail\n"
        + "ln -s " + shlex.quote(str(external_dir)) + " \"$2\"\n"
    )
    _write_executable(hook, body)
    return hook


def _write_bootstrap_handshake(
    sandbox: Sandbox,
    service: str,
    pid: int,
    cmd: str,
    service_value: str | None = None,
) -> None:
    prefix = "backend-bootstrap" if service == "backend" else "ollama-bootstrap"
    (sandbox.state / f"{prefix}-pid").write_text(
        f"{pid}\n", encoding="utf-8"
    )
    (sandbox.state / f"{prefix}-service").write_text(
        f"{service_value if service_value is not None else service}\n",
        encoding="utf-8",
    )
    (sandbox.state / f"{prefix}-cmd").write_text(
        f"{cmd}\n", encoding="utf-8"
    )


def _probe_fake_ps(
    sandbox: Sandbox, pid: int, key: str = "state=",
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["SANDBOX"] = str(sandbox.root)
    env["TEACHABLE_RUNTIME_DIR"] = str(sandbox.runtime)
    env["TEACHABLE_PROJECT_ROOT"] = str(sandbox.project)
    return subprocess.run(
        [str(sandbox.bin / "ps"), "-p", str(pid), "-o", key],
        capture_output=True, text=True, env=env, timeout=10,
    )


def _make_marker_publish_hook(sandbox: Sandbox, marker_text: str) -> Path:
    """Create a hook that publishes an external marker before ours can."""
    source = sandbox.state / "external-marker-source"
    source.write_text(marker_text, encoding="utf-8")
    hook = sandbox.bin / "marker-publish-hook"
    body = (
        _BASH_SHEBANG
        + "set -euo pipefail\n"
        + "cp " + shlex.quote(str(source)) + " \"$2\"\n"
    )
    _write_executable(hook, body)
    return hook


def _restore_normal_ollama_fakes(sandbox: Sandbox) -> None:
    _write_executable(
        sandbox.project / "scripts" / "start-local-ollama.sh",
        FAKE_OLLAMA_LAUNCHER,
    )
    _write_executable(
        sandbox.project / "local_llm" / "Ollama.app" / "Contents" /
        "Resources" / "ollama",
        FAKE_OLLAMA_CLI,
    )


def test_quarantine_marker_blocks_next_launches_and_clears_when_gone(
    sandbox: Sandbox,
) -> None:
    """A real bootstrap failure writes a marker and the next start obeys it.

    The first launcher starts the service, loses its state after TERM
    (unknown), leaves the real child alive and exits.  The marker must name
    that exact PID, survive alive and unknown probes, block both restarts
    before any service starts, and then disappear only after the PID is
    trusted gone.
    """
    _configure_bootstrap_fakes(sandbox, FAKE_STUBBORN_OLLAMA_CLI)
    signature = _bootstrap_expected_signature(sandbox)
    marker = sandbox.cleanup_failed_path()
    pid: int | None = None
    try:
        with sandbox.launch(**_bootstrap_launch_env()) as launcher:
            launcher_pid = launcher.proc.pid
            pid = _release_bootstrap_child(sandbox, sequence="S,-")
            out, err, rc = _wait_for_launcher_exit(launcher)
            assert rc == 1, out + err
            assert "lost track of the child's state" in err
            assert "could NOT be confirmed stopped" in err
            assert f"Inspect PID {pid}" in err
            assert "Inspect PID  " not in err
            assert not sandbox.record_exists("ollama")
            assert not sandbox.record_exists("backend")
            assert marker.is_file()
            marker_text = marker.read_text(encoding="utf-8")
            fields = _marker_fields(marker)
            assert fields.get("PID") == str(pid)
            assert fields.get("SERVICE") == "ollama"
            assert fields.get("PROJECT_ROOT") == str(sandbox.project)
            assert fields.get("CMD") == signature
            assert fields.get("RUN_ID", "").startswith(f"{launcher_pid}-")
            assert fields.get("REASON")
            assert "FINGERPRINT=" not in marker_text
            # The complete record was published, so the guard was released.
            assert not sandbox.bootstrap_guard_path().exists()
            # The child was neither killed nor signalled by the marker path.
            os.kill(pid, 0)
            # Keep the child alive when the context manager tears down.
            (sandbox.state / "fake-ollama.pid").unlink(missing_ok=True)

        # The fake ps must still report the residual child while the second
        # and third launches are blocked.
        sandbox.declare_process(pid, signature)

        for state, expected in (("S", "still alive"), ("-", "cannot be verified")):
            sandbox.set_ps_state_sequence(pid, state)
            blocked = sandbox.start(timeout=20)
            assert blocked.returncode == 1, blocked.stdout + blocked.stderr
            assert "cleanup-failed marker" in blocked.stderr
            assert expected in blocked.stderr
            assert f"PID {pid}" in blocked.stderr
            assert "Starting the project-local Ollama" not in blocked.stdout
            assert marker.read_text(encoding="utf-8") == marker_text
            assert not sandbox.record_exists("ollama")
            assert not sandbox.record_exists("backend")
            os.kill(pid, 0)

        # Trusted gone: the marker may be removed and the launcher may start.
        sandbox.terminate(pid, signal.SIGKILL)
        assert sandbox.wait_for_pid_exit(pid, timeout=10)
        sandbox.clear_ps_state_sequence()
        sandbox.clear_declared()
        sandbox.sync_ps_meta()
        _restore_normal_ollama_fakes(sandbox)

        with sandbox.launch() as third:
            third.wait_ready()
            third.signal(signal.SIGINT)
            third_out, third_err, third_rc = third.wait_exit()
        assert third_rc == 0, third_out + third_err
        assert not marker.exists()
    finally:
        _cleanup_bootstrap_case(sandbox, pid)


def test_malformed_cleanup_failed_marker_is_preserved(
    sandbox: Sandbox,
) -> None:
    """Missing, duplicated or unknown fields keep the marker and block."""
    good = sandbox.write_cleanup_failed_marker(DEAD_PID).read_text(
        encoding="utf-8"
    )
    cases = {
        "missing_reason": "\n".join(
            line for line in good.splitlines()
            if not line.startswith("REASON=")
        ) + "\n",
        "duplicate_pid": good + f"PID={DEAD_PID}\n",
        "garbage_line": good + "this is not a field\n",
        "unknown_key": good + "SURPRISE=value\n",
        "wrong_reason": "\n".join(
            "REASON=custom reason" if line.startswith("REASON=") else line
            for line in good.splitlines()
        ) + "\n",
    }
    for name, marker_text in cases.items():
        marker = sandbox.write_cleanup_failed_marker(
            0, text=marker_text
        )
        before = marker.read_text(encoding="utf-8")
        result = sandbox.start(timeout=20)
        assert result.returncode == 1, f"{name}: {result.stdout + result.stderr}"
        assert "cleanup-failed marker" in result.stderr
        assert marker.read_text(encoding="utf-8") == before
        assert sandbox.fake_ollama_pid() is None
        assert not sandbox.record_exists("ollama")
        marker.unlink(missing_ok=True)


def test_foreign_cleanup_failed_marker_is_preserved(
    sandbox: Sandbox,
) -> None:
    """A foreign project marker is never deleted, even for a gone PID."""
    marker = sandbox.write_cleanup_failed_marker(
        DEAD_PID, project_root="/some/other/project"
    )
    before = marker.read_bytes()
    result = sandbox.start(timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "another project root" in result.stderr
    assert marker.read_bytes() == before
    assert sandbox.fake_ollama_pid() is None
    assert not sandbox.record_exists("ollama")
    assert not sandbox.record_exists("backend")



@pytest.mark.parametrize("kind", ["marker", "guard"])
def test_symlinked_persistent_state_is_preserved(
    sandbox: Sandbox, kind: str,
) -> None:
    "A symlinked marker or guard is never verified, removed or replaced."
    target = sandbox.state / f"symlink-target-{kind}"
    target.write_text("external target content\n", encoding="utf-8")
    if kind == "marker":
        link = sandbox.cleanup_failed_path()
    else:
        link = sandbox.bootstrap_guard_path()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    before = target.read_text(encoding="utf-8")
    try:
        result = sandbox.start(timeout=20)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "is a symlink" in result.stderr
        assert link.is_symlink()
        assert target.read_text(encoding="utf-8") == before
        _assert_no_startup(sandbox)
    finally:
        link.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def test_existing_bootstrap_guard_blocks_before_any_child(
    sandbox: Sandbox,
) -> None:
    """A pre-existing/malformed guard fails closed before any fork."""
    guard = sandbox.write_bootstrap_guard(info_text="")
    result = sandbox.start(timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "bootstrap guard" in result.stderr
    _assert_no_startup(sandbox)
    assert guard.is_dir()
    assert (guard / "info").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("reason", ["custom reason", " ", ""])
def test_guard_with_nonfixed_reason_is_preserved(
    sandbox: Sandbox, reason: str,
) -> None:
    "The guard REASON must match the fixed text exactly."
    guard = sandbox.write_bootstrap_guard(reason=reason)
    before = (guard / "info").read_text(encoding="utf-8")
    result = sandbox.start(timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot be verified" in result.stderr
    assert (guard / "info").read_text(encoding="utf-8") == before
    _assert_no_startup(sandbox)


def test_guard_creation_failure_prevents_child_fork(
    sandbox: Sandbox,
) -> None:
    """If persistent protection cannot be claimed, no child is started."""
    result = sandbox.start(
        env_overrides={"TEACHABLE_TEST_FORCE_GUARD_FAIL": "1"},
        timeout=20,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "injected bootstrap guard creation failure" in result.stderr
    assert "could not establish the bootstrap guard" in result.stderr
    assert not sandbox.bootstrap_guard_path().exists()
    _assert_no_startup(sandbox)


def test_valid_bootstrap_guard_only_blocks_before_services(
    sandbox: Sandbox,
) -> None:
    """Guard without a PID record blocks; it is never auto-removed."""
    guard = sandbox.write_bootstrap_guard(run_id="crashed-run")
    before = (guard / "info").read_text(encoding="utf-8")
    result = sandbox.start(timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "bootstrap guard" in result.stderr
    assert "without a complete cleanup-failed record" in result.stderr
    assert (guard / "info").read_text(encoding="utf-8") == before
    _assert_no_startup(sandbox)


def test_foreign_bootstrap_guard_is_preserved(sandbox: Sandbox) -> None:
    guard = sandbox.write_bootstrap_guard(project_root="/some/other/project")
    before = (guard / "info").read_text(encoding="utf-8")
    result = sandbox.start(timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "another project root" in result.stderr
    assert (guard / "info").read_text(encoding="utf-8") == before
    _assert_no_startup(sandbox)


def test_malformed_bootstrap_guard_is_preserved(sandbox: Sandbox) -> None:
    guard = sandbox.write_bootstrap_guard(info_text="not-a-field\n")
    before = (guard / "info").read_text(encoding="utf-8")
    result = sandbox.start(timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "cannot be verified" in result.stderr
    assert (guard / "info").read_text(encoding="utf-8") == before
    _assert_no_startup(sandbox)


def test_successful_launch_and_cleanup_release_bootstrap_guard(
    sandbox: Sandbox,
) -> None:
    guard = sandbox.bootstrap_guard_path()
    marker = sandbox.cleanup_failed_path()
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        assert not guard.exists()
        assert not marker.exists()
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    assert not guard.exists()
    assert not marker.exists()
    sandbox.kill_all()


def test_bootstrap_guard_blocks_after_marker_temp_failure(
    sandbox: Sandbox,
) -> None:
    """If the full marker cannot even be written, the guard still blocks."""
    _configure_bootstrap_fakes(sandbox, FAKE_STUBBORN_OLLAMA_CLI)
    guard = sandbox.bootstrap_guard_path()
    marker = sandbox.cleanup_failed_path()
    pid: int | None = None
    env = dict(_bootstrap_launch_env())
    env["TEACHABLE_TEST_FORCE_MARKER_TEMP_FAIL"] = "1"
    try:
        with sandbox.launch(**env) as launcher:
            pid = _release_bootstrap_child(sandbox, sequence="S,-")
            out, err, rc = _wait_for_launcher_exit(launcher)
            assert rc == 1, out + err
            assert "injected cleanup-failed marker temporary-file failure" in err
            assert "bootstrap guard remains" in err
            assert guard.is_dir()
            assert not marker.exists()
            os.kill(pid, 0)
            (sandbox.state / "fake-ollama.pid").unlink(missing_ok=True)

        guard_before = (guard / "info").read_text(encoding="utf-8")
        blocked = sandbox.start(timeout=20)
        assert blocked.returncode == 1, blocked.stdout + blocked.stderr
        assert "bootstrap guard" in blocked.stderr
        assert "without a complete cleanup-failed record" in blocked.stderr
        assert "Starting the project-local Ollama" not in blocked.stdout
        assert (guard / "info").read_text(encoding="utf-8") == guard_before
        assert not marker.exists()
        os.kill(pid, 0)
    finally:
        _cleanup_bootstrap_case(sandbox, pid)


def test_backend_quarantine_keeps_successful_service_cleanup_and_pid(
    sandbox: Sandbox,
) -> None:
    "One service cleans up; the quarantined one keeps its real PID."
    backend_sig = (
        f"{sandbox.project}/.venv/bin/python -m uvicorn "
        "backend.main:app --host 127.0.0.1 --port 8000"
    )
    holder = _BASH_SHEBANG + r'''set -euo pipefail
printf '%s\n' "$$" > "$SANDBOX/state/backend-shell.pid"
: > "$SANDBOX/state/backend-shell-held"
until [ -f "$SANDBOX/state/backend-release" ]; do
    sleep 0.02
done
printf '%s\n' "$*" > "$SANDBOX/state/backend-started"
printf '%s\n' "$$" > "$SANDBOX/state/backend-started.pid"
trap '' TERM
while :; do
    sleep 0.05
done
'''
    _write_executable(
        sandbox.project / ".venv" / "bin" / "python", holder
    )
    pid: int | None = None
    try:
        with sandbox.launch(**_bootstrap_launch_env()) as launcher:
            _wait_for_path(sandbox.state / "backend-shell-held", timeout=10)
            pid = _read_positive_pid(sandbox.state / "backend-shell.pid")
            sandbox.break_lstart_for(pid)
            sandbox.declare_process(pid, backend_sig)
            sandbox.set_ps_state_sequence(pid, "S,-")
            (sandbox.state / "backend-release").write_text(
                "release\n", encoding="utf-8"
            )
            out, err, rc = _wait_for_launcher_exit(launcher)
            assert rc == 1, out + err
            assert "bootstrap-child cleanup" in err
            assert f"Inspect PID {pid}" in err
            marker = sandbox.cleanup_failed_path()
            assert marker.is_file()
            fields = _marker_fields(marker)
            assert fields.get("PID") == str(pid)
            assert fields.get("SERVICE") == "backend"
            assert fields.get("PROJECT_ROOT") == str(sandbox.project)
            assert fields.get("RUN_ID")
            assert "FINGERPRINT=" not in marker.read_text(encoding="utf-8")
            # The successfully started Ollama service was stopped and its
            # record/ownership persisted before the launcher exited.
            assert not sandbox.record_exists("ollama")
            assert not sandbox.record_exists("backend")
            assert not sandbox.bootstrap_guard_path().exists()
            os.kill(pid, 0)
            # Keep the residual backend alive through context teardown.
            (sandbox.state / "backend-started.pid").unlink(missing_ok=True)

        marker_before = marker.read_text(encoding="utf-8")
        sandbox.declare_process(pid, backend_sig)
        blocked = sandbox.start(timeout=20)
        assert blocked.returncode == 1, blocked.stdout + blocked.stderr
        assert "cleanup-failed marker" in blocked.stderr
        assert "Starting the project-local Ollama" not in blocked.stdout
        assert marker.read_text(encoding="utf-8") == marker_before
        os.kill(pid, 0)
    finally:
        _cleanup_bootstrap_case(sandbox, pid)
        for name in (
            "backend-shell.pid",
            "backend-shell-held",
            "backend-release",
            "backend-started.pid",
            "backend-started",
        ):
            (sandbox.state / name).unlink(missing_ok=True)


def test_marker_publish_race_preserves_external_marker_and_guard(
    sandbox: Sandbox,
) -> None:
    """A marker created after the temp write must never be replaced."""
    external = subprocess.Popen(["/bin/sleep", "120"])
    signature = _bootstrap_expected_signature(sandbox)
    external_text = (
        f"PID={external.pid}\n"
        "SERVICE=ollama\n"
        f"PROJECT_ROOT={sandbox.project}\n"
        f"CMD={signature}\n"
        "RUN_ID=external-run\n"
        "REASON=bootstrap child could not be confirmed stopped\n"
    )
    hook = _make_marker_publish_hook(sandbox, external_text)
    _configure_bootstrap_fakes(sandbox, FAKE_STUBBORN_OLLAMA_CLI)
    guard = sandbox.bootstrap_guard_path()
    marker = sandbox.cleanup_failed_path()
    pid: int | None = None
    env = dict(_bootstrap_launch_env())
    env["TEACHABLE_MARKER_PREPUBLISH_HOOK"] = str(hook)
    try:
        with sandbox.launch(**env) as launcher:
            pid = _release_bootstrap_child(sandbox, sequence="S,-")
            out, err, rc = _wait_for_launcher_exit(launcher)
            assert rc == 1, out + err
            assert marker.read_text(encoding="utf-8") == external_text
            assert guard.is_dir()
            assert list(sandbox.runtime.glob("cleanup-failed.tmp.*")) == []
            assert external.poll() is None
            os.kill(pid, 0)
            (sandbox.state / "fake-ollama.pid").unlink(missing_ok=True)

        guard_before = (guard / "info").read_text(encoding="utf-8")
        blocked = sandbox.start(timeout=20)
        assert blocked.returncode == 1, blocked.stdout + blocked.stderr
        assert "different runs" in blocked.stderr
        assert "Starting the project-local Ollama" not in blocked.stdout
        assert marker.read_text(encoding="utf-8") == external_text
        assert (guard / "info").read_text(encoding="utf-8") == guard_before
        assert external.poll() is None
    finally:
        if external.poll() is None:
            external.kill()
        external.wait(timeout=10)
        _cleanup_bootstrap_case(sandbox, pid)


def test_marker_publish_race_directory_target_preserves_guard_and_directory(
    sandbox: Sandbox,
) -> None:
    "A directory created at the exact marker path is never a link container."
    _configure_bootstrap_fakes(sandbox, FAKE_STUBBORN_OLLAMA_CLI)
    signature = _bootstrap_expected_signature(sandbox)
    hook, payload = _make_marker_publish_directory_hook(sandbox)
    guard = sandbox.bootstrap_guard_path()
    marker = sandbox.cleanup_failed_path()
    expected = {"payload": ("file", payload)}
    pid: int | None = None
    env = dict(_bootstrap_launch_env())
    env["TEACHABLE_MARKER_PREPUBLISH_HOOK"] = str(hook)
    try:
        with sandbox.launch(**env) as launcher:
            pid = _release_bootstrap_child(sandbox, sequence="S,-")
            out, err, rc = _wait_for_launcher_exit(launcher)
            assert rc == 1, out + err
            assert "could NOT be confirmed stopped" in err
            assert "was stopped and reaped" not in err
            assert marker.is_dir()
            assert _dir_snapshot(marker) == expected
            assert list(sandbox.runtime.glob("cleanup-failed.tmp.*")) == []
            assert guard.is_dir()
            assert not sandbox.record_exists("ollama")
            os.kill(pid, 0)
            (sandbox.state / "fake-ollama.pid").unlink(missing_ok=True)

        guard_before = (guard / "info").read_text(encoding="utf-8")
        sandbox.declare_process(pid, signature)
        blocked = sandbox.start(timeout=20)
        assert blocked.returncode == 1, blocked.stdout + blocked.stderr
        assert "cleanup-failed marker" in blocked.stderr
        assert "Starting the project-local Ollama" not in blocked.stdout
        assert (guard / "info").read_text(encoding="utf-8") == guard_before
        assert _dir_snapshot(marker) == expected
        os.kill(pid, 0)
    finally:
        _cleanup_bootstrap_case(sandbox, pid)


def test_marker_publish_race_symlink_to_directory_never_writes_outside(
    sandbox: Sandbox,
) -> None:
    "A symlinked marker pointing at a directory must not be followed."
    external = sandbox.root / "external-marker-target"
    external.mkdir(parents=True, exist_ok=True)
    (external / "payload.bin").write_bytes(b"external payload bytes\n")
    expected = _dir_snapshot(external)

    _configure_bootstrap_fakes(sandbox, FAKE_STUBBORN_OLLAMA_CLI)
    signature = _bootstrap_expected_signature(sandbox)
    hook = _make_marker_publish_symlink_hook(sandbox, external)
    guard = sandbox.bootstrap_guard_path()
    marker = sandbox.cleanup_failed_path()
    pid: int | None = None
    env = dict(_bootstrap_launch_env())
    env["TEACHABLE_MARKER_PREPUBLISH_HOOK"] = str(hook)
    try:
        with sandbox.launch(**env) as launcher:
            pid = _release_bootstrap_child(sandbox, sequence="S,-")
            out, err, rc = _wait_for_launcher_exit(launcher)
            assert rc == 1, out + err
            assert "could NOT be confirmed stopped" in err
            assert "was stopped and reaped" not in err
            assert marker.is_symlink()
            assert os.readlink(marker) == str(external)
            assert _dir_snapshot(external) == expected
            assert list(sandbox.runtime.glob("cleanup-failed.tmp.*")) == []
            assert guard.is_dir()
            assert not sandbox.record_exists("ollama")
            os.kill(pid, 0)
            (sandbox.state / "fake-ollama.pid").unlink(missing_ok=True)

        guard_before = (guard / "info").read_text(encoding="utf-8")
        blocked = sandbox.start(timeout=20)
        assert blocked.returncode == 1, blocked.stdout + blocked.stderr
        assert "is a symlink" in blocked.stderr
        assert "Starting the project-local Ollama" not in blocked.stdout
        assert (guard / "info").read_text(encoding="utf-8") == guard_before
        assert marker.is_symlink()
        assert os.readlink(marker) == str(external)
        assert _dir_snapshot(external) == expected
        os.kill(pid, 0)
    finally:
        _cleanup_bootstrap_case(sandbox, pid)
        shutil.rmtree(external, ignore_errors=True)


def test_trusted_gone_removes_matching_marker_and_guard(
    sandbox: Sandbox,
) -> None:
    """Crash after marker publish but before guard removal is recoverable."""
    run_id = "crashed-run"
    guard = sandbox.write_bootstrap_guard(run_id=run_id)
    marker = sandbox.write_cleanup_failed_marker(DEAD_PID, run_id=run_id)
    with sandbox.launch() as launcher:
        launcher.wait_ready()
        assert not guard.exists()
        assert not marker.exists()
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 0, out + err
    assert not guard.exists()
    assert not marker.exists()


_INTEGER_SETTINGS = [
    "TEACHABLE_LSTART_RETRIES",
    "TEACHABLE_SELF_LSTART_RETRIES",
    "TEACHABLE_READY_ATTEMPTS",
    "TEACHABLE_OLLAMA_STOP_TIMEOUT",
    "TEACHABLE_BACKEND_STOP_TIMEOUT",
]
_NUMBER_SETTINGS = [
    "TEACHABLE_LSTART_INTERVAL",
    "TEACHABLE_SELF_LSTART_INTERVAL",
    "TEACHABLE_READY_INTERVAL",
]
_PORT_SETTINGS = ["TEACHABLE_OLLAMA_PORT", "TEACHABLE_APP_PORT"]
_BAD_PS_STATES = ["garbage", "???", "123", "S unexpected"]


def _assert_no_startup(sandbox: Sandbox) -> None:
    assert sandbox.fake_ollama_pid() is None
    assert not sandbox.record_exists("ollama")
    assert not sandbox.record_exists("backend")
    assert not sandbox.has_flag("backend-started")


@pytest.mark.parametrize("name", _INTEGER_SETTINGS)
def test_start_rejects_non_integer_config(
    sandbox: Sandbox, name: str,
) -> None:
    result = sandbox.start(
        env_overrides={name: "abc"}, timeout=20
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "invalid configuration" in result.stderr
    assert name in result.stderr
    _assert_no_startup(sandbox)


@pytest.mark.parametrize(
    "value", ["", "0", "-1", "1.5", "1+1"],
)
def test_start_rejects_invalid_retry_counts(sandbox: Sandbox, value: str) -> None:
    result = sandbox.start(
        env_overrides={"TEACHABLE_LSTART_RETRIES": value}, timeout=20
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "invalid configuration" in result.stderr
    assert "TEACHABLE_LSTART_RETRIES" in result.stderr
    _assert_no_startup(sandbox)


@pytest.mark.parametrize(
    "value", ["", "0", "-0.1", "NaN", "1e-3", "1+1"],
)
def test_start_rejects_invalid_intervals(sandbox: Sandbox, value: str) -> None:
    result = sandbox.start(
        env_overrides={"TEACHABLE_LSTART_INTERVAL": value}, timeout=20
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "invalid configuration" in result.stderr
    assert "TEACHABLE_LSTART_INTERVAL" in result.stderr
    _assert_no_startup(sandbox)


@pytest.mark.parametrize("name", _PORT_SETTINGS)
def test_start_rejects_invalid_ports(sandbox: Sandbox, name: str) -> None:
    for value in ("", "0", "65536", "abc", "-1", "1.5"):
        result = sandbox.start(
            env_overrides={name: value}, timeout=20
        )
        assert result.returncode == 1, f"{name}={value}: {result.stderr}"
        assert "invalid configuration" in result.stderr
        assert name in result.stderr
        _assert_no_startup(sandbox)


@pytest.mark.parametrize("name", _NUMBER_SETTINGS)
def test_start_validates_every_interval_name(
    sandbox: Sandbox, name: str,
) -> None:
    result = sandbox.start(env_overrides={name: "1e-3"}, timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "invalid configuration" in result.stderr
    assert name in result.stderr
    _assert_no_startup(sandbox)


@pytest.mark.parametrize("value", ["", "0", "-1", "abc", "1.5"])
def test_stop_rejects_invalid_timeout_before_signal(
    sandbox: Sandbox, value: str,
) -> None:
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, signature)
        before = record.read_text(encoding="utf-8")
        result = sandbox.stop(
            env_overrides={"TEACHABLE_STOP_TIMEOUT": value}
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "invalid configuration" in result.stderr
        assert "TEACHABLE_STOP_TIMEOUT" in result.stderr
        assert proc.poll() is None
        assert record.read_text(encoding="utf-8") == before
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


def test_stop_rejects_invalid_port_before_signal(sandbox: Sandbox) -> None:
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, signature)
        before = record.read_text(encoding="utf-8")
        result = sandbox.stop(
            env_overrides={"TEACHABLE_APP_PORT": "65536"}
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "invalid configuration" in result.stderr
        assert "TEACHABLE_APP_PORT" in result.stderr
        assert proc.poll() is None
        assert record.read_text(encoding="utf-8") == before
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


@pytest.mark.parametrize("bad_state", _BAD_PS_STATES)
def test_malformed_ps_state_keeps_lock_and_never_starts(
    sandbox: Sandbox, bad_state: str,
) -> None:
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120")
    lock = _write_lock(sandbox, _valid_lock_text(sandbox, owner.pid))
    before = (lock / "info").read_text(encoding="utf-8")
    sandbox.set_ps_state_sequence(owner.pid, bad_state)
    try:
        result = sandbox.start(timeout=20)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "malformed state" in result.stderr
        assert lock.is_dir()
        assert (lock / "info").read_text(encoding="utf-8") == before
        assert owner.poll() is None
        _assert_no_startup(sandbox)
    finally:
        sandbox.clear_ps_state_sequence()
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


@pytest.mark.parametrize("bad_state", _BAD_PS_STATES)
def test_malformed_ps_state_never_signals_or_deletes_record(
    sandbox: Sandbox, bad_state: str,
) -> None:
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, signature)
        before = record.read_text(encoding="utf-8")
        sandbox.set_ps_state_sequence(proc.pid, bad_state)
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "malformed state" in result.stderr
        assert "Stopping FastAPI backend" not in result.stdout
        assert proc.poll() is None
        assert record.exists()
        assert record.read_text(encoding="utf-8") == before
    finally:
        sandbox.clear_ps_state_sequence()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


_MACOS_VALID_ALIVE_STATES = ["RWNA", "RE", "RV", "S+", "Ss"]
_MACOS_VALID_ZOMBIE_STATES = ["Z", "ZX"]
_MACOS_INVALID_STATES = ["A", "D", "W", "RI", "Rn", "R*"]


@pytest.mark.parametrize("state", _MACOS_VALID_ALIVE_STATES)
def test_macos_ps_state_start_alive_blocks_as_running_launcher(
    sandbox: Sandbox, state: str,
) -> None:
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120", state=state)
    lock = _write_lock(sandbox, _valid_lock_text(sandbox, owner.pid))
    before = (lock / "info").read_text(encoding="utf-8")
    try:
        result = sandbox.start(timeout=20)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "another launcher is already running" in result.stderr
        assert (lock / "info").read_text(encoding="utf-8") == before
        assert owner.poll() is None
        _assert_no_startup(sandbox)
    finally:
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


@pytest.mark.parametrize("state", _MACOS_VALID_ZOMBIE_STATES)
def test_macos_ps_state_start_zombie_blocks_as_unknown(
    sandbox: Sandbox, state: str,
) -> None:
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120", state=state)
    lock = _write_lock(sandbox, _valid_lock_text(sandbox, owner.pid))
    before = (lock / "info").read_text(encoding="utf-8")
    try:
        result = sandbox.start(timeout=20)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "cannot be checked" in result.stderr
        assert "is a zombie" in result.stderr
        assert (lock / "info").read_text(encoding="utf-8") == before
        assert owner.poll() is None
        _assert_no_startup(sandbox)
    finally:
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


@pytest.mark.parametrize("state", _MACOS_INVALID_STATES)
def test_macos_ps_state_start_invalid_blocks_as_unknown(
    sandbox: Sandbox, state: str,
) -> None:
    owner = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(owner.pid, "/bin/sleep 120", state=state)
    lock = _write_lock(sandbox, _valid_lock_text(sandbox, owner.pid))
    before = (lock / "info").read_text(encoding="utf-8")
    try:
        result = sandbox.start(timeout=20)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "malformed state" in result.stderr
        assert (lock / "info").read_text(encoding="utf-8") == before
        assert owner.poll() is None
        _assert_no_startup(sandbox)
    finally:
        sandbox.clear_declared()
        owner.terminate()
        owner.wait(timeout=10)


@pytest.mark.parametrize("state", _MACOS_VALID_ALIVE_STATES)
def test_macos_ps_state_stop_alive_terminates_managed_process(
    sandbox: Sandbox, state: str,
) -> None:
    proc, signature = _matching_backend_process(sandbox)
    # Suppress the fixture's auto-discovered backend row so the declared
    # state token is what the fake ps actually reports.
    _suppress_auto_backend_row(sandbox)
    sandbox.declare_process(proc.pid, signature, state=state)
    record = sandbox.write_pid_record("backend", proc.pid, signature)
    try:
        time.sleep(0.7)
        result = sandbox.stop()
        assert result.returncode == 0, result.stdout + result.stderr
        assert sandbox.wait_for_pid_exit(proc.pid, timeout=10)
        assert not record.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "state", _MACOS_VALID_ZOMBIE_STATES + _MACOS_INVALID_STATES,
)
def test_macos_ps_state_stop_zombie_or_invalid_never_signals(
    sandbox: Sandbox, state: str,
) -> None:
    proc, signature = _matching_backend_process(sandbox)
    # Suppress the fixture's auto-discovered backend row so the declared
    # state token is what the fake ps actually reports.
    _suppress_auto_backend_row(sandbox)
    sandbox.declare_process(proc.pid, signature, state=state)
    record = sandbox.write_pid_record("backend", proc.pid, signature)
    before = record.read_text(encoding="utf-8")
    try:
        time.sleep(0.7)
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Stopping FastAPI backend" not in result.stdout
        if state in _MACOS_VALID_ZOMBIE_STATES:
            assert "is a zombie" in result.stderr
        else:
            assert "malformed state" in result.stderr
        assert proc.poll() is None
        assert record.read_text(encoding="utf-8") == before
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


def test_foreign_lock_is_preserved_even_when_owner_gone(
    sandbox: Sandbox,
) -> None:
    lock = _write_lock(
        sandbox,
        f"PID={DEAD_PID}\n"
        "RUN_ID=foreign-run\n"
        f"FINGERPRINT={sandbox._lstart_for(int(DEAD_PID))}\n"
        "PROJECT_ROOT=/some/other/project\n",
    )
    before = (lock / "info").read_text(encoding="utf-8")
    result = sandbox.start(timeout=20)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "another project root" in result.stderr
    assert lock.is_dir()
    assert (lock / "info").read_text(encoding="utf-8") == before
    _assert_no_startup(sandbox)



# ---------------------------------------------------------------------------
# 10. lsof fail-closed behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rc", "out", "err", "should_start"),
    [
        # rc=1 + empty stdout/stderr  -> free -> the launcher proceeds
        (1, "", "", True),
        # everything else must fail closed
        (2, "", "lsof: unknown option", False),
        (126, "", "", False),
        (127, "", "", False),
        (0, "", "", False),
        (0, "not-a-pid\n", "", False),
        (0, "0\n", "", False),
        (0, "12abc\n", "", False),
        # rc=0 + a valid PID but non-empty stderr is still unknown
        (0, "12345\n", "lsof: something odd", False),
    ],
)
def test_lsof_fail_closed_states(
    sandbox: Sandbox,
    rc: int,
    out: str,
    err: str,
    should_start: bool,
) -> None:
    "Strict lsof table: only the documented free answer may proceed."
    sandbox.set_lsof_override(8000, rc, out, err)
    # Reuse an external healthy Ollama so the 8000 check is exercised.
    external = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.listen_marker(11435, external.pid)
    sandbox.flag("ollama-ready")
    sandbox.declare_process(external.pid, "/bin/sleep 120")
    try:
        if should_start:
            with sandbox.launch() as launcher:
                launcher.wait_ready()
                launcher.signal(signal.SIGINT)
                launcher_out, launcher_err, launcher_rc = launcher.wait_exit()
            assert launcher_rc == 0, launcher_out + launcher_err
            assert "cannot determine whether port 8000" not in launcher_err
        else:
            result = sandbox.start()
            assert result.returncode != 0
            # Either the port check itself is unknown, or (when the fake
            # lsof reported a valid PID) the launcher refuses to reuse an
            # unverified listener.
            assert (
                "cannot determine whether port 8000 is in use" in result.stderr
                or "port 8000 is already in use" in result.stderr
            )
            assert "was not started" not in result.stdout
    finally:
        sandbox.clear_declared()
        external.terminate()
        external.wait(timeout=10)
        sandbox.clear_lsof_override(8000)


# ---------------------------------------------------------------------------
# 11. stop failures keep records; cleanup reports failure
# ---------------------------------------------------------------------------


def test_start_cleanup_keeps_pid_record_when_sigkill_fails(
    sandbox: Sandbox,
) -> None:
    """A managed backend that survives everything: the record stays."""
    sig = (
        f"{sandbox.project}/.venv/bin/python -m uvicorn "
        "backend.main:app --host 127.0.0.1 --port 8000"
    )
    proc, _ = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, sig)
    try:
        time.sleep(0.7)
        record = sandbox.write_pid_record("backend", proc.pid, sig)
        # A "ghost" keeps reporting the PID as alive even after the real
        # process dies, so TERM/KILL can never be confirmed successful.
        sandbox.set_ghost_process(proc.pid, sig)
        result = sandbox.stop(env_overrides={"TEACHABLE_STOP_TIMEOUT": "1"})
        assert result.returncode == 1, result.stdout + result.stderr
        assert "escalating to SIGKILL" in result.stderr
        assert "Keeping the PID record" in result.stderr
        assert record.exists()
        assert sandbox.record_field("backend", "PID") == str(proc.pid)
    finally:
        sandbox.clear_ghost_processes()
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        record.unlink(missing_ok=True)


def test_cleanup_reports_failure_when_managed_process_remains(
    sandbox: Sandbox,
) -> None:
    """start-demo.sh exits non-zero when its own backend cannot be stopped."""
    sig = (
        f"{sandbox.project}/.venv/bin/python -m uvicorn "
        "backend.main:app --host 127.0.0.1 --port 8000"
    )
    with sandbox.launch(
        TEACHABLE_BACKEND_STOP_TIMEOUT="1",
        TEACHABLE_OLLAMA_STOP_TIMEOUT="1",
    ) as launcher:
        launcher.wait_ready()
        backend_pid = sandbox.backend_pid()
        assert backend_pid
        sandbox.set_ghost_process(backend_pid, sig)
        launcher.signal(signal.SIGINT)
        out, err, rc = launcher.wait_exit()
    assert rc == 1, out + err
    assert "Keeping the PID record" in err
    assert f"PID {backend_pid}" in err
    assert "Inspect PID  " not in err
    assert sandbox.record_exists("backend")
    assert sandbox.record_field("backend", "PID") is not None
    # The backend failed, but the Ollama service was confirmed stopped and
    # its record/ownership must be persisted and cleared before exit.
    assert not sandbox.record_exists("ollama")
    # A verifiable PID record is already the durable trace: no quarantine
    # marker is needed for this recorded failure.
    assert not (sandbox.runtime / "cleanup-failed").exists()
    sandbox.clear_ghost_processes()
    sandbox.kill_all()


# ---------------------------------------------------------------------------
# 12. strict PID record validation
# ---------------------------------------------------------------------------


def _base_record(sandbox: Sandbox, pid: int | str) -> str:
    sig = (
        f"{sandbox.project}/.venv/bin/python -m uvicorn "
        "backend.main:app --host 127.0.0.1 --port 8000"
    )
    fp = sandbox._lstart_for(int(pid)) if str(pid).isdigit() else "fp"
    return (
        "# comment\n"
        f"PID={pid}\n"
        "SERVICE=backend\n"
        f"PROJECT_ROOT={sandbox.project}\n"
        f"CMD={sig}\n"
        f"FINGERPRINT={fp}\n"
        "RUN_ID=test-run\n"
    )


def _assert_never_signals(sandbox: Sandbox, victim: subprocess.Popen) -> None:
    assert victim.poll() is None


def test_missing_project_root_never_signals(sandbox: Sandbox) -> None:
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    record = _base_record(sandbox, victim.pid).replace(
        f"PROJECT_ROOT={sandbox.project}\n", ""
    )
    path = sandbox.raw_pid_record("backend", record)
    try:
        result = sandbox.stop()
        assert "Refusing to signal" in result.stderr
        _assert_never_signals(sandbox, victim)
        assert path.exists()  # unverifiable records are kept
    finally:
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)
        path.unlink(missing_ok=True)


def test_missing_service_never_signals(sandbox: Sandbox) -> None:
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    record = _base_record(sandbox, victim.pid).replace(
        "SERVICE=backend\n", ""
    )
    path = sandbox.raw_pid_record("backend", record)
    try:
        result = sandbox.stop()
        assert "Refusing to signal" in result.stderr
        _assert_never_signals(sandbox, victim)
        assert path.exists()
    finally:
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)
        path.unlink(missing_ok=True)


def test_wrong_service_never_signals(sandbox: Sandbox) -> None:
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    record = _base_record(sandbox, victim.pid).replace(
        "SERVICE=backend\n", "SERVICE=ollama\n"
    )
    path = sandbox.raw_pid_record("backend", record)
    try:
        result = sandbox.stop()
        assert "does not belong to this record file" in result.stderr
        _assert_never_signals(sandbox, victim)
        assert path.exists()
    finally:
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)
        path.unlink(missing_ok=True)


def test_missing_cmd_never_signals(sandbox: Sandbox) -> None:
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    record = _base_record(sandbox, victim.pid)
    record = record.replace(f"CMD={sandbox.project}/.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000\n", "")
    path = sandbox.raw_pid_record("backend", record)
    try:
        result = sandbox.stop()
        assert "Refusing to signal" in result.stderr
        _assert_never_signals(sandbox, victim)
        assert path.exists()
    finally:
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)
        path.unlink(missing_ok=True)


def test_recorded_cmd_mismatch_never_signals(sandbox: Sandbox) -> None:
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    record = _base_record(sandbox, victim.pid).replace(
        "backend.main:app --host 127.0.0.1 --port 8000",
        "something.else --host 127.0.0.1 --port 9999",
    )
    path = sandbox.raw_pid_record("backend", record)
    try:
        result = sandbox.stop()
        assert "recorded CMD does not match" in result.stderr
        _assert_never_signals(sandbox, victim)
        assert path.exists()
    finally:
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)
        path.unlink(missing_ok=True)


def test_duplicate_record_fields_never_signal(sandbox: Sandbox) -> None:
    victim = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.declare_process(victim.pid, "/bin/sleep 120")
    record = _base_record(sandbox, victim.pid)
    record += "RUN_ID=another-run\n"
    path = sandbox.raw_pid_record("backend", record)
    try:
        result = sandbox.stop()
        assert "Refusing to signal" in result.stderr
        _assert_never_signals(sandbox, victim)
        assert path.exists()
    finally:
        sandbox.clear_declared()
        victim.terminate()
        victim.wait(timeout=10)
        path.unlink(missing_ok=True)


def test_same_command_pid_reuse_with_different_start_fingerprint_never_signals(
    sandbox: Sandbox,
) -> None:
    """Same PID, same command line, but a different process start time."""
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        # A record whose fingerprint is NOT this process's start time.
        path = sandbox.write_pid_record(
            "backend", proc.pid, signature,
            fingerprint="Mon Jan 01 00:00:00 2001",
        )
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Refusing to signal" in result.stderr
        # The real process is untouched despite the same command line.
        assert proc.poll() is None
        assert path.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()
        path.unlink(missing_ok=True)


def test_fingerprint_unreadable_fails_closed_on_stop(
    sandbox: Sandbox,
) -> None:
    """If the platform cannot read lstart, stop-demo.sh never signals."""
    proc, signature = _matching_backend_process(sandbox)
    sandbox.declare_process(proc.pid, signature)
    try:
        time.sleep(0.7)
        sandbox.write_pid_record("backend", proc.pid, signature)
        sandbox.flag("ps-lstart-fails")
        result = sandbox.stop()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "Refusing to signal" in result.stderr
        assert proc.poll() is None
    finally:
        sandbox.unflag("ps-lstart-fails")
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        sandbox.clear_declared()


@pytest.mark.parametrize("which", ["ollama", "backend"])
def test_fingerprint_unreadable_fails_closed_on_start(
    sandbox: Sandbox, which: str,
) -> None:
    """A bootstrap child whose start fingerprint cannot be read is stopped
    and reaped through the strictly limited bootstrap-child path.

    The launcher's fingerprint retry window is widened so the test can make
    the child's lstart unreadable at a deterministic point; the child then
    still appears in ps (for the command-line identity check) but can never
    be fingerprinted.
    """
    env = sandbox.env(TEACHABLE_LSTART_RETRIES="10",
                      TEACHABLE_LSTART_INTERVAL="0.5")
    # Verify only the guard-bound fallback: no service row may be published
    # by the background watcher, and the lstart fault is installed before the
    # child is forked.
    sandbox.suppress_service_meta = True
    sandbox.set_ps_probe_log()
    sandbox.break_bootstrap_lstart(which)
    proc = subprocess.Popen(
        ["/bin/bash", str(sandbox.project / "scripts" / "start-demo.sh")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd="/",
    )
    sandbox.declare_harness_process(
        proc.pid, f"/bin/bash {sandbox.project}/scripts/start-demo.sh"
    )
    watcher_stop = sandbox.start_meta_watcher()
    broken_pid: int | None = None
    handshake_pid = sandbox.state / (
        "ollama-bootstrap-pid" if which == "ollama" else "backend-bootstrap-pid"
    )
    try:
        deadline = time.time() + 40
        while time.time() < deadline:
            if handshake_pid.exists():
                raw = handshake_pid.read_text(encoding="utf-8").strip()
                if raw.isdigit():
                    broken_pid = int(raw)
                    break
            if proc.poll() is not None:
                break
            time.sleep(0.01)
        assert broken_pid, f"{which} bootstrap handshake never appeared"
        out, err = proc.communicate(timeout=120)
    finally:
        watcher_stop.set()
        sandbox.forget_harness_process(proc.pid)
        sandbox.clear_break_lstart_for()
        sandbox.clear_break_bootstrap_lstart(which)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert proc.returncode != 0, out + err
    assert "process start fingerprint" in err
    assert "bootstrap-child cleanup" in err
    # Prove the lstart probe reached the guard-bound fallback and was
    # rejected there; it did not succeed through a watcher-published row.
    probes = sandbox.ps_probe_log()
    assert f"{broken_pid}|lstart=|guard-lstart-fault" in probes
    assert f"{broken_pid}|" not in (
        sandbox.state / "ps-meta"
    ).read_text(encoding="utf-8")
    # Honest reporting: the child was confirmed stopped, not "verified".
    assert "was stopped and reaped" in err
    assert "could NOT be confirmed stopped" not in err

    # 1. no PID record, guard or quarantine marker were left behind
    assert not sandbox.record_exists("ollama")
    assert not sandbox.record_exists("backend")
    assert not (sandbox.runtime / "cleanup-failed").exists()
    assert not sandbox.bootstrap_guard_path().exists()
    # 2. the just-started child really exited
    assert sandbox.wait_for_pid_exit(broken_pid, timeout=15), (
        f"the {which} bootstrap child was left running"
    )
    # 3. it was terminated by this launcher's cleanup (not by the test)
    if which == "ollama":
        assert broken_pid in sandbox.ollama_terminated_pids()
        # 4. the backend was never started after the Ollama failure
        assert sandbox.backend_pid() is None
        assert not sandbox.has_flag("backend-started")
    # 5. the launcher released the lock it had acquired
    assert not (sandbox.runtime / "demo.lock").exists()
    sandbox.kill_all()


def test_fingerprint_unreadable_own_process_fails_closed_on_start(
    sandbox: Sandbox,
) -> None:
    """If the launcher cannot fingerprint itself, every ps read is broken and
    it must refuse to hold a lock at all."""
    sandbox.flag("ps-lstart-fails")
    result = sandbox.start()
    assert result.returncode != 0
    # Fail closed before creating a lock or starting any service.
    assert "cannot report this process's start time" in result.stderr
    assert not (sandbox.runtime / "demo.lock").exists()
    assert not sandbox.record_exists("ollama")
    assert not sandbox.record_exists("backend")
    assert sandbox.fake_ollama_pid() is None
    sandbox.unflag("ps-lstart-fails")


# ---------------------------------------------------------------------------
# 13. external Ollama and pre-existing records
# ---------------------------------------------------------------------------


def test_external_ollama_does_not_touch_existing_pid_records(
    sandbox: Sandbox,
) -> None:
    """Reusing an external Ollama must not overwrite or delete a pre-existing
    record at the ollama pid path, and stop-demo.sh must not signal the
    external process because of it."""
    external = subprocess.Popen(["/bin/sleep", "120"])
    sandbox.listen_marker(11435, external.pid)
    sandbox.flag("ollama-ready")
    sandbox.declare_process(external.pid, "/bin/sleep 120")

    # A pre-existing record for a dead PID.
    old = sandbox.write_pid_record(
        "ollama", DEAD_PID, run_id="ancient-run",
        fingerprint="Thu Jan 01 00:00:00 1970",
    )
    old_text = old.read_text(encoding="utf-8")
    try:
        with sandbox.launch() as launcher:
            launcher.wait_ready()
            assert old.read_text(encoding="utf-8") == old_text
            launcher.signal(signal.SIGINT)
            out, err, rc = launcher.wait_exit()
        assert rc == 0, out + err
        # The launcher never adopted or removed the old record.
        assert old.read_text(encoding="utf-8") == old_text

        # stop-demo.sh only touches the dead-PID stale record and never
        # signals the external process.
        stop = sandbox.stop()
        assert external.poll() is None, "external Ollama was signalled"
        assert stop.returncode == 0, stop.stdout + stop.stderr
        assert not old.exists()  # stale (dead PID) record is cleaned up
        assert sandbox.ollama_terminated_pids(timeout=1) == []
    finally:
        sandbox.clear_declared()
        external.terminate()
        external.wait(timeout=10)
