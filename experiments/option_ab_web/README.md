# Option A+B isolated web test harness

This directory is a test-only adapter for the frozen Option A+B experiment package
in `experiments/option_ab/`.  It does not implement Option A+B as a production
interaction mode and it does not change the production application.

The automated suite uses fake clients and temporary databases. Historical real
web pilots are recorded separately in the project review workspace; an isolated
test pass is not a new real-model result.

## Frozen experiment compatibility (2026-11-05)

After database isolation and manifest checks, this launcher explicitly injects
`FrozenReceiveTeachingPolicy` into its service instance. The exact frozen prompt
and version remain the contract; strict recorder hashes and snapshot checks are
preserved. Language adherence is observed/scored under the original prompt,
without the production `l1` instruction, version suffix, or pre-write language
rejection. The old prompts, manifests, and pilot evidence are unchanged.

New run metadata records `receive_policy.kind=frozen-experiment-v1`,
`reply_language=frozen-prompt-only`, and `history_tools=false`. This policy is
constructor-only, cannot be selected through HTTP/environment settings, and
cannot be combined with history retrieval services. Production ChatService
instances omit it and retain the backend language constraints.

## Boundary

- Only `experiments/option_ab_web/` is added.
- `backend/`, `frontend/`, `experiments/option_a/`,
  `experiments/option_a_web/` and `experiments/option_b/` are not modified.
- The real `backend.main.app`, `ChatService`, messages table, Receive history
  assembly and the real frontend assets are used by the test instance.
- Only the test process replaces the Receive prompt/version resolver and the
  test profile/client.  The session mode remains `receive_teaching`.
- `DATABASE_URL` is set to a SQLite file outside the repository before any
  `backend` import.  `.env` loading is disabled during preparation.
- No real credential is read by the default test path.  The real launcher
  requires explicit `--real-api` and credentials supplied by an operator.

## Files

| File | Role |
| --- | --- |
| `launcher.py` | preflight, isolated environment/DB, runtime Prompt override, test HTML, PID/stop identity |
| `state.py` | ten-run state machine, sessions, turn order, branch selection, budget |
| `recording.py` | actual request/response and diagnostic evidence |
| `asgi.py` | test-only ASGI wrapper: read-only allow list, arm/branch/snapshot controls, message gate |
| `tests/test_web_layer.py` | fake-client and in-process ASGI isolation tests |

## What is reused from `option_a_web`

The isolation pattern and evidence approach are adapted, not copied blindly:
repository-external run directory and SQLite, pre-import environment guard,
manifest preflight, runtime Receive prompt/version override, an in-memory test
HTML wrapper, an instrumented client, and stop-by-verified-PID.

The fixed-16-turn `TurnGate` from A is not used for AB.  AB has ten independent
runs, two human-pause points, and one-shot branch T2 turns.

## Preflight and runtime replacement

Before importing `backend`:

1. `experiments.option_ab.freeze_manifest.load_and_verify_manifest()` verifies
   the frozen A+B package, system hash and script counts.
2. A fresh run directory is created outside the repository; a SQLite
   `test.sqlite3` and evidence directory are created there.
3. `DATABASE_URL` points at that file, `LOCAL_LLM_ENABLED=false`, and the test
   process refuses to continue if any `backend.*` module is already loaded.

After import, the launcher checks the engine and `SessionLocal` binding, builds
one test profile and a real `ChatService`, replaces `main._profiles` and
`main.chat_service`, and patches only the Receive prompt/version getters.  It
restores those objects and the environment on close.

## Start a real test instance

This is the later Codex/operator step.  It is not executed by the isolation
tests.

```bash
cd /Users/luo/Documents/teachable_agent
LLM_API_KEY=... \
LLM_API_BASE_URL=... \
LLM_MODEL=... \
.venv/bin/python -m experiments.option_ab_web.launcher \
  --real-api \
  --run-dir /Users/luo/Documents/teachable_agent_runs/option-ab-web/<unique-run> \
  --host 127.0.0.1 --port 8765
```

- The run directory must be outside the repository and must not already exist.
- Credentials stay in the process environment.  Do not place them in a URL,
  command-line argument, README, browser page, or evidence file.
- Missing credentials cause a refusal; there is no automatic fake fallback.
- The launcher uses one Uvicorn worker and no reload.
- The generated page is a test-only copy of the current `frontend/index.html`
  with an always-visible A+B banner.  `/static` and `/api` still reach the real
  app.  The banner does not hide or rewrite assistant replies.

The page exposes only a test control surface; it is not a production feature.
It has a fixed top banner (condition AB, instance, prompt version, system hash,
current state) and a branch panel used only after an I2/I3 Q1.  I2/I3 each follow one human-selected path in the main batch; this is not full branch coverage.

## Normal run sequence

Ten independent sessions, 16 fixed calls plus 8-10 interactive calls:

| Run | Calls |
| --- | --- |
| S1-S6 | 2/2/2/2/4/4 |
| I1 | 2 |
| I2, I3 | 2 or 3 each |
| I4 | 2 |
| total | 24-26 |

For each ordinary turn:

1. Call `POST /__option_ab__/arm` with the instance, test session, script and
   turn.  This arms a turn; it does not call a model.
2. Send the frozen user text through the real frontend Send flow.  The wrapper
   atomically consumes the arm and forwards one message request to the real
   app.
3. The wrapper requires exactly one generation, verifies the response IDs and
   snapshots, writes evidence, and only then advances the state.

`POST /api/sessions` is allowed up to ten times.  Use a new session for each
script.  Do not use `/api/chat`.

### I2/I3 human pause

After Q1 of I2 or I3, the state becomes
`paused_human_classification`.  The branch panel appears.  This is a human
decision, not a model decision: no question mark, regex or keyword classifier
is used.

Send `POST /__option_ab__/branch` with:

```json
{
  "instance_id": "...",
  "session_id": 2,
  "script_id": "I2",
  "branch_id": "bc",
  "q1_assistant_message_id": 123,
  "selected_by": "human",
  "note": "neutral request for B-C"
}
```

- `none`: completes the run with two calls and no T2.
- Any other allowed branch: the state atomically selects the branch, arms the
  one frozen T2, and returns its exact text.  Send that text once through the
  real Send button.  The resulting assistant reply is FINAL; there is no Q2.
- Repeating a selection, switching branches, or arming a second T2 is rejected.
  A rejected competing request does not close or overwrite the first record.
- Allowed branch IDs are read from `/__option_ab__/state`; do not invent one.

## Evidence

The run directory contains `test.sqlite3` and `evidence/`:

- `run_meta.json`
- `calls.jsonl` - model calls, request/response hashes, original and effective
  parameters, request model, upstream model (or null plus reason), diagnostics
  without secrets or hidden reasoning.
- `http.jsonl` - message/control HTTP outcomes and associated message IDs.
- `snapshots.jsonl` - per-turn snapshot checks.
- `branches.jsonl` - human branch decisions and Q1 association.
- `screenshots/` is available for a later Codex browser run.

A failure, cancellation, duplicate call, missing message, snapshot mismatch,
empty/truncated/unconfirmed completion, or evidence-write failure stops the
batch.  There is no retry and no synthetic FINAL.

## Stop

Use the run-bound stopper, never `pkill`:

```bash
.venv/bin/python -c \
'from pathlib import Path; from experiments.option_ab_web.launcher import stop_instance; stop_instance(Path("/absolute/run/dir"))'
```

The stopper checks the PID start identity, run directory, recorded launch
arguments, run metadata hash and that the PID owns the test port before sending
SIGTERM.  If identity is not proven, it refuses to signal the process.  The
test database and evidence are retained.

## Isolation tests

```bash
.venv/bin/python -m unittest discover -s experiments/option_ab_web/tests -v
.venv/bin/python -m py_compile experiments/option_ab_web/*.py experiments/option_ab_web/tests/*.py
```

The suite uses fake clients, temporary SQLite and in-process ASGI; it does not
start a listening server or call a real model.  A passing fake test does not
prove real model behavior.

## Known limits

- The production OpenAI-compatible client does not expose the upstream response
  model; evidence records `null` and a reason for that field.
- `generate()` has no `timeout` argument.  The real client is constructed with a
  30-second timeout and the wrapper also applies a 30-second deadline; no
  production interface is changed.
- The test page is a test-only HTML wrapper around the actual frontend files.
  Structural tests are not a real browser acceptance test.
- Upstream/model behavior, real credentials, service isolation and browser
  acceptance remain Codex/operator work.
