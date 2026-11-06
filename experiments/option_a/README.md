# Option A isolated experiment

This directory contains an isolated, fake-client implementation of the frozen
Option A behavior contract. It is **not** wired into the production application,
does not implement Option B or A+B, and does not call a real model.

The purpose is to make the subsequent real-model pilot reviewable:
the prompt, scripts, expected calls, request history shape, and stored records
are frozen before any real output exists.

## Scope and safety

- Only `experiments/option_a/` is added. Production backend/frontend prompts,
  modes, configuration, database, and existing math-rendering changes are not
  modified.
- The default and only built-in client is a deterministic local fake. The
  runner imports no API/local-model client and does not read `.env`, tokens, or
  other credentials.
- The default test suite also avoids importing `backend.chat_service` (and thus
  `backend.config`/dotenv); it compares the two production history functions by
  loading their source definitions with `ast`.
- No network request is made by the fake-client path.
- The runner never writes the application database and never creates a
  production session.
- This is not a pilot execution. No real model output is generated here.

## Layout

```text
experiments/option_a/
  README.md
  manifest.json                 # frozen hashes and call counts
  freeze_manifest.py            # write/verify manifest
  history.py                    # production-shaped Receive history assembly
  runner.py                     # fake-client runner + exclusive run records
  prompts/
    base.txt                    # shared experimental base
    a_strategy.txt              # Option A policy
    system.txt                  # exact base + A assembly
  scripts/
    scripts.json                # machine-readable source of truth
    S1.txt ... S6.txt           # human-readable copies
  expected/
    per_turn_expectations.md
    coding_sheet.csv            # 16 rows, one per expected model call
  tests/
    test_option_a.py
  runs/                         # not populated by tests; exclusive run dirs
```

`scripts.json` is the machine-readable source used by the runner. The `S*.txt`
files are readable copies for review; the runner does not parse them.

## Frozen behavior

The shared base preserves the v7-style learner role, language priority,
faithfulness, correction handling, safety, mode control, and natural expression.
It removes the production prohibition on derivation only through the experiment
prompt family. The A strategy then limits derivation to:

- a visible, teacher-provided relation rule `R1`;
- at most one rule application, one conclusion, and two relation facts;
- every listed condition and exception checked from visible teaching;
- no common-sense transitivity, multi-step closure, contrapositive, symmetry,
  mathematical solving, or invented premises;
- `[inferred]` plus the exact premises and rule when inference occurs;
- direct teaching preferred and never labeled `[inferred]`;
- missing information stated and stopped, not turned into a request for teaching
  content;
- explicit negations, corrections, active exceptions, and revoked rules not
  bypassed by asking for new facts or a replacement rule.

This is prompt behavior only. The runner does not semantically verify the
model's answer. The coding sheet is for human review.

## History semantics

`history.py` mirrors the current Receive path:

- the last 20 **stored rows** (user + assistant) are selected first;
- historical assistant rows are dropped;
- remaining user rows are packed into one JSON user background message
  (`receive_teaching_history_background`, version `history-context-b1`);
- the current user message is appended separately and is not part of the
  20-row window.

This means the model does not see its own prior assistant replies, and the
background `index` is not a stable database message id.

## Scripts and call budget

The six scripts are frozen in `scripts/scripts.json`.

| Script | Turns | Expected calls | Focus |
|---|---:|---:|---|
| S1 | 2 | 2 | direct, inferred, cannot determine |
| S2 | 2 | 2 | missing B-C premise |
| S3 | 2 | 2 | combination rule not provided |
| S4 | 2 | 2 | active exception |
| S5 | 4 | 4 | licensed inference, then correction |
| S6 | 4 | 4 | licensed inference, then revocation |

One call is expected for every user turn. Teacher branches are not executed by
this runner. A question in a model reply never creates an extra call here.

## Run the isolated tests

From the repository root:

```bash
.venv/bin/python -m unittest discover -s experiments/option_a/tests -v
```

These tests use temporary directories and the fake client only. They cover:

- exact prompt assembly and hashes;
- six scripts and 16 expected calls;
- exact Receive history projection and background shape;
- current user not being counted in the 20-row window or duplicated;
- two/four-turn execution and record schema;
- fail-fast behavior with retained completed records;
- blank response as an error with its raw string retained (no response = null);
- wrong system hash rejected before a client call or run directory;
- exclusive run directories (no overwrite);
- a question in a reply not adding a call;
- default `run_all` using the fake client for 16 calls and not rewriting the
  manifest;
- manifest preflight rejecting missing files, changed schema/condition, a
  changed system, and a changed runner before any client is created;
- production history-source loading without `backend.config`/dotenv imports.

## Run the local fake experiment

This writes new run directories under `runs/`; it does not call a real model:

```bash
.venv/bin/python -m experiments.option_a.runner \
  --asset-root experiments/option_a \
  --output-dir experiments/option_a/runs
```

Each run gets a unique `run_id`. The directory is created exclusively and
contains:

- `turns.jsonl`: one append-only JSON object per attempted model call. Records
  include the client kind and `synthetic` flag, model/profile placeholders,
  requested generation parameters, response metadata placeholders, run-local
  source/event links, and separate model-self-report vs human-review sections.
  Human review also has null fields for verified source turn/record ids;
- `run_meta.json`: run-level status, prompt/script hashes, expected/actual calls,
  and the same client/model/source context.

For the fake path, `synthetic=true` and model/profile/response-metadata fields
are null with a reason. `source_message_id` is always null; correction and
revocation records use run-local record ids instead. A blank response is a
failed record whose `response` still preserves the raw string; a client
exception/no response records `response=null`. Neither path retries.

Existing run directories are never reused or overwritten. A failure is recorded
and stops that run; there is no automatic retry. A run that fails before all
turns has fewer records than expected; that is intentional.

## Freeze or verify the manifest

```bash
.venv/bin/python -m experiments.option_a.freeze_manifest
.venv/bin/python -m experiments.option_a.freeze_manifest --verify
```

The manifest records SHA-256 hashes and byte counts for the frozen assets,
including `freeze_manifest.py` itself, prompt assembly, and the expected 16
calls. `--verify` compares the complete frozen content (except the timestamp),
not only the file entries currently listed in the manifest. `run_all` performs
this verification before creating a client; a mismatch raises and produces zero
generations. Normal runs do not rewrite the manifest.

## Handoff boundary

This asset is ready for a later, separately authorized real-model pilot. It does
not establish:

- that a real model follows the A policy;
- that `[inferred]` output is a reliable provenance mechanism;
- that the behavior generalizes to other relations or mathematical tasks;
- that Option A should be integrated into the production application.

The real pilot should use the recorded prompt/script hashes, the same history
shape, and human coding. It must be explicitly authorized and should not reuse
the fake run records as evidence of model behavior.
