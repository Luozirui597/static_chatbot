# Option B isolated experiment

This directory is a self-contained, fake-client-only experiment package for
Option B. Option B receives the same shared base as Option A and a B strategy
that **never authorizes inference**. It may report direct teaching, state that
the current teaching is insufficient, and ask at most one focused question.

This package does **not** implement a B web layer. It does not import a
production LLM client, read credentials, open a socket, or touch the
application database or service.

## Files

| Path | Purpose |
|---|---|
| `prompts/base.txt` | Shared base, byte-for-byte identical to Option A's base. |
| `prompts/b_strategy.txt` | Complete B policy block. |
| `prompts/system.txt` | Exact assembly: base + blank line + B strategy + final LF. |
| `scripts/scripts.json` | Fixed S1-S6 plus interactive I1-I3, including all branch text. |
| `history.py` | Receive-shaped history assembly (last 20 stored rows, assistant filtering, JSON background, current user separate). |
| `runner.py` | Bounded fake runner and append-only evidence writer. |
| `freeze_manifest.py` | Full manifest builder/verifier. |
| `expected/per_turn_expectations.md` | Per-turn behavior, branch, pause, and question rules. |
| `expected/coding_sheet.csv` | Human coding sheet including optional branch rows. |
| `tests/test_option_b.py` | Direct isolation tests using only fakes/temp directories. |
| `manifest.json` | Frozen hashes/bytes and call budget. |

Frozen source hashes before manifest generation:

- shared base: 12436 bytes,
  `eda835c58da3c1cbd05d83d5ac34f4ff5bbd805b8a466796957b7b489e00bb6f`
- B strategy: 2513 bytes,
  `c2c72910d8a8381c43bbd6ff4b02d5aaf9e1d1d5957cd05b83be58305676a273`
- B system: 14950 bytes,
  `44aed8837a13413565a6dee866e00671a6fcefb3cc6f5f885dd78ac8495f9113`

The frozen prompt version is generated as
`option-b-isolated-v0.1+sys-44aed8837a13` (39 characters, within the production
`prompt_version_snapshot` limit of 50).

## Fixed flow

S1-S6 each have their own logical run and fresh storage/client:

- S1: 2 calls
- S2: 2 calls
- S3: 2 calls
- S4: 2 calls
- S5: 4 calls (T1, Q1, correction T2, Q2)
- S6: 4 calls (T1, Q1, revocation T2, Q2)

Fixed total: **16 calls**. Every fixed T1/Q1/T2/Q2 turn produces one generated response; fixed scripts have no teacher-supplement turns and no extra calls.

## Interactive flow

I1 always has 2 calls and no branch. I2/I3:

1. T1 and Q1 are generated.
2. The runner stops with `paused`; it does not classify Q1 automatically.
3. A human records `none`, or a selected branch. Classification is based on the Q1 assistant's actual request, not the later teacher text.
4. `none` ends at 2 calls with no T2.
5. A branch sends exactly one self-contained T2. The assistant response to T2
   is FINAL; it is not a new user turn, is not separately armed, and there is
   no interactive Q2. The T2 request and FINAL response are one generation.
6. Any branch makes the logical run 3 calls; `none` makes it 2 calls.

Branch values:

- I2: `none`, `bc`, `direct_ac`, `no_extra`
- I3: `none`, `rule_request`, `direct_ac`, `no_extra`

Branch texts are in `scripts/scripts.json`; the runner never infers a branch
from a question mark, keyword, or regex. A neutral request for B-C, the
combination rule, or direct teaching of A-C maps to `bc`, `rule_request`, or
`direct_ac` respectively. An actual request that is a candidate
confirmation, unsupported premise, generic request, or mixed request maps to
`no_extra`; those categories are recorded separately, mixed requests are not
routed by a dominant part, and candidate confirmation does not automatically
count as answer leakage or supply a new A-C fact. Only no actual supplement
request is `none`. Request type and compliance judgment are separate from the
teacher-supply branch; classification remains human-only.

## Budget

| Group | Runs | Calls |
|---|---:|---:|
| Fixed S1-S6 | 6 | 16 |
| I1 | 1 | 2 |
| I2 | 1 | 2-3 |
| I3 | 1 | 2-3 |
| **Total** | **9** | **22-24** |

There are no retries, no model-classification calls, and no repair messages.

## Commands

All commands are local/fake. Generate and verify the manifest before running:

```bash
.venv/bin/python -m experiments.option_b.freeze_manifest
.venv/bin/python -m experiments.option_b.freeze_manifest --verify
.venv/bin/python -m unittest discover -s experiments/option_b/tests -v
```

Run the fake batch without a branch (I2/I3 pause after Q1; 22 calls total):

```bash
.venv/bin/python -m experiments.option_b.runner \
  --run-prefix fake-no-branch \
  --output-dir /tmp/option-b-fake-no-branch
```

Run a complete fake batch with manual decisions (24 calls total):

```bash
.venv/bin/python -m experiments.option_b.runner \
  --run-prefix fake-with-branches \
  --output-dir /tmp/option-b-fake-with-branches \
  --branch I2=bc \
  --branch I3=direct_ac
```

Use `none` for a branch decision that adds no information. Omitting a branch
leaves that I2/I3 logical run paused, not failed.

These commands use `FakeLLMClient`. A successful fake run proves only that the
runner, pause boundary, hashes, records, and budget are wired correctly. It is
not evidence that a real model follows B.

## Records

Each logical run has a fresh directory under `runs/` (or the supplied output
root), created exclusively. `turns.jsonl` is append-only and contains one
record per actual fake call; `run_meta.json` records the final status, calls,
client, prompt version, system hash, branch decision, and pause state.

Records include:

- exact request messages and canonical request hash;
- exact visible response and response hash;
- `synthetic: true` for fake clients;
- `null` real-model, generation, and message-id fields with an explicit reason;
- model self-report fields left `null` (the runner does not parse trust claims);
- human-review fields left `null`, including candidate-confirmation,
  unsupported-premise, generic, and mixed-request distinctions;
- branch classification is `selected_by: human`, never inferred by the runner;
- run-local source links and branch/event data;
- `final_assistant` for a branch T2 response; the T2 user and FINAL assistant
  are linked to one generation, with no Q2.

Existing run directories and record files are never overwritten. A failure
stops recording and no retry is attempted.

## Isolation

The package imports no `backend`, `frontend`, production configuration, or
real LLM client. It does not read `.env`. Tests use temporary directories and
deterministic fakes. The production A package, A web layer, backend, frontend,
configuration, and database are untouched.

## Not included

This round does not include:

- a B web/server integration;
- real API/model clients;
- a real application acceptance run;
- browser automation;
- database writes or migrations;
- a branch classifier;
- a real-model claim that B never infers.

Those require a later explicit authorization and the same separation of frozen
assets from runtime evidence.
