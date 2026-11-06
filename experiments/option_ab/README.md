# Option A+B isolated experiment package

This directory is a separate A+B condition. It does not modify production,
Option A, Option B, or either web harness.

## Assets

- `prompts/base.txt` is byte-identical to the shared A/B base.
- `prompts/ab_strategy.txt` is the independent A+B strategy.
- `prompts/system.txt` is assembled exactly as `base.rstrip() + "\n\n" + ab_strategy.rstrip() + "\n"`.
- `scripts/scripts.json` defines S1–S6 and I1–I4.
- `manifest.json` is generated only after all frozen assets and tests exist.

The fixed S1–S6 user inputs are byte-identical to A/B. The interactive I2/I3
branches are A+B-specific and must be selected by a human, never by a model,
regex, question mark, or keyword classifier.

## Budget

- Fixed S1–S6: 6 runs / 16 calls.
- I1: 2 calls.
- I2: 2–3 calls.
- I3: 2–3 calls.
- I4: 2 calls.
- Main batch: 10 runs / 24–26 calls.

A failed batch may stop early. Do not pad calls.

## Running fake validation

```bash
.venv/bin/python -m unittest discover -s experiments/option_ab/tests -v
.venv/bin/python -m experiments.option_ab.freeze_manifest --verify
```

The default runner uses only `FakeLLMClient`; it performs no network call.

```bash
.venv/bin/python -m experiments.option_ab.runner \
  --output-dir /tmp/option-ab-runs \
  --branch I2=bc --branch I3=rule_supply
```

`none` is a runtime branch and is not stored as a script branch. Other I2/I3
branches require an explicit `--branch`; omitting one pauses that interactive
run. No hidden Q2, retry, or automatic model classification is permitted.

## Evidence

Each run has an exclusive run directory with `run_meta.json` and JSONL records
for calls and turns. Recorded fields separate model self-report from human
review, keep request/upstream model distinct, and mark fake/synthetic evidence.
A failed or cancelled run is never reported as completed.

This package contains no real-model evidence. Its tests validate assets,
state/ledger behavior, and fake-run evidence only; they do not show that a real
model follows the A+B strategy.

The standalone fake runner checks text/empty turns and ledger state; it does not
classify `finish_reason` because the fake client exposes only text. Truncation,
non-stop and other completion-status checks belong to the web recording layer
and are not claimed for this independent runner.
