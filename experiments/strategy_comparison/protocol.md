# Real-run protocol (for Codex/authorized human)

This file is operational guidance, not an API implementation.  No real call is
made by this repository's fake tests.

## Preconditions

- Current production code and a separate test instance/database.
- Temporary SQLite or another isolated test database; never the business DB.
- Same model, profile, generation parameters, English language policy, and
  current production language validation for all four conditions.
- Fresh browser/session for each cell; no cross-condition conversation reuse.
- A human present for I2/I3 branch selection.

## Per-session sequence

```
POST  /api/sessions
PATCH /api/sessions/{id}/teaching-strategy
POST  /api/sessions/{id}/messages   # T1
POST  /api/sessions/{id}/messages   # Q1
[pause: human branch decision]
POST  /api/sessions/{id}/messages   # optional branch T2 only
```

The strategy must be set before the first message.  The returned strategy must
equal the plan entry.  The `teaching_strategy` switch values are in
`conditions.json`; do not use the display label as the API value.

## Branch selection worksheet

Read Q1's assistant text, not the group label:

- asks for the missing B–C relation -> `bc`;
- asks for the missing combination rule -> `rule_supply`;
- asks for A–C directly -> `direct_ac`;
- asks something but receives no new fact/rule -> `no_extra`;
- no actual supplement request -> `none`.

Record the exact request type and rationale.  If the question is mixed, contains
an unsupported premise, repeats a question, or is unrelated, preserve that
specific violation; do not choose a branch merely to make the run continue.
`none` is a completed two-call session, not a crash.

## Failure handling

- HTTP error, timeout, language block, malformed response, or unknown call
  count: record it and stop that session.
- Do not automatically retry, resend a branch, or add a hidden Q2.
- Keep partial user/assistant messages and status.
- A failed session is part of the denominator for executed/failed reporting.

## Files and evidence

Real transcripts and scoring should be stored outside the repository under a
new run directory.  Do not overwrite prior evidence.  Do not put API keys,
authorization headers, `.env`, raw provider error bodies, or hidden reasoning in
the evidence.

## Budget reconciliation

Fixed: (4 tasks × 2 calls + 2 tasks × 4 calls) × 4 conditions × 3 repeats = 192.
Interactive: 24 sessions × 2 calls, plus one extra call for every actual branch:
48–72.  Total 240–264 for a normally completed batch. A failed/aborted batch may be lower; retain actual records and do not top up.  Pre-pilot is 32 sessions, 80–88 calls, and is excluded
if scripts/prompts/scoring change after it.

## Reporting

Report planned/executed/completed/scorable sessions separately; per-task,
per-condition results; paired within-block differences; repeated-run
consistency; failures, timeouts, language blocks, and call counts.  Do not claim
statistical significance or general superiority from three repeats per cell.
Do not claim A+B synergy from this design.
