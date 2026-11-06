# Standard / A / B / A+B strategy comparison

## Research questions

1. How often does each strategy comply with its own teaching contract?
2. When does it complete the original task using only taught facts and rules?
3. Are premises, conditions, exceptions, corrections, and citations correct?
4. Are questions focused, answerable, neutral, and free of unsupported
   premises?
5. After one supplement, does the assistant continue the original task?
6. What are the interaction and model-call costs, including failures?

This is the independent, frozen protocol for comparing the four production
Receive-teaching strategies.  It is a **known-task repeated evaluation**, not a
strict 2×2 factorial experiment.  In particular, B is allowed to request direct
target teaching while A+B is not; Standard and A have different question
permissions; their prompt structures differ.  Do not report an "A+B synergy"
from this design.

No real model, API, production service, business database, browser, `.env`, or
Git operation is used by the fake entry point in this directory.

## Conditions

| ID | Display | API switch | Prompt file | Prompt version (English) |
|---|---|---|---|---|
| `standard` | Standard | `baseline` | `prompts/standard.txt` | `receive-teaching-v7+history-context-b1+l1-en` |
| `a` | A | `a` | `prompts/a.txt` | `teach-a-v1+history-context-b1+l1-en` |
| `b` | B | `b` | `prompts/b.txt` | `teach-b-v1+history-context-b1+l1-en` |
| `ab` | A+B | `ab` | `prompts/ab.txt` | `teach-ab-v1+history-context-b1+l1-en` |

`conditions.json` contains the exact SHA-256 of each frozen prompt, the common
profile/model policy (`default`, `deepseek-v4-flash`), and the common generation
policy: no condition-specific temperature, max_tokens, or reasoning override is
sent.  The request model is fixed; provider-returned upstream model is recorded
when available.  The prompt files are the unconditional strategy prompts; the
actual system request also receives exactly one English target-language
instruction from the current production implementation.

## Task set

Eight tasks live in `tasks/tasks.json`; the user-visible texts are in
`tasks/*.txt`:

- S1: sufficient teaching; direct fact, one R1 inference, undetermined relation.
- S2: missing intermediate fact; state the gap and stop.
- S3: missing combination rule; do not assume transitivity.
- S4: active exception blocks the inference.
- S5: correction of a fact; Q2 re-evaluates.
- S6: revocation of a rule; Q2 re-evaluates.
- I2: missing B–C; one focused question and at most one supplement.
- I3: missing combination rule; one focused question and at most one supplement.

S1–S6 reuse the common user texts from the old frozen scripts.  I2/I3 are new
four-condition common texts.  No condition adds "do not infer", a correct
gap, or any strategy hint to the user message.  Strategy permission is carried
only by the system prompt.  The tasks are known development items; this is not
an unseen-task generalization claim.

Every initial teaching turn is inside the current visible window.  The
experiment deliberately does not mix long-history retrieval behavior into the
main comparison.

## Branches

`branches/branches.json` contains frozen I2/I3 supplements:

- I2 `bc`: supplies B links C.
- I3 `rule_supply`: supplies the combination rule.
- `direct_ac`: supplies the target A links C.
- `no_extra`: an actual request to which the teacher supplies no new fact/rule.
- `none`: no actual supplement request; end after Q1.

A human reads the actual Q1 assistant response and selects the branch from
meaning.  Never select by strategy, punctuation, keyword, regex, question mark,
or a model classifier.  B may make a legal direct-target request; A+B making
the same request is a target-solicitation violation even if a later answer is
correct.  A/Standard content questions may be violations under their contracts.

There is no hidden Q2.  A supplement is sent once and its assistant response is
FINAL.

## Run design

`run_plan.json` freezes:

- random seed `20261106`;
- 24 task × repetition blocks, each containing a random permutation of the
  four strategy IDs;
- 96 sessions (`8 tasks × 4 strategies × 3 repeats`);
- fixed S1–S6 cost: 16 model calls per strategy per repeat, 192 total;
- I2/I3 cost: 2 calls for `none`, 3 calls for a branch, 48–72 total;
- total planned range for a normally completed batch: 240–264 calls;
- a failed/aborted batch may be below 240; retain actual calls and failed
  sessions, and never add replacement sessions to reach the normal range;
- pre-pilot: 32 sessions and 80–88 calls, excluded from the formal result if
  the protocol changes afterwards.

The plan does not preassign I2/I3 branches.  The fake runner uses a clearly
labelled synthetic branch fixture for plumbing/budget tests only.

## API execution protocol

For each plan entry, create a **new** session. Never reuse a conversation across
conditions, tasks, or repeats.

1. `POST /api/sessions` with `{"llm_profile_id": "default"}`.
2. Before any message, call
   `PATCH /api/sessions/{id}/teaching-strategy` with
   `{"teaching_strategy": "<switch>"}` from `conditions.json`.
3. Verify the returned session has the expected strategy, `receive_teaching`
   mode, and real API profile/model.  Do not silently continue on mismatch.
4. Send each `tasks[].turns[].content` exactly once via
   `POST /api/sessions/{id}/messages` body `{"message": "..."}`.
   Do not retry; preserve every HTTP status and response.
5. For I2/I3, pause after Q1.  Select `none` or one branch from the actual
   semantic content.  `none` ends the session at two calls; a branch sends its
   content exactly once and its response is FINAL.
6. Record the returned user/assistant ids, model/version snapshots, status,
   timing, and call count.  The independent session is the analysis unit.
7. A failure is retained.  Do not rerun, replace, or supplement a failed
   session to obtain a preferred result.

The real run must use a separate test instance/database and the same current
production code, message endpoint, and strategy endpoint.  The test instance
must not point at the business database.  Real execution and browser
acceptance remain with Codex/authorized humans.

## Fake entry point

```bash
cd /Users/luo/Documents/teachable_agent
.venv/bin/python -m experiments.strategy_comparison.freeze_manifest --verify
.venv/bin/python -m experiments.strategy_comparison.runner
```

The runner verifies the frozen manifest first and runs 96 independent synthetic
sessions.  Its output is explicitly `"synthetic": true`; it is not model
behavior, a score, or evidence of strategy superiority.  It checks turn/call
accounting, branch plumbing, failure retention, session isolation, and budget.
`budget_status` is `complete` for a full batch and `partial_failure` when any
session failed; a partial batch may legitimately have fewer than 240 calls.
To save the synthetic summary, pass an explicit `--output` path outside the
frozen asset set; the runner refuses to overwrite an existing file.

Zero-network endpoint acceptance uses temporary SQLite and the current FastAPI
app with a fake client:

```bash
.venv/bin/python -m pytest experiments/strategy_comparison/tests/test_endpoint_acceptance.py -q
```

It actually creates sessions, patches each strategy, sends one message, and
reads the messages back.  It is not a substitute for the synthetic runner; the
runner is not endpoint evidence.

## Scoring

`expected/strategy_expectations.md` states the contracts and per-task expected
paths.  `scoring/scoring_guide.md` defines independent fields.

Scoring has two stages:

1. **Stage 1, blind:** `scoring/blind_coding_sheet.csv` contains one anonymous
   row per session, with answer, evidence, question, and execution fields but
   no strategy label.  Raters complete it without `scoring/blind_key.json`.
2. **Lock:** after Stage 1 and its disagreement review, record the CSV SHA-256
   in `scoring/stage1_lock.json` and set `locked=true`.
3. **Stage 2, strategy-aware:** only after the lock, use
   `scoring/blind_key.json` and `scoring/stage2_compliance_sheet.csv` to judge
   strategy compliance, branch validity, and the complete inquiry–supplement–
   inference chain.  Stage 2 is explicitly not blind; save reviewer and
   adjudication records separately.

S2 requires a gap statement and stop for all four strategies.  The Standard
Prompt's rule 7 still permits a necessary focused clarification in other tasks;
record an unclear boundary as `clarification_boundary=ambiguous` rather than
applying a blanket no-question rule.  Do not change production prompts to make
the rubric easier.

Scoring is human, not model self-scoring.  Do not collapse compliance, task
completion, evidence, question quality, continuation, and cost into one number.
`[inferred]` is a visible label, not proof of validity.  HTTP 200 is execution,
not success.
