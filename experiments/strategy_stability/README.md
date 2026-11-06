# Strategy stability prototype

Active controlled-query pilot assets and executable budget are in
`controlled_v1/`. They use preloaded atomic teaching and supersede the older
call-budget proposal below. Old narrative experiment assets remain unchanged.

The first controlled real pilot completed 32 sessions / 44 generations. It
found stale post-remediation gap metadata affecting B's rendered text; read
`FOLLOWUP_AFTER_CONTROLLED_V1.md` before further runs. The v1 frozen assets
and original results remain unchanged.

This directory contains an isolated verifier, a temporary-SQLite
integration service, and controlled pilot tooling. It is not connected
to production `ChatService`, production databases or the chat page.
The integration reuses the existing language validator; the pure-function
design notes below describe the verifier layer. See `integration.md` and
`controlled_v1/PROTOCOL.md` for the executable integration and pilot.

The prototype answers one narrow engineering question: can a server-side
contract prevent an unverified free-text answer from being saved when the
model was only allowed to report, infer under R1, say the evidence is
insufficient, or ask one focused question?

It is not evidence that any strategy is generally more compliant.

## What is verified

The candidate is a strict JSON object.  JSON shape is not semantic evidence.
After parsing, the verifier independently checks every claim against the
teacher messages supplied for the current run:

- the source message belongs to the current session;
- it is a teacher (`user`) message, not an assistant message;
- it is in the allowed evidence set;
- its quote occurs verbatim in the source;
- the source states the same subject/object and polarity;
- it is the current (possibly corrected) statement, not a stale one.

A source ID, a model-reported confidence, or a correct-looking quote is never
treated as proof by itself.

## Candidate contract

The preferred shape is a multi-item candidate.  One reply may contain at most
four items, at most one `infer` item, and at most one question item.  A question
item must be the last item in the reply.  The whole reply shares one R1 budget
and one question budget; items are validated independently and rendered only
after every item passes.

```json
{
  "items": [
    {"type": "direct", "claims": []},
    {"type": "infer", "claims": [], "inference": {}},
    {"type": "insufficient", "unknown_target": {}},
    {"type": "ask", "ask": {}},
    {"type": "clarify", "target": {}}
  ]
}
```

The single-action shape (`action` plus its fields) remains accepted for older
unit fixtures, but new candidates should use `items`.  The parser rejects
unknown fields.  In particular, `final_text`, `answer`, `reason`, and
model-supplied prose are invalid.  The final visible answer is rendered only
by the verifier.

## Server-owned context

`TurnContext` and `GoalContext` are supplied by the server, never by the
candidate:

```python
GoalContext(
    original=Relation("A", "C"),
    current_subgoals=(Relation("A", "D"), Relation("C", "A")),
    askable_gaps=(Relation("B", "C"),),
    missing_relations=(Relation("B", "C"),),
    missing_rule=False,
)
TurnContext(
    goal=goal,
    turn_kind="question" | "teaching" | "correction" | "revocation",
    allow_infer=True,
    allow_ask=True,
    allow_clarify=False,
    required_targets=(Relation("A", "D"), Relation("A", "C"), Relation("C", "A")),
)
```

The context owns the original goal, subgoals, the gaps the teacher can actually
supply, and the current turn permissions.  A model-reported `ask.kind`, a
question mark, or a strategy label cannot create a permission.  Public policy
validation requires context for **every** action; a missing or contradictory
context is rejected before a `verified` result.  There is no public
no-context evidence-only bypass.  A+B cannot turn the original target into a
missing fact by relabeling it, and an unrelated target cannot be answered or
asked.

### Direct claim

```json
{
  "claim_id": "c1",
  "kind": "direct",
  "subject": "A",
  "predicate": "links",
  "object": "B",
  "polarity": "positive",
  "source_message_id": 1,
  "quote": "A links to B."
}
```

The source must contain the exact quote.  The relation must match exactly
(`A links B` is not `B links A`) and the polarity must match.  Direct claims
win over inference.  A directly taught conclusion may not be re-labelled as
inferred.

### R1 inference

```json
{
  "premise_claim_ids": ["c1", "c2"],
  "rule_source_message_id": 3,
  "condition_source_message_ids": [4],
  "exception_source_message_ids": [5],
  "conclusion": {
    "subject": "A",
    "predicate": "links",
    "object": "C",
    "polarity": "positive"
  }
}
```

R1 is deliberately limited to:

- exactly two relation premises;
- one prefix chain `A-B, B-C -> A-C`;
- one rule application;
- one positive conclusion;
- no transitivity closure, contraposition, symmetry, or multi-step proof;
- no unstated condition or exception.

Condition statuses must be current and cited; conditions must be true;
exceptions must be current and inactive.  A missing status is `unknown`, not
`false`.  A revoked rule cannot be used.  An ad-hoc candidate rule is not a
teacher-adopted rule and is rejected as unsupported.

### Insufficient and ask

`insufficient` requires `unknown_target` and server context.  The target must
belong to the original goal, a current subgoal, or a required target.  A direct
positive or negative answer always blocks `insufficient`.  For A/A+B, a legally
available R1 result also blocks it; B/Standard may legitimately withhold when
the only route is R1 because their contracts do not authorize inference.  The
withholding text must not claim that existing premises or rules are missing.
It never produces `False` from missing evidence.

`ask` requires server context, a target, and one of:

- `missing_fact`;
- `missing_rule`;
- `direct_target`.

The server, not the candidate, owns the question count.  The prototype's
`InteractionState` records one asked question and one remediation.  A second
question, or a question after remediation, is rejected.  A question does not
require a question mark; an action is the contract, not punctuation.

Before any ask, direct positive/negative teaching of the **served target**
blocks the question.  A `missing_fact` target must be an askable gap, never the
original target.  A `missing_rule` request is allowed only when the server
context says the rule is missing and the target is the original target.
`direct_target` is allowed for B only; A+B is always rejected.  Standard may be
granted the separate `clarify` action when the server needs a necessary
clarification, but it may not ask a content question.  A question is bound to a
server-owned target (`GoalContext.gap_targets`, defaulting to `original`), not
to the model's claim.  A direct answer followed by one final focused question is
allowed; the question must be the last item.  The whole candidate is validated
before rendering; `ask` is not an early-success path.

## Strategy table

| Strategy | direct positive/negative | R1 infer | withhold when R1 path exists | ask one gap | direct target ask | server clarify |
|---|---:|---:|---:|---:|---:|---:|
| Standard | yes | no | yes | no | no | when granted |
| A | yes | one | **reject** | no | no | when granted |
| B | yes | no | yes | yes | yes (not if known) | when granted |
| A+B | yes | one | **reject** | yes | no | when granted |

Standard's real contract still permits a necessary clarification; this prototype
models it only through an explicit server `allow_clarify`.  It is not changed
to a blanket no-question strategy.  The matrix is the policy source of truth;
a direct+one-gap-question reply is allowed only with the question last.

B may ask for the target or a missing fact/rule.  A+B may ask for a missing
fact/rule but not directly request the target answer.  Neither B nor A+B is
forced to ask; an insufficient response is allowed by the frozen contracts.
A question is not automatically a violation when the strategy has no question
permission: the verifier rejects it under that strategy.

## Validation order and save semantics

The intended production order is:

1. freeze the strategy, model, prompt version, and evidence window;
2. obtain one strict candidate (no final prose);
3. parse and validate the candidate against evidence and policy;
4. render deterministic text from verified claims only;
5. run existing language validation on that rendered text;
6. atomically save assistant text and advance interaction state;
7. on any verifier/language failure, keep the already-saved user message and
   write no assistant row; return a structured error.

The prototype only implements steps 2–4.  It does not touch the production
save path.  A verifier rejection is not a network failure and must not be
retried automatically.

## Original S1–S6/I2–I3 coverage

The old eight T1 messages are **not accepted as single evidence messages** by
the controlled grammar.  They are multi-sentence narrative prompts containing
labels, preamble text, several facts, and sometimes rules/exceptions.  The
parser deliberately accepts atomic controlled statements, not arbitrary prose.
The old assets are not modified and no result below claims byte-level reuse.

| Task | Old T1 status | Controlled evidence needed | Adaptation |
|---|---|---|---|
| S1 | unsupported as one message | A-D direct, A-B/B-C, rule | split into atomic evidence; new controlled task |
| S2 | unsupported as one message | A-B only; B-C absent | split; assert specific gap |
| S3 | unsupported as one message | A-B/B-C; no rule | split; no transitivity |
| S4 | unsupported as one message | rule + explicit condition/exception status | split; active exception fixture |
| S5 | unsupported as one message | current A-B, negative/corrected B-C | controlled correction message |
| S6 | unsupported as one message | A-B/B-C current; rule revoked | controlled revocation message |
| I2 | unsupported as one message | A-B; B-C askable gap | controlled query + remediation |
| I3 | unsupported as one message | A-B/B-C; rule askable gap | controlled query + remediation |

If a controlled task is created, it is a **new task version** with new hashes
and must not be reported as a byte-identical replay of the old T1.  The old
assets remain frozen.  The prototype's multi-item S1 test uses the controlled
split form, not the old narrative message.

## Run budget

The real primary phase is 8 tasks × 4 strategies × 3 repetitions = 96 sessions.
With the reviewed call contract:

- fixed part: **192 calls** = (4×2 + 2×4) × 4 × 3;
- interaction part: **48–72 calls** = 2×4×3×(2–3);
- primary total: **240–264 calls**.

Held-out phase: 4 variants × 2 calls × 4 × 3 = **96 calls**.  Explicit-question
phase: 2 tasks × 3 calls × 4 × 3 = **72 calls**.  Grand plan: **408–432 calls**.
The fake run's 256 calls is a fixture exercise, not a real-call budget.

## Snapshot versions and paths

Current product versions (ordinary path):

| Path | Standard | A | B | A+B |
|---|---|---|---|---|
| ordinary language-suffixed | `receive-teaching-v7+history-context-b1+l1-en` (44) | `teach-a-v1+history-context-b1+l1-en` (35) | `teach-b-v1+history-context-b1+l1-en` (35) | `teach-ab-v1+history-context-b1+l1-en` (36) |
| programmatic fixed/general (current shared marker) | `receive-teaching-v7+programmatic-history-v1+l1-en` (49) | same shared path; strategy is not encoded in this marker | same shared path | same shared path |
| necessary clarification (current shared marker) | `receive-teaching-v7+clarify-v1+l1-en` (37) | same shared path | same shared path | same shared path |

A/B/A+B are **not** to be reported as ordinary Standard/`receive-teaching-v7`.
The shared programmatic/clarify markers are an integration gap, not evidence
that the strategy is Standard.  The proposed verified-route replacements are
`+r1-en`/`+r1-zh` (replacing the language marker), with lengths 44/49/36.  They
are a design proposal only, not implemented in this prototype.

## Threat model

The verifier is designed for these bypasses:

1. **Action mislabeling**: a candidate says `direct` but carries an inferred
   relation.  Evidence lookup rejects an untaught target.
2. **Split-brain prose**: valid structured fields plus unvalidated prose.
   The candidate has no final-text field; all prose is rendered server-side.
3. **Source spoofing**: fabricated IDs, wrong session, assistant messages, or
   quotes absent from the source.
4. **Relation inversion**: `A links B` used as `B links A`.
5. **Negation loss**: positive/negative polarity mismatch.
6. **Stale evidence**: old fact/rule used after a correction or revocation.
7. **Rule adoption attack**: a rule merely mentioned, quoted, or asked about
   is treated as adopted.
8. **Condition/exception laundering**: omitted, false, or active conditions.
9. **Unbounded deduction**: more than two premises or a multi-hop chain.
10. **Unknown-as-false**: missing evidence used to assert a negative relation.
11. **Question bypass**: a forbidden or repeated question expressed without
    a question mark.
12. **State spoofing**: model-asserted question/remediation counts; server
    state must own them.
13. **Strategy drift**: prompt text silently changing permission; policy is
    checked by the server contract, not inferred from the prompt.
14. **Persistence bypass**: assistant row written before verification or
    after a failed transaction.
15. **Parser/renderer mismatch**: structured fields pass but rendered output
    says something else; renderer uses only validated claims.

## What is not solved

- The evidence grammar is a controlled subset of English `links` sentences.
  Unsupported/messy natural language is rejected rather than interpreted.
- Natural-language parsing of arbitrary teaching remains outside this
  prototype.  The future adapter must run a deterministic parser whose
  output is itself reviewable; a model-generated fact table is still
  untrusted input.
- Condition names and statuses are a small closed grammar.
- Rule correction is represented by revocation; richer rule revision
  semantics are not implemented.
- The prototype uses an explicit `InteractionState`; a production database
  table/transaction is not implemented here.
- The renderer is English-only.  A future Chinese renderer must preserve the
  same claim-to-text correspondence and pass the existing language validator.
- The original T1 narrative is unsupported by the controlled grammar; a new
  atomic evidence fixture is required and must not be called a byte-level replay.
- This does not establish strategy quality, stability, or generalization.

## Running

```bash
cd /Users/luo/Documents/teachable_agent
.venv/bin/python -m pytest experiments/strategy_stability/tests -q
.venv/bin/python -m py_compile experiments/strategy_stability/verifier.py \
  experiments/strategy_stability/tests/test_verifier.py
```

The tests use only in-memory Python data and no model/API/database.
