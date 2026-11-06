# Controlled strategy-stability experiment (protocol)

Historical design proposal: the active, separately versioned query-only pilot
is frozen under `controlled_v1/PROTOCOL.md`, with 32 sessions and 40-44 calls.
The old call counts and human-branching design below are not its run plan.

Status: protocol and engineering prototype only.  No real run has been
executed in this phase.  The old pre-pilot (32 sessions / 80 calls) is not
part of this protocol's denominators.

## Research questions

1. Does a server-side R1 verifier prevent unverified assistant text from being
   saved without creating an unacceptable false-rejection rate?
2. With the verifier held fixed, do Standard/A/B/A+B differ in strategy
   compliance, task completion, evidence correctness, question quality,
   remediation continuation, and cost?
3. Are any observed differences stable across independent conversations and
   held-out task variants?
4. Does allowing a question improve task completion when the task genuinely
   needs one, separately from whether a strategy spontaneously asks?

These are engineering/behavioral questions.  The prototype and fake tests
answer only the first; they do not establish the others.

## Conditions and frozen contracts

Four product conditions are compared as whole strategies, not as a clean 2x2
factorial:

- **Standard**: actual frozen contract; no extra inference or question grant.
- **A**: one constrained R1 inference.
- **B**: no inference; one focused question or direct target request.
- **A+B**: one R1 inference or one focused gap question; no direct target
  request.

Common user inputs and evidence are identical across conditions.  Do not add
"do not infer" to B's user text or the correct gap to A+B's user text.  Policy
belongs in the system contract.

## Server-owned candidate and context

The model returns a strict JSON candidate, never final prose.  A new candidate
may contain up to four items: direct claims, one R1 inference, insufficient
unknown targets, and at most one ask/clarify item.  A question item must be the
last item, and the whole reply has one R1 budget.  Server-owned `TurnContext`
and `GoalContext` provide the original target, current subgoals, askable gaps,
turn kind, permissions, and required targets.  The model's `ask.kind` cannot
change those permissions.  `Standard` keeps its real necessary-clarification
permission via an explicit server `allow_clarify`; it is not made a blanket
no-question strategy.

## Tasks

The primary known-task set is eight tasks:

- S1 sufficient teaching; direct vs legal inference vs unknown;
- S2 missing intermediate fact; state gap and stop;
- S3 missing composition rule; do not assume transitivity;
- S4 explicit active exception;
- S5 corrected fact; re-evaluate;
- S6 revoked rule; re-evaluate;
- I2 missing intermediate fact; one focused question and one remediation;
- I3 missing composition rule; one focused question and one remediation.

S1–S6 reuse the common user wording after byte-level confirmation.  I2/I3 use
new group-independent text.  The known set is development-exposed; results
there are repeat measurements, not unseen-task generalization.

Held-out variants (separate phase):

- new names and order;
- a distractor sharing only a generic word;
- a negation variant;
- a condition/exception variant;
- a rule correction represented as revocation.

Lexical-only variants are labeled as such; they cannot support a claim of
open-task or structural generalization.

## Teacher branching

Branching is chosen by a human from the actual question semantics, never by
punctuation, keywords, regex, or the strategy label.

| Actual assistant request | Teacher action |
|---|---|
| focused missing fact question | supply exactly that fact |
| focused missing rule question | supply exactly that rule |
| direct target request | direct_target branch |
| no actual request | none; end without unsolicited teaching |
| unrelated, repetitive, unsupported premise, or mixed gaps | no_extra; retain violation code |

A branch is not forced by strategy.  B/A+B may legitimately say
"insufficient" and stop; no question is not automatically a violation.

When a matching question is answered, the remediation message must be
self-contained, refer to the original task, and ask for continuation.  The
next assistant response is FINAL.  There is no hidden Q2, re-ask, or repair
turn.  Only a matching gap question -> matching remediation -> valid inference
counts as a complete chain.  Unsolicited remediation, direct target teaching,
or no question never count.

## Run design and pairing

Each task × repetition is a block.  Within the block, the four conditions are
randomized and the seed/order are stored.  Each condition gets a fresh
session; no history crosses blocks or strategies.

Primary phase: 8 tasks × 4 strategies × 3 repetitions = 96 sessions.

Calls per strategy/repetition under the reviewed contract:

- fixed portion: 16 = 4×2 + 2×4;
- interaction portion: 4–6 = 2×(2–3);
- total 20–22.

Fixed protocol calls: **192** = (4×2 + 2×4) × 4×3.

Interaction calls: **48–72** = 2×4×3×(2–3).

Primary total: **240–264** calls.

Held-out phase: 4 variants × 2 calls × 4 strategies × 3 repetitions =
48 sessions, **96 calls**.

Explicit-question phase (separate tasks; tests question ability, not
spontaneous asking): 2 tasks × 3 calls (question, remediation, final) ×
4 strategies × 3 repetitions = 24 sessions, **72 calls**.

Total planned real calls: **408–432**, across **168 sessions**.  The old
pre-pilot 32 sessions/80 calls are excluded.  The fixed/primary/interaction
counts above must be recomputed from the frozen script manifest before a real
run; if they differ, stop and fix the manifest rather than silently changing
the denominator.

Engineering replay phase: reuse the old frozen candidate outputs offline
(0 new model calls) to measure the verifier.  This is not a strategy trial.
Do not mix those 80 historical calls into the new behavioral denominator.

## Metrics (no single total score)

Engineering fields:

- pre-verifier candidate schema/compliance rate;
- invalid-candidate pass-through rate (must be 0);
- valid-candidate false-rejection rate;
- assistant rows written for rejected candidates (must be 0);
- end-to-end task handling rate including rejected attempts;
- calls, latency, tokens, and error categories.

Behavior fields:

- strategy compliance;
- direct answer correctness;
- inference correctness: claim/rule/condition/exception/evidence IDs and
  quote-to-fact correspondence;
- question focus, answerability, neutrality, and unsupported-premise status;
- requested remediation kind vs actual teacher remediation kind;
- continuation of the original task;
- complete question -> remediation -> inference chain;
- extra/repeated questions and direct-target requests;
- calls, remediation count, latency;
- HTTP status, language rejection, timeout, empty response.
- `[inferred]` and HTTP 200 are never treated as correctness.

Report separate denominators: planned, executed, valid output, scoreable,
task success.  A strategy can be compliant but not complete the task; that is
not automatically a violation.  Asking is not automatically good; report
question use and compliance separately.

## Blind scoring

Stage 1 uses anonymous objective artifacts (teaching, question, evidence,
candidate/rendered response, branch, outcomes) with strategy labels removed.
Lock Stage 1 before unblinding.  Stage 2 adds the public strategy contract and
scores compliance.  Use independent human raters for content and compliance;
record disagreements/adjudication.  LLM judges, if used, are auxiliary and
never the final label.

## Budget and stop rules

Hard stop immediately on any of:

- verifier accepts an invalid action/claim or renders unvalidated prose;
- an assistant row is written for a rejected/invalid candidate;
- more than one generation in a turn or a second unplanned call;
- an interaction/question/remediation limit is exceeded;
- a prompt version exceeds the 50-character snapshot limit or an old version
  is rewritten;
- a planned session is duplicated or a strategy/version/prompt mismatch is
  found;
- the call budget is exhausted.

Pause and review (no retry, no backfill) after:

- any 5xx/transport/timeout burst of 3 consecutive failures or more than 10%
  of planned sessions in a phase;
- a non-zero invalid-candidate pass-through or assistant-write violation
  (this is actually a hard stop);
- more than 5% unscoreable sessions in a phase.

Proceed to the next phase only if the previous phase has zero verifier
bypasses and at least 95% of sessions have scoreable outcomes.  Do not repeat
sessions to fill quotas.  All failures remain in the denominator.

## Version table by path

Current ordinary versions are strategy-specific: Standard
`receive-teaching-v7+history-context-b1+l1-en` (44), A
`teach-a-v1+history-context-b1+l1-en` (35), B
`teach-b-v1+history-context-b1+l1-en` (35), A+B
`teach-ab-v1+history-context-b1+l1-en` (36).  The programmatic fixed/general
and clarification paths currently use shared markers;
`receive-teaching-v7+programmatic-history-v1+l1-en` is a shared integration
marker, not evidence that A/B/A+B are Standard.  Proposed verified-route
markers `+r1-en`/`+r1-zh` replace the language marker (ordinary 44,
programmatic 49, clarify 36); they are design-only and are not implemented.

## Version naming

Existing `+l1-en/zh` versions are unchanged for non-verified paths.  The
verified route needs a combined marker because the programmatic version is
already 49 characters with `+l1-en`; adding `+r1` would exceed the 50-character
snapshot limit.

Proposed verified-route suffixes are `+r1-en` and `+r1-zh`, replacing (not
appending to) the old `+l1` marker:

- ordinary: `receive-teaching-v7+history-context-b1+r1-en` (44);
- fixed/general: `receive-teaching-v7+programmatic-history-v1+r1-en` (49);
- clarify: `receive-teaching-v7+clarify-v1+r1-en` (36).

`r1-en/zh` denotes the new verified route and its target language; old
snapshots are never rewritten.  This is a proposed naming decision, not an
implemented production change.

## References and what they do not prove

- ProofWriter-style natural-language rule reasoning and explicit `Unknown`:
  useful inspiration for premises/rules/unknown, not evidence that this
  project's strategies are stable or compliant.
  https://aclanthology.org/2021.findings-acl.317/
- tau-bench: independent trials and terminal task outcomes are useful design
  ideas; our verifier and strategy-compliance metrics are project additions.
  https://arxiv.org/abs/2406.12045
- tau-bench environment patterns (state transitions and user simulator):
  https://github.com/sierra-research/tau-bench/blob/main/tau_bench/envs/base.py
- Question-asking/clarification literature informs question utility, not the
  exact contract here.
- LLM-as-judge work documents judge biases; it motivates locked human
  scoring, not a claim that a judge proves strategy superiority.

No paper cited here proves this project's policy compliance, stability, or
A+B synergy.  Those claims require the new independent sessions and held-out
variants.

## Not executed in this phase

No real model, provider, web browser, production database, production
service, `.env`, or Git write was used.  The prototype has only fake/in-memory
tests.  Production integration, database-backed question/remediation state,
and real behavior waits for review.
