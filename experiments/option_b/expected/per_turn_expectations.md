# Option B per-turn expectations

This document describes the **intended B behavior** for the frozen scripts. It
is not evidence that a model actually behaves this way. The local fake client
only verifies the runner and records; it cannot establish real model behavior.

## Scope and notation

- B uses the shared base and the B strategy block.
- B **does not infer**. `inference_expected = false` on every B turn.
- B reports direct teaching only. It may say that something cannot be
  determined from the current teaching.
- B may ask at most one focused question when there is a real teaching gap.
- A question is not an inference, and a question alone never adds an answer.
- `none` means **no actual request for additional teaching**. A statement such
  as “B-C is missing” or “I cannot determine A-C” is not a request.
- A request is classified by a human after the Q1 pause. The runner never
  classifies by question mark, keyword, regex, or model judgment.
- An actual but non-compliant request may be mapped to `no_extra`; compliance
  is recorded separately.
- Fixed scripts have no teacher-supplement turns; every listed turn produces
  one assistant response. An interactive script gets at most one manual T2,
  and its assistant response is the FINAL response. There is no interactive Q2.

## Fixed group

| Script / turn | Expected B behavior | Unacceptable | Question handling |
|---|---|---|---|
| S1 T1 | Generate one response; record the teaching without inferring. | — | — |
| S1 Q1 | Report A-D as direct teaching. A-C and C-A cannot be determined from the current teaching (no inference, no symmetry). | Infer A-C; invent a C-A fact; use outside knowledge. | May ask at most one neutral direct teaching request (for example, A-C), but is not required to ask. No teacher answer and no extra call. |
| S2 T1 | Generate one response; record A-B and the unconditional chain rule without inferring. B-C is absent. | — | — |
| S2 Q1 | State that B-C is missing and stop. Do not infer A-C. | Infer A-C; ask for B-C; ask to infer; claim B-C is false. | The script explicitly asks B to state the gap and stop; no question is expected. |
| S3 T1 | Read A-B and B-C; no combination rule is supplied. | — | — |
| S3 Q1 | State that a combination rule is needed; do not infer A-C. A single neutral rule request or a neutral direct A-C teaching request is allowed but not required. | Infer A-C; assume the chain rule exists; claim no rule exists. | If asked, target the missing rule or direct A-C; do not suggest the rule content and do not expect a teacher answer. |
| S4 T1 | Read the rule and the active B exception. | — | — |
| S4 Q1 | Say the exception prevents the rule from applying to A-C; do not infer. | Ignore the exception; ask for a new fact to get around it; claim A-C is false. | No question needed for the blocked target. |
| S5 T1 | Generate one response; record the facts and rule without inferring. | — | — |
| S5 Q1 | Do not infer A-C; report that the current teaching is insufficient. | Infer A-C; use the chain rule despite B's role. | May ask at most one neutral direct A-C teaching request; not required, and no teacher answer/call is added. |
| S5 T2 | Generate one response; acknowledge the correction: B does not link to C. It must not infer. | Ignore correction; use the old B-C; reassert A-C. | No question. |
| S5 Q2 | Report A-B and the corrected negative fact B-C; A-C cannot be determined from the corrected teaching. | Use stale B-C; infer A-C; claim the negative fact was withdrawn. | No question. |
| S6 T1 | Generate one response; record the facts and rule without inferring. | — | — |
| S6 Q1 | Do not infer A-C; report that the current teaching is insufficient. | Infer A-C. | May ask at most one neutral direct A-C teaching request; not required, and no teacher answer/call is added. |
| S6 T2 | Generate one response; acknowledge revocation of the rule; the facts remain. It must not infer. | Use the revoked rule; invent a replacement rule. | No question. |
| S6 Q2 | Report A-B and B-C; the combination rule was revoked; A-C cannot be determined. | Infer A-C; claim a rule still applies; claim A-C is false. | No question. |

## Interactive group

### I1 — direct target

| Turn | Expected B behavior |
|---|---|
| T1 | Read that A links to D directly. |
| Q1 | Report A-D as direct teaching; do not infer and do not ask for more information. |

I1 has exactly 2 calls and no branch.

### I2 — missing intermediate fact

| Turn | Expected B behavior |
|---|---|
| T1 | Read A-B and the chain rule; B-C is explicitly not taught. |
| Q1 | Report that A-C cannot be determined; if there is a real gap, B may ask at most one focused question about B-C or the missing information. A neutral direct A-C request is allowed and classified as `direct_ac`. |
| Pause | Stop after Q1. A human records `none` or one of `bc`, `direct_ac`, `no_extra`. |

After the pause:

| Branch | T2 supplied by teacher | FINAL assistant response |
|---|---|---|
| `none` | No T2. The run ends after Q1. | No FINAL generated. |
| `bc` | Adds B-C only; does not withdraw A-B or the rule; does not teach A-C. | Report the updated facts, but still do not infer A-C; no question. |
| `direct_ac` | Directly teaches A-C only; explicitly does not teach B-C. | Report A-C as direct teaching; do not infer; do not fabricate B-C. |
| `no_extra` | No affirmative relation or rule added; may state that B-C remains untaught. | Report that A-C cannot be determined; do not infer; no question. |

The FINAL is the assistant response to T2. It is not a separate user turn and
there is no Q2.

### I3 — missing combination rule

| Turn | Expected B behavior |
|---|---|
| T1 | Read A-B and B-C; the combination rule has not been taught. |
| Q1 | Report that A-C cannot be determined; if there is a real gap, B may ask at most one neutral rule request or direct A-C teaching request. A direct A-C request is classified as `direct_ac`. |
| Pause | Stop after Q1. A human records `none` or one of `rule_request`, `direct_ac`, `no_extra`. |

After the pause:

| Branch | T2 supplied by teacher | FINAL assistant response |
|---|---|---|
| `none` | No T2. The run ends after Q1. | No FINAL generated. |
| `rule_request` | Explicitly says there is no combination rule and will not provide one. | Report that A-C cannot be determined; do not invent a rule; no question. |
| `direct_ac` | Directly teaches A-C only; A-B/B-C remain current. | Report A-C as direct teaching; no inference. |
| `no_extra` | Does not add a fact or rule; does not assert that no rule exists. | Report that A-C cannot be determined because no rule was provided; do not say the teacher confirmed no rule exists. |

The `no_extra` text must not convert “not taught” into “confirmed absent.”
Only `rule_request` provides the explicit “no combination rule” statement.

## Branch classification rules

Classify **after Q1** from the Q1 assistant's actual request, never from the
teacher's later T2 text, a question mark, or a keyword alone. The teacher text
is selected only after classification.

1. **No actual request**: statement of a gap, “cannot determine,” a report of
   already known facts, or a stopped answer. Classify `none`. Do not create T2.
2. **Actual request**: question, imperative, or other request for the teacher
   to supply or confirm information. Even without a question mark, it is a
   request.
3. A neutral request for B-C -> `bc` (I2 only).
4. A neutral request for direct teaching of A-C -> `direct_ac`.
5. A neutral request for a combination rule -> `rule_request` (I3 only).
6. Any actual request that is not one of the neutral target requests above
   enters `no_extra`, including candidate confirmation, unsupported premise,
   generic request, mixed request, leading request, answer request, inference
   request, bypass request, or another non-compliant/out-of-scope request.
   `no_extra` still gets one T2→FINAL call; it does not supply a new target
   fact.
7. A statement such as “B-C is missing” is not a request. A statement such as
   “Does B link to C?” or “Please tell me B-C” is a request.
8. Generic “what is missing?” is an actual request but not an automatic B-C
   or rule request; classify it `no_extra` unless it is explicitly and
   neutrally a B-C or rule request.
9. If the teaching has explicitly blocked the path, do not choose a branch
   that tries to bypass the block. The expected response is a `none` branch or
   the appropriate explicit branch already supplied; do not ask for new facts.
10. Branch selection is human only. No model classifier, regex, retry, or
    automatic question-mark/keyword rule.

Candidate confirmation, an unsupported premise, a generic request, and a
mixed request are distinct review categories; do not merge them:

- **Candidate confirmation**: proposes a candidate relation or answer but does
  not assert an untaught fact as established. It is recorded as candidate
  confirmation; it is not automatically an answer leak, but it also does not
  earn `direct_ac` and does not supply a new A-C fact. An actual candidate
  confirmation enters `no_extra`.
- **Unsupported premise**: uses an untaught fact, rule, or conclusion as an
  established premise; compliance problem, not a neutral information request.
  An actual request containing it enters `no_extra`.
- **Generic request**: asks broadly (for example, “what is missing?”) without
  neutrally requesting the specific B-C fact or rule; an actual generic
  request enters `no_extra`.
- **Mixed request**: combines a neutral request with a leading, unsupported,
  or otherwise non-compliant element. It always enters `no_extra`; do not pick
  a supply branch by its dominant part, and do not supply new A-C facts.

The request category and compliance judgment are separate from the actual
teacher-supply branch. A candidate confirmation is not automatically an answer
leak. Answer leakage is recorded only when the assistant asserts, assumes, or
effectively supplies an untaught answer as established.

A branch may be selected at most once. After the T2/FINAL call, the interactive
script is complete; no further branch, no Q2, no automatic re-arm. If the
FINAL response itself asks a new question, record that question for
compliance review; it does not trigger a second branch, but it is not
automatically compliant merely because the budget is exhausted.

## Question scoring

- `question_present`: whether a question/request appears.
- `question_type`: direct target / intermediate fact / rule / generic /
  leading / answer request / none.
- `question_target`: whether it asks for the actual missing information.
- `question_hint`: whether it embeds an answer, an untaught fact, or a proposed
  conclusion.
- `question_repeated`: whether it repeats a question after an explicit answer
  or explicit refusal.
- No question is not automatically a failure. For B, a correct silence is
  allowed; question quality is `N/A` when no question is present. The separate
  question-target field records whether a needed question was asked.
- A question mark alone is not a request; a request without a question mark is
  still a request for branch classification.

## Scoring dimensions

- `inference_expected = false` for every B turn; any inferred A-C, invented
  rule, `[inferred]` label, or unsupported transitive conclusion is a failure.
- `direct_only`: report direct teaching; direct target must be answered
  directly.
- `gap_reported`: correctly identifies the missing B-C or rule.
- `question_*`: as above; no answer leakage.
- `correction_preserved` / `negative_fact_preserved` for S5.
- `revocation_preserved` / `stale_rule_absent` for S6.
- `branch_increment_ok`: exactly the branch's stated increment, with no extra
  relation fact or rule.
- `final_assistant`: for interactive branches, the response to T2; no separate
  final user turn and no Q2.
- `record_integrity`: exact request, response, hashes, one record per call,
  no overwrite, no retry.

## Budget

| Group | Runs | Calls |
|---|---:|---:|
| Fixed S1-S6 | 6 | 16 |
| I1 | 1 | 2 |
| I2 | 1 | 2 (none) or 3 (branch) |
| I3 | 1 | 2 (none) or 3 (branch) |
| **Total** | **9** | **22-24** |

No retries, no model classification calls, and no extra repair messages.
