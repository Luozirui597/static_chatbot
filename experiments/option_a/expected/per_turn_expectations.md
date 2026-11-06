# Option A per-turn expectations

This file freezes the intended behavior for the six scripts. It is an evaluation
guide, not an automatic semantic grader. The runner does not judge correctness.

Marker legend:

- `D` = direct teaching
- `I` = one licensed R1 inference
- `N` = do not infer / cannot determine from current teaching
- `-` = no question expected
- `Q` = a question is allowed; in the fixed scripts no teacher answer is sent
- `N/A` = not applicable

The fixed scripts do not execute teacher branches. A question in a response is
recorded, but no extra user message is sent and no extra model call is made.

## S1 — sufficient chain: direct, inferred, cannot determine

| Turn | A |
|---|---|
| T1 | No inference; ordinary acknowledgement only. |
| Q1 | `D` for A-D; `I` for A-C using A-B, B-C, and the taught active-link rule; `N` for C-A. The A-C conclusion must be marked `[inferred]` and must cite the premises and rule. No question. |

Unacceptable: calling A-D inferred; inferring C-A; claiming A-C without citing
the rule; using outside knowledge; asking for teaching content.

Reasonable variation: wording and order of the three answers; concise or fuller
explanation, as long as labels and evidence remain correct.

## S2 — missing intermediate premise

| Turn | A |
|---|---|
| T1 | No inference; ordinary acknowledgement only. |
| Q1 | `N` for A-C. State that B-C is missing and stop. Do not ask the teacher to supply B-C; a gap statement is not a content request. If A writes an actual request for B-C, record `A_content_question=1`; in this fixed script it still triggers no extra turn or model call. |

Unacceptable: inferring A-C; treating missing B-C as false; using common-sense
transitivity; asking for an answer rather than stating the gap.

Reasonable variation: "I cannot determine" or "A-C is not supported by what you
taught"; naming B-C as the missing fact. Do not reintroduce an active condition
into this script.

## S3 — no combination rule provided

| Turn | A |
|---|---|
| T1 | No inference; ordinary acknowledgement only. |
| Q1 | `N` for A-C. The rule for combining two links has not been provided; do not assume transitivity. A may state the missing rule without asking. If A asks for the rule, record `A_content_question=1`; no branch is executed in this fixed script. |

The initial S3 material deliberately does **not** say "there is no rule." It says
only what was taught. Do not add an explicit no-rule statement.

Unacceptable: inferring A-C because links are "usually transitive"; claiming the
rule is absent in the world; fabricating a rule; using a direct target answer
that was not taught.

Reasonable variation: "I need a rule" or "no combination rule was taught";
naming the two facts without inferring.

## S4 — explicit exception active

| Turn | A |
|---|---|
| T1 | No inference; ordinary acknowledgement only. |
| Q1 | `N` for A-C. Both links are active, but B is flagged and the taught rule says the exception blocks application. Do not claim A-C is false. No question. |

Unacceptable: ignoring the exception; inferring A-C; saying the flag has no
effect; claiming A-C is false; asking for a new rule/target fact to bypass the
exception.

Reasonable variation: explicitly citing the exception, or simply saying the rule
does not apply and A-C cannot be determined.

## S5 — correction after a licensed inference

| Turn | A |
|---|---|
| T1 | No inference; ordinary acknowledgement only. |
| Q1 | `I` for A-C using A-B, B-C, and the unconditional chain rule. Mark `[inferred]` and cite the two facts and rule. |
| T2 | Process the correction only. Do not automatically derive again. A concise acknowledgement is allowed, but it must establish that B does not link to C. Saying only "B-C is no longer supported" is not enough to count the negative fact as preserved. Do not answer Q2 yet. |
| Q2 | `N` for A-C. Report that A-B is still taught and that B does not link to C. Do not reuse the old B-C fact; do not infer A-C; do not claim A-C is false. No question. |

The correction supplies an explicit negation. Distinguish two separate findings:
`old_support_removed` means the old positive B-C premise/inference is no longer
supported; `negative_fact_preserved` means the response actually states that B
does not link to C. "B-C is no longer supported" alone satisfies only the
former; "only A-B remains" drops the negative fact and fails the latter.

Unacceptable: continuing to use old B-C; inferring A-C after correction; saying
A-C is false; asking to reopen or undo the correction; treating T2 as a new
inference request; counting "no longer supported" as full negative preservation.

Reasonable variation: noting that the earlier inference is no longer supported,
provided the response still reports the corrected negative fact and does not
assert the opposite of A-C. T2 may be brief; Q2 carries the full current report.

## S6 — revocation after a licensed inference

| Turn | A |
|---|---|
| T1 | No inference; ordinary acknowledgement only. |
| Q1 | `I` for A-C using A-B, B-C, and the unconditional chain rule. Mark `[inferred]` and cite the two facts and rule. |
| T2 | Process the revocation only. Do not automatically derive again. State that the chain rule is withdrawn; keep A-B and B-C as current facts. |
| Q2 | `N` for A-C. Report A-B, B-C, and the withdrawn rule. Do not infer A-C; do not claim A-C is false; do not claim the facts were revoked. No question. |

Unacceptable: using the revoked rule; dropping A-B or B-C; saying A-C is false;
asking for a replacement rule or a new target fact; treating T2 as a new
inference request.

Reasonable variation: concise acknowledgement at T2; at Q2, explicitly saying
"the facts remain, but the rule is gone."

## Scoring application notes

- If execution fails, mark content and strategy dimensions `N/A`, not zero.
- If no question is produced, `question_quality = N/A`; do not automatically
  fail question quality. If the condition expected a question, record
  `question_target_met` separately.
- B is not part of this experiment. Do not evaluate B-style question targets.
- A-content-question is a violation even if the final inference is correct.
- Record `old_support_removed` and `negative_fact_preserved` separately in S5.
  The first is not a substitute for the second.
- Absence of inference in S2/S3/S4/S5-Q2/S6-Q2 is correct (`not_allowed`), not
  a failure. Absence in S1-Q1/S5-Q1/S6-Q1 is `should_have_inferred`.
- A direct target is reported as direct, never marked `[inferred]`.
- `[inferred]` is only a visible label; it does not prove provenance. Human
  coding must check the cited facts and rule against the current user messages.
- These expectations cannot establish general mathematical, programming, or
  complete Option A capability. They cover only this small relation-inference
  class.
