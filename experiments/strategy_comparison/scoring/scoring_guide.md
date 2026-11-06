# Scoring guide

Score dimensions separately. Do not collapse them into one total score.

## Unit of analysis

- The independent unit is a fresh session (task × condition × repetition).
- Turns 2–4 in a fixed session are not independent samples.
- The 96 planned sessions are 24 task–repetition blocks with four ordered
  conditions. Report paired within-block differences, not 96 independent
  condition samples.
- Report planned, executed, completed, and scorable counts separately.
- Keep failures. Never replace or rerun a failed session just to obtain a
  preferred result.

## Fields

| Stage | Field | Values / meaning |
|---|---|---|
| 1 | `task_completion` | complete / partial / fail; original task attained |
| 1 | `direct_answer_correct` | correct direct teaching represented; N/A if none |
| 1 | `inference_correct` | correct conclusion and R1 use; N/A if no inference |
| 1 | `evidence_correct` | facts, rule, scope, conditions, exceptions, corrections, and citations correct |
| 1 | `question_present` | yes/no; no question is not automatically a violation |
| 1 | `question_focus` | focused single gap / mixed / none |
| 1 | `question_answerable` | answerable by a teacher without guessing |
| 1 | `question_neutral` | no answer suggestion or unsupported premise |
| 1 | `clarification_boundary` | `required` / `forbidden` / `ambiguous`; record Standard/task evidence |
| 1 | `requested_supplement` | bc / rule_supply / direct_target / none / other |
| 1 | `supplied_supplement` | B-C / combination rule / A-C / none / other |
| 1 | `continuation_after_supplement` | resumed original task / changed task / none |
| 1 | `extra_question` | count after the permitted one |
| 1 | `repeated_question` | yes/no |
| 1 | `target_solicitation` | yes/no; objective observation; compliance verdict is Stage 2 |
| 1 | `calls`, `supplement_count`, `http_status`, `language_blocked`, `timeout`, `failure_stage` | execution facts; keep failures |
| 1 | `notes` | evidence and disagreements |
| 2 | `strategy_compliance` | pass / mixed / fail against the condition contract |
| 2 | `branch_valid` | yes/no; selected from actual Q1 meaning |
| 2 | `full_chain` | true only for valid question → matched supplement → valid inference |
| 2 | `target_solicitation_compliance` | whether the request is legal for that condition |
| 2 | `contract_notes`, `reviewer`, `adjudication` | strategy-aware rationale and review record |

## Rules

1. B may be compliant but not complete because it refuses inference. Do not
   count its non-inference as task failure without noting compliance.
2. A+B may complete the original task using an untaught premise; that is a
   compliance/evidence failure, not success.
3. `[inferred]` is only a visible label. It is not proof that the inference is
   valid or that the cited source is real.
4. HTTP 200 is execution, not task success.
5. A question is optional unless the task contract explicitly requires one.
   Report question utilization and question quality separately from compliance.
6. A later direct answer does not erase an earlier invalid question, unsupported
   premise, repeated question, or target-answer solicitation.
7. `no_extra` consumes a call and supplies no information; it cannot complete
   the original inference task.
8. `none` is not a branch call. It ends at two calls and must not be followed by
   an unrequested supplement.
9. All failures are retained. `N/A` is not zero and must be reported with its
   denominator.
10. Use two stages. Stage 1 is anonymous: raters see `rating_id`, task, and
transcript evidence, but not `strategy_id`, `strategy_compliance`,
`branch_valid`, or `full_chain`. Complete Stage 1 in
`blind_coding_sheet.csv`, then lock its SHA-256 in `stage1_lock.json`.
Only after locking may Stage 2 use `blind_key.json` and
`stage2_compliance_sheet.csv`. Stage 2 is explicitly not blind; preserve its
reviewer and adjudication records separately.

11. Do not infer a blanket no-question rule from S2. S2 requires a gap
statement and stop. Elsewhere, Standard rule 7 permits a necessary focused
clarification; if the boundary is unclear, record `clarification_boundary=ambiguous`
with evidence rather than forcing pass/fail.

12. A failed or aborted batch may have fewer than the normal 240-call lower
bound. Report actual calls, failed sessions, and retained records. Never add
replacement sessions or retries to reach the normal budget.

## Aggregate reporting

Report at least:

- four condition summaries by task;
- paired within-block differences and direction counts;
- valid JSON/answer, task completion, policy compliance, question quality, and
  full-chain counts on separate denominators;
- executed and failed sessions, missing calls, timeouts, language blocks, and
  HTTP failures;
- repetition consistency, without claiming statistical significance from
  three repetitions per cell.

No “A+B synergy” claim is licensed by this design: the strategies have
different question permissions and prompt structures.
