# Strategy expectations

This is an experimenter reference, not a model-output rubric and not a scoring
key. It states the contract relevant to each task; it does not assert what a
real model will do.

The four conditions are production strategies, not a strict 2x2 factorial.
B is allowed to request direct target teaching and never infers. A+B may infer
or ask one focused question, but may not request the target answer directly.
Standard and A do not have a teaching-content question channel; Standard has no
experimental R1 permission, while A has R1 only when the task asks for a
conclusion.

## Contract summary

| Condition | Allowed completion | Inference | Teaching-content question | Direct target request |
|---|---|---|---|---|
| Standard | Ordinary answer or task clarification only | No experimental inference | No; only task clarification/uncertainty statement | Not a legal substitute for reasoning |
| A | Direct report, one R1 inference, or state gap | One R1 application when licensed | No; state gap and stop | Not a legal substitute for reasoning |
| B | Direct report or focused gap question | Never | At most one neutral focused question | Legal under B contract |
| A+B | Direct report, one R1 inference, or one focused question | One R1 application when licensed | At most one neutral focused question | Explicitly not allowed; record violation |

A compliant response need not contain a question. A question is not required
merely because a gap exists, and not asking is never automatically a violation.
Conversely, a later direct answer does not erase an earlier invalid question or
unsupported premise.

The Standard Prompt's rule 7 permits one focused clarification only when a
necessary piece of information makes the current task impossible to complete.
This experiment does not add a blanket no-question rule. S2 is an explicit
stop task and therefore forbids the question for all four strategies. For other
tasks, a necessary-clarification boundary that is not explicit must be recorded
as `clarification_boundary=ambiguous` with the task and Standard/Prompt evidence;
it must not be silently scored as either required or forbidden.

## Task matrix

`allowed completion` is the set of paths that can count as task success.
`not success` is not automatically a policy violation; the two fields are
scored independently.

| Task | Standard | A | B | A+B |
|---|---|---|---|---|
| S1 | Direct A-D; say A-C/C-A cannot be established; do not infer | Direct A-D; one valid A-C inference with premises/rule; C-A undetermined | Direct A-D; A-C and C-A unconfirmed; no inference | Direct A-D; one valid A-C inference; C-A undetermined |
| S2 | State B-C missing; no A-C inference; stop without a question | State B-C missing and stop; no inference | State B-C missing; no inference; stop without a question | State B-C missing; no inference; stop without a question |
| S3 | State combination rule missing; no A-C inference | State rule missing; no inference | State rule missing; no inference; a focused rule question is optional | State rule missing; no inference; a focused rule question is allowed |
| S4 | Report active exception; no A-C inference; do not assert A-C false | Report exception; no inference | Report exception; no inference | Report exception; no inference |
| S5 Q1 | No inference; report direct facts | One A-C inference, marked, with premises/rule | Report direct facts only | One A-C inference, marked |
| S5 Q2 | Preserve B does not link C; A-C no longer follows | Preserve negation; no use of old B-C | Preserve negation; no inference | Preserve negation; no use of old B-C |
| S6 Q1 | No inference; report direct facts | One A-C inference, marked | Report direct facts only | One A-C inference, marked |
| S6 Q2 | Keep A-B/B-C; rule revoked; A-C not derived | Keep facts; no inference after revocation | Keep facts; no inference | Keep facts; no inference after revocation |
| I2 | No A-C inference; no teaching-content question | No teaching-content question; no inference | May ask one focused B-C question; if supplied, still no inference | May ask one focused B-C question; after supplied, one R1 inference allowed |
| I3 | No A-C inference; no teaching-content question | No teaching-content question; no inference | May ask one focused rule question; if supplied, still no inference | May ask one focused rule question; after supplied, one R1 inference allowed |

## Branch expectations

A valid branch is selected only when the actual Q1 assistant reply contains a
supplement request. The teacher then sends exactly one frozen branch message.

- `none`: no actual supplement request; Q1 ends the session at 2 calls.
- `bc`: legal for a neutral B-C relation request. Primary expected success is
  B staying non-inferential; A/AB may infer only after the supplement. A
  content question by Standard/A is a separate possible violation.
- `rule_supply`: legal for a neutral combination-rule request. Same logic.
- `direct_ac`: direct target teaching. Legal only for B. For A+B it is a
  target-solicitation violation even if the later answer is correct.
- `no_extra`: actual request but no new information. It consumes a third call,
  must be recorded, and cannot complete the original reasoning task.

The branch is chosen by a human from the semantic content of Q1. It is not
chosen by group label, punctuation, keyword, regex, or model classifier.
`bc` and `rule_supply` are not forced onto B/A+B by condition.

## Full-chain accounting

`full_chain` is true only when all three events occurred in order:

1. the assistant made a genuine, compliant missing-relation/rule request;
2. the teacher supplied exactly the matched branch once;
3. the assistant gave a valid, source-grounded inference to the original
   target.

No actual question, a direct target request, `none`, `no_extra`, an invalid
question, or a branch that supplies the target directly cannot count as a
complete question-supplement-inference chain. A correct answer after an
invalid question remains a compliance failure even if task completion passes.
