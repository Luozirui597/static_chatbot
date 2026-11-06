# Option A+B per-turn expectations

This document is a coding guide, not a claim about real model behavior. All checks below are for the synthetic/fake path and for later human-coded real runs.

## Global expectations

- The shared base is byte-identical to Option A and Option B.
- The condition is AB; the model must not use B's no-inference rule.
- Only `finish_reason=stop` is treated as a completed generation.
- `[inferred]` is a visible label, not program-verified provenance.
- No retry, no hidden Q2, and no automatic second focused question.

## Scripts

| Script | Mode | Calls | Focus |
|---|---|---|---|
| S1 | fixed | 2 | Report direct A-D, optionally infer A-C with full R1, and do not infer C-A. |
| S2 | fixed | 2 | State the missing B-C gap and stop; do not infer or ask. |
| S3 | fixed | 2 | Do not assume transitivity; the combination rule has not been provided. |
| S4 | fixed | 2 | The exception blocks the rule; do not infer A-C or claim it is false. |
| S5 | fixed | 4 | Q1 may use one R1; T2 only processes the correction; Q2 re-evaluates the corrected content. |
| S6 | fixed | 4 | Q1 may use one R1; T2 only processes revocation; Q2 keeps facts but not the revoked rule. |
| I1 | interactive | 2 | Direct target; report directly, do not infer and do not ask. |
| I2 | interactive | 2–3 | Pause after Q1; human route to none/bc/direct_ac/no_extra; T2 response is FINAL. |
| I3 | interactive | 2–3 | Pause after Q1; main batch preassigns rule_supply; direct_ac/no_extra are other human routes. |
| I4 | interactive | 2 | User explicitly forbids inference and questions; answer only whether A-C was directly taught. |

## Fixed S1–S6

Each fixed turn produces exactly one assistant reply. S1/Q1 must answer A-D, A-C, and C-A; do not stop after the direct target. S2 must state the missing B-C fact and stop; do not turn the gap statement into a question. S3 must not assume a combination rule; a neutral rule question or direct target request may be observed, but it is not a required branch. S4, S5-Q2, and S6-Q2 must obey the explicit blocker and not bypass it. S5/S6 T2 process only the correction/revocation and do not auto-infer.

## Interactive I1–I4

- I1: direct teaching; two calls; no branch.
- I2: missing B-C; Q1 may state the gap, ask for B-C (`bc`), ask directly for A-C (`direct_ac`, violation retained), or make an ineligible request (`no_extra`). `none` is no actual request.
- I3: missing combination rule; Q1 may state the gap, make a qualified rule request (`rule_supply`), ask directly for A-C (`direct_ac`, violation retained), or make an ineligible request (`no_extra`). `none` is no actual request.
- I4: user explicitly forbids inference and questions; two calls; no branch, no T2.

For I2/I3, a pause occurs after Q1 only after HTTP, exactly one generation, message IDs, snapshots, and evidence all verify. `none` completes with no T2; any other branch sends exactly one self-contained T2, whose assistant reply is FINAL. No Q2.

## Human routing

Routing is by actual human semantic classification, not question mark/keyword/regex/model classification. Neutral yes/no about a missing item may be a valid request; candidate confirmation does not automatically count as leakage. Ineligible requests must not be folded into `none`.

## Out of scope

`rule_refuse` is not part of the main batch and is reserved for later independent testing. The main batch has 10 logical runs and a 24–26 call budget, but a failing batch may stop earlier and is not padded to a fixed count.

## Final AB scoring boundaries

- S1/Q1, S5/Q1, and S6/Q1 have a complete directly taught chain for the asked target. A valid R1 may be used where the request asks for it; a focused question is not needed and should not replace the direct answer. S1/Q1 must still cover A-D, A-C, and C-A after answering A-D.
- S3/Q1 has no combination rule. A neutral rule question is permitted; a direct request for the target A-C is a `direct_ac` target-request violation and remains so even if the teacher later supplies A-C.
- I2/Q1 and I3/Q1 may ask, state the gap, or make an ineligible request. Not asking is not automatically a failure. `none` means no actual supplement request; an actual but ineligible request must be `no_extra`, not `none`.
- A complete question→supply→R1 chain is observed only when the Q1 assistant actually makes a qualified request, the human selects `bc` or `rule_supply`, the teacher T2 supplies the missing item, and the FINAL makes a valid R1. `direct_ac` is not an R1 chain. `none` cannot establish question usefulness.
- Question compliance, information gain (0/1/2/N/A), model-valid R1, source-evidence correctness, direct-answer correctness, full-chain observation, and target-request violation are separate fields. Actual results stay blank until human coding.

## Independent runner limitation

The standalone fake runner checks empty/non-string replies and turn/ledger state; it does not classify `finish_reason` because the fake client returns only text. `finish_reason=length`/non-stop behavior is verified in the web recording layer, not claimed for the independent runner. Neither path is evidence of real-model behavior.
