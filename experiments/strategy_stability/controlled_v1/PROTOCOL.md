# Controlled links pilot v1

This is a **new task and input format**, not the old 32-session / 80-call
narrative pilot. No old scores, denominators or frozen assets are reused.
Production services, production databases and the webpage are not involved.

## What changes and what stays fixed

Eight development-exposed tasks S1-S6/I2/I3 use atomic teacher evidence. The
teacher evidence is preloaded; it generates **zero** acknowledgment calls.
S5/S6 have a frozen correction/revocation event between Q1 and Q2, also with
zero model calls. The runner applies that event under the isolated session
lock in a temporary database. This tests use of the current evidence snapshot,
not free-form teaching ingestion, teacher-turn responses or long-term memory.

The four conditions preserve their actual strategy text in frozen prompt
copies. A common JSON output contract replaces prose formatting, and a
deterministic renderer replaces the final prose. Standard and B do not gain
R1; A does not gain content asking; A+B cannot ask for its goal answer. These
are comparisons of four whole policies, not a clean factorial interaction
study. Template-constrained output cannot establish open-ended agent quality.

Evidence and user questions are identical across conditions. The model sees
the original goal, requested targets, current-turn permissions and interaction
counts. It **does not** see expected answers, missing_relations, missing_rule,
askable_gaps, gap_targets or the scoring gold. Those are server-only data.
The model must identify its own missing teaching from the supplied evidence.

Each strategy has its own frozen system prompt copy. All share the requested
model, JSON mode, output-token cap and request timeout in conditions.json.
There is no strategy-specific parameter tuning, repair, retry or backfill.

## Pairing and budget

The pilot is one repeat of 8 tasks x 4 conditions = **32 fresh sessions**.
For each task block, condition order is randomized with seed 20261106.

- S1-S4: one query generation each, 4 x 4 = 16 calls.
- S5-S6: Q1 and Q2, 2 x 2 x 4 = 16 calls.
- I2/I3: one initial query each, 2 x 4 = 8 calls.
- Conditional continuation: at most one for B and A+B per interactive task,
  2 x 2 = 4 additional calls.

Thus **40 base generations, at most 44**. Withholding/none causes no extra
call. A failed batch can have fewer than 40. A successful main batch of three
repeats would be 96 sessions / 120-132 generations under this new ingestion
contract, but it is **not scheduled by this pilot**. Historical 240-264 and
408-432 budgets concern the older proposed script, not these assets.

Only this 32-session / 44-call-cap pilot is executable here. Held-out variants,
explicit-question ability tasks and any larger run require separate assets.

## Teacher branches

This version uses a **controlled deterministic teacher**, explicitly different
from the old human semantic branch decisions. A branch is selected only from
an accepted candidate's normalized ask and the server's persisted pending
contract. It is not selected from question marks, keywords or the group name.

Task branches list the exact single statement supplied. A B direct_target
request can receive a direct target fact; that is a separate branch and never
counts as a question -> gap teaching -> inference chain. A valid intermediate
fact or rule question receives exactly that item, then one FINAL generation.
No question, rejected candidate, or optional withholding gets no supplement.
The same branch text applies to any condition legally requesting that item.

B remains forbidden to infer after receiving a premise or rule. A+B may infer
once when a valid path has been supplied. No second question or second
supplement is allowed. Questions are optional; no-ask is not a violation.

## Scoring and evidence

Record the raw candidate before the gate, rendered text after the gate, every
phase's HTTP/stage, sources, strategy/version snapshots, state transitions,
branch, model attempts, observed transport, response alias, usage and latency.
Write call intent before transport and persist results after every generation.
No credentials or authorization headers are written.

Report separate counts, without a combined score:

1. Parseable and canonical candidates, using all attempted generations.
2. Raw relational answers versus fixed task gold, including rejected outputs.
3. Accepted and appropriate turns, including every rejection in denominators.
4. Answered A-C targets, separate from lawful unknown handling.
5. Questions, branch kinds, and completed gap -> inference chains; zero
   questions makes conditional question quality N/A, not 0% quality.
6. Calls, transport failures, language failures, tokens and latency.

The relational gold checker is independent of validate_candidate. It is not
an independent proof-source oracle. A rejected output with matching relations
is a review flag, not automatically a verifier false rejection. Source proof
is gate-checked; no claim of independent human proof certification is made.

The label-removed anonymous artifact permits later review. Programmatic gold
scoring is unblinded, not two-rater blind evaluation. No significance,
generalization or A+B synergy claim is permitted from this small known set.
Inference structure can itself reveal policy; removing labels is not full
concealment. Old human scoring sheets and their locks are not modified.

## Stops

Hard stop on an accepted relational-gold mismatch, assistant saved after
rejection, state advance after rejection, version/policy mismatch, unsupported
frozen event, unplanned branch/call, asset drift, or the 44-generation cap.

Stop for review at three consecutive generation/transport failures or failures
in more than 10% of the 32 planned sessions. Candidate validation rejections
are model/contract outcomes, not automatically transport failures. No retry or
replacement follows either kind of failure. Preplanned Q2 can still run after
a rejected Q1; failed interactive Q1 cannot trigger a supplement.

Interrupted or stopped batches retain all attempted sessions and calls. Source
records are never overwritten; the CLI requires a new external evidence dir.
Real execution does not start before manifest verification and offline tests.
