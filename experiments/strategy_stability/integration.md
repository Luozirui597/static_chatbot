# Production integration assessment (design only)

No production file has been changed for this design.

## Candidate and context

The final model call returns a strict multi-item candidate (direct, one R1,
insufficient, and at most one ask/clarify).  The server owns the original goal,
subgoals, askable gaps, turn kind, permissions, and required targets; the
model cannot assert `ask.kind` or a question count.  All items are validated
before any text is rendered.  A direct response may be followed by one final
question, but no item may follow that question; one reply cannot contain two R1
items.  An `ask` gap must not be the same relation as the target it serves.

## Where it would sit

The verifier must sit after the one allowed final candidate and before any
assistant row is written.  In the current Receive flow the relevant sequence
is inside the per-session lock in `backend/chat_service.py`:

1. user row is saved with the actual interaction strategy, model, and prompt
   version;
2. the existing fixed-relation / general-history routing selects the final
   path;
3. the selected path obtains one structured candidate instead of free text;
4. `strategy_verifier` validates evidence and policy and renders the only
   permitted assistant text;
5. the existing language validator validates that rendered text;
6. assistant text and the interaction-state update commit atomically;
7. any verifier/language failure leaves the user row and its actual snapshot
   intact and writes zero assistant rows.

The four existing paths need the same hook:

- ordinary `Receive teaching`;
- fixed relation retrieval;
- general-history final answer;
- clarification.

A clarification is not an `ask` candidate.  It is a separately authorized
`clarify` item, allowed only when the server grants `allow_clarify`, and uses a
deterministic renderer.  It is not subject to the missing-fact gap rules.

## Required production modules (future change)

- `backend/strategy_verifier.py` (or equivalent pure module): the parser,
  evidence state, policy validation, renderer, structured errors.
- `backend/chat_service.py`: one pre-save hook shared by all four final paths;
  no assistant save before verification.
- `backend/general_history/engine.py` and
  `backend/programmatic_history/messages.py`: expose exactly one structured
  candidate from the selected final generation; do not append free prose.
- `backend/main.py`: structured 502 mapping for verification failures; 422 is
  not appropriate after the user row is saved.  Language 422 behavior is
  unchanged.
- `backend/models.py` + migration (later): session-scoped interaction state
  (asked count, remediation count, closed, target, version) and, if needed,
  a verifier-result marker on assistant messages.  No migration is part of
  this prototype.
- candidate-output prompt: the same strict JSON contract for all strategies;
  this is an experimental prompt, not a production switch in this phase.
- tests: pure verifier, endpoint failure matrix, cross-session/source,
  strategy mapping, state transitions, version lengths.

## Version table and snapshot impact

Current snapshot limit is 50 characters.  The current language-suffixed
programmatic version is already 49:

`receive-teaching-v7+programmatic-history-v1+l1-en`

Appending `+r1` would exceed the limit.  The proposed verified variants replace
the `l1` suffix with a combined `r1-en` / `r1-zh` marker:

- ordinary: `...history-context-b1+r1-en` = 44;
- programmatic: `...programmatic-history-v1+r1-en` = 49;
- clarify: `...clarify-v1+r1-en` = 36.

Old rows remain untouched.  The marker must be recorded in the same
`prompt_version_snapshot` that corresponds to the actual candidate prompt.
Do not truncate or rewrite an old snapshot.  Ordinary Standard/A/B/A+B use
their distinct actual versions; the current fixed/general and clarification
markers are shared path markers and must not be reported as Standard or as
strategy-specific evidence.  The `r1-en/zh` table is a proposal only.

## Interaction state

The prototype accepts `InteractionState(asked, remediated, closed)` from the
caller.  Production must own this state, not parse it from model text or count
question marks.

Minimum future state per session/target:

- asked count (max one);
- remediation count (max one);
- closed flag;
- original target identity;
- last verified status/version.

The state update and assistant save must be one transaction under the existing
session lock.  If the state commit fails, the assistant row must not become
visible.  A retry would need an idempotency key; this design does not add one.

## Compatibility and risk

- **Prompt drift**: the four production prompts currently do not enforce a
  candidate schema.  A candidate-generation prompt is a new behavior and must
  be separately versioned.  This prototype does not prove prompt compliance.
- **Evidence parser coverage**: only the controlled `links` grammar is
  supported.  Ordinary/free teaching remains on the old path.  Unsupported
  input is rejected, not silently accepted.
- **False rejection**: valid paraphrases outside the grammar can be rejected;
  this must be measured, not hidden.
- **Model-generated evidence mapping**: the model still proposes facts/rules
  from teacher messages.  The verifier checks them against the parsed
  evidence, but the parser itself is the remaining untrusted semantic layer.
  A future adapter must expose parser diagnostics and unsupported reasons.
- **Renderer expressiveness**: deterministic templates cannot produce rich
  explanations.  That is intentional for the controlled prototype; expanding
  it requires a separately audited renderer.
- **Language**: the prototype renderer is English.  Production Chinese output
  needs bilingual renderer templates and must still pass existing language
  validation.
- **Question state**: no production table exists yet; adding it is a schema
  and migration change, not included here.
- **Failure semantics**: verification failure after user save should be a
  structured 502 with no assistant row, not a 422.  Existing frontend
  saved-user handling is preserved.
- **No general Agent framework**: only the minimum relation grammar and
  focused question flow are designed.

## Explicit non-goals

- no production switch in this phase;
- no changes to the four frozen prompts;
- no changes to `.env`, business DB, services, frontend, or A/B/AB assets;
- no model call, browser, or formal acceptance in this phase.

## Implemented isolated integration (this round)

The isolated entry point is `experiments/strategy_stability/integration_app.py`.
It uses `experiments/strategy_stability/models.py` with its own SQLite engine;
it does not import or migrate `backend.models`.  The test suite starts the
FastAPI app through `TestClient` with an explicit `tmp_path` database.

Sequence implemented by `ExperimentService.send`:

1. In a first short transaction, save the user row with the session's actual
   strategy, language, model, contract, verifier, and behavior versions.
2. Outside that transaction, parse the server-owned evidence and target
   context, call the fake candidate source exactly once, and validate the
   candidate with `validate_candidate`.
3. Validate the deterministic rendered reply with the real
   `validate_reply_language`.
4. In one second transaction, insert the assistant row and update the session
   state (`asked`, `remediated`, `closed`, `pending_gap`).  The update and row
   are rolled back together on either failure.
5. A rejected candidate, language failure, model failure, or persistence
   failure leaves the user row intact and writes no assistant row.

The experiment state vocabulary is server-owned:

- a successful `ask` sets `asked=1` and records a normalized
  `pending_gap` object with `kind`, `target`, and `served_target`; it is the
  only way to open remediation;
- `pending_gap` is produced by the same `parse_candidate` path used for
  validation, not by a second raw-JSON parser; legacy `action: "ask"` and
  current `items` shapes therefore produce the same pending contract;
- `POST /sessions/{id}/remediation` accepts only the exact pending target and
  a positive teacher message id.  It reparses the submitted teacher text with
  the controlled evidence parser and checks its kind and target:
  `missing_fact`/`direct_target` require the exact relation (positive or
  negative), `missing_rule` requires a matching controlled rule, and a target
  fact cannot substitute for a rule.  Unrelated, unsupported, or
  kind-mismatched text leaves evidence, counts, and pending state unchanged;
- only a valid remediation appends evidence, sets `remediated=1`, and clears
  the pending gap in the same transaction;
- a server-authorized task `clarify` consumes the one-question allowance but
  does not create a content-remediation `pending_gap`; `direct + clarify` is a
  legal `verified` result and follows the same rule;
- a content `ask` must produce a matching pending contract; the integration
  rejects an accepted ask if pending extraction is missing;
- subsequent user text alone never counts as remediation;
- after remediation a second ask is rejected by the verifier state; a legal
  direct or R1 answer can still be saved;
- `pending_gap` is cleared on any later successful non-ask assistant save only
  by explicit remediation in this prototype; it is not inferred from model
  prose.

The fake model is a transport-free candidate source.  It receives a frozen
`CandidateRequest` containing the current question, the full server-selected
teacher evidence tuple, target/turn context, strategy, language, model, and
contract/verifier/behavior versions.  It records that exact request and the
generation count; it never validates, rewrites, or supplies server policy.  A
request in the candidate stage is called at most once.  The isolated tests
exercise transaction rollback with real SQLAlchemy events; they do not claim
to reproduce an uncertain post-commit outcome.

Error stages are explicit:

- malformed request fields or a missing/closed session, before a user row is
  saved: 422;
- candidate model failure, malformed/rejected candidate, source/policy
  validation failure, or language mismatch, after the user row is saved: 502;
- failure to save the user row: 500 with `user_message_id: null` and no model
  call;
- failure to commit the assistant row and state together: 500; the user row
  remains and no assistant row is visible.

### Running

```bash
cd /Users/luo/Documents/teachable_agent
PYTHONPATH=. .venv/bin/python -m pytest experiments/strategy_stability/tests -q
```

### Known limits

- This is an experimental app/database only.  It is not the production
  `ChatService`, fixed-relation retrieval, general-history retrieval, or the
  formal web page.
- The renderer is English-only; a Chinese target is intentionally exercised as
  a language-rejection case, not as bilingual support.
- The candidate is fake/deterministic.  Real-model candidate compliance and
  actual production prompt integration remain unverified.
- The candidate-to-context binding is limited to the controlled `links`
  grammar and the current target/evidence contract.
- An uncertain commit result after a successful SQL `COMMIT` is not modelled;
  all persistence failure tests fail before commit or during a flush.
