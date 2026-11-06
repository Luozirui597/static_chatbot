# General teaching-history retrieval experiment

This is an isolated cross-domain history experiment.  Its default path uses
fake clients; an opt-in real HTTP adapter is included for the later authorized
provider test.  It is not wired into the production endpoint and does not
modify the production relationship search, prompts, `.env`, or database.

## Scope

- Synthetic SQLite data only.
- Temporary databases and fixture messages.
- Fake decision/final clients by default; MockTransport tests for the opt-in
  real adapter.  The normal test suite makes no provider request.
- Read-only search over earlier teacher messages.
- No knowledge-state writes and no retry.

The experiment reuses the production window boundary and programmatic message
builder, but has its own `AnchorSearch` because the production tool accepts
plain string queries rather than structured anchors.

The CLI and experiment modules do not import `backend.database` or
`backend.chat_service`; the test database is created with the imported ORM
`Base.metadata` on an explicitly supplied test engine.  This prevents the
experiment CLI from indirectly loading the production dotenv configuration.

## Flow

```
window boundary
  -> direct answer / decision / forced explicit search
  -> zero or one structured decision model call
  -> zero or one AnchorSearch call
  -> one final model call
```

The production relationship parser and `decide()` routing are reused, not
reimplemented with broad `links` matching.  This experiment does not modify the
production branch.

## Decision protocol

The decision response must be exactly one JSON object:

```json
{
  "need_history": true,
  "query": "process P7 tighten screw",
  "visible_source_ids": [],
  "reason": "ordinary_recall"
}
```

Allowed reasons:

- `ordinary_recall`
- `visible_complete`
- `ambiguous`
- `not_needed`

`visible_source_ids` must be teacher IDs from the current visible window.
`need_history=true` requires a query of 1–200 characters and forbids visible
IDs.  `need_history=false` uses exactly `query=""` as the single no-query
representation; a non-empty query or any other form is invalid.  Each turn may
use at most six deduplicated, grounded anchors.  A malformed decision stops the
turn before search or final generation.

## Trigger and call rules

- No hidden teacher messages, simple confirmation, and pure teaching go
  directly to final: `0/0/1`.
- The production relation parser is used with ordered subject/relation/object
  and the latest effective visible statement.  A visible matching relation goes
  directly to final: `0/0/1`.  A hidden-only relation, a reversed visible
  relation, or an explicit original/current comparison searches once: `0/1/1`.
- A non-relation request that merely mentions “links” continues through the
  generic decision route.
- Explicit historical questions with a reliable anchor go directly to one
  search: `0/1/1`, even if a matching topic phrase is visible; visible phrasing
  is not the original/replacement/source evidence.
- Explicit historical questions without a unique anchor go to one final
  clarification: `0/0/1`.
- Ordinary history-dependent questions use one decision; if it requests
  history, at most one search and one final: `1/1/1`.
- A legal `ambiguous` decision uses one final clarification: `1/0/1`.
- A decision timeout with a reliable anchor searches directly; without an
  anchor it clarifies.
- Invalid decision: `1/0/0`. Search timeout/error: decision + retrieval, no
  final. Empty search: decision/retrieval + one evidence-limited final.

The decision request contains a clearly marked JSON data block with the
visible teacher message IDs, order, and exact text, followed by the current
question as a separate user message.  A query anchor must be grounded in the
question or in one uniquely located visible process theme.

Attempts and client calls are tracked separately from actual HTTP requests.
The fake path records `synthetic=True` and zero `*_http_requests`.  The real
adapter records `synthetic=False`, actual HTTP counts, request model, upstream
model when supplied, stage, status, and elapsed time.  It does not record
prompts, headers, credentials, reasoning, response bodies, or exception text.

## Search contract

- Read-only SQLite URI plus `PRAGMA query_only=ON`.
- Session-bound and limited to teacher messages with
  `id < visible_start_id`.
- 200 SQL candidates, 50,000 scanned characters, 3 returned results.
- Effective deadline is `min(turn absolute deadline, search start + 1s)`.
  SQL progress, Python matching, and result receipt share it.  If the turn
  budget is already exhausted, no scan/connection is started; late results are
  not passed to final generation.
- Up to 1,000 characters per result and 4,000 characters total.
- Formula and phrase anchors receive higher rank than terms.  The engine keeps
  at most six grounded anchors.
- A correction/exception requires an earlier scanned non-correction teacher
  message sharing a normalized anchor.
- When the question asks for comparison, a correction and its earlier version
  are paired before remaining slots are allocated.
- Corrections get the content budget first; final results are sorted in teacher
  order.
- `truncated`, `scan_truncated`, `corrections_truncated`, `omitted_count`, and
  `pair_missing` are preserved as separate evidence limitations.  A missing
  comparison pair is not a scan/content truncation.

## Cases in the fixture

`fixtures/general_cases.json` contains hidden and visible message sets for:

- process correction and reason;
- rule plus exception;
- formula original/correction and formula boundary;
- hidden and visible terminology;
- ambiguous reference;
- empty results and cross-session isolation;
- visible-complete answer;
- relationship visible/hidden/reversed routing and original/current comparison;
- non-relation mention of “links” through generic routing;
- unique and unlocatable process themes;
- final clarification, blank final, final timeout, and pair-missing paths.

## Real API adapter (opt-in)

The real adapter is never selected implicitly.  It uses an OpenAI-compatible
chat-completions endpoint, sends no native `tools`/`tool_choice`, and applies
one HTTP attempt per phase with no retries.  The decision request includes the
provider JSON response-format parameter when `decision_json_mode` is enabled;
the final request does not.  Local decision-schema validation always runs.

The runner reads the API key only from an explicitly named process environment
variable.  It never loads the production `.env`, never falls back to fake for
a real run, and requires the address, exact model, decision budget, and overall
budget explicitly:

```bash
export GENERAL_TOOL_API_KEY="..."
.venv/bin/python -m experiments.general_tool_search.runner \
  --real-api \
  --api-key-env GENERAL_TOOL_API_KEY \
  --base-url "https://provider.example/v1" \
  --model "exact-model-name" \
  --case alpha_explicit \
  --decision-timeout 15 \
  --overall-timeout 45
```

The code rejects missing credentials/address/model/budgets before starting a
provider request.  `--request-timeout` also bounds each HTTP request.  Retrieval
keeps its independent 1-second cap inside the overall turn deadline.

Both phases require `finish_reason="stop"`.  `length`, `tool_calls`, a missing
finish reason, or any other value is an explicit failure; no later phase is
entered.  A response that arrives after the effective deadline is discarded
even if its JSON is parseable.  HTTP counts are marked at the actual local
transport entry, for both the default transport and an injected MockTransport:
budget exhaustion and preparation failures are zero, while transport timeout,
cancellation, transport failure, and a received HTTP response are one.  The
count means the request reached the local transport; it does not claim provider
receipt.

The CLI prints only status, synthetic flag, counts, source IDs, and a safe list
of transport events (stage, attempt, HTTP-sent, status, elapsed, request model,
upstream model, HTTP status).  It does not print prompts, answers, credentials,
headers, or response bodies.

For a synthetic fake run, `--evidence-dir PATH` explicitly writes
`evidence.json` containing the final reply, injected/retrieved/cited/invalid
source IDs, retrieved teacher content with order, and truncation/pair-missing
flags.  It excludes prompts, requests, decisions, headers, credentials, and
raw provider errors.  Real API runs reject evidence export.

MockTransport tests cover direct retrieval, decision-then-retrieval,
visible-complete decisions, invalid JSON, pre-send budget exhaustion, post-send
timeout, HTTP errors, and final failure.  A real provider first test is still
required for JSON-output support, actual model selection, latency, and answer
quality.

```bash
.venv/bin/python -m experiments.general_tool_search.runner \
  --case no_hidden \
  --answer "The delta fold is [message 7]."
```

The fake CLI reports only experiment status/counts/source IDs and uses a
temporary SQLite database.

## Tests

```bash
.venv/bin/python -m pytest experiments/general_tool_search/tests -q
```

## Known limitations

1. The default clients are fake; the opt-in real adapter is covered only by
   MockTransport.  Provider JSON-mode/finish-reason support, model behavior,
   and first-token latency remain unverified until the authorized real test.
2. Anchor extraction and support detection are lexical; paraphrases, pronouns,
   and some CJK segmentation can miss evidence.
3. The 50,000-character scan and 3-result budget can omit relevant evidence.
4. Search failures are experiment statuses only; no HTTP endpoint is wired.
   Evidence export is synthetic-only and intentionally omits wire data.
5. The production relationship tool and production `.env` remain untouched;
   the CLI reads a key only from an explicitly named process environment
   variable and has a subprocess regression with a synthetic `.env`.
6. Fake calls are not HTTP requests.  Real transport evidence proves the local
   request contract, not provider acceptance or answer quality.
7. The final clarification path checks the request instruction and blank
   output, but does not prove a real model asks only one question.
8. Production relation parsing is single-token/English-limited; unsupported
   relation syntaxes go through the experiment's generic route or clarify.

## Decision-budget A/B/C experiment

`decision_budget.py` is an isolated, decision-only experiment.  It never runs
retrieval or final generation.

| Group | Thinking | max_tokens |
|---|---|---:|
| A | omit `thinking` / `reasoning_effort` | 800 |
| B | `thinking={"type":"disabled"}` | 800 |
| C | omit `thinking` / `reasoning_effort` | 2048 |

All groups use the same `DECISION_SYSTEM`, decision-window data, current user
question, `response_format={"type":"json_object"}`, temperature `0`, model,
and decision deadline.  Synthetic tests use 3 seconds; the authorized real
template uses 12 seconds.  The plan is four cases × three groups × two
independent repeats, for 24 decision requests.  Each item is attempted once;
there is no retry, retrieval, or final generation.

Before any request, the experiment runs every case through the actual
`GeneralSearchEngine` with a decision probe.  A case that reaches final
generation or retrieval first is rejected; the probe must stop at `_decide()`.
A question containing explicit-history language such as `original` or
`did I teach you` is a negative probe case, not a model-decision case.

### Cases

- `session51_original`: requires an external Codex-supplied input file.  The
  fixture contains only a clearly synthetic difference-quotient fallback for
  local tests; real execution refuses to use it.
- `visible_complete`: a normal recall question with sufficient in-window
  evidence and an unrelated older teacher message.
- `hidden_correction`: a non-math onboarding correction outside the window.
- `ambiguous_reference`: two plausible topics and a vague pronoun.

External session 51 input format:

```json
{
  "case_id": "session51_original",
  "current_message_id": 750,
  "messages": [
    {"message_id": 1, "role": "user", "content": "..."},
    {"message_id": 2, "role": "assistant", "content": "..."}
  ],
  "current": "the exact message 750 text",
  "expected": {
    "manual": true,
    "route_criteria": "human-reviewed route/source criteria"
  }
}
```

The external file must preserve the original message IDs, roles, content, and
order.  The fixed case ID `session51_original` always consumes the external
input in real mode; a custom fixture cannot disable the requirement with
`requires_external_input=false`.  The current ID must be exactly 750, history
IDs must be positive, unique, increasing, and below 750, and the external
evidence must include `manual=true` plus a non-empty `route_criteria`.  The
runner takes the most recent 20 stored messages before the current message,
then filters assistant messages.  It rejects any missing or invalid external
field before sending a request.

The real runner requires the file and rejects a missing/invalid case before
sending any request.

### Dry run

```bash
cd /Users/luo/Documents/teachable_agent
.venv/bin/python -m experiments.general_tool_search.decision_budget
```

The default is a fake 24-item dry run.  To persist synthetic evidence:

```bash
.venv/bin/python -m experiments.general_tool_search.decision_budget \
  --evidence-dir /tmp/decision-budget-dryrun
```

### Real run template

The API key is read only from the named environment variable; the runner does
not load `.env`.  Evidence must be outside the repository.

```bash
export DECISION_BUDGET_API_KEY='...'
.venv/bin/python -m experiments.general_tool_search.decision_budget \
  --real-api \
  --api-key-env DECISION_BUDGET_API_KEY \
  --base-url https://api.deepseek.com/v1 \
  --model deepseek-v4-flash \
  --decision-timeout 12 \
  --session51-input /outside/repo/session51_decision_input.json \
  --evidence-dir /outside/repo/decision-budget-runs/run-001
```

The real command requires explicit `--model` and a finite positive
`--decision-timeout`; it stops before any request if the key, model, timeout,
external input, or exclusive evidence directory is unavailable.  It writes `run_meta.json`,
`attempts.jsonl`, `attempts.csv`, and `summary.json`.  Its SQLite database is
temporary and is removed after success or failure; no database copy is left in
the evidence directory.  Synthetic runs are marked `synthetic: true`;
MockTransport runs must not be reported as real upstream evidence.

### Scoring

`output_valid` requires HTTP success, `finish_reason="stop"`, non-empty text,
and strict `parse_decision()` validation.  `finish_reason="length"` is always
truncated even if the body happens to parse.  A blank body on an HTTP response
is counted separately from a timeout or connection failure with no response.
`route_correct` is separate and is computed only for cases with frozen
expectations; session 51 is manual.  An invalid or truncated output is never
counted as a correct route.  HTTP non-2xx responses are recorded as
`http_error`, not as model-empty bodies.  HTTP 2xx is the success range here,
matching the production client; a 201 with a valid structure is therefore a
success.  A 2xx response with empty or whitespace-only model text is
`content_empty` and is not sent to the JSON parser.  A 2xx response that is
missing `choices` or `message`, or whose root/choice/message has the wrong
type, is `invalid_response`, not an empty body.  If the HTTP response already
exists, the invalid record still preserves the independently readable
`finish_reason`, `completion_tokens`, and `reasoning_tokens`; a `length`
response with null content therefore remains truncated and invalid, with no
response status reported as `no_response`.  A
connection failure or timeout with no HTTP response is
`no_response`/`timeout`.  A `length` response with an empty body is both
truncated and empty, and is never valid JSON.  The summary counts
`invalid_response`, `http_error`, `empty`, `no_response`, and `timeout`
separately.  The summary
reports separate denominators for valid JSON, correct routes, manual-pending
routes, HTTP errors, empty bodies, and no-response cases.

Evidence records only the case/group/repeat, request model, upstream model,
parameters, timing, transport status, HTTP status, finish reason, empty-body
flag, validation result, actual completion/reasoning token counts when
provided, structured decision fields, and a redacted error code.  It does not
save credentials, headers, full prompts/history, raw response bodies, or
internal reasoning.
