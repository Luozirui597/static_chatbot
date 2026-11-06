# search_teaching_history experiment

First isolated, fake-client experiment for structured, read-only retrieval of
earlier teacher messages.  This directory does not modify production
`LLMMessage`, `ChatService`, prompts, API routes, or databases.

## Scope

- Temporary SQLite only.
- Synthetic fixture (`fixtures/window_outsider.json`).
- Fake structured tool client only.
- No network, real API, browser, browser application, or production DB.
- No knowledge-state writes; no inference tool; no Skill layer.

The first round verifies orchestration, budget enforcement, session isolation,
and evidence records.  It does not verify real-provider tool-call compatibility
or real model behavior.

## Structured exchange

The experiment uses an experiment-only assistant turn:

```python
AssistantTurn(
    content=None,
    tool_calls=(
        ToolCall(
            id="call_1",
            name="search_teaching_history",
            arguments={"query": "A B C links correction", "limit": 3},
        ),
    ),
)
```

The runner appends:

1. `assistant` with `tool_calls`;
2. a `role="tool"` message with the same `tool_call_id`;
3. a second fake generation.

Tool exchanges stay in the run context and evidence records.  They are never
inserted into the teacher-message table and do not enter later Receive history.

## Conditions

`runner.py` implements three conditions over the same fixture and current
question, using a separate temporary database per condition:

- `window`: the production Receive window: last 20 stored rows before the
  current message, chronological, assistant rows filtered out; no tool.
- `fixed`: one program retrieval using `fixture.fixed_query`; the synthetic
  exchange is marked `program_generated=True` and `program_retrievals=1`.
- `tool`: the fake model may make one structured request; a second generation
  follows the `role="tool"` result.

Call limits are one tool call and two generations per turn.  A second tool
request is rejected without a third generation.

## Tool limits

- Query: 1–200 characters.
- Query tokens: first 6 distinct tokens, in first-occurrence order.
- SQL candidate rows: 200.
- Scanned characters: 50,000.
- Returned rows: up to 5.
- Per-message content: 1,000 characters.
- Total returned content: 4,000 characters.
- Tool deadline: 1 second, shared by SQLite progress interruption and Python
  scanning.
- SQL is parameterized and session-bound.  The model cannot pass a session id
  or current message id.

Results include `message_id`, teacher `order`, exact content (or a marked
prefix), `kind`, and flags:

- `scan_truncated`;
- `truncated`;
- `corrections_truncated`;
- `omitted_count` (only candidates confirmed omitted in the scanned range).

A message is a `correction_candidate` only when it has an explicit correction
cue and there is an earlier, already-scanned, non-correction teacher message
sharing at least one query token.  Isolated negations and corrections without
visible support are returned as ordinary teaching matches; they do not reserve
a correction slot.

Newer correction candidates are selected first and may reserve up to two slots
and use the total content-character budget first.  Remaining budget is then
allocated to teaching matches by the existing match ranking.  Only after
selection is the return list sorted into teacher order.  `truncated`,
`corrections_truncated`, and `omitted_count` are computed from the actual
selected/truncated content, not from a pre-budget count.

Corrections use an explicit cue plus a shared query token.  A pronoun-only
correction, a correction whose support is outside the scan, or one without a
shared token may be missed.  The fixture includes a pronoun-style limitation
case.  `status="ok"` does not mean the search was exhaustive; the model must
state when evidence is incomplete.

## Running tests

```bash
cd /Users/luo/Documents/teachable_agent
.venv/bin/python -m pytest experiments/tool_search_history/tests -q
```

The tests use temporary SQLite files and a scripted fake client.  They do not
call a model, network, or production database.

## Evidence in the fake round

`ConditionResult` records:

- condition, prompt SHA-256, status, final assistant id;
- model and tool call counts;
- program retrieval count;
- tool exchanges (`source`, call id, result);
- model message batches for the fake client;
- source IDs extracted from `[message N]` citations;
- model/tool latency.

No run is written to the production database.

## Known limitations

1. The structured messages are experiment-only; production `LLMMessage` only
   supports text `system/user/assistant`.
2. Native provider compatibility is deferred.
3. Correction detection is lexical, not semantic.
4. A search can miss relevant evidence due to the scan budget, the six-token
   query cap, missing visible correction support, or pronouns.
5. Fake-client checks do not validate real model tool selection, answer
   quality, or latency.

## DeepSeek adapter (mock-tested only)

`deepseek_client.py` implements the experiment-only structured client for
direct DeepSeek Chat Completions:

- URL: `https://api.deepseek.com/chat/completions`
- model: `deepseek-flash`
- thinking disabled: `{"thinking": {"type": "disabled"}}`
- non-streaming; temperature `0.0`; max tokens `800`
- first request `tool_choice="auto"`; second request `tool_choice="none"`
- both requests use the same `TOOL_SCHEMA`
- per-request timeout 30 seconds, limited by the remaining 65-second turn
  deadline; tool execution is limited to 1 second and the same turn deadline
- no retry

The runner passes `phase`, `tool_choice`, `tools`, and the absolute turn
`deadline` explicitly. The adapter creates one attempt before validation, records
whether an HTTP request was actually sent, and the runner collects that attempt
in a `finally` block so a later call cannot overwrite it.

The adapter preserves the provider's raw tool-call argument string separately
from the parsed dictionary. The second request appends the original assistant
tool-call message and the same-`tool_call_id` tool result. The adapter never
executes a tool and never accepts a tool call after the second phase.

Tests use `httpx.MockTransport` only. No real DeepSeek request, credential, or
network call is made by this test suite. A real first run remains a separate
Codex/operator acceptance step with explicitly supplied credentials.

## Programmatic condition (isolated)

`programmatic` is a new experiment-only condition. It does not alter
`window`, `fixed`, or the model-native `tool` condition.

- The runner loads the same last-20 stored-message window as Receive.
- It computes `visible_start_id` and checks for an earlier teacher message
  with a bounded `EXISTS` query.
- A small English parser handles `Fact: B links to C.`,
  `Correction: B does not link to C.`, `Does B link to C?`, the quoted
  comparison form, and mixed teaching-plus-question input.
- Retrieval is hidden-only: same session, `role=user`,
  `id < visible_start_id`.
- The model receives a single programmatic request with `tools=[]`,
  `tool_choice="none"`, and no `tools`/`tool_choice` HTTP fields.
- A programmatic run performs at most one retrieval and one model request.
  Tool timeout/error stops before the model request.
- Retried or unexpected tool calls stop the run; the model-requested tool
  execution count remains zero for programmatic runs.

Evidence separates:

- retrieved IDs;
- injected window/hidden source IDs;
- cited IDs and invalid cited IDs;
- programmatic status;
- human semantic judgment about correction use.

The old `window`, `fixed`, and native `tool` tests remain in the suite. The
programmatic tests use temporary SQLite databases, `FakeToolClient`, and
`httpx.MockTransport`; no real API is called.
