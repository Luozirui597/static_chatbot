# Static Chatbot

A persistent, multi-session chat application with a FastAPI backend
and a vanilla HTML/CSS/JavaScript frontend.  Supports local development
with a fake echo mode or integration with a configured
OpenAI-compatible chat-completions API.

## Development version

- `static-baseline-v1` freezes the original Static Chatbot baseline.
- The current `feature/history-review` branch is the Teachable Agent
  development version.
- **Iteration 1** adds two switchable interaction modes
  (`receive_teaching`, `corrective`) and records an immutable boundary
  for every switch.  Corrective mode only changes how *future* turns are
  handled; it never inspects prior history on its own.
- **Iteration 2** adds **History Review**: an explicit, user-started
  audit of the teaching history that Iteration 1 froze at a mode
  switch.  A review is a **separate operation with its own lifecycle** —
  switching to `corrective` never starts one automatically, and nothing
  is sent to a remote reviewer until the user confirms it.

## Features

- FastAPI backend with async LLM client
- Vanilla HTML / CSS / JavaScript frontend (no frameworks)
- Two LLM modes: **fake** (deterministic, no network) and **real**
  (OpenAI-compatible `/chat/completions` API)
- SQLite persistence — sessions and messages survive restarts
- Multi-turn conversations with a configurable history window
- Multiple isolated sessions — create, switch, rename, and delete
- Auto-generated session titles from the first user message
- Selectable LLM model for new chats (API / local / fake profiles)
  plus per-conversation model switching — the model is chosen when a
  session is created and can be re-bound at any time via the
  **Model for this chat** control; there is no automatic fallback
  between models
- Non-ready legacy sessions (model removed, model configuration
  changed, or created before model tracking) remain readable but are
  read-only
- Responsive layout with a collapsible sidebar on mobile
- Loading, empty, and error states in the UI
- Input validation (blank and over-length messages are rejected)
- Complete Python and Node test suites covering APIs, models, business
  logic, LLM client behaviour, session isolation, concurrency,
  auto-title generation, session rename, schema migration, history
  review selection/execution/persistence, and error handling
- Frontend unit tests for clipboard logic, copy button state machine,
  network-error recovery, model-selection logic, session
  model-switching logic, interaction-mode helpers, and the history
  review panel/controller (Node `node:test`)
- Two per-session interaction modes:
  `receive_teaching` (non-corrective teaching) and `corrective`
  (active correction for future turns)
- Interaction mode is independent of the LLM profile/model: switching
  mode never changes `llm_profile_id`, the model snapshot, or the
  **Model for this chat** control
- Per-message interaction-mode and prompt-version snapshots plus a
  persisted `mode_switch_events` log
- **History Review** — an explicit, user-started audit of prior
  teaching history:
  - a *frozen boundary*: each mode switch records the message id the
    history runs through, so a review always sees the same history no
    matter when it is started
  - *deterministic source selection*: eligible user messages are picked
    newest-first under a fixed token/character budget with a versioned
    policy, never re-selected while a review exists
  - a *reviewer snapshot*: the reviewer profile, model, prompt version
    and source ids are frozen when the review starts
  - an explicit *privacy confirmation* whenever the selected source
    messages were not produced by the same API reviewer profile — a
    non-API (local/fake) reviewer never asks, and an API reviewer asks
    only when at least one selected source message carries a different
    profile id, profile kind, or model snapshot
  - a strict *state machine* (`pending` → `running` →
    `completed` / `failed`) with a single CAS claim, at most one LLM
    call per review, and a strict JSON parser that fails a review with
    a stable error code instead of storing partial output
  - a per-session review panel with status, findings, coverage note,
    review selector, Recheck/Reload, and Continue actions

## Project structure

```
backend/
  main.py              FastAPI app, routes, static mount
  schemas.py           Pydantic request / response models
  chat_service.py      ChatService — business logic
  history_boundary.py  Deterministic mode-switch boundary capture
  history_review_selection.py  Deterministic review source selection
  history_review_prompt.py     Frozen prompt inputs / construction
  history_review_parser.py     Strict reviewer-output parser
  history_review_service.py    Review preparation, CAS claim, execution
  llm_client.py        LLMClient protocol, FakeLLMClient,
                       OpenAICompatibleLLMClient, factory
  llm_profiles.py      LLM profile registry and session binding
  config.py            Environment configuration
  database.py          SQLAlchemy engine, session factory, get_db
  models.py            ORM models — ChatSession, Message,
                       ModeSwitchEvent, HistoryReview,
                       HistoryReviewSource, HistoryReviewFinding
  exceptions.py        Service-level exception types
  system_prompt.py     Fixed legacy chat prompt
  interaction_modes.py Interaction modes, prompt versions, prompts
frontend/
  index.html           Multi-session chat page
  style.css            Responsive styles
  network-recovery.js  Pure helper for send-failure recovery
  clipboard.js         Clipboard API + execCommand fallback
  copy-controller.js   Per-button copy state machine
  interaction-mode.js  Pure helpers for the interaction-mode control
  model-selection.js   Pure helpers for the model selector
  session-profile-switch.js  Session model-switch controller,
                       confirmer, and outcome planners
  history-review.js    Pure history-review logic: validators, proposal
                       selection, cache reconciliation, panel model,
                       execution controller
  app.js               Frontend logic (vanilla JS)
tests/
  conftest.py          Forces LLM_MODE=fake for all tests
  test_health.py       Health endpoint
  test_chat.py         Legacy chat endpoint & route error tests
  test_llm_factory.py  LLM client factory
  test_real_client.py  OpenAI-compatible client unit tests
  test_models.py       ORM model constraints and relationships
  test_chat_service.py ChatService business logic & transactions
  test_sessions.py     Session CRUD API
  test_session_chat.py Session message send API, concurrency, lock safety
  test_interaction_mode.py    Interaction-mode API, prompts, snapshots, migration
  test_history_boundary.py    Boundary capture rules
  test_history_review_selection.py  Deterministic source selection
  test_history_review_prompt.py     Prompt construction / snapshots
  test_history_review_parser.py     Strict output parsing
  test_history_review_service.py    Preparation and execution service
  test_history_review_storage.py    Review/source/finding persistence
  test_history_review_storage_migration.py  Schema migration
  test_history_review_execution.py  CAS claim and execution outcomes
  test_history_review_api.py        History-review HTTP API
  test_clipboard.test.js         Frontend clipboard helper tests
  test_network_recovery.test.js  Frontend send-failure recovery tests
  test_interaction_mode.test.js  Frontend interaction-mode helper tests
  test_model_selection.test.js   Frontend model-selection helper tests
  test_session_profile_switch.test.js  Frontend session model-switch tests
  test_history_review.test.js    Frontend history-review controller tests
  test_history_review_dom.test.js  Panel/dialog DOM, CSS and wiring tests
.env.example           Documented environment variables
requirements.txt       Python dependencies
```

## Quick start

1.  **Clone the repository**

    ```bash
    git clone https://github.com/Luozirui597/static_chatbot.git
    cd static_chatbot
    ```

2.  **Create and activate a virtual environment**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**

    ```bash
    python -m pip install -r requirements.txt
    ```

4.  **Configure the environment**

    ```bash
    cp .env.example .env
    ```

    Edit `.env` to set `LLM_MODE` and, if using real mode, your API
    credentials.  See [Environment variables](#environment-variables)
    for the full list.

5.  **Start the server**

    ```bash
    python -m uvicorn backend.main:app --reload
    ```

6.  **Open the app**

    Visit **http://127.0.0.1:8000** in a browser.

## LLM modes

### Fake mode (default)

In fake mode the `FakeLLMClient` is used — every reply echoes back the
last user message as a Chinese test response.  No network calls are
made, no API key is required.  This mode is suitable for development
and for running the test suite.

Minimal `.env`:

```env
LLM_MODE=fake
```

### Real mode

In real mode the `OpenAICompatibleLLMClient` sends requests to the
configured `/chat/completions` endpoint.  It is designed for services
that implement the OpenAI chat-completions protocol.  DeepSeek is the
provider configuration currently verified for this project; other
providers may require adjustments.

Real mode requires an API key, base URL, and model name.  External
API requests **may incur costs** from your provider.

Example `.env` (placeholders only — use your own credentials):

```env
LLM_MODE=real
LLM_API_KEY=<your-api-key>
LLM_API_BASE_URL=<your-base-url>
LLM_MODEL=<your-model-name>
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LLM_MODE` | `fake` | `fake` — deterministic echo replies; `real` — calls a chat completions API |
| `LLM_API_KEY` | — | API key (required when `LLM_MODE=real`) |
| `LLM_API_BASE_URL` | — | API base URL, e.g. `https://api.openai.com/v1` |
| `LLM_MODEL` | — | Model name sent in the request body |
| `LLM_REASONING_EFFORT` | — | `none` / `low` / `medium` / `high`. Leave empty for model default. For local models set to `none` to suppress hidden reasoning chains. |
| `DATABASE_URL` | `sqlite:///data/chatbot.db` | SQLite connection string; the `data/` directory is created automatically on first run |

### Local LLM configuration (example)

When running against a local Ollama instance (e.g. `qwen3.5:4b`):

```env
LLM_MODE=real
LLM_API_KEY=ollama
LLM_API_BASE_URL=http://127.0.0.1:11435/v1
LLM_MODEL=qwen3.5:4b
LLM_REASONING_EFFORT=none
```

Start the backend with temporary environment variables:

```bash
LLM_MODE=real \
LLM_API_KEY=ollama \
LLM_API_BASE_URL=http://127.0.0.1:11435/v1 \
LLM_MODEL=qwen3.5:4b \
LLM_REASONING_EFFORT=none \
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`load_dotenv(override=False)` ensures terminal environment variables
take precedence over the existing `.env` file without overwriting it.

## Data persistence

Messages and sessions are stored in a local SQLite database.  The
default database file is `data/chatbot.db` — it is created
automatically on first startup if it does not exist.

Core chat data:

- Database tables (`chat_sessions`, `messages`) are initialised when
  the application starts.
- All sessions and their messages persist across server restarts.
- Deleting a session via the API cascades to its messages — all
  related data is removed.
- The full conversation history for each session is stored in the
  database.
- Only the **20 most recent prior messages** are sent to the LLM as
  conversational context.  Messages beyond that window remain in the
  database but are not included in LLM requests.

Interaction-mode and history-review data:

- `mode_switch_events` — one immutable row per **real interaction-mode
  change**, in both directions: `receive_teaching` → `corrective` and
  `corrective` → `receive_teaching`.  Every row stores the frozen upper
  boundary (`history_through_message_id`), the boundary version and the
  reviewable user-message count.  These fields are never recomputed
  after the switch, so a review started later still sees the history as
  it was at the switch.  Only `receive_teaching` → `corrective` events
  with a valid boundary are eligible for History Review: on the reverse
  direction (and when the boundary cannot be trusted) the reviewable
  count is `null`, and the review API rejects such an event as not
  reviewable.  The event row does **not** store a lower bound.
- `history_reviews` — the review state machine (`pending`, `running`,
  `completed`, `failed`) plus the frozen reviewer snapshot (profile id,
  label, model, prompt version), the selection/budget policy versions
  and source counters, the review summary, coverage note, error
  code/message, findings count, and the created/started/completed/
  updated timestamps.  For an eligible review the lower bound is
  *derived* while the review is prepared, from the preceding
  `corrective` → `receive_teaching` event in that session (`session_start`
  when there is none), and is then frozen in
  `history_reviews.lower_bound_kind` and
  `history_reviews.lower_bound_message_id` — never written back onto the
  mode-switch event.  At most **one** history review exists per
  mode-switch event (enforced by the
  `uq_history_reviews_session_event` constraint on
  `session_id` + `mode_switch_event_id`); execution is claimed with a
  compare-and-swap update so a review is never executed twice.
- `history_review_sources` — the **frozen** set of source messages for a
  review (message id, sequence, role, snapshot of the text and byte
  length).  The stored text is what the reviewer saw, so later message
  edits never rewrite review history.
- `history_review_findings` — the parsed findings for a completed
  review: sequence, source message id, verdict
  (`correct` / `incorrect` / `uncertain` / `not_a_claim`), the claim
  text, and the optional correction and explanation text.
- Deleting a session cascades to its mode-switch events, reviews,
  sources and findings.
- Schema upgrades are recorded in `schema_migrations`, so an existing
  database is migrated once and never rewritten on later startups.

## API

Interactive API documentation (Swagger UI) is available at:

> **http://127.0.0.1:8000/docs**

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat` | Send a stateless chat message (legacy) |
| `POST` | `/api/sessions` | Create a new chat session |
| `GET` | `/api/sessions` | List all sessions (newest first) |
| `GET` | `/api/sessions/{id}` | Get a single session |
| `GET` | `/api/sessions/{id}/messages` | Get messages for a session |
| `POST` | `/api/sessions/{id}/messages` | Send a message within a session |
| `PATCH` | `/api/sessions/{id}` | Rename a session |
| `PATCH` | `/api/sessions/{id}/llm-profile` | Switch the session's LLM profile |
| `PATCH` | `/api/sessions/{id}/interaction-mode` | Switch between `receive_teaching` and `corrective` |
| `GET` | `/api/sessions/{id}/mode-switch-events` | List the session's mode-switch events and their frozen boundaries |
| `POST` | `/api/sessions/{id}/history-reviews` | Start (or reuse) a history review for a mode-switch event |
| `GET` | `/api/sessions/{id}/history-reviews` | List the session's review summaries (newest first) |
| `GET` | `/api/sessions/{id}/history-reviews/{review_id}` | Get one review's detail, sources and findings |
| `DELETE` | `/api/sessions/{id}` | Delete a session and its messages |

The history-review endpoints behave as follows:

```http
GET  /api/sessions/{session_id}/mode-switch-events
POST /api/sessions/{session_id}/history-reviews
GET  /api/sessions/{session_id}/history-reviews
GET  /api/sessions/{session_id}/history-reviews/{review_id}
```

- `POST /api/sessions/{id}/history-reviews` returns **201** for a newly
  created review and **200** when an existing review for the same
  mode-switch event is reused.
- `GET /api/sessions/{id}/mode-switch-events` returns **every** recorded
  mode change for the session (newest first), in both directions.
  Events whose `reviewable_user_message_count` is `null` — the reverse
  `corrective` → `receive_teaching` direction, or a boundary that could
  not be trusted — are listed but cannot be reviewed.
- The acknowledgement requirement for a remote reviewer is *conditional*:
  - a **non-API** reviewer (local / fake) never requires it;
  - an **API** reviewer whose selected source messages all carry that
    same profile id, a profile kind of `api`, and the same model
    snapshot as the reviewer never requires it;
  - if **at least one** selected source message differs in profile id,
    profile kind, or model snapshot, the request returns **409**
    `history_review_remote_ack_required` with the acknowledgement
    metadata (source message count, reviewer label/model, truncation
    flag).  The frontend shows a privacy confirmation and retries once
    it is accepted.
- A concurrent execution returns **409**
  `history_review_execution_conflict`.
- Sessions with no reviewable history, an unusable boundary, or an
  event that is not a supported switch fail with **422** and never
  reach an LLM.
- The reviewer output is parsed strictly, and a parse failure fails the
  review instead of storing partial data.  The stable parser error
  codes are:
  `history_review_output_too_large`,
  `history_review_json_syntax_error`,
  `history_review_json_structure_error`,
  `history_review_json_semantic_error`,
  `history_review_source_reference_invalid`, and
  `history_review_source_uncovered`.

The web interface is served at:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat page (`frontend/index.html`) |
| * | `/static/*` | Static files (CSS, JavaScript) |

### Example: legacy stateless chat

```json
// POST /api/chat
{"message": "Hello"}

// Response (fake mode)
{"reply": "测试回复：Hello"}
```

### Example: create a session

```json
// POST /api/sessions → 201

{
  "id": 1,
  "title": "New Chat",
  "interaction_mode": "receive_teaching",
  "created_at": "2026-08-06T12:00:00",
  "updated_at": "2026-08-06T12:00:00"
}
```

### Example: send a message in a session

```json
// POST /api/sessions/1/messages
{"message": "Hello"}

// Response → 200
{
  "user_message": {
    "id": 1,
    "session_id": 1,
    "role": "user",
    "content": "Hello",
    "created_at": "2026-08-06T12:00:01"
  },
  "assistant_message": {
    "id": 2,
    "session_id": 1,
    "role": "assistant",
    "content": "测试回复：Hello",
    "created_at": "2026-08-06T12:00:02"
  }
}
```

### Example: get session messages

```json
// GET /api/sessions/1/messages → 200

[
  {
    "id": 1,
    "session_id": 1,
    "role": "user",
    "content": "Hello",
    "created_at": "2026-08-06T12:00:01"
  },
  {
    "id": 2,
    "session_id": 1,
    "role": "assistant",
    "content": "测试回复：Hello",
    "created_at": "2026-08-06T12:00:02"
  }
]
```

## Frontend behaviour

The frontend uses the session API for all chat interactions.  The
legacy `POST /api/chat` endpoint remains available but is **not used**
by the current UI.

- Every session has an **Interaction mode for this chat** control with
  `receive_teaching` and `corrective`; switching mode never changes the
  session model profile.
- `receive_teaching` is the default for new sessions.
- Iteration 1 `corrective` mode only changes how subsequent messages
  are handled.  Switching to it does **not** review, summarise or
  rewrite prior history — a history review is always a separate,
  explicitly started operation (Iteration 2, below).
- **History Review** panel:
  - A switch to `corrective` that has reviewable prior user messages
    shows a **Review available** proposal.  Nothing runs until the user
    presses **Review previous teaching** and confirms the start dialog;
    **Not now** dismisses the proposal for that switch.
  - Starting a review asks for confirmation first.  A **second** privacy
    confirmation appears only when the request needs remote
    acknowledgement — that is, when an API reviewer would receive
    selected source messages whose profile id, profile kind or model
    snapshot differs from its own.  It lists how many messages would be
    sent, which reviewer profile/model would receive them, and whether
    the history is truncated.  Local/fake reviewers, and API reviewers
    whose sources all match, go straight through.
  - A review always uses the history frozen at its mode switch, so a
    review started later sees exactly the same messages.
  - The panel shows the review status badge (`Pending`, `Running`,
    `Completed`, `Failed`, plus `Review available`, `Working`,
    `Uncertain`, `Error`, `Loading`), the summary, the coverage note,
    and the per-finding verdicts with their source message ids.
  - The review selector appears only when the session has **two or
    more** reviews; a session with a single review shows no dropdown.
  - **Recheck** re-queries an unreliable or running review and a failed
    one; **Reload** re-fetches a missing or outdated detail; **Continue
    review** resumes a pending review.
  - A review in flight keeps the session busy: sending, mode switching,
    model switching, renaming and deleting are blocked for that
    session until it settles.  A running review in one session never
    blocks another session.
  - If the review list cannot be loaded, the panel fails closed and
    offers **Reload**; a cached detail is only rendered while it still
    matches the authoritative review summary.

- The sidebar lists all sessions, newest first (ordered by
  `updated_at` descending on the server).
- **+ New Chat** creates a session immediately and selects it.
- A **Model for new chats** selector next to the button chooses which
  LLM profile the *next* new session uses.  Changing it never changes
  the current session — to use another model with an existing chat,
  use the **Model for this chat** bar described below.  There is no
  automatic fallback between models.
- The chat header shows the current session's actual model label
  (with an API / Local / Fake badge when known).
- Sessions whose model is no longer available, whose model
  configuration changed, or that were created before model tracking
  are readable but read-only — renaming, deleting, copying and
  creating new chats still work.
- Click a session in the sidebar to switch to it.
- Click the **×** button to delete a session (confirmation required).
- Messages are rendered with `textContent` — no HTML injection.
- Loading, empty, and error states are shown in the chat area and
  status bar.
- **Enter** sends the message; **Shift+Enter** inserts a newline.
- On page reload the most recently updated session is opened
  automatically and the model selector returns to the server-declared
  default profile.
- The sidebar collapses on narrow screens (≤ 767 px); tap the toggle
  button (☰) to open or close it.
- Each conversation shows a **Model for this chat** bar with an Apply
  button.  The model is chosen when the conversation is created, but
  existing conversations can be re-bound to another available model
  at any time; switching never deletes or rewrites history, so one
  conversation may contain a mix of models.  Messages created after
  model tracking was introduced record their own model provenance;
  messages from before that migration keep null snapshot fields and
  are never back-filled with a guessed model.
- Switching a conversation with history to a remote API model asks
  for confirmation first — the most recent chat history will be sent
  to the remote API service on the next message.  Local models are
  used without any such confirmation, and there is no automatic
  fallback between models.
- If a switch result cannot be confirmed (for example a network
  interruption), the conversation shows a persistent notice; press
  **Apply** again to check the current binding before retrying.
- If the local Ollama service is stopped, sending fails with the
  usual saved-message recovery.  Restart the local service and send
  the message again to continue — Apply does not start or repair the
  Ollama service, and re-applying the same profile on a ready binding
  is idempotent, so Apply stays disabled there.
- If an upstream API error occurs, the user message may still be
  saved (it is committed before the LLM is called).  The frontend
  re-synchronises message history after such errors to reflect the
  saved state.

## Local Ollama service

A project-local Ollama (v0.32.6) is installed under `local_llm/Ollama.app/`.
This workflow uses the project-local Ollama runtime and does not
use, install, or modify system-wide Ollama files or symlinks.  The `local_llm/` directory
is excluded by `.gitignore` — the Ollama app, models and runtime data are not
part of the Git repository.  These instructions describe the current research
machine; a fresh checkout requires its own Ollama runtime preparation.

### Start the service

```bash
bash scripts/start-local-ollama.sh
```

The service runs in the foreground, listening **only** on
`127.0.0.1:11435`.  All runtime state stays under `local_llm/`.

Expected log output (first few lines):

```text
Ollama cloud disabled: true
Listening on 127.0.0.1:11435 (version 0.32.6)
```

### Verify the service

In another terminal:

```bash
curl http://127.0.0.1:11435/api/version
```

A successful response confirms the service is running, **not** that any
model is installed:

```json
{"version": "0.32.6"}
```

### Stop the service

Press `Ctrl+C` in the terminal where `start-local-ollama.sh` is running.

### Notes

- This workflow uses only the project-local CLI.  It does not use or
  modify the existing `/usr/local/bin/ollama` symlink.
- The local model `qwen3.5:4b` (4.7B params, Q4_K_M, ~3.4 GB) is
  installed under `local_llm/models/` (excluded from Git).
- Digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- `LLM_REASONING_EFFORT=none` prevents this local model from producing
  unneeded extended hidden reasoning for ordinary chatbot replies.
- The environment variable `OLLAMA_NO_CLOUD=1` disables Ollama cloud
  features.  Verify the log contains `Ollama cloud disabled: true`.
- The `/api/version` success only confirms the service is running; it
  does not mean any model is installed.
- Step 2D verification logs and reports are preserved in
  `local_llm/logs/`.

## Testing

```bash
# Python tests
.venv/bin/python -m pytest -q

# Frontend tests (requires Node.js)
node --test tests/*.test.js
```

Both suites are complete and green: the Python suite and the Node
suite each run as a whole (`tests/*.test.js` collects every frontend
test file, including the history-review controller and DOM/CSS
contract tests).  Run them without a build step — no bundler, no
framework, and no network access.

- `conftest.py` forces `LLM_MODE=fake` and `DATABASE_URL=sqlite:///:memory:`
  before any test module is imported — no test ever touches a real
  LLM API or the production database.
- Session and message API tests use temporary SQLite files created
  per test run.
- The spy / mock LLM client records every `generate()` call and
  supports both configurable responses and injected errors.
- Concurrency tests use ``asyncio.Event``-controlled spies to assert
  structural invariants (``max_active``) instead of wall-clock
  thresholds.  Coverage includes lock-cancellation safety and
  delete-during-generation races.
- History-review tests cover boundary capture, deterministic source
  selection, prompt construction, strict parsing, preparation and CAS
  execution, persistence and migration, and the HTTP API.
- Frontend tests exercise clipboard logic, copy-button state, network
  recovery, model selection, session model switching, interaction
  modes, and the history-review controller/panel — including cache
  reconciliation, busy rules, dialog cancellation, and the panel's
  DOM and CSS contracts (a small in-repo CSS reader, no browser
  engine required).

## Security and privacy

- API credentials are read by the backend from environment variables
  and are **never** embedded in frontend source code or returned in
  API responses.
- `.env` is excluded by `.gitignore` and must not be committed.
- All user and assistant content in the UI is inserted using
  `textContent` — no `innerHTML` usage for dynamic content.
- Request validation limits messages to 4000 characters.
- Deleting a session cascades to its messages in the database.
- Database-level constraints reject empty or blank message content
  and enforce valid message roles.

**Important:** There is currently **no authentication or
authorisation**.  This is a local, single-user development
application.  Do not expose it on a public network without adding
appropriate security controls.

## Current limitations

- No authentication or multi-user support.
- Session titles are auto-generated from the first user message and
  can be manually renamed via the API and UI.
- Messages cannot be edited or deleted individually.
- No streaming (server-sent events) responses.
- No Markdown or rich-text rendering in message bubbles.
- No retry or regenerate-action for a failed assistant reply.
- No message search or conversation export.
- Context window hard-coded at 20 most recent prior messages per
  session.
- Only tested with a single OpenAI-compatible provider (DeepSeek);
  other providers may require adjustments.
- Not hardened for production deployment.

### Per-session lock scope

The per-session ``asyncio.Lock`` guarantees only **single-process**
serialisation — at most one request executes per session at a time
within a single uvicorn worker.  Multi-worker or multi-instance
deployments require additional coordination such as a database-level
lock (e.g. ``SELECT … FOR UPDATE``), an external queue, or a
distributed lock manager.

### Known lint / type-check items

The codebase has **23 ruff items** (B008×7, I001×5, DTZ001×5, RUF100×2,
UP037×2, UP035×1, UP006×1) and **3 mypy items** that are intentional or
pre-existing:

- Ruff ``B008``: ``Depends(get_db)`` in FastAPI route signatures is
  the standard dependency-injection pattern — these are not defects.
- Ruff ``DTZ001``: naive ``datetime`` objects used for SQLite
  compatibility (SQLite stores datetimes as strings without timezone).
- Ruff ``UP035`` / ``UP006`` / ``UP037``: legacy typing imports
  retained for clarity alongside SQLAlchemy ``Mapped[]`` types.
- Ruff ``I001``: import-block ordering (cosmetic).
- Ruff ``RUF100``: unused ``noqa`` for a non-enabled rule (cosmetic).
- Mypy 3 items: ``.reverse()`` on ``Sequence`` return type, and
  ORM-model / Pydantic-schema type mismatches in route handlers.
  Both are benign at runtime.

These are tracked but not treated as release blockers.

## Future work

Future versions may extend this baseline into a learning-by-teaching
chatbot with an explicit knowledge state and an adaptive learner
model.  Candidate next steps include a persistent learner/knowledge
model that consumes review findings, follow-up reviews as new teaching
accumulates, and streaming replies.  No such feature is implemented in
the current branch.
