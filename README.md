# Static Chatbot

A persistent, multi-session chat application with a FastAPI backend
and a vanilla HTML/CSS/JavaScript frontend.  Supports local development
with a fake echo mode or integration with a configured
OpenAI-compatible chat-completions API.

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
  — the model is fixed when a session is created; existing sessions
  cannot be switched to another model, and there is no automatic
  fallback between models
- Non-ready legacy sessions (model removed, model configuration
  changed, or created before model tracking) remain readable but are
  read-only
- Responsive layout with a collapsible sidebar on mobile
- Loading, empty, and error states in the UI
- Input validation (blank and over-length messages are rejected)
- 297 automated Python tests covering APIs, models, business logic,
  LLM client behaviour, session isolation, concurrency, auto-title
  generation, session rename, schema migration, and error handling
- Frontend unit tests for clipboard logic, copy button state machine,
  network-error recovery, and model-selection logic (Node `node:test`)

## Project structure

```
backend/
  main.py              FastAPI app, routes, static mount
  schemas.py           Pydantic request / response models
  chat_service.py      ChatService — business logic
  llm_client.py        LLMClient protocol, FakeLLMClient,
                       OpenAICompatibleLLMClient, factory
  config.py            Environment configuration
  database.py          SQLAlchemy engine, session factory, get_db
  models.py            ORM models — ChatSession, Message
  exceptions.py        LLMError, LLMInvalidResponseError
  system_prompt.py     Fixed system prompt
frontend/
  index.html           Multi-session chat page
  style.css            Responsive styles
  network-recovery.js  Pure helper for send-failure recovery
  clipboard.js         Clipboard API + execCommand fallback
  copy-controller.js   Per-button copy state machine
  model-selection.js   Pure helpers for the model selector
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
  test_network_recovery.test.js  Frontend send-failure recovery tests
  test_model_selection.test.js   Frontend model-selection helper tests
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
| `DELETE` | `/api/sessions/{id}` | Delete a session and its messages |

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

- The sidebar lists all sessions, newest first (ordered by
  `updated_at` descending on the server).
- **+ New Chat** creates a session immediately and selects it.
- A **Model for new chats** selector next to the button chooses which
  LLM profile the *next* new session uses.  Switching the selector
  never changes the current session — to use another model, choose it
  and start a new chat.  There is no automatic fallback between
  models.
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
node --test \
  tests/test_clipboard.test.js \
  tests/test_network_recovery.test.js \
  tests/test_model_selection.test.js
```

Current suite: **297 Python tests**, **142 frontend tests** (all passing).

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
- Frontend tests exercise the network-error recovery logic
  (`findSentMessages`) with ``node:test`` — no build system required.
- Coverage spans health checks, legacy chat, LLM client behaviour,
  client factory routing, ORM model constraints, chat service
  business logic and transactions, session CRUD, session message
  send, session isolation, concurrency, and lock safety.

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
model.
