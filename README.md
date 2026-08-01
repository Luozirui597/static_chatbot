# Static Chatbot

A simple browser-based chatbot with a FastAPI backend.

## LLM mode

The chatbot supports two modes controlled by the `LLM_MODE` environment variable:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODE` | `fake` | `fake` — returns echo replies; `real` — calls an LLM API |
| `LLM_API_KEY` | — | API key (required when `LLM_MODE=real`) |
| `LLM_API_BASE_URL` | — | API base URL, e.g. `https://api.openai.com/v1` |
| `LLM_MODEL` | — | Model name, e.g. `gpt-4o` |

When `LLM_MODE=fake` (the default) the `FakeLLMClient` is used — every
reply echoes back the user message.  No network calls are made.

When `LLM_MODE=real` the `OpenAICompatibleLLMClient` sends requests to:

```
POST {LLM_API_BASE_URL}/chat/completions
Authorization: Bearer <key>
```

and reads the reply from `choices[0].message.content`.  This is the
protocol used by OpenAI, DeepSeek, and many other providers — but the
application does **not** claim compatibility with every service.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to set `LLM_MODE` and, if using real mode, `LLM_API_KEY`,
`LLM_API_BASE_URL`, and `LLM_MODEL`.

## Run

```bash
.venv/bin/python -m uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000** in a browser.

## API

| Method | Path          | Description          |
|--------|---------------|----------------------|
| GET    | `/api/health` | Health check         |
| POST   | `/api/chat`   | Send a chat message  |

### POST /api/chat

Request:

```json
{"message": "你好"}
```

Response (fake mode):

```json
{"reply": "测试回复：你好"}
```

## Test

```bash
.venv/bin/python -m pytest -q
```

## Architecture

```
backend/
  main.py          FastAPI app, routes, static mount
  schemas.py       Pydantic request / response models
  chat_service.py  ChatService — business logic
  llm_client.py    LLMClient protocol, FakeLLMClient,
                   OpenAICompatibleLLMClient, factory
  config.py        Environment configuration
  exceptions.py    LLMError exception
  system_prompt.py Fixed system prompt
frontend/
  index.html       Chat page
  style.css        Styles
  app.js           Frontend logic (vanilla JS)
tests/
  conftest.py      Forces LLM_MODE=fake for all tests
  test_health.py   Health endpoint tests
  test_chat.py     Chat endpoint & route error tests
  test_real_client.py  OpenAICompatibleLLMClient unit tests
  test_llm_factory.py  Factory function tests
```
