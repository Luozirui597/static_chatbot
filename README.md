# Static Chatbot

A simple browser-based chatbot with a FastAPI backend.

Currently uses a **FakeLLMClient** that echoes back the user message —
no real model API is called.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

Response:

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
  llm_client.py    LLMClient protocol + FakeLLMClient
  config.py        Environment configuration
frontend/
  index.html       Chat page
  style.css        Styles
  app.js           Frontend logic (vanilla JS)
tests/
  test_health.py   Health endpoint tests
  test_chat.py     Chat endpoint tests
```
