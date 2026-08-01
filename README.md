# Static Chatbot

A simple browser-based chatbot with FastAPI backend and SQLite storage.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your API key
```

## Run

```bash
uvicorn backend.main:app --reload
```

## Test

```bash
pytest
```
