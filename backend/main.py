"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(title="Static Chatbot")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Static frontend files will be mounted here in a later step.
