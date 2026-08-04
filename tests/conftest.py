"""Pytest configuration — force fake mode for all tests.

This module is loaded by pytest *before* any test module is imported,
so the environment variables are set before ``backend.config`` or
``backend.main`` execute their module-level code.
"""

import os

# Force fake mode so no test ever touches a real LLM API.
os.environ["LLM_MODE"] = "fake"

# Blank out real-mode variables so that a local .env or shell
# environment cannot leak real credentials into a test.
os.environ["LLM_API_KEY"] = ""
os.environ["LLM_API_BASE_URL"] = ""
os.environ["LLM_MODEL"] = ""

# Prevent module-level engine creation in backend.database from
# touching the real data/chatbot.db during test collection.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
