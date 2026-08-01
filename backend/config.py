"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv(override=False)

LLM_MODE = os.getenv("LLM_MODE", "fake")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
