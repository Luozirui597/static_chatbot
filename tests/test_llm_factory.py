"""Tests for create_llm_client factory function.

All tests pass explicit parameters — they never read ``backend.config``
or the local environment.
"""

import pytest

from backend.llm_client import (
    FakeLLMClient,
    OpenAICompatibleLLMClient,
    create_llm_client,
)


# ---------------------------------------------------------------------------
# Fake mode
# ---------------------------------------------------------------------------


def test_fake_mode_returns_fake_client():
    client = create_llm_client(mode="fake")
    assert isinstance(client, FakeLLMClient)


# ---------------------------------------------------------------------------
# Real mode — full config
# ---------------------------------------------------------------------------


def test_real_mode_full_config_returns_real_client():
    client = create_llm_client(
        mode="real",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        model="gpt-4",
    )
    assert isinstance(client, OpenAICompatibleLLMClient)


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown LLM_MODE"):
        create_llm_client(mode="invalid")


# ---------------------------------------------------------------------------
# Real mode — missing / blank config
# ---------------------------------------------------------------------------


def test_real_mode_missing_api_key_raises():
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        create_llm_client(
            mode="real",
            api_key="",
            base_url="https://api.example.com/v1",
            model="gpt-4",
        )


def test_real_mode_blank_api_key_raises():
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        create_llm_client(
            mode="real",
            api_key="   ",
            base_url="https://api.example.com/v1",
            model="gpt-4",
        )


def test_real_mode_missing_base_url_raises():
    with pytest.raises(ValueError, match="LLM_API_BASE_URL"):
        create_llm_client(
            mode="real",
            api_key="sk-test",
            base_url="",
            model="gpt-4",
        )


def test_real_mode_missing_model_raises():
    with pytest.raises(ValueError, match="LLM_MODEL"):
        create_llm_client(
            mode="real",
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="",
        )


def test_real_mode_all_missing_never_falls_back_to_fake():
    """Missing config in real mode raises ValueError, not FakeLLMClient."""
    with pytest.raises(ValueError):
        create_llm_client(
            mode="real",
            api_key="",
            base_url="",
            model="",
        )
