"""Tests for LLM profile definitions, registry, and validation.

Every test uses injected fake/spy clients — no real network requests.
"""

from unittest import mock

import pytest
from sqlalchemy import text as sa_text

from backend.exceptions import (
    SessionProfileConflictError,
    SessionProfileUnavailableError,
)
from backend.llm_profiles import (
    LLMProfile,
    LLMProfileRegistry,
    SessionProfileStatus,
)

# ============================================================================
# Fake / Spy helpers
# ============================================================================


class _FakeClient:
    """Minimal fake LLM client for registry/profile tests."""

    def __init__(self, response: str = "fake reply") -> None:
        self.calls: list = []
        self.response = response

    async def generate(self, messages):
        self.calls.append(messages)
        return self.response


# ============================================================================
# Helpers
# ============================================================================


def _make_profile(**kwargs):
    """Create an LLMProfile with sensible test defaults."""
    defaults = {
        "id": "default", "label": "Default", "kind": "fake", "model": "fake",
        "client": _FakeClient(), "is_default": True,
    }
    defaults.update(kwargs)
    return LLMProfile(**defaults)


def _make_registry(*profiles):
    """Create a registry from profiles, defaulting to a single fake."""
    if not profiles:
        profiles = (_make_profile(),)
    return LLMProfileRegistry(list(profiles))


# ============================================================================
# LLMProfile field validation & normalisation
# ============================================================================


class TestLLMProfileFields:
    """LLMProfile.__post_init__ validates and normalises fields."""

    def test_valid_fake_profile(self):
        p = _make_profile(kind="fake", model="fake")
        assert p.kind == "fake"
        assert p.model == "fake"

    def test_fake_model_must_be_fake(self):
        with pytest.raises(ValueError, match="Fake profile model must be"):
            _make_profile(kind="fake", model="gpt-4")

    def test_blank_label_rejected(self):
        with pytest.raises(ValueError, match="label must not be blank"):
            _make_profile(label="")

    def test_whitespace_label_rejected(self):
        with pytest.raises(ValueError, match="label must not be blank"):
            _make_profile(label="   ")

    def test_label_is_normalised_on_construction(self):
        """Label is stored stripped."""
        p = _make_profile(label="  Cloud  ")
        assert p.label == "Cloud"

    def test_model_is_normalised_on_construction(self):
        """Model is stored stripped."""
        p = _make_profile(kind="api", model="  gpt-4  ")
        assert p.model == "gpt-4"

    def test_id_with_whitespace_rejected(self):
        with pytest.raises(ValueError, match="whitespace"):
            _make_profile(id=" default ")

    def test_id_not_auto_stripped(self):
        """ID with whitespace is rejected, not silently stripped."""
        with pytest.raises(ValueError):
            _make_profile(id=" default")

    def test_label_too_long_rejected(self):
        with pytest.raises(ValueError, match="label must be at most 100"):
            _make_profile(label="a" * 101)

    def test_blank_model_rejected(self):
        with pytest.raises(ValueError, match="model must not be blank"):
            _make_profile(kind="api", model="")

    def test_model_too_long_rejected(self):
        with pytest.raises(ValueError, match="model must be at most 255"):
            _make_profile(kind="api", model="a" * 256)

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            _make_profile(kind="cloud")

    def test_valid_kinds_accepted(self):
        for kind in ("fake", "api", "local"):
            model = "fake" if kind == "fake" else "some-model"
            extra = {}
            if kind == "local":
                extra["id"] = "local"
                extra["is_default"] = False
            p = _make_profile(kind=kind, model=model, **extra)
            assert p.kind == kind

    def test_id_too_long_rejected(self):
        with pytest.raises(ValueError, match="id must be at most 50"):
            _make_profile(id="a" * 51)

    def test_id_pattern_rejected(self):
        bad_ids = ["Default", "hello_world", "/default", ""]
        for bad_id in bad_ids:
            with pytest.raises(ValueError):
                _make_profile(id=bad_id)


# ============================================================================
# Registry construction
# ============================================================================


class TestRegistryConstruction:
    """LLMProfileRegistry validates internal consistency only."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one profile"):
            LLMProfileRegistry([])

    def test_exactly_one_default(self):
        with pytest.raises(ValueError, match="Exactly one default"):
            LLMProfileRegistry([
                _make_profile(id="a", is_default=True),
                _make_profile(id="b", is_default=True, kind="api", model="x"),
            ])

    def test_default_id_must_be_default(self):
        with pytest.raises(ValueError, match="must have id='default'"):
            _make_registry(_make_profile(id="other"))

    def test_duplicate_ids_rejected(self):
        c = _FakeClient()
        with pytest.raises(ValueError, match="Duplicate profile id"):
            LLMProfileRegistry([
                _make_profile(id="default", client=c),
                _make_profile(id="default", is_default=False,
                              kind="api", model="x", client=c),
            ])

    def test_local_must_have_id_local(self):
        with pytest.raises(ValueError, match="must have id='local'"):
            LLMProfileRegistry([
                _make_profile(id="default"),
                _make_profile(id="other", kind="local", model="llama3",
                              is_default=False),
            ])

    def test_registry_does_not_reject_local_when_env_disabled(self):
        """Registry itself does not check LOCAL_LLM_ENABLED.
        The caller decides whether to include the local profile."""
        reg = LLMProfileRegistry([
            _make_profile(id="default"),
            _make_profile(id="local", kind="local", model="llama3",
                          is_default=False),
        ])
        assert reg.get("local") is not None

    def test_registry_can_contain_local_without_global_config(self, monkeypatch):
        """Even with LOCAL_LLM_ENABLED=false, registry accepts local profiles.
        The production builder is responsible for the gating."""
        monkeypatch.setenv("LOCAL_LLM_ENABLED", "false")
        reg = LLMProfileRegistry([
            _make_profile(id="default"),
            _make_profile(id="local", kind="local", model="llama3",
                          is_default=False),
        ])
        assert reg.get("local") is not None


# ============================================================================
# from_single_client
# ============================================================================


class TestFromSingleClient:
    """LLMProfileRegistry.from_single_client() for tests."""

    def test_model_is_injected_test_model(self):
        c = _FakeClient()
        reg = LLMProfileRegistry.from_single_client(c)
        assert reg.default.model == "injected-test-model"

    def test_client_is_injected_client(self):
        c = _FakeClient()
        reg = LLMProfileRegistry.from_single_client(c)
        assert reg.default.client is c

    def test_id_is_default(self):
        reg = LLMProfileRegistry.from_single_client(_FakeClient())
        assert reg.default.id == "default"

    def test_is_default(self):
        reg = LLMProfileRegistry.from_single_client(_FakeClient())
        assert reg.default.is_default is True

    def test_kind_is_api(self):
        reg = LLMProfileRegistry.from_single_client(_FakeClient())
        assert reg.default.kind == "api"


# ============================================================================
# Registry operations
# ============================================================================


class TestRegistryOperations:
    def test_get_existing(self):
        reg = _make_registry()
        assert reg.get("default") is not None

    def test_get_missing(self):
        reg = _make_registry()
        assert reg.get("nonexistent") is None

    def test_list_all(self):
        c = _FakeClient()
        reg = LLMProfileRegistry([
            _make_profile(id="default", client=c),
            _make_profile(id="other", is_default=False,
                          kind="api", model="gpt-4", client=c),
        ])
        all_p = reg.list_all()
        assert len(all_p) == 2
        assert {p.id for p in all_p} == {"default", "other"}


# ============================================================================
# Session resolution
# ============================================================================


class TestSessionResolution:
    def test_ready(self):
        reg = _make_registry()
        r = reg.resolve("default", "fake")
        assert r.status == SessionProfileStatus.READY
        assert r.profile is not None

    def test_profile_unavailable(self):
        reg = _make_registry()
        r = reg.resolve("nonexistent", "fake")
        assert r.status == SessionProfileStatus.PROFILE_UNAVAILABLE

    def test_model_changed(self):
        reg = _make_registry()
        r = reg.resolve("default", "old-model")
        assert r.status == SessionProfileStatus.MODEL_CHANGED

    def test_legacy_unknown(self):
        reg = _make_registry()
        r = reg.resolve("default", None)
        assert r.status == SessionProfileStatus.LEGACY_UNKNOWN


# ============================================================================
# ChatService resolve_session_profile
# ============================================================================


class TestChatServiceResolveProfile:
    def test_ready_returns_full_profile(self):
        from backend.chat_service import ChatService
        from backend.models import ChatSession

        c = _FakeClient()
        svc = ChatService(llm_client=c)
        session = ChatSession(
            llm_profile_id="default",
            llm_model_snapshot="injected-test-model",
        )
        profile = svc.resolve_session_profile(session)
        assert profile.id == "default"
        assert profile.kind == "api"
        assert profile.model == "injected-test-model"
        assert profile.client is c

    def test_profile_unavailable_raises(self):
        from backend.chat_service import ChatService
        from backend.models import ChatSession

        svc = ChatService(profiles=_make_registry())
        session = ChatSession(
            llm_profile_id="nonexistent",
            llm_model_snapshot="injected-test-model",
        )
        with pytest.raises(SessionProfileUnavailableError) as e:
            svc.resolve_session_profile(session)
        assert e.value.profile_id == "nonexistent"

    def test_model_changed_raises_conflict(self):
        from backend.chat_service import ChatService
        from backend.models import ChatSession

        svc = ChatService(profiles=_make_registry())
        session = ChatSession(
            llm_profile_id="default",
            llm_model_snapshot="old-model",
        )
        with pytest.raises(SessionProfileConflictError) as e:
            svc.resolve_session_profile(session)
        assert e.value.status == "model_changed"

    def test_legacy_unknown_raises_conflict(self):
        from backend.chat_service import ChatService
        from backend.models import ChatSession

        svc = ChatService(profiles=_make_registry())
        session = ChatSession(
            llm_profile_id="default",
            llm_model_snapshot=None,
        )
        with pytest.raises(SessionProfileConflictError) as e:
            svc.resolve_session_profile(session)
        assert e.value.status == "legacy_unknown"
        assert "model has changed" not in str(e.value)


# ============================================================================
# ChatService constructor
# ============================================================================


class TestChatServiceConstructor:
    def test_positional_spy_works(self):
        from backend.chat_service import ChatService

        c = _FakeClient()
        svc = ChatService(c)
        # Verify via public behaviour: handle_message uses injected client
        assert svc.list_profiles_public()[0]["model"] == "injected-test-model"

    def test_keyword_llm_client_works(self):
        from backend.chat_service import ChatService

        c = _FakeClient()
        svc = ChatService(llm_client=c)
        assert svc.list_profiles_public()[0]["model"] == "injected-test-model"

    def test_keyword_profiles_works(self):
        from backend.chat_service import ChatService

        reg = _make_registry()
        svc = ChatService(profiles=reg)
        assert len(svc.list_profiles_public()) == 1

    def test_both_raises(self):
        from backend.chat_service import ChatService

        with pytest.raises(ValueError, match="not both"):
            ChatService(llm_client=_FakeClient(), profiles=_make_registry())

    def test_neither_raises(self):
        from backend.chat_service import ChatService

        with pytest.raises(ValueError, match="must be provided"):
            ChatService()


# ============================================================================
# build_session_response
# ============================================================================


class TestBuildSessionResponse:
    def test_includes_all_fields(self):
        from backend.chat_service import ChatService
        from backend.models import ChatSession

        svc = ChatService(llm_client=_FakeClient())
        session = ChatSession(
            id=1, title="Test",
            llm_profile_id="default",
            llm_model_snapshot="injected-test-model",
        )
        resp = svc.build_session_response(session)
        assert resp["id"] == 1
        assert resp["llm_profile_id"] == "default"
        assert resp["llm_profile_label"] == "Default"
        assert resp["llm_profile_status"] == SessionProfileStatus.READY
        assert resp["llm_model_snapshot"] == "injected-test-model"

    def test_legacy_session(self):
        from backend.chat_service import ChatService
        from backend.models import ChatSession

        svc = ChatService(llm_client=_FakeClient())
        session = ChatSession(
            id=2, title="Old",
            llm_profile_id="default",
            llm_model_snapshot=None,
        )
        resp = svc.build_session_response(session)
        assert resp["llm_profile_status"] == SessionProfileStatus.LEGACY_UNKNOWN

    def test_missing_profile_label_fallback(self):
        from backend.chat_service import ChatService
        from backend.models import ChatSession

        svc = ChatService(llm_client=_FakeClient())
        session = ChatSession(
            id=3, title="Ghost",
            llm_profile_id="removed",
            llm_model_snapshot="x",
        )
        resp = svc.build_session_response(session)
        assert resp["llm_profile_label"] == "removed"
        assert resp["llm_profile_status"] == SessionProfileStatus.PROFILE_UNAVAILABLE


# ============================================================================
# list_profiles_public
# ============================================================================


class TestListProfilesPublic:
    def test_returns_list_of_dicts(self):
        from backend.chat_service import ChatService

        svc = ChatService(llm_client=_FakeClient())
        profiles = svc.list_profiles_public()
        assert isinstance(profiles, list)
        assert len(profiles) == 1
        p = profiles[0]
        assert set(p.keys()) == {"id", "label", "kind", "model", "is_default"}
        assert p["id"] == "default"

    def test_no_sensitive_fields(self):
        from backend.chat_service import ChatService

        svc = ChatService(llm_client=_FakeClient())
        profiles = svc.list_profiles_public()
        for p in profiles:
            assert "client" not in p
            assert "api_key" not in p
            assert "base_url" not in p


# ============================================================================
# LOCAL_LLM_ENABLED config parsing
# ============================================================================


class TestLocalLLMEnabled:
    def test_not_set_returns_false(self, monkeypatch):
        from backend.config import _parse_local_llm_enabled

        monkeypatch.delenv("LOCAL_LLM_ENABLED", raising=False)
        assert _parse_local_llm_enabled() is False

    def test_true_variants(self, monkeypatch):
        from backend.config import _parse_local_llm_enabled

        for val in ("true", "True", "TRUE", " true ", " TRUE "):
            monkeypatch.setenv("LOCAL_LLM_ENABLED", val)
            assert _parse_local_llm_enabled() is True, f"failed for {val!r}"

    def test_false_variants(self, monkeypatch):
        from backend.config import _parse_local_llm_enabled

        for val in ("false", "False", "FALSE", " false ", " FALSE "):
            monkeypatch.setenv("LOCAL_LLM_ENABLED", val)
            assert _parse_local_llm_enabled() is False, f"failed for {val!r}"

    def test_empty_raises(self, monkeypatch):
        from backend.config import _parse_local_llm_enabled

        monkeypatch.setenv("LOCAL_LLM_ENABLED", "")
        with pytest.raises(ValueError, match="LOCAL_LLM_ENABLED"):
            _parse_local_llm_enabled()

    def test_one_raises(self, monkeypatch):
        from backend.config import _parse_local_llm_enabled

        monkeypatch.setenv("LOCAL_LLM_ENABLED", "1")
        with pytest.raises(ValueError, match="LOCAL_LLM_ENABLED"):
            _parse_local_llm_enabled()

    def test_zero_raises(self, monkeypatch):
        from backend.config import _parse_local_llm_enabled

        monkeypatch.setenv("LOCAL_LLM_ENABLED", "0")
        with pytest.raises(ValueError, match="LOCAL_LLM_ENABLED"):
            _parse_local_llm_enabled()

    def test_yes_raises(self, monkeypatch):
        from backend.config import _parse_local_llm_enabled

        monkeypatch.setenv("LOCAL_LLM_ENABLED", "yes")
        with pytest.raises(ValueError, match="LOCAL_LLM_ENABLED"):
            _parse_local_llm_enabled()


# ============================================================================
# CreateSessionRequest validation
# ============================================================================


class TestCreateSessionRequestSchema:
    def test_default_value(self):
        from backend.schemas import CreateSessionRequest

        req = CreateSessionRequest()
        assert req.llm_profile_id == "default"

    def test_empty_body_uses_default(self):
        from backend.schemas import CreateSessionRequest

        req = CreateSessionRequest()
        assert req.llm_profile_id == "default"

    def test_valid_id(self):
        from backend.schemas import CreateSessionRequest

        req = CreateSessionRequest(llm_profile_id="local")
        assert req.llm_profile_id == "local"

    def test_null_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError):
            CreateSessionRequest(llm_profile_id=None)

    def test_number_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError):
            CreateSessionRequest(llm_profile_id=123)

    def test_boolean_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError):
            CreateSessionRequest(llm_profile_id=True)

    def test_empty_string_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError):
            CreateSessionRequest(llm_profile_id="")

    def test_whitespace_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError):
            CreateSessionRequest(llm_profile_id="   ")

    def test_slash_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError):
            CreateSessionRequest(llm_profile_id="abc/def")

    def test_trailing_newline_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError) as e:
            CreateSessionRequest(llm_profile_id="default\n")
        assert "llm_profile_id" in str(e.value)

    def test_too_long_rejected(self):
        from backend.schemas import CreateSessionRequest

        with pytest.raises(ValueError):
            CreateSessionRequest(llm_profile_id="a" * 51)

    def test_dashes_accepted(self):
        from backend.schemas import CreateSessionRequest

        req = CreateSessionRequest(llm_profile_id="my-profile")
        assert req.llm_profile_id == "my-profile"


# ============================================================================
# Conflict error detail
# ============================================================================


class TestConflictErrorDetail:
    def test_model_changed_detail(self):
        err = SessionProfileConflictError("model_changed", "old-model")
        detail = str(err)
        assert "model" in detail.lower()
        assert "changed" in detail.lower()

    def test_legacy_unknown_does_not_say_model_changed(self):
        err = SessionProfileConflictError("legacy_unknown", None)
        detail = str(err)
        assert "model has changed" not in detail.lower()


# ============================================================================
# Production builder tests
# ============================================================================


class _RecordingSpyFactory:
    """Records every call to create_llm_client() for inspection."""

    def __init__(self, fake_client=None):
        self.calls: list[dict] = []
        self._fake = fake_client or _FakeClient()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self._fake


class TestProductionBuilder:
    """_build_production_registry constructs profiles correctly."""

    def test_fake_mode_produces_fake_default(self, monkeypatch):
        from backend import config as cfg
        monkeypatch.setattr(cfg, "LLM_MODE", "fake")
        monkeypatch.setattr(cfg, "LOCAL_LLM_ENABLED", False)

        from backend.main import _build_production_registry

        reg = _build_production_registry()
        default = reg.default
        assert default.kind == "fake"
        assert default.model == "fake"
        assert default.id == "default"

    def test_real_mode_produces_api_default(self, monkeypatch):
        from backend import config as cfg
        monkeypatch.setattr(cfg, "LLM_MODE", "real")
        monkeypatch.setattr(cfg, "LLM_API_KEY", "sk-test")
        monkeypatch.setattr(cfg, "LLM_API_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setattr(cfg, "LLM_MODEL", "test-model-v2")
        monkeypatch.setattr(cfg, "LLM_PROFILE_LABEL", "My API")
        monkeypatch.setattr(cfg, "LOCAL_LLM_ENABLED", False)

        from backend.main import _build_production_registry

        reg = _build_production_registry()
        default = reg.default
        assert default.kind == "api"
        assert default.model == "test-model-v2"
        assert default.label == "My API"

    def test_local_disabled_produces_only_default(self, monkeypatch):
        from backend import config as cfg
        monkeypatch.setattr(cfg, "LLM_MODE", "fake")
        monkeypatch.setattr(cfg, "LOCAL_LLM_ENABLED", False)

        from backend.main import _build_production_registry

        reg = _build_production_registry()
        assert len(reg.list_all()) == 1
        assert reg.get("local") is None

    def test_local_enabled_produces_default_and_local(self, monkeypatch):
        from backend import config as cfg
        monkeypatch.setattr(cfg, "LLM_MODE", "fake")
        monkeypatch.setattr(cfg, "LOCAL_LLM_ENABLED", True)
        monkeypatch.setattr(cfg, "LOCAL_LLM_MODEL", "qwen3.5:4b")
        monkeypatch.setattr(cfg, "LOCAL_LLM_API_BASE_URL",
                            "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(cfg, "LOCAL_LLM_REASONING_EFFORT", "none")

        from backend.main import _build_production_registry

        reg = _build_production_registry()
        profiles = reg.list_all()
        assert len(profiles) == 2
        local = reg.get("local")
        assert local is not None
        assert local.kind == "local"
        assert local.model == "qwen3.5:4b"

    def test_local_enabled_without_model_raises(self, monkeypatch):
        from backend import config as cfg
        monkeypatch.setattr(cfg, "LLM_MODE", "fake")
        monkeypatch.setattr(cfg, "LOCAL_LLM_ENABLED", True)
        monkeypatch.setattr(cfg, "LOCAL_LLM_MODEL", "")

        from backend.main import _build_production_registry

        with pytest.raises(ValueError, match="LOCAL_LLM_MODEL"):
            _build_production_registry()

    def test_local_client_receives_correct_params(self, monkeypatch):
        import backend.main as main_module
        from backend import config as cfg
        from backend.llm_client import FakeLLMClient

        monkeypatch.setattr(cfg, "LLM_MODE", "fake")
        monkeypatch.setattr(cfg, "LOCAL_LLM_ENABLED", True)
        monkeypatch.setattr(cfg, "LOCAL_LLM_MODEL", "qwen3.5:4b")
        monkeypatch.setattr(cfg, "LOCAL_LLM_API_BASE_URL",
                            "http://127.0.0.1:11435/v1")
        monkeypatch.setattr(cfg, "LOCAL_LLM_REASONING_EFFORT", "none")
        monkeypatch.setattr(cfg, "LOCAL_LLM_API_KEY", "ollama")

        recorder = _RecordingSpyFactory(FakeLLMClient())

        with mock.patch.object(main_module, "create_llm_client",
                               side_effect=recorder):
            reg = main_module._build_production_registry()

        assert len(recorder.calls) == 2
        local_call = recorder.calls[1]

        assert local_call["mode"] == "real"
        assert local_call["api_key"] == "ollama"
        assert local_call["base_url"] == "http://127.0.0.1:11435/v1"
        assert local_call["model"] == "qwen3.5:4b"
        assert local_call["reasoning_effort"] == "none"

        local = reg.get("local")
        assert local is not None
        assert local.model == "qwen3.5:4b"


# ============================================================================
# Compatibility edge-case tests (service-layer, no API)
# ============================================================================


class TestCompatibilityEdgeCases:
    """profile_unavailable / model_changed / legacy_unknown → raises
    before saving user message or calling any client."""

    @pytest.mark.anyio
    async def test_model_changed_no_message_saved_no_client_call(self):
        """Snapshot differs from current model → raises, no side effects."""
        from sqlalchemy import func, select
        from sqlalchemy.orm import sessionmaker

        from backend.chat_service import ChatService
        from backend.database import create_database_engine, create_tables
        from backend.models import Message

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )
        db = SessionLocal()

        try:
            spy = _FakeClient()
            reg = LLMProfileRegistry([LLMProfile(
                id="default", label="Default", kind="fake", model="fake",
                client=spy, is_default=True,
            )])
            svc = ChatService(profiles=reg)
            session = svc.create_session("default", db)
            session.llm_model_snapshot = "wrong-model"
            db.commit()

            msg_count_before = db.execute(
                select(func.count()).select_from(Message)
            ).scalar()
            title_before = session.title
            updated_before = session.updated_at

            with pytest.raises(SessionProfileConflictError) as e:
                await svc.handle_session_message(
                    session.id, "should not save", db,
                )
            assert e.value.status == "model_changed"

            msg_count_after = db.execute(
                select(func.count()).select_from(Message)
            ).scalar()
            assert msg_count_after == msg_count_before

            db.refresh(session)
            assert session.title == title_before
            assert session.updated_at == updated_before
            assert len(spy.calls) == 0
        finally:
            db.close()
            eng.dispose()

    @pytest.mark.anyio
    async def test_profile_unavailable_no_message_saved_no_client_call(self):
        """Session's profile is not in the registry → raises
        SessionProfileUnavailableError, no side effects."""
        from sqlalchemy import func, select
        from sqlalchemy.orm import sessionmaker

        from backend.chat_service import ChatService
        from backend.database import create_database_engine, create_tables
        from backend.models import ChatSession, Message

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )
        db = SessionLocal()

        try:
            spy = _FakeClient()
            reg = LLMProfileRegistry([LLMProfile(
                id="default", label="Default", kind="fake", model="fake",
                client=spy, is_default=True,
            )])
            svc = ChatService(profiles=reg)

            # Session references a profile that does not exist in the
            # registry.
            session = ChatSession(
                llm_profile_id="nonexistent-profile",
                llm_model_snapshot="some-model",
            )
            db.add(session)
            db.commit()
            db.refresh(session)

            msg_count_before = db.execute(
                select(func.count()).select_from(Message)
            ).scalar()
            title_before = session.title
            updated_before = session.updated_at

            with pytest.raises(SessionProfileUnavailableError) as e:
                await svc.handle_session_message(
                    session.id, "should not save", db,
                )
            assert e.value.profile_id == "nonexistent-profile"

            msg_count_after = db.execute(
                select(func.count()).select_from(Message)
            ).scalar()
            assert msg_count_after == msg_count_before

            db.refresh(session)
            assert session.title == title_before
            assert session.updated_at == updated_before
            assert len(spy.calls) == 0
        finally:
            db.close()
            eng.dispose()

    @pytest.mark.anyio
    async def test_legacy_unknown_no_message_saved(self):
        from sqlalchemy import func, select
        from sqlalchemy.orm import sessionmaker

        from backend.chat_service import ChatService
        from backend.database import create_database_engine, create_tables
        from backend.models import Message

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )
        db = SessionLocal()

        try:
            spy = _FakeClient()
            svc = ChatService(llm_client=spy)

            from backend.models import ChatSession
            session = ChatSession(
                llm_profile_id="default",
                llm_model_snapshot=None,
            )
            db.add(session)
            db.commit()
            db.refresh(session)

            msg_count_before = db.execute(
                select(func.count()).select_from(Message)
            ).scalar()
            title_before = session.title
            updated_before = session.updated_at

            with pytest.raises(SessionProfileConflictError) as e:
                await svc.handle_session_message(
                    session.id, "should not save", db,
                )
            assert e.value.status == "legacy_unknown"

            msg_count_after = db.execute(
                select(func.count()).select_from(Message)
            ).scalar()
            assert msg_count_after == msg_count_before

            db.refresh(session)
            assert session.title == title_before
            assert session.updated_at == updated_before
            assert len(spy.calls) == 0
        finally:
            db.close()
            eng.dispose()

    @pytest.mark.anyio
    async def test_ready_default_session_uses_default_client(self):
        from backend.chat_service import ChatService

        spy = _FakeClient(response="hi")
        svc = ChatService(llm_client=spy)

        from sqlalchemy.orm import sessionmaker

        from backend.database import create_database_engine, create_tables

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )
        db = SessionLocal()

        try:
            session = svc.create_session("default", db)
            await svc.handle_session_message(session.id, "hello", db)
            assert len(spy.calls) == 1
        finally:
            db.close()
            eng.dispose()

    @pytest.mark.anyio
    async def test_ready_local_session_uses_local_client(self):
        from backend.chat_service import ChatService

        default_spy = _FakeClient(response="default reply")
        local_spy = _FakeClient(response="local reply")
        reg = LLMProfileRegistry([
            LLMProfile(id="default", label="D", kind="fake", model="fake",
                       client=default_spy, is_default=True),
            LLMProfile(id="local", label="L", kind="local", model="llama3",
                       client=local_spy, is_default=False),
        ])
        svc = ChatService(profiles=reg)

        from sqlalchemy.orm import sessionmaker

        from backend.database import create_database_engine, create_tables

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )
        db = SessionLocal()

        try:
            session = svc.create_session("local", db)
            _u, a = await svc.handle_session_message(
                session.id, "hello", db,
            )
            assert a.content == "local reply"
            assert len(default_spy.calls) == 0
            assert len(local_spy.calls) == 1
        finally:
            db.close()
            eng.dispose()

    @pytest.mark.anyio
    async def test_two_sessions_different_profiles_isolated(self):
        from sqlalchemy.orm import sessionmaker

        from backend.chat_service import ChatService
        from backend.database import create_database_engine, create_tables

        spy_a = _FakeClient(response="reply-a")
        spy_b = _FakeClient(response="reply-b")
        reg = LLMProfileRegistry([
            LLMProfile(id="default", label="A", kind="fake", model="fake",
                       client=spy_a, is_default=True),
            LLMProfile(id="local", label="B", kind="local", model="llama3",
                       client=spy_b, is_default=False),
        ])
        svc = ChatService(profiles=reg)

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )
        db = SessionLocal()

        try:
            s_a = svc.create_session("default", db)
            s_b = svc.create_session("local", db)

            await svc.handle_session_message(s_a.id, "msg-a", db)
            await svc.handle_session_message(s_b.id, "msg-b", db)

            assert any("msg-a" in str(c) for c in spy_a.calls)
            assert not any("msg-b" in str(c) for c in spy_a.calls)
            assert any("msg-b" in str(c) for c in spy_b.calls)
            assert not any("msg-a" in str(c) for c in spy_b.calls)
        finally:
            db.close()
            eng.dispose()


# ============================================================================
# API integration tests (TestClient)
# ============================================================================


class _SpyForAPI:
    """Spy client that records calls; safe for API-level tests."""

    def __init__(self, response="api reply"):
        self.calls: list = []
        self.response = response

    async def generate(self, messages):
        self.calls.append(messages)
        return self.response


class TestAPIProfiles:
    """Tests that hit the real FastAPI app with overridden chat_service."""

    @pytest.fixture(autouse=True)
    def _override_chat_service(self, monkeypatch):
        """Replace main.chat_service with a test instance."""
        import backend.main as main_module

        saved = main_module.chat_service

        from backend.chat_service import ChatService
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry

        spy_default = _SpyForAPI("default reply")
        spy_local = _SpyForAPI("local reply")

        reg = LLMProfileRegistry([
            LLMProfile(id="default", label="My Default", kind="api",
                       model="test-default-model", client=spy_default,
                       is_default=True),
            LLMProfile(id="local", label="My Local", kind="local",
                       model="test-local-model", client=spy_local,
                       is_default=False),
        ])
        test_svc = ChatService(profiles=reg)
        monkeypatch.setattr(main_module, "chat_service", test_svc)
        yield
        monkeypatch.setattr(main_module, "chat_service", saved)

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from sqlalchemy.orm import sessionmaker

        from backend.database import create_database_engine, create_tables, get_db
        from backend.main import app

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )

        def override_get_db():
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.pop(get_db, None)
            eng.dispose()

    # -- GET /api/llm/profiles -------------------------------------------

    def test_list_profiles_returns_correct_fields(self, client):
        resp = client.get("/api/llm/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        for p in data:
            assert set(p.keys()) == {"id", "label", "kind", "model", "is_default"}
            assert p["kind"] in ("fake", "api", "local")

    def test_list_profiles_has_no_sensitive_data(self, client):
        resp = client.get("/api/llm/profiles")
        body = resp.text
        assert "test-default-model" in body
        # No sentinel secrets
        assert "sk-" not in body
        assert "api_key" not in body
        assert "base_url" not in body

    # -- POST /api/sessions ----------------------------------------------

    def test_create_session_no_body(self, client):
        resp = client.post("/api/sessions")
        assert resp.status_code == 201
        data = resp.json()
        assert data["llm_profile_id"] == "default"
        assert data["llm_profile_status"] == "ready"
        assert data["llm_model_snapshot"] == "test-default-model"

    def test_create_session_empty_object(self, client):
        resp = client.post("/api/sessions", json={})
        assert resp.status_code == 201
        assert resp.json()["llm_profile_id"] == "default"

    def test_create_session_json_null(self, client):
        resp = client.post("/api/sessions", json=None)
        assert resp.status_code == 201
        assert resp.json()["llm_profile_id"] == "default"

    def test_create_session_local_profile(self, client):
        resp = client.post("/api/sessions", json={"llm_profile_id": "local"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["llm_profile_id"] == "local"
        assert data["llm_model_snapshot"] == "test-local-model"
        assert data["llm_profile_label"] == "My Local"

    def test_create_session_unknown_profile_422(self, client):
        resp = client.post("/api/sessions",
                           json={"llm_profile_id": "nonexistent"})
        assert resp.status_code == 422

    def test_create_session_disabled_profile_422(self, client):
        resp = client.post("/api/sessions",
                           json={"llm_profile_id": "disabled"})
        assert resp.status_code == 422

    # -- Session responses include profile fields ------------------------

    def test_list_sessions_includes_profile_fields(self, client):
        client.post("/api/sessions")
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        sessions = resp.json()
        assert len(sessions) == 1
        s = sessions[0]
        for key in ("llm_profile_id", "llm_profile_label",
                     "llm_profile_status", "llm_model_snapshot"):
            assert key in s

    def test_get_session_includes_profile_fields(self, client):
        r = client.post("/api/sessions")
        sid = r.json()["id"]
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        s = resp.json()
        assert s["llm_profile_id"] == "default"
        assert s["llm_profile_status"] == "ready"

    def test_rename_session_includes_profile_fields(self, client):
        r = client.post("/api/sessions")
        sid = r.json()["id"]
        resp = client.patch(f"/api/sessions/{sid}",
                            json={"title": "Renamed"})
        assert resp.status_code == 200
        s = resp.json()
        assert "llm_profile_id" in s
        assert "llm_profile_status" in s


# ============================================================================
# Compatibility via API
# ============================================================================


class TestCompatibilityViaAPI:
    """profile_unavailable / model_changed / legacy_unknown via API."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        import backend.main as main_module

        saved = main_module.chat_service
        from backend.chat_service import ChatService
        from backend.llm_profiles import LLMProfile, LLMProfileRegistry

        self.spy = _SpyForAPI()
        reg = LLMProfileRegistry([
            LLMProfile(id="default", label="D", kind="fake", model="fake",
                       client=self.spy, is_default=True),
        ])
        self.test_svc = ChatService(profiles=reg)
        monkeypatch.setattr(main_module, "chat_service", self.test_svc)
        yield
        monkeypatch.setattr(main_module, "chat_service", saved)

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from sqlalchemy.orm import sessionmaker

        from backend.database import create_database_engine, create_tables, get_db
        from backend.main import app

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )

        def override_get_db():
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()

        self._SessionLocal = SessionLocal
        app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app) as c:
                yield c
        finally:
            app.dependency_overrides.pop(get_db, None)
            eng.dispose()

    def _message_count(self):
        db = self._SessionLocal()
        try:
            from sqlalchemy import func, select

            from backend.models import Message
            return db.execute(
                select(func.count()).select_from(Message)
            ).scalar()
        finally:
            db.close()

    def test_model_changed_409_no_message_saved(self, client):
        r = client.post("/api/sessions")
        sid = r.json()["id"]

        # Tamper with snapshot
        db = self._SessionLocal()
        try:
            from backend.models import ChatSession
            s = db.get(ChatSession, sid)
            s.llm_model_snapshot = "old-model"
            db.commit()
            title_before = s.title
            updated_before = s.updated_at
        finally:
            db.close()

        msg_before = self._message_count()
        calls_before = len(self.spy.calls)

        resp = client.post(f"/api/sessions/{sid}/messages",
                           json={"message": "hello"})
        assert resp.status_code == 409
        assert self._message_count() == msg_before
        assert len(self.spy.calls) == calls_before

        # Title and updated_at unchanged
        db = self._SessionLocal()
        try:
            from backend.models import ChatSession
            s = db.get(ChatSession, sid)
            assert s.title == title_before
            assert s.updated_at == updated_before
        finally:
            db.close()

    def test_legacy_unknown_409_no_message_saved(self, client):
        # Create a legacy session directly
        db = self._SessionLocal()
        try:
            from backend.models import ChatSession
            s = ChatSession(
                llm_profile_id="default",
                llm_model_snapshot=None,
            )
            db.add(s)
            db.commit()
            db.refresh(s)
            sid = s.id
            title_before = s.title
            updated_before = s.updated_at
        finally:
            db.close()

        msg_before = self._message_count()
        calls_before = len(self.spy.calls)

        resp = client.post(f"/api/sessions/{sid}/messages",
                           json={"message": "hello"})
        assert resp.status_code == 409
        assert self._message_count() == msg_before
        assert len(self.spy.calls) == calls_before

        db = self._SessionLocal()
        try:
            from backend.models import ChatSession
            s = db.get(ChatSession, sid)
            assert s.title == title_before
            assert s.updated_at == updated_before
        finally:
            db.close()

    def test_profile_unavailable_503_no_message_saved(self, client):
        # Create session with profile that doesn't exist
        db = self._SessionLocal()
        try:
            from backend.models import ChatSession
            s = ChatSession(
                llm_profile_id="nonexistent",
                llm_model_snapshot="some-model",
            )
            db.add(s)
            db.commit()
            db.refresh(s)
            sid = s.id
        finally:
            db.close()

        msg_before = self._message_count()
        calls_before = len(self.spy.calls)

        resp = client.post(f"/api/sessions/{sid}/messages",
                           json={"message": "hello"})
        assert resp.status_code == 503
        assert self._message_count() == msg_before
        assert len(self.spy.calls) == calls_before


# ============================================================================
# Transaction handling
# ============================================================================


class TestCreateSessionTransaction:
    def test_commit_failure_rollback_no_session_left(self, monkeypatch):
        from sqlalchemy import func, select
        from sqlalchemy.orm import sessionmaker

        from backend.chat_service import ChatService
        from backend.database import create_database_engine, create_tables

        eng = create_database_engine("sqlite:///:memory:")
        create_tables(bind=eng)
        SessionLocal = sessionmaker(
            bind=eng, autoflush=False, expire_on_commit=False,
        )
        db = SessionLocal()

        svc = ChatService(profiles=_make_registry())

        def _fail_commit():
            raise RuntimeError("simulated")

        monkeypatch.setattr(db, "commit", _fail_commit)

        count_before = db.execute(
            select(func.count()).select_from(
                __import__("backend.models", fromlist=["ChatSession"]).ChatSession
            )
        ).scalar()

        with pytest.raises(RuntimeError, match="simulated"):
            svc.create_session("default", db)

        # Rollback must have been called implicitly by the service
        count_after = db.execute(
            select(func.count()).select_from(
                __import__("backend.models", fromlist=["ChatSession"]).ChatSession
            )
        ).scalar()
        assert count_after == count_before

        # Restore commit and retry
        monkeypatch.undo()
        session = svc.create_session("default", db)
        assert session.id is not None

        db.close()
        eng.dispose()


# ============================================================================
# Migration tests
# ============================================================================


def _create_legacy_messages_table(raw):
    """Create the pre-snapshot messages table for legacy DB fixtures.

    This is the real table shape from before the message provenance
    snapshot migration: it must NOT include
    ``llm_profile_id_snapshot``, ``llm_profile_kind_snapshot`` or
    ``llm_model_snapshot`` — otherwise the old-database migration
    tests would no longer exercise the ALTER path.
    """
    raw.execute(
        "CREATE TABLE messages ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  session_id INTEGER NOT NULL "
        "REFERENCES chat_sessions(id) ON DELETE CASCADE,"
        "  role VARCHAR(20) NOT NULL,"
        "  content TEXT NOT NULL,"
        "  created_at DATETIME NOT NULL"
        ")"
    )


class TestLLMProfileMigration:
    """llm_profile_v1 migration behaviour."""

    def test_fresh_db_create_tables_then_migrate(self, tmp_path):
        from backend.database import (
            create_database_engine,
            create_tables,
            run_migrations,
        )

        eng = create_database_engine(f"sqlite:///{tmp_path}/fresh.db")
        try:
            create_tables(bind=eng)
            run_migrations(eng)

            with eng.begin() as conn:
                row = conn.execute(sa_text(
                    "SELECT version FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).fetchone()
                assert row is not None

                cols = [r[1] for r in conn.execute(
                    sa_text("PRAGMA table_info('chat_sessions')")
                ).fetchall()]
                assert "llm_profile_id" in cols
                assert "llm_model_snapshot" in cols
        finally:
            eng.dispose()

    def test_old_db_adds_columns(self, tmp_path):
        import sqlite3

        from backend.database import create_database_engine, run_migrations

        db_path = tmp_path / "old.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        _create_legacy_messages_table(raw)
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) "
            "VALUES ('Old Chat', '2026-01-01', '2026-01-01')"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                cols = [r[1] for r in conn.execute(
                    sa_text("PRAGMA table_info('chat_sessions')")
                ).fetchall()]
                assert "llm_profile_id" in cols
                assert "llm_model_snapshot" in cols

                # Old row: profile_id = 'default', snapshot = NULL
                row = conn.execute(sa_text(
                    "SELECT llm_profile_id, llm_model_snapshot "
                    "FROM chat_sessions WHERE title = 'Old Chat'"
                )).fetchone()
                assert row[0] == "default"
                assert row[1] is None
        finally:
            eng.dispose()

    def test_migration_idempotent(self, tmp_path):
        from backend.database import (
            create_database_engine,
            create_tables,
            run_migrations,
        )

        eng = create_database_engine(f"sqlite:///{tmp_path}/idem.db")
        try:
            create_tables(bind=eng)
            run_migrations(eng)
            run_migrations(eng)  # second run must not raise

            with eng.begin() as conn:
                count = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert count == 1
        finally:
            eng.dispose()

    def test_migration_record_correct(self, tmp_path):
        from backend.database import (
            create_database_engine,
            create_tables,
            run_migrations,
        )

        eng = create_database_engine(f"sqlite:///{tmp_path}/rec.db")
        try:
            create_tables(bind=eng)
            run_migrations(eng)

            with eng.begin() as conn:
                row = conn.execute(sa_text(
                    "SELECT version, applied_at FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).fetchone()
                assert row is not None
                assert row[1] is not None  # applied_at set
        finally:
            eng.dispose()

    def test_columns_exist_dirty_data_backfilled(self, tmp_path):
        """Scenario A: both columns already exist and rows hold
        NULL / empty / whitespace-only profile ids."""
        import sqlite3

        from backend.database import create_database_engine, run_migrations

        db_path = tmp_path / "dirty.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0,"
            "  llm_profile_id VARCHAR(50) DEFAULT 'default',"
            "  llm_model_snapshot VARCHAR(255)"
            ")"
        )
        _create_legacy_messages_table(raw)
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        # title_is_manual_v1 already recorded
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('title_is_manual_v1')"
        )
        # Rows: NULL, empty, spaces, tab, newline, mixed whitespace,
        # and valid ids that must stay untouched.
        rows = [
            (None, "null-id"),
            ("", "empty-id"),
            ("   ", "spaces-id"),
            ("\t", "tab-id"),
            ("\n", "lf-id"),
            ("\r", "cr-id"),
            (" \t\r\n ", "mixed-id"),
            ("default", "default-id"),
            ("local", "local-id"),
            ("removed-profile", "removed-id"),
        ]
        for pid, title in rows:
            raw.execute(
                "INSERT INTO chat_sessions "
                "(title, created_at, updated_at, llm_profile_id, "
                " llm_model_snapshot) "
                "VALUES (?, '2026-01-01', '2026-01-01', ?, NULL)",
                (title, pid),
            )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                # All dirty ids became 'default'
                dirty_titles = (
                    "null-id", "empty-id", "spaces-id", "tab-id",
                    "lf-id", "cr-id", "mixed-id",
                )
                for t in dirty_titles:
                    pid = conn.execute(sa_text(
                        "SELECT llm_profile_id FROM chat_sessions "
                        "WHERE title = :t"
                    ), {"t": t}).fetchone()[0]
                    assert pid == "default", f"{t!r} not backfilled"

                # Valid ids untouched
                for t, expected in (
                    ("default-id", "default"),
                    ("local-id", "local"),
                    ("removed-id", "removed-profile"),
                ):
                    pid = conn.execute(sa_text(
                        "SELECT llm_profile_id FROM chat_sessions "
                        "WHERE title = :t"
                    ), {"t": t}).fetchone()[0]
                    assert pid == expected, f"{t!r} was modified"

                # All snapshots still NULL
                snap = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM chat_sessions "
                    "WHERE llm_model_snapshot IS NOT NULL"
                )).scalar()
                assert snap == 0

                # Exactly one llm_profile_v1 record
                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 1

            # Second run must not change anything
            run_migrations(eng)
            with eng.begin() as conn:
                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 1
        finally:
            eng.dispose()

    def test_columns_exist_backfill_then_record_failure_recovers(
        self, tmp_path,
    ):
        """Scenario B: columns exist; record-write failure rolls back
        the backfill DML and a retry converges.

        Covers every required dirty shape: NULL, empty string, spaces,
        tab, LF, mixed whitespace — plus a valid non-blank id that
        must stay untouched.
        """
        import sqlite3

        from sqlalchemy.exc import IntegrityError

        from backend.database import create_database_engine, run_migrations

        # (title, original llm_profile_id, expected after retry)
        dirty_rows = [
            ("null-row", None),
            ("empty-row", ""),
            ("spaces-row", "   "),
            ("tab-row", "\t"),
            ("lf-row", "\n"),
            ("mixed-row", " \t\r\n "),
            ("valid-row", "removed-profile"),
        ]

        db_path = tmp_path / "fail_b.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0,"
            "  llm_profile_id VARCHAR(50) DEFAULT 'default',"
            "  llm_model_snapshot VARCHAR(255)"
            ")"
        )
        _create_legacy_messages_table(raw)
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('title_is_manual_v1')"
        )
        for title, pid in dirty_rows:
            raw.execute(
                "INSERT INTO chat_sessions "
                "(title, created_at, updated_at, llm_profile_id, "
                " llm_model_snapshot) "
                "VALUES (?, '2026-01-01', '2026-01-01', ?, NULL)",
                (title, pid),
            )
        # Trigger blocks ONLY the llm_profile_v1 record insert.
        raw.execute(
            "CREATE TRIGGER fail_profile_migration "
            "BEFORE INSERT ON schema_migrations "
            "WHEN NEW.version = 'llm_profile_v1' "
            "BEGIN "
            "  SELECT RAISE(FAIL, 'simulated failure'); "
            "END"
        )
        raw.commit()
        raw.close()

        def _fetch_pid(conn, title):
            return conn.execute(sa_text(
                "SELECT llm_profile_id FROM chat_sessions "
                "WHERE title = :t"
            ), {"t": title}).fetchone()[0]

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            # First run fails at the llm_profile_v1 record insert
            with pytest.raises(IntegrityError, match="simulated failure"):
                run_migrations(eng)

            with eng.begin() as conn:
                # No llm_profile_v1 record
                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 0

                # Backfill DML fully rolled back: every dirty value
                # keeps its exact original shape — no partial commit.
                for title, original in dirty_rows:
                    pid = _fetch_pid(conn, title)
                    if original is None:
                        assert pid is None, f"{title}: expected NULL, got {pid!r}"
                    else:
                        assert pid == original, (
                            f"{title}: expected {original!r}, got {pid!r}"
                        )

                # All snapshots still NULL
                snap = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM chat_sessions "
                    "WHERE llm_model_snapshot IS NOT NULL"
                )).scalar()
                assert snap == 0

            # Remove trigger, retry
            raw2 = sqlite3.connect(str(db_path))
            raw2.execute("DROP TRIGGER IF EXISTS fail_profile_migration")
            raw2.commit()
            raw2.close()

            run_migrations(eng)  # must succeed

            with eng.begin() as conn:
                # Every dirty value became 'default'
                for title, original in dirty_rows:
                    pid = _fetch_pid(conn, title)
                    if original == "removed-profile":
                        assert pid == "removed-profile", (
                            f"{title}: valid id was modified to {pid!r}"
                        )
                    else:
                        assert pid == "default", (
                            f"{title}: expected 'default', got {pid!r}"
                        )

                # Snapshots all still NULL
                snap = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM chat_sessions "
                    "WHERE llm_model_snapshot IS NOT NULL"
                )).scalar()
                assert snap == 0

                # Exactly one llm_profile_v1 record
                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 1

            # Idempotent on third run: record count and data unchanged
            run_migrations(eng)
            with eng.begin() as conn:
                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 1

                for title, original in dirty_rows:
                    pid = _fetch_pid(conn, title)
                    expected = (
                        "removed-profile" if original == "removed-profile"
                        else "default"
                    )
                    assert pid == expected, (
                        f"{title}: third run changed value to {pid!r}"
                    )
        finally:
            eng.dispose()

    def test_columns_missing_ddl_may_persist_after_failure(self, tmp_path):
        """Scenario C: columns missing; in this project's SQLite the
        ALTER TABLE results persist even though the migration record
        insert fails.  The retry must recognise the existing columns,
        skip the duplicate ALTER, and converge."""
        import sqlite3

        from sqlalchemy.exc import IntegrityError

        from backend.database import create_database_engine, run_migrations

        db_path = tmp_path / "fail_c.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL,"
            "  title_is_manual INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        _create_legacy_messages_table(raw)
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO schema_migrations (version) "
            "VALUES ('title_is_manual_v1')"
        )
        raw.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) "
            "VALUES ('Old Chat', '2026-01-01', '2026-01-01')"
        )
        raw.execute(
            "CREATE TRIGGER fail_profile_migration "
            "BEFORE INSERT ON schema_migrations "
            "WHEN NEW.version = 'llm_profile_v1' "
            "BEGIN "
            "  SELECT RAISE(FAIL, 'simulated failure'); "
            "END"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            with pytest.raises(IntegrityError, match="simulated failure"):
                run_migrations(eng)

            with eng.begin() as conn:
                # No llm_profile_v1 record
                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 0

                # In this project's SQLite/SQLAlchemy environment the
                # two ALTER TABLE ADD COLUMN statements survive the
                # failed record insert: the columns persist even
                # though the migration record does not.  Assert that
                # explicitly — it proves the retry must recognise the
                # existing columns and skip the duplicate ALTER.
                cols = [r[1] for r in conn.execute(
                    sa_text("PRAGMA table_info('chat_sessions')")
                ).fetchall()]
                assert cols.count("llm_profile_id") == 1
                assert cols.count("llm_model_snapshot") == 1

            # Remove trigger, retry — must not raise duplicate column
            raw2 = sqlite3.connect(str(db_path))
            raw2.execute("DROP TRIGGER IF EXISTS fail_profile_migration")
            raw2.commit()
            raw2.close()

            run_migrations(eng)  # must succeed regardless of DDL state

            with eng.begin() as conn:
                cols = [r[1] for r in conn.execute(
                    sa_text("PRAGMA table_info('chat_sessions')")
                ).fetchall()]
                # Each column exists exactly once
                assert cols.count("llm_profile_id") == 1
                assert cols.count("llm_model_snapshot") == 1

                # Old row: profile_id = 'default', snapshot = NULL
                row = conn.execute(sa_text(
                    "SELECT llm_profile_id, llm_model_snapshot "
                    "FROM chat_sessions WHERE title = 'Old Chat'"
                )).fetchone()
                assert row[0] == "default"
                assert row[1] is None

                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 1

            # Idempotent on third run
            run_migrations(eng)
            with eng.begin() as conn:
                rec = conn.execute(sa_text(
                    "SELECT COUNT(*) FROM schema_migrations "
                    "WHERE version = 'llm_profile_v1'"
                )).scalar()
                assert rec == 1
        finally:
            eng.dispose()

    def test_does_not_break_title_is_manual_v1(self, tmp_path):
        """llm_profile_v1 coexists with title_is_manual_v1."""
        import sqlite3

        from backend.database import create_database_engine, run_migrations

        db_path = tmp_path / "both.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "CREATE TABLE chat_sessions ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  title VARCHAR(255) NOT NULL DEFAULT 'New Chat',"
            "  created_at DATETIME NOT NULL,"
            "  updated_at DATETIME NOT NULL"
            ")"
        )
        _create_legacy_messages_table(raw)
        raw.execute(
            "CREATE TABLE schema_migrations ("
            "  version VARCHAR(255) PRIMARY KEY,"
            "  applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        raw.execute(
            "INSERT INTO chat_sessions (title, created_at, updated_at) "
            "VALUES ('Custom Title', '2026-01-02', '2026-01-02')"
        )
        raw.commit()
        raw.close()

        eng = create_database_engine(f"sqlite:///{db_path}")
        try:
            run_migrations(eng)

            with eng.begin() as conn:
                # Both migrations recorded
                for v in ("title_is_manual_v1", "llm_profile_v1"):
                    row = conn.execute(sa_text(
                        "SELECT version FROM schema_migrations "
                        "WHERE version = :v"
                    ), {"v": v}).fetchone()
                    assert row is not None, f"{v} not recorded"

                # title_is_manual backfill worked
                row = conn.execute(sa_text(
                    "SELECT title_is_manual FROM chat_sessions "
                    "WHERE title = 'Custom Title'"
                )).fetchone()
                assert row[0] == 1
        finally:
            eng.dispose()
