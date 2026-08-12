"""LLM profile registry — validated profile definitions and session resolution.

The registry is the single source of truth for which LLM profiles are
available.  It is built at application startup and never calls
``generate()`` or connects to the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from backend.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Session profile status
# ---------------------------------------------------------------------------


class SessionProfileStatus(str, Enum):
    """Compatibility status between a session's snapshot and the registry."""

    READY = "ready"
    """The session's profile exists and its model matches the snapshot."""

    PROFILE_UNAVAILABLE = "profile_unavailable"
    """The profile recorded in the session no longer exists in the registry."""

    MODEL_CHANGED = "model_changed"
    """The profile exists but its model has changed since the session was
    created."""

    LEGACY_UNKNOWN = "legacy_unknown"
    """The session was created before model snapshots were tracked — the
    original model is unknown."""


# ---------------------------------------------------------------------------
# LLMProfile
# ---------------------------------------------------------------------------

_VALID_KINDS = frozenset({"fake", "api", "local"})
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")


@dataclass(frozen=True)
class LLMProfile:
    """A validated, immutable LLM profile definition.

    Parameters
    ----------
    id:
        Unique identifier.  Must match ``^[a-z][a-z0-9-]*$``, max 50 chars.
        Must not contain whitespace (rejected, not stripped).
    label:
        Human-readable label, max 100 chars after stripping.  Stored
        normalised (stripped).
    kind:
        ``"fake"``, ``"api"``, or ``"local"``.
    model:
        Model name sent to the API, max 255 chars after stripping.  Stored
        normalised (stripped).
    client:
        The LLM client that will be used for this profile.
    is_default:
        Exactly one profile in a registry must have this set to ``True``.
    """

    id: str
    label: str
    kind: str
    model: str
    client: LLMClient
    is_default: bool = False

    def __post_init__(self) -> None:
        # -- id: reject whitespace, don't strip ---------------------------
        if self.id != self.id.strip():
            raise ValueError(
                f"Profile id must not contain leading/trailing whitespace, "
                f"got {self.id!r}"
            )
        if not _ID_PATTERN.fullmatch(self.id):
            raise ValueError(
                f"Profile id must match ^[a-z][a-z0-9-]*$, got {self.id!r}"
            )
        if len(self.id) > 50:
            raise ValueError(
                f"Profile id must be at most 50 characters, "
                f"got {len(self.id)}"
            )

        # -- label: store stripped value ----------------------------------
        stripped_label = self.label.strip()
        if not stripped_label:
            raise ValueError("Profile label must not be blank")
        if len(stripped_label) > 100:
            raise ValueError(
                f"Profile label must be at most 100 characters, "
                f"got {len(stripped_label)}"
            )
        if stripped_label != self.label:
            object.__setattr__(self, "label", stripped_label)

        # -- kind ---------------------------------------------------------
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"Profile kind must be one of {sorted(_VALID_KINDS)}, "
                f"got {self.kind!r}"
            )

        # -- model: store stripped value ----------------------------------
        stripped_model = self.model.strip()
        if not stripped_model:
            raise ValueError("Profile model must not be blank")
        if len(stripped_model) > 255:
            raise ValueError(
                f"Profile model must be at most 255 characters, "
                f"got {len(stripped_model)}"
            )
        if stripped_model != self.model:
            object.__setattr__(self, "model", stripped_model)

        # -- fake profiles must have model="fake" -------------------------
        if self.kind == "fake" and self.model != "fake":
            raise ValueError(
                f"Fake profile model must be 'fake', got {self.model!r}"
            )


# ---------------------------------------------------------------------------
# SessionProfileResolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionProfileResolution:
    """The result of resolving a session's profile against the registry."""

    status: SessionProfileStatus
    profile: LLMProfile | None = None


# ---------------------------------------------------------------------------
# LLMProfileRegistry
# ---------------------------------------------------------------------------


class LLMProfileRegistry:
    """An immutable registry of :class:`LLMProfile` instances.

    Construction validates internal consistency only:

    * Exactly one default profile.
    * All profile ids are unique.
    * The default profile has id ``"default"``.
    * The local profile (if present) has id ``"local"``.

    The registry does **not** read ``LOCAL_LLM_ENABLED`` or any other
    global config — the caller decides which profiles to include.

    Construction does **not** call ``generate()`` or connect to the
    network.
    """

    def __init__(self, profiles: list[LLMProfile]) -> None:
        if not profiles:
            raise ValueError("At least one profile is required")

        _validate_registry_consistency(profiles)

        self._profiles: dict[str, LLMProfile] = {p.id: p for p in profiles}
        self._default: LLMProfile = next(p for p in profiles if p.is_default)

    @property
    def default(self) -> LLMProfile:
        """The default profile."""
        return self._default

    def get(self, profile_id: str) -> LLMProfile | None:
        """Look up a profile by id, returning ``None`` when not found."""
        return self._profiles.get(profile_id)

    def list_all(self) -> list[LLMProfile]:
        """Return every profile in the registry."""
        return list(self._profiles.values())

    def resolve(self, session_profile_id: str,
                session_model_snapshot: str | None) -> SessionProfileResolution:
        """Determine the compatibility status for a session's stored profile.

        Parameters
        ----------
        session_profile_id:
            The ``llm_profile_id`` stored on the session.
        session_model_snapshot:
            The ``llm_model_snapshot`` stored on the session (may be
            ``None`` for legacy sessions).

        Returns
        -------
        SessionProfileResolution
            The status and, when *ready*, the matching profile.
        """
        profile = self._profiles.get(session_profile_id)
        if profile is None:
            return SessionProfileResolution(
                status=SessionProfileStatus.PROFILE_UNAVAILABLE,
            )

        if session_model_snapshot is None:
            return SessionProfileResolution(
                status=SessionProfileStatus.LEGACY_UNKNOWN,
            )

        if profile.model != session_model_snapshot:
            return SessionProfileResolution(
                status=SessionProfileStatus.MODEL_CHANGED,
            )

        return SessionProfileResolution(
            status=SessionProfileStatus.READY,
            profile=profile,
        )

    # -- factory methods ----------------------------------------------------

    @classmethod
    def from_single_client(cls, client: LLMClient) -> LLMProfileRegistry:
        """Build a registry with a single default profile.

        The profile model is always ``"injected-test-model"`` — this
        factory is intended for tests that inject a spy/fake client.
        """
        profile = LLMProfile(
            id="default",
            label="Default",
            kind="api",
            model="injected-test-model",
            client=client,
            is_default=True,
        )
        return cls([profile])


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_registry_consistency(profiles: list[LLMProfile]) -> None:
    """Validate a list of profiles for registry construction.

    Checks structural invariants (exactly one default, unique ids, etc.)
    but does **not** read global config or environment variables.
    Individual profile fields are already validated by
    :class:`LLMProfile.__post_init__`.
    """
    defaults = [p for p in profiles if p.is_default]
    if len(defaults) != 1:
        raise ValueError(
            f"Exactly one default profile is required; found {len(defaults)}"
        )

    if defaults[0].id != "default":
        raise ValueError(
            f"The default profile must have id='default', "
            f"got {defaults[0].id!r}"
        )

    seen_ids: set[str] = set()
    for p in profiles:
        if p.id in seen_ids:
            raise ValueError(f"Duplicate profile id: {p.id!r}")
        seen_ids.add(p.id)

        if p.kind == "local" and p.id != "local":
            raise ValueError(
                f"Local profile must have id='local', got {p.id!r}"
            )
