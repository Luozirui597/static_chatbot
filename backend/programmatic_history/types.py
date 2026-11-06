"""Structured, experiment-only tool-call types.

These types deliberately do not alter production ``LLMMessage`` or
``ChatService``.  The first isolated tool experiment owns its message
shapes and fake client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One provider-shaped function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass(frozen=True)
class AssistantTurn:
    """An assistant generation that may contain tool calls."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ToolResult:
    """A tool result correlated to exactly one call id."""

    tool_call_id: str
    name: str
    content: dict[str, Any]
    is_error: bool = False


@dataclass(frozen=True)
class ToolContext:
    """Server-bound execution context.  Never accepted from the model."""

    session_id: int
    current_message_id: int


class ExperimentError(RuntimeError):
    """Base class for experiment-only failures."""


class ToolProtocolError(ExperimentError):
    """A structured call or result is not valid."""


class ToolExecutionError(ExperimentError):
    """Tool execution failed safely."""


class ToolTimeout(ExperimentError):
    """Tool execution exceeded its deadline."""
