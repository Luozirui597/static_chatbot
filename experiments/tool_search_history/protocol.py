"""Validation and provider-shaped message construction for the experiment."""

from __future__ import annotations

import json
from typing import Any

from experiments.tool_search_history.types import (
    AssistantTurn,
    ToolCall,
    ToolProtocolError,
    ToolResult,
)

TOOL_NAME = "search_teaching_history"
MAX_QUERY_CHARS = 200
MAX_CALL_ID_CHARS = 128
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Search earlier teacher messages in the current session.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": MAX_QUERY_CHARS},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def parse_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ToolProtocolError("invalid_arguments_json") from exc
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ToolProtocolError("arguments_must_be_object")
    return parsed


def validate_tool_call(call: ToolCall) -> None:
    if not isinstance(call.id, str) or not call.id.strip():
        raise ToolProtocolError("missing_call_id")
    if len(call.id) > MAX_CALL_ID_CHARS:
        raise ToolProtocolError("call_id_too_long")
    if call.name != TOOL_NAME:
        raise ToolProtocolError("unknown_tool")
    if not isinstance(call.arguments, dict):
        raise ToolProtocolError("arguments_must_be_object")
    allowed = {"query", "limit"}
    extra = set(call.arguments) - allowed
    if extra:
        raise ToolProtocolError("unknown_arguments")
    query = call.arguments.get("query")
    if not isinstance(query, str):
        raise ToolProtocolError("query_must_be_string")
    if not query.strip():
        raise ToolProtocolError("query_empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ToolProtocolError("query_too_long")
    limit = call.arguments.get("limit", 3)
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ToolProtocolError("limit_must_be_integer")
    if limit < 1 or limit > 5:
        raise ToolProtocolError("limit_out_of_range")


def parse_assistant_turn(
    *,
    content: str | None,
    raw_calls: Any,
) -> AssistantTurn:
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ToolProtocolError("tool_calls_must_be_list")
    if len(raw_calls) > 1:
        raise ToolProtocolError("multiple_tool_calls")
    calls: list[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            raise ToolProtocolError("tool_call_must_be_object")
        call_id = item.get("id")
        function = item.get("function")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ToolProtocolError("missing_call_id")
        if item.get("type") != "function":
            raise ToolProtocolError("tool_type_must_be_function")
        if not isinstance(function, dict):
            raise ToolProtocolError("function_must_be_object")
        name = function.get("name")
        if not isinstance(name, str):
            raise ToolProtocolError("function_name_missing")
        raw_value = function.get("arguments", {})
        arguments = parse_arguments(raw_value)
        raw_arguments = (
            raw_value
            if isinstance(raw_value, str)
            else json.dumps(arguments, ensure_ascii=False)
        )
        call = ToolCall(
            id=call_id,
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
        )
        validate_tool_call(call)
        calls.append(call)
    if content is not None and not isinstance(content, str):
        raise ToolProtocolError("assistant_content_must_be_text_or_null")
    return AssistantTurn(content=content, tool_calls=tuple(calls))


def assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content,
    }
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": (
                        call.raw_arguments
                        if call.raw_arguments
                        else json.dumps(call.arguments, ensure_ascii=False)
                    ),
                },
            }
            for call in turn.tool_calls
        ]
    return message


def tool_message(result: ToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.name,
        "content": json.dumps(result.content, ensure_ascii=False),
    }


def validate_result_for_call(result: ToolResult, call: ToolCall) -> None:
    if result.tool_call_id != call.id:
        raise ToolProtocolError("tool_call_id_mismatch")
    if result.name != call.name:
        raise ToolProtocolError("tool_name_mismatch")
    if not isinstance(result.content, dict):
        raise ToolProtocolError("tool_content_must_be_object")
