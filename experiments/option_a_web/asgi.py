"""ASGI wrapper for the isolated Option A web test instance.

The wrapper changes only the test page and the admission gate.  Every chat
request still traverses the real FastAPI app and the real ChatService.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from typing import Any, Callable

from starlette.responses import Response

from experiments.option_a_web.gate import GateError, TurnGate
from experiments.option_a_web.recording import (
    RequestContext,
    RunRecorder,
    reset_request_context,
    set_request_context,
)

SnapshotVerifier = Callable[[int, list[int], str, str], dict[str, Any]]
SessionModeGetter = Callable[[int], str | None]


class TestApp:
    """ASGI wrapper around the real app."""

    def __init__(
        self,
        app: Any,
        *,
        test_html: str,
        gate: TurnGate,
        recorder: RunRecorder,
        instance_id: str,
        snapshot_verifier: SnapshotVerifier | None = None,
        session_mode_getter: SessionModeGetter | None = None,
    ) -> None:
        self.app = app
        self.test_html = test_html
        self.gate = gate
        self.recorder = recorder
        self.instance_id = instance_id
        self.snapshot_verifier = snapshot_verifier
        self.session_mode_getter = session_mode_getter

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        normalized_path = path.rstrip("/") or "/"
        method = scope.get("method", "").upper()

        if path == "/" and method == "GET":
            await self._send_html(scope, receive, send)
            return

        if path == "/__option_a__/arm" and method == "POST":
            await self._handle_arm(scope, receive, send)
            return

        if path == "/__option_a__/snapshot" and method == "POST":
            await self._handle_snapshot(scope, receive, send)
            return

        if self._is_blocked(path):
            await self._send_json(
                scope,
                receive,
                send,
                403,
                {
                    "error": "endpoint_not_available_in_option_a_test",
                    "path": path,
                },
            )
            return

        message_match = re.fullmatch(r"/api/sessions/(\d+)/messages", normalized_path)
        if message_match is not None and method == "POST":
            await self._handle_message(
                scope, receive, send, int(message_match.group(1))
            )
            return

        await self.app(scope, receive, send)

    async def _send_html(self, scope: dict, receive: Callable, send: Callable) -> None:
        response = Response(
            self.test_html,
            media_type="text/html",
            headers={"cache-control": "no-store"},
        )
        await response(scope, receive, send)

    async def _send_json(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
        status_code: int,
        payload: dict[str, Any],
    ) -> None:
        response = Response(
            json.dumps(payload, ensure_ascii=False),
            status_code=status_code,
            media_type="application/json",
            headers={"cache-control": "no-store"},
        )
        await response(scope, receive, send)

    @staticmethod
    def _is_blocked(path: str) -> bool:
        normalized = path.rstrip("/") or "/"
        if normalized == "/api/chat":
            return True
        if "/history-reviews" in normalized or normalized.endswith("/history-review"):
            return True
        if re.fullmatch(r"/api/sessions/\d+/interaction-mode", normalized):
            return True
        if re.fullmatch(r"/api/sessions/\d+/llm-profile", normalized):
            return True
        return False

    @staticmethod
    async def _read_body(receive: Callable) -> bytes:
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        return b"".join(chunks)

    @staticmethod
    def _replay_receive(body: bytes) -> Callable:
        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        return replay

    @staticmethod
    def _parse_json_object(body: bytes, label: str) -> dict[str, Any]:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{label} body is not valid UTF-8") from None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"{label} body is not JSON") from None
        if not isinstance(payload, dict):
            raise ValueError(f"{label} body must be a JSON object")
        return payload

    @staticmethod
    def _positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def _safe_fail(self, reason: str) -> None:
        self.gate.run_state.fail(reason)

    def _record_http_safely(self, record: dict[str, Any]) -> bool:
        try:
            self.recorder.record_http(record)
            return True
        except Exception:
            self._safe_fail("evidence write failed; batch stopped")
            return False

    def _record_snapshot_safely(self, record: dict[str, Any]) -> bool:
        try:
            self.recorder.record_snapshot(record)
            return True
        except Exception:
            self._safe_fail("evidence write failed; batch stopped")
            return False

    async def _handle_arm(
        self, scope: dict, receive: Callable, send: Callable
    ) -> None:
        body = await self._read_body(receive)
        try:
            payload = self._parse_json_object(body, "arm")
            instance_id = payload["instance_id"]
            session_id = payload["session_id"]
            script_id = payload["script_id"]
            turn_id = payload["turn_id"]
            if not isinstance(instance_id, str):
                raise ValueError("instance_id must be text")
            if not isinstance(script_id, str) or not isinstance(turn_id, str):
                raise ValueError("script_id and turn_id must be text")
            if not self._positive_int(session_id):
                raise ValueError("session_id must be a positive integer")
            armed = self.gate.arm(
                instance_id=instance_id,
                session_id=session_id,
                script_id=script_id,
                turn_id=turn_id,
            )
        except GateError:
            self._safe_fail("arm request rejected")
            await self._send_json(
                scope,
                receive,
                send,
                409,
                {"error": "arm_rejected"},
            )
            return
        except (KeyError, ValueError, TypeError):
            self._safe_fail("malformed arm request")
            await self._send_json(
                scope,
                receive,
                send,
                400,
                {"error": "malformed_arm_request"},
            )
            return

        await self._send_json(
            scope,
            receive,
            send,
            200,
            {
                "instance_id": armed.instance_id,
                "session_id": armed.session_id,
                "script_id": armed.script_id,
                "turn_id": armed.turn_id,
                "content_sha256": armed.content_sha256,
            },
        )

    async def _handle_snapshot(
        self, scope: dict, receive: Callable, send: Callable
    ) -> None:
        body = await self._read_body(receive)
        try:
            payload = self._parse_json_object(body, "snapshot")
            if payload.get("instance_id") != self.instance_id:
                raise ValueError("wrong instance")
            session_id = payload["session_id"]
            script_id = payload["script_id"]
            turn_id = payload["turn_id"]
            if not self._positive_int(session_id):
                raise ValueError("session_id must be a positive integer")
            if not isinstance(script_id, str) or not isinstance(turn_id, str):
                raise ValueError("script_id/turn_id must be text")
            completion = self.gate.completed_turn(script_id, turn_id)
            if completion is None:
                raise ValueError("snapshot turn is not completed")
            if completion.session_id != session_id:
                raise ValueError("snapshot session does not match completion")
            if self.snapshot_verifier is None:
                result: dict[str, Any] = {
                    "errors": ["no snapshot verifier configured"]
                }
            else:
                try:
                    result = self.snapshot_verifier(
                        session_id,
                        [completion.user_message_id, completion.assistant_message_id],
                        script_id,
                        turn_id,
                    )
                except Exception as exc:
                    result = {
                        "errors": [
                            f"snapshot verifier failed: {type(exc).__name__}"
                        ]
                    }
            if not isinstance(result, dict):
                raise ValueError("snapshot verifier returned a non-object")
            if result.get("errors"):
                self._safe_fail("snapshot verification failed")
                await self._send_json(
                    scope,
                    receive,
                    send,
                    409,
                    {"error": "snapshot_mismatch", "result": result},
                )
                return
        except (KeyError, ValueError, TypeError):
            self._safe_fail("malformed or missing snapshot request")
            await self._send_json(
                scope,
                receive,
                send,
                400,
                {"error": "malformed_snapshot_request"},
            )
            return

        if not self._record_snapshot_safely(
            {
                "instance_id": self.instance_id,
                "session_id": session_id,
                "script_id": script_id,
                "turn_id": turn_id,
                "result": result,
            }
        ):
            await self._send_json(
                scope,
                receive,
                send,
                500,
                {"error": "evidence_write_failed"},
            )
            return
        await self._send_json(scope, receive, send, 200, result)

    async def _reject_message(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
        *,
        session_id: int,
        body: bytes,
        status_code: int,
        error: str,
        reason: str,
    ) -> None:
        self._safe_fail(reason)
        self._record_http_safely(
            {
                "request_id": uuid.uuid4().hex,
                "instance_id": self.instance_id,
                "session_id": session_id,
                "script_id": None,
                "turn_id": None,
                "request_content_sha256": hashlib.sha256(body).hexdigest(),
                "status_code": status_code,
                "response_content_type": None,
                "response_body_sha256": None,
                "call_ids": [],
                "user_message_id": None,
                "assistant_message_id": None,
                "snapshot_result": None,
                "inner_error_class": None,
                "failure_reason": reason,
                "response_json_was_object": False,
            }
        )
        await self._send_json(
            scope,
            receive,
            send,
            status_code,
            {"error": error},
        )

    async def _handle_message(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
        session_id: int,
    ) -> None:
        body = await self._read_body(receive)
        try:
            payload = self._parse_json_object(body, "message")
            message = payload["message"]
            if not isinstance(message, str):
                raise ValueError("message must be text")
        except (KeyError, ValueError, TypeError):
            await self._reject_message(
                scope,
                receive,
                send,
                session_id=session_id,
                body=body,
                status_code=400,
                error="malformed_message_request",
                reason="malformed message request",
            )
            return

        if self.session_mode_getter is not None:
            try:
                mode = self.session_mode_getter(session_id)
            except Exception:
                await self._reject_message(
                    scope,
                    receive,
                    send,
                    session_id=session_id,
                    body=body,
                    status_code=503,
                    error="session_mode_check_failed",
                    reason="session mode check failed",
                )
                return
            if mode != "receive_teaching":
                await self._reject_message(
                    scope,
                    receive,
                    send,
                    session_id=session_id,
                    body=body,
                    status_code=403,
                    error="only_receive_teaching_is_allowed",
                    reason="non-receive session rejected",
                )
                return

        try:
            armed = self.gate.consume(session_id=session_id, content=message)
        except GateError:
            await self._reject_message(
                scope,
                receive,
                send,
                session_id=session_id,
                body=body,
                status_code=409,
                error="turn_rejected",
                reason="message request rejected",
            )
            return

        request_id = uuid.uuid4().hex
        context = RequestContext(
            request_id=request_id,
            session_id=session_id,
            arm=armed,
            run_state=self.gate.run_state,
        )
        token = set_request_context(context)
        status_code: int | None = None
        response_headers: dict[str, str] = {}
        response_chunks: list[bytes] = []
        response_started = False
        response_completed = False
        send_failed = False
        inner_error: str | None = None
        inner_error_detail: str | None = None

        async def capture_send(message: dict[str, Any]) -> None:
            nonlocal status_code, response_started, response_completed, send_failed
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                response_started = True
                for name, value in message.get("headers", []):
                    key = name.decode("latin-1").lower()
                    response_headers[key] = value.decode("latin-1")
            elif message.get("type") == "http.response.body":
                response_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    response_completed = True
            try:
                await send(message)
            except BaseException:
                send_failed = True
                raise

        try:
            await self.app(scope, self._replay_receive(body), capture_send)
        except asyncio.CancelledError:
            inner_error = "CancelledError"
            inner_error_detail = "request cancelled"
            self._safe_fail("chat request was cancelled; batch stopped")
            self._record_http_safely(
                {
                    "request_id": request_id,
                    "instance_id": self.instance_id,
                    "session_id": session_id,
                    "script_id": armed.script_id,
                    "turn_id": armed.turn_id,
                    "request_content_sha256": hashlib.sha256(
                        message.encode("utf-8")
                    ).hexdigest(),
                    "status_code": status_code,
                    "response_content_type": response_headers.get("content-type"),
                    "response_body_sha256": None,
                    "call_ids": list(context.calls),
                    "user_message_id": None,
                    "assistant_message_id": None,
                    "snapshot_result": None,
                    "inner_error_class": inner_error,
                    "failure_reason": inner_error_detail,
                    "response_json_was_object": False,
                    "response_started": response_started,
                    "response_completed": response_completed,
                    "send_failed": send_failed,
                }
            )
            reset_request_context(token)
            raise
        except Exception as exc:
            inner_error = type(exc).__name__
            inner_error_detail = "unhandled exception in chat request"
            self._safe_fail("unhandled exception in chat request; batch stopped")
            if not response_started and not send_failed:
                try:
                    await self._send_json(
                        scope,
                        receive,
                        capture_send,
                        502,
                        {"error": "chat_request_failed"},
                    )
                except Exception:
                    send_failed = True
        except BaseException:
            inner_error = "unhandled_base_exception"
            inner_error_detail = "unhandled base exception in chat request"
            self._safe_fail("unhandled exception in chat request; batch stopped")
            raise

        response_body = b"".join(response_chunks)
        response_json: dict[str, Any] | None = None
        user_message_id: int | None = None
        assistant_message_id: int | None = None
        response_parse_error: str | None = None
        if status_code == 200:
            try:
                parsed = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                response_parse_error = "response body was not JSON"
            else:
                if not isinstance(parsed, dict):
                    response_parse_error = "response body was not a JSON object"
                else:
                    response_json = parsed
                    user = parsed.get("user_message")
                    assistant = parsed.get("assistant_message")
                    if isinstance(user, dict) and self._positive_int(user.get("id")):
                        user_message_id = user["id"]
                    if isinstance(assistant, dict) and self._positive_int(
                        assistant.get("id")
                    ):
                        assistant_message_id = assistant["id"]

        snapshot_result: dict[str, Any] | None = None
        snapshot_record_ok = True
        if (
            status_code == 200
            and response_json is not None
            and user_message_id is not None
            and assistant_message_id is not None
        ):
            if self.snapshot_verifier is None:
                snapshot_result = {"errors": ["no snapshot verifier configured"]}
            else:
                try:
                    snapshot_result = self.snapshot_verifier(
                        session_id,
                        [user_message_id, assistant_message_id],
                        armed.script_id,
                        armed.turn_id,
                    )
                except Exception as exc:
                    snapshot_result = {
                        "errors": [f"snapshot verifier failed: {type(exc).__name__}"]
                    }
            snapshot_record_ok = self._record_snapshot_safely(
                {
                    "request_id": request_id,
                    "instance_id": self.instance_id,
                    "session_id": session_id,
                    "script_id": armed.script_id,
                    "turn_id": armed.turn_id,
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                    "result": snapshot_result,
                }
            )

        failure_reason: str | None = None
        if inner_error is not None:
            failure_reason = inner_error_detail or "unhandled exception in chat request"
        elif send_failed:
            failure_reason = "failed to send chat response"
        elif context.failure is not None:
            failure_reason = context.failure
        elif not snapshot_record_ok:
            failure_reason = "evidence write failed"
        elif status_code != 200:
            failure_reason = f"chat request returned HTTP {status_code}"
        elif response_parse_error is not None:
            failure_reason = response_parse_error
        elif len(context.calls) != 1:
            failure_reason = (
                f"expected exactly one model call, observed {len(context.calls)}"
            )
        elif user_message_id is None or assistant_message_id is None:
            failure_reason = "response did not contain both message ids"
        elif snapshot_result is not None and snapshot_result.get("errors"):
            failure_reason = "message snapshot verification failed"

        if failure_reason is not None:
            self._safe_fail(failure_reason)

        http_record = {
            "request_id": request_id,
            "instance_id": self.instance_id,
            "session_id": session_id,
            "script_id": armed.script_id,
            "turn_id": armed.turn_id,
            "request_content_sha256": hashlib.sha256(
                message.encode("utf-8")
            ).hexdigest(),
            "status_code": status_code,
            "response_content_type": response_headers.get("content-type"),
            "response_body_sha256": (
                hashlib.sha256(response_body).hexdigest() if response_body else None
            ),
            "call_ids": list(context.calls),
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "snapshot_result": snapshot_result,
            "inner_error_class": inner_error,
            "failure_reason": failure_reason,
            "response_json_was_object": response_json is not None,
            "response_started": response_started,
            "response_completed": response_completed,
            "send_failed": send_failed,
        }
        record_ok = self._record_http_safely(http_record)
        if failure_reason is not None or not record_ok:
            reset_request_context(token)
            return

        try:
            self.gate.complete(
                armed=armed,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
            )
        except GateError:
            self._safe_fail("gate completion failed; batch stopped")
            self._record_snapshot_safely(
                {
                    "event": "gate_completion_failed",
                    "request_id": request_id,
                    "instance_id": self.instance_id,
                    "session_id": session_id,
                    "script_id": armed.script_id,
                    "turn_id": armed.turn_id,
                    "reason": "gate rejected completion",
                }
            )
        else:
            self._record_snapshot_safely(
                {
                    "event": "turn_completed",
                    "request_id": request_id,
                    "instance_id": self.instance_id,
                    "session_id": session_id,
                    "script_id": armed.script_id,
                    "turn_id": armed.turn_id,
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                }
            )
        finally:
            reset_request_context(token)
