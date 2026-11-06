"""ASGI wrapper for the isolated Option A+B web test instance.

Only this process serves this wrapper.  The real FastAPI application remains
the backend for every chat/read request; the wrapper adds A+B test controls,
read-only routing, single-use turn gating, and evidence capture.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Callable

from starlette.responses import JSONResponse, Response

from experiments.option_ab_web.recording import (
    RecordingError,
    RequestContext,
    RunRecorder,
    reset_request_context,
    set_request_context,
)
from experiments.option_ab_web.state import (
    GateError,
    BatchState,
    BranchSelection,
)


_SESSION_PREFIX = "/api/sessions/"
_SESSION_ID_RE = re.compile(r"^/api/sessions/([1-9][0-9]*)$")
_MESSAGES_RE = re.compile(r"^/api/sessions/([1-9][0-9]*)/messages$")
_MODE_EVENTS_RE = re.compile(r"^/api/sessions/([1-9][0-9]*)/mode-switch-events$")
_HISTORY_RE = re.compile(r"^/api/sessions/([1-9][0-9]*)/history-reviews$")
_HISTORY_DETAIL_RE = re.compile(
    r"^/api/sessions/([1-9][0-9]*)/history-reviews/([1-9][0-9]*)$"
)


class TestApp:
    def __init__(
        self,
        app: Any,
        *,
        test_html: str,
        state: BatchState,
        recorder: RunRecorder,
        snapshot_verifier: Callable[[int, list[int], str, str], dict[str, Any]],
        session_mode_getter: Callable[[int], str | None],
    ) -> None:
        self.app = app
        self.test_html = test_html
        self.state = state
        self.recorder = recorder
        self.instance_id = state.instance_id
        self.snapshot_verifier = snapshot_verifier
        self.session_mode_getter = session_mode_getter

    # ------------------------------------------------------------------
    # ASGI plumbing
    # ------------------------------------------------------------------
    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path") or "/")
        normalized = path.rstrip("/") or "/"

        if method == "GET" and normalized == "/":
            response = Response(
                self.test_html,
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return

        if method == "GET" and normalized == "/__option_ab__/state":
            await self._send_json(send, 200, self.state.state_snapshot())
            return
        if method == "POST" and normalized == "/__option_ab__/arm":
            await self._handle_arm(scope, receive, send)
            return
        if method == "POST" and normalized == "/__option_ab__/branch":
            await self._handle_branch(scope, receive, send)
            return
        if method == "POST" and normalized == "/__option_ab__/snapshot":
            await self._handle_snapshot(scope, receive, send)
            return

        if method == "GET" and normalized.startswith("/static/"):
            await self.app(scope, receive, send)
            return

        if method == "GET" and self._allowed_read(normalized):
            await self.app(scope, receive, send)
            return

        if method == "POST" and normalized == "/api/sessions":
            await self._handle_session_create(scope, receive, send)
            return

        if method == "POST" and _MESSAGES_RE.fullmatch(normalized):
            await self._handle_message(scope, receive, send, normalized)
            return

        # Every non-allowed route is an A+B test boundary.  In particular this
        # blocks /api/chat, History Review execution, and all session/profile/
        # mode mutations.
        await self._send_json(
            send,
            403,
            {
                "error": "route_not_allowed_in_option_ab_web",
                "path": normalized,
                "method": method,
            },
        )

    # ------------------------------------------------------------------
    # Read-only API boundary
    # ------------------------------------------------------------------
    def _allowed_read(self, path: str) -> bool:
        if path in {"/api/health", "/api/llm/profiles", "/api/sessions"}:
            return True
        if _SESSION_ID_RE.fullmatch(path):
            return True
        if _MESSAGES_RE.fullmatch(path):
            return True
        if _MODE_EVENTS_RE.fullmatch(path):
            return True
        if _HISTORY_RE.fullmatch(path):
            return True
        if _HISTORY_DETAIL_RE.fullmatch(path):
            return True
        return False

    # ------------------------------------------------------------------
    # Control endpoints
    # ------------------------------------------------------------------
    async def _handle_arm(self, scope, receive, send) -> None:
        try:
            payload = await self._parse_json_object(receive)
            if payload.get("instance_id") != self.instance_id:
                raise GateError("instance mismatch", fatal=True)
            session_id = self._positive_int(payload.get("session_id"), "session_id")
            script_id = self._text(payload.get("script_id"), "script_id")
            turn_id = self._text(payload.get("turn_id"), "turn_id")
            armed = self.state.arm(
                session_id=session_id,
                script_id=script_id,
                turn_id=turn_id,
            )
        except GateError as exc:
            if exc.fatal:
                self.state.fail("control request rejected")
                await self._send_json(send, 400, {"error": "arm_rejected"})
            else:
                await self._send_json(send, 409, {"error": "arm_not_available"})
            return
        except (ValueError, TypeError):
            self.state.fail("malformed control request")
            await self._send_json(send, 400, {"error": "arm_rejected"})
            return
        await self._send_json(
            send,
            200,
            {
                "instance_id": self.instance_id,
                "session_id": armed.session_id,
                "script_id": armed.script_id,
                "turn_id": armed.turn_id,
                "content_sha256": armed.content_sha256,
            },
        )

    async def _handle_branch(self, scope, receive, send) -> None:
        try:
            payload = await self._parse_json_object(receive)
            if payload.get("instance_id") != self.instance_id:
                raise GateError("instance mismatch", fatal=True)
            selection = self.state.select_branch(
                session_id=self._positive_int(payload.get("session_id"), "session_id"),
                script_id=self._text(payload.get("script_id"), "script_id"),
                branch_id=self._text(payload.get("branch_id"), "branch_id"),
                q1_assistant_message_id=self._positive_int(
                    payload.get("q1_assistant_message_id"),
                    "q1_assistant_message_id",
                ),
                selected_by=self._text(
                    payload.get("selected_by") or "human", "selected_by"
                ),
                note=str(payload.get("note") or ""),
            )
            self.recorder.record_branch(
                {
                    "kind": "branch_selection",
                    "instance_id": selection.instance_id,
                    "session_id": selection.session_id,
                    "script_id": selection.script_id,
                    "branch_id": selection.branch_id,
                    "branch_label": selection.branch_label,
                    "q1_assistant_message_id": selection.q1_assistant_message_id,
                    "selected_at": selection.selected_at,
                    "selected_by": "human",
                    "t2_content": selection.t2_content,
                    "t2_content_sha256": selection.t2_content_sha256,
                    "t2_armed": not selection.branch_id == "none",
                }
            )
        except GateError as exc:
            if exc.fatal:
                self.state.fail("branch selection rejected")
                await self._send_json(send, 400, {"error": "branch_rejected"})
            else:
                await self._send_json(send, 409, {"error": "branch_not_available"})
            return
        except (ValueError, TypeError):
            self.state.fail("malformed branch control request")
            await self._send_json(send, 400, {"error": "branch_rejected"})
            return
        except RecordingError:
            self.state.fail("branch evidence write failed; batch stopped")
            await self._send_json(send, 500, {"error": "evidence_write_failed"})
            return
        except Exception:
            self.state.fail("branch control failed; batch stopped")
            await self._send_json(send, 500, {"error": "branch_failed"})
            return
        await self._send_json(send, 200, self._branch_response(selection))

    def _branch_response(self, selection: BranchSelection) -> dict[str, Any]:
        return {
            "instance_id": selection.instance_id,
            "session_id": selection.session_id,
            "script_id": selection.script_id,
            "branch_id": selection.branch_id,
            "branch_label": selection.branch_label,
            "selected_at": selection.selected_at,
            "t2_content": selection.t2_content,
            "t2_content_sha256": selection.t2_content_sha256,
            "t2_armed": selection.branch_id != "none",
            "t2_is_final": True,
        }

    async def _handle_snapshot(self, scope, receive, send) -> None:
        try:
            payload = await self._parse_json_object(receive)
            if payload.get("instance_id") != self.instance_id:
                raise GateError("instance mismatch", fatal=True)
            script_id = self._text(payload.get("script_id"), "script_id")
            turn_id = self._text(payload.get("turn_id"), "turn_id")
            completion = self.state.completed_turn(script_id, turn_id)
            if completion is None:
                raise GateError("turn is not completed", fatal=False)
            verified = self.snapshot_verifier(
                completion.session_id,
                [completion.user_message_id, completion.assistant_message_id],
                script_id,
                turn_id,
            )
            if verified.get("errors"):
                self.state.fail("snapshot verification failed; batch stopped")
            self.recorder.record_snapshot(
                {
                    "kind": "snapshot",
                    "instance_id": self.instance_id,
                    "script_id": script_id,
                    "turn_id": turn_id,
                    "session_id": completion.session_id,
                    "user_message_id": completion.user_message_id,
                    "assistant_message_id": completion.assistant_message_id,
                    "result": verified,
                }
            )
        except GateError as exc:
            if exc.fatal:
                self.state.fail("snapshot control rejected")
            await self._send_json(
                send,
                400 if exc.fatal else 409,
                {"error": "snapshot_rejected"},
            )
            return
        except RecordingError:
            self.state.fail("snapshot evidence write failed; batch stopped")
            await self._send_json(send, 500, {"error": "evidence_write_failed"})
            return
        except Exception:
            self.state.fail("snapshot verification failed; batch stopped")
            await self._send_json(send, 500, {"error": "snapshot_failed"})
            return
        status = 200 if not verified.get("errors") else 409
        await self._send_json(send, status, verified)

    # ------------------------------------------------------------------
    # Session creation and messages
    # ------------------------------------------------------------------
    async def _handle_session_create(self, scope, receive, send) -> None:
        if not self.state.reserve_session_creation():
            await self._send_json(send, 409, {"error": "session_limit_reached"})
            return
        status, headers, body = await self._capture_app(scope, receive)
        if status is None or status >= 400:
            self.state.release_session_creation()
        await self._forward_captured(send, status, headers, body)

    async def _handle_message(self, scope, receive, send, path: str) -> None:
        match = _MESSAGES_RE.fullmatch(path)
        if match is None:
            await self._send_json(send, 403, {"error": "message_route_not_allowed"})
            return
        session_id = int(match.group(1))
        try:
            payload, raw_body = await self._read_json_object(receive)
            message = self._text(payload.get("message"), "message")
            if self.session_mode_getter(session_id) != "receive_teaching":
                raise GateError("session is not receive_teaching", fatal=True)
            armed = self.state.consume(session_id=session_id, content=message)
        except GateError as exc:
            if exc.fatal:
                self.state.fail("message gate rejected; batch stopped")
            await self._send_json(
                send,
                400 if exc.fatal else 409,
                {"error": "message_rejected"},
            )
            return
        except (ValueError, TypeError):
            self.state.fail("malformed message request")
            await self._send_json(send, 400, {"error": "message_rejected"})
            return

        request_id = uuid.uuid4().hex
        context = RequestContext(
            request_id=request_id,
            session_id=session_id,
            armed=armed,
            state=self.state,
        )
        token = set_request_context(context)
        status_value: int | None = None
        headers: list[tuple[bytes, bytes]] = []
        body = b""
        delivery_status = "not_delivered"
        try:
            # Phase 1: generate/save/verify.  No HTTP response bytes are sent.
            try:
                status_value, headers, body = await self._capture_app(
                    scope, self._receive_from_body(raw_body)
                )
                self._check_message_response(armed, status_value, body)
                verified = self.snapshot_verifier(
                    session_id,
                    list(self._response_ids(body)),
                    armed.script_id,
                    armed.turn_id,
                )
                self.recorder.record_snapshot(
                    {
                        "kind": "snapshot",
                        "instance_id": self.instance_id,
                        "script_id": armed.script_id,
                        "turn_id": armed.turn_id,
                        "session_id": session_id,
                        "user_message_id": self._response_ids(body)[0],
                        "assistant_message_id": self._response_ids(body)[1],
                        "result": verified,
                    }
                )
                if verified.get("errors"):
                    raise GateError("snapshot verification failed", fatal=True)
            except asyncio.CancelledError:
                self.state.cancel()
                self._safe_record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status="cancelled",
                    body=body,
                    context=context,
                    error="cancelled",
                    delivery_status="not_delivered",
                )
                raise
            except GateError:
                self.state.fail("message post-check failed; batch stopped")
                self._safe_record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status=status_value if status_value is not None else "failed",
                    body=body,
                    context=context,
                    error="verification_failed",
                    delivery_status="not_delivered",
                )
                await self._send_json(send, 502, {"error": "message_verification_failed"})
                return
            except RecordingError:
                self.state.fail("message evidence write failed; batch stopped")
                await self._send_json(send, 500, {"error": "evidence_write_failed"})
                return
            except Exception:
                self.state.fail("message request failed; batch stopped")
                self._safe_record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status=status_value if status_value is not None else "error",
                    body=body,
                    context=context,
                    error="unhandled",
                    delivery_status="not_delivered",
                )
                await self._send_json(send, 502, {"error": "message_failed"})
                return

            # Phase 2: deliver the already saved response.  The gate remains
            # in flight until the body has been handed to the ASGI server.
            try:
                delivery_status = "start_pending"
                await send(
                    {
                        "type": "http.response.start",
                        "status": int(status_value),
                        "headers": headers,
                    }
                )
                delivery_status = "start_sent"
                await send({"type": "http.response.body", "body": body})
                delivery_status = "delivered"
            except asyncio.CancelledError:
                self.state.cancel()
                self._safe_record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status="cancelled",
                    body=body,
                    context=context,
                    error="cancelled",
                    delivery_status=delivery_status,
                )
                raise
            except BaseException:
                self.state.fail("response delivery failed; batch stopped")
                self._safe_record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status=status_value,
                    body=body,
                    context=context,
                    error="delivery_failed",
                    delivery_status=delivery_status,
                )
                # Never attempt a second HTTP status after delivery was
                # attempted; preserve the transport exception.
                raise

            # Phase 3: only now advance the gate.  The response has already
            # been delivered, so a failure here must not emit a second status.
            try:
                self.state.complete(
                    armed=armed,
                    user_message_id=self._response_ids(body)[0],
                    assistant_message_id=self._response_ids(body)[1],
                )
                self._record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status=status_value,
                    body=body,
                    context=context,
                    error=None,
                    delivery_status="delivered",
                )
            except asyncio.CancelledError:
                self.state.cancel()
                self._safe_record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status="cancelled",
                    body=body,
                    context=context,
                    error="cancelled",
                    delivery_status="delivered",
                )
                raise
            except Exception:
                self.state.fail("post-delivery completion failed; batch stopped")
                self._safe_record_message_http(
                    request_id=request_id,
                    armed=armed,
                    status=status_value,
                    body=body,
                    context=context,
                    error="post_delivery_failure",
                    delivery_status="delivered",
                )
                raise
        finally:
            reset_request_context(token)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_record_message_http(self, **kwargs: Any) -> None:
        """Record an HTTP attempt, preserving an earlier transport exception."""
        try:
            self._record_message_http(**kwargs)
        except Exception:
            # The caller is already failing/cancelling the batch; a second
            # evidence failure must never replace the original exception.
            pass

    def _record_message_http(
        self,
        *,
        request_id: str,
        armed: Any,
        status: int | str,
        body: bytes,
        context: Any,
        error: str | None,
        delivery_status: str,
    ) -> None:
        user_id, assistant_id = self._response_ids(body)
        model_calls = len(context.calls)
        model_status = context.model_status()
        database_status = "saved" if user_id and assistant_id else "unknown"
        if error is None and delivery_status == "delivered":
            outcome = "completed"
        elif error == "cancelled":
            outcome = "cancelled"
        else:
            outcome = "failed"
        self.recorder.record_http(
            {
                "kind": "message_http",
                "request_id": request_id,
                "instance_id": self.instance_id,
                "session_id": armed.session_id,
                "script_id": armed.script_id,
                "turn_id": armed.turn_id,
                "status": status,
                "request_message_sha256": armed.content_sha256,
                "user_message_id": user_id or None,
                "assistant_message_id": assistant_id or None,
                "model_calls": model_calls,
                "model_status": model_status,
                "database_status": database_status,
                "delivery_status": delivery_status,
                "calls": list(context.calls),
                "outcome": outcome,
                "error": error,
            }
        )

    async def _capture_app(self, scope, receive) -> tuple[int | None, list[tuple[bytes, bytes]], bytes]:
        messages: list[dict[str, Any]] = []

        async def capture(message):
            messages.append(message)

        await self.app(scope, receive, capture)
        status = next(
            (
                int(item["status"])
                for item in messages
                if item.get("type") == "http.response.start"
            ),
            None,
        )
        headers: list[tuple[bytes, bytes]] = []
        for item in messages:
            if item.get("type") == "http.response.start":
                headers = list(item.get("headers") or [])
                break
        body = b"".join(
            item.get("body", b"")
            for item in messages
            if item.get("type") == "http.response.body"
        )
        return status, headers, body

    async def _forward_captured(
        self,
        send,
        status: int | None,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
    ) -> None:
        if status is None:
            await self._send_json(send, 502, {"error": "upstream_no_response"})
            return
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _check_message_response(
        self, armed, status: int | None, body: bytes
    ) -> None:
        if status != 200:
            raise GateError("message endpoint did not return 200", fatal=True)
        user_id, assistant_id = self._response_ids(body)
        if not user_id or not assistant_id:
            raise GateError("message ids missing", fatal=True)
        context = None
        try:
            from experiments.option_ab_web.recording import current_request_context

            context = current_request_context()
        except RecordingError:
            context = None
        if context is None or len(context.calls) != 1:
            raise GateError("expected exactly one model call", fatal=True)

    def _response_ids(self, body: bytes) -> tuple[int, int]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 0, 0
        if not isinstance(payload, dict):
            return 0, 0
        user = payload.get("user_message")
        assistant = payload.get("assistant_message")
        if not isinstance(user, dict) or not isinstance(assistant, dict):
            return 0, 0
        user_id = user.get("id")
        assistant_id = assistant.get("id")
        if not isinstance(user_id, int) or not isinstance(assistant_id, int):
            return 0, 0
        if user_id <= 0 or assistant_id <= 0 or user_id == assistant_id:
            return 0, 0
        return user_id, assistant_id

    async def _read_json_object(self, receive) -> tuple[dict[str, Any], bytes]:
        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                break
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        raw = bytes(body)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("request body must be JSON")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload, raw

    async def _parse_json_object(self, receive) -> dict[str, Any]:
        payload, _ = await self._read_json_object(receive)
        return payload

    @staticmethod
    def _receive_from_body(body: bytes):
        state = {"sent": False}

        async def receive():
            if state["sent"]:
                return {"type": "http.disconnect"}
            state["sent"] = True
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        return receive

    @staticmethod
    def _positive_int(value: Any, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        return value

    @staticmethod
    def _text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be text")
        return value

    @staticmethod
    async def _send_json(send, status: int, payload: dict[str, Any]) -> None:
        response = JSONResponse(payload, status_code=status)
        await response({"type": "http"}, _empty_receive, send)


async def _empty_receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}

