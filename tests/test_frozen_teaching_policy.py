"""Frozen experiment contracts stay isolated from production language policy."""

import asyncio
import copy

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.chat_service import ChatService, FrozenReceiveTeachingPolicy
from backend.database import create_database_engine, create_tables
from backend.exceptions import LLMError
from backend.interaction_modes import CORRECTIVE_MODE, CORRECTIVE_PROMPT
from backend.models import Message


class ReplyClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def generate(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return self.reply


@pytest.fixture
def db(tmp_path):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'frozen.sqlite3'}")
    create_tables(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_frozen_contract_preserves_exact_request_snapshot_and_observed_reply(db):
    client = ReplyClient("这是实验中需要记录的错语回复。")
    policy = FrozenReceiveTeachingPolicy("Frozen prompt.\n", "experiment-v1")
    service = ChatService(client, frozen_receive_teaching_policy=policy)
    session = service.create_session("default", db)
    user, assistant = asyncio.run(
        service.handle_session_message(session.id, "Please confirm this fact.", db)
    )
    assert client.calls == [[
        {"role": "system", "content": policy.system_prompt},
        {"role": "user", "content": user.content},
    ]]
    assert user.prompt_version_snapshot == assistant.prompt_version_snapshot == policy.prompt_version
    assert assistant.content == client.reply


def test_default_service_still_blocks_same_wrong_language_before_assistant_write(db):
    client = ReplyClient("这是实验中需要记录的错语回复。")
    service = ChatService(client)
    session = service.create_session("default", db)
    with pytest.raises(LLMError) as caught:
        asyncio.run(service.handle_session_message(session.id, "Please confirm this fact.", db))
    assert caught.value.status_code == 502
    assert caught.value.detail["code"] == "receive_reply_language_mismatch"
    messages = db.scalars(select(Message).where(Message.session_id == session.id)).all()
    assert len(messages) == 1 and messages[0].role == "user"
    assert messages[0].prompt_version_snapshot.endswith("+l1-en")


def test_frozen_receive_policy_does_not_replace_corrective_policy(db):
    client = ReplyClient("Corrective reply.")
    service = ChatService(
        client,
        frozen_receive_teaching_policy=FrozenReceiveTeachingPolicy("Frozen prompt", "experiment-v1"),
    )
    session = service.create_session("default", db)
    session.interaction_mode = CORRECTIVE_MODE
    db.commit()
    user, assistant = asyncio.run(service.handle_session_message(session.id, "Check this claim.", db))
    assert client.calls[0][0]["content"] == CORRECTIVE_PROMPT
    assert user.prompt_version_snapshot == assistant.prompt_version_snapshot == "corrective-v1"


@pytest.mark.parametrize("tool", ["general_history", "programmatic_history"])
def test_frozen_experiment_rejects_history_tool_configuration(tool):
    with pytest.raises(ValueError, match="cannot enable history tools"):
        ChatService(
            ReplyClient("reply"),
            frozen_receive_teaching_policy=FrozenReceiveTeachingPolicy("Frozen prompt", "experiment-v1"),
            **{tool: object()},
        )
