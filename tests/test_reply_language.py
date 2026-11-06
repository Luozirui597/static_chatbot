"""Unit tests for reply-language resolution and validation."""

from __future__ import annotations

import pytest

from backend.llm_client import FakeLLMClient
from backend.reply_language import (
    LANGUAGE_AMBIGUOUS,
    LANGUAGE_CONFLICT,
    LANGUAGE_INSTRUCTIONS,
    LANGUAGE_UNSUPPORTED,
    ReplyLanguageInputError,
    ReplyLanguageMismatchError,
    append_language_instruction,
    language_prompt_version,
    resolve_reply_language,
    validate_reply_language,
)


def test_explicit_turn_languages():
    assert resolve_reply_language("Please reply in Chinese.").target == "zh"
    assert resolve_reply_language("Please reply in English.").target == "en"
    assert resolve_reply_language("请用中文解释。").target == "zh"


def test_persistent_history_is_used_but_quoted_history_is_not():
    assert resolve_reply_language(
        "Please continue.", ("Please use Chinese from now on.",)
    ).target == "zh"
    assert resolve_reply_language(
        "Please continue.", ("“Please use Chinese from now on.”",)
    ).target == "en"


def test_mixed_conflict_and_unsupported_are_rejected_before_generation():
    with pytest.raises(ReplyLanguageInputError) as exc:
        resolve_reply_language("Please continue. 请继续。")
    assert exc.value.code == LANGUAGE_AMBIGUOUS

    with pytest.raises(ReplyLanguageInputError) as exc:
        resolve_reply_language("Please answer in English or Chinese.")
    assert exc.value.code == LANGUAGE_CONFLICT

    with pytest.raises(ReplyLanguageInputError) as exc:
        resolve_reply_language("Please answer in French.")
    assert exc.value.code == LANGUAGE_UNSUPPORTED


def test_prompt_instruction_is_appended_once_and_versioned():
    prompt = append_language_instruction("base", "en")
    assert prompt.endswith(LANGUAGE_INSTRUCTIONS["en"] + "\n")
    assert prompt.count("Reply in English.") == 1
    assert language_prompt_version("receive-teaching-v7+history-context-b1", "zh") == (
        "receive-teaching-v7+history-context-b1+l1-zh"
    )


def test_english_replaces_chinese_prose_but_allows_evidence_quote():
    validate_reply_language(
        "A signed token. [message 3]",
        target="en",
        current_text="What is the protocol?",
    )
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "这是中文回答。",
            target="en",
            current_text="What is the protocol?",
        )
    validate_reply_language(
        "“以后都用中文。”",
        target="en",
        current_text="Please continue.",
        evidence_texts=("以后都用中文。",),
    )
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "“这是编造的引文。”",
            target="en",
            current_text="Please continue.",
            evidence_texts=("以后都用中文。",),
        )


def test_formula_and_code_boundaries():
    # A direct formula-only answer to a formula request is allowed.
    validate_reply_language(
        "$x=1$",
        target="en",
        current_text="公式是什么？",
    )
    # Asking for an English explanation cannot be satisfied by a bare formula.
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "$x=1$",
            target="en",
            current_text="Please only use English to explain this formula.",
        )
    # Code delimiters do not exempt ordinary Chinese prose.
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "`这是中文`",
            target="en",
            current_text="What is the label?",
        )


def test_chinese_rejects_unprotected_english():
    validate_reply_language(
        "规则要求除数不为零。",
        target="zh",
        current_text="这条规则说什么？",
    )
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "This is an English answer.",
            target="zh",
            current_text="这条规则说什么？",
        )


def test_fake_client_uses_instruction_language():
    import asyncio

    async def run():
        client = FakeLLMClient()
        english = await client.generate(
            [
                {
                    "role": "system",
                    "content": append_language_instruction("base", "en"),
                },
                {"role": "user", "content": "Hello"},
            ]
        )
        chinese = await client.generate(
            [
                {
                    "role": "system",
                    "content": append_language_instruction("base", "zh"),
                },
                {"role": "user", "content": "Hello"},
            ]
        )
        return english, chinese

    english, chinese = asyncio.run(run())
    assert "test reply" in english
    assert "测试回复" in chinese
    validate_reply_language(english, target="en", current_text="Hello")
    validate_reply_language(chinese, target="zh", current_text="请回复。")


def test_scoped_directives_apply_to_the_current_and_next_turn():
    plan = resolve_reply_language("这次请用中文回答，今后请用英文回复。")
    assert plan.target == "zh"
    assert plan.persistent_after == "en"
    follow_up = resolve_reply_language(
        "Please continue.",
        ("这次请用中文回答，今后请用英文回复。",),
    )
    assert follow_up.target == "en"


def test_pure_formula_uses_recent_visible_language_then_falls_back_to_english():
    assert resolve_reply_language("$x=1$", ("好。",)).target == "zh"
    assert resolve_reply_language("$x=1$").target == "en"


def test_code_and_french_bread_do_not_create_language_false_positives():
    assert resolve_reply_language("请解释 `for` 循环。").target == "zh"
    assert resolve_reply_language("Explain why French bread rises.").target == "en"
    assert resolve_reply_language(
        "They wrote “请用中文回答” in a note."
    ).target == "en"


def test_guard_cases_cannot_be_exempted_by_math_or_evidence():
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "Understood. $\\text{我不理解。}$",
            target="en",
            current_text="What is this?",
        )
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "Je ne comprends pas.",
            target="en",
            current_text="What is this?",
        )
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "收到，I DO NOT UNDERSTAND.",
            target="zh",
            current_text="请解释。",
        )
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "好。",
            target="en",
            current_text="Please continue.",
            evidence_texts=("好。",),
        )
    validate_reply_language(
        'He said "好。"',
        target="en",
        current_text="Please continue.",
        evidence_texts=("好。",),
    )
    validate_reply_language(
        "API",
        target="zh",
        current_text="这个标签是什么？",
    )
    with pytest.raises(ReplyLanguageMismatchError):
        validate_reply_language(
            "...",
            target="en",
            current_text="公式是什么？",
        )


def test_quoted_third_language_directive_is_not_executed():
    assert resolve_reply_language(
        "They wrote “Please answer in French.” in a note."
    ).target == "en"
    assert resolve_reply_language(
        "请解释 $\\text{Please answer in French.}$ 的含义。"
    ).target == "zh"
    assert resolve_reply_language(
        "The label [message 1] is shown."
    ).target == "en"


def test_real_code_and_evidence_quote_can_pass():
    validate_reply_language(
        "```python\nif x > 0:\n    return x\n```",
        target="en",
        current_text="Please show the code.",
    )
    validate_reply_language(
        "He said “除数不能为零。”",
        target="en",
        current_text="What did the teacher say?",
        evidence_texts=("除数不能为零。",),
    )


def test_language_revocation_is_scoped_and_for_conversation_persists():
    assert resolve_reply_language(
        "请停止这个步骤。",
        ("For this conversation, answer in English.",),
    ).target == "en"
    assert resolve_reply_language("请停止这个步骤，请用英文。").target == "en"
    plan = resolve_reply_language("For this conversation, answer in English.")
    assert plan.target == "en"
    assert plan.persistent_after == "en"


def test_language_scope_revoke_replace_and_turn_override():
    history = ("以后都用中文。",)
    assert resolve_reply_language("取消中文要求", history).target == "zh"
    assert resolve_reply_language("取消中文要求", history).persistent_after is None
    assert resolve_reply_language("请继续", (*history, "取消中文要求")).target == "zh"

    replace = resolve_reply_language("以后用英文回复", history)
    assert replace.target == "en"
    assert resolve_reply_language("请继续", (*history, "以后用英文回复")).target == "en"

    override = resolve_reply_language("这次用英文回答", history)
    assert override.target == "en"
    assert resolve_reply_language("请继续", (*history, "这次用英文回答")).target == "zh"

    english_revoke = resolve_reply_language(
        "Forget Chinese and use English from now on.", history
    )
    assert english_revoke.target == "en"
    assert resolve_reply_language(
        "请继续", (*history, "Forget Chinese and use English from now on.")
    ).target == "en"
