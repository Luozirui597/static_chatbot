"""Reply-language resolution and post-generation language validation.

This module is deliberately small and independent of the chat transport.  The
target language is resolved *before* the user message is persisted; the model's
reply is validated *before* the assistant message is persisted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LANGUAGE_INSTRUCTIONS = {
    "en": (
        "Reply in English. Use English for all natural-language prose. Keep "
        "formulas, code, fixed labels, proper names, and exact quotations "
        "unchanged; do not translate them."
    ),
    "zh": (
        "请用中文回复。所有自然语言正文使用中文；公式、代码、固定标签、专有名词和"
        "逐字引文保持原样，不要翻译。"
    ),
}

LANGUAGE_PROMPT_SUFFIX = {"en": "+l1-en", "zh": "+l1-zh"}
CLARIFY_PROMPT_VERSION = "receive-teaching-v7+clarify-v1"
SUPPORTED = ("en", "zh")

LANGUAGE_AMBIGUOUS = "receive_reply_language_ambiguous"
LANGUAGE_CONFLICT = "receive_reply_language_conflict"
LANGUAGE_UNSUPPORTED = "receive_reply_language_unsupported"
LANGUAGE_MISMATCH = "receive_reply_language_mismatch"


class ReplyLanguageInputError(Exception):
    """The current request cannot be assigned one safe target language."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ReplyLanguageMismatchError(Exception):
    """The generated reply does not use the frozen target language."""

    code = LANGUAGE_MISMATCH

    def __init__(self, target: str) -> None:
        super().__init__("reply language mismatch")
        self.target = target
        self.message = (
            "Your message was saved. The assistant reply did not use the "
            "required language and was not saved."
            if target == "en"
            else "你的消息已保存。助手回复的语言不符合要求，该回复未保存。"
        )


@dataclass(frozen=True)
class ReplyLanguagePlan:
    target: str
    source: str
    persistent_after: str | None


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_RE = re.compile(r"[A-Za-z]+")
_MATH_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos",
    "arctan", "log", "ln", "exp", "lim", "max", "min", "det", "sqrt",
    "sum", "prod", "int", "infty", "theta", "alpha", "beta", "gamma",
    "delta", "lambda", "mu", "pi",
}

_FUNCTION_WORDS = {
    "an", "the", "is", "are", "was", "were", "be", "been", "you",
    "he", "she", "it", "we", "they", "me", "my", "your", "our", "their",
    "to", "of", "and", "or", "but", "in", "on", "at", "for", "from", "with",
    "by", "as", "this", "that", "these", "those", "please", "ok", "okay",
    "yes", "no", "not", "do", "does", "did", "can", "could", "will", "would",
    "should", "what", "why", "how", "when", "where", "which", "who", "tell",
    "show", "give", "need", "want", "use", "using", "write", "read", "say",
    "said", "teach", "taught", "mean", "means", "about", "only", "just",
    "again", "more", "less", "than", "if", "then", "because", "so", "also",
    "here", "there", "now", "later", "still", "already", "never", "always",
    "some", "any", "all", "one", "two", "continue", "explain", "answer",
    "reply", "respond", "formula", "label", "quote", "text", "language",
    "english", "chinese", "understood", "thanks", "thank", "got", "sure",
    "hello", "hi",
}
_PERSISTENT_CUES = (
    "from now on", "from this point on", "for this conversation",
    "for the rest of this conversation",
    "for the remainder of this conversation", "for future replies",
    "以后", "今后", "从现在起", "接下来都", "后续都", "本对话", "这段对话",
)
_TURN_CUES = (
    "this time", "this turn", "this one", "this question", "this answer",
    "for this turn", "这次", "这一条", "这一轮", "本轮", "这道题", "这个回答",
)
_REVOKE_CUES = (
    "stop using", "stop replying in", "no longer use", "no longer reply",
    "forget the", "cancel the", "取消", "停止", "不用再", "不再用", "放弃",
    "no language preference", "without a language preference",
)
_LANGUAGE_ALIASES = (
    ("en", r"English|英语|英文"),
    ("zh", r"Chinese|Mandarin|中文|汉语|普通话"),
)
_OTHER_LANGUAGES = (
    r"French|Spanish|German|Japanese|Korean|Portuguese|Italian|Russian|"
    r"法语|西班牙语|德语|日语|韩语|葡萄牙语|意大利语|俄语"
)
_REQUEST_VERB = (
    r"reply|respond|answer|write|speak|explain|continue|use|talk|say|"
    r"回复|回答|写|说|解释|继续|使用|讲"
)


def _strip_quotes(text: str) -> str:
    text = re.sub(r'"[^"\n]{0,500}"', " ", text)
    text = re.sub(r"'[^'\n]{0,500}'", " ", text)
    text = re.sub(r"“[^”\n]{0,500}”", " ", text)
    text = re.sub(r"「[^」\n]{0,500}」", " ", text)
    text = re.sub(r"『[^』\n]{0,500}』", " ", text)
    return text

_FENCE_STRIP_RE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_INLINE_STRIP_RE = re.compile(r"`[^`\n]+`")
_MATH_STRIP_RE = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S
)
_SOURCE_STRIP_RE = re.compile(r"\[message\s+\d+\]", re.IGNORECASE)
_OTHER_LANGUAGE_REQUEST_RE = re.compile(
    rf"(?:(?:in|用)\s*(?:{_OTHER_LANGUAGES})|"
    rf"(?:{_OTHER_LANGUAGES})\s*(?:please|reply|respond|answer|回复|回答)|"
    rf"(?:use)\s*(?:{_OTHER_LANGUAGES}))",
    re.IGNORECASE,
)


def _strip_non_text(text: str) -> str:
    """Remove code, formulas, labels and quotations before language logic."""
    text = _FENCE_STRIP_RE.sub(" ", text)
    text = _INLINE_STRIP_RE.sub(" ", text)
    text = _MATH_STRIP_RE.sub(" ", text)
    text = _SOURCE_STRIP_RE.sub(" ", text)
    return _strip_quotes(text)


def _cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _english_cue(text: str) -> bool:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return any(w in _FUNCTION_WORDS for w in words)


def _is_language_revocation(text: str) -> bool:
    """Only a revocation aimed at language preference cancels it."""
    lowered = text.lower()
    language_words = re.compile(
        r"language|preference|english|chinese|mandarin|语言|偏好|英文|中文|汉语|普通话"
    )
    for cue in _REVOKE_CUES:
        for match in re.finditer(re.escape(cue.lower()), lowered):
            window = lowered[max(0, match.start() - 50): match.end() + 50]
            if language_words.search(window):
                return True
    return False


def _scope_for(text: str, start: int, end: int) -> str:
    """Find a scope cue in the same clause, or a previous persistent clause."""
    lowered = text.lower()
    boundary = re.compile(r"[,;，；]")
    boundaries = [match.start() for match in boundary.finditer(text)]
    clause_start = max([b for b in boundaries if b < start], default=-1) + 1
    clause_end = min(
        [b for b in boundaries if b >= end], default=len(text)
    )
    clause = lowered[clause_start:clause_end]
    candidates: list[tuple[int, str]] = []
    for scope, cues in (("persistent", _PERSISTENT_CUES), ("turn", _TURN_CUES)):
        for cue in cues:
            for match in re.finditer(re.escape(cue.lower()), clause):
                position = clause_start + match.start()
                distance = abs(position - start)
                candidates.append((distance, scope))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    # "From now on, reply in English." keeps the persistent scope across a
    # comma, but a directive in the next clause must not steal the previous
    # clause's scope.
    previous = lowered[:clause_start].rstrip(" ,;，；")
    if any(cue.lower() in previous[-50:] for cue in _PERSISTENT_CUES):
        return "persistent"
    return "turn"


def _language_directives(text: str) -> tuple[list[tuple[str, str]], bool]:
    """Return (language, scope) requests and whether a preference is revoked."""
    clean = _strip_non_text(text)
    if _OTHER_LANGUAGE_REQUEST_RE.search(clean):
        raise ReplyLanguageInputError(
            LANGUAGE_UNSUPPORTED,
            "Only English and Chinese replies are supported for this endpoint.",
        )

    lowered = clean.lower()
    revoked = _is_language_revocation(clean)
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for code, alias in _LANGUAGE_ALIASES:
        patterns = (
            # Reply/answer/explain + explicit "in <language>".
            rf"(?:reply|respond|answer|write|speak|explain|continue|talk|say|"
            rf"回复|回答|写|说|解释|继续)\s*"
            rf"(?:in|using|用)\s*(?:{alias})",
            # Chinese-style "用英文解释" / "使用中文回答".
            rf"(?:用|使用)\s*(?:{alias})\s*"
            rf"(?:reply|respond|answer|write|speak|explain|continue|talk|say|"
            rf"回复|回答|写|说|解释|继续)",
            # Language followed by a reply verb.
            rf"(?:{alias})\s*(?:please|reply|respond|answer|回复|回答)",
            # "Please use Chinese from now on." / "请用英文." but not
            # "use English variable names".
            rf"(?:use|使用)\s*(?:{alias})"
            rf"(?=\s*(?:[.!?。！？]|$|from now on|for (?:this|the) conversation|"
            rf"for the rest of|今后|以后|本次会话|之后))",
            # Bare "in English." / "请用英文。".
            rf"(?:in|用)\s*(?:{alias})\s*(?:[.!?。！？]|$)",
            # Coordinated targets such as "in English or Chinese" must
            # contribute both sides and then conflict.
            rf"(?:in|using|用)\s*(?:{alias})\s*(?:or|或|/)\s*"
            rf"(?:{'|'.join(other for other_code, other in _LANGUAGE_ALIASES if other_code != code)})",
            rf"(?:in|using|用)\s*"
            rf"(?:{'|'.join(other for other_code, other in _LANGUAGE_ALIASES if other_code != code)})"
            rf"\s*(?:or|或|/)\s*(?:{alias})",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, clean, re.IGNORECASE):
                prefix = clean[max(0, match.start() - 32): match.start()]
                if _is_language_revocation(prefix):
                    revoked_codes = {
                        revoked_code
                        for revoked_code, revoked_alias in _LANGUAGE_ALIASES
                        if re.search(revoked_alias, prefix, re.IGNORECASE)
                    }
                    if code in revoked_codes:
                        continue
                scope = _scope_for(clean, match.start(), match.end())
                marker = (code, match.start(), scope)
                if marker in seen:
                    continue
                seen.add(marker)
                found.append((code, scope))
    return found, revoked

def _persistent_from_history(history_texts: tuple[str, ...]) -> str | None:
    target: str | None = None
    for text in history_texts:
        try:
            directives, revoked = _language_directives(text)
        except ReplyLanguageInputError:
            continue
        if revoked:
            target = None
        persistent = [code for code, scope in directives if scope == "persistent"]
        if persistent:
            target = persistent[-1]
        elif any(scope == "turn" for _, scope in directives):
            # A turn-scoped instruction in history does not persist.
            continue
    return target


def _primary_language(text: str) -> str | None:
    clean = _strip_non_text(text)
    has_cjk = _cjk(clean)
    has_english = _english_cue(clean)
    if has_cjk and has_english:
        raise ReplyLanguageInputError(
            LANGUAGE_AMBIGUOUS,
            "The request mixes Chinese and English without a clear main language.",
        )
    if has_cjk:
        return "zh"
    if has_english:
        return "en"
    return None


def _recent_history_language(history_texts: tuple[str, ...]) -> str | None:
    for text in reversed(history_texts):
        clean = _strip_non_text(text)
        if not (_cjk(clean) or _english_cue(clean)):
            continue
        try:
            primary = _primary_language(clean)
        except ReplyLanguageInputError:
            continue
        if primary is not None:
            return primary
    return None


def resolve_reply_language(
    current_text: str,
    history_texts: tuple[str, ...] = (),
) -> ReplyLanguagePlan:
    """Resolve one target for the current user request.

    ``history_texts`` is oldest-to-newest user text from the recent stored
    history.  Only explicit persistent instructions survive a turn; a quoted
    sentence is never a language instruction.
    """
    current_directives, current_revoked = _language_directives(current_text)
    turn_targets = {code for code, scope in current_directives if scope == "turn"}
    persistent_targets = {
        code for code, scope in current_directives if scope == "persistent"
    }
    if len(turn_targets) > 1 or len(persistent_targets) > 1:
        raise ReplyLanguageInputError(
            LANGUAGE_CONFLICT,
            "The request contains conflicting reply-language instructions.",
        )

    history_target = _persistent_from_history(history_texts)
    persistent_after = history_target
    if persistent_targets:
        persistent_after = next(iter(persistent_targets))
    elif current_revoked:
        persistent_after = None

    if turn_targets:
        target = next(iter(turn_targets))
        source = "current_turn"
    elif persistent_targets:
        target = next(iter(persistent_targets))
        source = "current_persistent"
    elif persistent_after:
        target = persistent_after
        source = "history_persistent"
    else:
        primary = _primary_language(current_text)
        if primary is None:
            visible_history = _recent_history_language(history_texts)
            if visible_history is not None:
                target = visible_history
                source = "history_visible"
            else:
                target = "en"
                source = "v7_fallback"
        else:
            target = primary
            source = "primary"
    return ReplyLanguagePlan(
        target=target,
        source=source,
        persistent_after=persistent_after,
    )


# ---------------------------------------------------------------------------
# Prompt/version helpers
# ---------------------------------------------------------------------------


def append_language_instruction(system_prompt: str, target: str) -> str:
    if target not in SUPPORTED:
        raise ValueError("unsupported reply language")
    return system_prompt.rstrip() + "\n\n" + LANGUAGE_INSTRUCTIONS[target] + "\n"


def language_prompt_version(base_version: str, target: str) -> str:
    if target not in SUPPORTED:
        raise ValueError("unsupported reply language")
    version = base_version + LANGUAGE_PROMPT_SUFFIX[target]
    if len(version) > 50:
        raise ValueError("prompt version too long")
    return version


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.S)
_INLINE_RE = re.compile(r"`[^`\n]+`")
_CODE_HINT_RE = re.compile(
    r"[=(){}[\]<>;\\]|\b(?:def|class|return|import|print|function|const|let|var|if|for|while)\b"
)
_MATH_RE = re.compile(
    r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S
)
_LABEL_CUE_RE = re.compile(
    r"\b(?:label|tag|name|term|acronym|identifier|code)\b|标签|名称|术语|标记|缩写",
    re.IGNORECASE,
)
_LABEL_PHRASE_RE = re.compile(r"[A-Z][A-Za-z0-9_-]*(?:[ \t]+[A-Z][A-Za-z0-9_-]*)*")

_SOURCE_RE = re.compile(r"\[message\s+\d+\]", re.IGNORECASE)
_QUOTE_RE = re.compile(
    r'"([^"\n]{1,500})"|“([^”\n]{1,500})”|「([^」\n]{1,500})」|『([^』\n]{1,500})』'
)
_FORMULAISH_RE = re.compile(r"[=^+\-*/]|\\[A-Za-z]+|\$")
_NON_ENGLISH_WORDS = {
    "je", "ne", "pas", "comprends", "comprend", "bonjour", "merci",
    "vous", "nous", "suis", "est", "une", "des", "les", "pour",
    "ich", "nicht", "verstehe", "bitte", "danke",
    "hola", "gracias", "entiendo", "por", "favor",
}


def _content_requests_only(current_text: str) -> bool:
    """True for a direct formula/label request, not an explanation request."""
    lowered = _strip_non_text(current_text).lower()
    if not re.search(
        r"formula|expression|label|code|snippet|公式|表达式|标签|代码",
        lowered,
    ):
        return False
    if re.search(r"\b(?:explain|describe|why|how)\b|说明|解释|为什么|怎么", lowered):
        return False
    return True


def _has_natural_language(text: str) -> bool:
    return _cjk(text) or bool(_WORD_RE.search(text))


def _code_like(text: str) -> bool:
    return bool(_CODE_HINT_RE.search(text))


def _source_texts(current_text: str, evidence_texts: tuple[str, ...]) -> tuple[str, ...]:
    return (current_text,) + tuple(evidence_texts)


def _quoted_fragment(match: re.Match[str]) -> str:
    return next(group for group in match.groups() if group is not None)


def _is_exact_quote(
    fragment: str, current_text: str, evidence_texts: tuple[str, ...]
) -> bool:
    target = fragment.strip()
    if not target:
        return False
    return any(target in source for source in _source_texts(current_text, evidence_texts))


def _math_visible_text(match: re.Match[str]) -> str:
    body = match.group(0)
    # \text{...} contains visible natural language and must be judged.  Bare
    # CJK inside a math span is also visible prose, not a TeX variable.
    texts = re.findall(r"\\text\{([^{}]*)\}", body)
    texts.extend(_CJK_RE.findall(body))
    return " " + " ".join(dict.fromkeys(text for text in texts if text)) + " "


def _fence_body(raw: str) -> str:
    lines = raw.strip().splitlines()
    if lines and lines[0].lstrip().startswith(("```", "~~~")):
        lines = lines[1:]
    if lines and lines[-1].strip() in {"```", "~~~"}:
        lines = lines[:-1]
    return "\n".join(lines)


def _exact_source_fragment(
    fragment: str,
    current_text: str,
    evidence_texts: tuple[str, ...],
) -> bool:
    fragment = fragment.strip()
    if not fragment:
        return False
    return any(
        fragment in source
        for source in (current_text, *evidence_texts)
    )


def _source_code_fragments(
    current_text: str, evidence_texts: tuple[str, ...]
) -> tuple[str, ...]:
    fragments: list[str] = []
    for source in (current_text, *evidence_texts):
        for match in _FENCE_RE.finditer(source):
            body = _fence_body(match.group(0)).strip()
            if body:
                fragments.append(body)
        for match in _INLINE_RE.finditer(source):
            body = match.group(0)[1:-1].strip()
            if body and _code_like(body):
                fragments.append(body)
    return tuple(dict.fromkeys(fragments))


def _protect_exact_source_code(
    reply: str,
    current_text: str,
    evidence_texts: tuple[str, ...],
) -> str:
    """Remove verbatim code from this request/evidence before prose checks.

    This is deliberately exact: generated or altered code is not protected.
    New natural-language prose around/inside changed code is still checked.
    """
    for fragment in _source_code_fragments(current_text, evidence_texts):
        if len(fragment) >= 4 and fragment in reply:
            reply = reply.replace(fragment, " ")
    return reply


def _strip_protected(
    reply: str,
    *,
    current_text: str,
    evidence_texts: tuple[str, ...],
) -> str:
    label_fragments = _label_fragments(current_text, evidence_texts)
    reply = _protect_labels(reply, label_fragments)

    # A code delimiter only protects code-like content.  Natural-language
    # text wrapped in backticks still counts as visible prose.
    def _cjk_only(text: str) -> str:
        return " ".join(_CJK_RE.findall(text))

    def _fence(match: re.Match[str]) -> str:
        body = _fence_body(match.group(0))
        if _exact_source_fragment(body, current_text, evidence_texts):
            return " "
        # A non-source fence protects code structure, but its words (CJK or
        # Latin prose) remain visible evidence of the reply language.
        return " " + body + " "

    def _inline(match: re.Match[str]) -> str:
        body = match.group(0)[1:-1]
        if _exact_source_fragment(body, current_text, evidence_texts):
            return " "
        # Backticks are formatting, not a language exemption.
        return " " + body + " "

    def _math(match: re.Match[str]) -> str:
        raw = match.group(0)
        if _exact_source_fragment(raw, current_text, evidence_texts):
            return " "
        # \text{...} is prose.  A math span containing multiple Latin words
        # or an English function word is also prose (not a pure formula).
        texts = re.findall(r"\\text\{([^{}]*)\}", raw)
        # TeX commands and common math identifiers are syntax, not prose.
        word_source = re.sub(r"\\[A-Za-z]+", " ", raw)
        body_words = [
            word for word in _WORD_RE.findall(word_source)
            if len(word) > 1 and word.lower() not in _MATH_WORDS
        ]
        if len(body_words) >= 2 or any(
            word.lower() in _FUNCTION_WORDS for word in body_words
        ):
            texts.extend(body_words)
        texts.extend(_CJK_RE.findall(raw))
        return " " + " ".join(dict.fromkeys(t for t in texts if t)) + " "

    def _quote(match: re.Match[str]) -> str:
        fragment = _quoted_fragment(match)
        if _is_exact_quote(fragment, current_text, evidence_texts):
            return " "
        return match.group(0)

    text = _FENCE_RE.sub(_fence, reply)
    text = _INLINE_RE.sub(_inline, text)
    text = _MATH_RE.sub(_math, text)
    text = _SOURCE_RE.sub(" ", text)
    text = _QUOTE_RE.sub(_quote, text)
    return text


def _label_fragments(
    current_text: str, evidence_texts: tuple[str, ...]
) -> tuple[str, ...]:
    """Return exact label spans tied to a label cue, not nearby prose.

    A cue window is deliberately not used: ``The label is Foobar.`` followed by
    an unrelated sentence must not protect that sentence.  Only the candidate
    immediately governed by the cue (or immediately before it) is a label.
    """
    fragments: list[str] = []
    patterns = (
        re.compile(
            r"(?i:label|tag|name|term|acronym|identifier)\s*"
            r"(?:is|are|=|:)?\s*"
            r"([A-Z][A-Za-z0-9_-]*(?:[ \t]+[A-Z][A-Za-z0-9_-]*)*)"
        ),
        re.compile(
            r"([A-Z][A-Za-z0-9_-]*(?:[ \t]+[A-Z][A-Za-z0-9_-]*)*)"
            r"\s*(?i:label|tag|name|term|acronym|identifier)\b"
        ),
        re.compile(
            r"(?:标签|名称|术语|标记|缩写)\s*"
            r"(?:是|为|叫|：|:)?\s*"
            r"([A-Z][A-Za-z0-9_-]*(?:[ \t]+[A-Z][A-Za-z0-9_-]*)*)"
        ),
        re.compile(
            r"([A-Z][A-Za-z0-9_-]*(?:[ \t]+[A-Z][A-Za-z0-9_-]*)*)"
            r"\s*(?:标签|名称|术语|标记|缩写)"
        ),
        # "Foobar and it requires testing 标签" — the cue is postposed,
        # but only the leading capitalized label span is confirmed.
        re.compile(
            r"(?:^|[.!?。！？]\s*)\s*"
            r"([A-Z][A-Za-z0-9_-]*(?:[ \t]+[A-Z][A-Za-z0-9_-]*)*)"
            r"[^.!?。！？\n]{0,120}?"
            r"(?:标签|名称|术语|标记|缩写)"
        ),
    )
    for source in (current_text, *evidence_texts):
        for pattern in patterns:
            for match in pattern.finditer(source):
                phrase = match.group(1).strip(" .,;:!?()[]{}<>\"'`")
                if not phrase or len(phrase) < 2:
                    continue
                if phrase in source and phrase not in fragments:
                    fragments.append(phrase)
        # A standalone capitalized source token can be a proper name (Quux,
        # Foobar, API).  Multi-word capitalized prose is not a label unless a
        # cue pattern above confirmed the whole span.
        for match in re.finditer(r"[A-Z][A-Za-z0-9_-]{1,}", source):
            token = match.group(0)
            if token.lower() in _FUNCTION_WORDS or token.lower() in _MATH_WORDS:
                continue
            before = source[: match.start()].rstrip()
            after = source[match.end() :].lstrip()
            if before and before[-1].isupper():
                continue
            if after and after[0].isupper():
                continue
            if token not in fragments:
                fragments.append(token)
    return tuple(fragments)


def _protect_labels(
    reply: str, fragments: tuple[str, ...]
) -> str:
    for phrase in sorted(fragments, key=len, reverse=True):
        reply = reply.replace(phrase, " ")
    return reply


def _is_label_only_reply(reply: str, fragments: tuple[str, ...]) -> bool:
    if not fragments:
        return False
    rest = _protect_labels(reply, fragments)
    return not _has_natural_language(rest)


def _label_only_request(current_text: str) -> bool:
    lowered = _strip_non_text(current_text).lower()
    return bool(
        re.search(r"只(?:需|要|返回|输出|给)|only\s+(?:return|output|show)", lowered)
        and _LABEL_CUE_RE.search(lowered)
    )


def _is_standalone_acronym_reply(reply: str, current_text: str) -> bool:
    if not _LABEL_CUE_RE.search(current_text):
        return False
    stripped = reply.strip().rstrip(".!?。！？")
    if not re.fullmatch(r"[A-Z][A-Z0-9_-]{1,5}", stripped):
        return False
    # A single short acronym is acceptable only as a label answer, not as a
    # sentence.  Full uppercase prose has multiple words and is handled by the
    # uppercase-prose guard.
    return True


def _allowed_latin_token(
    token: str, current_text: str, evidence_texts: tuple[str, ...]
) -> bool:
    # Labels and proper names are removed by _protect_labels; no general
    # case-based or evidence-word exemption is permitted here.
    return len(token) <= 1 or token.isdigit()


def _non_english_latin(words: list[str]) -> bool:
    foreign = [word for word in words if word.lower() in _NON_ENGLISH_WORDS]
    if not foreign:
        return False
    # A single ambiguous word (e.g. English "pour") must not condemn an
    # otherwise English sentence.  Reject clearly foreign sentences, or a
    # lone foreign word with no English function-word context.
    if len(foreign) >= 2:
        return True
    has_english_context = any(
        word.lower() in _FUNCTION_WORDS
        and word.lower() not in _NON_ENGLISH_WORDS
        for word in words
    )
    return not has_english_context


def _has_evidence_quote(
    reply: str, current_text: str, evidence_texts: tuple[str, ...]
) -> bool:
    for match in _QUOTE_RE.finditer(reply):
        if _is_exact_quote(_quoted_fragment(match), current_text, evidence_texts):
            return True
    return False


def _is_formulaish(reply: str) -> bool:
    return bool(_FORMULAISH_RE.search(reply))


def _is_content_only_reply(reply: str) -> bool:
    return _is_formulaish(reply) or _code_like(reply) or bool(
        _FENCE_RE.search(reply)
    )


def validate_reply_language(
    reply: str,
    *,
    target: str,
    current_text: str,
    evidence_texts: tuple[str, ...] = (),
) -> None:
    if target not in SUPPORTED:
        raise ValueError("unsupported reply language")
    visible = _strip_protected(
        reply, current_text=current_text, evidence_texts=evidence_texts
    )
    has_cjk = _cjk(visible)
    words = _WORD_RE.findall(visible)
    has_latin = bool(words)
    natural = has_cjk or has_latin
    has_evidence_quote = _has_evidence_quote(reply, current_text, evidence_texts)
    content_only = _content_requests_only(current_text) or not _has_natural_language(
        _strip_non_text(current_text)
    )

    if target == "en":
        if has_cjk:
            raise ReplyLanguageMismatchError(target)
        if _non_english_latin(words):
            raise ReplyLanguageMismatchError(target)
        if not natural:
            if has_evidence_quote:
                return
            if content_only and _is_content_only_reply(reply):
                return
            if _label_only_request(current_text) and _is_label_only_reply(
                reply, _label_fragments(current_text, evidence_texts)
            ):
                return
            raise ReplyLanguageMismatchError(target)
        return

    # target == "zh"
    if not has_cjk:
        if words:
            if content_only and _is_standalone_acronym_reply(
                reply, current_text
            ):
                return
            uppercase_prose = [
                token for token in words if len(token) > 1 and token.isupper()
            ]
            if len(uppercase_prose) > 1:
                raise ReplyLanguageMismatchError(target)
            if any(
                not _allowed_latin_token(token, current_text, evidence_texts)
                for token in words
            ):
                raise ReplyLanguageMismatchError(target)
            return
        if has_evidence_quote:
            return
        if content_only and _is_content_only_reply(reply):
            return
        if _label_only_request(current_text) and _is_label_only_reply(
            reply, _label_fragments(current_text, evidence_texts)
        ):
            return
        if content_only and _is_standalone_acronym_reply(reply, current_text):
            return
        raise ReplyLanguageMismatchError(target)
    uppercase_prose = [
        token for token in words if len(token) > 1 and token.isupper()
    ]
    if len(uppercase_prose) > 1:
        raise ReplyLanguageMismatchError(target)
    if any(
        not _allowed_latin_token(token, current_text, evidence_texts)
        for token in words
    ):
        raise ReplyLanguageMismatchError(target)
