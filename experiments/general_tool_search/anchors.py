"""Extract and ground topic anchors without domain trigger words."""

from __future__ import annotations

import re

from experiments.general_tool_search.types import Anchor

_CJK_RE = re.compile(r"[\u3400-\u9fff]{2,}")
_ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FORMULA_RE = re.compile(
    r"(?<![A-Za-z0-9])((?:[A-Za-z]|\d+)\s*(?:\([^)]*\)|[A-Za-z0-9])?\s*/\s*"
    r"\(?\s*(?:\d+\s*\*\s*)?[A-Za-z0-9]+\s*\)?)(?![A-Za-z0-9])"
)
_LETTER_FORMULA_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]\([^)]*\))(?![A-Za-z0-9])"
)
_QUOTED_RE = re.compile(r'["“]([^"”]{2,80})["”]')
_PROCESS_RE = re.compile(r"\bprocess\s+[A-Za-z0-9_-]+\b", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "what", "which", "who", "when", "where", "why",
    "how", "about", "you", "your", "i", "me", "my", "we", "our", "it",
    "this", "that", "these", "those", "and", "or", "of", "to", "in", "on",
    "for", "with", "from", "by", "as", "at", "any", "some", "there", "here",
    "teach", "taught", "say", "said", "call", "called", "tell", "told",
    "explain", "explained", "show", "shown", "use", "used", "using",
    "original", "originally", "current", "currently", "before", "after",
    "first", "last", "step", "steps", "rule", "rules", "reason", "reasons",
    "formula", "formulas", "exception", "exceptions", "correction",
    "corrected", "replacement", "replace", "replaced", "version", "versions",
    "history", "earlier", "previous", "previously", "have", "has", "had",
    "can", "could", "would", "should", "will", "may", "might", "must",
    "please", "exactly", "include", "including", "than", "then", "also",
    "still", "just", "very", "more", "most", "less", "least", "same",
    "different", "other", "another", "one", "two", "three", "new", "old",
    "process", "procedure", "order", "actual", "really", "know", "think",
}
_CJK_STOP = {
    "什么", "怎么", "为什么", "那个", "这个", "那条", "这条", "规则", "例外",
    "流程", "步骤", "公式", "老师", "教过", "你教", "之前", "原来", "最初",
    "现在", "当前", "还有", "以及", "是否", "可以", "怎么", "如何", "那个",
}


def _normalise_formula(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _formula_anchors(text: str) -> list[Anchor]:
    anchors: list[Anchor] = []
    for match in _FORMULA_RE.finditer(text):
        value = _normalise_formula(match.group(1))
        if "/" in value and value not in {item.normalized for item in anchors}:
            anchors.append(Anchor("formula", value, value))
    for match in _LETTER_FORMULA_RE.finditer(text):
        value = _normalise_formula(match.group(1))
        if value not in {item.normalized for item in anchors}:
            anchors.append(Anchor("formula", value, value))
    return anchors


def _cjk_anchors(text: str) -> list[Anchor]:
    anchors: list[Anchor] = []
    for match in _CJK_RE.finditer(text):
        value = match.group(0)
        if value in _CJK_STOP or len(value) > 20:
            continue
        if value not in {item.text for item in anchors}:
            anchors.append(Anchor("cjk", value, value.casefold()))
    return anchors


def _term_anchors(text: str) -> list[Anchor]:
    anchors: list[Anchor] = []
    for match in _ASCII_TERM_RE.finditer(text):
        value = match.group(0)
        if value.casefold() in _STOPWORDS:
            continue
        if len(value) == 1:
            continue
        if value not in {item.text for item in anchors}:
            anchors.append(Anchor("term", value, value.casefold()))
    return anchors


def extract_anchors(text: str) -> list[Anchor]:
    """Extract distinct, typed anchors from the current question."""
    anchors = _formula_anchors(text)
    for match in _QUOTED_RE.finditer(text):
        value = " ".join(match.group(1).split())
        if len(value) >= 2 and value not in {item.text for item in anchors}:
            anchors.append(Anchor("phrase", value, value.casefold()))
    for match in _PROCESS_RE.finditer(text):
        value = " ".join(match.group(0).split())
        if value not in {item.text for item in anchors}:
            anchors.append(Anchor("phrase", value, value.casefold()))
    anchors.extend(_cjk_anchors(text))
    anchors.extend(_term_anchors(text))
    return anchors[:8]


def _contains_formula(content: str, formula: str) -> bool:
    body = re.sub(r"\s+", "", content).casefold()
    target = re.escape(formula)
    return re.search(rf"(?<![a-z0-9]){target}(?![a-z0-9])", body) is not None


def anchor_matches(content: str, anchor: Anchor) -> bool:
    if anchor.kind == "formula":
        return _contains_formula(content, anchor.normalized)
    if anchor.kind == "phrase":
        return anchor.normalized in " ".join(content.casefold().split())
    if anchor.kind == "cjk":
        return anchor.text in content
    if anchor.kind == "term":
        return re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(anchor.text)}(?![A-Za-z0-9_])",
            content,
            re.IGNORECASE,
        ) is not None
    return False


def unique_process_theme(messages: list[str]) -> Anchor | None:
    """Return a process anchor only when the visible window has one process."""
    phrases: dict[str, str] = {}
    for content in messages:
        for match in _PROCESS_RE.finditer(content):
            text = " ".join(match.group(0).split())
            phrases.setdefault(text.casefold(), text)
    if len(phrases) != 1:
        return None
    text = next(iter(phrases.values()))
    return Anchor("phrase", text, text.casefold())


def grounded_anchors(
    question: str,
    query: str,
    *,
    theme: Anchor | None = None,
) -> list[Anchor]:
    """Keep only query anchors grounded in the question or a located theme."""
    result: list[Anchor] = []
    for anchor in extract_anchors(query):
        if anchor.kind == "formula":
            exists = _contains_formula(question, anchor.normalized)
        elif anchor.kind == "cjk":
            exists = anchor.text in question
        elif anchor.kind == "phrase":
            exists = anchor.normalized in " ".join(question.casefold().split())
        else:
            exists = re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(anchor.text)}(?![A-Za-z0-9_])",
                question,
                re.IGNORECASE,
            ) is not None
        if not exists and theme is not None:
            exists = anchor_matches(theme.text, anchor)
        if exists:
            result.append(anchor)
    return result


def anchor_weight(anchor: Anchor) -> int:
    return {"formula": 100, "phrase": 30, "cjk": 20, "term": 10}[anchor.kind]
