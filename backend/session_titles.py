"""Pure helpers for session title normalisation and auto-generation.

These functions have no database or service dependencies and can be
tested in isolation.
"""

from __future__ import annotations

import re


def normalize_title_whitespace(text: str) -> str:
    """Collapse all whitespace runs and strip leading/trailing space.

    Returns a string with no newlines, tabs, or consecutive spaces.
    Does **not** truncate.

    >>> normalize_title_whitespace("  hello\\n\\tworld   ")
    'hello world'
    """
    return re.sub(r"\s+", " ", text).strip()


def derive_auto_title(content: str, max_chars: int = 40) -> str:
    """Derive a deterministic session title from message *content*.

    1. Normalise whitespace via :func:`normalize_title_whitespace`.
    2. If the result exceeds *max_chars* Unicode characters, truncate
       to *max_chars* and append ``…`` (U+2026).
    3. Clamp to 255 characters (the database column limit).

    >>> derive_auto_title("Explain quantum computing")
    'Explain quantum computing'
    """
    normalized = normalize_title_whitespace(content)
    if len(normalized) > max_chars:
        normalized = normalized[:max_chars] + "…"
    return normalized[:255]
