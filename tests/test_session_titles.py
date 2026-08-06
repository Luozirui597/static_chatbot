"""Unit tests for backend.session_titles — pure functions, no I/O."""

from backend.session_titles import derive_auto_title, normalize_title_whitespace


class TestNormalizeTitleWhitespace:
    """Tests for normalize_title_whitespace()."""

    def test_collapses_multiple_spaces(self):
        assert normalize_title_whitespace("hello   world") == "hello world"

    def test_collapses_newlines_and_tabs(self):
        assert normalize_title_whitespace("a\nb\tc") == "a b c"

    def test_strips_leading_whitespace(self):
        assert normalize_title_whitespace("  hi") == "hi"

    def test_strips_trailing_whitespace(self):
        assert normalize_title_whitespace("hi  ") == "hi"

    def test_all_whitespace_returns_empty(self):
        assert normalize_title_whitespace("   \n\t  ") == ""

    def test_single_word_passes_through(self):
        assert normalize_title_whitespace("Hello") == "Hello"

    def test_mixed_whitespace_normalised(self):
        result = normalize_title_whitespace(
            "  hello\n\nworld\t\tchat   "
        )
        assert result == "hello world chat"


class TestDeriveAutoTitle:
    """Tests for derive_auto_title()."""

    def test_short_text_unchanged(self):
        assert derive_auto_title("Explain quantum computing") == (
            "Explain quantum computing"
        )

    def test_calls_whitespace_normalisation_first(self):
        assert derive_auto_title("  hello\nworld  ") == "hello world"

    def test_exactly_40_chars_no_ellipsis(self):
        text = "a" * 40
        result = derive_auto_title(text, max_chars=40)
        assert result == text
        assert "…" not in result

    def test_41_chars_truncated_with_ellipsis(self):
        text = "a" * 41
        result = derive_auto_title(text, max_chars=40)
        assert len(result) == 41  # 40 chars + …
        assert result.endswith("…")
        assert result[:40] == "a" * 40

    def test_long_text_truncated(self):
        text = "a" * 100
        result = derive_auto_title(text, max_chars=40)
        assert len(result) == 41
        assert result.endswith("…")

    def test_clamped_to_255_chars(self):
        text = "a" * 300
        result = derive_auto_title(text, max_chars=40)
        assert len(result) <= 255

    def test_custom_max_chars(self):
        text = "a" * 20
        result = derive_auto_title(text, max_chars=10)
        assert len(result) == 11  # 10 + …
        assert result.endswith("…")
