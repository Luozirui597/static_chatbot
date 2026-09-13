"""Pure tests for Iteration 2B history-review source selection."""

import pytest

from backend.history_review_selection import (
    HISTORY_REVIEW_BUDGET_VERSION,
    HISTORY_REVIEW_CONTEXT_TOKENS,
    HISTORY_REVIEW_MAX_SOURCE_CHARS,
    HISTORY_REVIEW_MAX_USER_MESSAGES,
    HISTORY_REVIEW_MESSAGE_OVERHEAD_TOKENS,
    HISTORY_REVIEW_RESERVED_MARGIN_TOKENS,
    HISTORY_REVIEW_RESERVED_OUTPUT_TOKENS,
    HISTORY_REVIEW_RESERVED_PROMPT_TOKENS,
    HISTORY_REVIEW_SELECTION_POLICY_VERSION,
    HISTORY_REVIEW_SOURCE_TOKEN_BUDGET,
    HistoryReviewCandidate,
    HistoryReviewSelection,
    HistoryReviewSourceMessageTooLarge,
    estimate_history_review_tokens,
    select_history_review_sources,
)


class TestBudgetConstants:
    def test_source_token_budget_derivation(self):
        assert (
            HISTORY_REVIEW_CONTEXT_TOKENS
            - HISTORY_REVIEW_RESERVED_PROMPT_TOKENS
            - HISTORY_REVIEW_RESERVED_OUTPUT_TOKENS
            - HISTORY_REVIEW_RESERVED_MARGIN_TOKENS
            == HISTORY_REVIEW_SOURCE_TOKEN_BUDGET
        )
        assert HISTORY_REVIEW_SOURCE_TOKEN_BUDGET == 2016
        assert HISTORY_REVIEW_MESSAGE_OVERHEAD_TOKENS == 16


class TestEstimateHistoryReviewTokens:
    def test_empty_string_still_pays_message_overhead(self):
        assert estimate_history_review_tokens("") == 16

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("a", 17),
            ("ab", 17),
            ("abc", 17),
            ("abcd", 17),
            ("abcde", 18),
            ("你好", 18),
            ("ab中文", 19),
            ("😀", 17),
        ],
    )
    def test_fixed_estimator(self, text, expected):
        assert estimate_history_review_tokens(text) == expected

    @pytest.mark.parametrize("value", [None, 123, b"bytes", ["list"]])
    def test_non_string_is_rejected(self, value):
        with pytest.raises(TypeError):
            estimate_history_review_tokens(value)


class TestCandidateValidation:
    def test_valid_candidate(self):
        candidate = HistoryReviewCandidate(1, "hello")
        assert candidate.message_id == 1
        assert candidate.content == "hello"

    @pytest.mark.parametrize("bad_id", [True, False, 0, -1, 1.5])
    def test_invalid_message_id(self, bad_id):
        with pytest.raises(ValueError):
            HistoryReviewCandidate(bad_id, "hello")

    @pytest.mark.parametrize("bad_content", [None, 123, b"bytes"])
    def test_invalid_content(self, bad_content):
        with pytest.raises(TypeError):
            HistoryReviewCandidate(1, bad_content)


class TestSelectionValidation:
    def test_non_sequence_is_rejected(self):
        with pytest.raises(TypeError):
            select_history_review_sources("not a sequence")

    def test_non_candidate_item_is_rejected(self):
        with pytest.raises(TypeError):
            select_history_review_sources([{"message_id": 1}])

    def test_duplicate_ids_are_rejected(self):
        candidates = [
            HistoryReviewCandidate(1, "a"),
            HistoryReviewCandidate(1, "b"),
        ]
        with pytest.raises(ValueError):
            select_history_review_sources(candidates)

    def test_non_increasing_ids_are_rejected(self):
        candidates = [
            HistoryReviewCandidate(2, "a"),
            HistoryReviewCandidate(1, "b"),
        ]
        with pytest.raises(ValueError):
            select_history_review_sources(candidates)

    def test_input_is_not_mutated(self):
        candidates = [
            HistoryReviewCandidate(1, "a"),
            HistoryReviewCandidate(2, "b"),
        ]
        snapshot = list(candidates)
        select_history_review_sources(candidates)
        assert candidates == snapshot


class TestSelection:
    def test_empty_candidates(self):
        result = select_history_review_sources([])
        assert isinstance(result, HistoryReviewSelection)
        assert result.eligible_message_count == 0
        assert result.eligible_char_count == 0
        assert result.eligible_from_message_id is None
        assert result.eligible_through_message_id is None
        assert result.selected_candidates == ()
        assert result.source_message_count == 0
        assert result.source_char_count == 0
        assert result.source_token_estimate == 0
        assert result.source_from_message_id is None
        assert result.source_through_message_id is None
        assert result.truncated is False
        assert result.budget_version == HISTORY_REVIEW_BUDGET_VERSION
        assert (
            result.selection_policy_version
            == HISTORY_REVIEW_SELECTION_POLICY_VERSION
        )

    def test_single_message(self):
        result = select_history_review_sources([
            HistoryReviewCandidate(7, "hello"),
        ])
        assert result.eligible_message_count == 1
        assert result.eligible_char_count == 5
        assert result.eligible_from_message_id == 7
        assert result.eligible_through_message_id == 7
        assert result.source_message_count == 1
        assert result.source_char_count == 5
        assert result.source_token_estimate == 18
        assert result.source_from_message_id == 7
        assert result.source_through_message_id == 7
        assert result.truncated is False

    def test_time_order_is_restored(self):
        result = select_history_review_sources([
            HistoryReviewCandidate(1, "old"),
            HistoryReviewCandidate(2, "middle"),
            HistoryReviewCandidate(3, "new"),
        ])
        assert [c.message_id for c in result.selected_candidates] == [1, 2, 3]
        assert result.source_from_message_id == 1
        assert result.source_through_message_id == 3

    def test_newest_ten_of_eleven(self):
        candidates = [
            HistoryReviewCandidate(i, f"m{i}") for i in range(1, 12)
        ]
        result = select_history_review_sources(candidates)
        assert result.source_message_count == HISTORY_REVIEW_MAX_USER_MESSAGES
        assert result.eligible_message_count == 11
        assert [c.message_id for c in result.selected_candidates] == list(
            range(2, 12)
        )
        assert result.truncated is True

    def test_exactly_ten_is_not_truncated(self):
        candidates = [
            HistoryReviewCandidate(i, "x") for i in range(1, 11)
        ]
        result = select_history_review_sources(candidates)
        assert result.source_message_count == 10
        assert result.truncated is False

    def test_char_budget_stops_before_older_message(self):
        old = "a" * 3000
        middle = "b" * 1500
        new = "c" * 1000
        candidates = [
            HistoryReviewCandidate(1, old),
            HistoryReviewCandidate(2, middle),
            HistoryReviewCandidate(3, new),
        ]
        result = select_history_review_sources(candidates)
        assert [c.message_id for c in result.selected_candidates] == [2, 3]
        assert result.source_char_count == 2500
        assert result.truncated is True

    def test_stops_at_first_oversized_older_message_without_skipping(self):
        candidates = [
            HistoryReviewCandidate(1, "small-old"),
            HistoryReviewCandidate(2, "x" * 5000),
            HistoryReviewCandidate(3, "newest"),
        ]
        result = select_history_review_sources(candidates)
        assert [c.message_id for c in result.selected_candidates] == [3]
        assert result.truncated is True

    def test_token_budget_with_cjk(self):
        candidates = [
            HistoryReviewCandidate(1, "旧" * 1001),
            HistoryReviewCandidate(2, "新" * 1000),
        ]
        result = select_history_review_sources(candidates)
        assert [c.message_id for c in result.selected_candidates] == [2]
        assert result.source_token_estimate <= 2016
        assert result.truncated is True

    def test_exact_char_limit_is_allowed(self):
        text = "a" * HISTORY_REVIEW_MAX_SOURCE_CHARS
        result = select_history_review_sources([
            HistoryReviewCandidate(1, text),
        ])
        assert result.source_char_count == HISTORY_REVIEW_MAX_SOURCE_CHARS
        assert result.source_message_count == 1
        assert result.truncated is False

    def test_exact_token_limit_is_allowed(self):
        text = "中" * 2000
        assert estimate_history_review_tokens(text) == 2016
        result = select_history_review_sources([
            HistoryReviewCandidate(1, text),
        ])
        assert result.source_token_estimate == 2016
        assert result.source_message_count == 1

    def test_newest_message_over_char_limit_raises(self):
        with pytest.raises(HistoryReviewSourceMessageTooLarge) as exc:
            select_history_review_sources([
                HistoryReviewCandidate(1, "a" * 4001),
            ])
        assert exc.value.message_id == 1
        assert "a" * 10 not in str(exc.value)

    def test_newest_message_over_token_limit_raises(self):
        with pytest.raises(HistoryReviewSourceMessageTooLarge) as exc:
            select_history_review_sources([
                HistoryReviewCandidate(1, "中" * 3000),
            ])
        assert exc.value.message_id == 1

    def test_identical_inputs_are_deterministic(self):
        candidates = [
            HistoryReviewCandidate(1, "a"),
            HistoryReviewCandidate(2, "b"),
        ]
        first = select_history_review_sources(candidates)
        second = select_history_review_sources(candidates)
        assert first == second
