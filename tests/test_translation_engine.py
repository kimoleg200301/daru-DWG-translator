"""Tests for daru.translation.engine — pure utility functions (no API calls)."""

from daru.translation.engine import (
    chunked,
    prepare_for_translation,
    recover_after_translation,
    restore_edge_whitespace,
    DIM_PLACEHOLDER,
)


class TestChunked:
    def test_exact_division(self):
        result = list(chunked(["a", "b", "c", "d"], 2))
        assert result == [["a", "b"], ["c", "d"]]

    def test_remainder(self):
        result = list(chunked(["a", "b", "c"], 2))
        assert result == [["a", "b"], ["c"]]

    def test_empty(self):
        assert list(chunked([], 5)) == []

    def test_single_chunk(self):
        result = list(chunked(["a", "b"], 10))
        assert result == [["a", "b"]]


class TestRestoreEdgeWhitespace:
    def test_preserves_leading(self):
        assert restore_edge_whitespace("  hello", "привет") == "  привет"

    def test_preserves_trailing(self):
        assert restore_edge_whitespace("hello  ", "привет") == "привет  "

    def test_preserves_both(self):
        assert restore_edge_whitespace("  hello  ", "привет") == "  привет  "

    def test_no_whitespace(self):
        assert restore_edge_whitespace("hello", "привет") == "привет"

    def test_empty_translated(self):
        assert restore_edge_whitespace("hello", "") == ""


class TestPrepareForTranslation:
    def test_replaces_mtext_newlines(self):
        assert prepare_for_translation("A\\PB") == "A\nB"

    def test_replaces_dim_placeholder(self):
        result = prepare_for_translation("size<>mm")
        assert DIM_PLACEHOLDER in result
        assert "<>" not in result

    def test_empty(self):
        assert prepare_for_translation("") == ""


class TestRecoverAfterTranslation:
    def test_restores_dim_placeholder(self):
        prepared = prepare_for_translation("X<>Y")
        result = recover_after_translation("X<>Y", prepared)
        assert "<>" in result

    def test_restores_mtext_newlines(self):
        original = "A\\PB"
        translated = "X\nY"
        result = recover_after_translation(original, translated)
        assert "\\P" in result

    def test_empty(self):
        assert recover_after_translation("hello", "") == ""

    def test_preserves_whitespace(self):
        result = recover_after_translation("  hello  ", "  привет  ")
        assert result.startswith("  ")
        assert result.endswith("  ")
