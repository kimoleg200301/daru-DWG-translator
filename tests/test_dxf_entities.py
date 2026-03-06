"""Tests for daru.dxf.entities — shared DXF entity helpers."""

from daru.dxf.entities import (
    ENTITY_TARGETS,
    get_dim_override_text,
    protect_dim,
    replace_all_exact,
    safe_mtext_text,
    safe_set_mtext,
    translate_text_keep_dim_and_mtext_controls,
)


class TestEntityTargets:
    def test_contains_expected(self):
        assert "TEXT" in ENTITY_TARGETS
        assert "MTEXT" in ENTITY_TARGETS
        assert "DIMENSION" in ENTITY_TARGETS
        assert "TABLE" in ENTITY_TARGETS


class FakeDim:
    def __init__(self, text):
        self._text = text

    def get_text(self):
        return self._text

    def set_text(self, s):
        self._text = s


class FakeMtext:
    def __init__(self, text):
        self.text = text

    def plain_text(self):
        return self.text


class TestGetDimOverrideText:
    def test_normal(self):
        assert get_dim_override_text(FakeDim("100mm")) == "100mm"

    def test_placeholder(self):
        assert get_dim_override_text(FakeDim("<>")) == ""

    def test_empty(self):
        assert get_dim_override_text(FakeDim("")) == ""

    def test_exception(self):
        class Bad:
            def get_text(self):
                raise RuntimeError("broken")
        assert get_dim_override_text(Bad()) == ""


class TestSafeMtextText:
    def test_plain_text_method(self):
        assert safe_mtext_text(FakeMtext("hello")) == "hello"

    def test_empty(self):
        assert safe_mtext_text(FakeMtext("")) == ""


class TestSafeSetMtext:
    def test_set(self):
        m = FakeMtext("old")
        safe_set_mtext(m, "new")
        assert m.text == "new"


class TestProtectDim:
    def test_no_placeholder(self):
        parts, sep = protect_dim("hello world")
        assert parts == ["hello world"]
        assert sep == "<>"

    def test_with_placeholder(self):
        parts, sep = protect_dim("A<>B")
        assert parts == ["A", "B"]
        assert sep == "<>"


class TestReplaceAllExact:
    def test_basic(self):
        assert replace_all_exact("hello world", [("hello", "привет")]) == "привет world"

    def test_multiple(self):
        result = replace_all_exact("a b c", [("a", "x"), ("c", "z")])
        assert result == "x b z"

    def test_empty_source(self):
        assert replace_all_exact("hello", [("", "x")]) == "hello"


class TestTranslateTextKeepDimAndMtextControls:
    def test_empty(self):
        assert translate_text_keep_dim_and_mtext_controls("", []) == ""

    def test_simple_replacement(self):
        result = translate_text_keep_dim_and_mtext_controls(
            "hello world", [("hello", "привет"), ("world", "мир")]
        )
        assert result == "привет мир"

    def test_preserves_dim_placeholder(self):
        result = translate_text_keep_dim_and_mtext_controls(
            "size<>mm", [("size", "размер"), ("mm", "мм")]
        )
        assert "размер" in result
        assert "<>" in result
        assert "мм" in result
