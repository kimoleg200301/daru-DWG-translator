"""Tests for daru.translation.engine — pure utility functions (no API calls)."""

import json

import httpx
from openai import OpenAI

from daru.translation.engine import (
    TranslationEngine,
    _extract_google_free_translation,
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


class TestGoogleBackends:
    def test_google_free_joins_all_response_segments(self):
        payload = [
            [
                ["Первая строка\n", "First line\n"],
                ["Вторая строка", "Second line"],
            ]
        ]

        assert _extract_google_free_translation(payload) == "Первая строка\nВторая строка"

    def test_legacy_provider_names_use_google(self):
        assert TranslationEngine(provider="deep_google").provider == "google"
        assert TranslationEngine(provider="googletrans").provider == "google"

    def test_deep_google_failure_uses_http_fallback(self):
        class FailingTranslator:
            def translate_batch(self, texts):
                raise RuntimeError("network unavailable")

        engine = TranslationEngine(provider="noop")
        engine._translator = FailingTranslator()
        engine._google_free_translate_batch = lambda texts: [f"fallback:{text}" for text in texts]

        assert engine._deep_translate_batch(["Hello"]) == ["fallback:Hello"]


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


class TestOpenAIParameters:
    def test_modern_sdk_targets_responses_endpoint(self):
        captured = []

        def handler(request):
            captured.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "gpt-5.5",
                    "output": [],
                },
            )

        client = OpenAI(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        client.responses.create(
            model="gpt-5.5",
            input=[
                {"role": "developer", "content": "Translate accurately."},
                {"role": "user", "content": "Hello"},
            ],
            text={"format": {"type": "json_object"}, "verbosity": "low"},
            reasoning={"effort": "low"},
            store=True,
        )

        request = captured[0]
        payload = json.loads(request.content)
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert payload["reasoning"] == {"effort": "low"}
        assert payload["text"]["verbosity"] == "low"

    def test_reasoning_model_uses_responses_parameters(self):
        engine = TranslationEngine(
            provider="noop",
            openai_model="gpt-5.5",
            openai_base_url="https://api.openai.com/v1/responses",
            openai_reasoning_effort="xhigh",
            openai_verbosity="low",
        )

        assert engine.openai_base_url == "https://api.openai.com/v1"
        assert engine._openai_generation_kwargs("gpt-5.5", for_responses=True) == {}
        assert engine._openai_responses_reasoning("gpt-5.5") == {"effort": "xhigh"}
        assert engine._openai_responses_text_config("gpt-5.5") == {
            "format": {"type": "json_object"},
            "verbosity": "low",
        }
        assert engine._as_responses_input(
            [
                {"role": "system", "content": "Translate accurately."},
                {"role": "user", "content": "Hello"},
            ]
        ) == [
            {"role": "developer", "content": "Translate accurately."},
            {"role": "user", "content": "Hello"},
        ]

    def test_non_reasoning_model_uses_temperature(self):
        engine = TranslationEngine(
            provider="noop",
            openai_model="gpt-4.1",
            openai_temperature=0.35,
        )

        assert engine._openai_generation_kwargs("gpt-4.1") == {"temperature": 0.35}
        assert engine._openai_responses_reasoning("gpt-4.1") is None
        assert engine._openai_responses_text_config("gpt-4.1") == {
            "format": {"type": "json_object"}
        }
