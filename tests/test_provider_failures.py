"""Reject incomplete provider responses before a pipeline can checkpoint them."""

import json
from types import SimpleNamespace

import pytest

from daru.translation.contracts import TranslationBatchError
from daru.translation.engine import TranslationEngine
from daru.translation.providers.standard import DeepLProvider, GoogleHttpProvider


def openai_engine(payload):
    engine = TranslationEngine(provider="noop", source_lang="en")
    engine._backend = "chatgpt"
    engine._translator = {
        "model": "gpt-5.5",
        "mode": "new",
        "client": SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    output_text=json.dumps(payload)
                ),
            )
        ),
    }
    engine._translate_batch = engine._chatgpt_translate_batch
    return engine


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"id": "0:0", "text": "Перевод"}],
        [{"id": "0:0", "text": "Перевод"}, {"id": "0:0", "text": "Дубликат"}],
        [{"id": "0:0", "text": "Перевод"}, {"id": "1:0", "text": ""}],
        [{"id": "0:0", "text": "Перевод"}, {"id": "1:0", "text": 12}],
        [{"id": "0:0", "text": "Перевод"}, {"id": "other", "text": "Лишний"}],
        "not a list",
    ],
)
def test_openai_rejects_missing_duplicate_extra_or_invalid_items(items):
    engine = openai_engine({"translations": items})
    with pytest.raises(TranslationBatchError):
        engine.translate_many(["Inspect the valve", "Close the door"])
    assert engine._glossary == {}
    assert engine._last_originals is None


def test_openai_restores_response_order_and_preserves_intentional_skips():
    engine = openai_engine(
        {
            "translations": [
                {"id": "2:0", "text": "Закройте дверь"},
                {"id": "0:0", "text": "Проверьте клапан"},
            ]
        }
    )
    assert engine.translate_many(
        ["  Inspect the valve  ", " ", "A1", "Close the door"]
    ) == [
        "  Проверьте клапан  ",
        " ",
        "A1",
        "Закройте дверь",
    ]


@pytest.mark.parametrize("value", [None, "", "  ", 12, {"text": "wrong shape"}])
def test_engine_rejects_invalid_value_even_when_batch_length_matches(value):
    engine = TranslationEngine(provider="noop")
    engine._translate_batch = lambda texts: [value]
    with pytest.raises(TranslationBatchError):
        engine.translate_many(["Inspect the valve"])


def test_google_transport_failure_is_not_returned_as_original(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(TranslationBatchError, match="Google"):
        GoogleHttpProvider("en", "ru").translate(["Valve"])


def test_deepl_error_does_not_trigger_one_paid_retry_per_item():
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("quota exceeded")

    with pytest.raises(TranslationBatchError, match="DeepL"):
        DeepLProvider(SimpleNamespace(translate_text=fail), "auto", "ru").translate(
            ["One", "Two"]
        )
    assert len(calls) == 1


def test_deepl_short_response_is_rejected():
    client = SimpleNamespace(
        translate_text=lambda *args, **kwargs: [SimpleNamespace(text="Один")]
    )
    with pytest.raises(TranslationBatchError):
        DeepLProvider(client, "en", "ru").translate(["One", "Two"])


def test_codex_incomplete_inner_batch_is_rejected():
    engine = TranslationEngine(provider="noop")
    engine._backend = "codex-cli"
    engine._translator = SimpleNamespace(translate=lambda *args, **kwargs: ["Один"])
    engine._translate_batch = engine._codex_translate_batch
    with pytest.raises(TranslationBatchError):
        engine.translate_many(["One", "Two"])
