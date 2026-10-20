"""Provider boundary and malformed-checkpoint regression tests."""

import json

import pytest

from daru.translation.checkpoint import (
    TranslationCheckpointStore,
    file_sha256,
    has_valid_checkpoint,
)
from daru.translation.engine import TranslationEngine, chunked


@pytest.mark.parametrize("response", [[], ["one"], ["one", "two", "three"]])
def test_engine_rejects_incomplete_or_extra_translations(response):
    engine = TranslationEngine(provider="noop")
    engine._backend = "chatgpt"
    engine._translate_batch = lambda _texts: response

    with pytest.raises(RuntimeError, match="неверное количество фрагментов"):
        engine.translate_many(["a", "b"])
    assert engine._last_originals is None


def test_noop_preserves_whitespace_only_values():
    texts = ["", "  ", "\t\n", " hello "]
    assert TranslationEngine(provider="noop").translate_many(texts) == texts


@pytest.mark.parametrize("size", [0, -1])
def test_chunk_size_must_be_positive(size):
    with pytest.raises(ValueError, match="положительным"):
        list(chunked(["text"], size))


@pytest.mark.parametrize("version", ["broken", {}, [], None, 2])
def test_malformed_checkpoint_version_is_ignored(tmp_path, version):
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({
        "version": version, "job_type": "cad", "document_sha256": file_sha256(source),
        "source_lang": "en", "target_lang": "ru", "blocks": [],
    }), encoding="utf-8")

    store = TranslationCheckpointStore(
        path=checkpoint, job_type="cad", document_sha256=file_sha256(source),
        source_lang="en", target_lang="ru",
    )

    assert store.blocks == []
    assert not has_valid_checkpoint(
        checkpoint, input_path=source, job_type="cad", source_lang="en", target_lang="ru",
    )
