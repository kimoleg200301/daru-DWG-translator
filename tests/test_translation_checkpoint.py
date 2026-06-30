import json

from daru.translation.checkpoint import (
    TranslationCheckpointStore,
    file_sha256,
    has_valid_checkpoint,
    stable_text_id,
)


def test_checkpoint_loads_matching_payload(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    document_hash = file_sha256(source)

    store = TranslationCheckpointStore(
        path=checkpoint,
        job_type="docx",
        document_sha256=document_hash,
        source_lang="en",
        target_lang="ru",
        translator="fake",
    )
    block_id = stable_text_id("docx:encoded", "Hello")
    store.upsert(
        namespace="docx:encoded",
        block_id=block_id,
        source_text="Hello",
        translated_text="Привет",
    )
    store.save()

    loaded = TranslationCheckpointStore(
        path=checkpoint,
        job_type="docx",
        document_sha256=document_hash,
        source_lang="en",
        target_lang="ru",
    )

    assert loaded.get("docx:encoded", block_id, source_text="Hello") == "Привет"
    assert has_valid_checkpoint(
        checkpoint,
        input_path=source,
        job_type="docx",
        source_lang="en",
        target_lang="ru",
    )


def test_checkpoint_ignores_corrupt_json(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{not-json", encoding="utf-8")
    logs = []

    store = TranslationCheckpointStore(
        path=checkpoint,
        job_type="docx",
        document_sha256=file_sha256(source),
        source_lang="en",
        target_lang="ru",
        log=logs.append,
    )

    assert store.blocks == []
    assert logs
    assert not has_valid_checkpoint(
        checkpoint,
        input_path=source,
        job_type="docx",
        source_lang="en",
        target_lang="ru",
    )


def test_checkpoint_ignores_metadata_mismatch(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "version": 1,
                "job_type": "cad",
                "document_sha256": "different",
                "source_lang": "en",
                "target_lang": "ru",
                "blocks": [
                    {
                        "id": "1",
                        "namespace": "cad",
                        "source_text": "Hello",
                        "translated_text": "Привет",
                        "extra": {},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    store = TranslationCheckpointStore(
        path=checkpoint,
        job_type="cad",
        document_sha256=file_sha256(source),
        source_lang="en",
        target_lang="ru",
    )

    assert store.blocks == []


def test_checkpoint_save_replaces_existing_payload(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Hello", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("old", encoding="utf-8")

    store = TranslationCheckpointStore(
        path=checkpoint,
        job_type="cad",
        document_sha256=file_sha256(source),
        source_lang="en",
        target_lang="ru",
        resume_policy="ignore",
    )
    store.upsert(
        namespace="cad",
        block_id="hello",
        source_text="Hello",
        translated_text="Привет",
    )
    store.save()

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["blocks"][0]["translated_text"] == "Привет"
    assert list(tmp_path.glob("*.tmp")) == []
