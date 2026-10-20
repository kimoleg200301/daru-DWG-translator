"""Durability, incremental writes and backwards-compatible migration."""

import json
import sqlite3
import subprocess
import sys

import pytest

from daru.translation import checkpoint_sqlite
from daru.translation.checkpoint import (
    TranslationCheckpointStore,
    default_checkpoint_path,
    file_sha256,
    has_valid_checkpoint,
    remove_checkpoint,
)


@pytest.fixture
def options(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Source", encoding="utf-8")
    return dict(path=tmp_path / "source.checkpoint.sqlite3", job_type="cad",
                document_sha256=file_sha256(source), source_lang="en", target_lang="ru",
                settings_fingerprint="settings")


def add(store, key, value=None, **extra):
    store.upsert(namespace="text", block_id=key, source_text=f"source:{key}",
                 translated_text=value or f"translation:{key}", extra=extra)


def sql(path, statement):
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute(statement)
    finally:
        connection.close()


def test_sqlite_updates_only_changed_rows_and_does_not_rewrite_unchanged_batch(options):
    store = TranslationCheckpointStore(**options)
    add(store, "old")
    store.save()
    sql(store.path, """CREATE TRIGGER protect_old BEFORE INSERT ON checkpoint_blocks
        WHEN NEW.block_id = 'old' BEGIN SELECT RAISE(ABORT, 'old row was rewritten'); END""")
    unchanged = store.path.read_bytes()
    add(store, "old")
    store.save()
    assert store.path.read_bytes() == unchanged
    add(store, "new")
    store.save()
    assert len(TranslationCheckpointStore(**options).blocks) == 2


def test_failed_batch_rolls_back_and_remains_retryable(options):
    store = TranslationCheckpointStore(**options)
    add(store, "committed")
    store.save()
    sql(store.path, """CREATE TRIGGER fail_batch BEFORE INSERT ON checkpoint_blocks
        WHEN NEW.block_id = 'fail' BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
    add(store, "first")
    add(store, "fail")
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        store.save()
    assert [block.id for block in TranslationCheckpointStore(**options).blocks] == ["committed"]
    sql(store.path, "DROP TRIGGER fail_batch")
    store.save()
    assert len(TranslationCheckpointStore(**options).blocks) == 3


def test_uncommitted_transaction_recovers_after_process_exit(options):
    store = TranslationCheckpointStore(**options)
    add(store, "saved")
    store.save()
    code = """import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute('BEGIN IMMEDIATE')
connection.execute("UPDATE checkpoint_blocks SET translated_text = 'UNCOMMITTED'")
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", code, str(store.path)], capture_output=True, timeout=10)
    assert result.returncode == 17
    assert TranslationCheckpointStore(**options).get("text", "saved") == "translation:saved"


def test_default_path_migrates_matching_json_without_changing_original(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Source", encoding="utf-8")
    path = default_checkpoint_path(source, "cad", root=tmp_path)
    assert path.suffix == ".sqlite3"
    options = dict(path=path, job_type="cad", document_sha256=file_sha256(source),
                   source_lang="en", target_lang="ru", settings_fingerprint="fp")
    legacy = TranslationCheckpointStore(**dict(options, path=path.with_suffix(".json")))
    add(legacy, "old", bbox=[1, 2, 3, 4])
    legacy.save()
    original = legacy.path.read_bytes()
    # The GUI's read-only check must detect migration candidates without writing.
    assert has_valid_checkpoint(path, input_path=source, job_type="cad", source_lang="en",
                                target_lang="ru", settings_fingerprint="fp")
    assert not path.exists()
    migrated = TranslationCheckpointStore(**options)
    assert migrated.blocks == legacy.blocks
    assert path.read_bytes().startswith(b"SQLite format 3")
    assert legacy.path.read_bytes() == original
    add(migrated, "new")
    migrated.save()
    assert len(TranslationCheckpointStore(**options).blocks) == 2


@pytest.mark.parametrize("change", [{"target_lang": "fr"}, {"settings_fingerprint": "new"}, {"document_sha256": "new"}])
def test_incompatible_json_is_not_migrated(options, change):
    legacy = TranslationCheckpointStore(**dict(options, path=options["path"].with_suffix(".json")))
    add(legacy, "old")
    legacy.save()
    fresh = TranslationCheckpointStore(**dict(options, **change))
    assert not fresh.blocks
    assert not fresh.path.exists()
    add(fresh, "new")
    fresh.save()
    assert [block.id for block in TranslationCheckpointStore(**dict(options, **change)).blocks] == ["new"]


def test_generation_rejects_stale_writer_after_reset(options):
    first = TranslationCheckpointStore(**options)
    add(first, "old")
    first.save()
    restarted = TranslationCheckpointStore(**dict(options, resume_policy="reset"))
    assert not TranslationCheckpointStore(**options).blocks
    add(first, "late")
    with pytest.raises(checkpoint_sqlite.CheckpointConflictError):
        first.save()
    add(restarted, "new")
    restarted.save()
    assert [block.id for block in TranslationCheckpointStore(**options).blocks] == ["new"]


def test_same_generation_writers_merge_disjoint_batches(options):
    first = TranslationCheckpointStore(**options)
    add(first, "one")
    first.save()
    second = TranslationCheckpointStore(**options)
    add(first, "two")
    first.save()
    add(second, "three")
    second.save()
    assert {block.id for block in TranslationCheckpointStore(**options).blocks} == {"one", "two", "three"}


def test_ignore_replaces_rows_only_after_first_successful_save(options):
    original = TranslationCheckpointStore(**options)
    add(original, "old")
    original.save()
    ignored = TranslationCheckpointStore(**dict(options, resume_policy="ignore"))
    assert not ignored.blocks
    assert TranslationCheckpointStore(**options).get("text", "old")
    add(ignored, "new")
    ignored.save()
    assert [block.id for block in TranslationCheckpointStore(**options).blocks] == ["new"]


def test_remove_and_reset_do_not_resurrect_legacy_json(options):
    legacy = TranslationCheckpointStore(**dict(options, path=options["path"].with_suffix(".json")))
    add(legacy, "old")
    legacy.save()
    store = TranslationCheckpointStore(**options)
    remove_checkpoint(store.path)
    assert not TranslationCheckpointStore(**options).blocks
    assert legacy.path.exists()
    assert not has_valid_checkpoint(store.path, input_path=store.path.parent / "source.txt",
                                    job_type="cad", source_lang="en", target_lang="ru")


def test_metadata_check_does_not_decode_all_block_extras(options):
    store = TranslationCheckpointStore(**options)
    add(store, "row")
    store.save()
    sql(store.path, "UPDATE checkpoint_blocks SET extra = 'not JSON'")
    assert has_valid_checkpoint(store.path, input_path=store.path.parent / "source.txt",
                                job_type="cad", source_lang="en", target_lang="ru")


@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
def test_json_path_stays_json_and_sqlite_does_not_modify_foreign_database(options, journal_mode):
    legacy = TranslationCheckpointStore(**dict(options, path=options["path"].with_suffix(".json")))
    add(legacy, "row")
    legacy.save()
    assert json.loads(legacy.path.read_text(encoding="utf-8"))["version"] == 1
    sql(options["path"], "CREATE TABLE valuable_data (value TEXT)")
    sql(options["path"], f"PRAGMA journal_mode = {journal_mode}")
    original = options["path"].read_bytes()
    with pytest.raises(RuntimeError, match="другого приложения"):
        TranslationCheckpointStore(**options)
    # The backend must also reject a write without changing persistent settings.
    with pytest.raises(sqlite3.DatabaseError, match="другого приложения"):
        checkpoint_sqlite.write_batch(options["path"], {}, (), expected_generation=None, replace=True)
    assert options["path"].read_bytes() == original


def test_corrupt_sqlite_fails_when_job_opens_before_any_translation(options):
    options["path"].write_bytes(b"broken checkpoint")
    with pytest.raises(RuntimeError, match="Невозможно прочитать checkpoint"):
        TranslationCheckpointStore(**options)
    assert options["path"].read_bytes() == b"broken checkpoint"


def test_block_metadata_is_a_snapshot(options):
    store = TranslationCheckpointStore(**options)
    values = [1]
    add(store, "row", values=values)
    values.append(2)
    store.blocks[0].extra["values"].append(3)
    store.save()
    assert TranslationCheckpointStore(**options).blocks[0].extra == {"values": [1]}
