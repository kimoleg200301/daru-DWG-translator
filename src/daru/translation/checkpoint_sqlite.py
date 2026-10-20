"""Transactional checkpoint persistence using Python's standard-library SQLite.

Connections are short-lived so GUI threads and Windows file operations never
inherit an open handle. A generation guards against stale jobs writing after
another job has reset the checkpoint or changed its translation settings.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

SCHEMA_VERSION = 1


class CheckpointConflictError(RuntimeError):
    """A different job replaced the checkpoint after this job opened it."""


def read_snapshot(path: Path, *, include_blocks: bool = True) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, isolation_level=None)) as connection:
        connection.execute("BEGIN")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            _require_empty_database(connection)
            return None, None  # A first transaction may have been rolled back.
        if version != SCHEMA_VERSION:
            raise sqlite3.DatabaseError(f"Неизвестная версия SQLite checkpoint: {version}")
        row = connection.execute("SELECT payload, generation FROM checkpoint_meta WHERE id = 1").fetchone()
        if row is None:
            return None, None
        payload, generation = json.loads(row[0]), row[1]
        if not isinstance(payload, dict) or not isinstance(generation, str):
            raise ValueError("Некорректные метаданные checkpoint")
        if include_blocks:
            blocks = []
            for namespace, block_id, source, translated, extra in connection.execute(
                "SELECT namespace, block_id, source_text, translated_text, extra FROM checkpoint_blocks ORDER BY rowid"
            ):
                blocks.append({"namespace": namespace, "id": block_id, "source_text": source,
                               "translated_text": translated, "extra": json.loads(extra)})
            payload["blocks"] = blocks
        else:
            payload["has_blocks"] = connection.execute(
                "SELECT 1 FROM checkpoint_blocks WHERE block_id != '' AND translated_text != '' LIMIT 1"
            ).fetchone() is not None
        return payload, generation


def write_batch(
    path: Path,
    metadata: dict[str, Any],
    blocks: Iterable[dict[str, Any]],
    *,
    expected_generation: str | None,
    replace: bool,
) -> str:
    """Commit metadata and all dirty rows together, or roll the entire batch back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, isolation_level=None, timeout=5.0)) as connection:
        # New databases use SQLite's default rollback journal. Do not change a
        # pre-existing file's persistent settings before validating its schema.
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("BEGIN IMMEDIATE")
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise sqlite3.DatabaseError(f"Неизвестная версия SQLite checkpoint: {version}")
            if version == 0:
                _require_empty_database(connection)
            else:
                # Fail before any DDL when the file is not our schema.
                connection.execute("SELECT generation FROM checkpoint_meta LIMIT 1")
                connection.execute("SELECT namespace, block_id, extra FROM checkpoint_blocks LIMIT 1")
            connection.execute("""CREATE TABLE IF NOT EXISTS checkpoint_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1), payload TEXT NOT NULL, generation TEXT NOT NULL
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS checkpoint_blocks (
                namespace TEXT NOT NULL, block_id TEXT NOT NULL, source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL, extra TEXT NOT NULL,
                UNIQUE (namespace, block_id)
            )""")
            current = connection.execute("SELECT generation FROM checkpoint_meta WHERE id = 1").fetchone()
            generation = current[0] if current else None
            if generation != expected_generation:
                raise CheckpointConflictError(
                    "Checkpoint изменён другой задачей. Повторно откройте задание для продолжения."
                )
            if replace or generation is None:
                connection.execute("DELETE FROM checkpoint_blocks")
                generation = uuid4().hex
            connection.executemany(
                """INSERT INTO checkpoint_blocks(namespace, block_id, source_text, translated_text, extra)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT(namespace, block_id) DO UPDATE SET
                   source_text=excluded.source_text, translated_text=excluded.translated_text, extra=excluded.extra""",
                ((block["namespace"], block["id"], block["source_text"], block["translated_text"],
                  json.dumps(block["extra"], ensure_ascii=False)) for block in blocks),
            )
            connection.execute(
                """INSERT INTO checkpoint_meta(id, payload, generation) VALUES (1, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, generation=excluded.generation""",
                (json.dumps(metadata, ensure_ascii=False), generation),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return generation


def _require_empty_database(connection: sqlite3.Connection) -> None:
    if connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
    ).fetchone():
        raise sqlite3.DatabaseError("Файл SQLite содержит данные другого приложения")
