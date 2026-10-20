"""Resumable translation checkpoint storage."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..config import normalize_language_code
from ..utils.io import write_json_atomic
from ..utils.paths import user_data_dir
from . import checkpoint_sqlite

CheckpointLog = Callable[[str], None]

CHECKPOINT_VERSION = 1
CHECKPOINT_POLICIES = {"auto", "ignore", "reset"}
DEFAULT_CHECKPOINT_ROOT = user_data_dir() / "checkpoints"


@dataclass(frozen=True)
class CheckpointBlock:
    id: str
    source_text: str
    translated_text: str
    namespace: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)


class TranslationCheckpointStore:
    """Incremental SQLite checkpoint, with explicit JSON-path compatibility."""

    def __init__(
        self,
        *,
        path: Path,
        job_type: str,
        document_sha256: str,
        source_lang: str,
        target_lang: str,
        translator: Optional[str] = None,
        backend: Optional[str] = None,
        resume_policy: str = "auto",
        log: Optional[CheckpointLog] = None,
        settings_fingerprint: Optional[str] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.job_type = str(job_type or "").strip().lower()
        self.document_sha256 = str(document_sha256 or "")
        self.source_lang = normalize_language_code(source_lang, "en")
        self.target_lang = normalize_language_code(target_lang, "ru")
        self.settings_fingerprint = settings_fingerprint
        self.translator = str(translator or "")
        self.backend = str(backend or "")
        self._log = log or (lambda _message: None)
        self._blocks: Dict[Tuple[str, str], CheckpointBlock] = {}
        self._dirty: Dict[Tuple[str, str], CheckpointBlock] = {}
        self._sqlite = _is_sqlite_path(self.path)
        self._generation: Optional[str] = None
        self._replace = True
        self._metadata_dirty = True

        policy = normalize_resume_policy(resume_policy)
        if self._sqlite and policy != "auto":
            # Read only the generation: ignore/reset must not import old rows.
            _payload, self._generation = checkpoint_sqlite.read_snapshot(self.path, include_blocks=False)
            if policy == "reset":
                self.save()  # An empty checkpoint also prevents legacy re-import.
        elif policy == "reset":
            remove_checkpoint(self.path)
        elif policy == "auto":
            self._load_existing()

    @property
    def blocks(self) -> List[CheckpointBlock]:
        """Return snapshots; mutations must go through upsert()."""
        return deepcopy(list(self._blocks.values()))

    def set_backend(self, backend: Optional[str]) -> None:
        if backend and str(backend) != self.backend:
            self.backend = str(backend)
            self._metadata_dirty = True

    def get(
        self,
        namespace: str,
        block_id: str,
        *,
        source_text: Optional[str] = None,
    ) -> Optional[str]:
        block = self._blocks.get((namespace, block_id))
        if block is None:
            return None
        if source_text is not None and block.source_text != source_text:
            return None
        translated = str(block.translated_text or "")
        return translated if translated else None

    def text_map(self, namespace: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for block in self._blocks.values():
            if block.namespace != namespace:
                continue
            if block.source_text and block.translated_text:
                result[block.source_text] = block.translated_text
        return result

    def upsert(
        self,
        *,
        namespace: str,
        block_id: str,
        source_text: str,
        translated_text: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        namespace = str(namespace or "default")
        block_id = str(block_id or "")
        if not block_id:
            block_id = stable_text_id(namespace, source_text)
        block = CheckpointBlock(
            id=block_id,
            namespace=namespace,
            source_text=str(source_text or ""),
            translated_text=str(translated_text or ""),
            extra=deepcopy(dict(extra or {})),
        )
        key = (namespace, block_id)
        if self._blocks.get(key) != block:
            self._blocks[key] = block
            self._dirty[key] = block

    def upsert_many(
        self,
        blocks: Iterable[CheckpointBlock],
    ) -> None:
        for block in blocks:
            self.upsert(
                namespace=block.namespace,
                block_id=block.id,
                source_text=block.source_text,
                translated_text=block.translated_text,
                extra=block.extra,
            )

    def save(self) -> None:
        if not self._dirty and not self._metadata_dirty and not self._replace:
            return
        payload = {
            "version": CHECKPOINT_VERSION,
            "job_type": self.job_type,
            "document_sha256": self.document_sha256,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "translator": self.translator,
            "backend": self.backend,
            "settings_fingerprint": self.settings_fingerprint,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._sqlite:
            blocks = self._blocks if self._replace else self._dirty
            self._generation = checkpoint_sqlite.write_batch(
                self.path, payload, (_block_payload(block) for block in blocks.values()),
                expected_generation=self._generation, replace=self._replace,
            )
        else:
            payload["blocks"] = [_block_payload(block) for block in self._blocks.values()]
            write_json_atomic(self.path, payload)
        self._dirty.clear()
        self._replace = False
        self._metadata_dirty = False

    def _load_existing(self) -> None:
        migrate = False
        if self._sqlite:
            try:
                payload, self._generation = checkpoint_sqlite.read_snapshot(self.path)
            except (sqlite3.DatabaseError, ValueError) as exc:
                # Fail before spending translation requests on a job whose
                # checkpoint cannot be persisted safely.
                raise RuntimeError(f"Невозможно прочитать checkpoint {self.path}: {exc}") from exc
            if payload is None:
                legacy_path = self.path.with_suffix(".json")
                payload = _read_payload(legacy_path, self._log) if legacy_path.exists() else None
                migrate = payload is not None
        else:
            payload = _read_payload(self.path, self._log) if self.path.exists() else None
        if payload is None:
            return
        if self.settings_fingerprint is not None and payload.get("settings_fingerprint") != self.settings_fingerprint:
            self._log("Checkpoint: настройки перевода изменились или отсутствуют; начинаем заново")
            return
        if not _payload_matches(
            payload,
            job_type=self.job_type,
            document_sha256=self.document_sha256,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
        ):
            self._log(f"Checkpoint ignored: metadata mismatch ({self.path})")
            return
        self.translator = str(payload.get("translator") or self.translator)
        self.backend = str(payload.get("backend") or self.backend)
        blocks = payload.get("blocks")
        if not isinstance(blocks, list):
            return
        for raw_block in blocks:
            if not isinstance(raw_block, dict):
                continue
            namespace = str(raw_block.get("namespace") or "default")
            block_id = str(raw_block.get("id") or "")
            source_text = str(raw_block.get("source_text") or "")
            translated_text = str(raw_block.get("translated_text") or "")
            if not block_id or not translated_text:
                continue
            extra = raw_block.get("extra")
            self._blocks[(namespace, block_id)] = CheckpointBlock(
                id=block_id,
                namespace=namespace,
                source_text=source_text,
                translated_text=translated_text,
                extra=extra if isinstance(extra, dict) else {},
            )
        if self._blocks:
            self._log(
                f"Checkpoint loaded: {len(self._blocks)} translated blocks ({self.path})"
            )
        if migrate:
            self.save()
            self._log(f"Checkpoint перенесён в SQLite; исходный JSON сохранён: {self.path}")
        else:
            self._replace = False
            self._metadata_dirty = False


def normalize_resume_policy(value: Optional[str]) -> str:
    policy = str(value or "auto").strip().lower()
    return policy if policy in CHECKPOINT_POLICIES else "auto"


def default_checkpoint_path(
    input_path: Path,
    job_type: str,
    *,
    root: Optional[Path] = None,
) -> Path:
    path = Path(input_path).expanduser()
    resolved = str(path.resolve()) if path.exists() else str(path)
    path_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]
    safe_stem = _safe_filename(path.stem or "document")
    safe_job = _safe_filename(str(job_type or "job").lower())
    return (root or DEFAULT_CHECKPOINT_ROOT) / f"{safe_stem}.{safe_job}.{path_hash}.checkpoint.sqlite3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_text_id(namespace: str, text: str) -> str:
    payload = json.dumps(
        {"namespace": namespace, "text": text},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def has_valid_checkpoint(
    path: Optional[Path],
    *,
    input_path: Path,
    job_type: str,
    source_lang: str,
    target_lang: str,
    log: Optional[CheckpointLog] = None,
    settings_fingerprint: Optional[str] = None,
) -> bool:
    if path is None:
        return False
    resolved = Path(path).expanduser()
    logger = log or (lambda _message: None)
    sqlite_has_blocks = None
    if _is_sqlite_path(resolved):
        try:
            payload, _generation = checkpoint_sqlite.read_snapshot(resolved, include_blocks=False)
            if payload is not None:
                sqlite_has_blocks = bool(payload.get("has_blocks"))
        except (sqlite3.DatabaseError, ValueError) as exc:
            logger(f"Checkpoint: невозможно проверить {resolved}: {exc}")
            return False
        if payload is None:
            legacy = resolved.with_suffix(".json")
            payload = _read_payload(legacy, logger) if legacy.exists() else None
    else:
        payload = _read_payload(resolved, logger) if resolved.exists() else None
    if payload is None:
        return False
    if settings_fingerprint is not None and payload.get("settings_fingerprint") != settings_fingerprint:
        return False
    try:
        document_hash = file_sha256(input_path)
    except OSError:
        return False
    if not _payload_matches(
        payload,
        job_type=str(job_type or "").strip().lower(),
        document_sha256=document_hash,
        source_lang=normalize_language_code(source_lang, "en"),
        target_lang=normalize_language_code(target_lang, "ru"),
    ):
        return False
    blocks = payload.get("blocks")
    if sqlite_has_blocks is not None:
        return sqlite_has_blocks
    return isinstance(blocks, list) and any(
        isinstance(block, dict) and str(block.get("translated_text") or "")
        for block in blocks
    )


def remove_checkpoint(path: Optional[Path]) -> None:
    if path is None:
        return
    resolved = Path(path).expanduser()
    if _is_sqlite_path(resolved):
        # Preserve a tombstone instead of deleting an active database/journal;
        # otherwise an old JSON sibling could be imported again on the next run.
        _payload, generation = checkpoint_sqlite.read_snapshot(resolved, include_blocks=False)
        checkpoint_sqlite.write_batch(resolved, {}, (), expected_generation=generation, replace=True)
    else:
        try:
            resolved.unlink(missing_ok=True)
        except OSError:
            pass


def _is_sqlite_path(path: Path) -> bool:
    return path.suffix.lower() in {".sqlite3", ".sqlite", ".db"}


def _block_payload(block: CheckpointBlock) -> Dict[str, Any]:
    return {"id": block.id, "source_text": block.source_text, "translated_text": block.translated_text,
            "namespace": block.namespace, "extra": block.extra}


def _payload_matches(
    payload: Dict[str, Any],
    *,
    job_type: str,
    document_sha256: str,
    source_lang: str,
    target_lang: str,
) -> bool:
    try:
        version = int(payload.get("version", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        version == CHECKPOINT_VERSION
        and str(payload.get("job_type") or "").strip().lower() == job_type
        and str(payload.get("document_sha256") or "") == document_sha256
        and str(payload.get("source_lang") or "") == source_lang
        and str(payload.get("target_lang") or "") == target_lang
    )


def _read_payload(path: Path, log: CheckpointLog) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"Checkpoint ignored: cannot read {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        log(f"Checkpoint ignored: invalid payload {path}")
        return None
    return payload


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:80] or "document"


__all__ = [
    "CheckpointBlock",
    "TranslationCheckpointStore",
    "default_checkpoint_path",
    "file_sha256",
    "has_valid_checkpoint",
    "normalize_resume_policy",
    "remove_checkpoint",
    "stable_text_id",
]
