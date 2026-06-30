"""Resumable translation checkpoint storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

CheckpointLog = Callable[[str], None]

CHECKPOINT_VERSION = 1
CHECKPOINT_POLICIES = {"auto", "ignore", "reset"}
DEFAULT_CHECKPOINT_ROOT = Path.home() / ".daru" / "checkpoints"


@dataclass
class CheckpointBlock:
    id: str
    source_text: str
    translated_text: str
    namespace: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)


class TranslationCheckpointStore:
    """JSON checkpoint that can be updated after every successful batch."""

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
    ) -> None:
        self.path = Path(path).expanduser()
        self.job_type = str(job_type or "").strip().lower()
        self.document_sha256 = str(document_sha256 or "")
        self.source_lang = str(source_lang or "")
        self.target_lang = str(target_lang or "")
        self.translator = str(translator or "")
        self.backend = str(backend or "")
        self._log = log or (lambda _message: None)
        self._blocks: Dict[Tuple[str, str], CheckpointBlock] = {}

        policy = normalize_resume_policy(resume_policy)
        if policy == "reset":
            remove_checkpoint(self.path)
        elif policy == "auto":
            self._load_existing()

    @property
    def blocks(self) -> List[CheckpointBlock]:
        return list(self._blocks.values())

    def set_backend(self, backend: Optional[str]) -> None:
        if backend:
            self.backend = str(backend)

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
        self._blocks[(namespace, block_id)] = CheckpointBlock(
            id=block_id,
            namespace=namespace,
            source_text=str(source_text or ""),
            translated_text=str(translated_text or ""),
            extra=dict(extra or {}),
        )

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
        payload = {
            "version": CHECKPOINT_VERSION,
            "job_type": self.job_type,
            "document_sha256": self.document_sha256,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "translator": self.translator,
            "backend": self.backend,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "blocks": [
                {
                    "id": block.id,
                    "source_text": block.source_text,
                    "translated_text": block.translated_text,
                    "namespace": block.namespace,
                    "extra": block.extra,
                }
                for block in sorted(
                    self._blocks.values(),
                    key=lambda item: (item.namespace, item.id),
                )
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        payload = _read_payload(self.path, self._log)
        if payload is None:
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
    return (root or DEFAULT_CHECKPOINT_ROOT) / f"{safe_stem}.{safe_job}.{path_hash}.checkpoint.json"


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
) -> bool:
    if path is None or not Path(path).expanduser().exists():
        return False
    payload = _read_payload(Path(path).expanduser(), log or (lambda _message: None))
    if payload is None:
        return False
    try:
        document_hash = file_sha256(input_path)
    except OSError:
        return False
    if not _payload_matches(
        payload,
        job_type=str(job_type or "").strip().lower(),
        document_sha256=document_hash,
        source_lang=str(source_lang or ""),
        target_lang=str(target_lang or ""),
    ):
        return False
    blocks = payload.get("blocks")
    return isinstance(blocks, list) and any(
        isinstance(block, dict) and str(block.get("translated_text") or "")
        for block in blocks
    )


def remove_checkpoint(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        Path(path).expanduser().unlink(missing_ok=True)
    except OSError:
        pass


def _payload_matches(
    payload: Dict[str, Any],
    *,
    job_type: str,
    document_sha256: str,
    source_lang: str,
    target_lang: str,
) -> bool:
    return (
        int(payload.get("version", 0) or 0) == CHECKPOINT_VERSION
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
