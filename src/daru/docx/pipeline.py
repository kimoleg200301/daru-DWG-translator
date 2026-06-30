"""DOCX translation pipeline based on direct OOXML text replacement."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from zipfile import BadZipFile, ZipFile

from ..translation.analysis import CodexAnalysisSession
from ..translation.checkpoint import (
    TranslationCheckpointStore,
    default_checkpoint_path,
    file_sha256,
    stable_text_id,
)
from ..translation.engine import TranslationEngine
from ..utils.io import ensure_parent, paths_equal
from .segments import (
    DocumentModel,
    apply_scheduled_replacements,
    build_document_model,
    decode_formatted_translation,
    decode_plain_translation,
    is_text_part,
    parse_xml_part,
    schedule_unit_replacement,
)

DOCX_SYSTEM_PROMPT = (
    "You are a professional technical document translator. "
    "Translate the provided values from {source_lang} to {target_lang}. "
    "All values belong to the SAME editable DOCX document, so keep terminology consistent. "
    "Preserve numbers, model names, identifiers, URLs, tab characters, and marker tokens "
    "matching '[[DARU_FMT_N]]' and '[[/DARU_FMT_N]]' exactly and in the same order. "
    "Use the full sentence around tab characters as translation context and do not "
    "translate any marker token. "
    "Respond with strict JSON: "
    '{"translations": [{"id": "<id>", "text": "<translated>"}, ...]}'
)


def translate_docx(
    *,
    input_path: Path,
    output_path: Path,
    translator_name: str = "google",
    source_lang: str = "en",
    target_lang: str = "ru",
    deepl_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_project: Optional[str] = None,
    openai_temperature: float = 0.2,
    openai_reasoning_effort: Optional[str] = None,
    openai_verbosity: Optional[str] = None,
    openai_strict_mode: Optional[str] = None,
    openai_strict_value: Optional[float] = None,
    codex_cli_path: Optional[str] = None,
    codex_model: Optional[str] = None,
    codex_reasoning_effort: Optional[str] = None,
    codex_analysis_model: Optional[str] = None,
    codex_analysis_reasoning_effort: Optional[str] = None,
    codex_analysis_session: Optional[CodexAnalysisSession] = None,
    codex_timeout_seconds: int = 300,
    log: Optional[Callable[[str], None]] = None,
    checkpoint_path: Optional[Path] = None,
    resume_policy: str = "auto",
) -> Dict[str, Any]:
    """Translate visible DOCX text without OCR while preserving the OOXML package."""

    logger = log or (lambda _message: None)
    input_path = input_path.expanduser()
    output_path = output_path.expanduser().with_suffix(".docx")

    _validate_paths(input_path, output_path)
    logger(f"DOCX: загружаем документ: {input_path}")

    document_hash = file_sha256(input_path)
    checkpoint_store = TranslationCheckpointStore(
        path=checkpoint_path or default_checkpoint_path(input_path, "docx"),
        job_type="docx",
        document_sha256=document_hash,
        source_lang=source_lang,
        target_lang=target_lang,
        translator=translator_name,
        resume_policy=resume_policy,
        log=logger,
    )
    original_parts, model = _load_document_model(input_path, source_lang)
    logger(
        f"DOCX: найдено {len(model.units)} текстовых фрагментов для перевода, "
        f"пропущено {model.skipped_items}"
    )

    backend_name = "not-required"
    translator: Optional[TranslationEngine] = None

    def get_translator() -> TranslationEngine:
        nonlocal backend_name, translator
        if translator is None:
            translator = TranslationEngine(
                provider=translator_name,
                source_lang=source_lang,
                target_lang=target_lang,
                deepl_auth_key=deepl_key,
                openai_api_key=openai_key,
                openai_model=openai_model,
                openai_base_url=openai_base_url,
                openai_project=openai_project,
                openai_temperature=openai_temperature,
                openai_reasoning_effort=openai_reasoning_effort,
                openai_verbosity=openai_verbosity,
                openai_strict_mode=openai_strict_mode,
                openai_strict_value=openai_strict_value,
                codex_cli_path=codex_cli_path,
                codex_model=codex_model,
                codex_reasoning_effort=codex_reasoning_effort,
                codex_analysis_model=codex_analysis_model,
                codex_analysis_reasoning_effort=codex_analysis_reasoning_effort,
                codex_analysis_session=codex_analysis_session,
                codex_timeout_seconds=codex_timeout_seconds,
                system_prompt_template=DOCX_SYSTEM_PROMPT,
            )
            translator.set_document_context(
                [unit.source_text for unit in model.units],
                context_label="DOCX DOCUMENT",
            )
            backend_name = translator.backend_name()
            checkpoint_store.set_backend(backend_name)
            logger(f"DOCX: инициализирован движок перевода: {backend_name}")
        return translator

    if model.units:
        backend_name = _translate_units(model, get_translator, logger, checkpoint_store)

    modified_parts = apply_scheduled_replacements(model)
    _write_atomic_docx(input_path, output_path, original_parts, modified_parts)
    logger(f"DOCX: файл сохранен: {output_path}")

    return {
        "output_path": output_path,
        "backend": backend_name,
        "items_translated": len(model.units),
        "items_skipped": model.skipped_items,
        "checkpoint_path": checkpoint_store.path,
    }


def _validate_paths(input_path: Path, output_path: Path) -> None:
    if input_path.suffix.lower() != ".docx":
        raise RuntimeError("Поддерживаются только входные файлы DOCX")
    if not input_path.exists() or not input_path.is_file():
        raise RuntimeError(f"Входной DOCX-файл не найден: {input_path}")
    if paths_equal(input_path, output_path):
        raise RuntimeError("Исходный и выходной DOCX-файлы должны отличаться")


def _load_document_model(
    input_path: Path,
    source_lang: str,
) -> tuple[Dict[str, bytes], DocumentModel]:
    original_parts: Dict[str, bytes] = {}
    parsed_parts = {}
    try:
        with ZipFile(input_path, "r") as archive:
            names = archive.namelist()
            if "word/document.xml" not in names:
                raise RuntimeError("Файл не содержит word/document.xml и не является DOCX")
            for name in names:
                data = archive.read(name)
                original_parts[name] = data
                if is_text_part(name):
                    parsed_parts[name] = parse_xml_part(name, data)
    except BadZipFile as exc:
        raise RuntimeError(f"Некорректный DOCX ZIP-контейнер: {input_path}") from exc

    return original_parts, build_document_model(parsed_parts, source_lang)


def _translate_units(
    model: DocumentModel,
    translator_factory: Callable[[], TranslationEngine],
    logger: Callable[[str], None],
    checkpoint_store: TranslationCheckpointStore,
) -> str:
    encoded_values = _ordered_unique(unit.encoded_text for unit in model.units)
    translations = _translate_with_progress(
        translator_factory,
        encoded_values,
        logger,
        prefix="DOCX: перевод",
        checkpoint_store=checkpoint_store,
        namespace="docx:encoded",
    )
    translated_by_source = dict(zip(encoded_values, translations))

    decoded_by_unit: Dict[int, Optional[List[str]]] = {}
    fallback_values: List[str] = []
    for index, unit in enumerate(model.units):
        translated = translated_by_source.get(unit.encoded_text, unit.encoded_text)
        decoded = decode_formatted_translation(unit, translated)
        decoded_by_unit[index] = decoded
        if decoded is None:
            fallback_values.append(unit.source_text)

    fallback_map: Dict[str, str] = {}
    unique_fallback_values = _ordered_unique(fallback_values)
    if unique_fallback_values:
        logger(
            "DOCX: переводчик изменил служебные маркеры; "
            "повторяем цельные фразы без служебных маркеров"
        )
        fallback_translations = _translate_with_progress(
            translator_factory,
            unique_fallback_values,
            logger,
            prefix="DOCX: резервный перевод",
            checkpoint_store=checkpoint_store,
            namespace="docx:fallback",
        )
        fallback_map = dict(zip(unique_fallback_values, fallback_translations))

    for index, unit in enumerate(model.units):
        decoded = decoded_by_unit[index]
        if decoded is None:
            decoded = decode_plain_translation(
                unit,
                fallback_map.get(unit.source_text, unit.source_text),
            )
        schedule_unit_replacement(model, unit, decoded)
    return checkpoint_store.backend or "cached-checkpoint"


def _translate_with_progress(
    translator_factory: Callable[[], TranslationEngine],
    values: Sequence[str],
    logger: Callable[[str], None],
    *,
    prefix: str,
    checkpoint_store: TranslationCheckpointStore,
    namespace: str,
) -> List[str]:
    if not values:
        return []
    result: List[Optional[str]] = [None] * len(values)
    pending_values: List[str] = []
    pending_indices: List[int] = []
    cached_count = 0
    for index, value in enumerate(values):
        cached = checkpoint_store.get(
            namespace,
            stable_text_id(namespace, value),
            source_text=value,
        )
        if cached is not None:
            result[index] = cached
            cached_count += 1
        else:
            pending_values.append(value)
            pending_indices.append(index)

    logger(f"{prefix}... [0%]")
    if not pending_values:
        logger(f"{prefix}... [100%]")
        return [item if item is not None else "" for item in result]

    translator = translator_factory()
    batch_size = max(1, min(50, (len(values) + 19) // 20))
    processed = cached_count
    for start in range(0, len(pending_values), batch_size):
        batch = pending_values[start : start + batch_size]
        batch_indices = pending_indices[start : start + batch_size]
        translated = translator.translate_many(batch)
        if len(translated) != len(batch):
            raise RuntimeError("Движок перевода вернул неверное количество DOCX-фрагментов")
        for value, translated_value, result_index in zip(batch, translated, batch_indices):
            result[result_index] = translated_value
            checkpoint_store.upsert(
                namespace=namespace,
                block_id=stable_text_id(namespace, value),
                source_text=value,
                translated_text=translated_value,
            )
        checkpoint_store.save()
        processed = min(len(values), processed + len(translated))
        percent = min(100, int(processed / len(values) * 100))
        logger(f"{prefix}... [{percent}%]")
    return [item if item is not None else "" for item in result]


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _write_atomic_docx(
    input_path: Path,
    output_path: Path,
    original_parts: Dict[str, bytes],
    modified_parts: Dict[str, bytes],
) -> None:
    ensure_parent(output_path)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".docx",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

        with ZipFile(input_path, "r") as source_archive, ZipFile(
            temporary_path,
            "w",
        ) as output_archive:
            for info in source_archive.infolist():
                data = modified_parts.get(info.filename, original_parts[info.filename])
                output_archive.writestr(info, data)

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["translate_docx"]
