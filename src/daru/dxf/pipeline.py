"""DXF/DWG translation pipeline: extract → translate → apply → save."""


from ..translation.progress import report_progress
import argparse
import sys
import tempfile
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple

import ezdxf

from ..translation.analysis import CodexAnalysisSession
from ..translation.checkpoint import (
    TranslationCheckpointStore,
    default_checkpoint_path,
    file_sha256,
    stable_text_id,
)
from ..translation.engine import TranslationEngine
from ..translation.job import check_cancelled
from ..translation.options import TranslationOptions
from ..utils.io import (
    ensure_parent,
    read_map_csv,
    write_map_csv,
    write_translated_txt,
)
from ..utils.odafc import convert_with_odafc
from ..utils.spinner import Spinner, colorize
from . import applier
from . import extractor
from .entities import TextReplacer

DEFAULT_MAP_CACHE = Path("map_auto.csv")


def _determine_output_format(output_format: Optional[str], output_path: Path) -> str:
    if output_format:
        fmt = output_format.lower()
    else:
        suffix = output_path.suffix.lower()
        fmt = "dwg" if suffix == ".dwg" else "dxf"
    if fmt not in {"dwg", "dxf"}:
        raise RuntimeError("Поддерживаются только форматы DWG и DXF")
    return fmt


def _prepare_input_file(path: Path, logger: Callable[[str], None], stack: ExitStack) -> Path:
    ext = path.suffix.lower()
    if ext == ".dxf":
        return path
    if ext == ".dwg":
        logger("Преобразуем DWG в DXF для обработки...")
        temp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="daru_in_")))
        dxf_path = temp_dir / f"{path.stem}_source.dxf"
        convert_with_odafc(path, dxf_path, logger)
        return dxf_path
    raise RuntimeError("Поддерживаются только входные файлы с расширением DWG или DXF")


def _prepare_output_destination(path: Path, fmt: str) -> Path:
    suffix = ".dwg" if fmt == "dwg" else ".dxf"
    return path.with_suffix(suffix)


def apply_translations(
    doc,
    pairs: List[Tuple[str, str]],
    style_font: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    resolver = applier.FontStyleResolver(doc, style_font, log)
    replacer = TextReplacer(pairs)
    ru_style = ""
    for layout in [doc.modelspace(), *[l for l in doc.layouts if l.name != "Model"]]:
        applier.walk_layout(layout, replacer, ru_style, resolver)
    applier.walk_blocks(doc, replacer, ru_style, resolver)


def translate_dxf(
    *,
    input_path: Path,
    output_path: Path,
    translator_name: str = "google",
    source_lang: str = "en",
    target_lang: str = "ru",
    style_font: Optional[str] = None,
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
    output_format: Optional[str] = None,
    map_path: Optional[Path] = None,
    glossary_path: Optional[Path] = None,
    save_map: bool = True,
    extracted_txt_path: Optional[Path] = None,
    translated_txt_path: Optional[Path] = None,
    save_txt: bool = True,
    log: Optional[Callable[[str], None]] = None,
    translation_context_factory: Optional[Callable[[str], ContextManager[Any]]] = None,
    checkpoint_path: Optional[Path] = None,
    resume_policy: str = "auto",
) -> Dict[str, Any]:
    options = TranslationOptions.from_mapping(locals())
    report_progress("read")
    logger = log or (lambda message: None)

    input_path = input_path.expanduser()
    output_path = output_path.expanduser()

    final_format = _determine_output_format(output_format, output_path)

    with ExitStack() as stack:
        processing_input = _prepare_input_file(input_path, logger, stack)

        logger(f"Загружаем чертёж: {processing_input}")
        doc = ezdxf.readfile(str(processing_input))

        freq, entity_types = extractor.extract_text_counts_and_types(doc)
    document_hash = file_sha256(input_path)
    sorted_items = extractor.sort_frequency(freq)
    english_texts = [text for text, _ in sorted_items]
    csv_cache = read_map_csv(glossary_path)
    checkpoint_store = TranslationCheckpointStore(
        path=checkpoint_path or default_checkpoint_path(input_path, "cad"),
        job_type="cad",
        document_sha256=document_hash,
        source_lang=source_lang,
        target_lang=target_lang,
        translator=translator_name,
        settings_fingerprint=options.fingerprint("cad", glossary=csv_cache),
        resume_policy=resume_policy,
        log=logger,
    )
    logger(f"Найдено {len(english_texts)} текстовых элементов для перевода")
    checkpoint_cache = checkpoint_store.text_map("cad")
    translation_cache = dict(checkpoint_cache)
    if checkpoint_cache:
        logger(f"Checkpoint: loaded {len(checkpoint_cache)} CAD translations")
    if csv_cache and glossary_path:
        logger(f"Загружено {len(csv_cache)} переводов из выбранного глоссария {glossary_path}")
        translation_cache.update(csv_cache)

    backend_name = "cached-checkpoint" if checkpoint_cache else "cached-map"

    context_factory = translation_context_factory or (lambda _msg: nullcontext())
    logger("Начинаем перевод... [0%]")
    total_items = len(english_texts)
    translation_slots: List[Optional[str]] = [None] * total_items
    text_to_indices: Dict[str, List[int]] = {}
    cached_hits = 0
    for idx, text in enumerate(english_texts):
        cached_value = translation_cache.get(text) if translation_cache else None
        if cached_value is not None:
            translation_slots[idx] = cached_value
            cached_hits += 1
        else:
            bucket = text_to_indices.setdefault(text, [])
            bucket.append(idx)
    unique_pending = list(text_to_indices.keys())
    report_progress("translate", cached_hits, total_items)
    if total_items == 0 or not unique_pending:
        logger("Начинаем перевод... [100%]")
    else:
        translator = TranslationEngine(**options.engine_kwargs())
        backend_name = translator.backend_name()
        checkpoint_store.set_backend(backend_name)
        logger(f"Инициализирован движок перевода: {backend_name}")
        translator.set_drawing_context(english_texts, entity_types=entity_types)
        processed = cached_hits
        batch_size = max(1, len(unique_pending) // 20)
        with context_factory("Переводим..."):
            for start_idx in range(0, len(unique_pending), batch_size):
                check_cancelled()
                chunk_texts = unique_pending[start_idx : start_idx + batch_size]
                translated_chunk = translator.translate_many(chunk_texts)
                for original_text, translated_text in zip(chunk_texts, translated_chunk):
                    translation_cache[original_text] = translated_text
                    checkpoint_store.upsert(
                        namespace="cad",
                        block_id=stable_text_id("cad", original_text),
                        source_text=original_text,
                        translated_text=translated_text,
                    )
                    for text_index in text_to_indices.get(original_text, []):
                        translation_slots[text_index] = translated_text
                        processed += 1
                checkpoint_store.save()
                report_progress("translate", processed, total_items)
                percent = min(100, int(processed / total_items * 100)) if total_items else 100
                logger(f"Начинаем перевод... [{percent}%]")
        if processed < total_items:
            logger("Начинаем перевод... [100%]")
    russian_texts: List[str] = []
    for idx, original in enumerate(english_texts):
        translated_value = translation_slots[idx] if idx < len(translation_slots) else None
        if not translated_value:
            translated_value = original
        russian_texts.append(translated_value)
    logger("Перевод завершён")

    if save_txt and extracted_txt_path:
        ensure_parent(extracted_txt_path)
        extractor.write_txt(freq, extracted_txt_path)
        logger(f"Сохранён TXT с исходными текстами: {extracted_txt_path}")
    if save_txt and translated_txt_path:
        ensure_parent(translated_txt_path)
        write_translated_txt(translated_txt_path, sorted_items, russian_texts)
        logger(f"Сохранён TXT с переводами: {translated_txt_path}")

    if save_map and map_path:
        ensure_parent(map_path)
        write_map_csv(map_path, english_texts, russian_texts)
        logger(f"Сохранена карта переводов CSV: {map_path}")

    pairs = list(zip(english_texts, russian_texts))
    pairs.sort(key=lambda x: -len(x[0]))
    logger("Применяем переводы к DXF")
    report_progress("apply")
    apply_translations(doc, pairs, style_font=style_font, log=logger)

    final_output_path = _prepare_output_destination(output_path, final_format)
    ensure_parent(final_output_path)

    with ExitStack() as stack2:
        report_progress("export")
        if final_format == "dwg":
            temp_dir = Path(stack2.enter_context(tempfile.TemporaryDirectory(prefix="daru_out_")))
            dxf_output = temp_dir / (final_output_path.stem + ".dxf")
        else:
            dxf_output = final_output_path if final_output_path.suffix.lower() == ".dxf" else final_output_path.with_suffix(".dxf")

        doc.saveas(str(dxf_output))

        if final_format == "dwg":
            logger("Преобразуем результат в DWG...")
            convert_with_odafc(dxf_output, final_output_path, logger)
            logger(f"DWG сохранён: {final_output_path}")
        else:
            logger(f"DXF сохранён: {final_output_path}")

    return {
        "output_path": final_output_path,
        "backend": backend_name,
        "map_saved": bool(save_map and map_path),
        "extracted_txt_saved": bool(save_txt and extracted_txt_path),
        "translated_txt_saved": bool(save_txt and translated_txt_path),
        "items_translated": len(english_texts),
        "checkpoint_path": checkpoint_store.path,
    }


def run_pipeline(args: argparse.Namespace) -> None:
    input_path = Path(args.input_dxf)
    output_path = Path(args.output_dxf)
    map_path = Path(args.map_csv) if args.map_csv else None
    extracted_txt_path = Path(args.extracted_txt) if args.extracted_txt else None
    translated_txt_path = Path(args.translated_txt) if args.translated_txt else None

    def cli_log(message: str) -> None:
        print(colorize(message, "36")) if message else None

    output_format_cli = (args.output_format or "dwg").lower()
    if output_format_cli not in {"dwg", "dxf"}:
        raise SystemExit("Недопустимый формат вывода. Используйте dwg или dxf")
    output_path = output_path.with_suffix(".dwg" if output_format_cli == "dwg" else ".dxf")

    try:
        result = translate_dxf(
            input_path=input_path,
            output_path=output_path,
            translator_name=args.translator,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            style_font=args.style_font,
            deepl_key=getattr(args, "deepl_key", None),
            openai_key=getattr(args, "openai_key", None),
            openai_model=getattr(args, "openai_model", None),
            openai_base_url=getattr(args, "openai_base_url", None),
            openai_project=getattr(args, "openai_project", None),
            openai_temperature=getattr(args, "openai_temperature", 0.2),
            openai_reasoning_effort=getattr(args, "openai_reasoning_effort", None),
            openai_verbosity=getattr(args, "openai_verbosity", None),
            openai_strict_mode=getattr(args, "openai_strict_mode", None),
            openai_strict_value=getattr(args, "openai_strict_value", None),
            codex_cli_path=getattr(args, "codex_cli_path", None),
            codex_model=getattr(args, "codex_model", None),
            codex_reasoning_effort=getattr(args, "codex_reasoning_effort", None),
            codex_analysis_model=getattr(args, "codex_analysis_model", None),
            codex_analysis_reasoning_effort=getattr(
                args,
                "codex_analysis_reasoning_effort",
                None,
            ),
            codex_timeout_seconds=getattr(args, "codex_timeout_seconds", 300),
            output_format=output_format_cli,
            map_path=map_path,
            glossary_path=Path(args.glossary_csv) if args.glossary_csv else None,
            save_map=not args.no_map,
            extracted_txt_path=extracted_txt_path,
            translated_txt_path=translated_txt_path,
            save_txt=not args.skip_txt,
            log=cli_log,
            resume_policy="reset" if args.restart_translation else "auto",
            translation_context_factory=(
                (lambda msg: Spinner(colorize(msg, "30"), enabled=sys.stdout.isatty()))
                if sys.stdout.isatty()
                else None
            ),
        )
    except RuntimeError as exc:
        print(colorize(f"Ошибка: {exc}", "31"))
        raise SystemExit(3)

    print(colorize("Saved:", "32"), colorize(str(result["output_path"]), "36"))
    print(colorize("Translator backend:", "32"), colorize(result["backend"], "36"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract, translate and reapply text for DWG/DXF files.")
    parser.add_argument("input_dxf", help="Входной DWG/DXF файл")
    parser.add_argument("output_dxf", help="Выходной DWG/DXF файл")
    parser.add_argument("--map-csv", default=str(DEFAULT_MAP_CACHE), help="Путь для сохранения карты переводов CSV")
    parser.add_argument("--glossary-csv", help="Явно выбранный CSV-глоссарий для текущего языка перевода")
    parser.add_argument("--no-map", action="store_true", help="Не сохранять CSV карту переводов")
    parser.add_argument("--extracted-txt", default="extracted_texts.txt", help="TXT с исходным текстом")
    parser.add_argument("--translated-txt", default="extracted_texts_ru.txt", help="TXT с переводом")
    parser.add_argument("--skip-txt", action="store_true", help="Не сохранять TXT-файлы")
    parser.add_argument(
        "--restart-translation",
        action="store_true",
        help="Сбросить checkpoint и начать перевод заново",
    )
    parser.add_argument("--translator", default="google", help="Движок перевода")
    parser.add_argument("--source-lang", default="en", help="Язык оригинала")
    parser.add_argument("--target-lang", default="ru", help="Язык перевода")
    parser.add_argument(
        "--style-font",
        help="Имя TTF/SHX-файла для кириллицы или original для исходных стилей",
    )
    parser.add_argument("--deepl-key", help="Ключ DeepL")
    parser.add_argument("--openai-key", help="Ключ OpenAI")
    parser.add_argument("--openai-model", help="Модель OpenAI")
    parser.add_argument("--openai-base-url", help="Базовый URL для OpenAI-совместимого API")
    parser.add_argument("--openai-project", help="Идентификатор проекта OpenAI")
    parser.add_argument("--openai-temperature", type=float, default=0.2, help="Температура генерации")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Уровень reasoning.effort для reasoning-моделей OpenAI",
    )
    parser.add_argument(
        "--openai-verbosity",
        choices=["low", "medium", "high"],
        help="Уровень text.verbosity для GPT-5 моделей",
    )
    parser.add_argument(
        "--openai-strict-mode",
        choices=["verbosity", "effort"],
        help="Устаревший параметр совместимости; используйте --openai-reasoning-effort/--openai-verbosity",
    )
    parser.add_argument(
        "--openai-strict-value",
        type=float,
        help="Устаревшее значение совместимости 0.0-1.0",
    )
    parser.add_argument("--codex-cli-path", help="Путь к исполняемому файлу Codex CLI")
    parser.add_argument("--codex-model", default="gpt-5.4-mini", help="Модель Codex CLI")
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="low",
        help="Уровень reasoning для Codex CLI",
    )
    parser.add_argument(
        "--codex-analysis-model",
        default="gpt-5.5",
        help="Модель Codex CLI для предварительного анализа",
    )
    parser.add_argument(
        "--codex-analysis-reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="high",
        help="Уровень reasoning Codex CLI для предварительного анализа",
    )
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=300,
        help="Таймаут одного вызова Codex CLI",
    )
    parser.add_argument("--output-format", choices=["dwg", "dxf"], default="dwg", help="Формат итогового файла")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
