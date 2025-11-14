import argparse
import csv
import os
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, nullcontext
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple

import ezdxf
from ezdxf import options as ezdxf_options
from ezdxf.addons import odafc
from ezdxf.addons.odafc import ODAFCNotInstalledError, UnknownODAFCError

import apply_dxf_translation as applier
import extract_dxf_texts as extractor
from auto_translation import TranslationEngine

DEFAULT_MAP_CACHE = Path("map_auto.csv")


def paths_equal(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return a == b


class Spinner:
    def __init__(self, message: str = "", enabled: bool = True, interval: float = 0.1) -> None:
        self.message = message
        self.enabled = enabled
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join()
            sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
            sys.stdout.flush()

    def _spin(self) -> None:
        spinner = cycle("|/-\\")
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r{next(spinner)} {self.message}")
            sys.stdout.flush()
            time.sleep(self.interval)


def colorize(text: str, color_code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{color_code}m{text}\033[0m"
    return text


def ensure_parent(path: Path) -> None:
    if path and path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _determine_output_format(output_format: Optional[str], output_path: Path) -> str:
    if output_format:
        fmt = output_format.lower()
    else:
        suffix = output_path.suffix.lower()
        fmt = "dwg" if suffix == ".dwg" else "dxf"
    if fmt not in {"dwg", "dxf"}:
        raise RuntimeError("Поддерживаются только форматы DWG и DXF")
    return fmt


def _convert_with_odafc(source: Path, dest: Path, logger: Callable[[str], None], *, replace: bool = True) -> None:
    _ensure_odafc_config()
    try:
        odafc.convert(str(source), str(dest), version="R2018", replace=replace)
    except ODAFCNotInstalledError as exc:
        raise RuntimeError(
            "Не найден ODA File Converter. Установите ODAFileConverter и добавьте его в PATH."
        ) from exc
    except UnknownODAFCError as exc:
        raise RuntimeError(f"Ошибка ODA File Converter: {exc}") from exc
    except Exception as exc:  # pragma: no cover - best effort
        raise RuntimeError(f"Не удалось конвертировать файл через ODA FC: {exc}") from exc
    else:
        logger(f"Конвертация выполнена: {dest}")


def _prepare_input_file(path: Path, logger: Callable[[str], None], stack: ExitStack) -> Path:
    ext = path.suffix.lower()
    if ext == ".dxf":
        return path
    if ext == ".dwg":
        logger("Преобразуем DWG в DXF для обработки...")
        temp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="daru_in_")))
        dxf_path = temp_dir / f"{path.stem}_source.dxf"
        _convert_with_odafc(path, dxf_path, logger)
        return dxf_path
    raise RuntimeError("Поддерживаются только входные файлы с расширением DWG или DXF")


def _prepare_output_destination(path: Path, fmt: str) -> Path:
    suffix = ".dwg" if fmt == "dwg" else ".dxf"
    return path.with_suffix(suffix)


def _ensure_odafc_config() -> None:
    if odafc.is_installed():
        return

    candidates = []
    env_exec = os.environ.get("ODA_FILE_CONVERTER")
    if env_exec:
        candidates.append(Path(env_exec))
    env_path = os.environ.get("ODAFC_PATH") or os.environ.get("ODAFC_HOME")
    if env_path:
        candidates.append(Path(env_path) / "ODAFileConverter.exe")

    if os.name == "nt":
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
        for base in program_files:
            if not base:
                continue
            oda_dir = Path(base) / "ODA"
            if oda_dir.exists():
                for sub in sorted(oda_dir.glob("ODAFileConverter*"), reverse=True):
                    candidates.append(sub / "ODAFileConverter.exe")

    for candidate in candidates:
        if candidate and candidate.exists():
            ezdxf_options.set("odafc-addon", "win_exec_path", str(candidate))
            if odafc.is_installed():
                return

    # if still not installed, odafc.convert will raise the detailed error


def write_translated_txt(path: Path, items: List[Tuple[str, int]], translations: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for (text, count), translated in zip(items, translations):
            f.write(f"[{count}] {translated}\n")


def write_map_csv(path: Path, english: List[str], russian: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text_en", "text_ru"])
        for en, ru in zip(english, russian):
            writer.writerow([en, ru])


def read_map_csv(path: Optional[Path]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path:
        return cache
    try:
        if not path.exists():
            return cache
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                en = (row.get("text_en") or "").strip()
                ru = row.get("text_ru") or ""
                if en:
                    cache[en] = ru
    except Exception:
        return {}
    return cache


def apply_translations(doc, pairs: List[Tuple[str, str]], style_font: Optional[str] = None) -> None:
    if style_font:
        applier.STYLE_FONT = style_font
    ru_style = applier.ensure_ru_style(doc)
    for layout in [doc.modelspace(), *[l for l in doc.layouts if l.name != "Model"]]:
        applier.walk_layout(layout, pairs, ru_style)
    applier.walk_blocks(doc, pairs, ru_style)


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
    openai_temperature: float = 0.2,
    openai_strict_mode: Optional[str] = None,
    openai_strict_value: Optional[float] = None,
    output_format: Optional[str] = None,
    map_path: Optional[Path] = None,
    save_map: bool = True,
    extracted_txt_path: Optional[Path] = None,
    translated_txt_path: Optional[Path] = None,
    save_txt: bool = True,
    log: Optional[Callable[[str], None]] = None,
    translation_context_factory: Optional[Callable[[str], ContextManager[Any]]] = None,
) -> Dict[str, Any]:
    logger = log or (lambda message: None)

    input_path = input_path.expanduser()
    output_path = output_path.expanduser()

    final_format = _determine_output_format(output_format, output_path)

    with ExitStack() as stack:
        processing_input = _prepare_input_file(input_path, logger, stack)

        logger(f"Загружаем чертёж: {processing_input}")
        doc = ezdxf.readfile(str(processing_input)) # основная загрузка DXF и обработка

        freq = extractor.extract_text_counts(doc)
    sorted_items = extractor.sort_frequency(freq)
    english_texts = [text for text, _ in sorted_items]
    logger(f"Найдено {len(english_texts)} текстовых элементов для перевода")
    cache_lookup_path: Optional[Path] = None
    if map_path:
        cache_lookup_path = map_path
    elif DEFAULT_MAP_CACHE.exists():
        cache_lookup_path = DEFAULT_MAP_CACHE
    translation_cache = read_map_csv(cache_lookup_path)
    if translation_cache and cache_lookup_path:
        logger(f"Loaded {len(translation_cache)} cached translations from {cache_lookup_path}")

    translator = TranslationEngine(
        provider=translator_name,
        source_lang=source_lang,
        target_lang=target_lang,
        deepl_auth_key=deepl_key,
        openai_api_key=openai_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_temperature=openai_temperature,
        openai_strict_mode=openai_strict_mode,
        openai_strict_value=openai_strict_value,
    )
    logger(f"Инициализирован движок перевода: {translator.backend_name()}")

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
    if total_items == 0 or not unique_pending:
        logger("Начинаем перевод... [100%]")
    else:
        processed = cached_hits
        batch_size = max(1, len(unique_pending) // 20)
        with context_factory("Переводим..."):
            for start_idx in range(0, len(unique_pending), batch_size):
                chunk_texts = unique_pending[start_idx : start_idx + batch_size]
                translated_chunk = translator.translate_many(chunk_texts)
                for original_text, translated_text in zip(chunk_texts, translated_chunk):
                    translation_cache[original_text] = translated_text
                    for text_index in text_to_indices.get(original_text, []):
                        translation_slots[text_index] = translated_text
                        processed += 1
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
    cache_output_path: Optional[Path] = DEFAULT_MAP_CACHE
    if save_map and map_path and paths_equal(map_path, DEFAULT_MAP_CACHE):
        cache_output_path = None
    if cache_output_path:
        ensure_parent(cache_output_path)
        write_map_csv(cache_output_path, english_texts, russian_texts)
        if not (save_map and map_path and paths_equal(map_path, DEFAULT_MAP_CACHE)):
            logger(f"Обновлён локальный кэш переводов: {cache_output_path}")

    pairs = list(zip(english_texts, russian_texts))
    pairs.sort(key=lambda x: -len(x[0]))
    logger("Применяем переводы к DXF")
    apply_translations(doc, pairs, style_font=style_font)

    final_output_path = _prepare_output_destination(output_path, final_format)
    ensure_parent(final_output_path)

    with ExitStack() as stack2:
        if final_format == "dwg":
            temp_dir = Path(stack2.enter_context(tempfile.TemporaryDirectory(prefix="daru_out_")))
            dxf_output = temp_dir / (final_output_path.stem + ".dxf")
        else:
            dxf_output = final_output_path if final_output_path.suffix.lower() == ".dxf" else final_output_path.with_suffix(".dxf")

        doc.saveas(str(dxf_output))

        if final_format == "dwg":
            logger("Преобразуем результат в DWG...")
            _convert_with_odafc(dxf_output, final_output_path, logger)
            logger(f"DWG сохранён: {final_output_path}")
        else:
            logger(f"DXF сохранён: {final_output_path}")

    return {
        "output_path": final_output_path,
        "backend": translator.backend_name(),
        "map_saved": bool(save_map and map_path),
        "extracted_txt_saved": bool(save_txt and extracted_txt_path),
        "translated_txt_saved": bool(save_txt and translated_txt_path),
        "items_translated": len(english_texts),
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
            openai_temperature=getattr(args, "openai_temperature", 0.2),
            openai_strict_mode=getattr(args, "openai_strict_mode", None),
            openai_strict_value=getattr(args, "openai_strict_value", None),
            output_format=output_format_cli,
            map_path=map_path,
            save_map=not args.no_map,
            extracted_txt_path=extracted_txt_path,
            translated_txt_path=translated_txt_path,
            save_txt=not args.skip_txt,
            log=cli_log,
            translation_context_factory=(
                (lambda msg: Spinner(colorize(msg, 30), enabled=sys.stdout.isatty()))
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
    parser.add_argument("--map-csv", default="map_auto.csv", help="Путь для сохранения карты переводов CSV")
    parser.add_argument("--no-map", action="store_true", help="Не сохранять CSV карту переводов")
    parser.add_argument("--extracted-txt", default="extracted_texts.txt", help="TXT с исходным текстом")
    parser.add_argument("--translated-txt", default="extracted_texts_ru.txt", help="TXT с переводом")
    parser.add_argument("--skip-txt", action="store_true", help="Не сохранять TXT-файлы")
    parser.add_argument(
        "--translator",
        default="google",
        help="Движок перевода (google/googletrans/deep_google/deepl/chatgpt/noop)",
    )
    parser.add_argument("--source-lang", default="en", help="Язык оригинала (default: en)")
    parser.add_argument("--target-lang", default="ru", help="Язык перевода (default: ru)")
    parser.add_argument("--style-font", help="Имя TTF-файла для стиля шрифта кириллицы")
    parser.add_argument("--deepl-key", help="Ключ DeepL (можно также задать через переменную DEEPL_AUTH_KEY)")
    parser.add_argument("--openai-key", help="Ключ OpenAI совместимого API (или переменная OPENAI_API_KEY)")
    parser.add_argument("--openai-model", help="Модель OpenAI (default: gpt-4o-mini)")
    parser.add_argument("--openai-base-url", help="Базовый URL для OpenAI-совместимого API")
    parser.add_argument(
        "--openai-temperature",
        type=float,
        default=0.2,
        help="Температура генерации ChatGPT (default: 0.2)",
    )
    parser.add_argument(
        "--openai-strict-mode",
        choices=["verbosity", "effort"],
        help="Параметр строгого режима для GPT-5 моделей (verbosity/effort)",
    )
    parser.add_argument(
        "--openai-strict-value",
        type=float,
        help="Значение параметра verbosity/effort (0.0-1.0)",
    )
    parser.add_argument(
        "--output-format",
        choices=["dwg", "dxf"],
        default="dwg",
        help="Формат итогового файла",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
