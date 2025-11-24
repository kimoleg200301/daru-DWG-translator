from __future__ import annotations

import csv
import io
import json
import tempfile
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set

from auto_translation import TranslationEngine
from pdf_layout import (
    PdfLayout,
    PdfLayoutLine,
    PdfLineRef,
    collect_line_refs,
    extract_pdf_layout,
    fitz as pymupdf,
)

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    convert_from_path = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PyPDF2 import PdfReader, PdfWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore
    PdfWriter = None  # type: ignore

PdfLog = Callable[[str], None]

PDF_TYPE_SCANNED = "scanned"
PDF_TYPE_NATIVE = "native"
DEFAULT_MAP_CACHE = Path("map_auto.csv")
_FREETEXT_FONT_WARNINGS: Set[str] = set()
_FREETEXT_DRAW_WARNING = False



def translate_pdf(
    *,
    input_path: Path,
    output_path: Path,
    layer_json_path: Optional[Path] = None,
    translation_cache_path: Optional[Path] = None,
    translator_name: str,
    source_lang: str,
    target_lang: str,
    pdf_type: str,
    log: PdfLog,
    style_font: Optional[str] = None,
    deepl_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_project: Optional[str] = None,
    openai_temperature: float = 0.2,
    openai_strict_mode: Optional[str] = None,
    openai_strict_value: Optional[float] = None,
) -> Dict[str, object]:
    if not input_path.exists():
        raise RuntimeError("PDF файл не найден")
    if input_path.suffix.lower() != ".pdf":
        raise RuntimeError("Ошибка! Приложение обрабатывает только PDF документы")

    normalized_type = _normalize_pdf_type(pdf_type)
    log(f"PDF файл: {input_path.name}")
    log(f"Режим обработки: {'OCR' if normalized_type == PDF_TYPE_SCANNED else 'Без OCR'}")

    with tempfile.TemporaryDirectory(prefix="daru_pdf_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        if normalized_type == PDF_TYPE_SCANNED:
            searchable_source = _build_searchable_pdf(input_path, temp_dir, log)
        else:
            searchable_source = input_path
        layout = extract_pdf_layout(searchable_source, log)
        line_refs = collect_line_refs(layout)
        if not line_refs:
            raise RuntimeError("Не удалось извлечь текст из PDF документа")
        log(f"Найдено {len(line_refs)} текстовых строк для обработки")

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
            openai_strict_mode=openai_strict_mode,
            openai_strict_value=openai_strict_value,
        )
        log(f"Инициализирован движок перевода: {translator.backend_name()}")
        segments = [ref.text for ref in line_refs]
        log(f"Готовим {len(segments)} блоков текста к переводу")
        total_segments = len(segments)
        cache_path = _resolve_cache_path(translation_cache_path)
        translation_cache = _read_translation_cache(cache_path)
        if translation_cache and cache_path.exists():
            log(
                f"PDF: загружено {len(translation_cache)} переводов из кэша: {cache_path}"
            )
        translation_slots: List[Optional[str]] = [None] * total_segments
        text_to_indices: Dict[str, List[int]] = {}
        cache_updates: Dict[str, str] = {}
        cached_hits = 0
        for idx, segment in enumerate(segments):
            cached_value = translation_cache.get(segment) if translation_cache else None
            if cached_value is not None:
                translation_slots[idx] = cached_value
                cached_hits += 1
            else:
                bucket = text_to_indices.setdefault(segment, [])
                bucket.append(idx)
        unique_pending = list(text_to_indices.keys())
        processed = cached_hits
        if total_segments == 0 or not unique_pending:
            log("Начинаем перевод... [100%]")
            log("Перевод завершён")
        else:
            log("Начинаем перевод... [0%]")
            chunk_size = max(1, len(unique_pending) // 20)
            for start in range(0, len(unique_pending), chunk_size):
                chunk = unique_pending[start : start + chunk_size]
                translated_chunk = translator.translate_many(chunk)
                for original_text, translated_text in zip(chunk, translated_chunk):
                    previous_value = translation_cache.get(original_text)
                    if previous_value != translated_text:
                        translation_cache[original_text] = translated_text
                        cache_updates[original_text] = translated_text
                    for text_index in text_to_indices.get(original_text, []):
                        translation_slots[text_index] = translated_text
                        processed += 1
                percent = min(100, int(processed / total_segments * 100)) if total_segments else 100
                log(f"Начинаем перевод... [{percent}%]")
            if processed < total_segments:
                log("Начинаем перевод... [100%]")
            log("Перевод завершён")
        if any(slot is None for slot in translation_slots):
            raise RuntimeError("Сбой перевода: не совпадает количество строк")
        translated_segments = [slot or "" for slot in translation_slots]
        if cache_updates and cache_path:
            _write_translation_cache(cache_path, cache_updates, log)
        overlay_records = _render_translated_pdf(
            source_pdf=searchable_source,
            layout=layout,
            line_refs=line_refs,
            translated_segments=translated_segments,
            output_path=output_path,
            style_font=style_font,
            log=log,
        )
        if layer_json_path:
            _export_overlay_metadata(overlay_records, layer_json_path, log)

    return {
        "output_path": output_path,
        "backend": translator.backend_name(),
        "pages": len(layout.pages),
        "job_type": "pdf",
        "ocr_performed": normalized_type == PDF_TYPE_SCANNED,
    }


def _normalize_pdf_type(pdf_type: Optional[str]) -> str:
    if not pdf_type:
        return PDF_TYPE_SCANNED
    value = pdf_type.strip().lower()
    if value in {"scan", "scanned", "ocr", "отсканированный"}:
        return PDF_TYPE_SCANNED
    return PDF_TYPE_NATIVE


def _require_dependency(obj: object, message: str) -> None:
    if obj is None:
        raise RuntimeError(message)


def _build_searchable_pdf(source: Path, temp_dir: Path, log: PdfLog) -> Path:
    _require_dependency(convert_from_path, "Установите pdf2image (pip install pdf2image)")
    _require_dependency(pytesseract, "Установите pytesseract (pip install pytesseract)")
    _require_dependency(PdfWriter, "Установите PyPDF2 (pip install PyPDF2)")

    log("OCR: конвертируем страницы PDF в картинки для распознавания...")
    poppler_path = _detect_poppler_path()
    poppler_hint = " Укажите переменную POPPLER_PATH или добавьте папку bin в PATH." if not poppler_path else ""
    try:
        images = convert_from_path(  # type: ignore[arg-type]
            str(source),
            dpi=350,
            poppler_path=poppler_path,
        )
    except Exception as exc:
        raise RuntimeError(f"Не удалось обработать PDF с помощью Poppler: {exc}.{poppler_hint}") from exc
    writer = PdfWriter()  # type: ignore[call-arg]
    for idx, image in enumerate(images, start=1):
        log(f"OCR: распознаём страницу {idx}/{len(images)}")
        color_image = image if image.mode == "RGB" else image.convert("RGB")
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(color_image, extension="pdf")  # type: ignore[arg-type]
        reader = PdfReader(io.BytesIO(pdf_bytes))  # type: ignore[call-arg]
        writer.add_page(reader.pages[0])  # type: ignore[arg-type]
    searchable_path = temp_dir / f"{source.stem}_searchable.pdf"
    with searchable_path.open("wb") as handle:
        writer.write(handle)  # type: ignore[arg-type]
    log(f"OCR: собран распознанный PDF: {searchable_path}")
    return searchable_path


@dataclass
class _LineAnnotationStyle:
    font_name: str
    font_size: float
    text_color: Tuple[float, float, float]
    background_color: Tuple[float, float, float]


@dataclass
class _OverlayRecord:
    annotation: str
    page_index: int
    block_index: int
    line_index: int
    bbox: Tuple[float, float, float, float]
    font_name: str
    font_size: float
    text_color: Tuple[float, float, float]
    background_color: Tuple[float, float, float]
    source_text: str
    translated_text: str


def _render_translated_pdf(
    *,
    source_pdf: Path,
    layout: PdfLayout,
    line_refs: Sequence[PdfLineRef],
    translated_segments: Sequence[str],
    output_path: Path,
    style_font: Optional[str],
    log: PdfLog,
) -> List[_OverlayRecord]:
    _require_dependency(pymupdf, "Установите PyMuPDF (pip install pymupdf)")
    doc = pymupdf.open(str(source_pdf))  # type: ignore[attr-defined]
    try:
        custom_font_handle: Optional[object] = None
        custom_font_name: Optional[str] = None
        custom_font_handle: Optional[object] = None
        custom_font_name: Optional[str] = None
        if style_font:
            custom_font_name, custom_font_handle = _load_custom_font(style_font, log)
        overlay_count = 0
        page_cache: Dict[int, object] = {}
        records: List[_OverlayRecord] = []
        for ref, translated in zip(line_refs, translated_segments):
            text = _normalize_translated_text(translated)
            if not text:
                continue
            page = page_cache.get(ref.page_index)
            if page is None:
                page = doc.load_page(ref.page_index)  # type: ignore[attr-defined]
                page_cache[ref.page_index] = page
            page_meta = layout.pages[ref.page_index]
            style = _build_line_style(ref.line, style_font, custom_font_name)
            rect = _prepare_annotation_rect(
                ref.line,
                page_meta.width,
                page_meta.height,
                style.font_size,
            )
            annotation_name = _try_apply_freetext_overlay(
                page,
                rect,
                text,
                style,
                annotation_index=overlay_count,
                log=log,
            )
            overlay_count += 1
            records.append(
                _OverlayRecord(
                    annotation=annotation_name,
                    page_index=ref.page_index,
                    block_index=ref.block_index,
                    line_index=ref.line_index,
                    bbox=tuple(float(coord) for coord in ref.line.bbox),
                    font_name=style.font_name,
                    font_size=style.font_size,
                    text_color=style.text_color,
                    background_color=style.background_color,
                    source_text=ref.text,
                    translated_text=text,
                )
            )
        log(f"PDF: добавлено {overlay_count} переводов поверх оригинала")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))  # type: ignore[attr-defined]
        log(f"PDF: файл сохранён: {output_path}")
        return records
    finally:
        if custom_font_handle is not None:
            # keep reference scope explicit for clarity; object cleaned automatically
            custom_font_handle = None
        doc.close()  # type: ignore[attr-defined]


def _normalize_translated_text(value: str) -> str:
    stripped = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not stripped:
        return ""
    tokens = [token.strip() for token in stripped.split("\n") if token.strip()]
    normalized = " ".join(tokens)
    return normalized.strip()


def _build_line_style(
    line: PdfLayoutLine,
    style_font: Optional[str],
    custom_font_name: Optional[str],
) -> _LineAnnotationStyle:
    font_size = _infer_font_size(line)
    if custom_font_name:
        font_name = custom_font_name
    else:
        font_name = _resolve_annotation_font(line, style_font)
    text_rgb = _select_text_color(line)
    background_rgb = _select_background_color(line)
    text_color = _color_to_pdf_tuple(text_rgb, fallback=(0, 0, 0))
    background_color = _color_to_pdf_tuple(background_rgb, fallback=(255, 255, 255))
    if _color_distance(text_color, background_color) < 0.2:
        background_color = _nudge_background(background_color)
    return _LineAnnotationStyle(
        font_name=font_name,
        font_size=font_size,
        text_color=text_color,
        background_color=background_color,
    )


def _infer_font_size(line: PdfLayoutLine) -> float:
    sizes = [span.font_size for span in line.spans if span.font_size > 0]
    if not sizes:
        return 11.0
    average = sum(sizes) / len(sizes)
    clamped = max(6.0, min(72.0, average))
    return clamped


def _resolve_annotation_font(line: PdfLayoutLine, style_font: Optional[str]) -> str:
    preferred = _map_font_name(style_font)
    if preferred:
        return preferred
    for span in line.spans:
        mapped = _map_font_name(span.font)
        if mapped:
            return mapped
    return "helv"


def _map_font_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    basename = Path(name).stem.lower()
    if any(keyword in basename for keyword in ("mono", "cour", "fixed")):
        return "cour"
    if any(keyword in basename for keyword in ("times", "serif", "tr")):
        return "tiro"
    return "helv"


def _select_text_color(line: PdfLayoutLine) -> Optional[Tuple[int, int, int]]:
    for span in line.spans:
        if span.fill_color:
            return span.fill_color
    return None


def _select_background_color(line: PdfLayoutLine) -> Optional[Tuple[int, int, int]]:
    for span in line.spans:
        if span.background_color:
            return span.background_color
    return None


def _color_to_pdf_tuple(
    color: Optional[Tuple[int, int, int]],
    *,
    fallback: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    source = color or fallback
    return tuple(max(0.0, min(1.0, component / 255.0)) for component in source)  # type: ignore[return-value]


def _color_serialization(color: Tuple[float, float, float]) -> Dict[str, Sequence[float]]:
    rgb = [max(0, min(255, int(round(channel * 255)))) for channel in color]
    return {
        "normalized": [round(channel, 4) for channel in color],
        "rgb": rgb,
    }


def _color_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _nudge_background(color: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # Ensure text remains legible by slightly adjusting similar colors.
    factor = 0.15
    adjusted = tuple(min(1.0, max(0.0, channel + factor if channel < 0.5 else channel - factor)) for channel in color)
    return adjusted  # type: ignore[return-value]


def _prepare_annotation_rect(
    line: PdfLayoutLine,
    page_width: float,
    page_height: float,
    font_size: float,
):
    padding = max(0.8, font_size * 0.15)
    rect = pymupdf.Rect(line.bbox)  # type: ignore[attr-defined]
    min_height = max(font_size * 1.2, 5.0)
    if rect.height < min_height:
        delta = (min_height - rect.height) / 2
        rect.y0 -= delta
        rect.y1 += delta
    if rect.width < font_size:
        rect.x1 = rect.x0 + font_size
    rect.x0 -= padding
    rect.x1 += padding
    rect.y0 -= padding
    rect.y1 += padding
    rect.x0 = max(0.0, rect.x0)
    rect.y0 = max(0.0, rect.y0)
    rect.x1 = min(page_width, rect.x1)
    rect.y1 = min(page_height, rect.y1)
    if rect.x1 <= rect.x0:
        rect.x1 = min(page_width, rect.x0 + font_size * 1.5)
    if rect.y1 <= rect.y0:
        rect.y1 = min(page_height, rect.y0 + font_size * 1.4)
    return rect


def _apply_freetext_overlay(
    page: object,
    rect: "pymupdf.Rect",
    text: str,
    style: _LineAnnotationStyle,
    *,
    annotation_index: int,
) -> str:
    annot = page.add_freetext_annot(  # type: ignore[attr-defined]
        rect,
        text,
        fontsize=style.font_size,
        fontname=style.font_name,
        text_color=style.text_color,
    )
    annot.set_border(width=0)  # type: ignore[attr-defined]
    annot.set_colors(fill=style.background_color)  # type: ignore[attr-defined]
    annot.set_opacity(0.98)  # type: ignore[attr-defined]
    annot.set_flags(pymupdf.PDF_ANNOT_PRINT)  # type: ignore[attr-defined]
    annot.set_info(
        title="Daru Translator",
        content="Переведено Daru DWG Translator",
    )  # type: ignore[attr-defined]
    annotation_name = f"DARU_TEXT_{annotation_index:05d}"
    annot.set_name(annotation_name)  # type: ignore[attr-defined]
    annot.update()  # type: ignore[attr-defined]
    return annotation_name


def _draw_text_overlay(
    page: object,
    rect: "pymupdf.Rect",
    text: str,
    style: _LineAnnotationStyle,
    *,
    annotation_index: int,
) -> str:
    font_name = style.font_name or "helv"
    background = style.background_color or (1.0, 1.0, 1.0)
    text_color = style.text_color or (0.0, 0.0, 0.0)
    try:
        page.draw_rect(  # type: ignore[attr-defined]
            rect,
            color=background,
            fill=background,
            overlay=True,
        )
    except Exception:
        pass
    page.insert_textbox(  # type: ignore[attr-defined]
        rect,
        text,
        fontsize=style.font_size,
        fontname=font_name,
        color=text_color,
        overlay=True,
    )
    return f"DARU_DRAWN_{annotation_index:05d}"


def _try_apply_freetext_overlay(
    page: object,
    rect: "pymupdf.Rect",
    text: str,
    style: _LineAnnotationStyle,
    *,
    annotation_index: int,
    log: PdfLog,
) -> str:
    attempted_fonts = []
    font_candidates: List[str] = []
    seen_fonts = set()
    primary_font = style.font_name or "helv"
    font_candidates.append(primary_font)
    seen_fonts.add(primary_font.lower())
    for fallback_font in ("helv", "cour", "tiro"):
        if fallback_font.lower() not in seen_fonts:
            font_candidates.append(fallback_font)
            seen_fonts.add(fallback_font.lower())
    for font_name in font_candidates:
        variant = style
        if font_name.lower() != (style.font_name or "").lower():
            variant = _LineAnnotationStyle(
                font_name=font_name,
                font_size=style.font_size,
                text_color=style.text_color,
                background_color=style.background_color,
            )
        try:
            result = _apply_freetext_overlay(
                page,
                rect,
                text,
                variant,
                annotation_index=annotation_index,
            )
            style.font_name = variant.font_name
            return result
        except Exception as exc:
            attempted_fonts.append(font_name)
            if not _is_freetext_font_error(exc):
                raise
            _log_freetext_font_warning(font_name, log)
            continue
    _log_freetext_draw_warning(log)
    style.font_name = font_candidates[-1] if font_candidates else "helv"
    return _draw_text_overlay(
        page,
        rect,
        text,
        style,
        annotation_index=annotation_index,
    )


def _is_freetext_font_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "freetext" in message


def _log_freetext_font_warning(font_name: str, log: PdfLog) -> None:
    normalized = font_name.lower()
    if normalized in _FREETEXT_FONT_WARNINGS:
        return
    _FREETEXT_FONT_WARNINGS.add(normalized)
    log(f"PDF: шрифт '{font_name}' не подходит для FreeText, пытаемся другой")


def _log_freetext_draw_warning(log: PdfLog) -> None:
    global _FREETEXT_DRAW_WARNING
    if _FREETEXT_DRAW_WARNING:
        return
    _FREETEXT_DRAW_WARNING = True
    log("PDF: FreeText недоступен, рисуем текст прямо на странице")


def _export_overlay_metadata(
    records: Sequence[_OverlayRecord],
    output_path: Path,
    log: PdfLog,
) -> None:
    items = [
        {
            "annotation": record.annotation,
            "page_index": record.page_index,
            "block_index": record.block_index,
            "line_index": record.line_index,
            "bbox": list(record.bbox),
            "font_name": record.font_name,
            "font_size": record.font_size,
            "text_color": _color_serialization(record.text_color),
            "background_color": _color_serialization(record.background_color),
            "source_text": record.source_text,
            "translated_text": record.translated_text,
        }
        for record in records
    ]
    payload = {
        "version": 1,
        "items": items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    log(f"PDF: сохранён JSON слой: {output_path}")


def _detect_poppler_path() -> Optional[str]:
    for env_name in ("POPPLER_PATH", "POPPLER_BIN", "POPPLER_HOME"):
        candidate = os.environ.get(env_name)
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path)
    return None


def _load_custom_font(style_font: str, log: PdfLog) -> Tuple[Optional[str], Optional[object]]:
    path = _resolve_font_file(style_font)
    if not path:
        log(f"PDF: файл шрифта не найден: {style_font}")
        return None, None
    try:
        font_handle = pymupdf.Font(fontfile=str(path))  # type: ignore[attr-defined]
    except Exception as exc:
        log(f"PDF: не удалось загрузить шрифт {path}: {exc}")
        return None, None
    log(f"PDF: подключён шрифт для аннотаций: {path}")
    return font_handle.name, font_handle


def _resolve_cache_path(candidate: Optional[Path]) -> Path:
    target = candidate if candidate is not None else DEFAULT_MAP_CACHE
    return target.expanduser()


def _resolve_font_file(font_value: str) -> Optional[Path]:
    candidate = Path(font_value).expanduser()
    if candidate.exists():
        return candidate
    search_dirs: List[Path] = []
    env_dir = os.environ.get("DARU_FONT_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir).expanduser())
    app_dir = Path(__file__).resolve().parent
    search_dirs.append(app_dir)
    platform = sys.platform
    home = Path.home()
    if platform == "win32":
        windir = Path(os.environ.get("WINDIR", "C:/Windows"))
        search_dirs.append(windir / "Fonts")
    elif platform == "darwin":
        search_dirs.extend(
            [
                Path("/System/Library/Fonts"),
                Path("/System/Library/Fonts/Supplemental"),
                Path("/Library/Fonts"),
                home / "Library/Fonts",
            ]
        )
    else:
        search_dirs.extend(
            [
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                home / ".fonts",
                home / ".local/share/fonts",
            ]
        )
    name = Path(font_value).name
    stem = Path(font_value).stem
    suffix = Path(font_value).suffix
    possible_names: List[str] = []
    if suffix:
        possible_names.append(name)
    else:
        possible_names.extend(
            [
                f"{stem}.ttf",
                f"{stem}.ttc",
                f"{stem}.otf",
            ]
        )
    # ensure original literal checked as well
    if name not in possible_names:
        possible_names.append(name)
    seen = set()
    filtered_names = []
    for item in possible_names:
        lower = item.lower()
        if lower in seen:
            continue
        seen.add(lower)
        filtered_names.append(item)
    for directory in search_dirs:
        if not directory.exists():
            continue
        for item in filtered_names:
            font_path = directory / item
            if font_path.exists():
                return font_path
    return None


def _read_translation_cache(path: Optional[Path]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path:
        return cache
    try:
        if not path.exists():
            return cache
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                source = (row.get("text_en") or "").strip()
                if not source:
                    continue
                cache[source] = row.get("text_ru") or ""
    except Exception:
        return {}
    return cache


def _write_translation_cache(path: Optional[Path], updates: Dict[str, str], log: PdfLog) -> None:
    if not path or not updates:
        return
    merged_cache = _read_translation_cache(path)
    applied = 0
    for source, translated in updates.items():
        if merged_cache.get(source) != translated:
            merged_cache[source] = translated
            applied += 1
    if applied == 0 and path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text_en", "text_ru"])
            for source, translated in merged_cache.items():
                writer.writerow([source, translated])
    except Exception as exc:
        log(f"PDF: не удалось обновить кэш переводов ({exc})")
    else:
        log(
            f"PDF: обновлён кэш переводов ({applied} новых значений): {path}"
        )
