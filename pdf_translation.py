from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from auto_translation import TranslationEngine, prepare_for_translation
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
_FONT_CACHE: Dict[str, object] = {}
_TEXT_ALIGN_LEFT = getattr(pymupdf, "TEXT_ALIGN_LEFT", 0) if pymupdf is not None else 0



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
        paragraph_refs, skipped_blocks = _collect_paragraph_refs(line_refs)
        if not paragraph_refs:
            raise RuntimeError("Не удалось сформировать текстовые абзацы для перевода")
        log(f"Сформировано {len(paragraph_refs)} абзацев для перевода")
        if skipped_blocks:
            log(f"PDF: пропущено {skipped_blocks} блоков без значимого текста")

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
        raw_segments = [paragraph.text for paragraph in paragraph_refs]
        normalized_segments = [_prepare_source_text(text) for text in raw_segments]
        payload_segments = [raw if raw else normalized for raw, normalized in zip(raw_segments, normalized_segments)]
        filter_keys = [prepare_for_translation(text) for text in normalized_segments]
        log(f"Готовим {len(raw_segments)} абзацев текста к переводу")
        total_segments = len(raw_segments)
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
        for idx, (cache_key, filter_key) in enumerate(zip(normalized_segments, filter_keys)):
            cached_value = translation_cache.get(cache_key) if translation_cache else None
            if cached_value is not None:
                translation_slots[idx] = cached_value
                cached_hits += 1
            else:
                bucket = text_to_indices.setdefault(filter_key, [])
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
                chunk_keys = unique_pending[start : start + chunk_size]
                chunk_texts = [payload_segments[text_to_indices[key][0]] for key in chunk_keys]
                translated_chunk = translator.translate_many(chunk_texts)
                for key, translated_text in zip(chunk_keys, translated_chunk):
                    indices = text_to_indices.get(key, [])
                    for text_index in indices:
                        cache_key = normalized_segments[text_index]
                        previous_value = translation_cache.get(cache_key)
                        if previous_value != translated_text:
                            translation_cache[cache_key] = translated_text
                            cache_updates[cache_key] = translated_text
                        translation_slots[text_index] = translated_text
                        processed += 1
                percent = min(100, int(processed / total_segments * 100)) if total_segments else 100
                log(f"Начинаем перевод... [{percent}%]")
            if processed < total_segments:
                log("Начинаем перевод... [100%]")
            log("Перевод завершён")
        translated_segments: List[str] = []
        for idx, slot in enumerate(translation_slots):
            value = slot
            if not value:
                value = payload_segments[idx] if idx < len(payload_segments) else ""
            if not value and idx < len(paragraph_refs):
                value = paragraph_refs[idx].text
            translated_segments.append(value or "")
        if cache_updates and cache_path:
            _write_translation_cache(cache_path, cache_updates, log)
        overlay_records = _render_translated_pdf(
            source_pdf=searchable_source,
            layout=layout,
            paragraphs=paragraph_refs,
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
    font_handle: Optional[object] = None


@dataclass
class _ParagraphRef:
    page_index: int
    block_index: int
    paragraph_index: int
    lines: List[PdfLineRef]
    bbox: Tuple[float, float, float, float]
    text: str
    meaningful: bool


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
    paragraphs: Sequence[_ParagraphRef],
    translated_segments: Sequence[str],
    output_path: Path,
    style_font: Optional[str],
    log: PdfLog,
) -> List[_OverlayRecord]:
    _require_dependency(pymupdf, "Установите PyMuPDF (pip install pymupdf)")
    doc = pymupdf.open(str(source_pdf))  # type: ignore[attr-defined]
    try:
        custom_font_handle: Optional[object] = None
        if style_font:
            _, custom_font_handle = _load_custom_font(style_font, log)
        overlay_count = 0
        page_cache: Dict[int, object] = {}
        records: List[_OverlayRecord] = []
        for paragraph, translated in zip(paragraphs, translated_segments):
            text = _normalize_translated_text(translated)
            if not text:
                continue
            if not paragraph.lines:
                continue
            page = page_cache.get(paragraph.page_index)
            if page is None:
                page = doc.load_page(paragraph.page_index)  # type: ignore[attr-defined]
                page_cache[paragraph.page_index] = page
            page_meta = layout.pages[paragraph.page_index]
            style_line = paragraph.lines[0].line
            style = _build_line_style(style_line, style_font, custom_font_handle)
            rect = _prepare_annotation_rect(
                paragraph.bbox,
                page_meta.width,
                page_meta.height,
                style.font_size,
            )
            rect = _ensure_rect_capacity(
                rect,
                text,
                style,
                page_meta.width,
            )
            annotation_name = _draw_text_overlay(
                page,
                rect,
                text,
                style,
                annotation_index=overlay_count,
                page_rect=page.rect,  # type: ignore[attr-defined]
            )
            overlay_count += 1
            records.append(
                _OverlayRecord(
                    annotation=annotation_name,
                    page_index=paragraph.page_index,
                    block_index=paragraph.block_index,
                    line_index=paragraph.lines[0].line_index if paragraph.lines else 0,
                    bbox=tuple(float(coord) for coord in paragraph.bbox),
                    font_name=style.font_name,
                    font_size=style.font_size,
                    text_color=style.text_color,
                    background_color=style.background_color,
                    source_text=paragraph.text,
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


def _prepare_source_text(value: str) -> str:
    if not value:
        return ""
    flattened = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    tokens = flattened.split()
    return " ".join(tokens)


def _normalize_translated_text(value: str) -> str:
    stripped = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not stripped:
        return ""
    tokens = [token.strip() for token in stripped.split("\n") if token.strip()]
    normalized = " ".join(tokens)
    return normalized.strip()


def _collect_paragraph_refs(line_refs: Sequence[PdfLineRef]) -> Tuple[List[_ParagraphRef], int]:
    grouped: Dict[Tuple[int, int], List[PdfLineRef]] = {}
    for ref in line_refs:
        key = (ref.page_index, ref.block_index)
        bucket = grouped.setdefault(key, [])
        bucket.append(ref)
    paragraphs: List[_ParagraphRef] = []
    index = 0
    skipped = 0
    for key in sorted(grouped.keys()):
        refs = grouped[key]
        refs.sort(key=lambda item: item.line_index)
        current: List[PdfLineRef] = []
        for ref in refs:
            text = _extract_line_text(ref)
            if not text:
                if current:
                    paragraphs.append(_make_paragraph_ref(current, index))
                    index += 1
                    current = []
                continue
            current.append(ref)
        if current:
            paragraph = _make_paragraph_ref(current, index)
            if paragraph.meaningful:
                paragraphs.append(paragraph)
                index += 1
            else:
                skipped += 1
    return paragraphs, skipped


def _make_paragraph_ref(lines: Sequence[PdfLineRef], paragraph_index: int) -> _ParagraphRef:
    line_list = list(lines)
    first = line_list[0]
    bbox = _union_bboxes(line_list)
    parts = [_extract_line_text(ref) for ref in line_list if _extract_line_text(ref)]
    text = "\n".join(parts)
    if not text:
        text = "\n".join(ref.text for ref in line_list if ref.text)
    normalized = _prepare_source_text(text)
    meaningful = _is_meaningful_text(normalized)
    return _ParagraphRef(
        page_index=first.page_index,
        block_index=first.block_index,
        paragraph_index=paragraph_index,
        lines=line_list,
        bbox=bbox,
        text=text,
        meaningful=meaningful,
    )


def _union_bboxes(line_refs: Sequence[PdfLineRef]) -> Tuple[float, float, float, float]:
    x0 = min(ref.line.bbox[0] for ref in line_refs)
    y0 = min(ref.line.bbox[1] for ref in line_refs)
    x1 = max(ref.line.bbox[2] for ref in line_refs)
    y1 = max(ref.line.bbox[3] for ref in line_refs)
    return (float(x0), float(y0), float(x1), float(y1))


def _extract_line_text(ref: PdfLineRef) -> str:
    return _line_text_with_spacing(ref.line)


def _line_text_with_spacing(line: PdfLayoutLine) -> str:
    if not line.spans:
        return line.text().strip()
    parts: List[str] = []
    prev_end = None
    base_size = _infer_font_size(line)
    gap_threshold = max(base_size * 0.18, 0.6)
    for span in line.spans:
        segment = (span.text or "").replace("\xa0", " ")
        if not segment.strip():
            prev_end = span.bbox[2]
            continue
        start = span.bbox[0]
        if parts:
            gap = start - (prev_end or start)
            if gap > gap_threshold and not parts[-1].endswith(" ") and not segment.startswith(" "):
                if not parts[-1].endswith("-"):
                    parts.append(" ")
        parts.append(segment)
        prev_end = span.bbox[2]
    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    stripped = text.strip()
    if len(stripped) <= 1 and not stripped.isalnum():
        return ""
    return stripped


def _is_meaningful_text(value: str) -> bool:
    if not value:
        return False
    stripped = value.strip()
    if len(stripped) < 3:
        return False
    tokens = [token for token in stripped.split(" ") if token]
    if not tokens:
        return False
    long_tokens = sum(1 for token in tokens if len(token) >= 3)
    alpha_tokens = sum(1 for token in tokens if any(ch.isalpha() for ch in token))
    if long_tokens == 0 and alpha_tokens == 0:
        return False
    if alpha_tokens == 0:
        return False
    if long_tokens == 0 and len(tokens) > 4:
        return False
    return True


def _build_line_style(
    line: PdfLayoutLine,
    style_font: Optional[str],
    custom_font: Optional[object],
) -> _LineAnnotationStyle:
    font_size = _infer_font_size(line)
    font_handle: Optional[object]
    if custom_font is not None:
        font_name = getattr(custom_font, "name", None) or "helv"
        font_handle = custom_font
    else:
        font_name = _resolve_annotation_font(line, style_font)
        font_handle = _resolve_font_handle(font_name)
    text_rgb = _select_text_color(line)
    background_rgb = _select_background_color(line)
    text_color = _color_to_pdf_tuple(text_rgb, fallback=(0, 0, 0))
    background_color = _color_to_pdf_tuple(background_rgb, fallback=(255, 255, 255))
    return _LineAnnotationStyle(
        font_name=font_name,
        font_size=font_size,
        text_color=text_color,
        background_color=background_color,
        font_handle=font_handle,
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


def _resolve_font_handle(font_name: str) -> object:
    key = (font_name or "helv").lower()
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    target_name = font_name or "helv"
    try:
        handle = pymupdf.Font(target_name)  # type: ignore[attr-defined]
    except Exception:
        target_name = "helv"
        key = target_name
        handle = _FONT_CACHE.get(key)
        if handle is None:
            handle = pymupdf.Font(target_name)  # type: ignore[attr-defined]
            _FONT_CACHE[key] = handle
        return handle
    _FONT_CACHE[key] = handle
    return handle


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


def _prepare_annotation_rect(
    bbox: Tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    font_size: float,
):
    padding = max(0.8, font_size * 0.15)
    rect = pymupdf.Rect(bbox)  # type: ignore[attr-defined]
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


def _ensure_rect_capacity(
    rect: "pymupdf.Rect",
    text: str,
    style: _LineAnnotationStyle,
    page_width: float,
) -> "pymupdf.Rect":
    if not text or not style.font_name:
        return rect
    text_length = _estimate_text_width(text, style)
    if text_length is None:
        return rect
    available = rect.width
    if text_length <= available:
        return rect
    deficit = text_length - available
    extra_padding = max(0.0, style.font_size * 0.2)
    deficit += extra_padding
    space_right = max(0.0, page_width - rect.x1)
    extend_right = min(space_right, deficit)
    rect.x1 += extend_right
    remaining = deficit - extend_right
    if remaining > 0:
        rect.x0 = max(0.0, rect.x0 - remaining)
    if rect.x1 <= rect.x0:
        rect.x1 = min(page_width, rect.x0 + text_length + extra_padding)
    return rect


def _estimate_text_width(text: str, style: _LineAnnotationStyle) -> Optional[float]:
    if not text.strip():
        return None
    font_handle = style.font_handle
    if font_handle is None:
        font_handle = _resolve_font_handle(style.font_name or "helv")
    try:
        return float(font_handle.text_length(text, fontsize=style.font_size))  # type: ignore[attr-defined]
    except Exception:
        return None


def _draw_text_overlay(
    page: object,
    rect: "pymupdf.Rect",
    text: str,
    style: _LineAnnotationStyle,
    *,
    annotation_index: int,
    page_rect: "pymupdf.Rect",
) -> str:
    font_handle = style.font_handle or _resolve_font_handle(style.font_name or "helv")
    background = style.background_color or (1.0, 1.0, 1.0)
    text_color = style.text_color or (0.0, 0.0, 0.0)
    working_rect = pymupdf.Rect(rect)  # type: ignore[attr-defined]
    max_attempts = 8
    growth_step = max(style.font_size * 0.9, 4.0)
    writer_to_use: Optional[object] = None
    final_rect = pymupdf.Rect(working_rect)  # type: ignore[attr-defined]
    for _ in range(max_attempts):
        writer = pymupdf.TextWriter(page_rect)  # type: ignore[attr-defined]
        try:
            leftovers = writer.fill_textbox(
                working_rect,
                text,
                font=font_handle,
                fontsize=style.font_size,
                align=_TEXT_ALIGN_LEFT,
                warn=None,
            )
        except ValueError:
            leftovers = [("overflow", 0.0)]
        writer_to_use = writer
        final_rect = pymupdf.Rect(working_rect)  # type: ignore[attr-defined]
        if not leftovers:
            break
        if working_rect.y0 <= 0.0 and working_rect.y1 >= page_rect.height:
            break
        extend_down = min(growth_step, page_rect.height - working_rect.y1)
        working_rect.y1 += extend_down
        remaining = growth_step - extend_down
        if remaining > 0:
            working_rect.y0 = max(0.0, working_rect.y0 - remaining)
    _draw_background_rect(page, final_rect, background)
    if writer_to_use is not None:
        writer_to_use.write_text(page, color=text_color, overlay=True)  # type: ignore[attr-defined]
    return f"DARU_DRAWN_{annotation_index:05d}"


def _draw_background_rect(
    page: object,
    rect: "pymupdf.Rect",
    color: Optional[Tuple[float, float, float]],
) -> None:
    if not color:
        return
    try:
        page.draw_rect(  # type: ignore[attr-defined]
            rect,
            color=color,
            fill=color,
            overlay=True,
        )
    except Exception:
        pass


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
                source_raw = (row.get("text_en") or "").strip()
                if not source_raw:
                    continue
                key = _prepare_source_text(source_raw) or source_raw
                cache[key] = row.get("text_ru") or ""
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
                writer.writerow([_prepare_source_text(source) or source, translated])
    except Exception as exc:
        log(f"PDF: не удалось обновить кэш переводов ({exc})")
    else:
        log(
            f"PDF: обновлён кэш переводов ({applied} новых значений): {path}"
        )
