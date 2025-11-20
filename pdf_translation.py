from __future__ import annotations

import io
import re
import tempfile
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from auto_translation import TranslationEngine

try:  # pragma: no cover - optional dependency
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None  # type: ignore

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

try:  # pragma: no cover - optional dependency
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.pdfbase import pdfmetrics  # type: ignore
    from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    A4 = None  # type: ignore
    pdfmetrics = None  # type: ignore
    TTFont = None  # type: ignore
    canvas = None  # type: ignore

PdfLog = Callable[[str], None]

PDF_TYPE_SCANNED = "scanned"
PDF_TYPE_NATIVE = "native"

MAX_PARAGRAPH = 900



def translate_pdf(
    *,
    input_path: Path,
    output_path: Path,
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
        text_pages = _extract_pdf_text(searchable_source, log)
        if not any(text_pages):
            raise RuntimeError("Не удалось извлечь текст из PDF документа")

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
        log(f"Инициализирован движок перевода: {translator.backend_name()}")
        segments, mapping = _linearize_pages(text_pages)
        log(f"Готовим {len(segments)} блоков текста к переводу")
        total_segments = len(segments)
        translated_segments: List[str] = []
        if total_segments == 0:
            log("Начинаем перевод... [100%]")
            log("Перевод завершён")
        else:
            log("Начинаем перевод... [0%]")
            chunk_size = max(1, total_segments // 20)
            processed = 0
            for start in range(0, total_segments, chunk_size):
                chunk = segments[start : start + chunk_size]
                translated_chunk = translator.translate_many(chunk)
                translated_segments.extend(translated_chunk)
                processed += len(chunk)
                percent = min(100, int(processed / total_segments * 100))
                log(f"Начинаем перевод... [{percent}%]")
            if processed < total_segments:
                log("Начинаем перевод... [100%]")
            log("Перевод завершён")
        translated_pages = _rebuild_pages(translated_segments, mapping, len(text_pages))
        _export_translated_pdf(translated_pages, output_path, style_font, log)

    return {
        "output_path": output_path,
        "backend": translator.backend_name(),
        "pages": len(translated_pages),
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


def _extract_pdf_text(pdf_path: Path, log: PdfLog) -> List[str]:
    _require_dependency(pdfplumber, "Установите pdfplumber (pip install pdfplumber)")
    pages_text: List[str] = []
    log("Извлекаем текст со всех страниц PDF...")
    with pdfplumber.open(str(pdf_path)) as pdf:  # type: ignore[attr-defined]
        total = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text(layout=True) or ""
            cleaned = _clean_text(raw_text)
            pages_text.append(cleaned)
            log(f"Извлечён текст: страница {idx}/{total}")
    return pages_text


def _clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _split_block(block: str) -> List[str]:
    if len(block) <= MAX_PARAGRAPH:
        return [block.strip()]
    chunks: List[str] = []
    start = 0
    while start < len(block):
        end = min(len(block), start + MAX_PARAGRAPH)
        if end < len(block):
            candidate = block.rfind(" ", start, end)
            if candidate - start > 100:
                end = candidate
        chunk = block[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _linearize_pages(pages: Sequence[str]) -> Tuple[List[str], List[int]]:
    segments: List[str] = []
    mapping: List[int] = []
    for page_idx, text in enumerate(pages):
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        if not blocks and text.strip():
            blocks = [text.strip()]
        for block in blocks:
            for chunk in _split_block(block):
                segments.append(chunk)
                mapping.append(page_idx)
    return segments, mapping


def _rebuild_pages(translated: Sequence[str], mapping: Sequence[int], total_pages: int) -> List[str]:
    page_buckets: List[List[str]] = [[] for _ in range(total_pages)]
    for chunk, page_idx in zip(translated, mapping):
        block = chunk.strip()
        if block:
            page_buckets[page_idx].append(block)
    return ["\n\n".join(blocks).strip() for blocks in page_buckets]


def _resolve_font(style_font: Optional[str]) -> Optional[Tuple[str, Path]]:
    if not style_font:
        return None
    candidates = [
        Path(style_font),
        Path.cwd() / style_font,
        Path(__file__).resolve().parent / style_font,
    ]
    for candidate in candidates:
        if candidate.exists():
            font_id = candidate.stem.replace(" ", "_")
            return font_id, candidate
    return None


def _export_translated_pdf(
    pages: Sequence[str],
    output_path: Path,
    style_font: Optional[str],
    log: PdfLog,
) -> None:
    _require_dependency(canvas, "Установите reportlab (pip install reportlab)")
    _require_dependency(A4, "Установите reportlab (pip install reportlab)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4  # type: ignore[assignment]
    c = canvas.Canvas(str(output_path), pagesize=A4)  # type: ignore[attr-defined]
    font_name = "Helvetica"
    resolved_font = _resolve_font(style_font)
    if resolved_font and pdfmetrics and TTFont:
        font_name = resolved_font[0]
        try:
            if font_name not in pdfmetrics.getRegisteredFontNames():  # type: ignore[attr-defined]
                pdfmetrics.registerFont(TTFont(font_name, str(resolved_font[1])))  # type: ignore[attr-defined]
        except Exception:
            font_name = "Helvetica"

    total_pages = len(pages)
    for page_idx, page_text in enumerate(pages, start=1):
        text_object = c.beginText(42, height - 60)
        text_object.setFont(font_name, 12)
        lines = page_text.splitlines() or [""]
        for line in lines:
            if text_object.getY() <= 60:
                c.drawText(text_object)
                c.showPage()
                text_object = c.beginText(42, height - 60)
                text_object.setFont(font_name, 12)
            text_object.textLine(line)
        c.drawText(text_object)
        if page_idx < total_pages:
            c.showPage()
        log(f"PDF: отрисована переведённая страница {page_idx}/{total_pages}")
    c.save()
    log(f"PDF: файл сохранён: {output_path}")


def _detect_poppler_path() -> Optional[str]:
    for env_name in ("POPPLER_PATH", "POPPLER_BIN", "POPPLER_HOME"):
        candidate = os.environ.get(env_name)
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path)
    return None
