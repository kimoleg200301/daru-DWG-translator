"""Native PDF translation based on the existing searchable text layer."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import ORIGINAL_FONT_VALUE, normalize_style_font
from ..translation.analysis import CodexAnalysisSession
from ..translation.checkpoint import (
    TranslationCheckpointStore,
    default_checkpoint_path,
)
from ..translation.engine import TranslationEngine

try:  # pragma: no cover - dependency guard
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - dependency guard
    fitz = None  # type: ignore


PdfLog = Callable[[str], None]
FallbackPageTranslator = Callable[[int], bytes]
BBox = Tuple[float, float, float, float]

MAX_BATCH_ITEMS = 24
MAX_BATCH_CHARS = 12_000
MAX_DAMAGED_RATIO = 0.10
MIN_NATIVE_FONT_SCALE = 0.70
FORCED_FONT_SCALE = 0.35

NATIVE_PDF_SYSTEM_PROMPT = (
    "You are a professional technical translator of elevator operation and maintenance manuals. "
    "Translate the provided values from {source_lang} to {target_lang}. "
    "The values belong to the SAME searchable PDF document and are supplied in reading order. "
    "Use precise safety terminology, preserve the force of warnings and prohibitions, and keep "
    "terminology consistent across headings, paragraphs, lists, captions, and table cells. "
    "Do not add explanations, omit content, or rewrite technical meaning. Preserve company and "
    "brand names, model identifiers, addresses, URLs, phone numbers, units, and every token matching "
    "'[[DARU_P_*]]' exactly and in the same position. Ignore source line wrapping and repair only "
    "obvious line-break hyphenation. Each item may contain a type field describing its document role. "
    "Respond with strict JSON: "
    '{"translations": [{"id": "<id>", "text": "<translated>"}, ...]}'
)

_LIST_PREFIX_RE = re.compile(r"^\s*(?:[•●▪◆◇\-–—]|\(?\d{1,3}[.)]|[A-Za-z][.)])\s+")
_TOC_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")
_DAMAGED_RE = re.compile(r"[\uFFFD\u0000-\u0008\u000B\u000C\u000E-\u001F]")
_PLACEHOLDER_RE = re.compile(r"\[\[DARU_P_[A-Z]+\]\]")
_CONTACT_LINE_RE = re.compile(
    r"\b(?:address|head\s+office|factory|zip|p\.?\s*c\.?|tel\.?|fax|e-?mail)\b",
    re.IGNORECASE,
)
_LOCATION_LINE_RE = re.compile(
    r"\b(?:street|st\.|road|rd\.|avenue|district|province|building|bldg)\b",
    re.IGNORECASE,
)
_PROTECTED_VALUE_RE = re.compile(
    r"https?://[^\s<>]+"
    r"|www\.[^\s<>]+"
    r"|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r"|(?:\+?\d[\d\s().-]{6,}\d)"
    r"|(?:\b(?=[A-Z0-9./_-]*[A-Z])(?=[A-Z0-9./_-]*\d)[A-Z0-9][A-Z0-9./_-]{2,}\b)"
    r"|(?:\b\d+(?:[.,]\d+)?\s?(?:mm|cm|m|km|kg|t|V|A|Hz|kW|W|MPa|Pa|°C|%)\b)"
    r"|(?:\.{3,}\s*\d+)"
    r"|(?:\b\d+(?:[.,]\d+)?\b)"
)
_COMPANY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z&.-]*\s+){0,5}"
    r"(?:Co\.?\s*,?\s*Ltd\.?|Company|Corporation|Corp\.?|Inc\.?|LLC)\b",
    re.IGNORECASE,
)


@dataclass
class ProtectedText:
    value: str
    replacements: Dict[str, str]


@dataclass
class NativeTextUnit:
    page_index: int
    stable_id: str
    role: str
    bbox: BBox
    source_text: str
    protected: ProtectedText
    font_size: float
    font_name: str
    color: int
    bold: bool
    rotation: int
    alignment: str = "left"
    translated_text: Optional[str] = None
    rendered_bbox: Optional[BBox] = None
    font_scale: float = 1.0
    cached: bool = False
    layout_warning: Optional[str] = None


@dataclass
class NativePage:
    page_index: int
    is_native: bool
    damaged_ratio: float
    units: List[NativeTextUnit] = field(default_factory=list)


@dataclass
class NativeTranslationResult:
    backend: str
    native_pages: List[int]
    ocr_pages: List[int]
    units: List[NativeTextUnit]
    layout_warnings: List[Dict[str, Any]]


def translate_native_pdf(
    *,
    input_path: Path,
    output_path: Path,
    translator_name: str,
    source_lang: str,
    target_lang: str,
    log: PdfLog,
    layer_json_path: Optional[Path] = None,
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
    fallback_page_translator: Optional[FallbackPageTranslator] = None,
    checkpoint_path: Optional[Path] = None,
    resume_policy: str = "auto",
) -> Dict[str, object]:
    """Translate searchable PDF pages while preserving their native page content."""

    _require_fitz()
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser().with_suffix(".pdf")
    if not input_path.exists():
        raise FileNotFoundError(f"PDF не найден: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError("Нативный режим поддерживает только PDF")
    if os.path.normcase(str(input_path.resolve())) == os.path.normcase(str(output_path.resolve())):
        raise ValueError("Исходный и выходной PDF должны различаться")

    file_hash = _file_sha256(input_path)
    checkpoint_store = TranslationCheckpointStore(
        path=checkpoint_path or default_checkpoint_path(input_path, "pdf-native"),
        job_type="pdf-native",
        document_sha256=file_hash,
        source_lang=source_lang,
        target_lang=target_lang,
        translator=translator_name,
        resume_policy=resume_policy,
        log=log,
    )
    document = fitz.open(str(input_path))
    try:
        pages = [_extract_page(document[index], index, source_lang) for index in range(len(document))]
        native_pages = [page.page_index + 1 for page in pages if page.is_native]
        ocr_pages = [page.page_index + 1 for page in pages if not page.is_native]
        units = [unit for page in pages if page.is_native for unit in page.units]

        log(
            f"PDF: текстовый слой пригоден на {len(native_pages)}/{len(pages)} страницах"
        )
        if ocr_pages:
            log(
                "PDF: OCR fallback для страниц: "
                + ", ".join(str(page_number) for page_number in ocr_pages)
            )
            if fallback_page_translator is None:
                raise RuntimeError(
                    "PDF содержит страницы без пригодного текстового слоя, "
                    "но OCR fallback не настроен"
                )

        cached = _checkpoint_native_cache(checkpoint_store, units, log)
        layer_cached = _load_translation_layer(layer_json_path, file_hash, units, log)
        for key, value in layer_cached.items():
            cached.setdefault(key, value)
        backend = _translate_units(
            units,
            cached,
            translator_name=translator_name,
            source_lang=source_lang,
            target_lang=target_lang,
            deepl_key=deepl_key,
            openai_key=openai_key,
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
            log=log,
            checkpoint_store=checkpoint_store,
        )

        layout_warnings = _render_native_pages(document, pages, style_font, log)
        fallback_documents: Dict[int, bytes] = {}
        for page_index in (page.page_index for page in pages if not page.is_native):
            fallback_documents[page_index] = fallback_page_translator(page_index)

        _save_output_document(document, fallback_documents, output_path)
        _save_translation_layer(
            layer_json_path,
            file_hash=file_hash,
            pages=pages,
            native_pages=native_pages,
            ocr_pages=ocr_pages,
            log=log,
        )
    finally:
        document.close()

    result = NativeTranslationResult(
        backend=backend,
        native_pages=native_pages,
        ocr_pages=ocr_pages,
        units=units,
        layout_warnings=layout_warnings,
    )
    log(f"PDF: файл сохранён: {output_path}")
    return {
        "output_path": output_path,
        "backend": result.backend,
        "pages": len(pages),
        "job_type": "pdf",
        "processing_mode": "native" if not result.ocr_pages else "hybrid",
        "native_pages": result.native_pages,
        "ocr_pages": result.ocr_pages,
        "ocr_performed": bool(result.ocr_pages),
        "translated_blocks": len(result.units),
        "layout_warnings": result.layout_warnings,
        "checkpoint_path": checkpoint_store.path,
    }


def _require_fitz() -> None:
    if fitz is None:
        raise RuntimeError(
            "Для PDF с текстовым слоем требуется PyMuPDF>=1.27.2.2 "
            "(pip install --upgrade pymupdf)"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_page(page: Any, page_index: int, source_lang: str) -> NativePage:
    data = page.get_text("dict", sort=True)
    text_blocks = [block for block in data.get("blocks", []) if block.get("type") == 0]
    raw_text = "".join(
        str(span.get("text", ""))
        for block in text_blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    )
    damaged_ratio = _damaged_character_ratio(raw_text)
    page_has_text = _text_needs_translation(raw_text, source_lang)
    if not page_has_text or damaged_ratio > MAX_DAMAGED_RATIO:
        return NativePage(
            page_index=page_index,
            is_native=False,
            damaged_ratio=damaged_ratio,
        )

    page_font_sizes = [
        float(span.get("size", 0) or 0)
        for block in text_blocks
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if str(span.get("text", "")).strip() and float(span.get("size", 0) or 0) > 0
    ]
    median_font_size = median(page_font_sizes) if page_font_sizes else 10.0
    table_cells = _extract_table_cells(page, page_index, source_lang, median_font_size)
    accepted_cell_rects = [fitz.Rect(unit.bbox) for unit in table_cells]
    units = list(table_cells)

    for block in text_blocks:
        block_units = _units_from_block(
            block,
            page_index=page_index,
            source_lang=source_lang,
            median_font_size=median_font_size,
        )
        for unit in block_units:
            unit_rect = fitz.Rect(unit.bbox)
            if any(
                _rect_contains_center(cell_rect, unit_rect)
                for cell_rect in accepted_cell_rects
            ):
                continue
            units.append(unit)

    units.sort(key=lambda unit: (unit.bbox[1], unit.bbox[0], unit.stable_id))
    return NativePage(
        page_index=page_index,
        is_native=True,
        damaged_ratio=damaged_ratio,
        units=units,
    )


def _extract_table_cells(
    page: Any,
    page_index: int,
    source_lang: str,
    median_font_size: float,
) -> List[NativeTextUnit]:
    units: List[NativeTextUnit] = []
    try:
        finder = page.find_tables()
    except Exception:
        return units

    for table_index, table in enumerate(getattr(finder, "tables", [])):
        try:
            values = table.extract()
        except Exception:
            continue
        nonempty = [
            value
            for row in values
            for value in row
            if isinstance(value, str) and value.strip()
        ]
        total_cells = max(1, int(table.row_count) * int(table.col_count))
        density = len(nonempty) / total_cells
        if table.row_count < 2 or table.col_count < 2 or len(nonempty) < 4 or density < 0.4:
            continue

        for row_index, row in enumerate(table.rows):
            for column_index, cell in enumerate(row.cells):
                if cell is None:
                    continue
                try:
                    source_text = str(values[row_index][column_index] or "").strip()
                except (IndexError, TypeError):
                    source_text = str(page.get_textbox(fitz.Rect(cell)) or "").strip()
                source_text = _normalize_block_text(source_text.splitlines())
                if not _text_needs_translation(source_text, source_lang):
                    continue
                bbox = _coerce_bbox(cell)
                if bbox is None:
                    continue
                style = _style_from_clip(page, fitz.Rect(bbox), median_font_size)
                role = "CELL"
                units.append(
                    _make_unit(
                        page_index=page_index,
                        bbox=bbox,
                        role=role,
                        source_text=source_text,
                        font_size=style["font_size"],
                        font_name=style["font_name"],
                        color=style["color"],
                        bold=style["bold"] or row_index == 0,
                        rotation=style["rotation"],
                        alignment=style["alignment"],
                        stable_salt=f"table:{table_index}:{row_index}:{column_index}",
                    )
                )
    return units


def _units_from_block(
    block: Dict[str, Any],
    *,
    page_index: int,
    source_lang: str,
    median_font_size: float,
) -> List[NativeTextUnit]:
    fragments = _split_block_fragments(block)
    units: List[NativeTextUnit] = []
    for fragment_index, fragment in enumerate(fragments):
        lines = fragment["lines"]
        line_texts = [line["text"] for line in lines]
        source_text = _normalize_block_text(line_texts)
        if not _text_needs_translation(source_text, source_lang):
            continue
        spans = [
            span
            for line in lines
            for span in line["spans"]
            if str(span.get("text", "")).strip()
        ]
        font_size, font_name, color, bold = _dominant_span_style(
            spans,
            median_font_size,
        )
        rotation = _rotation_from_lines(lines)
        role = _classify_role(source_text, font_size, median_font_size)
        units.append(
            _make_unit(
                page_index=page_index,
                bbox=fragment["bbox"],
                role=role,
                source_text=source_text,
                font_size=font_size,
                font_name=font_name,
                color=color,
                bold=bold,
                rotation=rotation,
                alignment="left",
                stable_salt=f"fragment:{fragment_index}",
            )
        )
    return units


def _split_block_fragments(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for line in block.get("lines", []):
        segments.extend(_split_line_segments(line))
    if not segments:
        return []
    segments.sort(key=lambda segment: (segment["bbox"][1], segment["bbox"][0]))

    fragments: List[Dict[str, Any]] = []
    for segment in segments:
        segment_rect = fitz.Rect(segment["bbox"])
        best_fragment = None
        best_score = -1.0
        for fragment in fragments:
            previous_rect = fitz.Rect(fragment["lines"][-1]["bbox"])
            if _vertical_overlap_ratio(previous_rect, segment_rect) > 0.5:
                continue
            vertical_gap = segment_rect.y0 - previous_rect.y1
            max_gap = max(
                6.0,
                float(segment["font_size"]) * 1.8,
                float(fragment["lines"][-1]["font_size"]) * 1.8,
            )
            if vertical_gap < -1.0 or vertical_gap > max_gap:
                continue
            overlap = _horizontal_overlap_ratio(
                fitz.Rect(fragment["bbox"]),
                segment_rect,
            )
            left_distance = abs(segment_rect.x0 - float(fragment["bbox"][0]))
            if overlap < 0.15 and left_distance > max(24.0, segment["font_size"] * 3):
                continue
            score = overlap - left_distance / 10_000
            if score > best_score:
                best_score = score
                best_fragment = fragment

        if best_fragment is None:
            fragments.append(
                {
                    "bbox": segment["bbox"],
                    "lines": [segment],
                }
            )
            continue
        best_fragment["lines"].append(segment)
        best_fragment["bbox"] = _bbox_union(
            best_fragment["bbox"],
            segment["bbox"],
        )
    return fragments


def _split_line_segments(line: Dict[str, Any]) -> List[Dict[str, Any]]:
    spans = [
        span
        for span in line.get("spans", [])
        if _coerce_span_bbox(span.get("bbox")) is not None
    ]
    if not spans:
        return []
    spans.sort(key=lambda span: float(span["bbox"][0]))
    groups: List[List[Dict[str, Any]]] = [[spans[0]]]
    previous_bbox = _coerce_span_bbox(spans[0].get("bbox"))
    for span in spans[1:]:
        bbox = _coerce_span_bbox(span.get("bbox"))
        if bbox is None or previous_bbox is None:
            continue
        gap = bbox[0] - previous_bbox[2]
        font_size = max(
            float(span.get("size", 0) or 0),
            float(groups[-1][-1].get("size", 0) or 0),
            1.0,
        )
        if gap > max(36.0, font_size * 4):
            groups.append([])
        groups[-1].append(span)
        previous_bbox = bbox

    segments: List[Dict[str, Any]] = []
    for group in groups:
        bboxes = [
            bbox
            for span in group
            if (bbox := _coerce_bbox(span.get("bbox"))) is not None
        ]
        if not bboxes:
            continue
        text = "".join(str(span.get("text", "")) for span in group).strip()
        if not text:
            continue
        segments.append(
            {
                "bbox": _bbox_union_many(bboxes),
                "text": text,
                "spans": group,
                "dir": line.get("dir", (1.0, 0.0)),
                "font_size": median(
                    [float(span.get("size", 10) or 10) for span in group]
                ),
            }
        )
    return segments


def _make_unit(
    *,
    page_index: int,
    bbox: BBox,
    role: str,
    source_text: str,
    font_size: float,
    font_name: str,
    color: int,
    bold: bool,
    rotation: int,
    alignment: str,
    stable_salt: str = "",
) -> NativeTextUnit:
    normalized_bbox = ",".join(f"{value:.2f}" for value in bbox)
    stable_payload = f"{page_index}|{normalized_bbox}|{source_text}|{stable_salt}"
    stable_id = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:20]
    return NativeTextUnit(
        page_index=page_index,
        stable_id=stable_id,
        role=role,
        bbox=bbox,
        source_text=source_text,
        protected=_protect_text(source_text),
        font_size=max(4.0, font_size),
        font_name=font_name,
        color=color,
        bold=bold,
        rotation=rotation,
        alignment=alignment,
    )


def _normalize_block_text(lines: Sequence[str]) -> str:
    clean_lines = [re.sub(r"\s+", " ", line).strip() for line in lines if line.strip()]
    if not clean_lines:
        return ""
    if len(clean_lines) == 1:
        return clean_lines[0]

    output = clean_lines[0]
    preserve_breaks = bool(_LIST_PREFIX_RE.match(clean_lines[0]))
    for current in clean_lines[1:]:
        if output.endswith("-") and current[:1].islower():
            output = output[:-1] + current
        elif preserve_breaks or _LIST_PREFIX_RE.match(current):
            output += "\n" + current
            preserve_breaks = True
        else:
            output += " " + current
    return output.strip()


def _classify_role(source_text: str, font_size: float, median_font_size: float) -> str:
    stripped = source_text.strip()
    if _LIST_PREFIX_RE.match(stripped):
        return "LIST"
    if font_size >= max(14.0, median_font_size * 1.35):
        return "TITLE"
    if len(stripped) <= 100 and font_size <= median_font_size * 0.85:
        return "CAPTION"
    return "TEXT"


def _dominant_span_style(
    spans: Sequence[Dict[str, Any]],
    fallback_size: float,
) -> Tuple[float, str, int, bool]:
    if not spans:
        return fallback_size, "sans-serif", 0, False
    weighted: List[Tuple[int, Dict[str, Any]]] = [
        (max(1, len(str(span.get("text", "")).strip())), span) for span in spans
    ]
    dominant = max(weighted, key=lambda item: item[0])[1]
    sizes = [
        float(span.get("size", fallback_size) or fallback_size)
        for span in spans
        for _ in range(max(1, min(20, len(str(span.get("text", "")).strip()))))
    ]
    font_name = str(dominant.get("font") or "sans-serif")
    flags = int(dominant.get("flags", 0) or 0)
    bold = bool(flags & 16) or "bold" in font_name.lower()
    return (
        float(median(sizes)) if sizes else fallback_size,
        font_name,
        int(dominant.get("color", 0) or 0),
        bold,
    )


def _style_from_clip(page: Any, rect: Any, fallback_size: float) -> Dict[str, Any]:
    try:
        data = page.get_text("dict", clip=rect, sort=True)
    except Exception:
        data = {"blocks": []}
    lines = [
        line
        for block in data.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
    ]
    spans = [
        span for line in lines for span in line.get("spans", []) if str(span.get("text", "")).strip()
    ]
    size, name, color, bold = _dominant_span_style(spans, fallback_size)
    text_rects = [
        fitz.Rect(span["bbox"])
        for span in spans
        if _coerce_bbox(span.get("bbox")) is not None
    ]
    alignment = "left"
    if text_rects:
        union = text_rects[0]
        for text_rect in text_rects[1:]:
            union |= text_rect
        left_gap = max(0.0, union.x0 - rect.x0)
        right_gap = max(0.0, rect.x1 - union.x1)
        if abs(left_gap - right_gap) <= max(2.0, rect.width * 0.08):
            alignment = "center"
        elif left_gap > right_gap * 2:
            alignment = "right"
    return {
        "font_size": size,
        "font_name": name,
        "color": color,
        "bold": bold,
        "rotation": _rotation_from_lines(lines),
        "alignment": alignment,
    }


def _rotation_from_lines(lines: Sequence[Dict[str, Any]]) -> int:
    for line in lines:
        direction = line.get("dir")
        if not isinstance(direction, (tuple, list)) or len(direction) != 2:
            continue
        dx, dy = float(direction[0]), float(direction[1])
        angle = int(round(math.degrees(math.atan2(-dy, dx)) / 90.0) * 90) % 360
        return angle if angle in {0, 90, 180, 270} else 0
    return 0


def _text_needs_translation(text: str, source_lang: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False
    if (source_lang or "").lower().startswith("en"):
        return len(re.findall(r"[A-Za-z]", stripped)) >= 2
    return sum(1 for char in stripped if char.isalpha()) >= 2


def _damaged_character_ratio(text: str) -> float:
    nonspace = [char for char in str(text or "") if not char.isspace()]
    if not nonspace:
        return 1.0
    damaged = len(_DAMAGED_RE.findall("".join(nonspace)))
    return damaged / len(nonspace)


def _protect_text(text: str) -> ProtectedText:
    replacements: Dict[str, str] = {}

    def next_token(value: str) -> str:
        token = f"[[DARU_P_{_alpha_index(len(replacements))}]]"
        replacements[token] = value
        return token

    protected_lines: List[str] = []
    for line in text.splitlines() or [text]:
        if _looks_like_address_line(line):
            protected_lines.append(next_token(line))
            continue

        def replace_company(match: re.Match[str]) -> str:
            return next_token(match.group(0))

        company_protected = _COMPANY_RE.sub(replace_company, line)

        def replace_value(match: re.Match[str]) -> str:
            return next_token(match.group(0))

        protected_lines.append(_PROTECTED_VALUE_RE.sub(replace_value, company_protected))
    return ProtectedText(value="\n".join(protected_lines), replacements=replacements)


def _looks_like_address_line(line: str) -> bool:
    if _CONTACT_LINE_RE.search(line):
        return True
    return bool(re.search(r"\d", line) and _LOCATION_LINE_RE.search(line))


def _alpha_index(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _restore_protected_text(protected: ProtectedText, translated: str) -> Optional[str]:
    candidate = str(translated or "").strip()
    if not candidate:
        return None
    found = _PLACEHOLDER_RE.findall(candidate)
    expected = list(protected.replacements)
    if sorted(found) != sorted(expected) or len(found) != len(expected):
        return None
    for token, original in protected.replacements.items():
        candidate = candidate.replace(token, original)
    if _PLACEHOLDER_RE.search(candidate):
        return None
    return candidate.strip()


def _translation_batches(units: Sequence[NativeTextUnit]) -> Iterable[List[NativeTextUnit]]:
    current: List[NativeTextUnit] = []
    current_chars = 0
    for unit in units:
        unit_chars = len(unit.protected.value)
        if current and (
            len(current) >= MAX_BATCH_ITEMS or current_chars + unit_chars > MAX_BATCH_CHARS
        ):
            yield current
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += unit_chars
    if current:
        yield current


def _translate_units(
    units: Sequence[NativeTextUnit],
    cached: Dict[str, str],
    *,
    translator_name: str,
    source_lang: str,
    target_lang: str,
    deepl_key: Optional[str],
    openai_key: Optional[str],
    openai_model: Optional[str],
    openai_base_url: Optional[str],
    openai_project: Optional[str],
    openai_temperature: float,
    openai_reasoning_effort: Optional[str],
    openai_verbosity: Optional[str],
    openai_strict_mode: Optional[str],
    openai_strict_value: Optional[float],
    log: PdfLog,
    codex_cli_path: Optional[str] = None,
    codex_model: Optional[str] = None,
    codex_reasoning_effort: Optional[str] = None,
    codex_analysis_model: Optional[str] = None,
    codex_analysis_reasoning_effort: Optional[str] = None,
    codex_analysis_session: Optional[CodexAnalysisSession] = None,
    codex_timeout_seconds: int = 300,
    checkpoint_store: Optional[TranslationCheckpointStore] = None,
) -> str:
    pending: List[NativeTextUnit] = []
    for unit in units:
        cached_value = cached.get(unit.stable_id)
        if cached_value:
            unit.translated_text = cached_value
            unit.cached = True
        else:
            pending.append(unit)

    if not pending:
        if not units:
            return "not-required"
        if checkpoint_store is not None and checkpoint_store.blocks:
            return "cached-checkpoint"
        return "cached-layer"

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
        system_prompt_template=NATIVE_PDF_SYSTEM_PROMPT,
    )
    translator.set_document_context(
        [unit.protected.value for unit in units],
        context_label="NATIVE PDF MANUAL",
        entity_types={unit.protected.value: unit.role for unit in units},
    )
    backend = translator.backend_name()
    if checkpoint_store is not None:
        checkpoint_store.set_backend(backend)
    log(f"PDF: инициализирован нативный переводчик: {backend}")

    processed = 0
    for batch in _translation_batches(pending):
        values = [unit.protected.value for unit in batch]
        try:
            translated = translator.translate_many(values)
        except Exception as exc:
            if translator.backend_name() == "codex-cli":
                raise
            log(f"PDF: пакетный перевод не выполнен ({exc}), повторяем блоки отдельно")
            translated = []

        if len(translated) != len(batch):
            translated = [""] * len(batch)

        saved_batch = False
        for unit, translated_value in zip(batch, translated):
            restored = _restore_protected_text(unit.protected, translated_value)
            if restored is None:
                restored = _retry_single_translation(translator, unit, log)
            unit.translated_text = restored or unit.source_text
            if (
                checkpoint_store is not None
                and restored
                and _canonical_text(restored) != _canonical_text(unit.source_text)
            ):
                checkpoint_store.upsert(
                    namespace="pdf-native",
                    block_id=unit.stable_id,
                    source_text=unit.source_text,
                    translated_text=restored,
                    extra={
                        "page_index": unit.page_index,
                        "bbox": list(unit.bbox),
                        "role": unit.role,
                    },
                )
                saved_batch = True
        if saved_batch and checkpoint_store is not None:
            checkpoint_store.save()

        processed += len(batch)
        percent = int(processed / len(pending) * 100)
        log(f"PDF: перевод текстового слоя... [{percent}%]")
    return backend


def _retry_single_translation(
    translator: TranslationEngine,
    unit: NativeTextUnit,
    log: PdfLog,
) -> Optional[str]:
    try:
        translated = translator.translate_many([unit.protected.value])
    except Exception as exc:
        if translator.backend_name() == "codex-cli":
            raise
        log(
            f"PDF: блок {unit.stable_id} не переведён после повторной попытки: {exc}"
        )
        return None
    if len(translated) != 1:
        log(f"PDF: блок {unit.stable_id} вернул неверное число переводов")
        return None
    restored = _restore_protected_text(unit.protected, translated[0])
    if restored is None:
        log(f"PDF: блок {unit.stable_id} повредил защищённые значения; оставлен оригинал")
    return restored


def _render_native_pages(
    document: Any,
    pages: Sequence[NativePage],
    style_font: Optional[str],
    log: PdfLog,
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    original_mode = normalize_style_font(style_font or "") == ORIGINAL_FONT_VALUE
    fallback_resource = _resolve_font_resource(None if original_mode else style_font)
    original_resolver = (
        _OriginalFontResolver(document, fallback_resource, log)
        if original_mode
        else None
    )
    try:
        for page_model in pages:
            if not page_model.is_native:
                continue
            page = document[page_model.page_index]
            changed = [
                unit
                for unit in page_model.units
                if unit.translated_text
                and _canonical_text(unit.translated_text) != _canonical_text(unit.source_text)
            ]
            if not changed:
                continue

            font_resources = {
                unit.stable_id: (
                    original_resolver.resolve(unit)
                    if original_resolver is not None
                    else fallback_resource
                )
                for unit in changed
            }
            for unit in changed:
                page.add_redact_annot(
                    fitz.Rect(unit.bbox),
                    fill=None,
                    cross_out=False,
                )
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                text=fitz.PDF_REDACT_TEXT_REMOVE,
            )

            occupied = [fitz.Rect(unit.bbox) for unit in page_model.units]
            occupied.extend(_page_nontext_occupied(page))
            for unit in changed:
                warning = _insert_unit(
                    page,
                    unit,
                    occupied=occupied,
                    font_resource=font_resources[unit.stable_id],
                )
                if warning:
                    warnings.append(warning)
                    log(
                        f"PDF: предупреждение верстки, стр. {unit.page_index + 1}, "
                        f"блок {unit.stable_id}, масштаб {unit.font_scale:.2f}"
                    )
    finally:
        if original_resolver is not None:
            original_resolver.close()
    return warnings


def _page_nontext_occupied(page: Any) -> List[Any]:
    occupied: List[Any] = []
    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is None:
                continue
            drawing_rect = fitz.Rect(rect)
            drawing_rect.x0 -= 0.75
            drawing_rect.y0 -= 0.75
            drawing_rect.x1 += 0.75
            drawing_rect.y1 += 0.75
            occupied.append(drawing_rect)
    except Exception:
        pass
    try:
        for image in page.get_image_info():
            bbox = _coerce_bbox(image.get("bbox"))
            if bbox is not None:
                occupied.append(fitz.Rect(bbox))
    except Exception:
        pass
    return occupied


@dataclass
class _FontResource:
    family: str
    archive: Optional[Any]
    font_face_css: str
    font: Optional[Any] = None
    source_name: str = ""


def _resolve_font_resource(style_font: Optional[str]) -> _FontResource:
    _require_fitz()
    candidates: List[Path] = []
    if style_font:
        requested = Path(style_font).expanduser()
        candidates.append(requested)
        if os.name == "nt" and not requested.is_absolute():
            fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
            aliases = {
                "arialunicode.ttf": "arialuni.ttf",
                "arial unicode.ttf": "arialuni.ttf",
                "dejavusans.ttf": "DejaVuSans.ttf",
            }
            candidates.append(fonts_dir / aliases.get(requested.name.lower(), requested.name))
    if os.name == "nt":
        fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
        candidates.extend([fonts_dir / "arial.ttf", fonts_dir / "segoeui.ttf"])
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            ]
        )

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            archive = fitz.Archive(str(candidate.parent))
            font = fitz.Font(fontfile=str(candidate))
        except Exception:
            continue
        family = "DaruPdfFont"
        css = (
            f"@font-face {{ font-family: {family}; "
            f"src: url('{candidate.name}'); }}"
        )
        return _FontResource(
            family=family,
            archive=archive,
            font_face_css=css,
            font=font,
            source_name=candidate.name,
        )
    return _FontResource(
        family="sans-serif",
        archive=None,
        font_face_css="",
        source_name="sans-serif",
    )


def _font_name_keys(value: str) -> Tuple[str, str]:
    normalized = re.sub(
        r"^[A-Z]{6}\+",
        "",
        str(value or "").strip(),
        flags=re.IGNORECASE,
    )
    compact = re.sub(r"[^a-z0-9]", "", normalized.casefold())
    family = compact
    for suffix in (
        "bolditalic",
        "boldoblique",
        "semibold",
        "demibold",
        "regular",
        "italic",
        "oblique",
        "roman",
        "medium",
        "bold",
        "book",
        "psmt",
        "mt",
    ):
        if family.endswith(suffix) and len(family) > len(suffix):
            family = family[: -len(suffix)]
            break
    return compact, family


def _font_resource_supports_text(resource: _FontResource, text: str) -> bool:
    if resource.font is None:
        return False
    codepoints = {
        ord(char)
        for char in str(text or "")
        if not char.isspace() and not unicodedata.category(char).startswith("C")
    }
    try:
        return all(resource.font.has_glyph(codepoint) for codepoint in codepoints)
    except Exception:
        return False


class _OriginalFontResolver:
    def __init__(
        self,
        document: Any,
        fallback_resource: _FontResource,
        log: PdfLog,
    ) -> None:
        self.document = document
        self.fallback_resource = fallback_resource
        self.log = log
        self._temp_dir = tempfile.TemporaryDirectory(prefix="daru-pdf-fonts-")
        self._archive = fitz.Archive(self._temp_dir.name)
        self._page_fonts: Dict[int, List[Tuple[Any, ...]]] = {}
        self._resources: Dict[int, Optional[_FontResource]] = {}
        self._warned: set[Tuple[str, str]] = set()

    def close(self) -> None:
        self._temp_dir.cleanup()

    def resolve(self, unit: NativeTextUnit) -> _FontResource:
        entry = self._find_font_entry(unit.page_index, unit.font_name)
        if entry is None:
            self._warn_once(unit.font_name, "not_found")
            return self.fallback_resource
        resource = self._resource_from_xref(int(entry[0] or 0))
        if resource is None:
            self._warn_once(unit.font_name, "not_embedded")
            return self.fallback_resource
        text = str(unit.translated_text or unit.source_text)
        if not _font_resource_supports_text(resource, text):
            self._warn_once(unit.font_name, "missing_glyphs")
            return self.fallback_resource
        return resource

    def _find_font_entry(
        self,
        page_index: int,
        font_name: str,
    ) -> Optional[Tuple[Any, ...]]:
        entries = self._page_fonts.get(page_index)
        if entries is None:
            try:
                entries = list(self.document.get_page_fonts(page_index, full=True))
            except Exception:
                entries = []
            self._page_fonts[page_index] = entries
        target_compact, target_family = _font_name_keys(font_name)
        best_entry = None
        best_score = 0
        for entry in entries:
            names = [str(entry[index] or "") for index in (3, 4) if len(entry) > index]
            score = 0
            for candidate in names:
                compact, family = _font_name_keys(candidate)
                if compact and compact == target_compact:
                    score = max(score, 3)
                elif family and family == target_family:
                    score = max(score, 2)
                elif target_family and family and (
                    target_family in compact or family in target_compact
                ):
                    score = max(score, 1)
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_entry

    def _resource_from_xref(self, xref: int) -> Optional[_FontResource]:
        if xref in self._resources:
            return self._resources[xref]
        if xref <= 0:
            self._resources[xref] = None
            return None
        try:
            extracted_name, extension, _font_type, buffer = self.document.extract_font(xref)
        except Exception:
            self._resources[xref] = None
            return None
        if not buffer:
            self._resources[xref] = None
            return None
        safe_extension = re.sub(r"[^a-z0-9]", "", str(extension).casefold()) or "bin"
        filename = f"font-{xref}.{safe_extension}"
        font_path = Path(self._temp_dir.name) / filename
        try:
            font_path.write_bytes(buffer)
            font = fitz.Font(fontbuffer=buffer)
        except Exception:
            self._resources[xref] = None
            return None
        family = f"DaruOriginalFont{xref}"
        resource = _FontResource(
            family=family,
            archive=self._archive,
            font_face_css=(
                f"@font-face {{ font-family: {family}; src: url('{filename}'); }}"
            ),
            font=font,
            source_name=str(extracted_name or filename),
        )
        self._resources[xref] = resource
        return resource

    def _warn_once(self, font_name: str, reason: str) -> None:
        display_name = str(font_name or "неизвестный")
        key = (display_name.casefold(), reason)
        if key in self._warned:
            return
        self._warned.add(key)
        if reason == "missing_glyphs":
            detail = "не содержит всех символов перевода"
        elif reason == "not_embedded":
            detail = "не встроен в PDF или не может быть извлечен"
        else:
            detail = "не найден среди ресурсов страницы"
        self.log(
            f"PDF: исходный шрифт «{display_name}» {detail}; "
            f"используется {self.fallback_resource.source_name}"
        )


def _insert_unit(
    page: Any,
    unit: NativeTextUnit,
    *,
    occupied: Sequence[Any],
    font_resource: _FontResource,
) -> Optional[Dict[str, Any]]:
    text = str(unit.translated_text or unit.source_text)
    css = _unit_css(unit, font_resource)
    html_text = "<div>" + html.escape(text).replace("\n", "<br>") + "</div>"
    source_rect = fitz.Rect(unit.bbox)
    candidates = _candidate_rects(page.rect, source_rect, unit.role, occupied)

    for candidate in candidates:
        spare_height, scale = page.insert_htmlbox(
            candidate,
            html_text,
            css=css,
            scale_low=MIN_NATIVE_FONT_SCALE,
            archive=font_resource.archive,
            rotate=unit.rotation,
            overlay=True,
        )
        if spare_height >= 0:
            unit.rendered_bbox = _rect_tuple(candidate)
            unit.font_scale = float(scale)
            return None

    forced_rect = candidates[-1]
    spare_height, scale = page.insert_htmlbox(
        forced_rect,
        html_text,
        css=css,
        scale_low=FORCED_FONT_SCALE,
        archive=font_resource.archive,
        rotate=unit.rotation,
        overlay=True,
    )
    if spare_height < 0:
        spare_height, scale = page.insert_htmlbox(
            forced_rect,
            html_text,
            css=css,
            scale_low=0,
            archive=font_resource.archive,
            rotate=unit.rotation,
            overlay=True,
        )
    unit.rendered_bbox = _rect_tuple(forced_rect)
    unit.font_scale = float(scale)
    reason = (
        "translation_did_not_fit"
        if spare_height < 0
        else "font_scaled_below_preferred_minimum"
    )
    unit.layout_warning = reason
    return {
        "page": unit.page_index + 1,
        "stable_id": unit.stable_id,
        "role": unit.role,
        "reason": reason,
        "scale": unit.font_scale,
        "bbox": list(unit.rendered_bbox),
    }


def _unit_css(unit: NativeTextUnit, font_resource: _FontResource) -> str:
    color = f"#{unit.color & 0xFFFFFF:06x}"
    weight = "700" if unit.bold else "400"
    return (
        font_resource.font_face_css
        + f" * {{ font-family: {font_resource.family}; font-size: {unit.font_size:.2f}pt; "
        f"font-weight: {weight}; color: {color}; text-align: {unit.alignment}; "
        "line-height: 1.08; margin: 0; padding: 0; }"
    )


def _candidate_rects(
    page_rect: Any,
    source_rect: Any,
    role: str,
    occupied: Sequence[Any],
) -> List[Any]:
    base = fitz.Rect(source_rect)
    candidates = [base]
    if role == "CELL":
        return candidates

    width_extra = source_rect.width * 0.20
    height_extra = source_rect.height * 0.35
    expansions = [
        fitz.Rect(source_rect.x0, source_rect.y0, source_rect.x1 + width_extra, source_rect.y1),
        fitz.Rect(source_rect.x0, source_rect.y0, source_rect.x1, source_rect.y1 + height_extra),
        fitz.Rect(
            source_rect.x0,
            source_rect.y0,
            source_rect.x1 + width_extra,
            source_rect.y1 + height_extra,
        ),
        fitz.Rect(
            source_rect.x0 - width_extra / 2,
            source_rect.y0 - height_extra / 3,
            source_rect.x1 + width_extra / 2,
            source_rect.y1 + height_extra * 2 / 3,
        ),
    ]
    for candidate in expansions:
        candidate &= page_rect
        if candidate.is_empty or candidate in candidates:
            continue
        if _expansion_is_free(candidate, source_rect, occupied):
            candidates.append(candidate)
    return candidates


def _expansion_is_free(candidate: Any, source_rect: Any, occupied: Sequence[Any]) -> bool:
    for other in occupied:
        if _rects_equal(other, source_rect):
            continue
        intersection = candidate & other
        if intersection.is_empty:
            continue
        if intersection.get_area() > 1.0:
            return False
    return True


def _load_translation_layer(
    path: Optional[Path],
    file_hash: str,
    units: Sequence[NativeTextUnit],
    log: PdfLog,
) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"PDF: не удалось прочитать JSON слой {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        return {}

    by_id = {unit.stable_id: unit for unit in units}
    cached: Dict[str, str] = {}
    version = int(payload.get("version", 1) or 1)
    if version >= 2:
        if str(payload.get("document_sha256") or "") != file_hash:
            log("PDF: JSON слой относится к другой версии исходного файла")
            return {}
        for block in _iter_layer_blocks(payload):
            stable_id = str(block.get("stable_id") or "")
            unit = by_id.get(stable_id)
            if unit is None or not _layer_block_matches_unit(block, unit):
                continue
            translation = str(block.get("translated_text") or "").strip()
            if translation:
                cached[stable_id] = translation
    else:
        for block in _iter_layer_blocks(payload):
            for unit in units:
                if _legacy_layer_block_matches_unit(block, unit):
                    translation = str(block.get("translated_text") or "").strip()
                    if translation:
                        cached[unit.stable_id] = translation
    if cached:
        log(f"PDF: загружено переводов из JSON слоя: {len(cached)}")
    return cached


def _checkpoint_native_cache(
    checkpoint_store: TranslationCheckpointStore,
    units: Sequence[NativeTextUnit],
    log: PdfLog,
) -> Dict[str, str]:
    cached: Dict[str, str] = {}
    for unit in units:
        translated = checkpoint_store.get(
            "pdf-native",
            unit.stable_id,
            source_text=unit.source_text,
        )
        if translated:
            cached[unit.stable_id] = translated
    if cached:
        log(f"PDF: checkpoint loaded native translations: {len(cached)}")
    return cached


def _iter_layer_blocks(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for page in payload.get("pages") or []:
        if not isinstance(page, dict):
            continue
        for block in page.get("blocks") or page.get("regions") or []:
            if isinstance(block, dict):
                yield block


def _layer_block_matches_unit(block: Dict[str, Any], unit: NativeTextUnit) -> bool:
    return (
        str(block.get("text") or "") == unit.source_text
        and _bbox_matches(block.get("bbox"), unit.bbox)
    )


def _legacy_layer_block_matches_unit(block: Dict[str, Any], unit: NativeTextUnit) -> bool:
    return (
        str(block.get("text") or "") == unit.source_text
        and _bbox_matches(block.get("bbox"), unit.bbox)
    )


def _save_translation_layer(
    path: Optional[Path],
    *,
    file_hash: str,
    pages: Sequence[NativePage],
    native_pages: Sequence[int],
    ocr_pages: Sequence[int],
    log: PdfLog,
) -> None:
    if path is None:
        return
    payload_pages: List[Dict[str, Any]] = []
    for page in pages:
        blocks = [
            {
                "stable_id": unit.stable_id,
                "role": unit.role,
                "bbox": list(unit.bbox),
                "rendered_bbox": list(unit.rendered_bbox or unit.bbox),
                "text": unit.source_text,
                "translated_text": unit.translated_text or unit.source_text,
                "font_scale": unit.font_scale,
                "layout_warning": unit.layout_warning,
            }
            for unit in page.units
        ]
        payload_pages.append(
            {
                "page_index": page.page_index,
                "processing_mode": "native" if page.is_native else "ocr",
                "damaged_character_ratio": page.damaged_ratio,
                "blocks": blocks,
            }
        )
    payload = {
        "version": 2,
        "document_sha256": file_hash,
        "processing_mode": "native" if not ocr_pages else "hybrid",
        "native_pages": list(native_pages),
        "ocr_pages": list(ocr_pages),
        "pages": payload_pages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"PDF: сохранён JSON слой v2: {path}")


def _save_output_document(
    document: Any,
    fallback_documents: Dict[int, bytes],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_document = document
    composed = None
    fallback_handles: List[Any] = []
    if fallback_documents:
        composed = fitz.open()
        metadata = document.metadata
        toc = document.get_toc(simple=False)
        for page_index in range(len(document)):
            fallback_bytes = fallback_documents.get(page_index)
            if fallback_bytes is None:
                composed.insert_pdf(document, from_page=page_index, to_page=page_index)
                continue
            fallback = fitz.open(stream=fallback_bytes, filetype="pdf")
            fallback_handles.append(fallback)
            composed.insert_pdf(fallback, from_page=0, to_page=0)
        composed.set_metadata(metadata)
        if toc:
            try:
                composed.set_toc(toc)
            except Exception:
                pass
        output_document = composed

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        output_document.save(
            str(temporary_path),
            garbage=4,
            deflate=True,
            clean=True,
        )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        for fallback in fallback_handles:
            fallback.close()
        if composed is not None:
            composed.close()


def _canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _coerce_bbox(value: Any) -> Optional[BBox]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox  # type: ignore[return-value]


def _coerce_span_bbox(value: Any) -> Optional[BBox]:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        return None
    return bbox  # type: ignore[return-value]


def _bbox_union(first: BBox, second: BBox) -> BBox:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _bbox_union_many(values: Sequence[BBox]) -> BBox:
    if not values:
        raise ValueError("At least one bounding box is required")
    result = values[0]
    for value in values[1:]:
        result = _bbox_union(result, value)
    return result


def _horizontal_overlap_ratio(first: Any, second: Any) -> float:
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    denominator = min(first.width, second.width)
    return overlap / denominator if denominator > 0 else 0.0


def _vertical_overlap_ratio(first: Any, second: Any) -> float:
    overlap = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    denominator = min(first.height, second.height)
    return overlap / denominator if denominator > 0 else 0.0


def _rect_tuple(rect: Any) -> BBox:
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def _rect_contains_center(container: Any, item: Any) -> bool:
    center = fitz.Point((item.x0 + item.x1) / 2, (item.y0 + item.y1) / 2)
    return center in container


def _rects_equal(first: Any, second: Any, tolerance: float = 0.01) -> bool:
    return all(
        abs(a - b) <= tolerance
        for a, b in zip(_rect_tuple(first), _rect_tuple(second))
    )


def _bbox_matches(value: Any, expected: BBox, tolerance: float = 0.01) -> bool:
    actual = _coerce_bbox(value)
    if actual is None:
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(actual, expected))


__all__ = [
    "NativePage",
    "NativeTextUnit",
    "translate_native_pdf",
]
