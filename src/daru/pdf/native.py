"""Native PDF translation based on the existing searchable text layer."""

from __future__ import annotations

from ..translation.progress import report_progress

import hashlib
import html
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
from ..translation.job import check_cancelled
from ..translation.options import TranslationOptions
from ..utils.io import write_json_atomic
from .layers import read_layer_payload
from .watermarks import preserve_watermarks

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
PREFERRED_NATIVE_FONT_SCALE = 0.92
MIN_NATIVE_FONT_SCALE = 0.82
FORCED_FONT_SCALE = 0.68
EMERGENCY_FONT_SCALE = 0.50
NORMAL_LINE_HEIGHT = 1.08
COMPACT_LINE_HEIGHT = 1.0

from ..translation.prompts import NATIVE_PDF_SYSTEM_PROMPT

_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d{1,3}、\s*|(?:[•●▪◆◇\-–—]|\(?\d{1,3}[.),]|\d+(?:\.\d+)+|[A-Za-z][.)])\s+)")
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
class NativeTextSpanStyle:
    text: str
    bbox: BBox
    font_size: float
    font_name: str
    color: int
    bold: bool
    italic: bool
    font_family_kind: str
    flags: int = 0
    char_flags: int = 0
    alpha: int = 255
    ascender: Optional[float] = None
    descender: Optional[float] = None
    origin: Optional[Tuple[float, float]] = None
    bidi: int = 0
    char_count: int = 0
    average_char_width: float = 0.0


@dataclass
class NativeTextLineLayout:
    text: str
    bbox: BBox
    origin: Tuple[float, float]
    font_size: float
    direction: Tuple[float, float] = (1.0, 0.0)
    writing_mode: int = 0
    spans: List[NativeTextSpanStyle] = field(default_factory=list)


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
    italic: bool
    rotation: int
    layout_bounds: Optional[BBox] = None
    framed_label: bool = False
    lines: List[NativeTextLineLayout] = field(default_factory=list)
    style_runs: List[NativeTextSpanStyle] = field(default_factory=list)
    font_family_kind: str = "sans"
    alignment: str = "left"
    translated_text: Optional[str] = None
    rendered_bbox: Optional[BBox] = None
    font_scale: float = 1.0
    cached: bool = False
    layout_warning: Optional[str] = None
    font_fallback_reason: Optional[str] = None


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

    options = TranslationOptions.from_mapping(locals(), system_prompt_template=NATIVE_PDF_SYSTEM_PROMPT)
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
        settings_fingerprint=options.fingerprint("pdf-native"),
        resume_policy=resume_policy,
        log=log,
    )
    document = fitz.open(str(input_path))
    try:
        report_progress("read", 0, len(document))
        link_snapshots = _snapshot_document_links(document)
        page_labels = _snapshot_page_labels(document)
        pages = []
        for index in range(len(document)):
            check_cancelled()
            pages.append(_extract_page(document[index], index, source_lang))
            report_progress("read", index + 1, len(document))
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
        layer_cached = (
            _load_translation_layer(layer_json_path, file_hash, units, log,
                                    expected_identity=options.cache_identity(file_hash, "pdf-native"))
            if resume_policy == "auto" else {}
        )
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

        report_progress("apply")
        layout_warnings = _render_native_pages(document, pages, style_font, log)
        fallback_documents: Dict[int, bytes] = {}
        for page_index in (page.page_index for page in pages if not page.is_native):
            fallback_documents[page_index] = fallback_page_translator(page_index)

        report_progress("export")
        _save_output_document(
            document,
            fallback_documents,
            output_path,
            link_snapshots=link_snapshots,
            page_labels=page_labels,
        )
        _save_translation_layer(
            layer_json_path,
            file_hash=file_hash,
            cache_identity=options.cache_identity(file_hash, "pdf-native"),
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
    # Watermark Forms retain their original transform and transparency. They
    # must not become translation units or influence paragraph/table bounds.
    with preserve_watermarks(page):
        return _extract_page_content(page, page_index, source_lang)


def _extract_page_content(page: Any, page_index: int, source_lang: str) -> NativePage:
    text_blocks = _native_text_blocks(page)
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
    cell_rects = _table_cell_rects(page, text_blocks)
    units: List[NativeTextUnit] = []

    for block in text_blocks:
        block_units = _units_from_block(
            block,
            page_index=page_index,
            source_lang=source_lang,
            median_font_size=median_font_size,
            cell_rects=cell_rects,
        )
        units.extend(block_units)

    units = _merge_justified_cell_paragraphs(page, units, text_blocks=text_blocks)
    units = _merge_framed_labels(page, units)
    units = _split_units_around_artwork(page, units)
    units.sort(key=lambda unit: (unit.bbox[1], unit.bbox[0], unit.stable_id))
    return NativePage(
        page_index=page_index,
        is_native=True,
        damaged_ratio=damaged_ratio,
        units=units,
    )


def _native_text_blocks(page: Any) -> List[Dict[str, Any]]:
    """Keep visible glyph anchors, including in spans padded with spaces.

    TableFinder's text extraction may slice words at inferred cell boundaries.
    Always take text and typography from the original complete PDF spans.
    """
    data = page.get_text(
        "rawdict", sort=True,
        flags=fitz.TEXTFLAGS_RAWDICT & ~fitz.TEXT_PRESERVE_IMAGES,
    )
    blocks = [block for block in data.get("blocks", []) if block.get("type") == 0]
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                chars = span.pop("chars", [])
                span["text"] = "".join(char["c"] for char in chars)
                visible = [char for char in chars if char["c"].strip()]
                if visible:
                    span["bbox"] = _bbox_union_many([char["bbox"] for char in visible])
                    span["origin"] = visible[0]["origin"]
    return blocks


def _merge_framed_labels(page: Any, units: List[NativeTextUnit]) -> List[NativeTextUnit]:
    """Recover wrapped callouts even when a source line protrudes below its box."""
    frames = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        items = drawing.get("items", [])
        border = len(items) > 1 and all(item[0] in {"re", "l"} for item in items)
        if fill is not None and min(fill) < 0.95 and not border:
            continue
        for item in items:
            if item[0] != "re":
                continue
            rect = fitz.Rect(item[1])
            if 15 <= rect.width <= 160 and 8 <= rect.height <= 50:
                frames.append(rect)
    result = list(units)
    for frame in sorted(frames, key=lambda rect: rect.get_area()):
        members = []
        for unit in result:
            if unit.framed_label or unit.rotation:
                continue
            rect = fitz.Rect(unit.bbox)
            if (rect.x0 >= frame.x0 - 1 and rect.x1 <= frame.x1 + 1 and
                    rect.y0 >= frame.y0 - 1 and rect.y0 <= frame.y1 + unit.font_size * 0.5 and
                    rect.y1 <= frame.y1 + unit.font_size * 1.5):
                members.append(unit)
        if not members or not any(frame.contains(fitz.Point(u.bbox[0], u.bbox[1])) for u in members):
            continue
        members.sort(key=lambda unit: (unit.bbox[1], unit.bbox[0]))
        first = members[0]
        merged = _make_unit(
            page_index=first.page_index,
            bbox=_bbox_union_many([unit.bbox for unit in members]), role="CELL",
            source_text=_normalize_block_text([unit.source_text for unit in members]),
            font_size=first.font_size, font_name=first.font_name, color=first.color,
            bold=first.bold, italic=first.italic, rotation=first.rotation, alignment="center",
            lines=[line for unit in members for line in unit.lines],
            style_runs=[span for unit in members for span in unit.style_runs],
            font_family_kind=first.font_family_kind, stable_salt="framed-label",
        )
        merged.layout_bounds = _rect_tuple(frame + (1, 0.5, -1, -0.5))
        merged.framed_label = True
        member_ids = {id(unit) for unit in members}
        result = [unit for unit in result if id(unit) not in member_ids]
        result.append(merged)
    return result


def _merge_justified_cell_paragraphs(
    page: Any, units: List[NativeTextUnit], *, text_blocks: Sequence[Dict[str, Any]] = (),
) -> List[NativeTextUnit]:
    """Recover justified prose split into PDF word-lines and separate blocks.

    Cell geometry alone is insufficient: require a wrapped paragraph with a
    justified line spanning a narrow cell. Keep every original glyph/style
    anchor, and leave unaffected units (including their cache IDs) untouched.
    """
    by_cell: Dict[BBox, List[NativeTextUnit]] = {}
    for unit in units:
        if unit.layout_bounds is None or unit.rotation or unit.framed_label:
            continue
        if any(line.writing_mode or line.direction != (1.0, 0.0) for line in unit.lines):
            continue
        by_cell.setdefault(unit.layout_bounds, []).append(unit)

    removed = set()
    recovered: List[NativeTextUnit] = []
    occupied = None
    for cell, members in by_cell.items():
        size = median(unit.font_size for unit in members)
        width, height = cell[2] - cell[0], cell[3] - cell[1]
        if len(members) < 2 or width >= size * 20 or height > size * 24:
            continue
        segments = []
        for unit in members:
            for line in unit.lines:
                spans = [{"text": span.text, "bbox": span.bbox, "size": span.font_size,
                          "font": span.font_name, "flags": span.flags, "color": span.color}
                         for span in line.spans]
                segments.append({"text": line.text, "bbox": line.bbox, "spans": spans,
                                 "font_size": line.font_size, "baseline_y": line.origin[1],
                                 "dir": line.direction, "cell": cell, "layout": line,
                                 "owners": {id(unit)}, "original_lines": {id(line)}})
        segments.sort(key=lambda segment: (segment["baseline_y"], segment["bbox"][0]))
        rows = []
        for segment in segments:
            if rows and abs(segment["baseline_y"] - rows[-1][0]["baseline_y"]) < size * 0.25:
                rows[-1].append(segment)
            else:
                rows.append([segment])
        joined = []
        for row in rows:
            gaps = [right["bbox"][0] - left["bbox"][2] for left, right in zip(row, row[1:])]
            bbox = _bbox_union_many([segment["bbox"] for segment in row])
            styles = [_dominant_span_style(segment["spans"], size) for segment in row]
            justified = (
                len(row) > 1 and gaps and max(gaps) > size * 1.5
                and all(len(segment["text"].split()) == 1 for segment in row)
                and all(0 <= gap <= size * 4 for gap in gaps)
                and bbox[2] - bbox[0] >= width * 0.75
                and bbox[0] - cell[0] <= size and cell[2] - bbox[2] <= size
                and all(style[1:5] == styles[0][1:5] for style in styles)
                and max(style[0] for style in styles) - min(style[0] for style in styles) <= size * 0.15
                and not any(_LIST_PREFIX_RE.match(segment["text"]) for segment in row[1:])
            )
            if not justified:
                joined.extend(row)
                continue
            first = row[0]
            span_styles = [span for segment in row for span in segment["layout"].spans]
            text = " ".join(segment["text"].strip() for segment in row)
            layout = NativeTextLineLayout(
                text=text, bbox=bbox, origin=first["layout"].origin,
                font_size=first["font_size"], direction=first["dir"], spans=span_styles,
            )
            joined.append({**first, "text": text, "bbox": bbox, "layout": layout,
                           "spans": [span for segment in row for span in segment["spans"]],
                           "owners": set().union(*(segment["owners"] for segment in row)),
                           "original_lines": set().union(*(segment["original_lines"] for segment in row)),
                           "justified": True})
        if not any(segment.get("justified") for segment in joined):
            continue
        for fragment in _group_paragraph_segments(joined, stop_at_sentence=True):
            lines = fragment["lines"]
            owners = set().union(*(line["owners"] for line in lines))
            originals = set().union(*(line["original_lines"] for line in lines))
            original_units = [unit for unit in members if id(unit) in owners]
            if (len(owners) < 2 or len(lines) < 2 or not any(line.get("justified") for line in lines)
                    or originals != {id(line) for unit in original_units for line in unit.lines}):
                continue
            source = _normalize_block_text([line["text"] for line in lines])
            if len(source.split()) < 5:
                continue
            represented = {(span.text, span.bbox) for unit in original_units for span in unit.style_runs}
            # Numeric-only and one-letter fragments do not become translation
            # units. Never redact them by expanding across their glyphs here.
            if any(
                bbox is not None and fitz.Rect(bbox).intersects(fitz.Rect(fragment["bbox"]))
                and _containing_cell(bbox, [fitz.Rect(cell)]) is not None
                and (str(span.get("text", "")), bbox) not in represented
                for block in text_blocks for line in block.get("lines", [])
                if tuple(line.get("dir", (1.0, 0.0))) == (1.0, 0.0) and not line.get("wmode", 0)
                for span in line.get("spans", []) if str(span.get("text", "")).strip()
                for bbox in [_coerce_span_bbox(span.get("bbox"))]
            ):
                continue
            if occupied is None:
                occupied = _page_nontext_occupied(page, text_rects=[fitz.Rect(unit.bbox) for unit in units])
            if any(fitz.Rect(fragment["bbox"]).intersects(obstacle) for obstacle in occupied):
                continue
            font_size, font_name, color, bold, italic, family = _dominant_span_style(
                [span for line in lines for span in line["spans"]], size,
            )
            layouts = [line["layout"] for line in lines]
            merged = _make_unit(
                page_index=original_units[0].page_index, bbox=fragment["bbox"], role="CELL",
                source_text=source, font_size=font_size, font_name=font_name, color=color,
                bold=bold, italic=italic, rotation=0,
                alignment=_fragment_alignment(lines, fragment["bbox"], cell),
                lines=layouts, style_runs=[span for line in layouts for span in line.spans],
                font_family_kind=family, stable_salt="justified-cell-paragraph",
            )
            merged.layout_bounds = cell
            removed.update(owners)
            recovered.append(merged)
    return [unit for unit in units if id(unit) not in removed] + recovered


def _split_units_around_artwork(page: Any, units: List[NativeTextUnit]) -> List[NativeTextUnit]:
    """Keep changes in column width when a paragraph wraps around a figure."""
    obstacles = _page_nontext_occupied(page, text_rects=[fitz.Rect(unit.bbox) for unit in units])
    result = []
    for unit in units:
        rect = fitz.Rect(unit.bbox)
        if (unit.framed_label or unit.rotation or unit.alignment != "left" or
                len(unit.lines) < 3 or rect.width < unit.font_size * 12):
            result.append(unit)
            continue
        nearby = [other for other in obstacles if other.intersects(rect)]
        bands: List[Tuple[bool, float, List[NativeTextLineLayout]]] = []
        for line in unit.lines:
            right = rect.x1
            for other in nearby:
                overlap_y = min(line.bbox[3], other.y1) - max(line.bbox[1], other.y0)
                if (overlap_y > unit.font_size * 0.2 and
                        other.x0 >= line.bbox[2] + 1 and other.x0 < right):
                    right = other.x0 - 1
            constrained = right < rect.x1 - unit.font_size
            if bands and bands[-1][0] == constrained:
                previous = bands[-1]
                # A band must still contain every original line assigned to it.
                if min(previous[1], right) >= max(item.bbox[2] for item in previous[2] + [line]):
                    bands[-1] = (constrained, min(previous[1], right), previous[2] + [line])
                    continue
            bands.append((constrained, right, [line]))
        if len(bands) == 1 or not any(band[0] for band in bands):
            result.append(unit)
            continue
        for index, (_, right, lines) in enumerate(bands):
            bbox = _bbox_union_many([line.bbox for line in lines])
            fragment = _make_unit(
                page_index=unit.page_index, bbox=bbox, role=unit.role,
                source_text=_normalize_block_text([line.text for line in lines]),
                font_size=unit.font_size, font_name=unit.font_name, color=unit.color,
                bold=unit.bold, italic=unit.italic, rotation=unit.rotation,
                alignment=unit.alignment, lines=lines,
                style_runs=[span for line in lines for span in line.spans],
                font_family_kind=unit.font_family_kind, stable_salt=f"artwork-band:{index}",
            )
            bounds = fitz.Rect(unit.layout_bounds or page.rect)
            bounds.x1 = min(bounds.x1, right)
            bounds.y0 = max(bounds.y0, bbox[1])
            if index + 1 < len(bands):
                bounds.y1 = min(bounds.y1, bands[index + 1][2][0].bbox[1])
            fragment.layout_bounds = _rect_tuple(bounds)
            result.append(fragment)
    return result


def _table_cell_rects(page: Any, text_blocks: Sequence[Dict[str, Any]] = ()) -> List[Any]:
    """Table geometry constrains layout and validates narrow prose recovery."""
    cells: List[Any] = []
    try:
        finder = page.find_tables()
    except Exception:
        return cells

    for table in getattr(finder, "tables", []):
        if table.row_count < 2 or table.col_count < 2:
            continue
        for row in table.rows:
            for cell in row.cells:
                bbox = _coerce_bbox(cell)
                if bbox is not None and fitz.Rect(bbox) not in cells:
                    cells.append(fitz.Rect(bbox))
    segments = [segment for block in text_blocks for line in block.get("lines", [])
                for segment in _split_line_segments(line)]
    valid = []
    for cell in cells:
        cuts_text = False
        for segment in segments:
            r = fitz.Rect(segment["bbox"])
            if not cell.y0 + 1 < (r.y0 + r.y1) / 2 < cell.y1 - 1:
                continue
            if r.x0 < cell.x1 - 1 and r.x1 > cell.x0 + 1:
                if r.x0 < cell.x0 - 0.5 or r.x1 > cell.x1 + 0.5:
                    cuts_text = True
                    break
        if not cuts_text:
            valid.append(cell)
    return sorted(valid, key=lambda rect: rect.get_area())


def _containing_cell(bbox: BBox, cells: Sequence[Any]) -> Optional[BBox]:
    rect = fitz.Rect(bbox)
    for cell in cells:
        # Reject inferred borders that pass through a word or a complete line.
        tolerance = fitz.Rect(cell) + (-0.5, -1.0, 0.5, 1.0)
        if tolerance.contains(rect):
            return _rect_tuple(cell)
    return None


def _units_from_block(
    block: Dict[str, Any],
    *,
    page_index: int,
    source_lang: str,
    median_font_size: float,
    cell_rects: Sequence[Any] = (),
) -> List[NativeTextUnit]:
    fragments = _split_block_fragments(block, cell_rects=cell_rects)
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
        font_size, font_name, color, bold, italic, font_family_kind = _dominant_span_style(
            spans,
            median_font_size,
        )
        rotation = _rotation_from_lines(lines)
        role = _classify_role(source_text, font_size, median_font_size)
        line_layouts = [_line_layout_from_segment(line) for line in lines]
        style_runs = [
            span_style
            for line_layout in line_layouts
            for span_style in line_layout.spans
        ]
        cell = _containing_cell(fragment["bbox"], cell_rects)
        unit = _make_unit(
            page_index=page_index,
            bbox=fragment["bbox"],
            role="CELL" if cell and role != "TOC" else role,
            source_text=source_text,
            font_size=font_size,
            font_name=font_name,
            color=color,
            bold=bold,
            italic=italic,
            rotation=rotation,
            alignment=_fragment_alignment(lines, fragment["bbox"], cell),
            lines=line_layouts,
            style_runs=style_runs,
            font_family_kind=font_family_kind,
            stable_salt=f"fragment:{fragment_index}",
        )
        unit.layout_bounds = cell
        units.append(unit)
    return units


def _split_block_fragments(
    block: Dict[str, Any], *, cell_rects: Sequence[Any] = (),
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for line in block.get("lines", []):
        segments.extend(_split_line_segments(line))
    if not segments:
        return []
    segments.sort(key=lambda segment: (segment["baseline_y"], segment["bbox"][0]))
    joined: List[Dict[str, Any]] = []
    for segment in segments:
        segment["cell"] = _containing_cell(segment["bbox"], cell_rects)
        previous = joined[-1] if joined else None
        if previous and segment["cell"] is not None and segment["cell"] == previous["cell"]:
            size = min(segment["font_size"], previous["font_size"])
            gap = segment["bbox"][0] - previous["bbox"][2]
            cell_width = segment["cell"][2] - segment["cell"][0]
            max_gap = size * (2.5 if cell_width < size * 10 else 1.5)
            if (cell_width < size * 20 and
                    abs(segment["baseline_y"] - previous["baseline_y"]) < size * 0.25 and
                    0 <= gap < max_gap):
                previous["text"] += " " + segment["text"]
                previous["spans"].extend(segment["spans"])
                previous["bbox"] = _bbox_union(previous["bbox"], segment["bbox"])
                continue
        joined.append(segment)
    segments = joined

    return _group_paragraph_segments(segments)


def _group_paragraph_segments(
    segments: Sequence[Dict[str, Any]], *, stop_at_sentence: bool = False,
) -> List[Dict[str, Any]]:

    fragments: List[Dict[str, Any]] = []
    for segment in segments:
        segment_rect = fitz.Rect(segment["bbox"])
        best_fragment = None
        best_score = -1.0
        for fragment in fragments:
            previous_line = fragment["lines"][-1]
            if not _segments_share_paragraph(previous_line, segment):
                continue
            if stop_at_sentence and previous_line["text"].rstrip().endswith((".", ":", ";", "!", "?")):
                continue
            baseline_gap = float(segment["baseline_y"]) - float(previous_line["baseline_y"])
            same_baseline = abs(baseline_gap) <= max(
                1.5,
                min(float(segment["font_size"]), float(previous_line["font_size"])) * 0.25,
            )
            if same_baseline:
                continue
            max_gap = max(
                6.0,
                float(segment["font_size"]) * 1.8,
                float(previous_line["font_size"]) * 1.8,
            )
            if baseline_gap <= 0.0 or baseline_gap > max_gap:
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


def _segments_share_paragraph(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    if first.get("cell") != second.get("cell") or first.get("dir") != second.get("dir"):
        return False
    if _TOC_LEADER_RE.search(first["text"]) or _TOC_LEADER_RE.search(second["text"]):
        return False
    if not any(char.isalpha() for char in first["text"]):
        return False
    if _LIST_PREFIX_RE.match(second["text"]):
        return False
    first_style = _dominant_span_style(first["spans"], first["font_size"])
    second_style = _dominant_span_style(second["spans"], second["font_size"])
    if first_style[3:5] != second_style[3:5]:
        return False
    size = max(first_style[0], second_style[0])
    if abs(first_style[0] - second_style[0]) > size * 0.15:
        return False
    a, b = fitz.Rect(first["bbox"]), fitz.Rect(second["bbox"])
    if size < 20 and abs(a.x0 - b.x0) > size * 3:
        # Separate centered captions from nearby prose and adjacent columns.
        cell = first.get("cell")
        if not (cell and cell[2] - cell[0] < size * 20 and
                abs((a.x0 + a.x1) - (b.x0 + b.x1)) < size):
            return False
    if first["text"].rstrip().endswith((".", ":", ";")) and b.x0 - a.x0 > size:
        return False
    return True


def _fragment_alignment(
    lines: Sequence[Dict[str, Any]], bbox: BBox, cell: Optional[BBox],
) -> str:
    size = max(1.0, float(lines[0]["font_size"]))
    rects = [fitz.Rect(line["bbox"]) for line in lines]
    if len(rects) > 1:
        lefts = [rect.x0 for rect in rects]
        centers = [(rect.x0 + rect.x1) / 2 for rect in rects]
        rights = [rect.x1 for rect in rects]
        if max(lefts) - min(lefts) > size * 0.5:
            if max(centers) - min(centers) < size * 0.5:
                return "center"
            if max(rights) - min(rights) < size * 0.3:
                return "right"
    if cell:
        left, right = bbox[0] - cell[0], cell[2] - bbox[2]
        # Full-width prose has equal tiny margins too; that is not centering.
        if min(left, right) > size and abs(left - right) < size * 0.7:
            return "center"
    return "left"


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
            if str(span.get("text", "")).strip()
            if (bbox := _coerce_bbox(span.get("bbox"))) is not None
        ]
        if not bboxes:
            continue
        text = "".join(str(span.get("text", "")) for span in group).strip()
        if not text:
            continue
        bbox = _bbox_union_many(bboxes)
        segments.append(
            {
                "bbox": bbox,
                "text": text,
                "spans": group,
                "dir": line.get("dir", (1.0, 0.0)),
                "wmode": int(line.get("wmode", 0) or 0),
                "font_size": median(
                    [float(span.get("size", 10) or 10) for span in group]
                ),
                "origin": _segment_origin(line, group, bbox),
                "baseline_y": _segment_baseline_y(line, group, bbox),
            }
        )
    return segments


def _segment_origin(
    line: Dict[str, Any],
    spans: Sequence[Dict[str, Any]],
    bbox: BBox,
) -> Tuple[float, float]:
    origin = line.get("origin")
    if isinstance(origin, (tuple, list)) and len(origin) == 2:
        try:
            return (float(origin[0]), float(origin[1]))
        except (TypeError, ValueError):
            pass
    origins = [
        span.get("origin")
        for span in spans
        if isinstance(span.get("origin"), (tuple, list))
        and len(span.get("origin")) == 2
    ]
    if origins:
        try:
            return (float(origins[0][0]), float(origins[0][1]))
        except (TypeError, ValueError):
            pass
    return (float(bbox[0]), float(bbox[3]))


def _segment_baseline_y(
    line: Dict[str, Any],
    spans: Sequence[Dict[str, Any]],
    bbox: BBox,
) -> float:
    return _segment_origin(line, spans, bbox)[1]


def _line_layout_from_segment(segment: Dict[str, Any]) -> NativeTextLineLayout:
    spans = [
        style
        for span in segment.get("spans", [])
        if (style := _span_style_from_span(span)) is not None
    ]
    raw_origin = segment.get("origin") or (segment["bbox"][0], segment["bbox"][3])
    try:
        origin = (float(raw_origin[0]), float(raw_origin[1]))
    except (TypeError, ValueError, IndexError):
        origin = (float(segment["bbox"][0]), float(segment["bbox"][3]))
    direction = _coerce_point(segment.get("dir"), (1.0, 0.0))
    return NativeTextLineLayout(
        text=str(segment.get("text") or ""),
        bbox=segment["bbox"],
        origin=origin,
        font_size=float(segment.get("font_size", 10) or 10),
        direction=direction,
        writing_mode=int(segment.get("wmode", 0) or 0),
        spans=spans,
    )


def _span_style_from_span(span: Dict[str, Any]) -> Optional[NativeTextSpanStyle]:
    bbox = _coerce_span_bbox(span.get("bbox"))
    text = str(span.get("text", ""))
    if bbox is None or not text.strip():
        return None
    font_name = str(span.get("font") or "sans-serif")
    flags = int(span.get("flags", 0) or 0)
    bold = bool(flags & 16) or "bold" in font_name.lower()
    italic = bool(flags & 2) or any(
        marker in font_name.lower() for marker in ("italic", "oblique")
    )
    char_count = len([char for char in text if not char.isspace()])
    average_char_width = (bbox[2] - bbox[0]) / char_count if char_count else 0.0
    return NativeTextSpanStyle(
        text=text,
        bbox=bbox,
        font_size=max(4.0, float(span.get("size", 10) or 10)),
        font_name=font_name,
        color=int(span.get("color", 0) or 0),
        bold=bold,
        italic=italic,
        font_family_kind=_font_family_kind(font_name),
        flags=flags,
        char_flags=int(span.get("char_flags", 0) or 0),
        alpha=max(0, min(255, int(span.get("alpha", 255) or 255))),
        ascender=_coerce_optional_float(span.get("ascender")),
        descender=_coerce_optional_float(span.get("descender")),
        origin=_coerce_optional_point(span.get("origin")),
        bidi=int(span.get("bidi", 0) or 0),
        char_count=char_count,
        average_char_width=average_char_width,
    )


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
    italic: bool = False,
    lines: Optional[List[NativeTextLineLayout]] = None,
    style_runs: Optional[List[NativeTextSpanStyle]] = None,
    font_family_kind: str = "",
    stable_salt: str = "",
) -> NativeTextUnit:
    normalized_bbox = ",".join(f"{value:.2f}" for value in bbox)
    stable_payload = f"{page_index}|{normalized_bbox}|{source_text}|{stable_salt}"
    stable_id = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:20]
    resolved_family_kind = font_family_kind or _font_family_kind(font_name)
    resolved_lines = list(lines or [])
    resolved_runs = list(style_runs or [])
    if not resolved_runs:
        resolved_runs = [
            NativeTextSpanStyle(
                text=source_text,
                bbox=bbox,
                font_size=max(4.0, font_size),
                font_name=font_name,
                color=color,
                bold=bold,
                italic=italic,
                font_family_kind=resolved_family_kind,
            )
        ]
    if not resolved_lines:
        resolved_lines = [
            NativeTextLineLayout(
                text=source_text,
                bbox=bbox,
                origin=(bbox[0], bbox[3]),
                font_size=max(4.0, font_size),
                spans=resolved_runs,
            )
        ]
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
        italic=italic,
        rotation=rotation,
        lines=resolved_lines,
        style_runs=resolved_runs,
        font_family_kind=resolved_family_kind,
        alignment=alignment,
    )


def _normalize_block_text(lines: Sequence[str]) -> str:
    clean_lines = [re.sub(r"\s+", " ", line).strip() for line in lines if line.strip()]
    if not clean_lines:
        return ""
    if len(clean_lines) == 1:
        return clean_lines[0]

    output = clean_lines[0]
    for current in clean_lines[1:]:
        if output.endswith("-") and current[:1].islower():
            output = output[:-1] + current
        elif _LIST_PREFIX_RE.match(current):
            output += "\n" + current
        else:
            output += " " + current
    return output.strip()


def _classify_role(source_text: str, font_size: float, median_font_size: float) -> str:
    stripped = source_text.strip()
    if _TOC_LEADER_RE.search(stripped):
        return "TOC"
    if _LIST_PREFIX_RE.match(stripped):
        return "LIST"
    if font_size >= max(14.0, median_font_size * 1.35):
        return "TITLE"
    if len(stripped) <= 100 and font_size <= median_font_size * 0.85:
        return "CAPTION"
    return "TEXT"


def _font_family_kind(font_name: str) -> str:
    normalized = _font_name_keys(font_name)[1]
    if any(marker in normalized for marker in ("mono", "courier", "consol", "code")):
        return "mono"
    if any(
        marker in normalized
        for marker in (
            "serif",
            "times",
            "georgia",
            "cambria",
            "garamond",
            "baskerville",
            "minion",
            "caslon",
        )
    ):
        return "serif"
    return "sans"


def _dominant_span_style(
    spans: Sequence[Dict[str, Any]],
    fallback_size: float,
) -> Tuple[float, str, int, bool, bool, str]:
    if not spans:
        return fallback_size, "sans-serif", 0, False, False, "sans"
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
    italic = bool(flags & 2) or any(
        marker in font_name.lower() for marker in ("italic", "oblique")
    )
    return (
        float(median(sizes)) if sizes else fallback_size,
        font_name,
        int(dominant.get("color", 0) or 0),
        bold,
        italic,
        _font_family_kind(font_name),
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
    size, name, color, bold, italic, font_family_kind = _dominant_span_style(spans, fallback_size)
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
        "italic": italic,
        "font_family_kind": font_family_kind,
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
    if re.match(
        r"^\s*(?:address|head\s+office|factory|zip|p\.?\s*c\.?|tel\.?|fax|e-?mail)\s*:",
        line, re.IGNORECASE,
    ):
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
    options = TranslationOptions.from_mapping(locals(), system_prompt_template=NATIVE_PDF_SYSTEM_PROMPT)
    pending: List[NativeTextUnit] = []
    for unit in units:
        cached_value = cached.get(unit.stable_id)
        if cached_value:
            unit.translated_text = cached_value
            unit.cached = True
        else:
            pending.append(unit)

    report_progress("translate", len(units) - len(pending), len(units))
    if not pending:
        if not units:
            return "not-required"
        if checkpoint_store is not None and checkpoint_store.blocks:
            return "cached-checkpoint"
        return "cached-layer"

    translator = TranslationEngine(**options.engine_kwargs())
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
        check_cancelled()
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
        report_progress("translate", len(units) - len(pending) + processed, len(units))
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
    fallback_resolver = _FallbackFontResolver(None if original_mode else style_font)
    original_resolver = (
        _OriginalFontResolver(document, fallback_resolver, log)
        if original_mode
        else None
    )
    try:
        for page_model in pages:
            check_cancelled()
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

            # Keep decorative watermark glyphs out of redactions and layout.
            with preserve_watermarks(page):
                font_bundles = {
                    unit.stable_id: _resolve_unit_font_bundle(
                        unit,
                        original_resolver=original_resolver,
                        fallback_resolver=fallback_resolver,
                    )
                    for unit in changed
                }
                for unit in changed:
                    if unit.font_fallback_reason:
                        warnings.append(
                            {
                                "page": unit.page_index + 1,
                                "stable_id": unit.stable_id,
                                "role": unit.role,
                                "reason": "font_fallback",
                                "detail": unit.font_fallback_reason,
                                "font": font_bundles[unit.stable_id].default.source_name,
                                "bbox": list(unit.bbox),
                            }
                        )
                for unit in changed:
                    for line in unit.lines:
                        page.add_redact_annot(
                            fitz.Rect(line.bbox), fill=None, cross_out=False,
                        )
                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                    text=fitz.PDF_REDACT_TEXT_REMOVE,
                )

                occupied = [fitz.Rect(unit.bbox) for unit in page_model.units]
                occupied.extend(_page_nontext_occupied(page, text_rects=occupied))
                occupied.extend(
                    fitz.Rect(span["bbox"])
                    for block in _native_text_blocks(page)
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                    if str(span.get("text", "")).strip()
                )
                for unit in changed:
                    warning = _insert_unit(
                        page,
                        unit,
                        occupied=occupied,
                        font_resource=font_bundles[unit.stable_id],
                    )
                    if warning:
                        warnings.append(warning)
                        log(
                            f"PDF: предупреждение верстки, стр. {unit.page_index + 1}, "
                            f"блок {unit.stable_id}, масштаб {unit.font_scale:.2f}"
                        )
                    if unit.rendered_bbox is not None:
                        occupied.append(fitz.Rect(unit.rendered_bbox))
    finally:
        if original_resolver is not None:
            original_resolver.close()
    return warnings


def _page_nontext_occupied(page: Any, *, text_rects: Sequence[Any] = ()) -> List[Any]:
    occupied: List[Any] = []
    try:
        for drawing in page.get_drawings():
            rect = drawing.get("rect")
            if rect is None:
                continue
            items = drawing.get("items", [])
            if (len(items) == 1 and items[0][0] == "re" and
                    drawing.get("fill") is not None and drawing.get("color") is None and
                    any(fitz.Rect(rect).contains(text_rect) for text_rect in text_rects)):
                # A solid color behind source text is its background, not a
                # diagram that should force the translation into a tiny font.
                continue
            if (items and all(item[0] in {"l", "re"} for item in items) and
                    (drawing.get("fill") is None or any(item[0] == "re" for item in items))):
                # An enclosing table border is not an occupied solid rectangle.
                for item in items:
                    if item[0] == "l":
                        a, b = item[1:3]
                        boxes = [fitz.Rect(min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))]
                    else:
                        r = fitz.Rect(item[1])
                        boxes = [fitz.Rect(r.x0, r.y0, r.x1, r.y0), fitz.Rect(r.x0, r.y1, r.x1, r.y1),
                                 fitz.Rect(r.x0, r.y0, r.x0, r.y1), fitz.Rect(r.x1, r.y0, r.x1, r.y1)]
                    occupied.extend(box + (-0.75, -0.75, 0.75, 0.75) for box in boxes)
            else:
                occupied.append(fitz.Rect(rect) + (-0.75, -0.75, 0.75, 0.75))
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
    family_kind: str = "sans"
    bold: bool = False
    italic: bool = False


@dataclass
class _FontBundle:
    default: _FontResource
    by_style: Dict[Tuple[str, str, bool, bool], _FontResource] = field(default_factory=dict)
    archive: Optional[Any] = None
    font_face_css: str = ""

    def resource_for(self, style: NativeTextSpanStyle) -> _FontResource:
        return self.by_style.get(_style_resource_key(style), self.default)


def _style_resource_key(style: NativeTextSpanStyle) -> Tuple[str, str, bool, bool]:
    compact, family = _font_name_keys(style.font_name)
    font_key = family or compact or str(style.font_name or "").casefold()
    return (
        font_key,
        style.font_family_kind or _font_family_kind(style.font_name),
        bool(style.bold),
        bool(style.italic),
    )


def _build_font_bundle(
    default: _FontResource,
    by_style: Optional[Dict[Tuple[str, str, bool, bool], _FontResource]] = None,
) -> _FontBundle:
    style_resources = dict(by_style or {})
    resources = [default, *style_resources.values()]
    css_parts: List[str] = []
    seen_css = set()
    for resource in resources:
        css = resource.font_face_css
        if not css or css in seen_css:
            continue
        css_parts.append(css)
        seen_css.add(css)
    return _FontBundle(
        default=default,
        by_style=style_resources,
        archive=_combine_font_archives(resources),
        font_face_css="\n".join(css_parts),
    )


def _ensure_font_bundle(value: Any) -> _FontBundle:
    if isinstance(value, _FontBundle):
        return value
    return _build_font_bundle(value)


def _combine_font_archives(resources: Sequence[_FontResource]) -> Optional[Any]:
    archives = []
    seen = set()
    for resource in resources:
        archive = resource.archive
        if archive is None:
            continue
        key = id(archive)
        if key in seen:
            continue
        seen.add(key)
        archives.append(archive)
    if not archives:
        return None
    if len(archives) == 1:
        return archives[0]
    combined = fitz.Archive()
    for archive in archives:
        combined.add(archive)
    return combined


def _resolve_unit_font_bundle(
    unit: NativeTextUnit,
    *,
    original_resolver: Optional["_OriginalFontResolver"],
    fallback_resolver: "_FallbackFontResolver",
) -> _FontBundle:
    text = str(unit.translated_text or unit.source_text)
    default = (
        original_resolver.resolve(unit)
        if original_resolver is not None
        else fallback_resolver.resolve(unit, text)
    )
    by_style: Dict[Tuple[str, str, bool, bool], _FontResource] = {}
    for style in _unit_distinct_span_styles(unit):
        key = _style_resource_key(style)
        if key in by_style:
            continue
        by_style[key] = (
            original_resolver.resolve_style(unit, style, text)
            if original_resolver is not None
            else fallback_resolver.resolve_style(style, text)
        )
    return _build_font_bundle(default, by_style)


def _unit_distinct_span_styles(unit: NativeTextUnit) -> List[NativeTextSpanStyle]:
    styles: List[NativeTextSpanStyle] = []
    seen = set()
    for style in unit.style_runs:
        if not style.text.strip():
            continue
        key = _style_resource_key(style)
        if key in seen:
            continue
        seen.add(key)
        styles.append(style)
    return styles


def _resolve_font_resource(
    style_font: Optional[str],
    *,
    family_kind: str = "sans",
    bold: bool = False,
    italic: bool = False,
    text: str = "",
) -> _FontResource:
    _require_fitz()
    first_loaded: Optional[_FontResource] = None
    for candidate in _font_candidate_paths(style_font, family_kind, bold, italic):
        resource = _load_font_resource(candidate, family_kind, bold, italic)
        if resource is None:
            continue
        if first_loaded is None:
            first_loaded = resource
        if not text or _font_resource_supports_text(resource, text):
            return resource
    if first_loaded is not None:
        return first_loaded
    generic_family = family_kind if family_kind in {"serif", "mono"} else "sans-serif"
    return _FontResource(
        family=generic_family,
        archive=None,
        font_face_css="",
        source_name=generic_family,
        family_kind=family_kind,
        bold=bold,
        italic=italic,
    )


def _font_candidate_paths(
    style_font: Optional[str],
    family_kind: str,
    bold: bool,
    italic: bool,
) -> List[Path]:
    candidates: List[Path] = []
    if style_font:
        requested = Path(style_font).expanduser()
        variants = {
            "times": ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
            "arial": ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"),
            "calibri": ("calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"),
            "segoeui": ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"),
            "georgia": ("georgia.ttf", "georgiab.ttf", "georgiai.ttf", "georgiaz.ttf"),
        }
        names = variants.get(requested.stem.lower())
        if names:
            variant = names[int(bold) + 2 * int(italic)]
            candidates.append(requested.with_name(variant))
            if os.name == "nt" and not requested.is_absolute():
                candidates.append(Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / variant)
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
        windows: Dict[str, Dict[Tuple[bool, bool], List[str]]] = {
            "serif": {
                (False, False): ["times.ttf", "georgia.ttf", "cambria.ttc"],
                (True, False): ["timesbd.ttf", "georgiab.ttf", "cambriab.ttf"],
                (False, True): ["timesi.ttf", "georgiai.ttf", "cambriai.ttf"],
                (True, True): ["timesbi.ttf", "georgiaz.ttf", "cambriaz.ttf"],
            },
            "sans": {
                (False, False): ["arial.ttf", "segoeui.ttf", "calibri.ttf"],
                (True, False): ["arialbd.ttf", "segoeuib.ttf", "calibrib.ttf"],
                (False, True): ["ariali.ttf", "segoeuii.ttf", "calibrii.ttf"],
                (True, True): ["arialbi.ttf", "segoeuiz.ttf", "calibriz.ttf"],
            },
            "mono": {
                (False, False): ["consola.ttf", "cour.ttf"],
                (True, False): ["consolab.ttf", "courbd.ttf"],
                (False, True): ["consolai.ttf", "couri.ttf"],
                (True, True): ["consolaz.ttf", "courbi.ttf"],
            },
        }
        names = windows.get(family_kind, windows["sans"]).get((bold, italic), [])
        candidates.extend(fonts_dir / name for name in names)
        candidates.extend([fonts_dir / "arial.ttf", fonts_dir / "segoeui.ttf"])
    else:
        linux: Dict[str, Dict[Tuple[bool, bool], List[Path]]] = {
            "serif": {
                (False, False): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")],
                (True, False): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")],
                (False, True): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf")],
                (True, True): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf")],
            },
            "sans": {
                (False, False): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")],
                (True, False): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")],
                (False, True): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf")],
                (True, True): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf")],
            },
            "mono": {
                (False, False): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")],
                (True, False): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")],
                (False, True): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf")],
                (True, True): [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf")],
            },
        }
        candidates.extend(linux.get(family_kind, linux["sans"]).get((bold, italic), []))
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            ]
        )

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _load_font_resource(
    candidate: Path,
    family_kind: str,
    bold: bool,
    italic: bool,
) -> Optional[_FontResource]:
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        archive = fitz.Archive(str(candidate.parent))
        font = fitz.Font(fontfile=str(candidate))
    except Exception:
        return None
    style_key = (
        f"{family_kind}{'Bold' if bold else ''}{'Italic' if italic else ''}"
    )
    family = re.sub(r"[^A-Za-z0-9_]", "", f"DaruPdfFont{style_key}") or "DaruPdfFont"
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
        family_kind=family_kind,
        bold=bold,
        italic=italic,
    )


class _FallbackFontResolver:
    def __init__(self, style_font: Optional[str]) -> None:
        self.style_font = style_font
        self._cache: Dict[Tuple[str, bool, bool, str], _FontResource] = {}

    def resolve(self, unit: NativeTextUnit, text: str) -> _FontResource:
        family_kind = unit.font_family_kind or _font_family_kind(unit.font_name)
        key = (family_kind, bool(unit.bold), bool(unit.italic), _text_script_key(text))
        cached = self._cache.get(key)
        if cached is not None and (
            not text or cached.font is None or _font_resource_supports_text(cached, text)
        ):
            return cached
        resource = _resolve_font_resource(
            self.style_font,
            family_kind=family_kind,
            bold=unit.bold,
            italic=unit.italic,
            text=text,
        )
        self._cache[key] = resource
        return resource

    def resolve_style(self, style: NativeTextSpanStyle, text: str) -> _FontResource:
        family_kind = style.font_family_kind or _font_family_kind(style.font_name)
        key = (family_kind, bool(style.bold), bool(style.italic), _text_script_key(text))
        cached = self._cache.get(key)
        if cached is not None and (
            not text or cached.font is None or _font_resource_supports_text(cached, text)
        ):
            return cached
        resource = _resolve_font_resource(
            self.style_font,
            family_kind=family_kind,
            bold=style.bold,
            italic=style.italic,
            text=text,
        )
        self._cache[key] = resource
        return resource

    def default_resource(self) -> _FontResource:
        return _resolve_font_resource(self.style_font)


def _text_script_key(text: str) -> str:
    if re.search(r"[\u0400-\u04FF]", str(text or "")):
        return "cyrillic"
    if re.search(r"[^\x00-\x7F]", str(text or "")):
        return "unicode"
    return "latin"


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
        fallback_resource: Any,
        log: PdfLog,
    ) -> None:
        self.document = document
        if isinstance(fallback_resource, _FallbackFontResolver):
            self.fallback_resolver: Optional[_FallbackFontResolver] = fallback_resource
            self.fallback_resource = fallback_resource.default_resource()
        else:
            self.fallback_resolver = None
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
        text = str(unit.translated_text or unit.source_text)
        return self._resolve_named_font(
            page_index=unit.page_index,
            font_name=unit.font_name,
            text=text,
            fallback=lambda: self._fallback_for(unit, text),
            mark_fallback=lambda reason: self._mark_unit_fallback(unit, reason),
        )

    def resolve_style(
        self,
        unit: NativeTextUnit,
        style: NativeTextSpanStyle,
        text: str,
    ) -> _FontResource:
        return self._resolve_named_font(
            page_index=unit.page_index,
            font_name=style.font_name,
            text=text,
            fallback=lambda: self._fallback_for_style(style, text),
            mark_fallback=lambda reason: self._mark_unit_fallback(unit, reason),
        )

    def _resolve_named_font(
        self,
        *,
        page_index: int,
        font_name: str,
        text: str,
        fallback: Callable[[], _FontResource],
        mark_fallback: Callable[[str], None],
    ) -> _FontResource:
        entry = self._find_font_entry(page_index, font_name)
        if entry is None:
            self._warn_once(font_name, "not_found")
            mark_fallback("font_fallback_not_found")
            return fallback()
        resource = self._resource_from_xref(int(entry[0] or 0))
        if resource is None:
            self._warn_once(font_name, "not_embedded")
            mark_fallback("font_fallback_not_embedded")
            return fallback()
        if not _font_resource_supports_text(resource, text):
            self._warn_once(font_name, "missing_glyphs")
            mark_fallback("font_fallback_missing_glyphs")
            return fallback()
        return resource

    def _fallback_for(self, unit: NativeTextUnit, text: str) -> _FontResource:
        if self.fallback_resolver is not None:
            return self.fallback_resolver.resolve(unit, text)
        return self.fallback_resource

    def _fallback_for_style(self, style: NativeTextSpanStyle, text: str) -> _FontResource:
        if self.fallback_resolver is not None:
            return self.fallback_resolver.resolve_style(style, text)
        return self.fallback_resource

    def _mark_unit_fallback(self, unit: NativeTextUnit, reason: str) -> None:
        if unit.font_fallback_reason is None:
            unit.font_fallback_reason = reason

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
            family_kind=_font_family_kind(extracted_name or ""),
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
    font_resource: Any,
) -> Optional[Dict[str, Any]]:
    text = str(unit.translated_text or unit.source_text)
    font_bundle = _ensure_font_bundle(font_resource)
    if unit.role == "TOC" and _insert_toc_unit(page, unit, font_bundle):
        return None
    line_height = _unit_line_height(unit, NORMAL_LINE_HEIGHT)
    compact_line_height = min(line_height, COMPACT_LINE_HEIGHT)
    css = _unit_css(unit, font_bundle, line_height=line_height)
    compact_css = _unit_css(unit, font_bundle, line_height=compact_line_height)
    html_text = _unit_html(unit, font_bundle, text)
    source_rect = fitz.Rect(unit.layout_bounds if unit.framed_label else unit.bbox)
    candidates = _candidate_rects(
        (page.rect & fitz.Rect(unit.layout_bounds)) if unit.layout_bounds else page.rect,
        source_rect,
        "TEXT" if unit.layout_bounds else unit.role,
        occupied,
        alignment=unit.alignment,
        rotation=unit.rotation,
    )
    relaxed_candidates = _relaxed_candidate_order(candidates)
    if _insert_single_line_unit(page, unit, font_bundle, candidates):
        return None

    for candidate in candidates:
        warning = _try_insert_htmlbox(
            page,
            unit,
            candidate,
            html_text,
            css,
            font_bundle,
            scale_low=PREFERRED_NATIVE_FONT_SCALE,
        )
        if warning is not None or unit.rendered_bbox is not None:
            return warning

    for candidate in relaxed_candidates:
        warning = _try_insert_htmlbox(
            page,
            unit,
            candidate,
            html_text,
            compact_css,
            font_bundle,
            scale_low=PREFERRED_NATIVE_FONT_SCALE,
        )
        if warning is not None or unit.rendered_bbox is not None:
            return warning

    for candidate in relaxed_candidates:
        warning = _try_insert_htmlbox(
            page,
            unit,
            candidate,
            html_text,
            compact_css,
            font_bundle,
            scale_low=MIN_NATIVE_FONT_SCALE,
        )
        if warning is not None or unit.rendered_bbox is not None:
            return warning

    forced_rect = relaxed_candidates[0] if relaxed_candidates else candidates[-1]
    warning = _try_insert_htmlbox(
        page,
        unit,
        forced_rect,
        html_text,
        compact_css,
        font_bundle,
        scale_low=FORCED_FONT_SCALE,
    )
    if warning is not None or unit.rendered_bbox is not None:
        return warning

    warning = _try_insert_htmlbox(
        page,
        unit,
        forced_rect,
        html_text,
        compact_css,
        font_bundle,
        scale_low=EMERGENCY_FONT_SCALE,
    )
    if warning is not None or unit.rendered_bbox is not None:
        return warning

    warning = _try_insert_htmlbox(
        page,
        unit,
        forced_rect,
        html_text,
        compact_css,
        font_bundle,
        scale_low=0,
    )
    if warning is not None or unit.rendered_bbox is not None:
        return warning

    unit.rendered_bbox = _rect_tuple(forced_rect)
    unit.font_scale = 0.0
    unit.layout_warning = "translation_did_not_fit"
    return _layout_warning(unit, "translation_did_not_fit")


def _insert_single_line_unit(
    page: Any, unit: NativeTextUnit, fonts: _FontBundle, candidates: Sequence[Any],
) -> bool:
    """Preserve the native baseline without fitting an EM box into a glyph box."""
    text = str(unit.translated_text or unit.source_text)
    resource = fonts.default
    if len(unit.lines) != 1 or "\n" in text or unit.rotation or resource.font is None or unit.framed_label:
        return False
    if not _font_resource_supports_text(resource, text):
        return False
    styles = unit.style_runs or unit.lines[0].spans
    if any(_style_resource_key(span) != _style_resource_key(_unit_default_style(unit)) or
           abs(span.font_size - unit.font_size) > 0.1 or span.color != unit.color or span.alpha != 255
           for span in styles):
        return False
    width = resource.font.text_length(text, fontsize=unit.font_size)
    for candidate in candidates:
        if width > candidate.width:
            continue
        left = candidate.x0
        if unit.alignment == "center":
            left += (candidate.width - width) / 2
        elif unit.alignment == "right":
            left = candidate.x1 - width
        page.insert_font(fontname=resource.family, fontbuffer=resource.font.buffer)
        color = tuple(((unit.color >> shift) & 255) / 255 for shift in (16, 8, 0))
        page.insert_text((left, unit.lines[0].origin[1]), text, fontname=resource.family,
                         fontsize=unit.font_size, color=color)
        unit.rendered_bbox = _rect_tuple(candidate)
        unit.font_scale = 1.0
        return True
    return False


def _insert_toc_unit(page: Any, unit: NativeTextUnit, fonts: _FontBundle) -> bool:
    """Keep the page-number column fixed instead of translating literal leaders."""
    match = re.match(r"^(.*?)\s*\.{3,}\s*(\d+)\s*$", unit.translated_text or unit.source_text)
    resource = fonts.default
    if match is None or resource.font is None or unit.rotation:
        return False
    if not _font_resource_supports_text(resource, match.group(0)):
        return False
    title, number = match.groups()
    size = unit.font_size
    font = resource.font
    rect = fitz.Rect(unit.bbox)
    number_width = font.text_length(number, fontsize=size)
    available = max(size, rect.width - number_width - size * 1.5)
    title_width = font.text_length(title, fontsize=size)
    scale = min(1.0, available / max(1.0, title_width))
    baseline = unit.lines[0].origin[1] if unit.lines else rect.y1 - size * 0.2
    name = resource.family
    page.insert_font(fontname=name, fontbuffer=font.buffer)
    color = tuple(((unit.color >> shift) & 255) / 255 for shift in (16, 8, 0))
    for x, value, font_size in (
        (rect.x0, title, size * scale),
        (rect.x1 - number_width, number, size),
    ):
        page.insert_text((x, baseline), value, fontname=name, fontsize=font_size, color=color)
    dot_width = font.text_length(".", fontsize=size)
    start = rect.x0 + title_width * scale + size * 0.4
    count = int(max(0, rect.x1 - number_width - size * 0.4 - start) / max(0.1, dot_width))
    if count:
        page.insert_text((start, baseline), "." * count, fontname=name, fontsize=size, color=color)
    unit.rendered_bbox = unit.bbox
    unit.font_scale = scale
    return True


def _try_insert_htmlbox(
    page: Any,
    unit: NativeTextUnit,
    candidate: Any,
    html_text: str,
    css: str,
    font_bundle: _FontBundle,
    *,
    scale_low: float,
) -> Optional[Dict[str, Any]]:
    spare_height, scale = page.insert_htmlbox(
        candidate,
        html_text,
        css=css,
        scale_low=scale_low,
        archive=font_bundle.archive,
        rotate=unit.rotation,
        overlay=True,
    )
    if spare_height < 0:
        return None

    unit.rendered_bbox = _rect_tuple(candidate)
    unit.font_scale = float(scale)
    if unit.font_scale >= PREFERRED_NATIVE_FONT_SCALE:
        return None
    if unit.font_scale < EMERGENCY_FONT_SCALE:
        reason = "font_scaled_below_readable_minimum"
    elif unit.font_scale < MIN_NATIVE_FONT_SCALE:
        reason = "font_scaled_below_minimum"
    else:
        reason = "font_scaled_below_preferred_minimum"
    unit.layout_warning = reason
    return _layout_warning(unit, reason)


def _layout_warning(unit: NativeTextUnit, reason: str) -> Dict[str, Any]:
    return {
        "page": unit.page_index + 1,
        "stable_id": unit.stable_id,
        "role": unit.role,
        "reason": reason,
        "scale": unit.font_scale,
        "bbox": list(unit.rendered_bbox or unit.bbox),
    }


def _unit_line_height(unit: NativeTextUnit, default: float) -> float:
    baselines = sorted(
        line.origin[1]
        for line in unit.lines
        if isinstance(line.origin, tuple) and len(line.origin) == 2
    )
    if len(baselines) < 2:
        return default
    gaps = [
        baselines[index + 1] - baselines[index]
        for index in range(len(baselines) - 1)
        if baselines[index + 1] > baselines[index]
    ]
    if not gaps:
        return default
    sizes = [line.font_size for line in unit.lines if line.font_size > 0]
    if not sizes:
        return default
    ratio = median(gaps) / median(sizes)
    return max(0.92, min(1.40, float(ratio)))


def _relaxed_candidate_order(candidates: Sequence[Any]) -> List[Any]:
    return sorted(
        candidates,
        key=lambda rect: (float(rect.get_area()), float(rect.width), float(rect.height)),
        reverse=True,
    )


def _unit_css(
    unit: NativeTextUnit,
    font_resource: Any,
    *,
    line_height: float = NORMAL_LINE_HEIGHT,
) -> str:
    font_bundle = _ensure_font_bundle(font_resource)
    default_resource = font_bundle.default
    color = f"#{unit.color & 0xFFFFFF:06x}"
    weight = "700" if unit.bold else "400"
    font_style = "italic" if unit.italic else "normal"
    return (
        font_bundle.font_face_css
        + f" * {{ font-family: {default_resource.family}; font-size: {unit.font_size:.2f}pt; "
        f"font-weight: {weight}; font-style: {font_style}; color: {color}; text-align: {unit.alignment}; "
        f"line-height: {line_height:.2f}; margin: 0; padding: 0; }}"
    )


def _unit_html(unit: NativeTextUnit, font_resource: Any, text: str) -> str:
    font_bundle = _ensure_font_bundle(font_resource)
    translated_lines = str(text or "").splitlines()
    if translated_lines and len(translated_lines) == len(unit.lines):
        rendered_lines = []
        for translated_line, source_line in zip(translated_lines, unit.lines):
            rendered_lines.append(
                "<div>"
                + _styled_line_html(translated_line, source_line.spans, unit, font_bundle)
                + "</div>"
            )
        return "".join(rendered_lines)

    indent = 0.0
    if len(unit.lines) > 1 and unit.alignment == "left" and not unit.rotation:
        indent = max(0.0, min(unit.font_size * 3, unit.lines[0].bbox[0] - unit.bbox[0]))
    opening = f'<div style="text-indent:{indent:.2f}pt">' if indent > 0.5 else "<div>"
    return (
        opening
        + _styled_line_html(
            str(text or ""),
            unit.style_runs if len(unit.style_runs) > 1 else [],
            unit,
            font_bundle,
        ).replace("\n", "<br>")
        + "</div>"
    )


class EditableTextInserter:
    """Shared HTMLBox writer for exact OCR boxes and native text units."""

    def __init__(self, style_font: Optional[str] = None) -> None:
        self._resolver = _FallbackFontResolver(style_font)

    def insert_units(
        self,
        page: Any,
        units: Sequence[NativeTextUnit],
        *,
        allow_expansion: bool = False,
        occupied: Sequence[Any] = (),
    ) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []
        for unit in units:
            text = str(unit.translated_text or unit.source_text or "").strip()
            if not text or fitz.Rect(unit.bbox).is_empty:
                continue
            font_bundle = _resolve_unit_font_bundle(
                unit,
                original_resolver=None,
                fallback_resolver=self._resolver,
            )
            if allow_expansion:
                warning = _insert_unit(
                    page,
                    unit,
                    occupied=occupied,
                    font_resource=font_bundle,
                )
            else:
                warning = self._insert_exact(page, unit, font_bundle)
            if warning is not None:
                warnings.append(warning)
        return warnings

    @staticmethod
    def _insert_exact(
        page: Any,
        unit: NativeTextUnit,
        font_bundle: _FontBundle,
    ) -> Optional[Dict[str, Any]]:
        html_text = _unit_html(
            unit,
            font_bundle,
            str(unit.translated_text or unit.source_text),
        )
        css = _unit_css(unit, font_bundle)
        compact_css = _unit_css(
            unit,
            font_bundle,
            line_height=COMPACT_LINE_HEIGHT,
        )
        warning: Optional[Dict[str, Any]] = None
        for attempt_css, scale_low in (
            (css, PREFERRED_NATIVE_FONT_SCALE),
            (compact_css, PREFERRED_NATIVE_FONT_SCALE),
            (compact_css, MIN_NATIVE_FONT_SCALE),
            (compact_css, FORCED_FONT_SCALE),
            (compact_css, EMERGENCY_FONT_SCALE),
            (compact_css, 0.0),
        ):
            warning = _try_insert_htmlbox(
                page,
                unit,
                fitz.Rect(unit.bbox),
                html_text,
                attempt_css,
                font_bundle,
                scale_low=scale_low,
            )
            if unit.rendered_bbox is not None:
                return warning
        unit.layout_warning = "translation_did_not_fit"
        return _layout_warning(unit, "translation_did_not_fit")


def _styled_line_html(
    text: str,
    source_styles: Sequence[NativeTextSpanStyle],
    unit: NativeTextUnit,
    font_resource: Any,
) -> str:
    font_bundle = _ensure_font_bundle(font_resource)
    styles = [style for style in source_styles if style.text.strip()]
    if len(styles) <= 1:
        style = styles[0] if styles else _unit_default_style(unit)
        return (
            f'<span style="{_inline_style(style, font_bundle)}">'
            + html.escape(text)
            + "</span>"
        )
    parts = _split_text_by_style_runs(text, styles)
    return "".join(
        f'<span style="{_inline_style(style, font_bundle)}">{html.escape(part)}</span>'
        for part, style in zip(parts, styles)
        if part
    )


def _unit_default_style(unit: NativeTextUnit) -> NativeTextSpanStyle:
    return NativeTextSpanStyle(
        text=unit.source_text,
        bbox=unit.bbox,
        font_size=unit.font_size,
        font_name=unit.font_name,
        color=unit.color,
        bold=unit.bold,
        italic=unit.italic,
        font_family_kind=unit.font_family_kind,
    )


def _inline_style(style: NativeTextSpanStyle, font_resource: Any) -> str:
    font_bundle = _ensure_font_bundle(font_resource)
    resolved_resource = font_bundle.resource_for(style)
    weight = "700" if style.bold else "400"
    font_style = "italic" if style.italic else "normal"
    color = f"#{style.color & 0xFFFFFF:06x}"
    opacity = ""
    if style.alpha < 255:
        opacity = f"opacity:{max(0.0, min(1.0, style.alpha / 255)):.3f};"
    return (
        f"font-family:{resolved_resource.family};"
        f"font-size:{max(4.0, style.font_size):.2f}pt;"
        f"font-weight:{weight};"
        f"font-style:{font_style};"
        f"color:{color};"
        f"{opacity}"
    )


def _split_text_by_style_runs(
    text: str,
    styles: Sequence[NativeTextSpanStyle],
) -> List[str]:
    if not styles:
        return [text]
    source_lengths = [max(1, len(style.text.strip())) for style in styles]
    total_source = sum(source_lengths)
    if total_source <= 0 or len(styles) == 1:
        return [text]

    parts: List[str] = []
    start = 0
    total_len = len(text)
    cumulative = 0
    for source_length in source_lengths[:-1]:
        cumulative += source_length
        target = int(round(total_len * cumulative / total_source))
        split_at = _nearest_word_boundary(text, target, start)
        parts.append(text[start:split_at])
        start = split_at
    parts.append(text[start:])
    return parts


def _nearest_word_boundary(text: str, target: int, minimum: int) -> int:
    target = max(minimum, min(len(text), target))
    if target >= len(text):
        return len(text)
    if target <= minimum:
        return target
    left = text.rfind(" ", minimum, target + 1)
    right = text.find(" ", target)
    candidates = [candidate for candidate in (left, right) if candidate >= minimum]
    if not candidates:
        return target
    boundary = min(candidates, key=lambda candidate: abs(candidate - target))
    return min(len(text), boundary + 1) if text[boundary : boundary + 1] == " " else boundary


def _candidate_rects(
    page_rect: Any,
    source_rect: Any,
    role: str,
    occupied: Sequence[Any],
    alignment: str = "left",
    rotation: int = 0,
) -> List[Any]:
    base = fitz.Rect(source_rect)
    candidates = [base]
    if role == "CELL":
        return candidates

    expansions: List[Any] = []
    ratios = [(0.20, 0.0)]
    if rotation % 180 == 0:
        ratios.extend([(0.50, 0.0), (1.00, 0.0), (1.50, 0.0), (2.00, 0.0), (3.00, 0.0)])
    ratios.extend((
        (0.0, 0.35),
        (0.20, 0.35),
        (0.50, 0.50),
        (1.00, 0.50),
        (1.50, 1.00),
    ))
    for width_ratio, height_ratio in ratios:
        width_extra = source_rect.width * width_ratio
        height_extra = source_rect.height * height_ratio
        if str(alignment or "left").strip().lower() == "center":
            symmetric_room = max(
                0.0,
                2.0
                * min(
                    source_rect.x0 - page_rect.x0,
                    page_rect.x1 - source_rect.x1,
                ),
            )
            width_extra = min(width_extra, symmetric_room)
        x0, x1 = _anchored_horizontal_bounds(
            source_rect,
            width_extra,
            alignment,
        )
        expansions.append(
            fitz.Rect(
                x0,
                source_rect.y0,
                x1,
                source_rect.y1 + height_extra,
            )
        )
    for candidate in expansions:
        candidate &= page_rect
        if candidate.is_empty or candidate in candidates:
            continue
        if _expansion_is_free(candidate, source_rect, occupied):
            candidates.append(candidate)
    return candidates


def _anchored_horizontal_bounds(
    source_rect: Any,
    width_extra: float,
    alignment: str,
) -> Tuple[float, float]:
    normalized = str(alignment or "left").strip().lower()
    if normalized == "right":
        return (source_rect.x0 - width_extra, source_rect.x1)
    if normalized == "center":
        return (source_rect.x0 - width_extra / 2, source_rect.x1 + width_extra / 2)
    return (source_rect.x0, source_rect.x1 + width_extra)


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
    *,
    expected_identity: Optional[dict] = None,
) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = read_layer_payload(path, log, expected_identity)
    if not payload:
        return {}

    by_id = {unit.stable_id: unit for unit in units}
    cached: Dict[str, str] = {}
    try:
        version = int(payload.get("version", 1) or 1)
    except (TypeError, ValueError, OverflowError):
        return {}
    if version >= 2:
        if str(payload.get("document_sha256") or "") != file_hash:
            log("PDF: JSON слой относится к другой версии исходного файла")
            return {}
        for block in _iter_layer_blocks(payload):
            stable_id = str(block.get("stable_id") or "")
            unit = by_id.get(stable_id)
            if unit is not None and not _layer_block_matches_unit(block, unit):
                unit = None
            if unit is None:
                unit = _find_layer_unit_by_text_bbox(block, units)
            if unit is None:
                continue
            translation = str(block.get("translated_text") or "").strip()
            if translation:
                cached[unit.stable_id] = translation
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
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return
    for page in pages:
        if not isinstance(page, dict):
            continue
        try:
            page_index = int(page.get("page_index", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        blocks = page.get("blocks") or page.get("regions") or []
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict):
                yield {**block, "page_index": page_index}


def _layer_block_matches_unit(block: Dict[str, Any], unit: NativeTextUnit) -> bool:
    return (
        str(block.get("text") or "") == unit.source_text
        and block.get("page_index", unit.page_index) == unit.page_index
        and _bbox_matches(block.get("bbox"), unit.bbox)
    )


def _legacy_layer_block_matches_unit(block: Dict[str, Any], unit: NativeTextUnit) -> bool:
    return _layer_block_matches_unit(block, unit)


def _find_layer_unit_by_text_bbox(
    block: Dict[str, Any],
    units: Sequence[NativeTextUnit],
) -> Optional[NativeTextUnit]:
    source_text = str(block.get("text") or "")
    bbox = _coerce_bbox(block.get("bbox"))
    if not source_text or bbox is None:
        return None
    block_rect = fitz.Rect(bbox)
    best_unit: Optional[NativeTextUnit] = None
    best_score = 0.0
    for unit in units:
        if block.get("page_index", unit.page_index) != unit.page_index:
            continue
        if source_text != unit.source_text:
            continue
        score = _rect_overlap_score(block_rect, fitz.Rect(unit.bbox))
        if score > best_score:
            best_score = score
            best_unit = unit
    return best_unit if best_score >= 0.80 else None


def _save_translation_layer(
    path: Optional[Path],
    *,
    file_hash: str,
    pages: Sequence[NativePage],
    native_pages: Sequence[int],
    ocr_pages: Sequence[int],
    log: PdfLog,
    cache_identity: Optional[dict] = None,
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
                "layout_bounds": list(unit.layout_bounds) if unit.layout_bounds else None,
                "framed_label": unit.framed_label,
                "text": unit.source_text,
                "translated_text": unit.translated_text or unit.source_text,
                "font_scale": unit.font_scale,
                "layout_warning": unit.layout_warning,
                "font_fallback_reason": unit.font_fallback_reason,
                "style": {
                    "font_size": unit.font_size,
                    "font_name": unit.font_name,
                    "font_family_kind": unit.font_family_kind,
                    "color": unit.color,
                    "bold": unit.bold,
                    "italic": unit.italic,
                    "rotation": unit.rotation,
                    "alignment": unit.alignment,
                },
                "lines": [
                    {
                        "text": line.text,
                        "bbox": list(line.bbox),
                        "origin": list(line.origin),
                        "font_size": line.font_size,
                        "direction": list(line.direction),
                        "writing_mode": line.writing_mode,
                        "spans": [
                            {
                                "text": span.text,
                                "bbox": list(span.bbox),
                                "font_size": span.font_size,
                                "font_name": span.font_name,
                                "font_family_kind": span.font_family_kind,
                                "color": span.color,
                                "bold": span.bold,
                                "italic": span.italic,
                                "flags": span.flags,
                                "char_flags": span.char_flags,
                                "alpha": span.alpha,
                                "ascender": span.ascender,
                                "descender": span.descender,
                                "origin": list(span.origin) if span.origin else None,
                                "bidi": span.bidi,
                                "char_count": span.char_count,
                                "average_char_width": span.average_char_width,
                            }
                            for span in line.spans
                        ],
                    }
                    for line in unit.lines
                ],
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
        "version": 4,
        "cache_identity": cache_identity,
        "document_sha256": file_hash,
        "processing_mode": "native" if not ocr_pages else "hybrid",
        "native_pages": list(native_pages),
        "ocr_pages": list(ocr_pages),
        "pages": payload_pages,
    }
    write_json_atomic(path, payload)
    log(f"PDF: сохранён JSON слой v4: {path}")


def _save_output_document(
    document: Any,
    fallback_documents: Dict[int, bytes],
    output_path: Path,
    *,
    link_snapshots: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    page_labels: Optional[List[Dict[str, Any]]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_document = document
    composed = None
    fallback_handles: List[Any] = []
    if fallback_documents:
        composed = fitz.open()
        metadata = document.metadata
        try:
            xml_metadata = document.get_xml_metadata()
        except Exception:
            xml_metadata = ""
        toc = document.get_toc(simple=False)
        for page_index in range(len(document)):
            check_cancelled()
            fallback_bytes = fallback_documents.get(page_index)
            if fallback_bytes is None:
                composed.insert_pdf(document, from_page=page_index, to_page=page_index)
                continue
            fallback = fitz.open(stream=fallback_bytes, filetype="pdf")
            fallback_handles.append(fallback)
            composed.insert_pdf(fallback, from_page=0, to_page=0)
        composed.set_metadata(metadata)
        if xml_metadata:
            try:
                composed.set_xml_metadata(xml_metadata)
            except Exception:
                pass
        if toc:
            try:
                composed.set_toc(toc)
            except Exception:
                pass
        output_document = composed

    _restore_page_labels(output_document, page_labels or [])
    _restore_document_links(output_document, link_snapshots or {})

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


def _snapshot_document_links(document: Any) -> Dict[int, List[Dict[str, Any]]]:
    snapshots: Dict[int, List[Dict[str, Any]]] = {}
    for page_index in range(len(document)):
        check_cancelled()
        try:
            links = document[page_index].get_links()
        except Exception:
            links = []
        clean_links = [
            clean
            for link in links
            if (clean := _clean_link_dict(link)) is not None
        ]
        if clean_links:
            snapshots[page_index] = clean_links
    return snapshots


def _restore_document_links(
    document: Any,
    snapshots: Dict[int, List[Dict[str, Any]]],
) -> None:
    if not snapshots:
        return
    for page_index, links in snapshots.items():
        if page_index < 0 or page_index >= len(document):
            continue
        page = document[page_index]
        try:
            existing = [
                clean
                for link in page.get_links()
                if (clean := _clean_link_dict(link)) is not None
            ]
        except Exception:
            existing = []
        for link in links:
            if any(_links_equal(link, candidate) for candidate in existing):
                continue
            try:
                page.insert_link(dict(link))
                existing.append(link)
            except Exception:
                continue


def _clean_link_dict(link: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(link, dict):
        return None
    cleaned: Dict[str, Any] = {}
    for key, value in link.items():
        if key in {"xref", "id"}:
            continue
        if key == "from":
            try:
                cleaned[key] = fitz.Rect(value)
            except Exception:
                return None
        elif key == "to":
            try:
                cleaned[key] = fitz.Point(value)
            except Exception:
                cleaned[key] = value
        else:
            cleaned[key] = value
    return cleaned if "kind" in cleaned and "from" in cleaned else None


def _links_equal(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    if first.get("kind") != second.get("kind"):
        return False
    if not _bbox_matches_rect(first.get("from"), second.get("from")):
        return False
    for key in ("uri", "file", "page", "xref", "nameddest"):
        if first.get(key) != second.get(key):
            return False
    return _points_equal(first.get("to"), second.get("to"))


def _points_equal(first: Any, second: Any, tolerance: float = 0.01) -> bool:
    if first is None and second is None:
        return True
    if first is None or second is None:
        return False
    try:
        first_point = fitz.Point(first)
        second_point = fitz.Point(second)
    except Exception:
        return first == second
    return (
        abs(first_point.x - second_point.x) <= tolerance
        and abs(first_point.y - second_point.y) <= tolerance
    )


def _bbox_matches_rect(first: Any, second: Any, tolerance: float = 0.01) -> bool:
    try:
        first_rect = fitz.Rect(first)
        second_rect = fitz.Rect(second)
    except Exception:
        return False
    return all(
        abs(a - b) <= tolerance
        for a, b in zip(_rect_tuple(first_rect), _rect_tuple(second_rect))
    )


def _snapshot_page_labels(document: Any) -> List[Dict[str, Any]]:
    try:
        labels = document.get_page_labels()
    except Exception:
        return []
    return [dict(label) for label in labels or [] if isinstance(label, dict)]


def _restore_page_labels(document: Any, labels: Sequence[Dict[str, Any]]) -> None:
    if not labels:
        return
    try:
        document.set_page_labels([dict(label) for label in labels])
    except Exception:
        pass


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


def _coerce_point(
    value: Any,
    default: Tuple[float, float],
) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return default
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return default


def _coerce_optional_point(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _rect_overlap_score(first: Any, second: Any) -> float:
    intersection = first & second
    if intersection.is_empty:
        return 0.0
    denominator = min(first.get_area(), second.get_area())
    return intersection.get_area() / denominator if denominator > 0 else 0.0


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
