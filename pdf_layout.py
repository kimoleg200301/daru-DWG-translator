from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

PdfLayoutLog = Callable[[str], None]
PdfColor = Tuple[int, int, int]
PdfBBox = Tuple[float, float, float, float]


@dataclass
class PdfLayoutSpan:
    text: str
    bbox: PdfBBox
    font: Optional[str]
    font_size: float
    fill_color: Optional[PdfColor]
    background_color: Optional[PdfColor]
    span_index: int


@dataclass
class PdfLayoutLine:
    bbox: PdfBBox
    spans: List[PdfLayoutSpan]
    line_index: int

    def text(self) -> str:
        content = "".join(span.text for span in self.spans)
        return content.strip()


@dataclass
class PdfLayoutBlock:
    bbox: PdfBBox
    lines: List[PdfLayoutLine]
    block_index: int


@dataclass
class PdfLayoutPage:
    page_index: int
    width: float
    height: float
    blocks: List[PdfLayoutBlock]

    def to_text(self) -> str:
        chunks: List[str] = []
        for block in self.blocks:
            block_lines = []
            for line in block.lines:
                text = line.text()
                if text:
                    block_lines.append(text)
            if block_lines:
                chunks.append("\n".join(block_lines))
        return "\n\n".join(chunks)


@dataclass
class PdfLayout:
    pages: List[PdfLayoutPage] = field(default_factory=list)

    def page_texts(self) -> List[str]:
        return [page.to_text() for page in self.pages]


@dataclass
class PdfLineRef:
    page_index: int
    block_index: int
    line_index: int
    text: str
    line: PdfLayoutLine


def collect_line_refs(layout: PdfLayout) -> List[PdfLineRef]:
    refs: List[PdfLineRef] = []
    for page in layout.pages:
        page_idx = page.page_index
        for block in page.blocks:
            block_idx = block.block_index
            for line in block.lines:
                text = line.text()
                if not text:
                    continue
                refs.append(
                    PdfLineRef(
                        page_index=page_idx,
                        block_index=block_idx,
                        line_index=line.line_index,
                        text=text,
                        line=line,
                    )
                )
    return refs


def extract_pdf_layout(pdf_path: Path, log: Optional[PdfLayoutLog] = None) -> PdfLayout:
    _require_pymupdf("Установите PyMuPDF (pip install pymupdf)")
    if log:
        log("Извлекаем макет PDF: координаты текста и стили...")
    doc = fitz.open(str(pdf_path))  # type: ignore[call-arg]
    pages: List[PdfLayoutPage] = []
    try:
        total_pages = doc.page_count  # type: ignore[attr-defined]
        for page_index in range(total_pages):
            page = doc.load_page(page_index)  # type: ignore[attr-defined]
            snapshot = _PageSnapshot(page)
            blocks, fallback_used = _extract_blocks(page, snapshot)
            pages.append(
                PdfLayoutPage(
                    page_index=page_index,
                    width=float(page.rect.width),  # type: ignore[attr-defined]
                    height=float(page.rect.height),  # type: ignore[attr-defined]
                    blocks=blocks,
                )
            )
            if log:
                suffix = ", резервный парсер слов" if fallback_used else ""
                log(
                    f"PDF: макет страницы {page_index + 1}/{total_pages} собран ({len(blocks)} блоков{suffix})"
                )
    finally:
        doc.close()  # type: ignore[attr-defined]
    return PdfLayout(pages)


def _extract_blocks(page: "fitz.Page", snapshot: "_PageSnapshot") -> Tuple[List[PdfLayoutBlock], bool]:
    raw_blocks = _extract_blocks_from_rawdict(page, snapshot)
    if raw_blocks:
        return raw_blocks, False
    return _extract_blocks_from_words(page, snapshot), True


def _extract_blocks_from_rawdict(page: "fitz.Page", snapshot: "_PageSnapshot") -> List[PdfLayoutBlock]:
    page_dict = page.get_text("rawdict")  # type: ignore[attr-defined]
    blocks: List[PdfLayoutBlock] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_bbox = _to_bbox(block.get("bbox"))
        lines: List[PdfLayoutLine] = []
        for line_index, line in enumerate(block.get("lines", [])):
            spans = _build_spans(line.get("spans", []), snapshot)
            if not spans:
                continue
            line_bbox = _to_bbox(line.get("bbox"))
            lines.append(PdfLayoutLine(bbox=line_bbox, spans=spans, line_index=line_index))
        if lines:
            blocks.append(
                PdfLayoutBlock(
                    bbox=block_bbox,
                    lines=lines,
                    block_index=len(blocks),
                )
            )
    return blocks


def _extract_blocks_from_words(page: "fitz.Page", snapshot: "_PageSnapshot") -> List[PdfLayoutBlock]:
    word_items = page.get_text("words")  # type: ignore[attr-defined]
    if not word_items:
        return []
    block_order: List[int] = []
    block_map: Dict[int, Dict[str, object]] = {}
    for entry in word_items:
        if len(entry) < 8:
            continue
        x0, y0, x1, y1, raw_text, block_idx, line_idx, _word_idx = entry[:8]
        try:
            block_key = int(block_idx)
            line_key = int(line_idx)
        except (TypeError, ValueError):
            continue
        text_value = "" if raw_text is None else str(raw_text)
        if not text_value.strip():
            continue
        block_data = block_map.get(block_key)
        if block_data is None:
            block_data = {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "lines": {},
                "line_order": [],
            }
            block_map[block_key] = block_data
            block_order.append(block_key)
        else:
            bbox = block_data["bbox"]
            bbox[0] = min(bbox[0], float(x0))
            bbox[1] = min(bbox[1], float(y0))
            bbox[2] = max(bbox[2], float(x1))
            bbox[3] = max(bbox[3], float(y1))
        lines: Dict[int, Dict[str, object]] = block_data["lines"]
        line_data = lines.get(line_key)
        if line_data is None:
            line_data = {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "spans": [],
                "last_x1": None,
            }
            lines[line_key] = line_data
            block_data["line_order"].append(line_key)
        else:
            line_bbox = line_data["bbox"]
            line_bbox[0] = min(line_bbox[0], float(x0))
            line_bbox[1] = min(line_bbox[1], float(y0))
            line_bbox[2] = max(line_bbox[2], float(x1))
            line_bbox[3] = max(line_bbox[3], float(y1))
        bbox_tuple = (float(x0), float(y0), float(x1), float(y1))
        font_size = abs(float(y1) - float(y0))
        if font_size <= 0:
            font_size = 11.0
        span_text = text_value
        previous_x1 = line_data.get("last_x1")
        if previous_x1 is not None:
            gap = float(x0) - float(previous_x1)
            threshold = max(0.8, font_size * 0.25)
            if gap > threshold:
                span_text = f" {span_text}"
        spans_list: List[PdfLayoutSpan] = line_data["spans"]
        spans_list.append(
            PdfLayoutSpan(
                text=span_text,
                bbox=bbox_tuple,
                font=None,
                font_size=float(max(6.0, min(72.0, font_size))),
                fill_color=None,
                background_color=snapshot.sample_background(bbox_tuple),
                span_index=len(spans_list),
            )
        )
        line_data["last_x1"] = float(x1)
    blocks: List[PdfLayoutBlock] = []
    for block_idx in block_order:
        block_data = block_map.get(block_idx)
        if not block_data:
            continue
        line_objects: List[PdfLayoutLine] = []
        line_order: List[int] = block_data["line_order"]
        lines = block_data["lines"]
        for display_idx, line_idx in enumerate(line_order):
            line_data = lines.get(line_idx)
            if not line_data:
                continue
            spans = line_data["spans"]
            if not spans:
                continue
            line_objects.append(
                PdfLayoutLine(
                    bbox=tuple(line_data["bbox"]) if line_data["bbox"] else (0.0, 0.0, 0.0, 0.0),  # type: ignore[arg-type]
                    spans=spans,
                    line_index=display_idx,
                )
            )
        if line_objects:
            bbox = block_data["bbox"] if block_data["bbox"] else [0.0, 0.0, 0.0, 0.0]
            blocks.append(
                PdfLayoutBlock(
                    bbox=tuple(bbox),  # type: ignore[arg-type]
                    lines=line_objects,
                    block_index=len(blocks),
                )
            )
    return blocks


def _build_spans(spans_raw: Iterable[dict], snapshot: "_PageSnapshot") -> List[PdfLayoutSpan]:
    spans: List[PdfLayoutSpan] = []
    for span_index, span in enumerate(spans_raw):
        text = span.get("text", "")
        if not text:
            continue
        bbox = _to_bbox(span.get("bbox"))
        font = span.get("font")
        size = float(span.get("size", 0.0))
        fill_color = _decode_color(span.get("color"))
        background_color = snapshot.sample_background(bbox)
        spans.append(
            PdfLayoutSpan(
                text=text,
                bbox=bbox,
                font=font,
                font_size=size,
                fill_color=fill_color,
                background_color=background_color,
                span_index=span_index,
            )
        )
    return spans


def _to_bbox(value: Optional[Sequence[float]]) -> PdfBBox:
    if not value:
        return (0.0, 0.0, 0.0, 0.0)
    if len(value) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(v) for v in value)  # type: ignore[return-value]


def _decode_color(value: Optional[float]) -> Optional[PdfColor]:
    if value is None:
        return None
    try:
        color_value = int(value)
    except (TypeError, ValueError):
        return None
    r = (color_value >> 16) & 0xFF
    g = (color_value >> 8) & 0xFF
    b = color_value & 0xFF
    return (r, g, b)


def _require_pymupdf(message: str) -> None:
    if fitz is None:  # pragma: no cover - optional dependency
        raise RuntimeError(message)


class _PageSnapshot:
    def __init__(self, page: "fitz.Page") -> None:
        self.rect = page.rect  # type: ignore[attr-defined]
        self.matrix = fitz.Matrix(1, 1)  # type: ignore[attr-defined]
        pixmap = page.get_pixmap(matrix=self.matrix, alpha=False)  # type: ignore[attr-defined]
        self.width = pixmap.width
        self.height = pixmap.height
        self.n = pixmap.n
        self.samples = pixmap.samples
        self.stride = pixmap.stride
        self.scale_x = self.width / self.rect.width if self.rect.width else 1.0
        self.scale_y = self.height / self.rect.height if self.rect.height else 1.0

    def sample_background(self, bbox: PdfBBox) -> Optional[PdfColor]:
        rect = fitz.Rect(bbox)  # type: ignore[attr-defined]
        if rect.is_empty:  # type: ignore[attr-defined]
            return None
        margin = max(1.5, max(rect.width, rect.height) * 0.05)  # type: ignore[attr-defined]
        expanded = fitz.Rect(
            rect.x0 - margin,
            rect.y0 - margin,
            rect.x1 + margin,
            rect.y1 + margin,
        )  # type: ignore[attr-defined]
        clipped = expanded & self.rect  # type: ignore[attr-defined]
        if clipped.is_empty:  # type: ignore[attr-defined]
            return None
        x0 = self._to_pixel_x(clipped.x0)
        y0 = self._to_pixel_y(clipped.y0)
        x1 = self._to_pixel_x(clipped.x1)
        y1 = self._to_pixel_y(clipped.y1)
        if x1 - x0 > 220:
            center_x = (x0 + x1) // 2
            half = 110
            x0 = max(0, center_x - half)
            x1 = min(self.width, center_x + half)
        if y1 - y0 > 220:
            center_y = (y0 + y1) // 2
            half = 110
            y0 = max(0, center_y - half)
            y1 = min(self.height, center_y + half)
        if x1 <= x0 or y1 <= y0:
            return None
        total_r = total_g = total_b = 0
        count = 0
        data = self.samples
        stride = self.stride
        channels = self.n
        for py in range(y0, y1):
            row_offset = py * stride
            for px in range(x0, x1):
                offset = row_offset + px * channels
                if offset + 2 >= len(data):
                    continue
                total_r += data[offset]
                total_g += data[offset + 1]
                total_b += data[offset + 2]
                count += 1
        if count == 0:
            return None
        return (
            int(total_r / count),
            int(total_g / count),
            int(total_b / count),
        )

    def _to_pixel_x(self, value: float) -> int:
        relative = (value - self.rect.x0) * self.scale_x
        return max(0, min(self.width, int(relative)))

    def _to_pixel_y(self, value: float) -> int:
        relative = (value - self.rect.y0) * self.scale_y
        return max(0, min(self.height, int(relative)))
