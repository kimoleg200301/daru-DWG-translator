"""Local OCR detection, layout analysis and text preprocessing."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .common import _noop_log, _require_dependency, _safe_float, _safe_int
from .config import PdfProcessingConfig
from .imaging import _estimate_background_color, _estimate_text_color, build_text_mask
from .models import (
    BBoxWH,
    DetectedSegment,
    MergedRegion,
    PageImage,
    PdfLog,
    RegionInfo,
    TextBlock,
    WordBox,
)

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pytesseract import Output
except ImportError:  # pragma: no cover
    Output = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore


class RegionDetector:
    """
    Detect text regions on a page image using EasyOCR as the primary detector,
    with pytesseract word boxes as a fallback. Detected segments are merged into
    coherent line/block regions before further analysis.
    """

    def __init__(self, config: PdfProcessingConfig, log: PdfLog = _noop_log) -> None:
        self.config = config
        self.log = log
        self._easyocr_reader = self._init_easyocr_reader()

    def detect(self, page: PageImage) -> List[MergedRegion]:
        _require_dependency(cv2, "Установите opencv-python (pip install opencv-python)")
        bgr = cv2.cvtColor(np.array(page.image.convert("RGB")), cv2.COLOR_RGB2BGR)
        segments: List[DetectedSegment] = []

        if self._easyocr_reader is not None:
            segments = self._detect_with_easyocr(bgr)

        if not segments:
            segments = self._detect_with_tesseract(page.image, page.width, page.height)

        if not segments:
            self.log(f"Региональный детектор: на странице {page.page_index + 1} не найдено кандидатов")
            return []

        merged = self._merge_segments(segments, page.width, page.height)
        merged = self._filter_regions(merged, page.width, page.height)
        self.log(f"Региональный детектор: объединено {len(merged)} областей на странице {page.page_index + 1}")
        return merged

    def _init_easyocr_reader(self) -> Optional["easyocr.Reader"]:
        if easyocr is None:
            self.log("EasyOCR не установлен, используется pytesseract для fallback-детекции")
            return None
        languages = self._resolve_easyocr_languages()
        try:
            reader = easyocr.Reader(languages, gpu=bool(self.config.easyocr_gpu))
            self.log(f"EasyOCR: инициализирован с языками {languages}")
            return reader
        except Exception as exc:  # pragma: no cover - GPU/CPU differences
            self.log(f"EasyOCR: не удалось инициализировать ({exc}), используется pytesseract fallback")
            return None

    def _resolve_easyocr_languages(self) -> List[str]:
        if self.config.easyocr_languages:
            langs = [lang.strip() for lang in self.config.easyocr_languages if lang.strip()]
            if langs:
                return langs
        raw = self.config.ocr_languages or "eng"
        tokens = [token for token in re.split(r"[+,/;\\s]+", raw) if token]
        mapping = {"eng": "en", "en": "en", "rus": "ru", "ru": "ru"}
        langs = [mapping.get(token.lower(), token.lower()) for token in tokens]
        return langs or ["en"]

    def _detect_with_easyocr(self, image_bgr: "np.ndarray") -> List[DetectedSegment]:
        """Primary detection path using EasyOCR (detection + OCR)."""
        _require_dependency(easyocr, "Установите easyocr (pip install easyocr)")
        try:
            results = self._easyocr_reader.readtext(image_bgr, detail=1)  # type: ignore[operator]
        except Exception as exc:
            self.log(f"EasyOCR: ошибка детекции ({exc}), используем fallback")
            return []

        h, w = image_bgr.shape[:2]
        min_w = max(1, self.config.min_box_width)
        min_h = max(1, self.config.min_box_height)
        min_conf = max(0.0, min(1.0, self.config.detector_min_confidence))

        segments: List[DetectedSegment] = []
        for result in results:
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                continue
            bbox, text, conf = result[0], result[1], result[2]
            try:
                conf_value = float(conf)
            except Exception:
                conf_value = 0.0
            if conf_value < min_conf:
                continue
            coords_x = [int(p[0]) for p in bbox]
            coords_y = [int(p[1]) for p in bbox]
            x0 = max(0, min(coords_x))
            y0 = max(0, min(coords_y))
            x1 = min(w, max(coords_x))
            y1 = min(h, max(coords_y))
            if x1 <= x0 or y1 <= y0:
                continue
            bw = x1 - x0
            bh = y1 - y0
            if bw < min_w or bh < min_h:
                continue
            clean_text = str(text or "").strip()
            if not clean_text:
                continue
            segments.append(DetectedSegment(bbox=(x0, y0, x1, y1), text=clean_text, confidence=conf_value))
        return segments

    def _detect_with_tesseract(self, image: "Image.Image", width: int, height: int) -> List[DetectedSegment]:
        """Fallback detection path using pytesseract word boxes."""
        data = run_ocr(image, languages=self.config.ocr_languages)
        segments: List[DetectedSegment] = []
        length = len(data.get("text", []))
        for idx in range(length):
            level = _safe_int(data.get("level", []), idx)
            if level != 5:
                continue
            text = (data.get("text", [""])[idx] or "").strip()
            if not text:
                continue
            conf_value = _safe_float(data.get("conf", []), idx)
            if conf_value < self.config.min_confidence:
                continue
            left = _safe_int(data.get("left", []), idx)
            top = _safe_int(data.get("top", []), idx)
            w = _safe_int(data.get("width", []), idx)
            h = _safe_int(data.get("height", []), idx)
            if w <= 0 or h <= 0:
                continue
            x0 = max(0, left)
            y0 = max(0, top)
            x1 = min(width, x0 + w)
            y1 = min(height, y0 + h)
            if x1 <= x0 or y1 <= y0:
                continue
            segments.append(DetectedSegment(bbox=(x0, y0, x1, y1), text=text, confidence=conf_value / 100.0))
        if segments:
            self.log(f"Tesseract fallback: найдено {len(segments)} слов для объединения")
        return segments

    def _merge_segments(self, segments: List[DetectedSegment], page_w: int, page_h: int) -> List[MergedRegion]:
        """Merge touching/adjacent boxes into unified regions."""
        if not segments:
            return []
        heights = [seg.bbox[3] - seg.bbox[1] for seg in segments]
        median_height = float(np.median(heights)) if heights else 0.0
        if median_height <= 0:
            median_height = max(8.0, page_h * 0.01)
        line_y_threshold = max(4.0, self.config.merge_line_y_ratio * median_height)
        x_gap_threshold = max(4.0, self.config.merge_x_gap_ratio * median_height)

        graph = self._build_graph(segments, line_y_threshold, x_gap_threshold)
        components = self._connected_components(graph, len(segments))

        merged: List[MergedRegion] = []
        for comp in components:
            comp_segments = [segments[i] for i in comp]
            x0 = min(seg.bbox[0] for seg in comp_segments)
            y0 = min(seg.bbox[1] for seg in comp_segments)
            x1 = max(seg.bbox[2] for seg in comp_segments)
            y1 = max(seg.bbox[3] for seg in comp_segments)
            text = self._merge_text(comp_segments, line_y_threshold)
            if not text.strip():
                continue
            confidence = float(np.median([seg.confidence for seg in comp_segments]))
            merged.append(
                MergedRegion(
                    bbox=(x0, y0, x1, y1),
                    text=text,
                    segments=comp_segments,
                    angle=0.0,
                    confidence=confidence,
                )
            )

        merged.sort(key=lambda r: ((r.bbox[1] + r.bbox[3]) / 2.0, r.bbox[0]))
        return merged

    def _build_graph(
        self, segments: List[DetectedSegment], line_y_threshold: float, x_gap_threshold: float
    ) -> List[Set[int]]:
        graph: List[Set[int]] = [set() for _ in range(len(segments))]
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if self._should_connect(segments[i], segments[j], line_y_threshold, x_gap_threshold):
                    graph[i].add(j)
                    graph[j].add(i)
        return graph

    def _should_connect(
        self, a: DetectedSegment, b: DetectedSegment, line_y_threshold: float, x_gap_threshold: float
    ) -> bool:
        ax0, ay0, ax1, ay1 = a.bbox
        bx0, by0, bx1, by1 = b.bbox
        a_cy = (ay0 + ay1) / 2.0
        b_cy = (by0 + by1) / 2.0
        if abs(a_cy - b_cy) > line_y_threshold:
            return False
        overlap = min(ax1, bx1) - max(ax0, bx0)
        if overlap >= 0:
            return True
        if not self.config.merge_allow_horizontal_gap:
            return False
        gap = bx0 - ax1 if ax1 < bx0 else ax0 - bx1
        return gap <= x_gap_threshold

    def _connected_components(self, graph: List[Set[int]], size: int) -> List[List[int]]:
        visited: Set[int] = set()
        components: List[List[int]] = []
        for node in range(size):
            if node in visited:
                continue
            stack = [node]
            component: List[int] = []
            visited.add(node)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return components

    def _merge_text(self, segments: List[DetectedSegment], line_y_threshold: float) -> str:
        if not segments:
            return ""
        sorted_segments = sorted(
            segments, key=lambda s: (((s.bbox[1] + s.bbox[3]) / 2.0), s.bbox[0])
        )
        lines: List[str] = []
        current_line: List[str] = []
        current_y: Optional[float] = None
        for seg in sorted_segments:
            cy = (seg.bbox[1] + seg.bbox[3]) / 2.0
            clean_text = str(seg.text or "").strip()
            if not clean_text:
                continue
            if current_y is None:
                current_y = cy
                current_line.append(clean_text)
            elif abs(cy - current_y) < line_y_threshold:
                current_line.append(clean_text)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [clean_text]
                current_y = cy
        if current_line:
            lines.append(" ".join(current_line))
        return "\n".join(lines).strip()

    def _filter_regions(self, regions: List[MergedRegion], page_w: int, page_h: int) -> List[MergedRegion]:
        filtered: List[MergedRegion] = []
        min_w = max(1, self.config.min_box_width)
        min_h = max(1, self.config.min_box_height)
        for region in regions:
            x0, y0, x1, y1 = region.bbox
            if x1 <= x0 or y1 <= y0:
                continue
            if (x1 - x0) < min_w or (y1 - y0) < min_h:
                continue
            region.bbox = (
                max(0, int(x0)),
                max(0, int(y0)),
                min(page_w, int(x1)),
                min(page_h, int(y1)),
            )
            if not region.text.strip():
                continue
            filtered.append(region)
        return filtered


class LayoutAnalyzer:
    """
    Runs OCR and converts pytesseract word-level data into logical TextBlocks.
    Grouping preserves block/paragraph/line structure and keeps pixel-based bboxes.
    """

    def __init__(self, config: PdfProcessingConfig, log: PdfLog = _noop_log) -> None:
        self.languages = config.ocr_languages
        self.min_confidence = config.min_confidence
        self.log = log

    def analyze(self, page: PageImage) -> List[TextBlock]:
        ocr_data = run_ocr(page.image, languages=self.languages)
        blocks = build_blocks_from_tesseract(
            ocr_data,
            page_index=page.page_index,
            min_confidence=self.min_confidence,
            page_width=page.width,
            page_height=page.height,
        )
        for block in blocks:
            block.block_type = self._classify_block(block)
        self.log(f"OCR: найдено блоков на странице {page.page_index + 1}: {len(blocks)}")
        return blocks

    def _classify_block(self, block: TextBlock) -> str:
        """Simple heuristic classification to support table/header/note styling."""
        x, y, w, h = block.bbox
        aspect = w / max(1, h)
        avg_height = 0
        if block.word_boxes:
            heights = [wb.bbox[3] for wb in block.word_boxes if wb.bbox[3] > 0]
            if heights:
                avg_height = sum(heights) / len(heights)
        lines = len(block.source_text.splitlines())
        text_len = len(block.source_text)
        if aspect > 5 and lines == 1 and avg_height > 0:
            return "header"
        if lines > 1 and aspect < 2:
            return "note"
        if aspect > 3 and avg_height and avg_height < 16:
            return "table_cell"
        if text_len < 20 and h < 30:
            return "label"
        return "paragraph"


class TextPreprocessor:
    """
    Cleans raw OCR text so that the translator receives stable, paragraph-preserving input.
    """

    def __init__(self, log: PdfLog = _noop_log) -> None:
        self.log = log

    def preprocess(self, blocks: Sequence[TextBlock]) -> List[TextBlock]:
        processed: List[TextBlock] = []
        for block in blocks:
            cleaned = self._clean_text(block.source_text)
            block.source_text = cleaned
            processed.append(block)
        return processed

    def _clean_text(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines()]
        merged: List[str] = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if not line:
                merged.append("")
                idx += 1
                continue
            if line.endswith("-") and idx + 1 < len(lines):
                # Merge hyphenated breaks: "TEMPERA-" + "TURE" -> "TEMPERATURE"
                next_line = lines[idx + 1].lstrip()
                merged.append((line[:-1] + next_line).strip())
                idx += 2
                continue
            merged.append(line)
            idx += 1
        # Collapse multiple blank lines to a single blank to preserve paragraphs.
        normalized: List[str] = []
        blank = False
        for line in merged:
            if not line:
                if not blank:
                    normalized.append("")
                blank = True
            else:
                normalized.append(" ".join(line.split()))
                blank = False
        return "\n".join(normalized).strip("\n")


class RegionAnalyzer:
    """
    Performs per-region analysis: dynamic expansion, OCR, color sampling, and mask extraction.
    """

    def __init__(self, config: PdfProcessingConfig, log: PdfLog = _noop_log) -> None:
        self.config = config
        self.log = log
        self.preprocessor = TextPreprocessor(log)

    def analyze(self, page: PageImage, merged_regions: Sequence[MergedRegion]) -> List[RegionInfo]:
        _require_dependency(cv2, "Установите opencv-python (pip install opencv-python)")
        regions: List[RegionInfo] = []
        if not merged_regions:
            return regions
        bgr_page = cv2.cvtColor(np.array(page.image.convert("RGB")), cv2.COLOR_RGB2BGR)
        for idx, merged in enumerate(merged_regions):
            x0 = max(0, min(page.width, int(merged.bbox[0])))
            y0 = max(0, min(page.height, int(merged.bbox[1])))
            x1 = max(x0, min(page.width, int(merged.bbox[2])))
            y1 = max(y0, min(page.height, int(merged.bbox[3])))
            bbox_wh: BBoxWH = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
            expanded_bbox = self._expand_bbox(bbox_wh, page.width, page.height)
            info = self._analyze_single(page, bgr_page, expanded_bbox, idx, merged)
            if info and info.source_text.strip():
                info.source_text = self.preprocessor._clean_text(info.source_text)
                # Skip only single-symbol noise.
                if len(info.source_text.strip()) <= 1:
                    continue
                regions.append(info)
        return regions

    def _expand_bbox(self, bbox: BBoxWH, page_w: int, page_h: int) -> BBoxWH:
        x, y, w, h = bbox
        margin_base = h if h > 0 else max(page_w, page_h) * 0.01
        mx = max(self.config.min_region_margin_px, int(self.config.region_margin_scale_x * margin_base))
        my = max(self.config.min_region_margin_px, int(self.config.region_margin_scale_y * margin_base))
        max_margin_x = int(page_w * self.config.max_region_margin_ratio)
        max_margin_y = int(page_h * self.config.max_region_margin_ratio)
        mx = min(mx, max_margin_x)
        my = min(my, max_margin_y)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(page_w, x + w + mx)
        y1 = min(page_h, y + h + my)
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _analyze_single(
        self,
        page: PageImage,
        bgr_page: "np.ndarray",
        bbox: BBoxWH,
        region_id: int,
        merged_region: MergedRegion,
    ) -> Optional[RegionInfo]:
        x, y, w, h = bbox
        x1 = x + w
        y1 = y + h
        roi = bgr_page[y:y1, x:x1].copy()
        if roi.size == 0:
            return None

        mask = build_text_mask(roi, dilation_kernel=self.config.dilation_kernel_size)
        bg_color = _estimate_background_color(roi, mask if mask is not None else np.zeros(roi.shape[:2], dtype=np.uint8))
        text_color = _estimate_text_color(roi, mask if mask is not None else np.ones(roi.shape[:2], dtype=np.uint8))

        font_height = self._estimate_font_size_from_segments(merged_region, h)

        return RegionInfo(
            page_index=page.page_index,
            region_id=region_id,
            bbox=bbox,
            text_orientation_deg=float(merged_region.angle),
            text_color=text_color,
            background_color=bg_color,
            source_text=merged_region.text,
            translated_text=None,
            mask=mask,
            font_size_estimate=font_height,
            source_font_size_estimate=font_height,
        )

    def _estimate_font_size_from_segments(self, merged_region: MergedRegion, fallback_height: int) -> int:
        heights = [seg.bbox[3] - seg.bbox[1] for seg in merged_region.segments if (seg.bbox[3] - seg.bbox[1]) > 0]
        if heights:
            heights.sort()
            median = heights[len(heights) // 2]
            return max(8, min(96, median))
        return max(8, min(96, int(fallback_height * 0.6)))


def run_ocr(image: "Image.Image", *, languages: str) -> Dict[str, List]:
    """Run pytesseract and return a parsed dictionary structure."""
    _require_dependency(pytesseract, "Установите pytesseract (pip install pytesseract)")
    _require_dependency(Output, "Модуль pytesseract.Output недоступен")
    try:
        data = pytesseract.image_to_data(image, lang=languages, output_type=Output.DICT)  # type: ignore[arg-type]
    except Exception as exc:
        raise RuntimeError(f"Ошибка OCR (pytesseract): {exc}") from exc
    return data


def build_blocks_from_tesseract(
    ocr_data: Dict[str, List],
    *,
    page_index: int,
    min_confidence: int,
    page_width: int,
    page_height: int,
) -> List[TextBlock]:
    """
    Build coherent TextBlocks from pytesseract output.

    Groups by (block_num, par_num), then by line_num, keeps reading order, and
    returns bounding boxes in image pixels as (x, y, w, h).
    """
    words_by_block: Dict[Tuple[int, int], Dict[int, List[WordBox]]] = {}
    length = len(ocr_data.get("text", []))
    for idx in range(length):
        level = _safe_int(ocr_data.get("level", []), idx)
        if level != 5:
            continue
        text = (ocr_data.get("text", [""])[idx] or "").strip()
        if not text:
            continue
        conf_value = _safe_float(ocr_data.get("conf", []), idx)
        if conf_value < min_confidence:
            continue
        left = _safe_int(ocr_data.get("left", []), idx)
        top = _safe_int(ocr_data.get("top", []), idx)
        width = _safe_int(ocr_data.get("width", []), idx)
        height = _safe_int(ocr_data.get("height", []), idx)
        if width <= 0 or height <= 0:
            continue
        x0 = max(0, left)
        y0 = max(0, top)
        x1 = min(page_width, x0 + width)
        y1 = min(page_height, y0 + height)
        if x1 <= x0 or y1 <= y0:
            continue
        bbox_wh: BBoxWH = (x0, y0, x1 - x0, y1 - y0)
        block_key = (_safe_int(ocr_data.get("block_num", []), idx), _safe_int(ocr_data.get("par_num", []), idx))
        line_num = _safe_int(ocr_data.get("line_num", []), idx)
        bucket_lines = words_by_block.setdefault(block_key, {})
        bucket_words = bucket_lines.setdefault(line_num, [])
        bucket_words.append(
            WordBox(
                bbox=bbox_wh,
                text=text,
                confidence=conf_value,
                line_num=line_num,
            )
        )

    blocks: List[TextBlock] = []
    block_id = 0
    for block_key in sorted(words_by_block.keys(), key=lambda k: (k[0], k[1])):
        lines = words_by_block[block_key]
        line_texts: List[str] = []
        all_word_boxes: List[WordBox] = []
        x0 = y0 = 10**9
        x1 = y1 = -10**9
        for line_num in sorted(lines.keys()):
            words = sorted(lines[line_num], key=lambda w: (w.bbox[0], w.bbox[1]))
            if not words:
                continue
            line_parts: List[str] = []
            prev_x_end = None
            for w in words:
                wx, wy, ww, wh = w.bbox
                if prev_x_end is not None and wx > prev_x_end:
                    line_parts.append(" ")
                line_parts.append(str(w.text))
                prev_x_end = wx + ww
                all_word_boxes.append(w)
                x0 = min(x0, wx)
                y0 = min(y0, wy)
                x1 = max(x1, wx + ww)
                y1 = max(y1, wy + wh)
            line_texts.append("".join(line_parts).strip())
        if not line_texts or x1 <= x0 or y1 <= y0:
            continue
        bbox_wh: BBoxWH = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        blocks.append(
            TextBlock(
                page_index=page_index,
                block_id=block_id,
                bbox=bbox_wh,
                source_text="\n".join(line_texts),
                word_boxes=all_word_boxes,
            )
        )
        block_id += 1
    return blocks
