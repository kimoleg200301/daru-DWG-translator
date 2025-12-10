from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re

from io import BytesIO
import numpy as np

from auto_translation import TranslationEngine, chunked

try:  # pragma: no cover - optional dependency guards
    from pdf2image import convert_from_path  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guards
    convert_from_path = None  # type: ignore

try:  # pragma: no cover - optional dependency guards
    import pytesseract  # type: ignore
    from pytesseract import Output  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guards
    pytesseract = None  # type: ignore
    Output = None  # type: ignore

try:  # pragma: no cover - optional dependency guards
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guards
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency guards
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guards
    easyocr = None  # type: ignore

try:  # pragma: no cover - optional dependency guards
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guards
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

try:  # pragma: no cover - optional dependency guards
    import boto3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guards
    boto3 = None  # type: ignore

PdfLog = Callable[[str], None]
BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)
BBoxWH = Tuple[int, int, int, int]  # (x, y, w, h)

PDF_TYPE_SCANNED = "scanned"
PDF_TYPE_NATIVE = "native"
TRANSLATABLE_BLOCK_TYPES: Set[str] = {"TEXT", "TITLE", "LIST", "CELL", "KEY", "VALUE"}
FIGURE_BLOCK_TYPES: Set[str] = {"LAYOUT_FIGURE", "FIGURE"}
# Use a dedicated cache folder next to this script instead of a generic logs/ path.
_SCRIPT_ROOT = Path(__file__).resolve().parent
_CACHE_DIR = _SCRIPT_ROOT / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEXTRACT_LOG_PATH = _CACHE_DIR / "textract_requests.log"


def _noop_log(_message: str) -> None:
    return


def _require_dependency(obj: object, message: str) -> None:
    if obj is None:
        raise RuntimeError(message)


def _log_textract_exchange(
    *,
    page_index: int,
    image_bytes: int,
    image_sha256: Optional[str],
    feature_types: Sequence[str],
    response: Dict[str, object],
    log: PdfLog,
    log_path: Path = TEXTRACT_LOG_PATH,
    file_hash: Optional[str] = None,
    input_name: Optional[str] = None,
    render_dpi: Optional[int] = None,
    cache_version: int = 2,
) -> None:
    """
    Append request/response metadata for Textract to a project-level log file.
    Response is stored as-is (JSON-serializable) without image bytes to keep size manageable.
    """
    entry = {
        "version": cache_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "page_index": page_index,
        "file_hash": file_hash,
        "input_name": input_name or "",
        "request": {
            "feature_types": list(feature_types),
            "image_bytes": image_bytes,
            "image_sha256": image_sha256,
            "render_dpi": render_dpi,
            "file_hash": file_hash,
            "input_name": input_name or "",
        },
        "response": {
            "block_count": len(response.get("Blocks", []) if isinstance(response, dict) else []),
            "data": response,
        },
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
    except Exception as exc:  # pragma: no cover - logging should not break pipeline
        log(f"Textract log: не удалось записать обмен для страницы {page_index + 1}: {exc}")


def _load_textract_cache(log_path: Path, log: PdfLog) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not log_path.exists():
        return entries
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if not clean:
                    continue
                try:
                    entry = json.loads(clean)
                except json.JSONDecodeError:
                    log("Textract log: некорректная JSON-строка, пропускаю")
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
    except Exception as exc:
        log(f"Textract log: не удалось прочитать {log_path}: {exc}")
    return entries


def _compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PdfProcessingConfig:
    """High-quality PDF processing knobs."""

    dpi: int = 400  # output DPI when writing PDF
    render_dpi: int = 400  # DPI for rasterizing PDF pages
    image_format: str = "png"  # format for intermediate page images
    ocr_languages: str = "eng"
    easyocr_languages: Optional[List[str]] = None
    easyocr_gpu: bool = False
    min_confidence: int = 60
    detector_min_confidence: float = 0.45
    min_box_width: int = 10
    min_box_height: int = 8
    merge_line_y_ratio: float = 0.6
    merge_x_gap_ratio: float = 1.0
    textract_split_lines: bool = True
    textract_vertical_aspect_ratio: float = 2.2
    textract_vertical_top_margin_ratio: float = 0.2 # добавляет больше рабочую область переведенного вертикального текста
    text_bold_fill_ratio: float = 0.45
    merge_allow_horizontal_gap: bool = False
    blur_kernel_size: int = 23
    dilation_kernel_size: int = 3
    region_margin_scale_x: float = 0.35
    region_margin_scale_y: float = 0.25
    min_region_margin_px: int = 3
    max_region_margin_ratio: float = 0.2
    textract_bbox_height_boost: float = 0.25 # добавляет больше рабочую область переведенного текста
    textract_bbox_width_boost: float = 0.1

    def normalized(self) -> "PdfProcessingConfig":
        blur = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size + 1
        blur = max(3, blur)
        dilation = max(1, self.dilation_kernel_size)
        render_dpi = max(150, self.render_dpi or self.dpi)
        output_dpi = max(72, self.dpi, render_dpi)
        image_format = (self.image_format or "png").lower()
        detector_conf = max(0.0, min(1.0, self.detector_min_confidence))
        min_box_w = max(1, self.min_box_width)
        min_box_h = max(1, self.min_box_height)
        languages_value: Optional[List[str]] = None
        if self.easyocr_languages:
            if isinstance(self.easyocr_languages, str):
                languages_value = [token for token in re.split(r"[+,/;\\s]+", self.easyocr_languages) if token]
            else:
                languages_value = [str(lang).strip() for lang in self.easyocr_languages if str(lang).strip()]
        return PdfProcessingConfig(
            dpi=output_dpi,
            render_dpi=render_dpi,
            image_format=image_format,
            ocr_languages=self.ocr_languages or "eng",
            min_confidence=max(0, min(100, self.min_confidence)),
            blur_kernel_size=blur,
            dilation_kernel_size=dilation,
            easyocr_languages=languages_value,
            easyocr_gpu=bool(self.easyocr_gpu),
            detector_min_confidence=detector_conf,
            min_box_width=min_box_w,
            min_box_height=min_box_h,
            merge_line_y_ratio=max(0.1, min(2.0, self.merge_line_y_ratio)),
            merge_x_gap_ratio=max(0.1, min(3.0, self.merge_x_gap_ratio)),
            textract_split_lines=bool(self.textract_split_lines),
            textract_vertical_aspect_ratio=max(1.0, min(10.0, self.textract_vertical_aspect_ratio)),
            textract_vertical_top_margin_ratio=max(0.0, min(0.5, self.textract_vertical_top_margin_ratio)),
            text_bold_fill_ratio=max(0.05, min(0.9, self.text_bold_fill_ratio)),
            merge_allow_horizontal_gap=bool(self.merge_allow_horizontal_gap),
            region_margin_scale_x=max(0.05, self.region_margin_scale_x),
            region_margin_scale_y=max(0.05, self.region_margin_scale_y),
            min_region_margin_px=max(0, self.min_region_margin_px),
            max_region_margin_ratio=max(0.05, min(0.9, self.max_region_margin_ratio)),
            textract_bbox_height_boost=max(0.0, min(0.5, self.textract_bbox_height_boost)),
            textract_bbox_width_boost=max(0.0, min(0.5, self.textract_bbox_width_boost)),
        )


@dataclass
class WordBox:
    """Single OCR word with its bounding box and confidence."""

    bbox: BBoxWH  # (x, y, w, h) in image pixels
    text: str
    confidence: float
    line_num: int


@dataclass
class PageImage:
    """Normalized page/image representation used across the pipeline."""

    page_index: int
    image: "Image.Image"
    width: int
    height: int
    dpi: int


@dataclass
class TextBlock:
    """Logical text block used end-to-end (OCR -> translation -> overlay)."""

    page_index: int
    block_id: int  # unique within page
    bbox: BBoxWH  # (x, y, w, h) in image pixels
    source_text: str
    block_type: str = "paragraph"  # e.g. paragraph, table_cell, header, note
    translated_text: Optional[str] = None
    word_boxes: List[WordBox] = field(default_factory=list)
    background_color: Optional[Tuple[int, int, int]] = None
    text_color: Optional[Tuple[int, int, int]] = None


@dataclass
class DetectedSegment:
    """Single detection hit returned by OCR engine (axis-aligned)."""

    bbox: BBox  # (x0, y0, x1, y1)
    text: str
    confidence: float


@dataclass
class MergedRegion:
    """Merged region built from multiple touching segments."""

    bbox: BBox  # (x0, y0, x1, y1)
    text: str
    segments: List[DetectedSegment] = field(default_factory=list)
    angle: float = 0.0
    confidence: float = 0.0


@dataclass
class RegionInfo:
    """Per-region analysis container used for translation and rendering."""

    page_index: int
    region_id: int
    bbox: BBoxWH  # expanded box in page coordinates
    text_orientation_deg: float
    text_color: Tuple[int, int, int]
    background_color: Tuple[int, int, int]
    source_text: str
    translated_text: Optional[str] = None
    mask: Optional["np.ndarray"] = None  # text mask in ROI coordinates
    font_size_estimate: int = 12
    is_vertical: bool = False
    is_bold: bool = False


@dataclass
class PdfProcessingResult:
    blurred_pdf_path: Path
    regions: List[RegionInfo]
    processed_images: List["Image.Image"]


@dataclass
class LineRegion:
    """Single Textract line mapped into image coordinates."""

    page_index: int
    line_id: str
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    source_text: str
    translated_text: Optional[str] = None


@dataclass
class BlockRegion:
    """Logical block coming from Textract (text, table cell, key/value, etc.)."""

    page_index: int
    block_id: str
    block_type: str
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    lines: List[LineRegion] = field(default_factory=list)
    cells: List["BlockRegion"] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    source_text: str = ""
    translated_text: Optional[str] = None
    render_id: Optional[int] = None


@dataclass
class TextractPageData:
    """Raw Textract structures and derived regions for a single page."""

    page: PageImage
    blocks_by_id: Dict[str, Dict[str, object]]
    lines: Dict[str, LineRegion]
    blocks: List[BlockRegion]


def _union_bbox(bboxes: Sequence[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not bboxes:
        return None
    x0 = min(box[0] for box in bboxes)
    y0 = min(box[1] for box in bboxes)
    x1 = max(box[2] for box in bboxes)
    y1 = max(box[3] for box in bboxes)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bbox_iou_wh(a: BBoxWH, b: BBoxWH) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = aw * ah
    area_b = bw * bh
    denom = float(area_a + area_b - inter_area)
    if denom <= 0:
        return 0.0
    return inter_area / denom


def _bbox_intersection_area_wh(a: BBoxWH, b: BBoxWH) -> int:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0
    return int((inter_x1 - inter_x0) * (inter_y1 - inter_y0))


# === Pipeline components ======================================================


class PageLoader:
    """
    Converts user-selected PDFs/images into normalized PageImage objects.

    - PDFs are rasterized at a high DPI.
    - Standalone images are loaded as RGB without scaling so pixel coordinates stay stable.
    """

    def __init__(self, config: PdfProcessingConfig, log: PdfLog = _noop_log) -> None:
        self.config = config
        self.log = log

    def load(self, input_path: Path) -> List[PageImage]:
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")
        suffix = input_path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(input_path)
        return [self._load_image(input_path)]

    def _load_pdf(self, pdf_path: Path) -> List[PageImage]:
        images = pdf_to_images(pdf_path, self.config.render_dpi, self.config.image_format, self.log)
        pages: List[PageImage] = []
        for idx, image in enumerate(images):
            width, height = image.size
            pages.append(PageImage(page_index=idx, image=image, width=width, height=height, dpi=self.config.render_dpi))
        return pages

    def _load_image(self, image_path: Path) -> PageImage:
        _require_dependency(Image, "Установите Pillow (pip install pillow)")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Не удалось открыть изображение {image_path}: {exc}") from exc
        width, height = image.size
        return PageImage(page_index=0, image=image, width=width, height=height, dpi=self.config.dpi)


class TextractLayoutExtractor:
    """Builds LineRegion/BlockRegion structures from AWS Textract layout API."""

    def __init__(
        self,
        textract_client: Optional[object],
        log: PdfLog = _noop_log,
        *,
        config: Optional[PdfProcessingConfig] = None,
        log_path: Path = TEXTRACT_LOG_PATH,
        file_hash: Optional[str] = None,
        input_name: str = "",
    ) -> None:
        self.client = textract_client or self._init_client()
        self.log = log
        self.config = (config or PdfProcessingConfig()).normalized()
        self.log_path = log_path
        self.file_hash = file_hash
        self.input_name = input_name
        self._cached_entries = self._load_cache()

    def _init_client(self) -> object:
        _require_dependency(boto3, "Установите boto3 (pip install boto3)")
        try:
            return boto3.client("textract")
        except Exception as exc:  # pragma: no cover - network/auth related
            raise RuntimeError(f"Textract: не удалось создать клиент: {exc}") from exc

    def _load_cache(self) -> List[Dict[str, object]]:
        cached_rows = _load_textract_cache(self.log_path, self.log)
        cache: List[Dict[str, object]] = []
        for entry in cached_rows:
            req = entry.get("request", {}) if isinstance(entry, dict) else {}
            resp = entry.get("response", {}) if isinstance(entry, dict) else {}
            data = resp.get("data") if isinstance(resp, dict) else None
            if not isinstance(data, dict):
                continue
            try:
                page_idx = int(entry.get("page_index"))
            except Exception:
                continue
            cache.append(
                {
                    "page_index": page_idx,
                    "file_hash": entry.get("file_hash") or req.get("file_hash"),
                    "image_bytes": req.get("image_bytes"),
                    "image_sha256": req.get("image_sha256"),
                    "response": data,
                }
            )
        if cache:
            self.log(f"Textract: загружено {len(cache)} ответов из JSON-лога {self.log_path}")
        return cache

    def _find_cached_response(self, page_index: int, meta: Dict[str, object]) -> Optional[Dict[str, object]]:
        candidates = [item for item in self._cached_entries if item.get("page_index") == page_index]
        if self.file_hash:
            hash_filtered = [item for item in candidates if item.get("file_hash") == self.file_hash]
            if hash_filtered:
                candidates = hash_filtered
        sha = meta.get("image_sha256")
        if sha:
            for item in reversed(candidates):
                if item.get("image_sha256") == sha:
                    return item.get("response")
        img_bytes = meta.get("image_bytes")
        if img_bytes is not None:
            for item in reversed(candidates):
                if item.get("image_bytes") == img_bytes:
                    return item.get("response")
        return None

    def analyze(self, page: PageImage) -> TextractPageData:
        """Run Textract analyze_document on the given page image."""
        img_bytes, img_meta = self._encode_image_for_textract(page.image, page.page_index, render_dpi=page.dpi)
        width, height = page.image.size
        feature_types = ["LAYOUT", "SIGNATURES"]
        cached = self._find_cached_response(page.page_index, img_meta)
        if cached is not None:
            resp = cached
            self.log(f"Textract: страница {page.page_index + 1} загружена из лога, запрос не отправлялся")
        else:
            try:
                resp = self.client.analyze_document(
                    Document={"Bytes": img_bytes},
                    FeatureTypes=feature_types,
                )
            except Exception as exc:  # pragma: no cover - Textract call can fail remotely
                raise RuntimeError(f"Textract: ошибка анализа страницы {page.page_index + 1}: {exc}") from exc

            _log_textract_exchange(
                page_index=page.page_index,
                image_bytes=img_meta.get("image_bytes", 0),
                image_sha256=img_meta.get("image_sha256"),
                feature_types=feature_types,
                response=resp,
                log=self.log,
                log_path=self.log_path,
                file_hash=self.file_hash,
                input_name=self.input_name,
                render_dpi=img_meta.get("render_dpi"),
            )
        blocks_by_id: Dict[str, Dict[str, object]] = {
            block["Id"]: block for block in resp.get("Blocks", []) if block.get("Id")
        }
        lines = self._build_lines(blocks_by_id, page_index=page.page_index, page_width=width, page_height=height)
        used_line_ids: Set[str] = set()
        blocks = self._build_blocks(
            blocks_by_id,
            lines,
            used_line_ids,
            page_index=page.page_index,
            page_width=width,
            page_height=height,
            page_image=page,
            allow_figures=True,
        )
        self.log(f"Textract: страница {page.page_index + 1} — строк: {len(lines)}, блоков: {len(blocks)}")
        return TextractPageData(page=page, blocks_by_id=blocks_by_id, lines=lines, blocks=blocks)

    def _encode_image_for_textract(
        self, image: "Image.Image", page_index: int, render_dpi: Optional[int] = None
    ) -> Tuple[bytes, Dict[str, object]]:
        """
        Encode page image to PNG, automatically downscaling if the payload exceeds Textract limits (~5 MB).
        Keeps original aspect ratio; coordinates are still mapped using the original page size.
        """
        _require_dependency(Image, "Установите Pillow (pip install pillow)")
        max_bytes = 4_500_000  # keep a buffer below the 5 MB limit
        attempt_image = image
        scale = 1.0
        for attempt in range(6):
            buf = BytesIO()
            attempt_image.save(buf, format="PNG")
            data = buf.getvalue()
            if len(data) <= max_bytes:
                if attempt > 0:
                    w, h = attempt_image.size
                    self.log(
                        f"Textract: страница {page_index + 1} уменьшена до {w}x{h} для ограничения 5 МБ "
                        f"({len(data) / 1024:.1f} КБ)"
                    )
                meta = {
                    "image_bytes": len(data),
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "render_dpi": render_dpi,
                }
                return data, meta
            scale *= 0.85
            new_w = max(1, int(image.width * scale))
            new_h = max(1, int(image.height * scale))
            if new_w < 100 or new_h < 100:
                break
            attempt_image = image.resize((new_w, new_h), Image.LANCZOS)

        # Fallback to JPEG compression if PNG still too large.
        for quality in (85, 70, 60):
            buf = BytesIO()
            attempt_image.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                w, h = attempt_image.size
                self.log(
                    f"Textract: страница {page_index + 1} сжата JPEG q={quality} до {w}x{h} "
                    f"({len(data) / 1024:.1f} КБ)"
                )
                meta = {
                    "image_bytes": len(data),
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "render_dpi": render_dpi,
                }
                return data, meta

        raise RuntimeError(
            f"Textract: не удалось уменьшить страницу {page_index + 1} ниже ограничения 5 МБ (последний размер "
            f"{len(data) / 1024:.1f} КБ)"
        )

    def _build_lines(
        self,
        blocks_by_id: Dict[str, Dict[str, object]],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> Dict[str, LineRegion]:
        lines: List[LineRegion] = []
        for block in blocks_by_id.values():
            if str(block.get("BlockType", "")).upper() != "LINE":
                continue
            text = str(block.get("Text", "") or "").strip()
            if not text:
                continue
            bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
            if not bbox:
                continue
            lines.append(LineRegion(page_index=page_index, line_id=str(block["Id"]), bbox=bbox, source_text=text))
        lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        return {line.line_id: line for line in lines}

    def _build_blocks(
        self,
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
        page_image: Optional[PageImage] = None,
        allow_figures: bool = True,
    ) -> List[BlockRegion]:
        blocks: List[BlockRegion] = []
        figure_blocks: List[Dict[str, object]] = []
        figure_line_ids: Set[str] = set()
        for block in blocks_by_id.values():
            block_type_raw = str(block.get("BlockType", "")).upper()
            if block_type_raw in FIGURE_BLOCK_TYPES and allow_figures and page_image is not None:
                figure_blocks.append(block)
                block_id = str(block.get("Id", ""))
                if block_id:
                    figure_line_ids.update(self._collect_line_ids_from_block(block_id, blocks_by_id, set()))
                continue
            if block_type_raw == "CELL":
                cell_region = self._build_cell_block(
                    block,
                    blocks_by_id,
                    lines_by_id,
                    used_line_ids,
                    page_index=page_index,
                    page_width=page_width,
                    page_height=page_height,
                )
                if cell_region:
                    blocks.append(cell_region)
                continue
            normalized = self._normalize_text_block_type(block_type_raw)
            if normalized:
                text_blocks = self._build_text_block(
                    block,
                    blocks_by_id,
                    lines_by_id,
                    used_line_ids,
                    figure_line_ids,
                    block_type=normalized,
                    page_index=page_index,
                    page_width=page_width,
                    page_height=page_height,
                )
                blocks.extend(text_blocks)

        for block in figure_blocks:
            figure_container, child_blocks = self._build_figure_block(
                block,
                blocks_by_id,
                lines_by_id,
                used_line_ids,
                page_index=page_index,
                page_width=page_width,
                page_height=page_height,
                page_image=page_image,
            )
            if figure_container:
                blocks.append(figure_container)
            blocks.extend(child_blocks)
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        return blocks

    def _collect_line_ids_from_block(
        self, block_id: str, blocks_by_id: Dict[str, Dict[str, object]], visited: Optional[Set[str]] = None
    ) -> Set[str]:
        visited = visited or set()
        if block_id in visited:
            return set()
        visited.add(block_id)
        target = blocks_by_id.get(block_id)
        if not target:
            return set()
        collected: Set[str] = set()
        for rel in target.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                child_block = blocks_by_id.get(child_id)
                child_type = str(child_block.get("BlockType", "")).upper() if child_block else ""
                if child_type == "LINE":
                    collected.add(str(child_id))
                else:
                    collected.update(self._collect_line_ids_from_block(str(child_id), blocks_by_id, visited))
        return collected

    def _build_text_block(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        figure_line_ids: Set[str],
        *,
        block_type: str,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> List[BlockRegion]:
        line_regions: List[LineRegion] = []
        has_line_children = False
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                if child_id in lines_by_id:
                    has_line_children = True
                if child_id in used_line_ids:
                    continue
                line = lines_by_id.get(child_id)
                if line:
                    line_regions.append(line)
        line_regions = self._dedup_and_sort_lines(line_regions)
        if any(ln.line_id in used_line_ids for ln in line_regions):
            return []
        if any(ln.line_id in figure_line_ids for ln in line_regions):
            return []

        blocks_out: List[BlockRegion] = []
        split_lines = bool(getattr(self.config, "textract_split_lines", False))
        split_by_gap = split_lines and self._should_split_lines_by_horizontal_gap(line_regions)
        if split_by_gap and line_regions:
            base_id = str(block.get("Id", ""))
            for idx, ln in enumerate(line_regions):
                used_line_ids.add(ln.line_id)
                blocks_out.append(
                    BlockRegion(
                        page_index=page_index,
                        block_id=f"{base_id}:{idx}" if base_id else f"line:{idx}",
                        block_type=block_type,
                        bbox=ln.bbox,
                        lines=[ln],
                        cells=[],
                        source_text=ln.source_text,
                    )
                )
            return blocks_out

        bbox = _union_bbox([ln.bbox for ln in line_regions])
        if bbox is None:
            bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
        source_text = "\n".join(line.source_text for line in line_regions).strip() if line_regions else ""
        if not source_text:
            source_text = str(block.get("Text", "") or "").strip()
        if has_line_children and not line_regions:
            return []
        if not source_text or not bbox:
            return []
        for ln in line_regions:
            used_line_ids.add(ln.line_id)
        blocks_out.append(
            BlockRegion(
                page_index=page_index,
                block_id=str(block.get("Id", "")),
                block_type=block_type,
                bbox=bbox if bbox else (line_regions[0].bbox if line_regions else (0, 0, 0, 0)),
                lines=line_regions,
                cells=[],
                source_text=source_text,
            )
        )
        return blocks_out

    def _should_split_lines_by_horizontal_gap(self, lines: Sequence[LineRegion]) -> bool:
        """Detect a significant horizontal gap between lines on the same row."""
        if len(lines) < 2:
            return False
        heights = [max(1, ln.bbox[3] - ln.bbox[1]) for ln in lines]
        median_h = float(np.median(heights)) if heights else 1.0
        gap_threshold = max(4.0, min(3.0, self.config.merge_x_gap_ratio) * median_h)
        ordered = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                if not self._lines_share_row(a, b, median_h):
                    continue
                gap = self._horizontal_gap(a, b)
                if gap > gap_threshold:
                    return True
        return False

    def _lines_share_row(self, a: LineRegion, b: LineRegion, median_h: float) -> bool:
        ay0, ay1 = a.bbox[1], a.bbox[3]
        by0, by1 = b.bbox[1], b.bbox[3]
        overlap = min(ay1, by1) - max(ay0, by0)
        min_h = float(min(max(1, ay1 - ay0), max(1, by1 - by0)))
        if overlap > 0 and (overlap / min_h) >= 0.5:
            return True
        cy_diff = abs(((ay0 + ay1) / 2.0) - ((by0 + by1) / 2.0))
        return cy_diff <= max(4.0, 0.4 * median_h)

    def _horizontal_gap(self, a: LineRegion, b: LineRegion) -> float:
        ax0, ax1 = a.bbox[0], a.bbox[2]
        bx0, bx1 = b.bbox[0], b.bbox[2]
        overlap = min(ax1, bx1) - max(ax0, bx0)
        if overlap >= 0:
            return 0.0
        return float(bx0 - ax1 if ax1 < bx0 else ax0 - bx1)

    def _build_cell_block(
        self,
        cell_block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> Optional[BlockRegion]:
        bbox = self._bbox_from_geometry(cell_block.get("Geometry"), page_width, page_height)
        if not bbox:
            return None
        text, lines = self._collect_text(cell_block, blocks_by_id, lines_by_id, used_line_ids)
        if any(ln.line_id in used_line_ids for ln in lines):
            return None
        for ln in lines:
            used_line_ids.add(ln.line_id)
        return BlockRegion(
            page_index=page_index,
            block_id=str(cell_block.get("Id", "")),
            block_type="CELL",
            bbox=bbox,
            lines=lines,
            cells=[],
            source_text=text.strip(),
        )

    def _build_figure_block(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
        page_image: PageImage,
    ) -> Tuple[Optional[BlockRegion], List[BlockRegion]]:
        bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
        child_ids: List[str] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                child_ids.append(str(child_id))

        line_ids = self._collect_line_ids_from_block(str(block.get("Id", "")), blocks_by_id, set())
        ordered_lines: List[LineRegion] = []
        for line_id in line_ids:
            line = lines_by_id.get(line_id)
            if line:
                ordered_lines.append(line)
        ordered_lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        for line in ordered_lines:
            used_line_ids.add(line.line_id)
        # Delegate figure text detection to a dedicated Textract pass on the cropped region.
        detected_blocks = self._analyze_figure_region(bbox, page_image, block_id=str(block.get("Id", "")))
        container: Optional[BlockRegion] = None
        if bbox:
            container = BlockRegion(
                page_index=page_index,
                block_id=str(block.get("Id", "")),
                block_type="FIGURE",
                bbox=bbox,
                lines=[],
                cells=[],
                child_ids=child_ids,
                source_text="",
            )
        return container, detected_blocks

    def _analyze_figure_region(self, bbox: Optional[Tuple[int, int, int, int]], page: PageImage, block_id: str) -> List[BlockRegion]:
        if not bbox:
            return []
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(page.width, x0))
        y0 = max(0, min(page.height, y0))
        x1 = max(x0, min(page.width, x1))
        y1 = max(y0, min(page.height, y1))
        if x1 <= x0 or y1 <= y0:
            return []

        _require_dependency(Image, "Установите Pillow (pip install pillow)")
        crop = page.image.crop((x0, y0, x1, y1)).convert("RGB")
        img_bytes, img_meta = self._encode_image_for_textract(crop, page.page_index, render_dpi=page.dpi)
        feature_types = ["LAYOUT", "SIGNATURES"]
        cached = self._find_cached_response(page.page_index, img_meta)
        if cached is not None:
            resp = cached
            self.log(f"Textract: фигура {block_id} на стр. {page.page_index + 1} загружена из лога")
        else:
            try:
                resp = self.client.analyze_document(
                    Document={"Bytes": img_bytes},
                    FeatureTypes=feature_types,
                )
            except Exception as exc:
                raise RuntimeError(f"Textract: ошибка анализа фигуры на странице {page.page_index + 1}: {exc}") from exc

            _log_textract_exchange(
                page_index=page.page_index,
                image_bytes=img_meta.get("image_bytes", 0),
                image_sha256=img_meta.get("image_sha256"),
                feature_types=feature_types,
                response=resp,
                log=self.log,
                log_path=self.log_path,
                file_hash=self.file_hash,
                input_name=f"{self.input_name}#figure:{block_id}",
                render_dpi=img_meta.get("render_dpi"),
            )

        blocks_by_id: Dict[str, Dict[str, object]] = {
            blk["Id"]: blk for blk in resp.get("Blocks", []) if blk.get("Id")
        }
        lines = self._build_lines(
            blocks_by_id,
            page_index=page.page_index,
            page_width=crop.width,
            page_height=crop.height,
        )

        # Shift sub-block coordinates back to original page space.
        shifted_blocks: List[BlockRegion] = []
        for line in lines.values():
            lx0, ly0, lx1, ly1 = line.bbox
            shifted_bbox = (lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0)
            shifted_line = LineRegion(
                page_index=line.page_index,
                line_id=line.line_id,
                bbox=shifted_bbox,
                source_text=line.source_text,
                translated_text=line.translated_text,
            )
            shifted_blocks.append(
                BlockRegion(
                    page_index=page.page_index,
                    block_id=f"{block_id}-LINE-{line.line_id}",
                    block_type="TEXT",
                    bbox=shifted_bbox,
                    lines=[shifted_line],
                    cells=[],
                    source_text=line.source_text,
                )
            )
        return shifted_blocks

    def _collect_text(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Optional[Set[str]] = None,
    ) -> Tuple[str, List[LineRegion]]:
        line_regions: List[LineRegion] = []
        words: List[str] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                if used_line_ids is not None and child_id in used_line_ids:
                    continue
                child = blocks_by_id.get(child_id)
                if not child:
                    continue
                child_type = str(child.get("BlockType", "")).upper()
                if child_type == "LINE":
                    line = lines_by_id.get(child_id)
                    if line:
                        line_regions.append(line)
                elif child_type == "WORD":
                    word_text = str(child.get("Text", "") or "").strip()
                    if word_text:
                        words.append(word_text)
        line_regions = self._dedup_and_sort_lines(line_regions)
        if line_regions:
            return "\n".join(line.source_text for line in line_regions), line_regions
        return " ".join(words).strip(), []

    def _normalize_text_block_type(self, block_type_raw: str) -> Optional[str]:
        if block_type_raw in FIGURE_BLOCK_TYPES:
            return None
        if block_type_raw in {"LAYOUT_TEXT", "PARAGRAPH", "TEXT"}:
            return "TEXT"
        if block_type_raw in {"LAYOUT_TITLE", "TITLE"}:
            return "TITLE"
        if block_type_raw in {"LAYOUT_LIST", "LIST"}:
            return "LIST"
        if block_type_raw in {"LAYOUT_TABLE", "TABLE", "LAYOUT_CELL", "CELL"}:
            return "TEXT"
        if block_type_raw in {"KEY_VALUE_SET", "KEY", "VALUE", "LAYOUT_KEY_VALUE_SET"}:
            return "TEXT"
        if block_type_raw.startswith("LAYOUT_"):
            return "TEXT"
        return None

    def _bbox_from_geometry(
        self, geometry: Optional[Dict[str, object]], page_width: int, page_height: int
    ) -> Optional[Tuple[int, int, int, int]]:
        if not geometry:
            return None
        bbox = geometry.get("BoundingBox") if isinstance(geometry, dict) else None
        if not bbox:
            return None
        try:
            left = float(bbox.get("Left", 0.0))
            top = float(bbox.get("Top", 0.0))
            width = float(bbox.get("Width", 0.0))
            height = float(bbox.get("Height", 0.0))
        except Exception:
            return None
        x0 = int(left * page_width)
        y0 = int(top * page_height)
        x1 = int((left + width) * page_width)
        y1 = int((top + height) * page_height)
        x0 = max(0, min(page_width, x0))
        y0 = max(0, min(page_height, y0))
        x1 = max(x0, min(page_width, x1))
        y1 = max(y0, min(page_height, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _dedup_and_sort_lines(self, lines: Sequence[LineRegion]) -> List[LineRegion]:
        seen: Dict[str, LineRegion] = {}
        for line in lines:
            seen[line.line_id] = line
        sorted_lines = list(seen.values())
        sorted_lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        return sorted_lines


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
        )

    def _estimate_font_size_from_segments(self, merged_region: MergedRegion, fallback_height: int) -> int:
        heights = [seg.bbox[3] - seg.bbox[1] for seg in merged_region.segments if (seg.bbox[3] - seg.bbox[1]) > 0]
        if heights:
            heights.sort()
            median = heights[len(heights) // 2]
            return max(8, min(96, median))
        return max(8, min(96, int(fallback_height * 0.6)))


def _canonicalize_text_for_dedup(text: str) -> str:
    """Normalize text to detect duplicates regardless of whitespace or case."""
    collapsed = re.sub(r"\s+", " ", str(text or ""))
    return collapsed.strip().lower()


def _collect_canonical_cache(blocks: Sequence[BlockRegion]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    for block in blocks:
        existing_value = str(block.translated_text or "").strip()
        if not existing_value:
            continue
        canonical = _canonicalize_text_for_dedup(block.source_text)
        if canonical:
            cache.setdefault(canonical, existing_value)
    return cache


def _has_pending_translation(blocks: Sequence[BlockRegion], canonical_cache: Dict[str, str]) -> bool:
    for block in blocks:
        if block.block_type not in TRANSLATABLE_BLOCK_TYPES:
            continue
        source_text = str(block.source_text or "").strip()
        if not source_text:
            continue
        canonical = _canonicalize_text_for_dedup(source_text)
        if canonical and canonical not in canonical_cache:
            return True
    return False


def _apply_canonical_translations(blocks: Sequence[BlockRegion], canonical_cache: Dict[str, str]) -> None:
    for block in blocks:
        canonical = _canonicalize_text_for_dedup(block.source_text)
        translated_value = canonical_cache.get(canonical, "").strip()
        if translated_value:
            block.translated_text = translated_value
        elif not block.translated_text:
            block.translated_text = block.source_text


class RegionTranslator:
    """Translate RegionInfo objects in batches using the existing TranslationEngine."""

    def __init__(self, engine: TranslationEngine, log: PdfLog = _noop_log, batch_size: int = 10) -> None:
        self.engine = engine
        self.batch_size = max(1, batch_size)
        self.log = log

    def translate(self, regions: Sequence[RegionInfo]) -> None:
        texts = [region.source_text for region in regions]
        if not texts:
            return
        translated: List[str] = []
        total = len(texts)
        processed = 0
        for chunk in chunked(texts, self.batch_size):
            chunk_translated = self.engine.translate_many(list(chunk))
            translated.extend(chunk_translated)
            processed = min(total, processed + len(chunk_translated))
            percent = int((processed / total) * 100)
            self.log(f"Перевод областей... [{percent}%]")
        for region, translated_text in zip(regions, translated):
            region.translated_text = translated_text or region.source_text


class BlockRegionTranslator:
    """Translate BlockRegion objects as independent units."""

    def __init__(self, engine: TranslationEngine, log: PdfLog = _noop_log, batch_size: int = 10) -> None:
        self.engine = engine
        self.batch_size = max(1, batch_size)
        self.log = log

    def translate(
        self, blocks: Sequence[BlockRegion], canonical_cache: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Translate unique block texts once (based on canonical form) and reuse results.
        Returns the canonical cache used for translation.
        """
        targets = [block for block in blocks if block.block_type in TRANSLATABLE_BLOCK_TYPES]
        targets = [block for block in targets if str(block.source_text or "").strip()]
        cache: Dict[str, str] = dict(canonical_cache or {})
        if not targets:
            return cache

        # Pre-seed cache with translations already present on blocks.
        for block in targets:
            existing_value = str(block.translated_text or "").strip()
            if not existing_value:
                continue
            canonical = _canonicalize_text_for_dedup(block.source_text)
            if canonical:
                cache.setdefault(canonical, existing_value)

        queued: List[Tuple[str, str]] = []
        queued_keys: Set[str] = set()
        for block in targets:
            source_text = str(block.source_text or "")
            canonical = _canonicalize_text_for_dedup(source_text)
            if not canonical or not source_text.strip():
                continue
            if canonical in cache or canonical in queued_keys:
                continue
            queued.append((canonical, source_text))
            queued_keys.add(canonical)

        if queued:
            canonical_order = [item[0] for item in queued]
            source_order = [item[1] for item in queued]
            translated: List[str] = []
            total = len(source_order)
            processed = 0
            for chunk in chunked(source_order, self.batch_size):
                chunk_translated = self.engine.translate_many(list(chunk))
                translated.extend(chunk_translated)
                processed = min(total, processed + len(chunk_translated))
                percent = int((processed / total) * 100)
                self.log(f"Перевод блоков... [{percent}%]")
            for canonical_key, source_text, translated_text in zip(canonical_order, source_order, translated):
                cache[canonical_key] = translated_text or source_text

        for block in targets:
            canonical = _canonicalize_text_for_dedup(block.source_text)
            translated_value = cache.get(canonical, "").strip()
            if translated_value:
                block.translated_text = translated_value
            else:
                block.translated_text = block.source_text
        return cache


class Renderer:
    """
    Removes original glyphs and draws translated text back into the same regions.
    """

    def __init__(self, config: PdfProcessingConfig, log: PdfLog = _noop_log, style_font: Optional[str] = None) -> None:
        self.config = config
        self.log = log
        self.font_loader = _FontLoader(style_font)

    def render_page(self, page: PageImage, regions: Sequence[RegionInfo]) -> "Image.Image":
        """Remove text per region and render translations back into place."""
        _require_dependency(cv2, "Установите opencv-python (pip install opencv-python)")
        base_rgb = page.image.convert("RGB")
        bgr = cv2.cvtColor(np.array(base_rgb), cv2.COLOR_RGB2BGR)

        # First pass: remove source text region-by-region using masks/background color.
        for region in regions:
            x, y, w, h = region.bbox
            x1, y1 = x + w, y + h
            roi = bgr[y:y1, x:x1]
            if roi.size == 0:
                continue
            mask = region.mask
            if mask is None:
                mask = build_text_mask(roi, dilation_kernel=self.config.dilation_kernel_size)
            if mask is None:
                continue
            # Defensive: ensure mask matches ROI size and type.
            if mask.shape[:2] != roi.shape[:2]:
                mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if mask.shape[:2] != roi.shape[:2]:
                continue
            bg_color = region.background_color
            fill = np.full_like(roi, bg_color, dtype=np.uint8)
            text_region = cv2.bitwise_and(fill, fill, mask=mask)
            background_region = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            cleaned = cv2.add(background_region, text_region)
            cleaned = cv2.GaussianBlur(
                cleaned,
                (max(3, self.config.blur_kernel_size | 1), max(3, self.config.blur_kernel_size | 1)),
                0,
            )
            bgr[y:y1, x:x1] = cleaned

        cleaned_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", cleaned_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for region in regions:
            self._draw_region(draw, region, overlay, cleaned_pil)

        return Image.alpha_composite(cleaned_pil, overlay).convert("RGB")

    def _draw_region(
        self,
        draw: "ImageDraw.ImageDraw",
        region: RegionInfo,
        overlay: "Image.Image",
        base_image: "Image.Image",
    ) -> None:
        x0, y0, w, h = region.bbox
        text = str(region.translated_text or region.source_text or "").strip()
        if not text:
            return
        padding = max(1, int(min(w, h) * 0.06))
        max_text_width = max(1, w - 2 * padding)
        max_text_height = max(1, h - 2 * padding)

        start_size = max(8, min(96, region.font_size_estimate))
        chosen_font: Optional["ImageFont.FreeTypeFont"] = None
        wrapped_lines: List[str] = []
        line_spacing = 0
        for size in range(start_size, 7, -1):
            font = self.font_loader.get(size)
            lines = _wrap_text(text, draw, font, max_text_width)
            if not lines:
                continue
            line_height = _measure_text("Ay", draw, font)[1]
            spacing = max(1, int(line_height * 0.25))
            total_height = len(lines) * line_height + max(0, len(lines) - 1) * spacing
            max_width = max(_measure_text(line, draw, font)[0] for line in lines)
            if total_height <= max_text_height and max_width <= max_text_width:
                chosen_font = font
                wrapped_lines = lines
                line_spacing = spacing
                break
        if chosen_font is None:
            chosen_font = self.font_loader.get(8)
            wrapped_lines = _wrap_text(text, draw, chosen_font, max_text_width)
            line_spacing = max(1, int(_measure_text("Ay", draw, chosen_font)[1] * 0.25))

        if region.text_color:
            b, g, r = region.text_color
            text_color = (r, g, b, 255)
        else:
            text_color = _sample_text_color(base_image, (x0, y0, x0 + w, y0 + h))
        stroke_w = 1 if region.is_bold else 0
        stroke_fill = text_color

        if region.is_vertical:
            # Pick font size starting from the estimated size of the source block; shrink only if не помещается.
            layer_w, layer_h = h, w  # swapped canvas before rotation
            available_w = max(1, layer_w - 2 * padding)
            available_h = max(1, layer_h - 2 * padding)
            target_size = max(8, min(96, region.font_size_estimate))
            chosen_font = None
            char_lines: List[str] = []
            for paragraph in wrapped_lines:
                if not paragraph:
                    char_lines.append("")
                    continue
                # reversed so that after -90 rotation чтение идёт сверху вниз
                char_lines.extend(list(reversed(paragraph)))
                char_lines.append("")
            for size in range(target_size, 7, -1):
                font = self.font_loader.get(size)
                line_height = _measure_text("Ay", draw, font)[1]
                spacing = max(1, int(line_height * 0.25))
                max_width_needed = 0
                width_acc = 0
                total_h = 0
                for ch in char_lines:
                    cw, ch_h = _measure_text(ch or " ", draw, font)
                    if ch == "":
                        max_width_needed = max(max_width_needed, width_acc)
                        width_acc = 0
                        total_h += ch_h
                        continue
                    width_acc += cw
                    max_width_needed = max(max_width_needed, width_acc)
                    width_acc += spacing
                    total_h += ch_h
                max_width_needed = max(max_width_needed, width_acc)
                total_h += max(0, len(char_lines) - 1) * spacing
                if max_width_needed <= available_w and total_h <= available_h:
                    chosen_font = font
                    line_spacing = spacing
                    break
            if chosen_font is None:
                chosen_font = self.font_loader.get(8)
                line_spacing = max(1, int(_measure_text("Ay", draw, chosen_font)[1] * 0.25))

            # Draw along X (right->left) on swapped canvas; after -90 rotation it flows top->bottom.
            layer_w, layer_h = h, w
            region_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
            region_draw = ImageDraw.Draw(region_layer)
            current_x = layer_w - padding
            baseline_y = padding
            for paragraph in wrapped_lines:
                if not paragraph:
                    current_x -= line_spacing
                    continue
                for ch in reversed(paragraph):
                    char_w, char_h = _measure_text(ch, region_draw, chosen_font)
                    if current_x - char_w < padding:
                        break
                    region_draw.text(
                        (current_x - char_w, baseline_y),
                        ch,
                        fill=text_color,
                        font=chosen_font,
                        stroke_width=stroke_w,
                        stroke_fill=stroke_fill,
                    )
                    current_x -= char_w + line_spacing
                current_x -= line_spacing  # gap between words/paragraphs
            angle = region.text_orientation_deg or -90.0
            rotated = region_layer.rotate(angle, expand=True, resample=Image.BICUBIC)
            overlay.alpha_composite(rotated, dest=(x0, y0))
        else:
            region_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            region_draw = ImageDraw.Draw(region_layer)
            current_y = padding
            for line in wrapped_lines:
                line_width, line_height = _measure_text(line, region_draw, chosen_font)
                region_draw.text(
                    (padding, current_y),
                    line,
                    fill=text_color,
                    font=chosen_font,
                    stroke_width=stroke_w,
                    stroke_fill=stroke_fill,
                )
                current_y += line_height + line_spacing

            angle = region.text_orientation_deg or 0.0
            if abs(angle) > 1e-1:
                rotated = region_layer.rotate(angle, expand=False, resample=Image.BICUBIC)
                overlay.alpha_composite(rotated, dest=(x0, y0))
            else:
                overlay.alpha_composite(region_layer, dest=(x0, y0))


class Exporter:
    """Exports processed page images back to a multi-page PDF (or image sequence)."""

    def __init__(self, log: PdfLog = _noop_log) -> None:
        self.log = log

    def to_pdf(self, images: Sequence["Image.Image"], output_path: Path, *, dpi: int) -> None:
        images_to_pdf(images, output_path, dpi=dpi, log=self.log)


def _iter_translatable_blocks(blocks: Sequence[BlockRegion]) -> Iterable[BlockRegion]:
    for block in blocks:
        if block.block_type in TRANSLATABLE_BLOCK_TYPES:
            yield block
        if block.cells:
            yield from _iter_translatable_blocks(block.cells)


def _assign_render_ids_for_page(blocks: Sequence[BlockRegion]) -> None:
    ordered = sorted(_iter_translatable_blocks(blocks), key=lambda b: (b.bbox[1], b.bbox[0]))
    for idx, block in enumerate(ordered):
        block.render_id = idx

def _apply_cached_translations(blocks: Sequence[BlockRegion], existing: Dict[Tuple[int, int], str]) -> None:
    for block in _iter_translatable_blocks(blocks):
        if block.render_id is None:
            continue
        cached_value = str(existing.get((block.page_index, block.render_id), "") or "").strip()
        if cached_value:
            block.translated_text = cached_value

def _apply_canonical_map(blocks: Sequence[BlockRegion], canonical_map: Dict[str, str]) -> None:
    """Apply translations by canonicalized source text (ignores render_id mismatch)."""
    if not canonical_map:
        return
    for block in _iter_translatable_blocks(blocks):
        if block.translated_text:
            continue
        canonical = _canonicalize_text_for_dedup(block.source_text)
        translated = canonical_map.get(canonical, "").strip()
        if translated:
            block.translated_text = translated


def _estimate_font_size_from_block(block: BlockRegion) -> int:
    heights = [line.bbox[3] - line.bbox[1] for line in block.lines if (line.bbox[3] - line.bbox[1]) > 0]
    if heights:
        heights.sort()
        median = heights[len(heights) // 2]
        return max(8, min(96, median))
    block_height = max(1, block.bbox[3] - block.bbox[1])
    return max(8, min(96, int(block_height * 0.6)))


def _clamp_bbox_to_page(
    bbox: Tuple[int, int, int, int], page_width: int, page_height: int
) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(page_width, int(x0)))
    y0 = max(0, min(page_height, int(y0)))
    x1 = max(x0, min(page_width, int(x1)))
    y1 = max(y0, min(page_height, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _block_to_region_info(
    block: BlockRegion,
    bgr_page: "np.ndarray",
    config: PdfProcessingConfig,
    page_width: int,
    page_height: int,
) -> Optional[RegionInfo]:
    clamped = _clamp_bbox_to_page(block.bbox, page_width, page_height)
    if not clamped:
        return None
    x0, y0, x1, y1 = clamped
    # Add a small vertical margin so descenders are not clipped after rendering.
    height_boost = max(0.0, min(0.5, config.textract_bbox_height_boost))
    if height_boost > 0:
        extra = int((y1 - y0) * height_boost)
        if extra > 0:
            grow_down = min(extra, page_height - y1)
            grow_up = extra - grow_down
            y1 = min(page_height, y1 + grow_down)
            y0 = max(0, y0 - grow_up)
    # Add a small horizontal margin to avoid clipping long translated words.
    width_boost = max(0.0, min(0.5, config.textract_bbox_width_boost))
    if width_boost > 0:
        extra = int((x1 - x0) * width_boost)
        if extra > 0:
            grow_right = min(extra, page_width - x1)
            grow_left = extra - grow_right
            x1 = min(page_width, x1 + grow_right)
            x0 = max(0, x0 - grow_left)
    roi = bgr_page[y0:y1, x0:x1].copy()
    if roi.size == 0:
        return None
    mask = build_text_mask(roi, dilation_kernel=config.dilation_kernel_size)
    if mask is None:
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    bg_color = _estimate_background_color(roi, mask)
    raw_text_color = _estimate_text_color(roi, mask if mask is not None else np.ones(roi.shape[:2], dtype=np.uint8))
    is_vertical = False
    is_bold = False
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    aspect = height / float(width)
    if aspect >= config.textract_vertical_aspect_ratio:
        is_vertical = True
        top_extra = int(height * max(0.0, min(0.5, config.textract_vertical_top_margin_ratio)))
        if top_extra > 0:
            y0 = max(0, y0 - top_extra)
    if mask is not None and mask.size >= 400:
        fill_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if fill_ratio >= max(0.05, min(0.9, config.text_bold_fill_ratio)):
            is_bold = True
    boosted_text_color = _boost_text_contrast(raw_text_color, bg_color)
    return RegionInfo(
        page_index=block.page_index,
        region_id=int(block.render_id if block.render_id is not None else 0),
        bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        text_orientation_deg=-90.0 if is_vertical else 0.0,
        text_color=boosted_text_color,
        background_color=bg_color,
        source_text=block.source_text,
        translated_text=block.translated_text or block.source_text,
        mask=mask,
        font_size_estimate=_estimate_font_size_from_block(block),
        is_vertical=is_vertical,
        is_bold=is_bold,
    )


def _build_regions_from_blocks(
    page: PageImage, blocks: Sequence[BlockRegion], config: PdfProcessingConfig, log: PdfLog = _noop_log
) -> List[RegionInfo]:
    _require_dependency(cv2, "Установите opencv-python (pip install opencv-python)")
    bgr_page = cv2.cvtColor(np.array(page.image.convert("RGB")), cv2.COLOR_RGB2BGR)
    regions: List[RegionInfo] = []
    seen_render_ids: Set[int] = set()
    seen_by_text: Dict[str, List[RegionInfo]] = {}
    for block in _iter_translatable_blocks(blocks):
        if block.render_id is None:
            continue
        if block.render_id in seen_render_ids:
            # Skip duplicate blocks that would draw text multiple times over the same place.
            log(f"Textract: повторяющийся render_id {block.render_id} на странице {page.page_index + 1}, пропускаю дубль")
            continue
        seen_render_ids.add(block.render_id)
        text_value = block.translated_text or block.source_text
        if not str(text_value or "").strip():
            continue
        region = _block_to_region_info(block, bgr_page, config, page.width, page.height)
        if region:
            canonical = _canonicalize_text_for_dedup(region.translated_text or region.source_text)
            overlaps = False
            for prev in seen_by_text.get(canonical, []):
                iou = _bbox_iou_wh(region.bbox, prev.bbox)
                inter_area = _bbox_intersection_area_wh(region.bbox, prev.bbox)
                area_new = region.bbox[2] * region.bbox[3]
                area_prev = prev.bbox[2] * prev.bbox[3]
                min_area = min(area_new, area_prev)
                if iou >= 0.6 or (min_area > 0 and inter_area / min_area >= 0.8):
                    overlaps = True
                    break
            if overlaps:
                log(
                    f"Render: пропускаю дубль текста на стр. {page.page_index + 1} блок #{block.render_id} "
                    f"({canonical[:40]}...)"
                )
                continue
            preview = str(region.translated_text or region.source_text or "")
            log(
                f"Render: стр. {page.page_index + 1} блок #{block.render_id} "
                f"TextractId={block.block_id} bbox={region.bbox} \"{preview[:60]}\""
            )
            regions.append(region)
            seen_by_text.setdefault(canonical, []).append(region)
    regions.sort(key=lambda r: r.region_id)
    return regions


def process_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    *,
    config: Optional[PdfProcessingConfig] = None,
    log: Optional[PdfLog] = None,
    write_blurred_pdf: bool = True,
) -> PdfProcessingResult:
    """
    Convert a PDF into page images, detect text blocks with Textract, remove the original
    glyphs, and (optionally) save the blurred/clean PDF. This is the "analysis only"
    path — translation happens in translate_pdf().

    Raises descriptive exceptions on dependency or parsing issues.
    """
    logger = log or _noop_log
    cfg = (config or PdfProcessingConfig()).normalized()
    input_path = Path(input_pdf_path)
    output_path = Path(output_pdf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"PDF не найден: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError("Поддерживаются только файлы PDF")

    file_hash = _compute_file_sha256(input_path)
    loader = PageLoader(cfg, logger)
    renderer = Renderer(cfg, logger)
    textract = TextractLayoutExtractor(
        None,
        logger,
        config=cfg,
        log_path=TEXTRACT_LOG_PATH,
        file_hash=file_hash,
        input_name=str(input_path),
    )

    pages = loader.load(input_path)
    processed_images: List["Image.Image"] = []
    all_regions: List[RegionInfo] = []

    for page in pages:
        logger(f"PDF: обработка страницы {page.page_index + 1}/{len(pages)}")
        page_layout = textract.analyze(page)
        _assign_render_ids_for_page(page_layout.blocks)
        for block in _iter_translatable_blocks(page_layout.blocks):
            block.translated_text = block.source_text
        regions = _build_regions_from_blocks(page, page_layout.blocks, cfg, log=logger)
        all_regions.extend(regions)
        processed_image = renderer.render_page(page, regions)
        processed_images.append(processed_image)

    if write_blurred_pdf:
        Exporter(logger).to_pdf(processed_images, output_path, dpi=cfg.dpi)
        logger(f"PDF: файл сохранён: {output_path}")

    return PdfProcessingResult(
        blurred_pdf_path=output_path,
        regions=all_regions,
        processed_images=processed_images,
    )


def translate_pdf(
    *,
    input_path: Path,
    output_path: Path,
    translator_name: str,
    source_lang: str,
    target_lang: str,
    log: PdfLog,
    pdf_type: str = PDF_TYPE_SCANNED,
    layer_json_path: Optional[Path] = None,
    processing_options: Optional[Dict[str, object]] = None,
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
    """
    High-quality PDF translation pipeline driven purely by Textract layout blocks.
    """
    _ = pdf_type  # retained for UI compatibility
    config = PdfProcessingConfig().normalized()
    if processing_options:
        config = PdfProcessingConfig(
            dpi=processing_options.get("dpi", config.dpi),
            render_dpi=processing_options.get("render_dpi", config.render_dpi),
            image_format=processing_options.get("image_format", config.image_format),
            ocr_languages=processing_options.get("ocr_languages", config.ocr_languages),
            easyocr_languages=processing_options.get("easyocr_languages", config.easyocr_languages),
            easyocr_gpu=processing_options.get("easyocr_gpu", config.easyocr_gpu),
            min_confidence=processing_options.get("min_confidence", config.min_confidence),
            detector_min_confidence=processing_options.get("detector_min_confidence", config.detector_min_confidence),
            min_box_width=processing_options.get("min_box_width", config.min_box_width),
            min_box_height=processing_options.get("min_box_height", config.min_box_height),
            merge_line_y_ratio=processing_options.get("merge_line_y_ratio", config.merge_line_y_ratio),
            merge_x_gap_ratio=processing_options.get("merge_x_gap_ratio", config.merge_x_gap_ratio),
            textract_split_lines=processing_options.get("textract_split_lines", config.textract_split_lines),
            textract_vertical_aspect_ratio=processing_options.get(
                "textract_vertical_aspect_ratio", config.textract_vertical_aspect_ratio
            ),
            textract_vertical_top_margin_ratio=processing_options.get(
                "textract_vertical_top_margin_ratio", config.textract_vertical_top_margin_ratio
            ),
            text_bold_fill_ratio=processing_options.get("text_bold_fill_ratio", config.text_bold_fill_ratio),
            merge_allow_horizontal_gap=processing_options.get(
                "merge_allow_horizontal_gap", config.merge_allow_horizontal_gap
            ),
            blur_kernel_size=processing_options.get("blur_kernel_size", config.blur_kernel_size),
            dilation_kernel_size=processing_options.get("dilation_kernel_size", config.dilation_kernel_size),
            region_margin_scale_x=processing_options.get("region_margin_scale_x", config.region_margin_scale_x),
            region_margin_scale_y=processing_options.get("region_margin_scale_y", config.region_margin_scale_y),
            min_region_margin_px=processing_options.get("min_region_margin_px", config.min_region_margin_px),
            max_region_margin_ratio=processing_options.get("max_region_margin_ratio", config.max_region_margin_ratio),
        ).normalized()

    if layer_json_path is None:
        layer_json_path = _CACHE_DIR / f"{input_path.stem}.translation.json"

    file_hash = _compute_file_sha256(input_path)
    loader = PageLoader(config, log)
    renderer = Renderer(config, log, style_font=style_font)
    exporter = Exporter(log)
    textract = TextractLayoutExtractor(
        None,
        log,
        config=config,
        log_path=TEXTRACT_LOG_PATH,
        file_hash=file_hash,
        input_name=str(input_path),
    )

    pages = loader.load(input_path)
    page_layouts: List[TextractPageData] = []
    for page in pages:
        layout = textract.analyze(page)
        _assign_render_ids_for_page(layout.blocks)
        page_layouts.append(layout)

    all_blocks: List[BlockRegion] = []
    for layout in page_layouts:
        all_blocks.extend(list(_iter_translatable_blocks(layout.blocks)))

    if not all_blocks:
        raise RuntimeError("Не удалось извлечь текст из PDF/изображения")

    existing_map: Dict[Tuple[int, int], str] = {}
    existing_canonicals: Dict[str, str] = {}
    if layer_json_path and layer_json_path.exists():
        existing_map = load_translation_mapping(layer_json_path, log)
        existing_canonicals = load_translation_canonicals(layer_json_path, log)
        if existing_map:
            log(f"PDF: найден готовый слой перевода, используем без обращения к API ({len(existing_map)} областей)")

    for layout in page_layouts:
        _apply_cached_translations(layout.blocks, existing_map)
        _apply_canonical_map(layout.blocks, existing_canonicals)

    backend_name = "cached-layer"
    canonical_cache = _collect_canonical_cache(all_blocks)
    missing_canonicals: Set[str] = set()
    pending_block_count = 0
    for block in all_blocks:
        if block.block_type not in TRANSLATABLE_BLOCK_TYPES:
            continue
        source_value = str(block.source_text or "").strip()
        if not source_value:
            continue
        canonical_value = _canonicalize_text_for_dedup(source_value)
        if canonical_value and canonical_value not in canonical_cache:
            pending_block_count += 1
            missing_canonicals.add(canonical_value)

    pdf_prompt_template = (
        "You are a professional translator for elevator manuals and catalogs (non-technical marketing/guide tone). "
        "Translate the provided values from {source_lang} to {target_lang}. "
        "Keep company names, brand names, product/model names, part codes, and addresses exactly as in the source. "
        "Preserve numbers, placeholders like '__DXF_DIM__', and DXF control sequences such as \"\\P\". Respond "
        "with strict JSON: {\"translations\": [{\"id\": \"<id>\", \"text\": \"<translated>\"}, ...]}"
    )

    if missing_canonicals:
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
            system_prompt_template=pdf_prompt_template,
        )
        backend_name = translator.backend_name()
        log(f"Инициализирован движок перевода: {backend_name}")

        log(
            f"Готовим {len(missing_canonicals)} уникальных блоков текста к переводу "
            f"(из {pending_block_count} без готового перевода)"
        )
        canonical_cache = BlockRegionTranslator(translator, log).translate(all_blocks, canonical_cache)

    _apply_canonical_translations(all_blocks, canonical_cache)

    for block in all_blocks:
        if block.render_id is not None and block.translated_text:
            existing_map[(block.page_index, block.render_id)] = block.translated_text

    page_regions: Dict[int, List[RegionInfo]] = {}
    all_regions: List[RegionInfo] = []
    for layout in page_layouts:
        regions = _build_regions_from_blocks(layout.page, layout.blocks, config, log=log)
        page_regions[layout.page.page_index] = regions
        all_regions.extend(regions)

    if layer_json_path:
        save_translation_mapping(all_regions, layer_json_path, log)

    rendered_images: List["Image.Image"] = []
    for layout in page_layouts:
        rendered_images.append(renderer.render_page(layout.page, page_regions.get(layout.page.page_index, [])))

    exporter.to_pdf(rendered_images, output_path, dpi=config.dpi)
    log(f"PDF: файл сохранён: {output_path}")

    return {
        "output_path": output_path,
        "backend": backend_name,
        "pages": len(rendered_images),
        "job_type": "pdf",
        "ocr_performed": True,
        "translated_blocks": len(all_blocks),
    }


def pdf_to_images(pdf_path: Path, dpi: int, image_format: str, log: PdfLog) -> List["Image.Image"]:
    """Convert a PDF to high-resolution RGB images (lossless/hi-res where possible)."""
    _require_dependency(convert_from_path, "Установите pdf2image (pip install pdf2image)")
    _require_dependency(Image, "Установите Pillow (pip install pillow)")
    poppler_path = _detect_poppler_path()
    poppler_hint = " Укажите переменную POPPLER_PATH или добавьте папку bin в PATH." if not poppler_path else ""
    try:
        images = convert_from_path(  # type: ignore[arg-type]
            str(pdf_path),
            dpi=int(dpi),
            fmt=image_format,
            poppler_path=poppler_path,
        )
    except Exception as exc:
        raise RuntimeError(f"Не удалось обработать PDF с помощью Poppler: {exc}{poppler_hint}") from exc
    if not images:
        raise RuntimeError("pdf2image не вернул страницы для обработки")
    rgb_images: List["Image.Image"] = []
    for idx, image in enumerate(images):
        mode = image.mode
        if mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        elif mode == "RGBA":
            image = image.convert("RGB")
        rgb_images.append(image)
        log(f"PDF: страница {idx + 1}/{len(images)} конвертирована в изображение")
    return rgb_images


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


def _remove_block_text_background_aware(image: "np.ndarray", block: TextBlock, config: PdfProcessingConfig) -> None:
    """
    Remove text within a block by filling letter shapes with the local background color
    and optionally smoothing. This mimics the "Google Translate" disappearance effect.
    """
    x, y, w, h = block.bbox
    height, width = image.shape[:2]
    x0 = max(0, min(width, int(x)))
    y0 = max(0, min(height, int(y)))
    x1 = max(0, min(width, int(x + w)))
    y1 = max(0, min(height, int(y + h)))
    if x1 <= x0 or y1 <= y0:
        return

    roi = image[y0:y1, x0:x1].copy()
    if roi.size == 0:
        return

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for word in block.word_boxes:
        wx, wy, ww, wh = word.bbox
        rx0 = max(0, min(roi.shape[1], int(wx - x0)))
        ry0 = max(0, min(roi.shape[0], int(wy - y0)))
        rx1 = max(0, min(roi.shape[1], int(wx - x0 + ww)))
        ry1 = max(0, min(roi.shape[0], int(wy - y0 + wh)))
        if rx1 > rx0 and ry1 > ry0:
            cv2.rectangle(mask, (rx0, ry0), (rx1, ry1), color=255, thickness=-1)

    if np.count_nonzero(mask) == 0:
        # Fallback: derive mask from the ROI itself (useful when word boxes are missing).
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(thresh) == 0:
            return
        white_pixels = np.count_nonzero(thresh == 255)
        black_pixels = np.count_nonzero(thresh == 0)
        if white_pixels > black_pixels:
            mask = 255 - thresh
        else:
            mask = thresh

    kernel_size = max(1, config.dilation_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Estimate colors before modification.
    bg_color = _estimate_background_color(roi, dilated_mask)
    text_color = _boost_text_contrast(_estimate_text_color(roi, dilated_mask), bg_color)
    block.background_color = bg_color
    block.text_color = text_color

    # Fill text area with background color.
    fill = np.full_like(roi, bg_color, dtype=np.uint8)
    text_region = cv2.bitwise_and(fill, fill, mask=dilated_mask)
    background_region = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(dilated_mask))
    cleaned = cv2.add(background_region, text_region)

    # Light blur to blend edges.
    cleaned = cv2.GaussianBlur(cleaned, (max(3, config.blur_kernel_size | 1), max(3, config.blur_kernel_size | 1)), 0)

    image[y0:y1, x0:x1] = cleaned


def build_text_mask(roi: "np.ndarray", *, dilation_kernel: int) -> Optional["np.ndarray"]:
    """Construct a binary mask highlighting text pixels inside ROI."""
    _require_dependency(cv2, "Установите opencv-python (pip install opencv-python)")
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pixels = np.count_nonzero(thresh == 255)
    black_pixels = np.count_nonzero(thresh == 0)
    if white_pixels > black_pixels:
        mask = 255 - thresh
    else:
        mask = thresh
    kernel_size = max(1, int(dilation_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


def images_to_pdf(images: Sequence["Image.Image"], output_path: Path, *, dpi: int, log: PdfLog) -> None:
    """Save processed page images back into a single PDF."""
    if not images:
        raise RuntimeError("Нет изображений для сохранения PDF")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base, *rest = images
    base.save(
        output_path,
        "PDF",
        resolution=dpi,
        save_all=True,
        append_images=rest,
    )
    log(f"PDF: сохранено {len(images)} страниц в {output_path}")


def _has_translation(existing: Dict[Tuple[int, int], str], region: RegionInfo) -> bool:
    key = (region.page_index, region.region_id)
    value = existing.get(key, "")
    return bool(str(value).strip())


def load_translation_mapping(path: Path, log: PdfLog) -> Dict[Tuple[int, int], str]:
    """Load an existing translation layer file if available."""
    mapping: Dict[Tuple[int, int], str] = {}
    if not path.exists():
        return mapping
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        log(f"PDF: не удалось прочитать слой перевода {path}: {exc}")
        return mapping

    pages = data.get("pages") if isinstance(data, dict) else None
    if not pages:
        return mapping
    for page in pages:
        try:
            page_index = int(page.get("page_index", 0))
        except Exception:
            continue
        blocks = page.get("regions") or page.get("blocks") or []
        for block in blocks:
            try:
                raw_id = block.get("region_id", block.get("block_id", 0))
                region_id = int(raw_id)
                translated_text = str(block.get("translated_text", "") or "").strip()
                if not translated_text:
                    continue
                mapping[(page_index, region_id)] = translated_text
            except Exception:
                continue
    return mapping

def load_translation_canonicals(path: Path, log: PdfLog) -> Dict[str, str]:
    """Load canonicalized text->translation pairs from a saved layer (ignoring region ids)."""
    canonicals: Dict[str, str] = {}
    if not path.exists():
        return canonicals
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        log(f"PDF: не удалось прочитать слой перевода {path}: {exc}")
        return canonicals
    pages = data.get("pages") if isinstance(data, dict) else None
    if not pages:
        return canonicals
    for page in pages:
        blocks = page.get("regions") or page.get("blocks") or []
        for block in blocks:
            try:
                source_text = str(block.get("text", "") or "").strip()
                translated_text = str(block.get("translated_text", "") or "").strip()
                if not source_text or not translated_text:
                    continue
                canonical = _canonicalize_text_for_dedup(source_text)
                if canonical:
                    canonicals.setdefault(canonical, translated_text)
                # Additionally add line-wise pairs when counts match (helps if Textract разбил блоки).
                src_lines = [ln.strip() for ln in source_text.splitlines() if ln.strip()]
                dst_lines = [ln.strip() for ln in translated_text.splitlines() if ln.strip()]
                if len(src_lines) == len(dst_lines) and src_lines:
                    for s, t in zip(src_lines, dst_lines):
                        can_line = _canonicalize_text_for_dedup(s)
                        if can_line:
                            canonicals.setdefault(can_line, t)
            except Exception:
                continue
    return canonicals


class _FontLoader:
    """Lazy font loader with fallback to default PIL font."""

    def __init__(self, font_value: Optional[str]) -> None:
        self.font_value = font_value or ""
        self._cache: Dict[int, "ImageFont.FreeTypeFont"] = {}
        self._preferred_font_path = self._resolve_font_path()

    def get(self, size: int) -> "ImageFont.FreeTypeFont":
        size = max(8, min(128, int(size)))
        cached = self._cache.get(size)
        if cached is not None:
            return cached
        font_paths = []
        if self.font_value:
            font_paths.append(Path(self.font_value).expanduser())
        if self._preferred_font_path:
            font_paths.append(self._preferred_font_path)
        for candidate in font_paths:
            try:
                font = ImageFont.truetype(str(candidate), size=size, encoding="utf-8")
                self._cache[size] = font
                return font
            except Exception:
                continue
        try:
            font = ImageFont.load_default()
            self._cache[size] = font
            return font
        except Exception:
            raise RuntimeError("Не удалось загрузить шрифт для наложения перевода")

    def _resolve_font_path(self) -> Optional[Path]:
        """Pick a Unicode-capable font (Cyrillic-friendly) for stable rendering."""
        candidates: List[Path] = []
        # PIL ships with DejaVu fonts; prefer them because they cover Cyrillic.
        try:
            import PIL

            pil_root = Path(PIL.__file__).resolve().parent
            for name in ("DejaVuSans.ttf", "DejaVuSansCondensed.ttf"):
                candidate = pil_root / "fonts" / name
                if candidate.exists():
                    candidates.append(candidate)
        except Exception:
            pass
        # Common system fonts that typically include Cyrillic glyphs.
        system_fonts = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/Library/Fonts/Arial Unicode MS.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        ]
        candidates.extend([p for p in system_fonts if p.exists()])
        return candidates[0] if candidates else None


def _measure_text(text: str, draw: "ImageDraw.ImageDraw", font: "ImageFont.FreeTypeFont") -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    size = draw.textsize(text, font=font)
    return int(size[0]), int(size[1])


def _wrap_text(text: str, draw: "ImageDraw.ImageDraw", font: "ImageFont.FreeTypeFont", max_width: int) -> List[str]:
    if max_width <= 0:
        return [text]
    lines: List[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            width, _ = _measure_text(candidate, draw, font)
            if width <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _estimate_font_size(block_info: Optional[TextBlock], block_height: int) -> int:
    """Estimate a reasonable font size from OCR word box heights."""
    if block_info and block_info.word_boxes:
        heights: List[int] = []
        for word in block_info.word_boxes:
            _, _, _, h = word.bbox
            heights.append(h)
        if heights:
            heights.sort()
            median = heights[len(heights) // 2]
            return max(8, min(72, median))
        approximate = int(block_height / max(1, len(block_info.word_boxes)))
        return max(8, min(72, approximate))
    return max(8, min(72, int(block_height * 0.7)))


def _sample_text_color(image: "Image.Image", bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    region = image.crop((bbox[0], bbox[1], bbox[2], bbox[3])).convert("RGB")
    if region.width <= 0 or region.height <= 0:
        return (20, 20, 20, 255)
    np_region = np.array(region)
    flat = np_region.reshape(-1, 3)
    brightness = flat.mean(axis=1)
    dark_pixels = flat[brightness < 180]
    target = dark_pixels if len(dark_pixels) >= 10 else flat
    if len(target) == 0:
        return (20, 20, 20, 255)
    median_color = np.median(target, axis=0).astype(int)
    r, g, b = [int(max(0, min(255, c))) for c in median_color.tolist()]
    return (r, g, b, 255)


def _estimate_background_color(roi: "np.ndarray", text_mask: "np.ndarray") -> Tuple[int, int, int]:
    """Estimate local background color from pixels outside the text mask (BGR input)."""
    if roi.size == 0 or text_mask.size == 0:
        return (255, 255, 255)
    if text_mask.shape[:2] != roi.shape[:2]:
        mask = cv2.resize(text_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = text_mask
    bg_pixels = roi[mask == 0]
    if bg_pixels.size == 0:
        bg_pixels = roi.reshape(-1, 3)
    median = np.median(bg_pixels, axis=0).astype(int)
    b, g, r = [int(max(0, min(255, c))) for c in median.tolist()]
    return (b, g, r)


def _estimate_text_color(roi: "np.ndarray", text_mask: "np.ndarray") -> Tuple[int, int, int]:
    """Estimate original text color from pixels inside the text mask (BGR input)."""
    if roi.size == 0 or text_mask.size == 0:
        return (20, 20, 20)
    if text_mask.shape[:2] != roi.shape[:2]:
        mask = cv2.resize(text_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = text_mask
    text_pixels = roi[mask > 0]
    if text_pixels.size == 0:
        text_pixels = roi.reshape(-1, 3)
    median = np.median(text_pixels, axis=0).astype(int)
    b, g, r = [int(max(0, min(255, c))) for c in median.tolist()]
    return (b, g, r)


def _boost_text_contrast(text_color: Tuple[int, int, int], bg_color: Tuple[int, int, int], *, min_diff: int = 40) -> Tuple[int, int, int]:
    """Make text color more distinguishable from background without changing hue drastically."""
    tb, tg, tr = text_color
    bb, bg, br = bg_color
    text_l = 0.299 * tr + 0.587 * tg + 0.114 * tb
    bg_l = 0.299 * br + 0.587 * bg + 0.114 * bb
    diff = abs(text_l - bg_l)
    if diff >= min_diff:
        return (tb, tg, tr)
    adjust = int(min_diff - diff)
    if text_l >= bg_l:
        tb = max(0, tb - adjust)
        tg = max(0, tg - adjust)
        tr = max(0, tr - adjust)
    else:
        tb = min(255, tb + adjust)
        tg = min(255, tg + adjust)
        tr = min(255, tr + adjust)
    return (tb, tg, tr)


def save_translation_mapping(
    regions: Sequence[RegionInfo],
    output_path: Path,
    log: PdfLog,
) -> None:
    """Persist mapping between original and translated regions for later overlay."""
    pages: Dict[int, List[RegionInfo]] = {}
    for region in regions:
        pages.setdefault(region.page_index, []).append(region)

    payload_pages: List[Dict[str, object]] = []
    for page_index in sorted(pages.keys()):
        page_regions = pages[page_index]
        blocks_payload: List[Dict[str, object]] = []
        for region in page_regions:
            blocks_payload.append(
                {
                    "region_id": region.region_id,
                    "bbox": list(region.bbox),
                    "text": region.source_text,
                    "translated_text": region.translated_text or "",
                }
            )
        payload_pages.append(
            {
                "page_index": page_index,
                "blocks": blocks_payload,
                "regions": blocks_payload,
            }
        )
    payload = {
        "version": 1,
        "pages": payload_pages,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    log(f"PDF: сохранён JSON слой: {output_path}")


def _safe_int(seq: Sequence, index: int, default: int = 0) -> int:
    try:
        return int(seq[index])
    except Exception:
        return default


def _safe_float(seq: Sequence, index: int, default: float = 0.0) -> float:
    try:
        return float(seq[index])
    except Exception:
        return default


def _detect_poppler_path() -> Optional[str]:
    for env_name in ("POPPLER_PATH", "POPPLER_BIN", "POPPLER_HOME"):
        candidate = os.environ.get(env_name)
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path)
    return None


if __name__ == "__main__":  # pragma: no cover - manual usage hint
    sample_input = Path("sample.pdf")
    sample_output = Path("sample_blurred.pdf")
    if sample_input.exists():
        print(f"Processing {sample_input} -> {sample_output}")
        process_pdf(str(sample_input), str(sample_output))
    else:
        print("Поместите sample.pdf рядом со скриптом для быстрой проверки")
