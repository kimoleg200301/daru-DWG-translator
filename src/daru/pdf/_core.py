from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import re

from io import BytesIO
import numpy as np

from daru.config import ORIGINAL_FONT_VALUE, normalize_style_font
from daru.pdf.vision import (
    CodexCliVisionLayoutRefiner,
    CodexCliVisionQaReviewer,
    OpenAIVisionLayoutRefiner,
    OpenAIVisionQaReviewer,
    VisionBlockUpdate,
    VisionQaIssue,
    VISION_IMAGE_QUALITIES,
    VISION_IMAGE_QUALITY_STABLE,
    VISION_REQUEST_MODE_BATCHED,
    VISION_REQUEST_MODES,
)
from daru.translation.analysis import CodexAnalysisSession
from daru.translation.checkpoint import (
    TranslationCheckpointStore,
    default_checkpoint_path,
)
from daru.translation.engine import TranslationEngine, chunked

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
PDF_PROCESSING_TEXTRACT = "textract"
PDF_PROCESSING_TEXTRACT_VISION = "textract_vision"
TRANSLATABLE_BLOCK_TYPES: Set[str] = {"TEXT", "TITLE", "LIST", "CELL", "KEY", "VALUE"}
FIGURE_BLOCK_TYPES: Set[str] = {"LAYOUT_FIGURE", "FIGURE"}
# Use a dedicated cache folder next to this script instead of a generic logs/ path.
_SCRIPT_ROOT = Path(__file__).resolve().parent
_CACHE_DIR = _SCRIPT_ROOT / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEXTRACT_LOG_PATH = _CACHE_DIR / "textract_requests.log"


def _noop_log(_message: str) -> None:
    return


def _resolve_scanned_style_font(
    style_font: Optional[str],
    log: PdfLog,
) -> Optional[str]:
    resolved = normalize_style_font(style_font or "")
    if resolved == ORIGINAL_FONT_VALUE:
        log(
            "PDF: для отсканированной страницы невозможно надежно определить "
            "исходное семейство шрифта; используется Unicode-шрифт по умолчанию"
        )
        return None
    return resolved or None


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
    textract_split_lines: bool = False
    textract_vertical_aspect_ratio: float = 2.2
    textract_vertical_top_margin_ratio: float = 0.15 # добавляет больше рабочую область переведенного вертикального текста
    text_bold_fill_ratio: float = 0.45
    merge_allow_horizontal_gap: bool = False
    blur_kernel_size: int = 23
    dilation_kernel_size: int = 3
    region_margin_scale_x: float = 0.35
    region_margin_scale_y: float = 0.25
    min_region_margin_px: int = 3
    max_region_margin_ratio: float = 0.2
    textract_bbox_height_boost: float = 0.15 # добавляет больше рабочую область переведенного текста
    textract_bbox_width_boost: float = 0.1
    vision_backend: str = "openai_api"
    vision_model: Optional[str] = "gpt-5.5"
    vision_reasoning_effort: Optional[str] = "medium"
    vision_api_key: Optional[str] = None
    vision_base_url: Optional[str] = None
    vision_project: Optional[str] = None
    vision_codex_cli_path: Optional[str] = None
    vision_codex_timeout_seconds: int = 300
    vision_request_mode: str = VISION_REQUEST_MODE_BATCHED
    vision_image_quality: str = VISION_IMAGE_QUALITY_STABLE
    vision_cache_path: Optional[Path] = None

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
            vision_backend=(
                str(self.vision_backend or "openai_api").strip().lower()
                if str(self.vision_backend or "openai_api").strip().lower()
                in {"openai_api", "codex_cli"}
                else "openai_api"
            ),
            vision_model=self.vision_model or "gpt-5.5",
            vision_reasoning_effort=self.vision_reasoning_effort or "medium",
            vision_api_key=self.vision_api_key or None,
            vision_base_url=self.vision_base_url or None,
            vision_project=self.vision_project or None,
            vision_codex_cli_path=self.vision_codex_cli_path or None,
            vision_codex_timeout_seconds=max(
                10, int(self.vision_codex_timeout_seconds or 300)
            ),
            vision_request_mode=(
                str(self.vision_request_mode or VISION_REQUEST_MODE_BATCHED).strip().lower()
                if str(self.vision_request_mode or VISION_REQUEST_MODE_BATCHED).strip().lower()
                in VISION_REQUEST_MODES
                else VISION_REQUEST_MODE_BATCHED
            ),
            vision_image_quality=(
                str(self.vision_image_quality or VISION_IMAGE_QUALITY_STABLE).strip().lower()
                if str(self.vision_image_quality or VISION_IMAGE_QUALITY_STABLE).strip().lower()
                in VISION_IMAGE_QUALITIES
                else VISION_IMAGE_QUALITY_STABLE
            ),
            vision_cache_path=Path(self.vision_cache_path) if self.vision_cache_path else None,
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
    physical_size_points: Optional[Tuple[float, float]] = None


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
    source_font_size_estimate: int = 0
    rendered_font_size: int = 0
    is_vertical: bool = False
    is_bold: bool = False
    font_weight: str = "auto"
    is_table_cell: bool = False
    stable_id: str = ""
    alignment: str = "left"
    confidence: float = 0.0
    parent_block_id: Optional[str] = None
    content_bbox: Optional[BBoxWH] = None
    corrected_text: str = ""
    semantic_type: str = "unknown"
    fit_strategy: str = "default"
    vision_confidence: float = 0.0
    qa_flags: List[str] = field(default_factory=list)


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
    word_boxes: List[BBoxWH] = field(default_factory=list)
    confidence: float = 0.0
    rotation_deg: float = 0.0
    parent_block_id: Optional[str] = None


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
    word_boxes: List[BBoxWH] = field(default_factory=list)
    is_table: bool = False
    confidence: float = 0.0
    rotation_deg: float = 0.0
    parent_block_id: Optional[str] = None
    parent_block_type: Optional[str] = None
    parent_bbox: Optional[Tuple[int, int, int, int]] = None
    alignment: str = "left"
    font_weight: str = "auto"
    stable_id: str = ""
    should_translate: bool = True
    render_enabled: bool = True
    corrected_text: str = ""
    semantic_type: str = "unknown"
    fit_strategy: str = "default"
    vision_confidence: float = 0.0
    qa_flags: List[str] = field(default_factory=list)


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


def _bbox_intersection_ratio(
    inner: Tuple[int, int, int, int],
    outer: Tuple[int, int, int, int],
) -> float:
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter_area = float((ix1 - ix0) * (iy1 - iy0))
    inner_area = float(max(1, (inner[2] - inner[0]) * (inner[3] - inner[1])))
    return inter_area / inner_area


def _median_float(values: Sequence[float], default: float = 0.0) -> float:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return default
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


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
        physical_sizes: List[Optional[Tuple[float, float]]] = [None] * len(images)
        try:
            import fitz  # type: ignore

            document = fitz.open(str(pdf_path))
            try:
                for idx in range(min(len(images), len(document))):
                    rect = document[idx].rect
                    physical_sizes[idx] = (float(rect.width), float(rect.height))
            finally:
                document.close()
        except Exception:
            pass
        pages: List[PageImage] = []
        for idx, image in enumerate(images):
            width, height = image.size
            pages.append(
                PageImage(
                    page_index=idx,
                    image=image,
                    width=width,
                    height=height,
                    dpi=self.config.render_dpi,
                    physical_size_points=physical_sizes[idx],
                )
            )
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
                    "feature_types": tuple(entry.get("feature_types") or req.get("feature_types") or []),
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
        desired_feature_types = meta.get("feature_types")
        if desired_feature_types:
            candidates = [item for item in candidates if item.get("feature_types") == desired_feature_types]
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
        feature_types = ["LAYOUT", "SIGNATURES", "TABLES"]
        img_meta["feature_types"] = tuple(feature_types)
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
            confidence = float(block.get("Confidence") or 0.0)
            rotation_deg = self._rotation_from_geometry(block.get("Geometry"))
            word_boxes: List[BBoxWH] = []
            for rel in block.get("Relationships", []) or []:
                if str(rel.get("Type", "")).upper() != "CHILD":
                    continue
                for child_id in rel.get("Ids", []) or []:
                    child = blocks_by_id.get(child_id)
                    if not child:
                        continue
                    if str(child.get("BlockType", "")).upper() != "WORD":
                        continue
                    word_bbox = self._bbox_from_geometry(child.get("Geometry"), page_width, page_height)
                    if not word_bbox:
                        continue
                    wx0, wy0, wx1, wy1 = word_bbox
                    if wx1 <= wx0 or wy1 <= wy0:
                        continue
                    word_boxes.append((wx0, wy0, wx1 - wx0, wy1 - wy0))
            lines.append(
                LineRegion(
                    page_index=page_index,
                    line_id=str(block["Id"]),
                    bbox=bbox,
                    source_text=text,
                    word_boxes=word_boxes,
                    confidence=confidence,
                    rotation_deg=rotation_deg,
                )
            )
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
        if allow_figures and page_image is not None:
            figure_blocks = [
                block
                for block in blocks_by_id.values()
                if str(block.get("BlockType", "")).upper() in FIGURE_BLOCK_TYPES
            ]
            for figure_block in figure_blocks:
                figure_id = str(figure_block.get("Id", ""))
                if figure_id:
                    figure_line_ids.update(
                        self._collect_line_ids_from_block(
                            figure_id,
                            blocks_by_id,
                            set(),
                        )
                    )
        has_table_cells = any(
            str(block.get("BlockType", "")).upper() in {"CELL", "MERGED_CELL"} for block in blocks_by_id.values()
        )
        cell_bboxes: List[Tuple[int, int, int, int]] = []
        merged_children_map: Dict[str, List[str]] = {}
        skip_cell_ids: Set[str] = set()
        if has_table_cells:
            for block in blocks_by_id.values():
                block_type = str(block.get("BlockType", "")).upper()
                if block_type not in {"CELL", "MERGED_CELL"}:
                    continue
                block_id = str(block.get("Id", ""))
                bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
                if bbox:
                    cell_bboxes.append(bbox)
                if block_type == "MERGED_CELL":
                    child_cells: List[str] = []
                    for rel in block.get("Relationships", []) or []:
                        if str(rel.get("Type", "")).upper() != "CHILD":
                            continue
                        for child_id in rel.get("Ids", []) or []:
                            child = blocks_by_id.get(child_id)
                            if not child:
                                continue
                            if str(child.get("BlockType", "")).upper() == "CELL":
                                child_cells.append(str(child_id))
                                skip_cell_ids.add(str(child_id))
                    merged_children_map[block_id] = child_cells
        for block in blocks_by_id.values():
            block_type_raw = str(block.get("BlockType", "")).upper()
            if block_type_raw in FIGURE_BLOCK_TYPES and allow_figures and page_image is not None:
                continue
            if has_table_cells and block_type_raw in {"TABLE", "LAYOUT_TABLE"}:
                blocks.extend(
                    self._build_layout_table_lines(
                        block,
                        lines_by_id,
                        used_line_ids,
                        cell_bboxes,
                        page_index=page_index,
                    )
                )
                continue
            if block_type_raw in {"CELL", "MERGED_CELL"}:
                if block_type_raw == "CELL":
                    block_id = str(block.get("Id", ""))
                    if block_id in skip_cell_ids:
                        continue
                    cell_region = self._build_cell_block(
                        block,
                        blocks_by_id,
                        lines_by_id,
                        used_line_ids,
                        page_index=page_index,
                        page_width=page_width,
                        page_height=page_height,
                    )
                else:
                    block_id = str(block.get("Id", ""))
                    child_cells = merged_children_map.get(block_id, [])
                    cell_region = self._build_merged_cell_block(
                        block,
                        blocks_by_id,
                        lines_by_id,
                        used_line_ids,
                        child_cells,
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
        self._finalize_block_metadata(blocks)
        blocks = self._deduplicate_blocks(blocks)
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
        parent_id = str(block.get("Id", "")) or None
        parent_bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
        if block_type == "LIST" and line_regions:
            groups = self._group_list_lines(line_regions)
            base_id = str(block.get("Id", ""))
            for idx, group in enumerate(groups):
                for line in group:
                    used_line_ids.add(line.line_id)
                group_bbox = _union_bbox([line.bbox for line in group]) or group[0].bbox
                blocks_out.append(
                    BlockRegion(
                        page_index=page_index,
                        block_id=f"{base_id}:item:{idx}" if base_id else f"list:item:{idx}",
                        block_type=block_type,
                        bbox=group_bbox,
                        lines=list(group),
                        source_text="\n".join(
                            line.source_text.strip()
                            for line in group
                            if line.source_text.strip()
                        ),
                        confidence=_median_float([line.confidence for line in group]),
                        rotation_deg=_median_float([line.rotation_deg for line in group]),
                        parent_block_id=parent_id,
                        parent_block_type=block_type,
                        parent_bbox=parent_bbox,
                    )
                )
            return blocks_out
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
                        confidence=ln.confidence,
                        rotation_deg=ln.rotation_deg,
                        parent_block_id=parent_id,
                        parent_block_type=block_type,
                        parent_bbox=parent_bbox,
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
                parent_block_id=parent_id,
                parent_block_type=block_type,
                parent_bbox=parent_bbox,
            )
        )
        return blocks_out

    def _group_list_lines(self, lines: Sequence[LineRegion]) -> List[List[LineRegion]]:
        ordered = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
        if len(ordered) <= 1:
            return [list(ordered)] if ordered else []
        heights = [max(1, line.bbox[3] - line.bbox[1]) for line in ordered]
        median_height = _median_float(heights, 1.0)
        bullet_re = re.compile(r"^(?:[-*•▪◦◇◆]|\(?\d+[.)])\s*")
        groups: List[List[LineRegion]] = [[ordered[0]]]
        for line in ordered[1:]:
            previous = groups[-1][-1]
            vertical_gap = max(0, line.bbox[1] - previous.bbox[3])
            starts_item = bool(bullet_re.match(line.source_text.strip()))
            same_indent = abs(line.bbox[0] - groups[-1][0].bbox[0]) <= median_height
            if starts_item or (same_indent and vertical_gap > median_height * 0.75):
                groups.append([line])
            else:
                groups[-1].append(line)
        return groups

    def _line_center_in_any_bbox(self, line: LineRegion, bboxes: Sequence[Tuple[int, int, int, int]]) -> bool:
        if not bboxes:
            return False
        x0, y0, x1, y1 = line.bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        for bx0, by0, bx1, by1 in bboxes:
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                return True
        return False

    def _bbox_intersection_ratio(
        self, inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]
    ) -> float:
        ix0 = max(inner[0], outer[0])
        iy0 = max(inner[1], outer[1])
        ix1 = min(inner[2], outer[2])
        iy1 = min(inner[3], outer[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter_area = float((ix1 - ix0) * (iy1 - iy0))
        inner_area = float(max(1, (inner[2] - inner[0]) * (inner[3] - inner[1])))
        return inter_area / inner_area

    def _build_layout_table_lines(
        self,
        block: Dict[str, object],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        cell_bboxes: Sequence[Tuple[int, int, int, int]],
        *,
        page_index: int,
    ) -> List[BlockRegion]:
        line_regions: List[LineRegion] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                if child_id in used_line_ids:
                    continue
                line = lines_by_id.get(child_id)
                if line:
                    line_regions.append(line)
        line_regions = self._dedup_and_sort_lines(line_regions)
        if cell_bboxes:
            line_regions = [ln for ln in line_regions if not self._line_center_in_any_bbox(ln, cell_bboxes)]
        if not line_regions:
            return []
        grouped = self._group_layout_table_lines(line_regions)
        blocks_out: List[BlockRegion] = []
        base_id = str(block.get("Id", ""))
        for idx, group in enumerate(grouped):
            for ln in group:
                used_line_ids.add(ln.line_id)
            bbox = _union_bbox([ln.bbox for ln in group]) or group[0].bbox
            source_text = " ".join(ln.source_text.strip() for ln in group if ln.source_text.strip())
            word_boxes: List[BBoxWH] = []
            for ln in group:
                if ln.word_boxes:
                    word_boxes.extend(ln.word_boxes)
            block_id = f"{base_id}:para:{idx}" if base_id else f"layout_table:para:{idx}"
            blocks_out.append(
                BlockRegion(
                    page_index=page_index,
                    block_id=block_id,
                    block_type="TEXT",
                    bbox=bbox,
                    lines=group,
                    cells=[],
                    source_text=source_text,
                    word_boxes=word_boxes,
                    is_table=True,
                    parent_block_id=base_id or None,
                    parent_block_type="LAYOUT_TABLE",
                    parent_bbox=bbox,
                )
            )
        return blocks_out

    def _group_layout_table_lines(self, lines: Sequence[LineRegion]) -> List[List[LineRegion]]:
        if not lines:
            return []
        ordered = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        heights = [max(1, ln.bbox[3] - ln.bbox[1]) for ln in ordered]
        median_h = float(np.median(heights)) if heights else 1.0
        gap_threshold = max(2.0, 0.6 * median_h)
        align_threshold = max(6.0, 0.6 * median_h)
        groups: List[List[LineRegion]] = []
        current: List[LineRegion] = [ordered[0]]
        for ln in ordered[1:]:
            prev = current[-1]
            gap = float(ln.bbox[1] - prev.bbox[3])
            if gap < 0:
                gap = 0.0
            overlap = min(prev.bbox[2], ln.bbox[2]) - max(prev.bbox[0], ln.bbox[0])
            min_w = float(
                max(1, min(prev.bbox[2] - prev.bbox[0], ln.bbox[2] - ln.bbox[0]))
            )
            overlap_ratio = max(0.0, overlap / min_w)
            left_delta = abs(ln.bbox[0] - prev.bbox[0])
            if gap <= gap_threshold and (overlap_ratio >= 0.3 or left_delta <= align_threshold):
                current.append(ln)
            else:
                groups.append(current)
                current = [ln]
        groups.append(current)
        return groups

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
        word_boxes = self._collect_word_boxes(cell_block, blocks_by_id, page_width, page_height)
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
            word_boxes=word_boxes,
            is_table=True,
            parent_block_id=str(cell_block.get("Id", "")) or None,
            parent_block_type="CELL",
            parent_bbox=bbox,
        )

    def _build_merged_cell_block(
        self,
        merged_block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        child_cell_ids: Sequence[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> Optional[BlockRegion]:
        bbox = self._bbox_from_geometry(merged_block.get("Geometry"), page_width, page_height)
        if not bbox:
            return None
        ordered_texts: List[Tuple[int, int, str]] = []
        word_boxes: List[BBoxWH] = []
        for child_id in child_cell_ids:
            child = blocks_by_id.get(child_id)
            if not child:
                continue
            if str(child.get("BlockType", "")).upper() != "CELL":
                continue
            text, _ = self._collect_text(child, blocks_by_id, lines_by_id, None)
            text = text.strip()
            if text:
                row_idx = int(child.get("RowIndex") or 0)
                col_idx = int(child.get("ColumnIndex") or 0)
                ordered_texts.append((row_idx, col_idx, text))
            word_boxes.extend(self._collect_word_boxes(child, blocks_by_id, page_width, page_height))
        source_text = ""
        if ordered_texts:
            ordered_texts.sort(key=lambda t: (t[0], t[1]))
            source_text = " ".join(text for _, _, text in ordered_texts if text)
        if not source_text:
            fallback_text, lines = self._collect_text(merged_block, blocks_by_id, lines_by_id, used_line_ids)
            source_text = fallback_text.strip()
        if not source_text:
            return None
        return BlockRegion(
            page_index=page_index,
            block_id=str(merged_block.get("Id", "")),
            block_type="CELL",
            bbox=bbox,
            lines=[],
            cells=[],
            source_text=source_text,
            word_boxes=word_boxes,
            is_table=True,
            parent_block_id=str(merged_block.get("Id", "")) or None,
            parent_block_type="MERGED_CELL",
            parent_bbox=bbox,
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
        figure_id = str(block.get("Id", ""))
        usable_lines = [line for line in ordered_lines if self._is_usable_figure_line(line)]
        if usable_lines:
            detected_blocks = self._figure_blocks_from_lines(
                usable_lines,
                figure_id=figure_id,
                figure_bbox=bbox,
                page_index=page_index,
            )
        else:
            detected_blocks = self._analyze_figure_region(bbox, page_image, block_id=figure_id)
        container: Optional[BlockRegion] = None
        if bbox:
            container = BlockRegion(
                page_index=page_index,
                block_id=figure_id,
                block_type="FIGURE",
                bbox=bbox,
                lines=[],
                cells=[],
                child_ids=child_ids,
                source_text="",
            )
        return container, detected_blocks

    def _is_usable_figure_line(self, line: LineRegion) -> bool:
        text = re.sub(r"\s+", " ", str(line.source_text or "")).strip()
        if not text:
            return False
        if line.confidence and line.confidence < max(40.0, float(self.config.min_confidence) - 10.0):
            return False
        alnum_count = sum(character.isalnum() for character in text)
        letter_count = sum(character.isalpha() for character in text)
        if alnum_count < 2 or (letter_count == 0 and alnum_count < 4):
            return False
        if len(text) <= 3 and text.isupper():
            # Short diagram labels such as "AND" are meaningful. Keep them only
            # when Textract is highly confident; this still rejects common OCR
            # fragments such as "#2", stray punctuation, and uncertain initials.
            if line.confidence < 90.0 or letter_count < 2:
                return False
        symbols = sum(not character.isalnum() and not character.isspace() for character in text)
        return symbols <= max(4, int(len(text) * 0.55))

    def _figure_blocks_from_lines(
        self,
        lines: Sequence[LineRegion],
        *,
        figure_id: str,
        figure_bbox: Optional[Tuple[int, int, int, int]],
        page_index: int,
    ) -> List[BlockRegion]:
        groups = self._group_layout_table_lines(lines)
        output: List[BlockRegion] = []
        for idx, group in enumerate(groups):
            bbox = _union_bbox([line.bbox for line in group])
            if bbox is None:
                continue
            text = "\n".join(line.source_text.strip() for line in group if line.source_text.strip()).strip()
            if not text:
                continue
            word_boxes = [box for line in group for box in line.word_boxes]
            output.append(
                BlockRegion(
                    page_index=page_index,
                    block_id=f"{figure_id}:text:{idx}",
                    block_type="TEXT",
                    bbox=bbox,
                    lines=list(group),
                    source_text=text,
                    word_boxes=word_boxes,
                    confidence=_median_float([line.confidence for line in group]),
                    rotation_deg=_median_float([line.rotation_deg for line in group]),
                    parent_block_id=figure_id or None,
                    parent_block_type="FIGURE",
                    parent_bbox=figure_bbox,
                )
            )
        return output

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
        img_meta["feature_types"] = tuple(feature_types)
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
        shifted_lines: List[LineRegion] = []
        for line in lines.values():
            if not self._is_usable_figure_line(line):
                continue
            lx0, ly0, lx1, ly1 = line.bbox
            shifted_bbox = (lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0)
            shifted_word_boxes: List[BBoxWH] = []
            for wx, wy, ww, wh in line.word_boxes:
                shifted_word_boxes.append((wx + x0, wy + y0, ww, wh))
            shifted_line = LineRegion(
                page_index=line.page_index,
                line_id=line.line_id,
                bbox=shifted_bbox,
                source_text=line.source_text,
                translated_text=line.translated_text,
                word_boxes=shifted_word_boxes,
                confidence=line.confidence,
                rotation_deg=line.rotation_deg,
                parent_block_id=block_id or None,
            )
            shifted_lines.append(shifted_line)
        return self._figure_blocks_from_lines(
            shifted_lines,
            figure_id=block_id,
            figure_bbox=bbox,
            page_index=page.page_index,
        )

    def _finalize_block_metadata(self, blocks: Sequence[BlockRegion]) -> None:
        for block in blocks:
            if block.block_type not in TRANSLATABLE_BLOCK_TYPES:
                continue
            if block.lines:
                for line in block.lines:
                    if line.parent_block_id is None:
                        line.parent_block_id = block.parent_block_id or block.block_id
                if not block.word_boxes:
                    block.word_boxes = [box for line in block.lines for box in line.word_boxes]
                if block.confidence <= 0:
                    block.confidence = _median_float([line.confidence for line in block.lines])
                if abs(block.rotation_deg) < 1e-6:
                    block.rotation_deg = _median_float([line.rotation_deg for line in block.lines])
            block.alignment = self._infer_alignment(block)

    def _infer_alignment(self, block: BlockRegion) -> str:
        if block.block_type == "TITLE":
            return "center"
        content_bbox = _block_content_bbox(block) or block.bbox
        container = block.parent_bbox or block.bbox
        if content_bbox is None:
            return "left"
        left_gap = max(0, content_bbox[0] - container[0])
        right_gap = max(0, container[2] - content_bbox[2])
        width = max(1, container[2] - container[0])
        if abs(left_gap - right_gap) <= width * 0.12:
            return "center"
        if right_gap <= width * 0.08 and left_gap > right_gap * 1.5:
            return "right"
        return "left"

    def _deduplicate_blocks(self, blocks: Sequence[BlockRegion]) -> List[BlockRegion]:
        blocks = self._remove_table_text_overlays(blocks)
        kept: List[BlockRegion] = []
        removed = 0

        def priority(block: BlockRegion) -> Tuple[int, float, int]:
            type_priority = {
                "CELL": 5,
                "TITLE": 4,
                "LIST": 4,
                "KEY": 4,
                "VALUE": 4,
                "TEXT": 3,
            }.get(block.block_type, 1)
            return (type_priority, block.confidence, len(block.source_text or ""))

        for block in blocks:
            if block.block_type not in TRANSLATABLE_BLOCK_TYPES:
                kept.append(block)
                continue
            canonical = re.sub(r"\s+", " ", str(block.source_text or "")).strip().casefold()
            if not canonical:
                continue
            duplicate_index: Optional[int] = None
            for idx, previous in enumerate(kept):
                if previous.block_type not in TRANSLATABLE_BLOCK_TYPES:
                    continue
                previous_canonical = re.sub(
                    r"\s+", " ", str(previous.source_text or "")
                ).strip().casefold()
                if canonical != previous_canonical:
                    continue
                overlap = max(
                    _bbox_intersection_ratio(block.bbox, previous.bbox),
                    _bbox_intersection_ratio(previous.bbox, block.bbox),
                )
                if overlap >= 0.78:
                    duplicate_index = idx
                    break
            if duplicate_index is None:
                kept.append(block)
                continue
            removed += 1
            previous = kept[duplicate_index]
            if priority(block) > priority(previous):
                kept[duplicate_index] = block
        if removed:
            self.log(f"Textract: удалено геометрических дублей блоков: {removed}")
        return kept

    def _remove_table_text_overlays(self, blocks: Sequence[BlockRegion]) -> List[BlockRegion]:
        cells = [
            block
            for block in blocks
            if block.block_type == "CELL" and block.bbox[2] > block.bbox[0] and block.bbox[3] > block.bbox[1]
        ]
        if not cells:
            return list(blocks)

        def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
            x0 = max(a[0], b[0])
            y0 = max(a[1], b[1])
            x1 = min(a[2], b[2])
            y1 = min(a[3], b[3])
            if x1 <= x0 or y1 <= y0:
                return 0
            return (x1 - x0) * (y1 - y0)

        filtered: List[BlockRegion] = []
        removed = 0
        for block in blocks:
            if block.block_type == "CELL" or block.block_type not in TRANSLATABLE_BLOCK_TYPES or block.is_table:
                filtered.append(block)
                continue
            area = max(1, (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1]))
            overlap_area = 0
            overlap_cells = 0
            for cell in cells:
                inter = intersection_area(block.bbox, cell.bbox)
                if inter <= 0:
                    continue
                overlap_area += inter
                overlap_cells += 1
            if overlap_cells >= 2 and overlap_area / float(area) >= 0.45:
                removed += 1
                continue
            filtered.append(block)
        if removed:
            self.log(f"Textract: removed table text overlays: {removed}")
        return filtered

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
            joined = " ".join(line.source_text.strip() for line in line_regions if line.source_text.strip())
            return joined, line_regions
        return " ".join(words).strip(), []

    def _collect_word_boxes(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> List[BBoxWH]:
        word_boxes: List[BBoxWH] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                child = blocks_by_id.get(child_id)
                if not child:
                    continue
                if str(child.get("BlockType", "")).upper() != "WORD":
                    continue
                word_bbox = self._bbox_from_geometry(child.get("Geometry"), page_width, page_height)
                if not word_bbox:
                    continue
                wx0, wy0, wx1, wy1 = word_bbox
                if wx1 <= wx0 or wy1 <= wy0:
                    continue
                word_boxes.append((wx0, wy0, wx1 - wx0, wy1 - wy0))
        return word_boxes

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

    def _rotation_from_geometry(self, geometry: Optional[Dict[str, object]]) -> float:
        if not isinstance(geometry, dict):
            return 0.0
        try:
            explicit = geometry.get("RotationAngle")
            if explicit is not None:
                return float(explicit) % 360.0
        except Exception:
            pass
        polygon = geometry.get("Polygon")
        if not isinstance(polygon, list) or len(polygon) < 2:
            return 0.0
        try:
            first = polygon[0]
            second = polygon[1]
            dx = float(second.get("X", 0.0)) - float(first.get("X", 0.0))
            dy = float(second.get("Y", 0.0)) - float(first.get("Y", 0.0))
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                return 0.0
            angle = math.degrees(math.atan2(dy, dx)) % 360.0
            nearest = round(angle / 90.0) * 90.0
            return nearest % 360.0 if abs(angle - nearest) <= 8.0 else angle
        except Exception:
            return 0.0

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
            source_font_size_estimate=font_height,
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
        self,
        blocks: Sequence[BlockRegion],
        canonical_cache: Optional[Dict[str, str]] = None,
        checkpoint_store: Optional[TranslationCheckpointStore] = None,
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
            total = len(source_order)
            processed = 0
            for start in range(0, len(source_order), self.batch_size):
                chunk = source_order[start : start + self.batch_size]
                chunk_keys = canonical_order[start : start + self.batch_size]
                chunk_translated = self.engine.translate_many(list(chunk))
                for canonical_key, source_text, translated_text in zip(
                    chunk_keys,
                    chunk,
                    chunk_translated,
                ):
                    final_text = translated_text or source_text
                    cache[canonical_key] = final_text
                    if checkpoint_store is not None and final_text:
                        checkpoint_store.upsert(
                            namespace="pdf-scanned:canonical",
                            block_id=canonical_key,
                            source_text=canonical_key,
                            translated_text=final_text,
                            extra={"sample_source_text": source_text},
                        )
                if checkpoint_store is not None and chunk_translated:
                    checkpoint_store.save()
                processed = min(total, processed + len(chunk_translated))
                percent = int((processed / total) * 100)
                self.log(f"Перевод блоков... [{percent}%]")

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
            if region.is_table_cell:
                if mask is None:
                    continue
            elif mask is None:
                mask = build_text_mask(roi, dilation_kernel=self.config.dilation_kernel_size)
            if mask is None:
                continue
            if region.is_table_cell and np.count_nonzero(mask) == 0:
                continue
            # Defensive: ensure mask matches ROI size and type.
            if mask.shape[:2] != roi.shape[:2]:
                mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            if mask.shape[:2] != roi.shape[:2]:
                continue
            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            inpaint_radius = max(
                3,
                min(5, int(math.ceil(self.config.dilation_kernel_size / 2.0)) + 2),
            )
            cleaned = cv2.inpaint(
                roi,
                mask,
                inpaint_radius,
                cv2.INPAINT_TELEA,
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
        if region.source_font_size_estimate <= 0:
            region.source_font_size_estimate = region.font_size_estimate
        dense_fit = region.fit_strategy in {"tight", "shrink", "preserve_small"} or region.semantic_type in {
            "map_label",
            "caption",
            "contact",
        }
        padding_ratio = 0.025 if dense_fit else (0.04 if region.is_table_cell else 0.06)
        padding = max(1, int(min(w, h) * padding_ratio))
        max_text_width = max(1, w - 2 * padding)
        max_text_height = max(1, h - 2 * padding)

        min_font_size = 5 if dense_fit or region.fit_strategy in {"wrap", "shrink"} else 8
        start_size = max(min_font_size, min(96, region.font_size_estimate))
        chosen_font: Optional["ImageFont.FreeTypeFont"] = None
        chosen_size = min_font_size
        wrapped_lines: List[str] = []
        line_spacing = 0
        for size in range(start_size, min_font_size - 1, -1):
            font = self.font_loader.get(size)
            lines = _wrap_text(text, draw, font, max_text_width)
            if not lines:
                continue
            line_height = _measure_text("Ay", draw, font)[1]
            spacing = max(0 if dense_fit else 1, int(line_height * (0.12 if dense_fit else 0.25)))
            total_height = len(lines) * line_height + max(0, len(lines) - 1) * spacing
            max_width = max(_measure_text(line, draw, font)[0] for line in lines)
            if total_height <= max_text_height and max_width <= max_text_width:
                chosen_font = font
                chosen_size = size
                wrapped_lines = lines
                line_spacing = spacing
                break
        if chosen_font is None:
            chosen_font = self.font_loader.get(min_font_size)
            chosen_size = min_font_size
            wrapped_lines = _wrap_text(text, draw, chosen_font, max_text_width)
            line_spacing = max(0 if dense_fit else 1, int(_measure_text("Ay", draw, chosen_font)[1] * 0.15))

        if region.text_color:
            b, g, r = region.text_color
            text_color = (r, g, b, 255)
        else:
            text_color = _sample_text_color(base_image, (x0, y0, x0 + w, y0 + h))
        stroke_w = 1 if region.is_bold else 0
        stroke_fill = text_color

        total_text_height = 0
        if wrapped_lines and chosen_font is not None:
            line_heights = [_measure_text(line or " ", draw, chosen_font)[1] for line in wrapped_lines]
            total_text_height = sum(line_heights) + max(0, len(wrapped_lines) - 1) * line_spacing

        if region.is_vertical:
            # Pick font size starting from the estimated size of the source block; shrink only if не помещается.
            layer_w, layer_h = h, w  # swapped canvas before rotation
            available_w = max(1, layer_w - 2 * padding)
            available_h = max(1, layer_h - 2 * padding)
            target_size = max(min_font_size, min(96, region.font_size_estimate))
            chosen_font = None
            char_lines: List[str] = []
            for paragraph in wrapped_lines:
                if not paragraph:
                    char_lines.append("")
                    continue
                # reversed so that after -90 rotation чтение идёт сверху вниз
                char_lines.extend(list(reversed(paragraph)))
                char_lines.append("")
            for size in range(target_size, min_font_size - 1, -1):
                font = self.font_loader.get(size)
                line_height = _measure_text("Ay", draw, font)[1]
                spacing = max(0 if dense_fit else 1, int(line_height * (0.12 if dense_fit else 0.25)))
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
                    chosen_size = size
                    line_spacing = spacing
                    break
            if chosen_font is None:
                chosen_font = self.font_loader.get(min_font_size)
                chosen_size = min_font_size
                line_spacing = max(0 if dense_fit else 1, int(_measure_text("Ay", draw, chosen_font)[1] * 0.15))
            region.rendered_font_size = chosen_size

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
            region.rendered_font_size = chosen_size
            region_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            region_draw = ImageDraw.Draw(region_layer)
            current_y = padding
            if region.is_table_cell and region.content_bbox:
                _cx, cy, _cw, ch = region.content_bbox
                target_y = cy + max(0, int((ch - total_text_height) / 2))
                current_y = max(padding, min(target_y, max(padding, h - padding - max(1, total_text_height))))
            for line_index, line in enumerate(wrapped_lines):
                line_width, line_height = _measure_text(line, region_draw, chosen_font)
                if region.is_table_cell and region.content_bbox:
                    cx, _cy, cw, _ch = region.content_bbox
                    if region.alignment == "center":
                        line_x = int(cx + (cw - line_width) / 2)
                    elif region.alignment == "right":
                        line_x = int(cx + cw - line_width)
                    else:
                        line_x = int(cx)
                    line_x = max(padding, min(line_x, max(padding, w - padding - line_width)))
                elif region.alignment == "center":
                    line_x = max(padding, int((w - line_width) / 2))
                elif region.alignment == "right":
                    line_x = max(padding, w - padding - line_width)
                else:
                    line_x = padding
                should_justify = (
                    region.alignment == "justify"
                    and line_index < len(wrapped_lines) - 1
                    and len(line.split()) > 1
                )
                if should_justify:
                    self._draw_justified_line(
                        region_draw,
                        line,
                        x=padding,
                        y=current_y,
                        width=max_text_width,
                        font=chosen_font,
                        fill=text_color,
                        stroke_width=stroke_w,
                        stroke_fill=stroke_fill,
                    )
                else:
                    region_draw.text(
                        (line_x, current_y),
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

    def _draw_justified_line(
        self,
        draw: "ImageDraw.ImageDraw",
        text: str,
        *,
        x: int,
        y: int,
        width: int,
        font: "ImageFont.FreeTypeFont",
        fill: Tuple[int, int, int, int],
        stroke_width: int,
        stroke_fill: Tuple[int, int, int, int],
    ) -> None:
        words = text.split()
        if len(words) <= 1:
            draw.text(
                (x, y),
                text,
                fill=fill,
                font=font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            return
        word_widths = [_measure_text(word, draw, font)[0] for word in words]
        total_words_width = sum(word_widths)
        gap = max(0.0, (width - total_words_width) / float(len(words) - 1))
        current_x = float(x)
        for word, word_width in zip(words, word_widths):
            draw.text(
                (int(round(current_x)), y),
                word,
                fill=fill,
                font=font,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            current_x += word_width + gap


class Exporter:
    """Exports processed page images back to a multi-page PDF (or image sequence)."""

    def __init__(self, log: PdfLog = _noop_log) -> None:
        self.log = log

    def to_pdf(self, images: Sequence["Image.Image"], output_path: Path, *, dpi: int) -> None:
        images_to_pdf(images, output_path, dpi=dpi, log=self.log)


def _iter_candidate_blocks(blocks: Sequence[BlockRegion]) -> Iterable[BlockRegion]:
    for block in blocks:
        if block.block_type in TRANSLATABLE_BLOCK_TYPES:
            yield block
        if block.cells:
            yield from _iter_candidate_blocks(block.cells)


def _iter_translatable_blocks(blocks: Sequence[BlockRegion]) -> Iterable[BlockRegion]:
    for block in _iter_candidate_blocks(blocks):
        if block.should_translate:
            yield block


def _stable_block_id(block: BlockRegion, page: Optional[PageImage]) -> str:
    page_width = max(1, page.width if page is not None else max(block.bbox[2], 1))
    page_height = max(1, page.height if page is not None else max(block.bbox[3], 1))
    normalized_bbox = (
        round(block.bbox[0] / page_width, 4),
        round(block.bbox[1] / page_height, 4),
        round(block.bbox[2] / page_width, 4),
        round(block.bbox[3] / page_height, 4),
    )
    identity = json.dumps(
        {
            "page": block.page_index,
            "type": block.block_type,
            "text": _canonicalize_text_for_dedup(block.source_text),
            "bbox": normalized_bbox,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _assign_render_ids_for_page(
    blocks: Sequence[BlockRegion],
    page: Optional[PageImage] = None,
) -> None:
    ordered = sorted(_iter_candidate_blocks(blocks), key=lambda b: (b.bbox[1], b.bbox[0]))
    for idx, block in enumerate(ordered):
        block.render_id = idx
        block.stable_id = _stable_block_id(block, page)


_EMAIL_RE = re.compile(
    r"^(?:(?:e|email)\s*[:.]?\s*)?[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"^(?:(?:tel|telephone|phone|fax|t|f)\s*[:.]?\s*)?"
    r"[+()\d][\d\s()+./-]{5,}$",
    re.IGNORECASE,
)


def _nontranslatable_text_reason(text: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return "empty"
    compact_lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if compact_lines and all(_EMAIL_RE.fullmatch(line) for line in compact_lines):
        return "email"
    if compact_lines and all(_PHONE_RE.fullmatch(line) for line in compact_lines):
        return "phone"
    alnum_count = sum(character.isalnum() for character in normalized)
    letter_count = sum(character.isalpha() for character in normalized)
    if alnum_count < 2:
        return "decorative"
    if letter_count == 0 and alnum_count < 4:
        return "decorative"
    symbols = sum(
        not character.isalnum() and not character.isspace()
        for character in normalized
    )
    if symbols > max(5, int(len(normalized) * 0.6)):
        return "decorative"
    return None


def _uppercase_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9&-]{2,}", str(text or "").upper())


def _mark_nontranslatable_blocks(
    page_layouts: Sequence[TextractPageData],
    log: PdfLog = _noop_log,
) -> None:
    candidates = [
        block
        for layout in page_layouts
        for block in _iter_candidate_blocks(layout.blocks)
    ]
    token_pages: Dict[str, Set[int]] = {}
    for block in candidates:
        for token in set(_uppercase_tokens(block.source_text)):
            token_pages.setdefault(token, set()).add(block.page_index)
    repeated_tokens = {
        token
        for token, pages in token_pages.items()
        if len(pages) >= 2 and len(token) >= 4
    }
    brand_tokens: Set[str] = set()
    for block in candidates:
        if block.parent_block_type != "FIGURE":
            continue
        tokens = _uppercase_tokens(block.source_text)
        letters = [character for character in str(block.source_text or "") if character.isalpha()]
        uppercase_ratio = (
            sum(character.isupper() for character in letters) / len(letters)
            if letters
            else 0.0
        )
        repeated_in_text = {token for token in tokens if token in repeated_tokens}
        if uppercase_ratio >= 0.9 and 2 <= len(tokens) <= 5 and len(repeated_in_text) >= 2:
            brand_tokens.update(repeated_in_text)

    skipped: Dict[str, int] = {}
    company_suffixes = {
        "CO",
        "COMPANY",
        "CORP",
        "CORPORATION",
        "INC",
        "LTD",
        "LLC",
    }
    logo_blocks: List[BlockRegion] = []
    for block in candidates:
        reason = _nontranslatable_text_reason(block.source_text)
        tokens = _uppercase_tokens(block.source_text)
        if reason is None and tokens and not block.is_table:
            uppercase_text = str(block.source_text or "").strip()
            letters = [character for character in uppercase_text if character.isalpha()]
            uppercase_ratio = (
                sum(character.isupper() for character in letters) / len(letters)
                if letters
                else 0.0
            )
            def is_brand_prefix(token: str, *, allow_repeated: bool) -> bool:
                pools: List[str] = list(brand_tokens)
                if allow_repeated:
                    pools.extend(repeated_tokens)
                return any(
                    len(token) >= 3 and len(repeated) > len(token) and repeated.startswith(token)
                    for repeated in pools
                )

            def is_brand_like(token: str, *, allow_repeated: bool) -> bool:
                return (
                    token in brand_tokens
                    or (allow_repeated and token in repeated_tokens)
                    or is_brand_prefix(token, allow_repeated=allow_repeated)
                    or token in company_suffixes
                )

            allow_repeated = block.parent_block_type != "FIGURE"
            repeated_or_prefix = all(
                is_brand_like(token, allow_repeated=allow_repeated)
                for token in tokens
            )
            brand_token_count = sum(
                1
                for token in tokens
                if is_brand_like(token, allow_repeated=True)
                and token not in company_suffixes
            )
            single_brand_fragment = (
                len(tokens) == 1
                and (
                    tokens[0] in brand_tokens
                    or is_brand_prefix(tokens[0], allow_repeated=True)
                )
            )
            if (
                uppercase_ratio >= 0.9
                and len(tokens) <= 4
                and repeated_or_prefix
                and (block.parent_block_type == "FIGURE" or brand_token_count >= 2 or single_brand_fragment)
            ):
                reason = "logo"
        if reason is None:
            continue
        block.should_translate = False
        block.render_enabled = False
        if reason == "logo":
            logo_blocks.append(block)
        skipped[reason] = skipped.get(reason, 0) + 1
    if logo_blocks:
        initial_logo_blocks = list(logo_blocks)

        def is_near_logo(candidate: BlockRegion, logo: BlockRegion) -> bool:
            if candidate.page_index != logo.page_index:
                return False
            text = str(candidate.source_text or "").strip()
            if not text or len(text.split()) > 4:
                return False
            bx0, by0, bx1, by1 = candidate.bbox
            lx0, ly0, lx1, ly1 = logo.bbox
            bw = max(1, bx1 - bx0)
            bh = max(1, by1 - by0)
            lw = max(1, lx1 - lx0)
            lh = max(1, ly1 - ly0)
            if bh > max(90, int(lh * 1.5)):
                return False
            vertical_gap = max(0, max(by0, ly0) - min(by1, ly1))
            center_gap = abs(((bx0 + bx1) / 2.0) - ((lx0 + lx1) / 2.0))
            max_width = max(bw, lw)
            return (
                vertical_gap <= max(80, int(lh * 0.9))
                and center_gap <= max(80, int(max_width * 0.35))
            )

        for block in candidates:
            if not block.render_enabled or block.is_table:
                continue
            if block.parent_block_type == "FIGURE":
                continue
            if any(is_near_logo(block, logo) for logo in initial_logo_blocks):
                block.should_translate = False
                block.render_enabled = False
                skipped["logo"] = skipped.get("logo", 0) + 1
    if skipped:
        details = ", ".join(f"{reason}: {count}" for reason, count in sorted(skipped.items()))
        log(f"Textract: исключены из перевода неизменяемые блоки ({details})")


def _disable_unchanged_blocks(blocks: Sequence[BlockRegion]) -> None:
    for block in _iter_translatable_blocks(blocks):
        source = _canonicalize_text_for_dedup(block.source_text)
        translated = _canonicalize_text_for_dedup(block.translated_text or "")
        if source and source == translated:
            block.render_enabled = False


def _apply_vision_block_updates(
    blocks: Sequence[BlockRegion],
    updates: Dict[str, VisionBlockUpdate],
) -> int:
    if not updates:
        return 0
    applied = 0
    for block in _iter_candidate_blocks(blocks):
        update = updates.get(block.stable_id)
        if update is None:
            continue
        corrected = str(update.corrected_text or "").strip()
        if corrected:
            block.corrected_text = corrected
            block.source_text = corrected
        block.semantic_type = update.semantic_type or "unknown"
        block.fit_strategy = update.fit_strategy or "default"
        if update.font_weight in {"auto", "normal", "bold"}:
            block.font_weight = update.font_weight
        if update.alignment in {"left", "center", "right", "justify"}:
            block.alignment = update.alignment
        block.vision_confidence = max(0.0, min(1.0, float(update.confidence or 0.0)))
        block.qa_flags = list(update.qa_flags)
        if not update.translate or block.semantic_type in {"brand", "contact", "decorative"}:
            block.should_translate = False
            block.render_enabled = False
        if block.semantic_type == "table_cell":
            block.is_table = True
        if block.semantic_type == "vertical" and block.fit_strategy == "default":
            block.fit_strategy = "vertical"
        applied += 1
    return applied


def _vision_cache_path(config: PdfProcessingConfig) -> Path:
    if config.vision_cache_path:
        return Path(config.vision_cache_path)
    return _CACHE_DIR / "vision_requests.json"


def _vision_mode_enabled(processing_mode: object) -> bool:
    return str(processing_mode or "").strip().lower() == PDF_PROCESSING_TEXTRACT_VISION


def _run_vision_layout_refinement(
    page_layouts: Sequence[TextractPageData],
    *,
    config: PdfProcessingConfig,
    openai_key: Optional[str],
    openai_base_url: Optional[str],
    openai_project: Optional[str],
    codex_cli_path: Optional[str],
    codex_timeout_seconds: int,
    log: PdfLog,
) -> None:
    if not page_layouts:
        return
    try:
        if config.vision_backend == "codex_cli":
            refiner = CodexCliVisionLayoutRefiner(
                cli_path=config.vision_codex_cli_path or codex_cli_path or "",
                model=config.vision_model or "gpt-5.5",
                reasoning_effort=config.vision_reasoning_effort or "medium",
                timeout_seconds=(
                    config.vision_codex_timeout_seconds
                    or codex_timeout_seconds
                ),
                cache_path=_vision_cache_path(config),
                request_mode=config.vision_request_mode,
                image_quality=config.vision_image_quality,
                log=log,
            )
        else:
            refiner = OpenAIVisionLayoutRefiner(
                api_key=config.vision_api_key or openai_key,
                model=config.vision_model or "gpt-5.5",
                base_url=config.vision_base_url or openai_base_url,
                project=config.vision_project or openai_project,
                reasoning_effort=config.vision_reasoning_effort or "medium",
                image_quality=config.vision_image_quality,
                cache_path=_vision_cache_path(config),
                log=log,
            )
    except Exception as exc:
        log(f"GPT Vision: layout refinement skipped ({exc})")
        return

    total = 0
    for layout in page_layouts:
        try:
            updates = refiner.refine_page(
                page_index=layout.page.page_index,
                image=layout.page.image,
                blocks=list(_iter_candidate_blocks(layout.blocks)),
            )
        except Exception as exc:
            log(f"GPT Vision: page {layout.page.page_index + 1} layout refinement failed ({exc})")
            continue
        applied = _apply_vision_block_updates(layout.blocks, updates)
        total += applied
        log(f"GPT Vision: page {layout.page.page_index + 1} refined blocks: {applied}")
    if total:
        log(f"GPT Vision: total refined blocks: {total}")


def _apply_vision_qa_issues(
    regions: Sequence[RegionInfo],
    issues: Sequence[VisionQaIssue],
) -> bool:
    if not issues:
        return False
    regions_by_id: Dict[str, List[RegionInfo]] = {}
    for region in regions:
        if region.stable_id:
            regions_by_id.setdefault(region.stable_id, []).append(region)
    changed = False
    deterministic_actions = {
        "shrink_text",
        "wrap_text",
        "reduce_padding",
        "preserve_font_size",
        "set_bold",
        "set_normal",
        "align_left",
        "align_center",
        "align_right",
        "align_justify",
    }
    for issue in issues:
        target_regions = regions_by_id.get(issue.stable_id, [])
        for region in target_regions:
            flag = issue.issue_type or issue.action
            if flag and flag not in region.qa_flags:
                region.qa_flags.append(flag)
            if issue.action not in deterministic_actions or issue.confidence < 0.45:
                continue
            if issue.action == "wrap_text":
                region.fit_strategy = "wrap"
            elif issue.action == "reduce_padding":
                region.fit_strategy = "tight"
            elif issue.action == "preserve_font_size":
                region.fit_strategy = "tight"
                region.font_size_estimate = max(
                    region.font_size_estimate,
                    region.source_font_size_estimate or region.font_size_estimate,
                )
            elif issue.action == "set_bold":
                region.font_weight = "bold"
                region.is_bold = True
            elif issue.action == "set_normal":
                region.font_weight = "normal"
                region.is_bold = False
            elif issue.action.startswith("align_"):
                alignment = issue.action.removeprefix("align_")
                if alignment in {"left", "center", "right", "justify"}:
                    region.alignment = alignment
            elif issue.action == "shrink_text":
                region.fit_strategy = "shrink"
                region.font_size_estimate = max(
                    5, int(region.font_size_estimate * 0.85)
                )
            changed = True
    return changed


def _make_vision_qa_reviewer(
    *,
    config: PdfProcessingConfig,
    openai_key: Optional[str],
    openai_base_url: Optional[str],
    openai_project: Optional[str],
    codex_cli_path: Optional[str],
    codex_timeout_seconds: int,
    log: PdfLog,
) -> Optional[object]:
    try:
        if config.vision_backend == "codex_cli":
            return CodexCliVisionQaReviewer(
                cli_path=config.vision_codex_cli_path or codex_cli_path or "",
                model=config.vision_model or "gpt-5.5",
                reasoning_effort=config.vision_reasoning_effort or "medium",
                timeout_seconds=(
                    config.vision_codex_timeout_seconds
                    or codex_timeout_seconds
                ),
                cache_path=_vision_cache_path(config),
                request_mode=config.vision_request_mode,
                image_quality=config.vision_image_quality,
                log=log,
            )
        return OpenAIVisionQaReviewer(
            api_key=config.vision_api_key or openai_key,
            model=config.vision_model or "gpt-5.5",
            base_url=config.vision_base_url or openai_base_url,
            project=config.vision_project or openai_project,
            reasoning_effort=config.vision_reasoning_effort or "medium",
            image_quality=config.vision_image_quality,
            cache_path=_vision_cache_path(config),
            log=log,
        )
    except Exception as exc:
        log(f"GPT Vision: QA skipped ({exc})")
        return None

def _apply_cached_translations(blocks: Sequence[BlockRegion], existing: Dict[Tuple[int, int], str]) -> None:
    for block in _iter_candidate_blocks(blocks):
        if block.translated_text:
            continue
        if block.render_id is None:
            continue
        cached_value = str(existing.get((block.page_index, block.render_id), "") or "").strip()
        if cached_value:
            block.translated_text = cached_value


def _apply_stable_cached_translations(
    blocks: Sequence[BlockRegion],
    existing: Dict[str, str],
) -> None:
    if not existing:
        return
    for block in _iter_candidate_blocks(blocks):
        if not block.stable_id:
            continue
        cached_value = str(existing.get(block.stable_id, "") or "").strip()
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
    word_heights: List[int] = []
    if block.word_boxes:
        word_heights = [int(h) for _, _, _, h in block.word_boxes if h > 0]
    elif block.lines:
        for line in block.lines:
            if line.word_boxes:
                word_heights.extend([int(h) for _, _, _, h in line.word_boxes if h > 0])
    if word_heights:
        word_heights.sort()
        median = word_heights[len(word_heights) // 2]
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


def _collect_block_word_boxes(block: BlockRegion) -> List[BBoxWH]:
    boxes = list(block.word_boxes)
    for line in block.lines:
        boxes.extend(line.word_boxes)
    unique: List[BBoxWH] = []
    seen: Set[BBoxWH] = set()
    for box in boxes:
        normalized = tuple(int(value) for value in box)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _block_content_bbox(block: BlockRegion) -> Optional[Tuple[int, int, int, int]]:
    word_boxes = _collect_block_word_boxes(block)
    if word_boxes:
        return _union_bbox([(x, y, x + w, y + h) for x, y, w, h in word_boxes])
    if block.lines:
        return _union_bbox([line.bbox for line in block.lines])
    return None


def _bounded_block_bbox(
    block: BlockRegion,
    candidates: Sequence[BlockRegion],
    config: PdfProcessingConfig,
    page_width: int,
    page_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    clamped = _clamp_bbox_to_page(block.bbox, page_width, page_height)
    if clamped is None:
        return None
    x0, y0, x1, y1 = clamped
    if block.is_table:
        return clamped

    loose_parent = _needs_large_heading_recovery(block)
    parent = (
        (0, 0, page_width, page_height)
        if loose_parent
        else _clamp_bbox_to_page(
            block.parent_bbox or (0, 0, page_width, page_height),
            page_width,
            page_height,
        )
        or (0, 0, page_width, page_height)
    )
    left_limit, top_limit, right_limit, bottom_limit = parent
    block_width = max(1, x1 - x0)
    block_height = max(1, y1 - y0)

    for neighbor in candidates:
        if neighbor is block or not neighbor.render_enabled:
            continue
        nx0, ny0, nx1, ny1 = neighbor.bbox
        vertical_overlap = max(0, min(y1, ny1) - max(y0, ny0))
        horizontal_overlap = max(0, min(x1, nx1) - max(x0, nx0))
        min_height = max(1, min(block_height, ny1 - ny0))
        min_width = max(1, min(block_width, nx1 - nx0))
        if vertical_overlap / min_height >= 0.3:
            if nx1 <= x0:
                left_limit = max(left_limit, nx1)
            elif nx0 >= x1:
                right_limit = min(right_limit, nx0)
        if horizontal_overlap / min_width >= 0.3:
            if ny1 <= y0:
                top_limit = max(top_limit, ny1)
            elif ny0 >= y1:
                bottom_limit = min(bottom_limit, ny0)

    extra_x = int(block_width * max(0.0, min(0.5, config.textract_bbox_width_boost)))
    extra_y = int(block_height * max(0.0, min(0.5, config.textract_bbox_height_boost)))
    if loose_parent:
        extra_x = max(extra_x, int(block_width * 1.4), int(block_height * 0.8))
    left_extra = min(extra_x // 4, int(block_width * 0.15)) if loose_parent else extra_x // 2
    top_extra = extra_y // 2
    expanded = (
        max(left_limit, x0 - left_extra),
        max(top_limit, y0 - top_extra),
        min(right_limit, x1 + (extra_x - left_extra)),
        min(bottom_limit, y1 + (extra_y - top_extra)),
    )
    return _clamp_bbox_to_page(expanded, page_width, page_height)


def _needs_large_heading_recovery(block: BlockRegion) -> bool:
    if block.parent_block_type == "FIGURE":
        return False
    text = str(block.source_text or "").strip()
    if not text:
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    if block.block_type == "TITLE" and uppercase_ratio >= 0.6:
        return True
    heights = [line.bbox[3] - line.bbox[1] for line in block.lines if line.bbox[3] > line.bbox[1]]
    median_height = _median_float(heights, 0.0)
    return median_height >= 55.0 or (uppercase_ratio >= 0.75 and median_height >= 35.0)


def _trim_recovery_bbox_to_light_background(
    bgr_page: "np.ndarray",
    expanded: Tuple[int, int, int, int],
    original: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = expanded
    ox0, _oy0, ox1, _oy1 = original
    if x1 <= x0 or y1 <= y0 or bgr_page.size == 0:
        return expanded
    height, width = bgr_page.shape[:2]
    y0 = max(0, min(height, y0))
    y1 = max(y0 + 1, min(height, y1))
    x0 = max(0, min(width, x0))
    x1 = max(x0 + 1, min(width, x1))
    ox0 = max(x0, min(x1, ox0))
    ox1 = max(x0, min(x1, ox1))
    step = max(3, min(12, (y1 - y0) // 20 or 3))

    def is_light_column(x: int) -> bool:
        left = max(0, min(width - 1, x))
        right = min(width, left + max(2, step))
        column = bgr_page[y0:y1, left:right]
        if column.size == 0:
            return True
        sample = column.reshape(-1, 3)
        b = sample[:, 0].astype(float)
        g = sample[:, 1].astype(float)
        r = sample[:, 2].astype(float)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        saturation = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        paper_pixels = (brightness >= 205.0) & (saturation <= 80.0)
        return float(np.mean(paper_pixels)) >= 0.35

    trimmed_left = x0
    probe = ox0
    while probe > x0:
        if not is_light_column(probe - step):
            look = probe - step
            found_light = False
            lower_bound = max(x0, probe - step * 12)
            while look > lower_bound:
                if is_light_column(look):
                    found_light = True
                    break
                look -= step
            if not found_light:
                trimmed_left = max(x0, probe)
                break
            trimmed_left = max(x0, look)
            probe = look
            continue
        trimmed_left = max(x0, probe - step)
        probe -= step

    trimmed_right = x1
    probe = ox1
    while probe < x1:
        if not is_light_column(probe):
            look = probe + step
            found_light = False
            upper_bound = min(x1, probe + step * 12)
            while look < upper_bound:
                if is_light_column(look):
                    found_light = True
                    break
                look += step
            if not found_light:
                trimmed_right = min(x1, probe)
                break
            trimmed_right = min(x1, look + step)
            probe = look + step
            continue
        trimmed_right = min(x1, probe + step)
        probe += step

    if trimmed_right <= trimmed_left:
        return expanded
    return (trimmed_left, y0, trimmed_right, y1)


def _block_to_region_info(
    block: BlockRegion,
    bgr_page: "np.ndarray",
    config: PdfProcessingConfig,
    page_width: int,
    page_height: int,
    candidates: Sequence[BlockRegion],
) -> Optional[RegionInfo]:
    clamped = _bounded_block_bbox(
        block,
        candidates,
        config,
        page_width,
        page_height,
    )
    if not clamped:
        return None
    if _needs_large_heading_recovery(block):
        clamped = _trim_recovery_bbox_to_light_background(bgr_page, clamped, block.bbox)
    x0, y0, x1, y1 = clamped
    is_table = bool(block.is_table)
    roi = bgr_page[y0:y1, x0:x1].copy()
    if roi.size == 0:
        return None
    word_boxes = _collect_block_word_boxes(block)
    mask_for_cleaning = build_text_mask_from_word_boxes(
        roi,
        word_boxes,
        offset_x=x0,
        offset_y=y0,
        dilation_kernel=config.dilation_kernel_size,
    )
    mask_for_color = build_text_mask_from_word_boxes(
        roi,
        word_boxes,
        offset_x=x0,
        offset_y=y0,
        dilation_kernel=1,
    )
    if mask_for_cleaning is None:
        mask_for_cleaning = build_text_mask(
            roi,
            dilation_kernel=config.dilation_kernel_size,
        )
    if mask_for_color is None:
        mask_for_color = build_text_mask(roi, dilation_kernel=1)
    if mask_for_color is None:
        mask_for_color = np.zeros(roi.shape[:2], dtype=np.uint8)
    if _needs_large_heading_recovery(block):
        fallback_mask = build_text_mask(roi, dilation_kernel=config.dilation_kernel_size)
        if fallback_mask is not None:
            mask_for_cleaning = fallback_mask if mask_for_cleaning is None else cv2.bitwise_or(mask_for_cleaning, fallback_mask)
        fallback_color_mask = build_text_mask(roi, dilation_kernel=1)
        if fallback_color_mask is not None:
            mask_for_color = cv2.bitwise_or(mask_for_color, fallback_color_mask)
    bg_color = _estimate_background_color(roi, mask_for_color)
    raw_text_color = _estimate_text_color(
        roi, mask_for_color if mask_for_color is not None else np.ones(roi.shape[:2], dtype=np.uint8)
    )
    rotation = float(block.rotation_deg or 0.0) % 360.0
    signed_rotation = rotation if rotation <= 180.0 else rotation - 360.0
    is_vertical = abs(abs(signed_rotation) - 90.0) <= 12.0
    if block.semantic_type == "vertical" or block.fit_strategy == "vertical":
        is_vertical = True
        signed_rotation = 90.0 if abs(signed_rotation) < 1e-1 else signed_rotation
    is_bold = False
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    aspect = height / float(width)
    if not is_vertical and aspect >= config.textract_vertical_aspect_ratio:
        is_vertical = True
        signed_rotation = 90.0
    if mask_for_color is not None and mask_for_color.size >= 400:
        fill_ratio = float(np.count_nonzero(mask_for_color)) / float(mask_for_color.size)
        if fill_ratio >= max(0.05, min(0.9, config.text_bold_fill_ratio)):
            is_bold = True
    if block.font_weight == "bold":
        is_bold = True
    elif block.font_weight == "normal":
        is_bold = False
    boosted_text_color = _boost_text_contrast(raw_text_color, bg_color)
    font_size = _estimate_font_size_from_block(block)
    if block.semantic_type in {"map_label", "caption", "contact"}:
        if block.fit_strategy == "default":
            block.fit_strategy = "preserve_small"
    content_bbox: Optional[BBoxWH] = None
    content_abs = _block_content_bbox(block)
    if content_abs:
        cx0, cy0, cx1, cy1 = content_abs
        clipped = _clamp_bbox_to_page(
            (
                max(x0, cx0),
                max(y0, cy0),
                min(x1, cx1),
                min(y1, cy1),
            ),
            page_width,
            page_height,
        )
        if clipped:
            content_bbox = (
                clipped[0] - x0,
                clipped[1] - y0,
                max(1, clipped[2] - clipped[0]),
                max(1, clipped[3] - clipped[1]),
            )
    return RegionInfo(
        page_index=block.page_index,
        region_id=int(block.render_id if block.render_id is not None else 0),
        bbox=(x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
        text_orientation_deg=-signed_rotation if is_vertical else -signed_rotation,
        text_color=boosted_text_color,
        background_color=bg_color,
        source_text=block.source_text,
        translated_text=block.translated_text or block.source_text,
        mask=mask_for_cleaning,
        font_size_estimate=font_size,
        source_font_size_estimate=font_size,
        is_vertical=is_vertical,
        is_bold=is_bold,
        font_weight=block.font_weight,
        is_table_cell=is_table,
        stable_id=block.stable_id,
        alignment=block.alignment,
        confidence=block.confidence,
        parent_block_id=block.parent_block_id,
        content_bbox=content_bbox,
        corrected_text=block.corrected_text,
        semantic_type=block.semantic_type,
        fit_strategy=block.fit_strategy,
        vision_confidence=block.vision_confidence,
        qa_flags=list(block.qa_flags),
    )


def _build_regions_from_blocks(
    page: PageImage, blocks: Sequence[BlockRegion], config: PdfProcessingConfig, log: PdfLog = _noop_log
) -> List[RegionInfo]:
    _require_dependency(cv2, "Установите opencv-python (pip install opencv-python)")
    bgr_page = cv2.cvtColor(np.array(page.image.convert("RGB")), cv2.COLOR_RGB2BGR)
    regions: List[RegionInfo] = []
    seen_render_ids: Set[int] = set()
    seen_by_text: Dict[str, List[RegionInfo]] = {}
    candidates = list(_iter_candidate_blocks(blocks))
    for block in _iter_translatable_blocks(blocks):
        if not block.render_enabled:
            continue
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
        region = _block_to_region_info(
            block,
            bgr_page,
            config,
            page.width,
            page.height,
            candidates,
        )
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
    page_layouts: List[TextractPageData] = []

    for page in pages:
        logger(f"PDF: обработка страницы {page.page_index + 1}/{len(pages)}")
        page_layout = textract.analyze(page)
        _assign_render_ids_for_page(page_layout.blocks, page)
        page_layouts.append(page_layout)

    _mark_nontranslatable_blocks(page_layouts, logger)
    for page_layout in page_layouts:
        page = page_layout.page
        for block in _iter_translatable_blocks(page_layout.blocks):
            block.translated_text = block.source_text
        regions = _build_regions_from_blocks(page, page_layout.blocks, cfg, log=logger)
        all_regions.extend(regions)
        processed_image = renderer.render_page(page, regions)
        processed_images.append(processed_image)

    if write_blurred_pdf:
        Exporter(logger).to_pdf(processed_images, output_path, dpi=cfg.render_dpi)
        logger(f"PDF: файл сохранён: {output_path}")

    return PdfProcessingResult(
        blurred_pdf_path=output_path,
        regions=all_regions,
        processed_images=processed_images,
    )


def _translate_scanned_pdf(
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
    checkpoint_path: Optional[Path] = None,
    resume_policy: str = "auto",
) -> Dict[str, object]:
    """
    High-quality PDF translation pipeline driven purely by Textract layout blocks.
    """
    _ = pdf_type
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
            textract_bbox_height_boost=processing_options.get(
                "textract_bbox_height_boost", config.textract_bbox_height_boost
            ),
            textract_bbox_width_boost=processing_options.get(
                "textract_bbox_width_boost", config.textract_bbox_width_boost
            ),
            vision_backend=processing_options.get(
                "vision_backend", config.vision_backend
            ),
            vision_model=processing_options.get("vision_model", config.vision_model),
            vision_reasoning_effort=processing_options.get(
                "vision_reasoning_effort", config.vision_reasoning_effort
            ),
            vision_api_key=processing_options.get(
                "vision_api_key", config.vision_api_key
            ),
            vision_base_url=processing_options.get(
                "vision_base_url", config.vision_base_url
            ),
            vision_project=processing_options.get(
                "vision_project", config.vision_project
            ),
            vision_codex_cli_path=processing_options.get(
                "vision_codex_cli_path", config.vision_codex_cli_path
            ),
            vision_codex_timeout_seconds=processing_options.get(
                "vision_codex_timeout_seconds",
                config.vision_codex_timeout_seconds,
            ),
            vision_request_mode=processing_options.get(
                "vision_request_mode",
                config.vision_request_mode,
            ),
            vision_image_quality=processing_options.get(
                "vision_image_quality",
                config.vision_image_quality,
            ),
            vision_cache_path=processing_options.get("vision_cache_path", config.vision_cache_path),
        ).normalized()

    if layer_json_path is None:
        layer_json_path = _CACHE_DIR / f"{input_path.stem}.translation.json"

    file_hash = _compute_file_sha256(input_path)
    checkpoint_store = TranslationCheckpointStore(
        path=checkpoint_path or default_checkpoint_path(input_path, "pdf-scanned"),
        job_type="pdf-scanned",
        document_sha256=file_hash,
        source_lang=source_lang,
        target_lang=target_lang,
        translator=translator_name,
        resume_policy=resume_policy,
        log=log,
    )
    pdf_processing_mode = (processing_options.get("pdf_processing_mode", "textract") if processing_options else "textract") or "textract"
    use_vision_mode = _vision_mode_enabled(pdf_processing_mode)
    textract_client_override: Optional[object] = None
    if str(pdf_processing_mode).lower() in {PDF_PROCESSING_TEXTRACT, PDF_PROCESSING_TEXTRACT_VISION}:
        region_name = processing_options.get("textract_region") if processing_options else None
        access_key = processing_options.get("textract_access_key") if processing_options else None
        secret_key = processing_options.get("textract_secret_key") if processing_options else None
        session_token = processing_options.get("textract_session_token") if processing_options else None
        if access_key and secret_key:
            _require_dependency(boto3, "Установите boto3 (pip install boto3) для Textract")
            try:
                textract_client_override = boto3.client(
                    "textract",
                    region_name=region_name or None,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    aws_session_token=session_token or None,
                )
            except Exception as exc:
                raise RuntimeError(f"Textract: не удалось создать клиент с заданными ключами: {exc}") from exc

    loader = PageLoader(config, log)
    renderer = Renderer(
        config,
        log,
        style_font=_resolve_scanned_style_font(style_font, log),
    )
    exporter = Exporter(log)
    textract = TextractLayoutExtractor(
        textract_client_override,
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
        _assign_render_ids_for_page(layout.blocks, page)
        page_layouts.append(layout)

    if use_vision_mode:
        _run_vision_layout_refinement(
            page_layouts,
            config=config,
            openai_key=openai_key,
            openai_base_url=openai_base_url,
            openai_project=openai_project,
            codex_cli_path=codex_cli_path,
            codex_timeout_seconds=codex_timeout_seconds,
            log=log,
        )

    _mark_nontranslatable_blocks(page_layouts, log)
    all_blocks: List[BlockRegion] = []
    for layout in page_layouts:
        all_blocks.extend(list(_iter_translatable_blocks(layout.blocks)))

    if not all_blocks:
        raise RuntimeError("Не удалось извлечь текст из PDF/изображения")

    existing_map: Dict[Tuple[int, int], str] = {}
    existing_stable_map: Dict[str, str] = {}
    checkpoint_canonicals = checkpoint_store.text_map("pdf-scanned:canonical")
    existing_canonicals: Dict[str, str] = dict(checkpoint_canonicals)
    if checkpoint_canonicals:
        log(f"PDF: checkpoint loaded canonical translations: {len(checkpoint_canonicals)}")
    if layer_json_path and layer_json_path.exists():
        existing_map = load_translation_mapping(layer_json_path, log)
        existing_stable_map = load_translation_stable_mapping(layer_json_path, log)
        layer_canonicals = load_translation_canonicals(layer_json_path, log)
        for key, value in layer_canonicals.items():
            existing_canonicals.setdefault(key, value)
        cached_count = len(existing_stable_map) or len(existing_map)
        if cached_count:
            log(f"PDF: найден готовый слой перевода, используем без обращения к API ({cached_count} областей)")

    for layout in page_layouts:
        _apply_stable_cached_translations(layout.blocks, existing_stable_map)
        _apply_cached_translations(layout.blocks, existing_map)
        _apply_canonical_map(layout.blocks, existing_canonicals)

    backend_name = "cached-checkpoint" if checkpoint_canonicals else "cached-layer"
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
        "All texts belong to the SAME document — maintain consistent terminology across all items. "
        "Ignore OCR artifacts such as random mixed casing, stray spaces inside words, and line-break hyphenation. "
        "Only remove a hyphen when it clearly indicates a line-break split (hyphen followed by whitespace/newline); "
        "keep true compound-word hyphens. Use natural casing and spacing in the target language, but keep company "
        "names, brand names, product/model names, part codes, and addresses exactly as in the source. "
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
            system_prompt_template=pdf_prompt_template,
        )
        all_source_texts = [
            str(b.source_text or "").strip()
            for b in all_blocks
            if b.block_type in TRANSLATABLE_BLOCK_TYPES and str(b.source_text or "").strip()
        ]
        translator.set_drawing_context(all_source_texts)
        backend_name = translator.backend_name()
        checkpoint_store.set_backend(backend_name)
        log(f"Инициализирован движок перевода: {backend_name}")

        log(
            f"Готовим {len(missing_canonicals)} уникальных блоков текста к переводу "
            f"(из {pending_block_count} без готового перевода)"
        )
        canonical_cache = BlockRegionTranslator(translator, log).translate(
            all_blocks,
            canonical_cache,
            checkpoint_store=checkpoint_store,
        )

    _apply_canonical_translations(all_blocks, canonical_cache)
    for layout in page_layouts:
        _disable_unchanged_blocks(layout.blocks)

    for block in all_blocks:
        if block.render_id is not None and block.translated_text:
            existing_map[(block.page_index, block.render_id)] = block.translated_text

    page_regions: Dict[int, List[RegionInfo]] = {}
    all_regions: List[RegionInfo] = []
    for layout in page_layouts:
        regions = _build_regions_from_blocks(layout.page, layout.blocks, config, log=log)
        page_regions[layout.page.page_index] = regions
        all_regions.extend(regions)

    rendered_images: List["Image.Image"] = []
    layout_warnings: List[str] = []
    vision_reviewer = (
        _make_vision_qa_reviewer(
            config=config,
            openai_key=openai_key,
            openai_base_url=openai_base_url,
            openai_project=openai_project,
            codex_cli_path=codex_cli_path,
            codex_timeout_seconds=codex_timeout_seconds,
            log=log,
        )
        if use_vision_mode
        else None
    )
    for layout in page_layouts:
        regions = page_regions.get(layout.page.page_index, [])
        rendered = renderer.render_page(layout.page, regions)
        if vision_reviewer is not None and regions:
            try:
                qa_issues = vision_reviewer.review_page(
                    page_index=layout.page.page_index,
                    source_image=layout.page.image,
                    rendered_image=rendered,
                    regions=regions,
                )
            except Exception as exc:
                log(f"GPT Vision: page {layout.page.page_index + 1} QA failed ({exc})")
                qa_issues = []
            for issue in qa_issues:
                warning = (
                    f"page={layout.page.page_index + 1} stable_id={issue.stable_id} "
                    f"issue={issue.issue_type} action={issue.action}: {issue.message}"
                )
                layout_warnings.append(warning)
            if _apply_vision_qa_issues(regions, qa_issues):
                log(f"GPT Vision: page {layout.page.page_index + 1} rerendered after QA adjustments")
                rendered = renderer.render_page(layout.page, regions)
        rendered_images.append(rendered)

    if layer_json_path:
        save_translation_mapping(all_regions, layer_json_path, log)

    exporter.to_pdf(rendered_images, output_path, dpi=config.render_dpi)
    log(f"PDF: файл сохранён: {output_path}")

    return {
        "output_path": output_path,
        "backend": backend_name,
        "pages": len(rendered_images),
        "job_type": "pdf",
        "processing_mode": PDF_PROCESSING_TEXTRACT_VISION if use_vision_mode else "scanned",
        "native_pages": [],
        "ocr_pages": list(range(1, len(rendered_images) + 1)),
        "ocr_performed": True,
        "translated_blocks": len(all_blocks),
        "layout_warnings": layout_warnings,
        "checkpoint_path": checkpoint_store.path,
    }


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
    checkpoint_path: Optional[Path] = None,
    resume_policy: str = "auto",
) -> Dict[str, object]:
    """Route searchable PDFs to the native pipeline and scans to OCR/Textract."""

    if str(translator_name or "").strip().lower() == "codex":
        codex_analysis_session = codex_analysis_session or CodexAnalysisSession()

    normalized_type = str(pdf_type or PDF_TYPE_SCANNED).strip().lower()
    common_args = {
        "input_path": input_path,
        "output_path": output_path,
        "translator_name": translator_name,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "log": log,
        "layer_json_path": layer_json_path,
        "processing_options": processing_options,
        "style_font": style_font,
        "deepl_key": deepl_key,
        "openai_key": openai_key,
        "openai_model": openai_model,
        "openai_base_url": openai_base_url,
        "openai_project": openai_project,
        "openai_temperature": openai_temperature,
        "openai_reasoning_effort": openai_reasoning_effort,
        "openai_verbosity": openai_verbosity,
        "openai_strict_mode": openai_strict_mode,
        "openai_strict_value": openai_strict_value,
        "codex_cli_path": codex_cli_path,
        "codex_model": codex_model,
        "codex_reasoning_effort": codex_reasoning_effort,
        "codex_analysis_model": codex_analysis_model,
        "codex_analysis_reasoning_effort": codex_analysis_reasoning_effort,
        "codex_analysis_session": codex_analysis_session,
        "codex_timeout_seconds": codex_timeout_seconds,
        "checkpoint_path": checkpoint_path,
        "resume_policy": resume_policy,
    }
    if normalized_type != PDF_TYPE_NATIVE:
        return _translate_scanned_pdf(pdf_type=PDF_TYPE_SCANNED, **common_args)

    from .native import translate_native_pdf

    def translate_fallback_page(page_index: int) -> bytes:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Для гибридной обработки PDF требуется PyMuPDF>=1.27.2.2"
            ) from exc

        import tempfile

        log(f"PDF: запускаем OCR fallback для страницы {page_index + 1}")
        with tempfile.TemporaryDirectory(prefix="daru-pdf-fallback-") as temp_dir:
            temp_root = Path(temp_dir)
            fallback_input = temp_root / f"page-{page_index + 1}.pdf"
            fallback_output = temp_root / f"page-{page_index + 1}-translated.pdf"
            fallback_layer = temp_root / f"page-{page_index + 1}.translation.json"

            source_document = fitz.open(str(input_path))
            one_page = fitz.open()
            try:
                one_page.insert_pdf(
                    source_document,
                    from_page=page_index,
                    to_page=page_index,
                )
                one_page.save(str(fallback_input), garbage=4, deflate=True)
            finally:
                one_page.close()
                source_document.close()

            fallback_args = dict(common_args)
            fallback_args.update(
                {
                    "input_path": fallback_input,
                    "output_path": fallback_output,
                    "layer_json_path": fallback_layer,
                    "checkpoint_path": default_checkpoint_path(
                        input_path,
                        f"pdf-fallback-{page_index + 1}",
                    ),
                }
            )
            _translate_scanned_pdf(pdf_type=PDF_TYPE_SCANNED, **fallback_args)
            return fallback_output.read_bytes()

    native_args = dict(common_args)
    native_args.pop("processing_options", None)
    native_args["fallback_page_translator"] = translate_fallback_page
    return translate_native_pdf(**native_args)


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


def build_text_mask_from_word_boxes(
    roi: "np.ndarray",
    word_boxes: Sequence[BBoxWH],
    *,
    offset_x: int,
    offset_y: int,
    dilation_kernel: int,
) -> Optional["np.ndarray"]:
    """Construct a glyph-level mask constrained by Textract word boxes."""
    _require_dependency(cv2, "opencv-python (pip install opencv-python)")
    if roi.size == 0 or not word_boxes:
        return None
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for wx, wy, ww, wh in word_boxes:
        rx0 = max(0, min(roi.shape[1], int(wx - offset_x)))
        ry0 = max(0, min(roi.shape[0], int(wy - offset_y)))
        rx1 = max(0, min(roi.shape[1], int(wx + ww - offset_x)))
        ry1 = max(0, min(roi.shape[0], int(wy + wh - offset_y)))
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        patch = roi[ry0:ry1, rx0:rx1]
        if patch.size == 0:
            continue
        border = np.concatenate(
            (
                patch[0, :, :],
                patch[-1, :, :],
                patch[:, 0, :],
                patch[:, -1, :],
            ),
            axis=0,
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(
            patch.astype(np.float32) - background.astype(np.float32),
            axis=2,
        )
        distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
        if np.max(distance_u8) <= 8:
            continue
        otsu_threshold, local_mask = cv2.threshold(
            distance_u8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        if otsu_threshold < 8:
            local_mask = np.where(distance_u8 >= 10, 255, 0).astype(np.uint8)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        threshold_inverse = 255 - threshold
        threshold_ratio = float(np.count_nonzero(threshold)) / float(threshold.size)
        inverse_ratio = float(np.count_nonzero(threshold_inverse)) / float(threshold_inverse.size)
        usable_threshold = 0.003 <= threshold_ratio <= 0.68
        usable_inverse = 0.003 <= inverse_ratio <= 0.68
        if usable_threshold and usable_inverse:
            gray_mask = threshold if threshold_ratio <= inverse_ratio else threshold_inverse
        elif usable_threshold:
            gray_mask = threshold
        elif usable_inverse:
            gray_mask = threshold_inverse
        else:
            gray_mask = local_mask
        combined = cv2.bitwise_or(local_mask, gray_mask)
        combined_ratio = float(np.count_nonzero(combined)) / float(combined.size)
        if combined_ratio <= 0.72:
            local_mask = combined
        local_size = max(
            3,
            min(
                9,
                int(round(min(patch.shape[0], patch.shape[1]) * 0.08)) | 1,
            ),
        )
        local_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (local_size, local_size),
        )
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, local_kernel)
        local_mask = cv2.dilate(local_mask, local_kernel, iterations=1)
        target = mask[ry0:ry1, rx0:rx1]
        mask[ry0:ry1, rx0:rx1] = cv2.bitwise_or(target, local_mask)
    if np.count_nonzero(mask) == 0:
        return None
    word_heights = [max(1, int(height)) for _, _, _, height in word_boxes if height > 0]
    median_word_height = _median_float(word_heights, 1.0)
    shadow_kernel = int(round(median_word_height * 0.05)) | 1
    kernel_size = max(
        1,
        int(dilation_kernel),
        min(9, shadow_kernel),
    )
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    shadow_offset = max(1, min(8, int(round(median_word_height * 0.05))))
    if shadow_offset < mask.shape[0] and shadow_offset < mask.shape[1]:
        shifted = np.zeros_like(mask)
        shifted[shadow_offset:, shadow_offset:] = mask[:-shadow_offset, :-shadow_offset]
        mask = cv2.bitwise_or(mask, shifted)
    return mask


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
    """Load legacy page/render-id mappings from any supported layer version."""
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


def load_translation_stable_mapping(path: Path, log: PdfLog) -> Dict[str, str]:
    """Load stable block-id mappings from a version 2 scanned-PDF layer."""
    mapping: Dict[str, str] = {}
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
        blocks = page.get("blocks") or page.get("regions") or []
        for block in blocks:
            stable_id = str(block.get("stable_id", "") or "").strip()
            translated_text = str(block.get("translated_text", "") or "").strip()
            if stable_id and translated_text:
                mapping[stable_id] = translated_text
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
                corrected_text = str(block.get("corrected_text", "") or "").strip()
                translated_text = str(block.get("translated_text", "") or "").strip()
                if not source_text or not translated_text:
                    continue
                for candidate_source in (source_text, corrected_text):
                    canonical = _canonicalize_text_for_dedup(candidate_source)
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
        self._windows_fonts_dir = self._resolve_windows_fonts_dir()
        self._preferred_font_path = self._resolve_font_path()

    def get(self, size: int) -> "ImageFont.FreeTypeFont":
        size = max(5, min(128, int(size)))
        cached = self._cache.get(size)
        if cached is not None:
            return cached
        font_paths = []
        if self.font_value:
            requested = Path(self.font_value).expanduser()
            font_paths.append(requested)
            if self._windows_fonts_dir and not requested.exists():
                resolved = self._resolve_windows_font_name(requested.name)
                if resolved is not None:
                    font_paths.append(resolved)
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

    def _resolve_windows_fonts_dir(self) -> Optional[Path]:
        if os.name != "nt":
            return None
        windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C:\\Windows"
        fonts_dir = Path(windir) / "Fonts"
        return fonts_dir if fonts_dir.exists() else None

    def _resolve_windows_font_name(self, name: str) -> Optional[Path]:
        if not self._windows_fonts_dir or not name:
            return None
        candidate = self._windows_fonts_dir / name
        if candidate.exists():
            return candidate
        alias_map = {
            "arialunicode.ttf": "arialuni.ttf",
            "arial unicode.ttf": "arialuni.ttf",
            "arial unicode ms.ttf": "arialuni.ttf",
            "arial.ttf": "arial.ttf",
            "segoeui.ttf": "segoeui.ttf",
            "calibri.ttf": "calibri.ttf",
        }
        alias = alias_map.get(name.lower())
        if alias:
            candidate = self._windows_fonts_dir / alias
            if candidate.exists():
                return candidate
        return None

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
        if os.name == "nt":
            if self._windows_fonts_dir:
                win_fonts = [
                    self._windows_fonts_dir / "arialuni.ttf",
                    self._windows_fonts_dir / "arial.ttf",
                    self._windows_fonts_dir / "segoeui.ttf",
                    self._windows_fonts_dir / "calibri.ttf",
                ]
                candidates.extend([p for p in win_fonts if p.exists()])
        else:
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


def _split_word_to_width(
    word: str,
    draw: "ImageDraw.ImageDraw",
    font: "ImageFont.FreeTypeFont",
    max_width: int,
) -> List[str]:
    if not word:
        return [word]
    parts: List[str] = []
    current = ""
    for character in word:
        candidate = current + character
        if current and _measure_text(candidate, draw, font)[0] > max_width:
            parts.append(current)
            current = character
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [word]


def _wrap_text(text: str, draw: "ImageDraw.ImageDraw", font: "ImageFont.FreeTypeFont", max_width: int) -> List[str]:
    if max_width <= 0:
        return [text]
    lines: List[str] = []
    for paragraph in text.splitlines() or [""]:
        raw_words = paragraph.split()
        if not raw_words:
            lines.append("")
            continue
        current = ""
        for raw_word in raw_words:
            split_parts = (
                _split_word_to_width(raw_word, draw, font, max_width)
                if _measure_text(raw_word, draw, font)[0] > max_width
                else [raw_word]
            )
            if len(split_parts) > 1:
                if current:
                    lines.append(current)
                    current = ""
                lines.extend(split_parts[:-1])
                current = split_parts[-1]
                continue
            word = split_parts[0]
            candidate = f"{current} {word}"
            width, _ = _measure_text(candidate, draw, font)
            if not current:
                current = word
            elif width <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
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
    bg_pixels = roi[mask == 0]
    if bg_pixels.size:
        bg_median = np.median(bg_pixels, axis=0)
    else:
        bg_median = np.median(roi.reshape(-1, 3), axis=0)
    bb, bg, br = [float(value) for value in bg_median.tolist()]
    bg_luma = 0.299 * br + 0.587 * bg + 0.114 * bb
    pixels = text_pixels.reshape(-1, 3)
    b = pixels[:, 0].astype(float)
    g = pixels[:, 1].astype(float)
    r = pixels[:, 2].astype(float)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if bg_luma >= 128.0:
        threshold = np.percentile(luma, 35)
        selected = pixels[luma <= threshold]
    else:
        threshold = np.percentile(luma, 65)
        selected = pixels[luma >= threshold]
    if len(selected) >= 5:
        text_pixels = selected
    median = np.median(text_pixels, axis=0).astype(int)
    b, g, r = [int(max(0, min(255, c))) for c in median.tolist()]
    return (b, g, r)


def _boost_text_contrast(text_color: Tuple[int, int, int], bg_color: Tuple[int, int, int], *, min_diff: int = 100) -> Tuple[int, int, int]:
    """Make text color more distinguishable from background without changing hue drastically."""
    tb, tg, tr = text_color
    bb, bg, br = bg_color
    text_l = 0.299 * tr + 0.587 * tg + 0.114 * tb
    bg_l = 0.299 * br + 0.587 * bg + 0.114 * bb
    diff = abs(text_l - bg_l)
    if diff >= min_diff:
        return (tb, tg, tr)
    adjust = int(min_diff - diff)
    if text_l > bg_l or (abs(text_l - bg_l) < 1e-6 and bg_l < 128):
        tb = min(255, tb + adjust)
        tg = min(255, tg + adjust)
        tr = min(255, tr + adjust)
    else:
        tb = max(0, tb - adjust)
        tg = max(0, tg - adjust)
        tr = max(0, tr - adjust)
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
                    "stable_id": region.stable_id,
                    "bbox": list(region.bbox),
                    "text": region.source_text,
                    "translated_text": region.translated_text or "",
                    "corrected_text": region.corrected_text or "",
                    "semantic_type": region.semantic_type or "unknown",
                    "translate": bool(region.translated_text),
                    "fit_strategy": region.fit_strategy or "default",
                    "vision_confidence": region.vision_confidence,
                    "qa_flags": list(region.qa_flags),
                    "alignment": region.alignment,
                    "font_weight": region.font_weight,
                    "is_bold": region.is_bold,
                    "source_font_size_estimate": region.source_font_size_estimate,
                    "rendered_font_size": region.rendered_font_size,
                    "rotation_deg": region.text_orientation_deg,
                    "confidence": region.confidence,
                    "parent_block_id": region.parent_block_id,
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
        "version": 3,
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
