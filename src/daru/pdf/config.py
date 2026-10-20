"""PDF processing options without OCR, rendering, or network dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.paths import user_data_dir

__all__ = ["PdfProcessingConfig", "TEXTRACT_LOG_PATH", "_CACHE_DIR"]

PDF_TYPE_SCANNED = "scanned"
PDF_TYPE_NATIVE = "native"
PDF_PROCESSING_TEXTRACT = "textract"
PDF_PROCESSING_TEXTRACT_VISION = "textract_vision"

TRANSLATABLE_BLOCK_TYPES = {"TEXT", "TITLE", "LIST", "CELL", "KEY", "VALUE"}
FIGURE_BLOCK_TYPES = {"LAYOUT_FIGURE", "FIGURE"}

_LEGACY_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_DIR = user_data_dir() / "cache" / "pdf"
TEXTRACT_LOG_PATH = _CACHE_DIR / "textract_requests.log"

VISION_IMAGE_QUALITY_STABLE = "stable"
VISION_IMAGE_QUALITY_HIGH = "high"
VISION_IMAGE_QUALITY_ORIGINAL = "original"
VISION_IMAGE_QUALITIES = {
    VISION_IMAGE_QUALITY_STABLE,
    VISION_IMAGE_QUALITY_HIGH,
    VISION_IMAGE_QUALITY_ORIGINAL,
}
VISION_IMAGE_QUALITY_MAX_PIXELS = {
    VISION_IMAGE_QUALITY_STABLE: 4_000_000,
    VISION_IMAGE_QUALITY_HIGH: 10_000_000,
    VISION_IMAGE_QUALITY_ORIGINAL: None,
}
VISION_REQUEST_MODE_BATCHED = "batched"
VISION_REQUEST_MODE_SINGLE_PAGE = "single_page"
VISION_REQUEST_MODES = {
    VISION_REQUEST_MODE_BATCHED,
    VISION_REQUEST_MODE_SINGLE_PAGE,
}


@dataclass
class PdfProcessingConfig:
    """High-quality PDF processing knobs."""

    dpi: int = 400  # output DPI when writing PDF
    render_dpi: int = 400  # DPI for rasterizing PDF pages
    image_format: str = "png"  # format for intermediate page images
    editable_text: bool = True  # keep OCR translations as embedded PDF text
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

    @classmethod
    def from_options(
        cls,
        options: Optional[Dict[str, object]] = None,
        *,
        base: Optional["PdfProcessingConfig"] = None,
    ) -> "PdfProcessingConfig":
        """Apply known UI / CLI overrides without duplicating every field."""
        config = base or cls()
        if not options:
            return config.normalized()
        known_names = {item.name for item in fields(cls)}
        overrides = {
            name: options[name]
            for name in known_names
            if name in options and options[name] is not None
        }
        return replace(config, **overrides).normalized()

    def normalized(self) -> "PdfProcessingConfig":
        blur = self.blur_kernel_size if self.blur_kernel_size % 2 == 1 else self.blur_kernel_size + 1
        blur = max(3, blur)
        dilation = max(1, self.dilation_kernel_size)
        render_dpi = max(150, self.render_dpi or self.dpi)
        # OCR/render resolution and the final PDF resolution serve different
        # purposes.  Keeping them independent lets us analyse small text at a
        # high DPI and downsample only after translated text has been drawn.
        output_dpi = max(72, int(self.dpi or render_dpi))
        image_format = (self.image_format or "png").lower()
        detector_conf = max(0.0, min(1.0, self.detector_min_confidence))
        min_box_w = max(1, self.min_box_width)
        min_box_h = max(1, self.min_box_height)
        languages_value: Optional[List[str]] = None
        if self.easyocr_languages:
            if isinstance(self.easyocr_languages, str):
                languages_value = [token for token in re.split(r"[+,/;\s]+", self.easyocr_languages) if token]
            else:
                languages_value = [str(lang).strip() for lang in self.easyocr_languages if str(lang).strip()]
        return PdfProcessingConfig(
            dpi=output_dpi,
            render_dpi=render_dpi,
            image_format=image_format,
            editable_text=bool(self.editable_text),
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
