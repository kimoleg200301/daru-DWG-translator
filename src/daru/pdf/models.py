"""PDF data models independent of OCR providers and rendering backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from .config import (
    PDF_PROCESSING_TEXTRACT,
    PDF_PROCESSING_TEXTRACT_VISION,
    PDF_TYPE_NATIVE,
    PDF_TYPE_SCANNED,
)

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

PdfLog = Callable[[str], None]
BBox = Tuple[int, int, int, int]
BBoxWH = Tuple[int, int, int, int]

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
    source_dpi: Optional[int] = None

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
    block_type: str = "TEXT"
    vector_font_scale: float = 1.0
    rendered_bbox_points: Optional[Tuple[float, float, float, float]] = None
    layout_warning: Optional[str] = None

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
    alignment_locked: bool = False
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


__all__ = [
    "PDF_TYPE_NATIVE",
    "PDF_TYPE_SCANNED",
    "PDF_PROCESSING_TEXTRACT",
    "PDF_PROCESSING_TEXTRACT_VISION",
    "BlockRegion",
    "DetectedSegment",
    "LineRegion",
    "MergedRegion",
    "PageImage",
    "PdfProcessingResult",
    "RegionInfo",
    "TextBlock",
    "TextractPageData",
    "WordBox",
]
