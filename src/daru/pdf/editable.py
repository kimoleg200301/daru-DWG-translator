"""Adapters for writing OCR translations as editable PDF text."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - dependency guard
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - dependency guard
    fitz = None  # type: ignore

from .geometry import text_frame_bbox as _text_frame_bbox
from .native import EditableTextInserter, NativeTextUnit, ProtectedText


PdfLog = Callable[[str], None]
EDITABLE_ROTATION_TOLERANCE_DEG = 3.0


def _noop_log(_message: str) -> None:
    return None


@dataclass(frozen=True)
class PageTransform:
    """Independent pixel-to-point scaling for one rendered PDF page."""

    scale_x: float
    scale_y: float

    @classmethod
    def from_page(cls, page_image: Any, page_rect: Any) -> "PageTransform":
        return cls(
            scale_x=float(page_rect.width) / max(1, int(page_image.width)),
            scale_y=float(page_rect.height) / max(1, int(page_image.height)),
        )

    def rect(self, bbox: Tuple[float, float, float, float]) -> Any:
        x0, y0, x1, y1 = bbox
        return fitz.Rect(
            x0 * self.scale_x,
            y0 * self.scale_y,
            x1 * self.scale_x,
            y1 * self.scale_y,
        )




def _quarter_turn(region: Any) -> Optional[int]:
    angle = float(getattr(region, "text_orientation_deg", 0.0) or 0.0)
    if bool(getattr(region, "is_vertical", False)) and abs(angle) < 45.0:
        angle = -90.0
    nearest = round(angle / 90.0) * 90.0
    delta = ((angle - nearest + 180.0) % 360.0) - 180.0
    if abs(delta) > EDITABLE_ROTATION_TOLERANCE_DEG:
        return None
    return int(nearest) % 360


def supports_editable_rotation(region: Any) -> bool:
    return _quarter_turn(region) is not None


def _region_rotation(region: Any) -> int:
    rotation = _quarter_turn(region)
    if rotation is None:
        angle = float(getattr(region, "text_orientation_deg", 0.0) or 0.0)
        raise ValueError(
            f"Editable PDF text supports only quarter-turn rotation; got {angle:g} degrees"
        )
    return rotation


def _region_color(region: Any) -> int:
    blue, green, red = getattr(region, "text_color", (0, 0, 0))
    return (
        (max(0, min(255, int(red))) << 16)
        | (max(0, min(255, int(green))) << 8)
        | max(0, min(255, int(blue)))
    )


def _region_role(region: Any) -> str:
    if bool(getattr(region, "is_table_cell", False)):
        return "CELL"
    semantic = str(getattr(region, "semantic_type", "") or "").lower()
    if semantic == "map_label":
        # Diagram callouts must remain inside their source frames; expanding a
        # translated label puts it over leaders and adjacent artwork.
        return "CELL"
    if semantic == "caption":
        return "CAPTION"
    return str(getattr(region, "block_type", "TEXT") or "TEXT").upper()


_STRUCTURAL_LINE_RE = re.compile(
    r"^\s*(?:\(?\d+[.)]|[A-Za-zА-Яа-яЁё][.)]|[-–—•])\s+"
)


def _reflow_ocr_prose(text: str) -> str:
    """Remove OCR wrap breaks while retaining paragraph and list boundaries."""
    lines = str(text or "").splitlines()
    if len(lines) < 2:
        return str(text or "").strip()
    paragraphs: List[str] = []
    current = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            if paragraphs and paragraphs[-1] != "":
                paragraphs.append("")
            continue
        if current and _STRUCTURAL_LINE_RE.match(line):
            paragraphs.append(current)
            current = line
        elif current:
            current = f"{current} {line}"
        else:
            current = line
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs).strip()


def region_to_text_unit(page_image: Any, region: Any, page_rect: Any) -> NativeTextUnit:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for editable OCR PDF export")
    transform = PageTransform.from_page(page_image, page_rect)
    rect = transform.rect(_text_frame_bbox(region)) & page_rect
    text = str(
        getattr(region, "translated_text", "")
        or getattr(region, "source_text", "")
        or ""
    ).strip()
    role = _region_role(region)
    if role not in {"CELL", "LIST"}:
        text = _reflow_ocr_prose(text)
    # The legacy Pillow preview may already have shrunk rendered_font_size.
    # HTMLBox must fit from the preferred OCR / Vision size exactly once.
    pixel_font_size = int(
        getattr(region, "font_size_estimate", 0)
        or getattr(region, "source_font_size_estimate", 0)
        or getattr(region, "rendered_font_size", 0)
        or 12
    )
    font_scale = (
        transform.scale_x
        if bool(getattr(region, "is_vertical", False))
        else transform.scale_y
    )
    return NativeTextUnit(
        page_index=int(page_image.page_index),
        stable_id=str(getattr(region, "stable_id", "") or region.region_id),
        role=role,
        bbox=tuple(rect),
        source_text=str(getattr(region, "source_text", "") or ""),
        protected=ProtectedText(value=text, replacements={}),
        font_size=max(4.0, float(pixel_font_size) * font_scale),
        font_name="",
        color=_region_color(region),
        bold=bool(
            getattr(region, "is_bold", False)
            or getattr(region, "font_weight", "") == "bold"
        ),
        italic=False,
        rotation=_region_rotation(region),
        font_family_kind="sans",
        alignment=str(getattr(region, "alignment", "left") or "left"),
        translated_text=text,
    )


class EditableTextLayer:
    """Stateful OCR overlay that reuses embedded font resources across pages."""

    def __init__(self, style_font: Optional[str] = None, log: PdfLog = _noop_log) -> None:
        self._inserter = EditableTextInserter(style_font)
        self.log = log
        self.warnings: List[Dict[str, Any]] = []
        self.inserted_regions = 0

    def insert_page(
        self,
        pdf_page: Any,
        page_image: Any,
        regions: Sequence[Any],
        *,
        record_layout: bool = True,
    ) -> List[Dict[str, Any]]:
        units = [region_to_text_unit(page_image, region, pdf_page.rect) for region in regions]
        warnings = self._inserter.insert_units(
            pdf_page,
            units,
            allow_expansion=True,
            occupied=[fitz.Rect(unit.bbox) for unit in units],
        )
        if not record_layout:
            return warnings
        self.warnings.extend(warnings)
        for region, unit in zip(regions, units):
            if unit.rendered_bbox is None:
                region.layout_warning = unit.layout_warning or "translation_did_not_fit"
                continue
            self.inserted_regions += 1
            region.vector_font_scale = float(unit.font_scale)
            region.rendered_bbox_points = tuple(unit.rendered_bbox)
            region.layout_warning = unit.layout_warning
            source_pixels = max(
                1,
                int(
                    getattr(region, "font_size_estimate", 0)
                    or getattr(region, "source_font_size_estimate", 0)
                    or 12
                ),
            )
            region.rendered_font_size = max(
                1,
                int(round(source_pixels * max(0.0, float(unit.font_scale)))),
            )
        return warnings

    def render_preview(
        self,
        cleaned_page: Any,
        page_image: Any,
        regions: Sequence[Any],
    ) -> Any:
        """Rasterize the same vector overlay that will be committed to the PDF."""
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for editable OCR PDF export")
        points = getattr(page_image, "physical_size_points", None)
        if not points:
            dpi = max(72, int(getattr(page_image, "dpi", 72) or 72))
            points = (
                float(page_image.width) * 72.0 / dpi,
                float(page_image.height) * 72.0 / dpi,
            )
        document = fitz.open()
        try:
            pdf_page = document.new_page(width=float(points[0]), height=float(points[1]))
            buffer = BytesIO()
            cleaned_page.convert("RGB").save(
                buffer,
                format="JPEG",
                quality=90,
                subsampling=0,
            )
            pdf_page.insert_image(
                pdf_page.rect,
                stream=buffer.getvalue(),
                keep_proportion=False,
            )
            self.insert_page(
                pdf_page,
                page_image,
                regions,
                record_layout=False,
            )
            pixmap = pdf_page.get_pixmap(
                matrix=fitz.Matrix(
                    float(page_image.width) / max(1.0, float(pdf_page.rect.width)),
                    float(page_image.height) / max(1.0, float(pdf_page.rect.height)),
                ),
                alpha=False,
            )
            if pixmap.n != 3:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            from PIL import Image

            preview = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            if preview.size != (page_image.width, page_image.height):
                preview = preview.resize(
                    (page_image.width, page_image.height),
                    Image.Resampling.LANCZOS,
                )
            return preview
        finally:
            document.close()
