"""Raster text cleanup and rendering, including font fitting."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import _noop_log, _require_dependency
from .config import PdfProcessingConfig
from .geometry import text_frame_bbox
from .imaging import _sample_text_color, build_text_mask
from .models import PageImage, PdfLog, RegionInfo

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import ImageDraw
except ImportError:  # pragma: no cover
    ImageDraw = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import ImageFont
except ImportError:  # pragma: no cover
    ImageFont = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


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
        return self.render_text_layer(self.clean_page(page, regions), regions)

    def clean_page(self, page: PageImage, regions: Sequence[RegionInfo]) -> "Image.Image":
        """Remove source glyphs while preserving photographs, lines, and tables."""
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

        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

    def render_text_layer(
        self,
        cleaned_page: "Image.Image",
        regions: Sequence[RegionInfo],
    ) -> "Image.Image":
        """Draw the legacy raster preview/fallback on an already-cleaned page."""
        cleaned_pil = cleaned_page.convert("RGBA")
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
        if region.content_bbox and not region.is_table_cell and not region.is_vertical:
            # The cleaning mask may extend far beyond the visible source text.
            # Share the editable renderer's anchor instead of drawing at that margin.
            left, top, right, bottom = text_frame_bbox(region)
            right, bottom = min(x0 + w, int(right)), min(y0 + h, int(bottom))
            x0, y0 = max(x0, int(left)), max(y0, int(top))
            w, h = max(1, right - x0), max(1, bottom - y0)
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
        # Vertical text uses exactly the same paragraph layout as horizontal
        # text on a swapped canvas.  Rotating the completed layer is both
        # simpler and avoids the old per-character height accumulation bug.
        layout_w, layout_h = (h, w) if region.is_vertical else (w, h)
        padding_ratio = 0.025 if dense_fit else (0.04 if region.is_table_cell else 0.06)
        padding = max(1, int(min(layout_w, layout_h) * padding_ratio))
        max_text_width = max(1, layout_w - 2 * padding)
        max_text_height = max(1, layout_h - 2 * padding)

        min_font_size = 5 if dense_fit or region.fit_strategy in {"wrap", "shrink"} else 8
        start_size = max(min_font_size, min(96, region.font_size_estimate))
        chosen_font: Optional["ImageFont.FreeTypeFont"] = None
        chosen_size = min_font_size
        wrapped_lines: List[str] = []
        line_spacing = 0
        line_advance = 1
        for size in range(start_size, min_font_size - 1, -1):
            font = self.font_loader.get(size)
            lines = _wrap_text(text, draw, font, max_text_width)
            if not lines:
                continue
            candidate_advance = _font_line_advance(draw, font)
            spacing = max(
                0 if dense_fit else 1,
                int(candidate_advance * (0.12 if dense_fit else 0.25)),
            )
            total_height = len(lines) * candidate_advance + max(0, len(lines) - 1) * spacing
            max_width = max(_measure_text(line, draw, font)[0] for line in lines)
            if total_height <= max_text_height and max_width <= max_text_width:
                chosen_font = font
                chosen_size = size
                wrapped_lines = lines
                line_spacing = spacing
                line_advance = candidate_advance
                break
        if chosen_font is None:
            chosen_font = self.font_loader.get(min_font_size)
            chosen_size = min_font_size
            wrapped_lines = _wrap_text(text, draw, chosen_font, max_text_width)
            line_advance = _font_line_advance(draw, chosen_font)
            line_spacing = max(
                0 if dense_fit else 1,
                int(line_advance * 0.15),
            )

        if region.text_color:
            b, g, r = region.text_color
            text_color = (r, g, b, 255)
        else:
            text_color = _sample_text_color(base_image, (x0, y0, x0 + w, y0 + h))
        stroke_w = 1 if region.is_bold else 0
        stroke_fill = text_color

        total_text_height = len(wrapped_lines) * line_advance + max(
            0, len(wrapped_lines) - 1
        ) * line_spacing
        region.rendered_font_size = chosen_size

        if region.is_vertical:
            region_layer = Image.new("RGBA", (layout_w, layout_h), (0, 0, 0, 0))
            region_draw = ImageDraw.Draw(region_layer)
            current_y = max(padding, int((layout_h - total_text_height) / 2))
            for line_index, line in enumerate(wrapped_lines):
                line_width = _measure_text(line, region_draw, chosen_font)[0]
                if region.alignment == "center":
                    line_x = max(padding, int((layout_w - line_width) / 2))
                elif region.alignment == "right":
                    line_x = max(padding, layout_w - padding - line_width)
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
                current_y += line_advance + line_spacing
            angle = region.text_orientation_deg or -90.0
            rotated = region_layer.rotate(angle, expand=True, resample=Image.BICUBIC)
            clipped = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            offset = ((w - rotated.width) // 2, (h - rotated.height) // 2)
            clipped.paste(rotated, offset, rotated)
            overlay.alpha_composite(clipped, dest=(x0, y0))
        else:
            region_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            region_draw = ImageDraw.Draw(region_layer)
            current_y = padding
            if region.is_table_cell and region.content_bbox:
                _cx, cy, _cw, ch = region.content_bbox
                target_y = cy + max(0, int((ch - total_text_height) / 2))
                current_y = max(padding, min(target_y, max(padding, h - padding - max(1, total_text_height))))
            for line_index, line in enumerate(wrapped_lines):
                line_width = _measure_text(line, region_draw, chosen_font)[0]
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
                current_y += line_advance + line_spacing

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


def _font_line_advance(
    draw: "ImageDraw.ImageDraw",
    font: "ImageFont.FreeTypeFont",
) -> int:
    """Return one stable line advance independent of the glyphs in a line."""
    return max(1, _measure_text("Ag", draw, font)[1])


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
    # A single OCR newline is normally a visual line wrap, not a semantic
    # paragraph break.  Reflow it; preserve only blank-line separators.
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = re.split(r"\n\s*\n", normalized)
    paragraphs = [
        " ".join(part.strip() for part in paragraph.splitlines() if part.strip())
        for paragraph in raw_paragraphs
    ] or [""]
    for paragraph_index, paragraph in enumerate(paragraphs):
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
        if paragraph_index < len(paragraphs) - 1:
            lines.append("")
    return lines
