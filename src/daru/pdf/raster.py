"""PDF rasterization and streaming page loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..translation.job import check_cancelled
from .common import _noop_log, _require_dependency
from .config import PdfProcessingConfig
from .geometry import _median_float
from .models import PageImage, PdfLog

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover
    convert_from_path = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore


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

    def iter_pages(self, input_path: Path) -> Iterable[PageImage]:
        """Rasterize one page at a time while preserving Poppler crop semantics."""
        if input_path.suffix.lower() != ".pdf":
            yield self._load_image(input_path)
            return
        _require_dependency(fitz, "Для постраничной обработки требуется PyMuPDF")
        source_dpi = _detect_pdf_raster_dpi(input_path, self.log)
        with fitz.open(str(input_path)) as document:
            for index in range(len(document)):
                check_cancelled()
                rect = document[index].rect
                images = pdf_to_images(input_path, self.config.render_dpi, self.config.image_format,
                                       self.log, first_page=index + 1, last_page=index + 1)
                if len(images) != 1:
                    for image in images:
                        image.close()
                    raise RuntimeError(f"PDF: не удалось получить страницу {index + 1}")
                image = images[0]
                yield PageImage(index, image, image.width, image.height, self.config.render_dpi,
                                physical_size_points=(rect.width, rect.height), source_dpi=source_dpi)

    def _load_pdf(self, pdf_path: Path) -> List[PageImage]:
        return list(self.iter_pages(pdf_path))

    def _load_image(self, image_path: Path) -> PageImage:
        _require_dependency(Image, "Установите Pillow (pip install pillow)")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Не удалось открыть изображение {image_path}: {exc}") from exc
        width, height = image.size
        return PageImage(page_index=0, image=image, width=width, height=height, dpi=self.config.dpi)


def _detect_pdf_raster_dpi(pdf_path: Path, log: PdfLog = _noop_log) -> Optional[int]:
    """Detect a native DPI when most PDF pages are full-page raster scans."""
    if fitz is None:
        return None
    try:
        document = fitz.open(str(pdf_path))
    except Exception:
        return None
    detected: List[float] = []
    page_count = len(document)
    try:
        for page in document:
            page_area = max(1.0, float(page.rect.get_area()))
            best: Optional[Tuple[float, float]] = None
            for image_info in page.get_images(full=True):
                xref = int(image_info[0])
                pixel_width = int(image_info[2])
                pixel_height = int(image_info[3])
                if pixel_width <= 0 or pixel_height <= 0:
                    continue
                try:
                    placements = page.get_image_rects(xref)
                except Exception:
                    placements = []
                for rect in placements:
                    if rect.width <= 0 or rect.height <= 0:
                        continue
                    coverage = min(1.0, float(rect.get_area()) / page_area)
                    if coverage < 0.80:
                        continue
                    direct = (
                        pixel_width * 72.0 / float(rect.width),
                        pixel_height * 72.0 / float(rect.height),
                    )
                    swapped = (
                        pixel_height * 72.0 / float(rect.width),
                        pixel_width * 72.0 / float(rect.height),
                    )
                    axes = min((direct, swapped), key=lambda pair: abs(pair[0] - pair[1]))
                    if max(axes) / max(1.0, min(axes)) > 1.25:
                        continue
                    candidate = (coverage, (axes[0] + axes[1]) / 2.0)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is not None:
                detected.append(best[1])
    finally:
        document.close()
    # A document-wide DPI cap is safe only when every page is a compatible
    # full-page raster.  Mixed raster/vector manuals must keep their requested
    # output resolution for the vector pages and newly drawn text.
    required_pages = max(1, page_count)
    if len(detected) < required_pages:
        return None
    native_dpi = int(round(_median_float(detected, 0.0)))
    if native_dpi < 72 or native_dpi > 1200:
        return None
    log(
        f"PDF: определено исходное разрешение полноформатного растра: {native_dpi} DPI "
        f"({len(detected)}/{page_count} стр.)"
    )
    return native_dpi


def _effective_output_dpi(
    requested_dpi: int,
    pages: Sequence[PageImage],
    log: PdfLog = _noop_log,
) -> int:
    output_dpi = max(72, int(requested_dpi))
    source_values = [int(page.source_dpi) for page in pages if page.source_dpi]
    if source_values:
        source_dpi = int(round(_median_float(source_values, float(output_dpi))))
        if source_dpi < output_dpi:
            log(
                f"PDF: итоговое разрешение уменьшено с {output_dpi} до {source_dpi} DPI; "
                "апскейлинг исходного скана не добавляет детализации"
            )
            output_dpi = max(72, source_dpi)
    return output_dpi


def pdf_to_images(pdf_path: Path, dpi: int, image_format: str, log: PdfLog,
                  *, first_page: Optional[int] = None, last_page: Optional[int] = None) -> List["Image.Image"]:
    """Convert a PDF to high-resolution RGB images (lossless/hi-res where possible)."""
    _require_dependency(convert_from_path, "Установите pdf2image (pip install pdf2image)")
    _require_dependency(Image, "Установите Pillow (pip install pillow)")
    poppler_path = _detect_poppler_path()
    poppler_hint = " Укажите переменную POPPLER_PATH или добавьте папку bin в PATH." if not poppler_path else ""
    try:
        page_range = {}
        if first_page is not None:
            page_range["first_page"] = first_page
        if last_page is not None:
            page_range["last_page"] = last_page
        images = convert_from_path(  # type: ignore[arg-type]
            str(pdf_path),
            dpi=int(dpi),
            fmt=image_format,
            poppler_path=poppler_path,
            use_cropbox=True,
            **page_range,
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


def _detect_poppler_path() -> Optional[str]:
    for env_name in ("POPPLER_PATH", "POPPLER_BIN", "POPPLER_HOME"):
        candidate = os.environ.get(env_name)
        if candidate:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.exists():
                return str(candidate_path)
    return None
