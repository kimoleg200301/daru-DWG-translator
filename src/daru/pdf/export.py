"""Atomic PDF export with optional editable text overlays."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from ..translation.job import check_cancelled
from .common import _noop_log, _require_dependency
from .models import PdfLog

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import ImageChops
except ImportError:  # pragma: no cover
    ImageChops = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore


class Exporter:
    """Exports processed page images back to a multi-page PDF (or image sequence)."""

    def __init__(self, log: PdfLog = _noop_log) -> None:
        self.log = log

    def to_pdf(
        self,
        images: Sequence["Image.Image"],
        output_path: Path,
        *,
        dpi: int,
        input_dpi: Optional[int] = None,
        page_sizes_points: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
        page_overlay: Optional[Callable[[int, object], None]] = None,
    ) -> None:
        images_to_pdf(
            images,
            output_path,
            dpi=dpi,
            input_dpi=input_dpi,
            page_sizes_points=page_sizes_points,
            page_overlay=page_overlay,
            log=self.log,
        )


def _image_is_grayscale(image: "Image.Image", *, tolerance: int = 2) -> bool:
    if image.mode in {"1", "L"}:
        return True
    _require_dependency(ImageChops, "Pillow ImageChops недоступен")
    rgb = image.convert("RGB")
    red, green, blue = rgb.split()
    return max(
        ImageChops.difference(red, green).getextrema()[1],
        ImageChops.difference(red, blue).getextrema()[1],
        ImageChops.difference(green, blue).getextrema()[1],
    ) <= max(0, int(tolerance))


def _prepare_pdf_page_image(
    image: "Image.Image",
    target_size: Tuple[int, int],
) -> "Image.Image":
    _require_dependency(Image, "Установите Pillow (pip install pillow)")
    if image.mode == "RGBA":
        rgb = Image.new("RGB", image.size, "white")
        rgb.paste(image, mask=image.getchannel("A"))
    else:
        rgb = image.convert("RGB")
    normalized_size = (max(1, int(target_size[0])), max(1, int(target_size[1])))
    if rgb.size != normalized_size:
        resampling = getattr(Image, "Resampling", Image)
        rgb = rgb.resize(normalized_size, resampling.LANCZOS)
    return rgb.convert("L") if _image_is_grayscale(rgb) else rgb


def _pdf_jpeg_quality(image: "Image.Image") -> int:
    """Choose scan-friendly compression without penalising genuine colour pages."""
    # A grayscale scan has only one channel and tolerates the conventional JPEG
    # quality used by scanners.  Keeping colour pages a little higher avoids
    # introducing visible banding in photographs and colour-coded diagrams.
    return 75 if image.mode == "L" else 88


def images_to_pdf(
    images: Sequence["Image.Image"],
    output_path: Path,
    *,
    dpi: int,
    log: PdfLog,
    input_dpi: Optional[int] = None,
    page_sizes_points: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    page_overlay: Optional[Callable[[int, object], None]] = None,
) -> None:
    """Downsample, optimize, and save processed page images to a PDF."""
    if not images:
        raise RuntimeError("Нет изображений для сохранения PDF")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dpi = max(72, int(dpi))
    working_dpi = max(72, int(input_dpi or output_dpi))

    def page_geometry(index: int, image: "Image.Image") -> Tuple[Tuple[float, float], Tuple[int, int]]:
        points: Optional[Tuple[float, float]] = None
        if page_sizes_points is not None and index < len(page_sizes_points):
            candidate = page_sizes_points[index]
            if candidate and candidate[0] > 0 and candidate[1] > 0:
                points = (float(candidate[0]), float(candidate[1]))
        if points is None:
            points = (
                float(image.width) * 72.0 / working_dpi,
                float(image.height) * 72.0 / working_dpi,
            )
        pixels = (
            max(1, int(round(points[0] * output_dpi / 72.0))),
            max(1, int(round(points[1] * output_dpi / 72.0))),
        )
        return points, pixels

    grayscale_pages = 0
    if fitz is not None:
        document = fitz.open()
        temporary_path = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.tmp.pdf"
        )
        try:
            if temporary_path.exists():
                temporary_path.unlink()
            try:
                for index, image in enumerate(images):
                    check_cancelled()
                    points, target_size = page_geometry(index, image)
                    prepared = _prepare_pdf_page_image(image, target_size)
                    grayscale_pages += int(prepared.mode == "L")
                    buffer = BytesIO()
                    prepared.save(
                        buffer,
                        format="JPEG",
                        quality=_pdf_jpeg_quality(prepared),
                        optimize=True,
                        subsampling=0,
                    )
                    page = document.new_page(width=points[0], height=points[1])
                    page.insert_image(
                        page.rect,
                        stream=buffer.getvalue(),
                        keep_proportion=False,
                    )
                    if page_overlay is not None:
                        page_overlay(index, page)
                document.save(
                    str(temporary_path),
                    garbage=4,
                    deflate=True,
                    clean=True,
                )
            finally:
                if not getattr(document, "is_closed", False):
                    document.close()
            os.replace(temporary_path, output_path)
        except Exception:
            if temporary_path.exists():
                temporary_path.unlink()
            raise
    else:  # pragma: no cover - PyMuPDF is a project dependency; keep a safe fallback
        if page_overlay is not None:
            raise RuntimeError(
                "PyMuPDF is required to export editable OCR text"
            )
        prepared_pages: List["Image.Image"] = []
        for index, image in enumerate(images):
            check_cancelled()
            _points, target_size = page_geometry(index, image)
            prepared = _prepare_pdf_page_image(image, target_size)
            grayscale_pages += int(prepared.mode == "L")
            prepared_pages.append(prepared)
        base, *rest = prepared_pages
        fallback_quality = min(_pdf_jpeg_quality(page) for page in prepared_pages)
        base.save(
            output_path,
            "PDF",
            resolution=output_dpi,
            save_all=True,
            append_images=rest,
            quality=fallback_quality,
            optimize=True,
            subsampling=0,
        )
    log(
        f"PDF: сохранено {len(images)} страниц в {output_path}; "
        f"{output_dpi} DPI, grayscale {grayscale_pages}/{len(images)}"
    )
