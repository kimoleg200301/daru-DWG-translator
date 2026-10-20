"""Text masks and color estimation used by OCR and PDF rendering."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .common import _require_dependency
from .geometry import _median_float
from .models import BBoxWH

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


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
    # Word-box masks include whitespace. With thin glyphs, the 35th percentile
    # of the entire box can still be background, washing black text into gray.
    foreground = np.abs(luma - bg_luma) >= 16.0
    if np.count_nonzero(foreground) >= 5:
        pixels = pixels[foreground]
        luma = luma[foreground]
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
