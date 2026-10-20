"""Pure bounding-box and block geometry helpers, independent of OCR and rendering."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set, Tuple

from .models import BBoxWH, BlockRegion


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


def text_frame_bbox(region: Any) -> Tuple[float, float, float, float]:
    """Use visible text as the anchor while retaining safe translated-text room."""
    x, y, width, height = region.bbox
    semantic = str(getattr(region, "semantic_type", "") or "").lower()
    font_size = max(1.0, float(getattr(region, "font_size_estimate", 0) or 1.0))
    single_line_map_label = (
        semantic == "map_label"
        and not bool(getattr(region, "is_table_cell", False))
        and "\n" not in str(getattr(region, "source_text", "") or "")
        and float(height) <= font_size * 1.5
    )
    if single_line_map_label:
        frame = (
            float(x),
            float(y) - float(height) * 0.40,
            float(x + width),
            float(y + height) + float(height) * 0.20,
        )
    else:
        frame = (float(x), float(y), float(x + width), float(y + height))
    content = getattr(region, "content_bbox", None)
    if not content:
        return frame
    content_x, content_y, content_width, content_height = content
    content_abs = (
        float(x + content_x),
        float(y + content_y),
        float(x + content_x + content_width),
        float(y + content_y + content_height),
    )
    sparse_table_label = (
        bool(getattr(region, "is_table_cell", False))
        and not bool(getattr(region, "is_vertical", False))
        and float(content_width) < float(width) * 0.45
    )
    if sparse_table_label:
        pad_x = max(4.0, font_size * 0.60)
        pad_top = max(3.0, font_size * 0.55)
        pad_bottom = max(2.0, font_size * 0.20)
        frame = (
            max(frame[0], content_abs[0] - pad_x),
            max(frame[1], content_abs[1] - pad_top),
            min(frame[2], content_abs[2] + pad_x),
            min(frame[3], content_abs[3] + pad_bottom),
        )
    alignment = str(getattr(region, "alignment", "left") or "left").lower()
    top = frame[1] if single_line_map_label or sparse_table_label else content_abs[1]
    if alignment == "right":
        return (frame[0], top, content_abs[2], frame[3])
    if alignment == "center":
        return (frame[0], top, frame[2], frame[3])
    left = content_abs[0]
    if (
        bool(getattr(region, "is_table_cell", False))
        and str(getattr(region, "block_type", "") or "").upper() == "CELL"
        and not sparse_table_label
        and float(content_x) < font_size * 0.75
    ):
        # Keep translated descriptions clear of preserved A-D circle markers.
        left += min(font_size * 0.25, max(0.0, frame[2] - left) * 0.04)
    return (left, top, frame[2], frame[3])
