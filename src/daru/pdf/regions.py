"""OCR block normalization, deduplication and render-region construction."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .common import _noop_log, _require_dependency
from .config import TRANSLATABLE_BLOCK_TYPES, PdfProcessingConfig
from .geometry import (
    _bbox_intersection_area_wh,
    _bbox_iou_wh,
    _block_content_bbox,
    _clamp_bbox_to_page,
    _collect_block_word_boxes,
    _median_float,
    _union_bbox,
)
from .imaging import (
    _boost_text_contrast,
    _estimate_background_color,
    _estimate_text_color,
    build_text_mask,
    build_text_mask_from_word_boxes,
)
from .layers import canonical_text as _canonicalize_text_for_dedup
from .models import BBoxWH, BlockRegion, PageImage, PdfLog, RegionInfo, TextractPageData

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


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


_WRAPPED_LINE_END_RE = re.compile(r"[-,(/]$")


_NEW_ITEM_RE = re.compile(
    r"^(?:[-*•▪◦◇◆]|\(?\d+[.)]|[A-Za-z][.)])\s+"
)


def _starts_with_lowercase_letter(text: str) -> bool:
    for character in str(text or "").lstrip():
        if character.isalpha():
            return character.islower()
        if character.isdigit():
            return False
    return False


def _is_wrapped_text_continuation(first: BlockRegion, second: BlockRegion) -> bool:
    """Return True for two Textract blocks that are really adjacent lines of one paragraph."""
    if first is second or first.is_table or second.is_table:
        return False
    if first.block_type not in {"TEXT", "TITLE"} or first.block_type != second.block_type:
        return False
    if not first.parent_block_id or first.parent_block_id != second.parent_block_id:
        return False
    if abs(float(first.rotation_deg or 0.0) - float(second.rotation_deg or 0.0)) > 5.0:
        return False

    first_text = str(first.source_text or "").strip()
    second_text = str(second.source_text or "").strip()
    if not first_text or not second_text or _NEW_ITEM_RE.match(second_text):
        return False
    if first_text.endswith((".", "!", "?", ";", ":")):
        return False
    if not (_starts_with_lowercase_letter(second_text) or _WRAPPED_LINE_END_RE.search(first_text)):
        return False

    ax0, ay0, ax1, ay1 = first.bbox
    bx0, by0, bx1, by1 = second.bbox
    first_height = max(1, ay1 - ay0)
    second_height = max(1, by1 - by0)
    line_height = _median_float([first_height, second_height], 1.0)
    if max(first_height, second_height) / max(1, min(first_height, second_height)) > 1.6:
        return False
    vertical_gap = by0 - ay1
    if vertical_gap < -line_height * 0.15 or vertical_gap > max(8.0, line_height * 0.85):
        return False
    horizontal_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
    min_width = max(1, min(ax1 - ax0, bx1 - bx0))
    if horizontal_overlap / min_width < 0.65:
        return False
    left_tolerance = max(line_height * 1.6, min(ax1 - ax0, bx1 - bx0) * 0.08)
    return abs(ax0 - bx0) <= left_tolerance


def _join_wrapped_text(first: str, second: str) -> str:
    left = str(first or "").rstrip()
    right = str(second or "").lstrip()
    if left.endswith("-") and len(left) > 1 and right:
        return left[:-1] + right
    return f"{left} {right}".strip()


def _merge_wrapped_text_blocks(
    blocks: Sequence[BlockRegion],
    log: PdfLog = _noop_log,
) -> List[BlockRegion]:
    """Merge obvious line fragments before translation so prose is fitted as a paragraph."""
    merged_blocks = list(blocks)
    removed: Set[int] = set()
    merge_count = 0
    eligible = [
        block
        for block in merged_blocks
        if block.block_type in {"TEXT", "TITLE"} and not block.is_table
    ]
    for current in sorted(eligible, key=lambda item: (item.bbox[1], item.bbox[0])):
        if id(current) in removed:
            continue
        while True:
            continuations = [
                candidate
                for candidate in eligible
                if id(candidate) not in removed
                and candidate is not current
                and candidate.bbox[1] >= current.bbox[1]
                and _is_wrapped_text_continuation(current, candidate)
            ]
            if not continuations:
                break
            continuation = min(
                continuations,
                key=lambda item: (
                    max(0, item.bbox[1] - current.bbox[3]),
                    abs(item.bbox[0] - current.bbox[0]),
                ),
            )
            combined_bbox = _union_bbox([current.bbox, continuation.bbox])
            if combined_bbox is None:
                break
            current_source = current.source_text
            continuation_source = continuation.source_text
            current_corrected = current.corrected_text
            continuation_corrected = continuation.corrected_text
            current.bbox = combined_bbox
            current.source_text = _join_wrapped_text(current_source, continuation_source)
            if current_corrected or continuation_corrected:
                current.corrected_text = _join_wrapped_text(
                    current_corrected or current_source,
                    continuation_corrected or continuation_source,
                )
            current.lines.extend(continuation.lines)
            current.word_boxes.extend(continuation.word_boxes)
            current.child_ids.extend(continuation.child_ids)
            current.confidence = _median_float(
                [current.confidence, continuation.confidence],
                current.confidence,
            )
            current.fit_strategy = "wrap"
            current.render_id = None
            current.stable_id = ""
            removed.add(id(continuation))
            merge_count += 1
    if merge_count:
        log(f"Textract: объединено переносов строк в абзацы: {merge_count}")
    return [block for block in merged_blocks if id(block) not in removed]


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


def _estimate_font_size_from_block(block: BlockRegion) -> int:
    rotation = float(block.rotation_deg or 0.0) % 360.0
    signed_rotation = rotation if rotation <= 180.0 else rotation - 360.0
    vertical = (
        abs(abs(signed_rotation) - 90.0) <= 12.0
        or block.semantic_type == "vertical"
        or block.fit_strategy == "vertical"
    )

    def word_extent(box: BBoxWH, token: str = "") -> int:
        _x, _y, width, height = box
        token_letters = re.sub(r"[^\w]+", "", token, flags=re.UNICODE)
        tall_word = (
            not vertical
            and len(token_letters) >= 2
            and height >= max(1, width) * 2.2
        )
        return int(width if vertical or tall_word else height)

    def extents(boxes: Sequence[BBoxWH], text: str) -> List[int]:
        tokens = re.findall(r"\S+", str(text or ""))
        aligned_tokens = tokens if len(tokens) == len(boxes) else [""] * len(boxes)
        values: List[int] = []
        for box, token in zip(boxes, aligned_tokens):
            value = word_extent(box, token)
            if value > 0:
                values.append(value)
        return values

    word_heights: List[int] = []
    if block.word_boxes:
        word_heights = extents(block.word_boxes, block.source_text)
    elif block.lines:
        for line in block.lines:
            if line.word_boxes:
                word_heights.extend(extents(line.word_boxes, line.source_text))
    if word_heights:
        word_heights.sort()
        median = word_heights[len(word_heights) // 2]
        return max(8, min(96, median))
    heights: List[int] = []
    for line in block.lines:
        extent = (
            line.bbox[2] - line.bbox[0]
            if vertical
            else line.bbox[3] - line.bbox[1]
        )
        if extent > 0:
            heights.append(extent)
    if heights:
        heights.sort()
        median = heights[len(heights) // 2]
        return max(8, min(96, median))
    block_extent = (
        block.bbox[2] - block.bbox[0]
        if vertical
        else block.bbox[3] - block.bbox[1]
    )
    return max(8, min(96, int(max(1, block_extent) * 0.6)))


def _harmonize_region_font_sizes(
    regions: Sequence[RegionInfo],
    log: PdfLog = _noop_log,
) -> None:
    """Normalize nearby body-text estimates while preserving distinct styles."""
    compatible: Dict[str, List[RegionInfo]] = {}
    excluded_semantics = {
        "caption",
        "contact",
        "footnote",
        "heading",
        "map_label",
        "table_cell",
        "title",
        "vertical",
    }
    for region in regions:
        semantic = str(region.semantic_type or "unknown").strip().casefold()
        block_type = str(region.block_type or "TEXT").strip().upper()
        if (
            region.is_vertical
            or region.is_table_cell
            or block_type == "TITLE"
            or semantic in excluded_semantics
            or region.fit_strategy in {"preserve_small", "vertical"}
            or region.font_size_estimate < 8
        ):
            continue
        weight = "bold" if region.is_bold or region.font_weight == "bold" else "normal"
        role = f"{block_type}:{weight}"
        compatible.setdefault(role, []).append(region)

    changed = 0
    for candidates in compatible.values():
        ordered = sorted(candidates, key=lambda item: item.font_size_estimate)
        clusters: List[List[RegionInfo]] = []
        for region in ordered:
            if not clusters:
                clusters.append([region])
                continue
            current = clusters[-1]
            center = _median_float(
                [item.font_size_estimate for item in current],
                float(region.font_size_estimate),
            )
            tolerance = max(3.0, center * 0.12)
            prospective_min = min(current[0].font_size_estimate, region.font_size_estimate)
            prospective_max = max(current[-1].font_size_estimate, region.font_size_estimate)
            if (
                abs(region.font_size_estimate - center) <= tolerance
                and prospective_max / max(1, prospective_min) <= 1.20
            ):
                current.append(region)
            else:
                clusters.append([region])
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            target = int(round(_median_float(
                [item.font_size_estimate for item in cluster],
                float(cluster[0].font_size_estimate),
            )))
            for region in cluster:
                if region.font_size_estimate != target:
                    region.font_size_estimate = target
                    changed += 1
    if changed:
        log(f"Render: harmonized font size for {changed} compatible text regions")


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
    if block.block_type != "TITLE" and block.semantic_type != "heading":
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
        block_type=block.block_type,
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
            # Duplicate detection must follow OCR identity, not the translated
            # wording: different labels can legitimately share one translation.
            canonical = _canonicalize_text_for_dedup(
                region.source_text or region.translated_text
            )
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
    _harmonize_region_font_sizes(regions, log)
    regions.sort(key=lambda r: r.region_id)
    return regions
