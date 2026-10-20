"""Read translation layers once and build indexes for compatibility formats."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from ..utils.io import write_json_atomic
from .models import RegionInfo, PdfLog


@dataclass
class LayerMappings:
    by_region: dict[tuple[int, int], str] = field(default_factory=dict)
    by_stable_id: dict[str, str] = field(default_factory=dict)
    by_text: dict[str, str] = field(default_factory=dict)


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def read_layer_payload(
    path: Path, log: Callable[[str], None], expected_identity: dict | None = None
) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log(f"PDF: не удалось прочитать слой перевода {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        return {}
    if (
        expected_identity is not None
        and data.get("cache_identity") != expected_identity
    ):
        log(
            "PDF: слой перевода относится к другому документу, языку или настройкам; кэш пропущен"
        )
        return {}
    return data


def load_layer_mappings(
    path: Path, log: Callable[[str], None], expected_identity: dict | None = None
) -> LayerMappings:
    data = read_layer_payload(path, log, expected_identity)
    result = LayerMappings()
    pages = data.get("pages")
    if not isinstance(pages, list):
        return result
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = page.get("blocks") or page.get("regions") or []
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            translated = str(block.get("translated_text") or "").strip()
            if not translated:
                continue
            stable_id = str(block.get("stable_id") or "").strip()
            if stable_id:
                result.by_stable_id[stable_id] = translated
            try:
                result.by_region[
                    (
                        int(page.get("page_index", 0)),
                        int(block.get("region_id", block.get("block_id", 0))),
                    )
                ] = translated
            except (TypeError, ValueError, OverflowError):
                pass
            source = str(block.get("text") or "").strip()
            for text in (source, block.get("corrected_text")):
                key = canonical_text(text)
                if key:
                    result.by_text.setdefault(key, translated)
            sources = [line.strip() for line in source.splitlines() if line.strip()]
            translations = [
                line.strip() for line in translated.splitlines() if line.strip()
            ]
            if len(sources) == len(translations):
                for text, value in zip(sources, translations):
                    result.by_text.setdefault(canonical_text(text), value)
    return result


def load_translation_mapping(
    path: Path, log: Callable[[str], None]
) -> dict[tuple[int, int], str]:
    return load_layer_mappings(path, log).by_region


def load_translation_stable_mapping(
    path: Path, log: Callable[[str], None]
) -> dict[str, str]:
    return load_layer_mappings(path, log).by_stable_id


def load_translation_canonicals(
    path: Path, log: Callable[[str], None]
) -> dict[str, str]:
    return load_layer_mappings(path, log).by_text


def save_translation_mapping(
    regions: Sequence[RegionInfo],
    output_path: Path,
    log: PdfLog,
    *,
    cache_identity: Optional[dict] = None,
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
                    "content_bbox": list(region.content_bbox)
                    if region.content_bbox
                    else None,
                    "block_type": region.block_type,
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
                    "vector_font_scale": region.vector_font_scale,
                    "rendered_bbox_points": (
                        list(region.rendered_bbox_points)
                        if region.rendered_bbox_points
                        else None
                    ),
                    "layout_warning": region.layout_warning,
                    "rotation_deg": region.text_orientation_deg,
                    "confidence": region.confidence,
                    "parent_block_id": region.parent_block_id,
                }
            )
        payload_pages.append(
            {
                "page_index": page_index,
                "blocks": blocks_payload,
            }
        )
    payload = {
        "version": 5,
        "cache_identity": cache_identity,
        "pages": payload_pages,
    }
    write_json_atomic(output_path, payload)
    log(f"PDF: сохранён JSON слой: {output_path}")
