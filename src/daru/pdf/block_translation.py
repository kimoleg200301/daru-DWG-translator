"""Translation of OCR regions with shared translation-cache handling."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..translation.checkpoint import TranslationCheckpointStore
from ..translation.engine import TranslationEngine, chunked
from ..translation.job import check_cancelled
from ..translation.progress import report_progress
from .common import _noop_log
from .config import TRANSLATABLE_BLOCK_TYPES
from .layers import canonical_text as _canonicalize_text_for_dedup
from .models import BlockRegion, PdfLog, RegionInfo
from .regions import _iter_candidate_blocks, _iter_translatable_blocks


def _collect_canonical_cache(blocks: Sequence[BlockRegion]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    for block in blocks:
        existing_value = str(block.translated_text or "").strip()
        if not existing_value:
            continue
        canonical = _canonicalize_text_for_dedup(block.source_text)
        if canonical:
            cache.setdefault(canonical, existing_value)
    return cache


def _apply_canonical_translations(blocks: Sequence[BlockRegion], canonical_cache: Dict[str, str]) -> None:
    for block in blocks:
        canonical = _canonicalize_text_for_dedup(block.source_text)
        translated_value = canonical_cache.get(canonical, "").strip()
        if translated_value:
            block.translated_text = translated_value
        elif not block.translated_text:
            block.translated_text = block.source_text


class RegionTranslator:
    """Translate RegionInfo objects in batches using the existing TranslationEngine."""

    def __init__(self, engine: TranslationEngine, log: PdfLog = _noop_log, batch_size: int = 10) -> None:
        self.engine = engine
        self.batch_size = max(1, batch_size)
        self.log = log

    def translate(self, regions: Sequence[RegionInfo]) -> None:
        texts = [region.source_text for region in regions]
        if not texts:
            return
        translated: List[str] = []
        total = len(texts)
        processed = 0
        report_progress("translate", 0, total)
        for chunk in chunked(texts, self.batch_size):
            check_cancelled()
            chunk_translated = self.engine.translate_many(list(chunk))
            translated.extend(chunk_translated)
            processed = min(total, processed + len(chunk_translated))
            report_progress("translate", processed, total)
            percent = int((processed / total) * 100)
            self.log(f"Перевод областей... [{percent}%]")
        for region, translated_text in zip(regions, translated):
            region.translated_text = translated_text or region.source_text


class BlockRegionTranslator:
    """Translate BlockRegion objects as independent units."""

    def __init__(self, engine: TranslationEngine, log: PdfLog = _noop_log, batch_size: int = 10) -> None:
        self.engine = engine
        self.batch_size = max(1, batch_size)
        self.log = log

    def translate(
        self,
        blocks: Sequence[BlockRegion],
        canonical_cache: Optional[Dict[str, str]] = None,
        checkpoint_store: Optional[TranslationCheckpointStore] = None,
    ) -> Dict[str, str]:
        """
        Translate unique block texts once (based on canonical form) and reuse results.
        Returns the canonical cache used for translation.
        """
        targets = [block for block in blocks if block.block_type in TRANSLATABLE_BLOCK_TYPES]
        targets = [block for block in targets if str(block.source_text or "").strip()]
        cache: Dict[str, str] = dict(canonical_cache or {})
        if not targets:
            return cache

        # Pre-seed cache with translations already present on blocks.
        for block in targets:
            existing_value = str(block.translated_text or "").strip()
            if not existing_value:
                continue
            canonical = _canonicalize_text_for_dedup(block.source_text)
            if canonical:
                cache.setdefault(canonical, existing_value)

        queued: List[Tuple[str, str]] = []
        queued_keys: Set[str] = set()
        for block in targets:
            source_text = str(block.source_text or "")
            canonical = _canonicalize_text_for_dedup(source_text)
            if not canonical or not source_text.strip():
                continue
            if canonical in cache or canonical in queued_keys:
                continue
            queued.append((canonical, source_text))
            queued_keys.add(canonical)

        if queued:
            canonical_order = [item[0] for item in queued]
            source_order = [item[1] for item in queued]
            total = len(source_order)
            processed = 0
            report_progress("translate", 0, total)
            for start in range(0, len(source_order), self.batch_size):
                chunk = source_order[start : start + self.batch_size]
                chunk_keys = canonical_order[start : start + self.batch_size]
                chunk_translated = self.engine.translate_many(list(chunk))
                for canonical_key, source_text, translated_text in zip(
                    chunk_keys,
                    chunk,
                    chunk_translated,
                ):
                    final_text = translated_text or source_text
                    cache[canonical_key] = final_text
                    if checkpoint_store is not None and final_text:
                        checkpoint_store.upsert(
                            namespace="pdf-scanned:canonical",
                            block_id=canonical_key,
                            source_text=canonical_key,
                            translated_text=final_text,
                            extra={"sample_source_text": source_text},
                        )
                if checkpoint_store is not None and chunk_translated:
                    checkpoint_store.save()
                processed = min(total, processed + len(chunk_translated))
                report_progress("translate", processed, total)
                percent = int((processed / total) * 100)
                self.log(f"Перевод блоков... [{percent}%]")

        for block in targets:
            canonical = _canonicalize_text_for_dedup(block.source_text)
            translated_value = cache.get(canonical, "").strip()
            if translated_value:
                block.translated_text = translated_value
            else:
                block.translated_text = block.source_text
        return cache


def _apply_cached_translations(blocks: Sequence[BlockRegion], existing: Dict[Tuple[int, int], str]) -> None:
    for block in _iter_candidate_blocks(blocks):
        if block.translated_text:
            continue
        if block.render_id is None:
            continue
        cached_value = str(existing.get((block.page_index, block.render_id), "") or "").strip()
        if cached_value:
            block.translated_text = cached_value


def _apply_stable_cached_translations(
    blocks: Sequence[BlockRegion],
    existing: Dict[str, str],
) -> None:
    if not existing:
        return
    for block in _iter_candidate_blocks(blocks):
        if not block.stable_id:
            continue
        cached_value = str(existing.get(block.stable_id, "") or "").strip()
        if cached_value:
            block.translated_text = cached_value


def _apply_canonical_map(blocks: Sequence[BlockRegion], canonical_map: Dict[str, str]) -> None:
    """Apply translations by canonicalized source text (ignores render_id mismatch)."""
    if not canonical_map:
        return
    for block in _iter_translatable_blocks(blocks):
        if block.translated_text:
            continue
        canonical = _canonicalize_text_for_dedup(block.source_text)
        translated = canonical_map.get(canonical, "").strip()
        if translated:
            block.translated_text = translated


def _apply_translation_caches(
    blocks: Sequence[BlockRegion],
    *,
    stable_map: Dict[str, str],
    legacy_render_map: Dict[Tuple[int, int], str],
    canonical_map: Dict[str, str],
) -> None:
    """Prefer identity-safe caches; render-id fallback is for legacy layers only."""
    _apply_stable_cached_translations(blocks, stable_map)
    if not stable_map:
        _apply_cached_translations(blocks, legacy_render_map)
    _apply_canonical_map(blocks, canonical_map)
