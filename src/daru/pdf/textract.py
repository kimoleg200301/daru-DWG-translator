"""Textract requests, response caching and structured layout extraction."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..translation.job import check_cancelled
from .common import _noop_log, _require_dependency
from .config import (
    FIGURE_BLOCK_TYPES,
    TEXTRACT_LOG_PATH,
    TRANSLATABLE_BLOCK_TYPES,
    PdfProcessingConfig,
)
from .geometry import (
    _bbox_intersection_ratio,
    _block_content_bbox,
    _clamp_bbox_to_page,
    _median_float,
    _union_bbox,
)
from .models import BBoxWH, BlockRegion, LineRegion, PageImage, PdfLog, TextractPageData

try:  # pragma: no cover - optional dependency
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore


def _log_textract_exchange(
    *,
    page_index: int,
    image_bytes: int,
    image_sha256: Optional[str],
    feature_types: Sequence[str],
    response: Dict[str, object],
    log: PdfLog,
    log_path: Path = TEXTRACT_LOG_PATH,
    file_hash: Optional[str] = None,
    input_name: Optional[str] = None,
    render_dpi: Optional[int] = None,
    cache_version: int = 2,
) -> None:
    """
    Append request/response metadata for Textract to the user cache log.
    Response is stored as-is (JSON-serializable) without image bytes to keep size manageable.
    """
    entry = {
        "version": cache_version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "page_index": page_index,
        "file_hash": file_hash,
        "input_name": input_name or "",
        "request": {
            "feature_types": list(feature_types),
            "image_bytes": image_bytes,
            "image_sha256": image_sha256,
            "render_dpi": render_dpi,
            "file_hash": file_hash,
            "input_name": input_name or "",
        },
        "response": {
            "block_count": len(response.get("Blocks", []) if isinstance(response, dict) else []),
            "data": response,
        },
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
    except Exception as exc:  # pragma: no cover - logging should not break pipeline
        log(f"Textract log: не удалось записать обмен для страницы {page_index + 1}: {exc}")


def _load_textract_cache(
    log_path: Path, log: PdfLog, *, file_hash: Optional[str] = None,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    if not log_path.exists():
        return entries
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                check_cancelled()
                clean = line.strip()
                if not clean:
                    continue
                try:
                    entry = json.loads(clean)
                except json.JSONDecodeError:
                    log("Textract log: некорректная JSON-строка, пропускаю")
                    continue
                if isinstance(entry, dict):
                    request = entry.get("request")
                    request = request if isinstance(request, dict) else {}
                    if file_hash and (entry.get("file_hash") or request.get("file_hash")) != file_hash:
                        continue
                    entries.append(entry)
    except (OSError, UnicodeError) as exc:
        log(f"Textract log: не удалось прочитать {log_path}: {exc}")
    return entries


class TextractLayoutExtractor:
    """Builds LineRegion/BlockRegion structures from AWS Textract layout API."""

    def __init__(
        self,
        textract_client: Optional[object],
        log: PdfLog = _noop_log,
        *,
        config: Optional[PdfProcessingConfig] = None,
        log_path: Path = TEXTRACT_LOG_PATH,
        file_hash: Optional[str] = None,
        input_name: str = "",
    ) -> None:
        self.client = textract_client
        self.log = log
        self.config = (config or PdfProcessingConfig()).normalized()
        self.log_path = log_path
        self.file_hash = file_hash
        self.input_name = input_name
        self._cached_entries = self._load_cache()

    def _init_client(self) -> object:
        _require_dependency(boto3, "Установите boto3 (pip install boto3)")
        try:
            return boto3.client("textract")
        except Exception as exc:  # pragma: no cover - network/auth related
            raise RuntimeError(f"Textract: не удалось создать клиент: {exc}") from exc

    def _load_cache(self) -> List[Dict[str, object]]:
        cached_rows = _load_textract_cache(self.log_path, self.log, file_hash=self.file_hash)
        cache: List[Dict[str, object]] = []
        for entry in cached_rows:
            req = entry.get("request", {}) if isinstance(entry, dict) else {}
            if not isinstance(req, dict):
                continue
            resp = entry.get("response", {}) if isinstance(entry, dict) else {}
            data = resp.get("data") if isinstance(resp, dict) else None
            if not isinstance(data, dict):
                continue
            try:
                page_idx = int(entry.get("page_index"))
            except Exception:
                continue
            cache.append(
                {
                    "page_index": page_idx,
                    "file_hash": entry.get("file_hash") or req.get("file_hash"),
                    "image_bytes": req.get("image_bytes"),
                    "image_sha256": req.get("image_sha256"),
                    "feature_types": tuple(entry.get("feature_types") or req.get("feature_types") or []),
                    "response": data,
                }
            )
        if cache:
            self.log(f"Textract: загружено {len(cache)} ответов из JSON-лога {self.log_path}")
        return cache

    def _find_cached_response(self, page_index: int, meta: Dict[str, object]) -> Optional[Dict[str, object]]:
        candidates = [item for item in self._cached_entries if item.get("page_index") == page_index]
        if self.file_hash:
            candidates = [item for item in candidates if item.get("file_hash") == self.file_hash]
        desired_feature_types = meta.get("feature_types")
        if desired_feature_types:
            candidates = [item for item in candidates if item.get("feature_types") == desired_feature_types]
        sha = meta.get("image_sha256")
        if sha:
            for item in reversed(candidates):
                if item.get("image_sha256") == sha:
                    return item.get("response")
        return None

    def analyze(self, page: PageImage) -> TextractPageData:
        """Run Textract analyze_document on the given page image."""
        img_bytes, img_meta = self._encode_image_for_textract(page.image, page.page_index, render_dpi=page.dpi)
        width, height = page.image.size
        feature_types = ["LAYOUT", "SIGNATURES", "TABLES"]
        img_meta["feature_types"] = tuple(feature_types)
        cached = self._find_cached_response(page.page_index, img_meta)
        if cached is not None:
            resp = cached
            self.log(f"Textract: страница {page.page_index + 1} загружена из лога, запрос не отправлялся")
        else:
            if self.client is None:
                self.client = self._init_client()
            try:
                resp = self.client.analyze_document(
                    Document={"Bytes": img_bytes},
                    FeatureTypes=feature_types,
                )
            except Exception as exc:  # pragma: no cover - Textract call can fail remotely
                raise RuntimeError(f"Textract: ошибка анализа страницы {page.page_index + 1}: {exc}") from exc

            _log_textract_exchange(
                page_index=page.page_index,
                image_bytes=img_meta.get("image_bytes", 0),
                image_sha256=img_meta.get("image_sha256"),
                feature_types=feature_types,
                response=resp,
                log=self.log,
                log_path=self.log_path,
                file_hash=self.file_hash,
                input_name=self.input_name,
                render_dpi=img_meta.get("render_dpi"),
            )
        blocks_by_id: Dict[str, Dict[str, object]] = {
            block["Id"]: block for block in resp.get("Blocks", []) if block.get("Id")
        }
        lines = self._build_lines(blocks_by_id, page_index=page.page_index, page_width=width, page_height=height)
        used_line_ids: Set[str] = set()
        blocks = self._build_blocks(
            blocks_by_id,
            lines,
            used_line_ids,
            page_index=page.page_index,
            page_width=width,
            page_height=height,
            page_image=page,
            allow_figures=True,
        )
        self.log(f"Textract: страница {page.page_index + 1} — строк: {len(lines)}, блоков: {len(blocks)}")
        return TextractPageData(page=page, blocks_by_id=blocks_by_id, lines=lines, blocks=blocks)

    def _encode_image_for_textract(
        self, image: "Image.Image", page_index: int, render_dpi: Optional[int] = None
    ) -> Tuple[bytes, Dict[str, object]]:
        """
        Encode page image to PNG, automatically downscaling if the payload exceeds Textract limits (~5 MB).
        Keeps original aspect ratio; coordinates are still mapped using the original page size.
        """
        _require_dependency(Image, "Установите Pillow (pip install pillow)")
        max_bytes = 4_500_000  # keep a buffer below the 5 MB limit
        attempt_image = image
        scale = 1.0
        for attempt in range(6):
            buf = BytesIO()
            attempt_image.save(buf, format="PNG")
            data = buf.getvalue()
            if len(data) <= max_bytes:
                if attempt > 0:
                    w, h = attempt_image.size
                    self.log(
                        f"Textract: страница {page_index + 1} уменьшена до {w}x{h} для ограничения 5 МБ "
                        f"({len(data) / 1024:.1f} КБ)"
                    )
                meta = {
                    "image_bytes": len(data),
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "render_dpi": render_dpi,
                }
                return data, meta
            scale *= 0.85
            new_w = max(1, int(image.width * scale))
            new_h = max(1, int(image.height * scale))
            if new_w < 100 or new_h < 100:
                break
            attempt_image = image.resize((new_w, new_h), Image.LANCZOS)

        # Fallback to JPEG compression if PNG still too large.
        for quality in (85, 70, 60):
            buf = BytesIO()
            attempt_image.save(buf, format="JPEG", quality=quality)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                w, h = attempt_image.size
                self.log(
                    f"Textract: страница {page_index + 1} сжата JPEG q={quality} до {w}x{h} "
                    f"({len(data) / 1024:.1f} КБ)"
                )
                meta = {
                    "image_bytes": len(data),
                    "image_sha256": hashlib.sha256(data).hexdigest(),
                    "render_dpi": render_dpi,
                }
                return data, meta

        raise RuntimeError(
            f"Textract: не удалось уменьшить страницу {page_index + 1} ниже ограничения 5 МБ (последний размер "
            f"{len(data) / 1024:.1f} КБ)"
        )

    def _build_lines(
        self,
        blocks_by_id: Dict[str, Dict[str, object]],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> Dict[str, LineRegion]:
        lines: List[LineRegion] = []
        for block in blocks_by_id.values():
            if str(block.get("BlockType", "")).upper() != "LINE":
                continue
            text = str(block.get("Text", "") or "").strip()
            if not text:
                continue
            bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
            if not bbox:
                continue
            confidence = float(block.get("Confidence") or 0.0)
            rotation_deg = self._rotation_from_geometry(block.get("Geometry"))
            word_boxes: List[BBoxWH] = []
            for rel in block.get("Relationships", []) or []:
                if str(rel.get("Type", "")).upper() != "CHILD":
                    continue
                for child_id in rel.get("Ids", []) or []:
                    child = blocks_by_id.get(child_id)
                    if not child:
                        continue
                    if str(child.get("BlockType", "")).upper() != "WORD":
                        continue
                    word_bbox = self._bbox_from_geometry(child.get("Geometry"), page_width, page_height)
                    if not word_bbox:
                        continue
                    wx0, wy0, wx1, wy1 = word_bbox
                    if wx1 <= wx0 or wy1 <= wy0:
                        continue
                    word_boxes.append((wx0, wy0, wx1 - wx0, wy1 - wy0))
            lines.append(
                LineRegion(
                    page_index=page_index,
                    line_id=str(block["Id"]),
                    bbox=bbox,
                    source_text=text,
                    word_boxes=word_boxes,
                    confidence=confidence,
                    rotation_deg=rotation_deg,
                )
            )
        lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        return {line.line_id: line for line in lines}

    def _build_blocks(
        self,
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
        page_image: Optional[PageImage] = None,
        allow_figures: bool = True,
    ) -> List[BlockRegion]:
        blocks: List[BlockRegion] = []
        figure_blocks: List[Dict[str, object]] = []
        figure_line_ids: Set[str] = set()
        if allow_figures and page_image is not None:
            figure_blocks = [
                block
                for block in blocks_by_id.values()
                if str(block.get("BlockType", "")).upper() in FIGURE_BLOCK_TYPES
            ]
            for figure_block in figure_blocks:
                figure_id = str(figure_block.get("Id", ""))
                if figure_id:
                    figure_line_ids.update(
                        self._collect_line_ids_from_block(
                            figure_id,
                            blocks_by_id,
                            set(),
                        )
                    )
        has_table_cells = any(
            str(block.get("BlockType", "")).upper() in {"CELL", "MERGED_CELL"} for block in blocks_by_id.values()
        )
        cell_bboxes: List[Tuple[int, int, int, int]] = []
        merged_children_map: Dict[str, List[str]] = {}
        expanded_merged_ids: Set[str] = set()
        skip_cell_ids: Set[str] = set()
        if has_table_cells:
            for block in blocks_by_id.values():
                block_type = str(block.get("BlockType", "")).upper()
                if block_type not in {"CELL", "MERGED_CELL"}:
                    continue
                block_id = str(block.get("Id", ""))
                bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
                if bbox:
                    cell_bboxes.append(bbox)
                if block_type == "MERGED_CELL":
                    child_cells: List[str] = []
                    for rel in block.get("Relationships", []) or []:
                        if str(rel.get("Type", "")).upper() != "CHILD":
                            continue
                        for child_id in rel.get("Ids", []) or []:
                            child = blocks_by_id.get(child_id)
                            if not child:
                                continue
                            if str(child.get("BlockType", "")).upper() == "CELL":
                                child_cells.append(str(child_id))
                    try:
                        merged_confidence = float(block.get("Confidence", 0.0) or 0.0)
                    except Exception:
                        merged_confidence = 0.0
                    if (
                        len(child_cells) > 1
                        and merged_confidence < float(self.config.min_confidence)
                    ):
                        expanded_merged_ids.add(block_id)
                    else:
                        skip_cell_ids.update(child_cells)
                    merged_children_map[block_id] = child_cells
        for block in blocks_by_id.values():
            block_type_raw = str(block.get("BlockType", "")).upper()
            if block_type_raw in FIGURE_BLOCK_TYPES and allow_figures and page_image is not None:
                continue
            if has_table_cells and block_type_raw in {"TABLE", "LAYOUT_TABLE"}:
                blocks.extend(
                    self._build_layout_table_lines(
                        block,
                        lines_by_id,
                        used_line_ids,
                        cell_bboxes,
                        page_index=page_index,
                    )
                )
                continue
            if block_type_raw in {"CELL", "MERGED_CELL"}:
                if block_type_raw == "CELL":
                    block_id = str(block.get("Id", ""))
                    if block_id in skip_cell_ids:
                        continue
                    cell_region = self._build_cell_block(
                        block,
                        blocks_by_id,
                        lines_by_id,
                        used_line_ids,
                        page_index=page_index,
                        page_width=page_width,
                        page_height=page_height,
                    )
                else:
                    block_id = str(block.get("Id", ""))
                    if block_id in expanded_merged_ids:
                        continue
                    child_cells = merged_children_map.get(block_id, [])
                    cell_region = self._build_merged_cell_block(
                        block,
                        blocks_by_id,
                        lines_by_id,
                        used_line_ids,
                        child_cells,
                        page_index=page_index,
                        page_width=page_width,
                        page_height=page_height,
                    )
                if cell_region:
                    blocks.extend(self._split_table_cell_zones(cell_region))
                continue
            normalized = self._normalize_text_block_type(block_type_raw)
            if normalized:
                text_blocks = self._build_text_block(
                    block,
                    blocks_by_id,
                    lines_by_id,
                    used_line_ids,
                    figure_line_ids,
                    block_type=normalized,
                    page_index=page_index,
                    page_width=page_width,
                    page_height=page_height,
                )
                blocks.extend(text_blocks)

        for block in figure_blocks:
            figure_container, child_blocks = self._build_figure_block(
                block,
                blocks_by_id,
                lines_by_id,
                used_line_ids,
                page_index=page_index,
                page_width=page_width,
                page_height=page_height,
                page_image=page_image,
            )
            if figure_container:
                blocks.append(figure_container)
            blocks.extend(child_blocks)
        self._finalize_block_metadata(blocks)
        blocks = self._deduplicate_blocks(blocks)
        blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        return blocks

    def _collect_line_ids_from_block(
        self, block_id: str, blocks_by_id: Dict[str, Dict[str, object]], visited: Optional[Set[str]] = None
    ) -> Set[str]:
        visited = visited or set()
        if block_id in visited:
            return set()
        visited.add(block_id)
        target = blocks_by_id.get(block_id)
        if not target:
            return set()
        collected: Set[str] = set()
        for rel in target.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                child_block = blocks_by_id.get(child_id)
                child_type = str(child_block.get("BlockType", "")).upper() if child_block else ""
                if child_type == "LINE":
                    collected.add(str(child_id))
                else:
                    collected.update(self._collect_line_ids_from_block(str(child_id), blocks_by_id, visited))
        return collected

    def _build_text_block(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        figure_line_ids: Set[str],
        *,
        block_type: str,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> List[BlockRegion]:
        line_regions: List[LineRegion] = []
        has_line_children = False
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                if child_id in lines_by_id:
                    has_line_children = True
                if child_id in used_line_ids:
                    continue
                line = lines_by_id.get(child_id)
                if line:
                    line_regions.append(line)
        line_regions = self._dedup_and_sort_lines(line_regions)
        if any(ln.line_id in used_line_ids for ln in line_regions):
            return []
        if any(ln.line_id in figure_line_ids for ln in line_regions):
            return []

        blocks_out: List[BlockRegion] = []
        parent_id = str(block.get("Id", "")) or None
        parent_bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
        if block_type == "LIST" and line_regions:
            groups = self._group_list_lines(line_regions)
            base_id = str(block.get("Id", ""))
            for idx, group in enumerate(groups):
                for line in group:
                    used_line_ids.add(line.line_id)
                group_bbox = _union_bbox([line.bbox for line in group]) or group[0].bbox
                blocks_out.append(
                    BlockRegion(
                        page_index=page_index,
                        block_id=f"{base_id}:item:{idx}" if base_id else f"list:item:{idx}",
                        block_type=block_type,
                        bbox=group_bbox,
                        lines=list(group),
                        source_text="\n".join(
                            line.source_text.strip()
                            for line in group
                            if line.source_text.strip()
                        ),
                        confidence=_median_float([line.confidence for line in group]),
                        rotation_deg=_median_float([line.rotation_deg for line in group]),
                        parent_block_id=parent_id,
                        parent_block_type=block_type,
                        parent_bbox=parent_bbox,
                    )
                )
            return blocks_out
        split_lines = bool(getattr(self.config, "textract_split_lines", False))
        split_by_gap = split_lines and self._should_split_lines_by_horizontal_gap(line_regions)
        if split_by_gap and line_regions:
            base_id = str(block.get("Id", ""))
            for idx, ln in enumerate(line_regions):
                used_line_ids.add(ln.line_id)
                blocks_out.append(
                    BlockRegion(
                        page_index=page_index,
                        block_id=f"{base_id}:{idx}" if base_id else f"line:{idx}",
                        block_type=block_type,
                        bbox=ln.bbox,
                        lines=[ln],
                        cells=[],
                        source_text=ln.source_text,
                        confidence=ln.confidence,
                        rotation_deg=ln.rotation_deg,
                        parent_block_id=parent_id,
                        parent_block_type=block_type,
                        parent_bbox=parent_bbox,
                    )
                )
            return blocks_out

        # Textract occasionally returns one page-spanning LAYOUT_TEXT block
        # containing prose, diagram labels, and table text.  Split only these
        # clearly oversized containers into coherent, aligned line groups.
        raw_layout_type = str(block.get("BlockType", "")).upper()
        is_splittable_layout = raw_layout_type in {
            "LAYOUT_TEXT",
            "LAYOUT_KEY_VALUE",
        }
        automatic_groups = (
            self._group_layout_table_lines(line_regions)
            if is_splittable_layout
            else []
        )
        block_height = (
            max(line.bbox[3] for line in line_regions)
            - min(line.bbox[1] for line in line_regions)
            if line_regions
            else 0
        )
        auto_split = (
            is_splittable_layout
            and len(automatic_groups) > 1
            and len(line_regions) >= 6
            and block_height >= page_height * 0.20
        )
        if auto_split:
            base_id = str(block.get("Id", ""))
            for idx, group in enumerate(automatic_groups):
                group_bbox = _union_bbox([line.bbox for line in group])
                if group_bbox is None:
                    continue
                for line in group:
                    used_line_ids.add(line.line_id)
                blocks_out.append(
                    BlockRegion(
                        page_index=page_index,
                        block_id=f"{base_id}:zone:{idx}" if base_id else f"text:zone:{idx}",
                        block_type=block_type,
                        bbox=group_bbox,
                        lines=list(group),
                        source_text="\n".join(
                            line.source_text.strip()
                            for line in group
                            if line.source_text.strip()
                        ),
                        confidence=_median_float([line.confidence for line in group]),
                        rotation_deg=_median_float([line.rotation_deg for line in group]),
                        parent_block_id=parent_id,
                        parent_block_type=block_type,
                        parent_bbox=parent_bbox,
                    )
                )
            return blocks_out

        bbox = _union_bbox([ln.bbox for ln in line_regions])
        if bbox is None:
            bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
        source_text = "\n".join(line.source_text for line in line_regions).strip() if line_regions else ""
        if not source_text:
            source_text = str(block.get("Text", "") or "").strip()
        if has_line_children and not line_regions:
            return []
        if not source_text or not bbox:
            return []
        for ln in line_regions:
            used_line_ids.add(ln.line_id)
        blocks_out.append(
            BlockRegion(
                page_index=page_index,
                block_id=str(block.get("Id", "")),
                block_type=block_type,
                bbox=bbox if bbox else (line_regions[0].bbox if line_regions else (0, 0, 0, 0)),
                lines=line_regions,
                cells=[],
                source_text=source_text,
                parent_block_id=parent_id,
                parent_block_type=block_type,
                parent_bbox=parent_bbox,
            )
        )
        return blocks_out

    def _group_list_lines(self, lines: Sequence[LineRegion]) -> List[List[LineRegion]]:
        ordered = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
        if len(ordered) <= 1:
            return [list(ordered)] if ordered else []
        heights = [max(1, line.bbox[3] - line.bbox[1]) for line in ordered]
        median_height = _median_float(heights, 1.0)
        bullet_re = re.compile(r"^(?:[-*•▪◦◇◆]|\(?\d+[.)])\s*")
        groups: List[List[LineRegion]] = [[ordered[0]]]
        for line in ordered[1:]:
            previous = groups[-1][-1]
            vertical_gap = max(0, line.bbox[1] - previous.bbox[3])
            starts_item = bool(bullet_re.match(line.source_text.strip()))
            same_indent = abs(line.bbox[0] - groups[-1][0].bbox[0]) <= median_height
            if starts_item or (same_indent and vertical_gap > median_height * 0.75):
                groups.append([line])
            else:
                groups[-1].append(line)
        return groups

    def _line_center_in_any_bbox(self, line: LineRegion, bboxes: Sequence[Tuple[int, int, int, int]]) -> bool:
        if not bboxes:
            return False
        x0, y0, x1, y1 = line.bbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        for bx0, by0, bx1, by1 in bboxes:
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                return True
        return False

    def _bbox_intersection_ratio(
        self, inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]
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

    def _build_layout_table_lines(
        self,
        block: Dict[str, object],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        cell_bboxes: Sequence[Tuple[int, int, int, int]],
        *,
        page_index: int,
    ) -> List[BlockRegion]:
        line_regions: List[LineRegion] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                if child_id in used_line_ids:
                    continue
                line = lines_by_id.get(child_id)
                if line:
                    line_regions.append(line)
        line_regions = self._dedup_and_sort_lines(line_regions)
        if cell_bboxes:
            line_regions = [ln for ln in line_regions if not self._line_center_in_any_bbox(ln, cell_bboxes)]
        if not line_regions:
            return []
        grouped = self._group_layout_table_lines(line_regions)
        blocks_out: List[BlockRegion] = []
        base_id = str(block.get("Id", ""))
        for idx, group in enumerate(grouped):
            for ln in group:
                used_line_ids.add(ln.line_id)
            bbox = _union_bbox([ln.bbox for ln in group]) or group[0].bbox
            source_text = " ".join(ln.source_text.strip() for ln in group if ln.source_text.strip())
            word_boxes: List[BBoxWH] = []
            for ln in group:
                if ln.word_boxes:
                    word_boxes.extend(ln.word_boxes)
            block_id = f"{base_id}:para:{idx}" if base_id else f"layout_table:para:{idx}"
            blocks_out.append(
                BlockRegion(
                    page_index=page_index,
                    block_id=block_id,
                    block_type="TEXT",
                    bbox=bbox,
                    lines=group,
                    cells=[],
                    source_text=source_text,
                    word_boxes=word_boxes,
                    is_table=True,
                    parent_block_id=base_id or None,
                    parent_block_type="LAYOUT_TABLE",
                    parent_bbox=bbox,
                )
            )
        return blocks_out

    def _group_layout_table_lines(self, lines: Sequence[LineRegion]) -> List[List[LineRegion]]:
        if not lines:
            return []
        ordered = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        heights = [max(1, ln.bbox[3] - ln.bbox[1]) for ln in ordered]
        median_h = float(np.median(heights)) if heights else 1.0
        gap_threshold = max(2.0, 0.6 * median_h)
        align_threshold = max(6.0, 0.6 * median_h)
        groups: List[List[LineRegion]] = []
        for ln in ordered:
            line_width = max(1, ln.bbox[2] - ln.bbox[0])
            matches: List[Tuple[float, float, int]] = []
            for group_index, group in enumerate(groups):
                prev = group[-1]
                gap = max(0.0, float(ln.bbox[1] - prev.bbox[3]))
                left_delta = abs(ln.bbox[0] - prev.bbox[0])
                prev_width = max(1, prev.bbox[2] - prev.bbox[0])
                center_delta = abs(
                    ((ln.bbox[0] + ln.bbox[2]) / 2.0)
                    - ((prev.bbox[0] + prev.bbox[2]) / 2.0)
                )
                width_ratio = min(prev_width, line_width) / max(prev_width, line_width)
                same_column = left_delta <= max(
                    align_threshold,
                    min(prev_width, line_width) * 0.12,
                )
                centered_stack = center_delta <= align_threshold and width_ratio >= 0.45
                if gap <= gap_threshold and (same_column or centered_stack):
                    matches.append((gap, min(left_delta, center_delta), group_index))
            if matches:
                groups[min(matches)[2]].append(ln)
            else:
                groups.append([ln])
        return groups

    def _should_split_lines_by_horizontal_gap(self, lines: Sequence[LineRegion]) -> bool:
        """Detect a significant horizontal gap between lines on the same row."""
        if len(lines) < 2:
            return False
        heights = [max(1, ln.bbox[3] - ln.bbox[1]) for ln in lines]
        median_h = float(np.median(heights)) if heights else 1.0
        gap_threshold = max(4.0, min(3.0, self.config.merge_x_gap_ratio) * median_h)
        ordered = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                if not self._lines_share_row(a, b, median_h):
                    continue
                gap = self._horizontal_gap(a, b)
                if gap > gap_threshold:
                    return True
        return False

    def _lines_share_row(self, a: LineRegion, b: LineRegion, median_h: float) -> bool:
        ay0, ay1 = a.bbox[1], a.bbox[3]
        by0, by1 = b.bbox[1], b.bbox[3]
        overlap = min(ay1, by1) - max(ay0, by0)
        min_h = float(min(max(1, ay1 - ay0), max(1, by1 - by0)))
        if overlap > 0 and (overlap / min_h) >= 0.5:
            return True
        cy_diff = abs(((ay0 + ay1) / 2.0) - ((by0 + by1) / 2.0))
        return cy_diff <= max(4.0, 0.4 * median_h)

    def _horizontal_gap(self, a: LineRegion, b: LineRegion) -> float:
        ax0, ax1 = a.bbox[0], a.bbox[2]
        bx0, bx1 = b.bbox[0], b.bbox[2]
        overlap = min(ax1, bx1) - max(ax0, bx0)
        if overlap >= 0:
            return 0.0
        return float(bx0 - ax1 if ax1 < bx0 else ax0 - bx1)

    def _build_cell_block(
        self,
        cell_block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> Optional[BlockRegion]:
        bbox = self._bbox_from_geometry(cell_block.get("Geometry"), page_width, page_height)
        if not bbox:
            return None
        original_bbox = bbox
        text, lines = self._collect_text(cell_block, blocks_by_id, lines_by_id, used_line_ids)
        if any(ln.line_id in used_line_ids for ln in lines):
            return None
        if not lines:
            lines = self._build_cell_word_lines(
                cell_block,
                blocks_by_id,
                page_index=page_index,
                page_width=page_width,
                page_height=page_height,
            )
            if lines:
                text = "\n".join(line.source_text for line in lines if line.source_text)
        # Directly reconstructed line words are more reliable than a recursive
        # CELL traversal: low-confidence merged cells may expose unrelated WORD
        # children outside the visible label.
        line_word_boxes = [box for line in lines for box in line.word_boxes]
        word_boxes = line_word_boxes or self._collect_word_boxes(
            cell_block,
            blocks_by_id,
            page_width,
            page_height,
        )
        content_bounds = _union_bbox(
            [(x, y, x + width, y + height) for x, y, width, height in word_boxes]
        )
        if content_bounds is not None:
            bx0, by0, bx1, by1 = bbox
            cx0, cy0, cx1, cy1 = content_bounds
            median_height = _median_float([height for _, _, _, height in word_boxes], 1.0)
            tolerance = max(2, int(round(median_height * 0.10)))
            content_is_clipped = (
                cx0 < bx0 - tolerance
                or cy0 < by0 - tolerance
                or cx1 > bx1 + tolerance
                or cy1 > by1 + tolerance
            )
            if content_is_clipped:
                pad_x = max(4, int(round(median_height * 0.60)))
                pad_y = max(3, int(round(median_height * 0.35)))
                recovered = _clamp_bbox_to_page(
                    (
                        cx0 - pad_x if cx0 < bx0 - tolerance else bx0,
                        cy0 - pad_y if cy0 < by0 - tolerance else by0,
                        cx1 + pad_x if cx1 > bx1 + tolerance else bx1,
                        cy1 + pad_y if cy1 > by1 + tolerance else by1,
                    ),
                    page_width,
                    page_height,
                )
                if recovered is not None:
                    bbox = recovered
        for ln in lines:
            used_line_ids.add(ln.line_id)
        return BlockRegion(
            page_index=page_index,
            block_id=str(cell_block.get("Id", "")),
            block_type="CELL",
            bbox=bbox,
            lines=lines,
            cells=[],
            source_text=text.strip(),
            word_boxes=word_boxes,
            is_table=True,
            parent_block_id=str(cell_block.get("Id", "")) or None,
            parent_block_type="CELL",
            parent_bbox=original_bbox,
        )

    def _build_cell_word_lines(
        self,
        cell_block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> List[LineRegion]:
        """Reconstruct table-cell lines when Textract relates CELL directly to WORD."""
        words: List[Tuple[Tuple[int, int, int, int], str, float]] = []
        for relationship in cell_block.get("Relationships", []) or []:
            if str(relationship.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in relationship.get("Ids", []) or []:
                child = blocks_by_id.get(child_id)
                if not child or str(child.get("BlockType", "")).upper() != "WORD":
                    continue
                text = str(child.get("Text", "") or "").strip()
                bbox = self._bbox_from_geometry(child.get("Geometry"), page_width, page_height)
                if not text or bbox is None:
                    continue
                try:
                    confidence = float(child.get("Confidence", 0.0) or 0.0)
                except Exception:
                    confidence = 0.0
                if confidence < max(15.0, float(self.config.min_confidence) * 0.5):
                    continue
                words.append((bbox, text, confidence))
        if not words:
            return []
        words.sort(key=lambda item: (item[0][1], item[0][0]))
        median_height = _median_float(
            [bbox[3] - bbox[1] for bbox, _text, _confidence in words],
            1.0,
        )
        rows: List[List[Tuple[Tuple[int, int, int, int], str, float]]] = []
        for word in words:
            bbox = word[0]
            center_y = (bbox[1] + bbox[3]) / 2.0
            target: Optional[List[Tuple[Tuple[int, int, int, int], str, float]]] = None
            for row in reversed(rows):
                row_box = _union_bbox([item[0] for item in row])
                if row_box is None:
                    continue
                row_center_y = (row_box[1] + row_box[3]) / 2.0
                overlap = max(0, min(row_box[3], bbox[3]) - max(row_box[1], bbox[1]))
                min_height = max(1, min(row_box[3] - row_box[1], bbox[3] - bbox[1]))
                if overlap / min_height >= 0.35 or abs(center_y - row_center_y) <= median_height * 0.55:
                    target = row
                    break
                if center_y - row_center_y > median_height * 1.5:
                    break
            if target is None:
                rows.append([word])
            else:
                target.append(word)

        cell_id = str(cell_block.get("Id", "cell"))
        lines: List[LineRegion] = []
        for index, row in enumerate(rows):
            row.sort(key=lambda item: item[0][0])
            bbox = _union_bbox([item[0] for item in row])
            if bbox is None:
                continue
            lines.append(
                LineRegion(
                    page_index=page_index,
                    line_id=f"{cell_id}:word-line:{index}",
                    bbox=bbox,
                    source_text=" ".join(item[1] for item in row),
                    word_boxes=[
                        (box[0], box[1], box[2] - box[0], box[3] - box[1])
                        for box, _text, _confidence in row
                    ],
                    confidence=_median_float([item[2] for item in row]),
                    parent_block_id=cell_id,
                )
            )
        return lines

    def _split_table_cell_zones(self, block: BlockRegion) -> List[BlockRegion]:
        """Separate text bands divided by photos or diagrams inside a table cell."""
        horizontal_zones = self._split_single_line_cell_by_gaps(block)
        if len(horizontal_zones) > 1:
            return horizontal_zones
        ordered = sorted(block.lines, key=lambda line: (line.bbox[1], line.bbox[0]))
        if len(ordered) < 2:
            return [block]
        median_height = _median_float(
            [line.bbox[3] - line.bbox[1] for line in ordered],
            1.0,
        )
        groups: List[List[LineRegion]] = [[ordered[0]]]
        for line in ordered[1:]:
            gap = line.bbox[1] - groups[-1][-1].bbox[3]
            # A gap of roughly one-and-a-half glyph heights already denotes a
            # separate label in technical cells; ordinary wrapped lines are
            # much closer together.
            if gap > max(24.0, median_height * 1.5):
                groups.append([line])
            else:
                groups[-1].append(line)
        if len(groups) == 1:
            return [block]

        cell_x0, cell_y0, cell_x1, cell_y1 = block.bbox
        zones: List[BlockRegion] = []
        for index, group in enumerate(groups):
            content_box = _union_bbox([line.bbox for line in group])
            if content_box is None:
                continue
            previous_bottom = groups[index - 1][-1].bbox[3] if index > 0 else cell_y0
            next_top = groups[index + 1][0].bbox[1] if index + 1 < len(groups) else cell_y1
            top = (
                max(cell_y0, int((previous_bottom + content_box[1]) / 2))
                if index > 0
                else cell_y0
            )
            bottom = (
                min(cell_y1, int((content_box[3] + next_top) / 2))
                if index + 1 < len(groups)
                else cell_y1
            )
            vertical_margin = max(2, int(median_height * 0.35))
            top = max(top, content_box[1] - vertical_margin)
            bottom = min(bottom, content_box[3] + vertical_margin)
            if bottom <= top:
                top, bottom = content_box[1], content_box[3]
            zones.append(
                BlockRegion(
                    page_index=block.page_index,
                    block_id=f"{block.block_id}:zone:{index}",
                    block_type="CELL",
                    bbox=(cell_x0, top, cell_x1, bottom),
                    lines=list(group),
                    source_text="\n".join(
                        line.source_text.strip()
                        for line in group
                        if line.source_text.strip()
                    ),
                    word_boxes=[box for line in group for box in line.word_boxes],
                    is_table=True,
                    confidence=_median_float([line.confidence for line in group]),
                    rotation_deg=_median_float([line.rotation_deg for line in group]),
                    parent_block_id=block.parent_block_id or block.block_id,
                    parent_block_type=block.parent_block_type or "CELL",
                    parent_bbox=block.parent_bbox or block.bbox,
                )
            )
        return zones or [block]

    def _split_single_line_cell_by_gaps(
        self,
        block: BlockRegion,
    ) -> List[BlockRegion]:
        """Split marker, description, and quantity columns inside one OCR cell."""
        if len(block.lines) != 1:
            return [block]
        line = block.lines[0]
        boxes = sorted(line.word_boxes, key=lambda box: (box[0], box[1]))
        tokens = re.findall(r"\S+", str(line.source_text or ""))
        if len(boxes) < 2 or len(tokens) != len(boxes):
            return [block]
        median_height = _median_float([box[3] for box in boxes], 1.0)
        cell_width = max(1, block.bbox[2] - block.bbox[0])
        gap_threshold = max(24.0, median_height * 1.25, cell_width * 0.06)
        grouped: List[List[Tuple[str, BBoxWH]]] = [[(tokens[0], boxes[0])]]
        first_token = tokens[0].strip()
        first_box = boxes[0]
        first_center_y = first_box[1] + first_box[3] / 2.0
        for token_index, (token, box) in enumerate(
            zip(tokens[1:], boxes[1:]),
            start=1,
        ):
            previous_box = grouped[-1][-1][1]
            previous_right = previous_box[0] + previous_box[2]
            center_y = box[1] + box[3] / 2.0
            leading_circle_marker = (
                token_index == 1
                and re.fullmatch(r"[A-D]", first_token, flags=re.IGNORECASE)
                and first_box[2] <= median_height * 1.25
                and abs(center_y - first_center_y) >= median_height * 0.25
            )
            # One-pixel tolerance compensates for integer bbox quantization.
            if leading_circle_marker or box[0] - previous_right + 1.0 >= gap_threshold:
                grouped.append([])
            grouped[-1].append((token, box))
        if len(grouped) == 1:
            return [block]

        cell_x0, cell_y0, cell_x1, cell_y1 = block.bbox
        content_boxes = [
            _union_bbox(
                [
                    (box[0], box[1], box[0] + box[2], box[1] + box[3])
                    for _token, box in group
                ]
            )
            for group in grouped
        ]
        if any(box is None for box in content_boxes):
            return [block]
        concrete_boxes = [box for box in content_boxes if box is not None]
        zones: List[BlockRegion] = []
        for index, (group, content_box) in enumerate(zip(grouped, concrete_boxes)):
            left = (
                cell_x0
                if index == 0
                else int((concrete_boxes[index - 1][2] + content_box[0]) / 2)
            )
            right = (
                cell_x1
                if index + 1 == len(concrete_boxes)
                else int((content_box[2] + concrete_boxes[index + 1][0]) / 2)
            )
            if right <= left:
                left, right = content_box[0], content_box[2]
            group_text = " ".join(token for token, _box in group)
            group_word_boxes = [box for _token, box in group]
            if re.fullmatch(r"[A-D]", group_text, flags=re.IGNORECASE):
                group_alignment = "center"
            elif re.fullmatch(r"\d+(?:[.,]\d+)?\s*(?:pcs?\.?|pieces?)", group_text, flags=re.IGNORECASE):
                group_alignment = "right"
            else:
                group_alignment = "left"
            zones.append(
                BlockRegion(
                    page_index=block.page_index,
                    block_id=f"{block.block_id}:column:{index}",
                    block_type="CELL",
                    bbox=(left, cell_y0, right, cell_y1),
                    lines=[
                        LineRegion(
                            page_index=line.page_index,
                            line_id=f"{line.line_id}:column:{index}",
                            bbox=content_box,
                            source_text=group_text,
                            word_boxes=group_word_boxes,
                            confidence=line.confidence,
                            rotation_deg=line.rotation_deg,
                            parent_block_id=block.block_id,
                        )
                    ],
                    source_text=group_text,
                    word_boxes=group_word_boxes,
                    is_table=True,
                    confidence=line.confidence or block.confidence,
                    rotation_deg=line.rotation_deg or block.rotation_deg,
                    parent_block_id=block.parent_block_id or block.block_id,
                    parent_block_type=block.parent_block_type or "CELL",
                    parent_bbox=block.parent_bbox or block.bbox,
                    alignment=group_alignment,
                    alignment_locked=True,
                    font_weight=block.font_weight,
                    semantic_type=block.semantic_type,
                    fit_strategy=block.fit_strategy,
                )
            )
        return zones or [block]

    def _build_merged_cell_block(
        self,
        merged_block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        child_cell_ids: Sequence[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
    ) -> Optional[BlockRegion]:
        bbox = self._bbox_from_geometry(merged_block.get("Geometry"), page_width, page_height)
        if not bbox:
            return None
        ordered_texts: List[Tuple[int, int, str]] = []
        word_boxes: List[BBoxWH] = []
        for child_id in child_cell_ids:
            child = blocks_by_id.get(child_id)
            if not child:
                continue
            if str(child.get("BlockType", "")).upper() != "CELL":
                continue
            text, _ = self._collect_text(child, blocks_by_id, lines_by_id, None)
            text = text.strip()
            if text:
                row_idx = int(child.get("RowIndex") or 0)
                col_idx = int(child.get("ColumnIndex") or 0)
                ordered_texts.append((row_idx, col_idx, text))
            word_boxes.extend(self._collect_word_boxes(child, blocks_by_id, page_width, page_height))
        source_text = ""
        if ordered_texts:
            ordered_texts.sort(key=lambda t: (t[0], t[1]))
            source_text = " ".join(text for _, _, text in ordered_texts if text)
        if not source_text:
            fallback_text, lines = self._collect_text(merged_block, blocks_by_id, lines_by_id, used_line_ids)
            source_text = fallback_text.strip()
        if not source_text:
            return None
        return BlockRegion(
            page_index=page_index,
            block_id=str(merged_block.get("Id", "")),
            block_type="CELL",
            bbox=bbox,
            lines=[],
            cells=[],
            source_text=source_text,
            word_boxes=word_boxes,
            is_table=True,
            parent_block_id=str(merged_block.get("Id", "")) or None,
            parent_block_type="MERGED_CELL",
            parent_bbox=bbox,
        )

    def _build_figure_block(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Set[str],
        *,
        page_index: int,
        page_width: int,
        page_height: int,
        page_image: PageImage,
    ) -> Tuple[Optional[BlockRegion], List[BlockRegion]]:
        bbox = self._bbox_from_geometry(block.get("Geometry"), page_width, page_height)
        child_ids: List[str] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                child_ids.append(str(child_id))

        line_ids = self._collect_line_ids_from_block(str(block.get("Id", "")), blocks_by_id, set())
        ordered_lines: List[LineRegion] = []
        for line_id in line_ids:
            line = lines_by_id.get(line_id)
            if line:
                ordered_lines.append(line)
        ordered_lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        for line in ordered_lines:
            used_line_ids.add(line.line_id)
        figure_id = str(block.get("Id", ""))
        usable_lines = [line for line in ordered_lines if self._is_usable_figure_line(line)]
        if usable_lines:
            detected_blocks = self._figure_blocks_from_lines(
                usable_lines,
                figure_id=figure_id,
                figure_bbox=bbox,
                page_index=page_index,
            )
        else:
            detected_blocks = self._analyze_figure_region(bbox, page_image, block_id=figure_id)
        container: Optional[BlockRegion] = None
        if bbox:
            container = BlockRegion(
                page_index=page_index,
                block_id=figure_id,
                block_type="FIGURE",
                bbox=bbox,
                lines=[],
                cells=[],
                child_ids=child_ids,
                source_text="",
            )
        return container, detected_blocks

    def _is_usable_figure_line(self, line: LineRegion) -> bool:
        text = re.sub(r"\s+", " ", str(line.source_text or "")).strip()
        if not text:
            return False
        if line.confidence and line.confidence < max(40.0, float(self.config.min_confidence) - 10.0):
            return False
        alnum_count = sum(character.isalnum() for character in text)
        letter_count = sum(character.isalpha() for character in text)
        if alnum_count < 2 or (letter_count == 0 and alnum_count < 4):
            return False
        if len(text) <= 3 and text.isupper():
            # Short diagram labels such as "AND" are meaningful. Keep them only
            # when Textract is highly confident; this still rejects common OCR
            # fragments such as "#2", stray punctuation, and uncertain initials.
            if line.confidence < 90.0 or letter_count < 2:
                return False
        symbols = sum(not character.isalnum() and not character.isspace() for character in text)
        return symbols <= max(4, int(len(text) * 0.55))

    def _figure_blocks_from_lines(
        self,
        lines: Sequence[LineRegion],
        *,
        figure_id: str,
        figure_bbox: Optional[Tuple[int, int, int, int]],
        page_index: int,
    ) -> List[BlockRegion]:
        groups = self._group_layout_table_lines(lines)
        output: List[BlockRegion] = []
        for idx, group in enumerate(groups):
            bbox = _union_bbox([line.bbox for line in group])
            if bbox is None:
                continue
            text = "\n".join(line.source_text.strip() for line in group if line.source_text.strip()).strip()
            if not text:
                continue
            word_boxes = [box for line in group for box in line.word_boxes]
            output.append(
                BlockRegion(
                    page_index=page_index,
                    block_id=f"{figure_id}:text:{idx}",
                    block_type="TEXT",
                    bbox=bbox,
                    lines=list(group),
                    source_text=text,
                    word_boxes=word_boxes,
                    confidence=_median_float([line.confidence for line in group]),
                    rotation_deg=_median_float([line.rotation_deg for line in group]),
                    parent_block_id=figure_id or None,
                    parent_block_type="FIGURE",
                    parent_bbox=figure_bbox,
                )
            )
        return output

    def _analyze_figure_region(self, bbox: Optional[Tuple[int, int, int, int]], page: PageImage, block_id: str) -> List[BlockRegion]:
        if not bbox:
            return []
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(page.width, x0))
        y0 = max(0, min(page.height, y0))
        x1 = max(x0, min(page.width, x1))
        y1 = max(y0, min(page.height, y1))
        if x1 <= x0 or y1 <= y0:
            return []

        _require_dependency(Image, "Установите Pillow (pip install pillow)")
        crop = page.image.crop((x0, y0, x1, y1)).convert("RGB")
        img_bytes, img_meta = self._encode_image_for_textract(crop, page.page_index, render_dpi=page.dpi)
        feature_types = ["LAYOUT", "SIGNATURES"]
        img_meta["feature_types"] = tuple(feature_types)
        cached = self._find_cached_response(page.page_index, img_meta)
        if cached is not None:
            resp = cached
            self.log(f"Textract: фигура {block_id} на стр. {page.page_index + 1} загружена из лога")
        else:
            if self.client is None:
                self.client = self._init_client()
            try:
                resp = self.client.analyze_document(
                    Document={"Bytes": img_bytes},
                    FeatureTypes=feature_types,
                )
            except Exception as exc:
                raise RuntimeError(f"Textract: ошибка анализа фигуры на странице {page.page_index + 1}: {exc}") from exc

            _log_textract_exchange(
                page_index=page.page_index,
                image_bytes=img_meta.get("image_bytes", 0),
                image_sha256=img_meta.get("image_sha256"),
                feature_types=feature_types,
                response=resp,
                log=self.log,
                log_path=self.log_path,
                file_hash=self.file_hash,
                input_name=f"{self.input_name}#figure:{block_id}",
                render_dpi=img_meta.get("render_dpi"),
            )

        blocks_by_id: Dict[str, Dict[str, object]] = {
            blk["Id"]: blk for blk in resp.get("Blocks", []) if blk.get("Id")
        }
        lines = self._build_lines(
            blocks_by_id,
            page_index=page.page_index,
            page_width=crop.width,
            page_height=crop.height,
        )

        # Shift sub-block coordinates back to original page space.
        shifted_lines: List[LineRegion] = []
        for line in lines.values():
            if not self._is_usable_figure_line(line):
                continue
            lx0, ly0, lx1, ly1 = line.bbox
            shifted_bbox = (lx0 + x0, ly0 + y0, lx1 + x0, ly1 + y0)
            shifted_word_boxes: List[BBoxWH] = []
            for wx, wy, ww, wh in line.word_boxes:
                shifted_word_boxes.append((wx + x0, wy + y0, ww, wh))
            shifted_line = LineRegion(
                page_index=line.page_index,
                line_id=line.line_id,
                bbox=shifted_bbox,
                source_text=line.source_text,
                translated_text=line.translated_text,
                word_boxes=shifted_word_boxes,
                confidence=line.confidence,
                rotation_deg=line.rotation_deg,
                parent_block_id=block_id or None,
            )
            shifted_lines.append(shifted_line)
        return self._figure_blocks_from_lines(
            shifted_lines,
            figure_id=block_id,
            figure_bbox=bbox,
            page_index=page.page_index,
        )

    def _finalize_block_metadata(self, blocks: Sequence[BlockRegion]) -> None:
        for block in blocks:
            if block.block_type not in TRANSLATABLE_BLOCK_TYPES:
                continue
            if block.lines:
                for line in block.lines:
                    if line.parent_block_id is None:
                        line.parent_block_id = block.parent_block_id or block.block_id
                if not block.word_boxes:
                    block.word_boxes = [box for line in block.lines for box in line.word_boxes]
                if block.confidence <= 0:
                    block.confidence = _median_float([line.confidence for line in block.lines])
                if abs(block.rotation_deg) < 1e-6:
                    block.rotation_deg = _median_float([line.rotation_deg for line in block.lines])
            if not block.alignment_locked:
                block.alignment = self._infer_alignment(block)

    def _infer_alignment(self, block: BlockRegion) -> str:
        if block.block_type == "TITLE":
            return "center"
        content_bbox = _block_content_bbox(block) or block.bbox
        container = block.parent_bbox or block.bbox
        if content_bbox is None:
            return "left"
        left_gap = max(0, content_bbox[0] - container[0])
        right_gap = max(0, container[2] - content_bbox[2])
        width = max(1, container[2] - container[0])
        if abs(left_gap - right_gap) <= width * 0.12:
            return "center"
        if right_gap <= width * 0.08 and left_gap > right_gap * 1.5:
            return "right"
        return "left"

    def _deduplicate_blocks(self, blocks: Sequence[BlockRegion]) -> List[BlockRegion]:
        blocks = self._remove_table_text_overlays(blocks)
        kept: List[BlockRegion] = []
        removed = 0

        def priority(block: BlockRegion) -> Tuple[int, float, int]:
            type_priority = {
                "CELL": 5,
                "TITLE": 4,
                "LIST": 4,
                "KEY": 4,
                "VALUE": 4,
                "TEXT": 3,
            }.get(block.block_type, 1)
            return (type_priority, block.confidence, len(block.source_text or ""))

        for block in blocks:
            if block.block_type not in TRANSLATABLE_BLOCK_TYPES:
                kept.append(block)
                continue
            canonical = re.sub(r"\s+", " ", str(block.source_text or "")).strip().casefold()
            if not canonical:
                continue
            duplicate_index: Optional[int] = None
            for idx, previous in enumerate(kept):
                if previous.block_type not in TRANSLATABLE_BLOCK_TYPES:
                    continue
                previous_canonical = re.sub(
                    r"\s+", " ", str(previous.source_text or "")
                ).strip().casefold()
                if canonical != previous_canonical:
                    continue
                overlap = max(
                    _bbox_intersection_ratio(block.bbox, previous.bbox),
                    _bbox_intersection_ratio(previous.bbox, block.bbox),
                )
                if overlap >= 0.78:
                    duplicate_index = idx
                    break
            if duplicate_index is None:
                kept.append(block)
                continue
            removed += 1
            previous = kept[duplicate_index]
            if priority(block) > priority(previous):
                kept[duplicate_index] = block
        if removed:
            self.log(f"Textract: удалено геометрических дублей блоков: {removed}")
        return self._remove_composite_text_duplicates(kept)

    def _remove_composite_text_duplicates(
        self,
        blocks: Sequence[BlockRegion],
    ) -> List[BlockRegion]:
        """Drop a broad OCR block when smaller child blocks contain the same text."""

        def canonical(block: BlockRegion) -> str:
            return re.sub(r"\s+", " ", str(block.source_text or "")).strip().casefold()

        removed_ids: Set[int] = set()
        for candidate in blocks:
            candidate_text = canonical(candidate)
            if candidate.block_type not in {"TEXT", "TITLE"} or len(candidate_text) < 8:
                continue
            candidate_area = max(
                1,
                (candidate.bbox[2] - candidate.bbox[0])
                * (candidate.bbox[3] - candidate.bbox[1]),
            )
            children = []
            for other in blocks:
                if other is candidate or other.block_type not in {"TEXT", "TITLE"}:
                    continue
                if candidate.parent_block_id != other.parent_block_id:
                    continue
                other_text = canonical(other)
                if not other_text or other_text not in candidate_text:
                    continue
                other_area = max(
                    1,
                    (other.bbox[2] - other.bbox[0])
                    * (other.bbox[3] - other.bbox[1]),
                )
                if other_area >= candidate_area * 0.75:
                    continue
                if _bbox_intersection_ratio(other.bbox, candidate.bbox) < 0.90:
                    continue
                children.append(other)
            if len(children) < 2:
                continue
            covered = [False] * len(candidate_text)
            for child in children:
                child_text = canonical(child)
                if not child_text:
                    continue
                best_start: Optional[int] = None
                best_gain = 0
                search_from = 0
                while True:
                    start = candidate_text.find(child_text, search_from)
                    if start < 0:
                        break
                    gain = sum(
                        not covered[position]
                        for position in range(start, start + len(child_text))
                    )
                    if gain > best_gain:
                        best_start = start
                        best_gain = gain
                    search_from = start + 1
                if best_start is not None:
                    for position in range(best_start, best_start + len(child_text)):
                        covered[position] = True
            uncovered_has_text = any(
                character.isalnum()
                for index, character in enumerate(candidate_text)
                if not covered[index]
            )
            if not uncovered_has_text:
                removed_ids.add(id(candidate))
        if removed_ids:
            self.log(
                f"Textract: удалено составных дублей подписей: {len(removed_ids)}"
            )
        return [block for block in blocks if id(block) not in removed_ids]

    def _remove_table_text_overlays(self, blocks: Sequence[BlockRegion]) -> List[BlockRegion]:
        cells = [
            block
            for block in blocks
            if block.block_type == "CELL" and block.bbox[2] > block.bbox[0] and block.bbox[3] > block.bbox[1]
        ]
        if not cells:
            return list(blocks)

        def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
            x0 = max(a[0], b[0])
            y0 = max(a[1], b[1])
            x1 = min(a[2], b[2])
            y1 = min(a[3], b[3])
            if x1 <= x0 or y1 <= y0:
                return 0
            return (x1 - x0) * (y1 - y0)

        filtered: List[BlockRegion] = []
        removed = 0
        for block in blocks:
            if block.block_type == "CELL" or block.block_type not in TRANSLATABLE_BLOCK_TYPES or block.is_table:
                filtered.append(block)
                continue
            area = max(1, (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1]))
            overlap_area = 0
            overlap_cells = 0
            block_text = re.sub(r"\s+", " ", str(block.source_text or "")).strip().casefold()
            covered_by_cell_text = False
            for cell in cells:
                inter = intersection_area(block.bbox, cell.bbox)
                if inter <= 0:
                    continue
                overlap_area += inter
                overlap_cells += 1
                cell_text = re.sub(
                    r"\s+", " ", str(cell.source_text or "")
                ).strip().casefold()
                if (
                    len(block_text) >= 4
                    and block_text in cell_text
                    and inter / float(area) >= 0.80
                ):
                    covered_by_cell_text = True
            if covered_by_cell_text or (
                overlap_cells >= 2 and overlap_area / float(area) >= 0.45
            ):
                removed += 1
                continue
            filtered.append(block)
        if removed:
            self.log(f"Textract: removed table text overlays: {removed}")
        return filtered

    def _collect_text(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        lines_by_id: Dict[str, LineRegion],
        used_line_ids: Optional[Set[str]] = None,
    ) -> Tuple[str, List[LineRegion]]:
        line_regions: List[LineRegion] = []
        words: List[str] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                if used_line_ids is not None and child_id in used_line_ids:
                    continue
                child = blocks_by_id.get(child_id)
                if not child:
                    continue
                child_type = str(child.get("BlockType", "")).upper()
                if child_type == "LINE":
                    line = lines_by_id.get(child_id)
                    if line:
                        line_regions.append(line)
                elif child_type == "WORD":
                    word_text = str(child.get("Text", "") or "").strip()
                    if word_text:
                        words.append(word_text)
        line_regions = self._dedup_and_sort_lines(line_regions)
        if line_regions:
            joined = " ".join(line.source_text.strip() for line in line_regions if line.source_text.strip())
            return joined, line_regions
        return " ".join(words).strip(), []

    def _collect_word_boxes(
        self,
        block: Dict[str, object],
        blocks_by_id: Dict[str, Dict[str, object]],
        page_width: int,
        page_height: int,
    ) -> List[BBoxWH]:
        word_boxes: List[BBoxWH] = []
        for rel in block.get("Relationships", []) or []:
            if str(rel.get("Type", "")).upper() != "CHILD":
                continue
            for child_id in rel.get("Ids", []) or []:
                child = blocks_by_id.get(child_id)
                if not child:
                    continue
                if str(child.get("BlockType", "")).upper() != "WORD":
                    continue
                word_bbox = self._bbox_from_geometry(child.get("Geometry"), page_width, page_height)
                if not word_bbox:
                    continue
                wx0, wy0, wx1, wy1 = word_bbox
                if wx1 <= wx0 or wy1 <= wy0:
                    continue
                word_boxes.append((wx0, wy0, wx1 - wx0, wy1 - wy0))
        return word_boxes

    def _normalize_text_block_type(self, block_type_raw: str) -> Optional[str]:
        if block_type_raw in FIGURE_BLOCK_TYPES:
            return None
        if block_type_raw in {"LAYOUT_TEXT", "PARAGRAPH", "TEXT"}:
            return "TEXT"
        if block_type_raw in {"LAYOUT_TITLE", "TITLE"}:
            return "TITLE"
        if block_type_raw in {"LAYOUT_LIST", "LIST"}:
            return "LIST"
        if block_type_raw in {"LAYOUT_TABLE", "TABLE", "LAYOUT_CELL", "CELL"}:
            return "TEXT"
        if block_type_raw in {"KEY_VALUE_SET", "KEY", "VALUE", "LAYOUT_KEY_VALUE_SET"}:
            return "TEXT"
        if block_type_raw.startswith("LAYOUT_"):
            return "TEXT"
        return None

    def _bbox_from_geometry(
        self, geometry: Optional[Dict[str, object]], page_width: int, page_height: int
    ) -> Optional[Tuple[int, int, int, int]]:
        if not geometry:
            return None
        bbox = geometry.get("BoundingBox") if isinstance(geometry, dict) else None
        if not bbox:
            return None
        try:
            left = float(bbox.get("Left", 0.0))
            top = float(bbox.get("Top", 0.0))
            width = float(bbox.get("Width", 0.0))
            height = float(bbox.get("Height", 0.0))
        except Exception:
            return None
        x0 = int(left * page_width)
        y0 = int(top * page_height)
        x1 = int((left + width) * page_width)
        y1 = int((top + height) * page_height)
        x0 = max(0, min(page_width, x0))
        y0 = max(0, min(page_height, y0))
        x1 = max(x0, min(page_width, x1))
        y1 = max(y0, min(page_height, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _rotation_from_geometry(self, geometry: Optional[Dict[str, object]]) -> float:
        if not isinstance(geometry, dict):
            return 0.0
        try:
            explicit = geometry.get("RotationAngle")
            if explicit is not None:
                return float(explicit) % 360.0
        except Exception:
            pass
        polygon = geometry.get("Polygon")
        if not isinstance(polygon, list) or len(polygon) < 2:
            return 0.0
        try:
            first = polygon[0]
            second = polygon[1]
            dx = float(second.get("X", 0.0)) - float(first.get("X", 0.0))
            dy = float(second.get("Y", 0.0)) - float(first.get("Y", 0.0))
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                return 0.0
            angle = math.degrees(math.atan2(dy, dx)) % 360.0
            nearest = round(angle / 90.0) * 90.0
            return nearest % 360.0 if abs(angle - nearest) <= 8.0 else angle
        except Exception:
            return 0.0

    def _dedup_and_sort_lines(self, lines: Sequence[LineRegion]) -> List[LineRegion]:
        seen: Dict[str, LineRegion] = {}
        for line in lines:
            seen[line.line_id] = line
        sorted_lines = list(seen.values())
        sorted_lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        return sorted_lines
