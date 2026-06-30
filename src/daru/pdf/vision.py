from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..translation.codex_cli import CodexCliStructuredClient

try:  # pragma: no cover - optional dependency guard
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional dependency guard
    Image = None  # type: ignore

PdfLog = Callable[[str], None]

LAYOUT_SCHEMA_VERSION = 4
QA_SCHEMA_VERSION = 4
CODEX_LAYOUT_MAX_BLOCKS_PER_REQUEST = 16
CODEX_QA_MAX_REGIONS_PER_REQUEST = 16
CODEX_VISION_MAX_IMAGE_PIXELS = 4_000_000
VISION_IMAGE_QUALITY_STABLE = "stable"
VISION_IMAGE_QUALITY_HIGH = "high"
VISION_IMAGE_QUALITY_ORIGINAL = "original"
VISION_IMAGE_QUALITIES = {
    VISION_IMAGE_QUALITY_STABLE,
    VISION_IMAGE_QUALITY_HIGH,
    VISION_IMAGE_QUALITY_ORIGINAL,
}
VISION_IMAGE_QUALITY_MAX_PIXELS = {
    VISION_IMAGE_QUALITY_STABLE: 4_000_000,
    VISION_IMAGE_QUALITY_HIGH: 10_000_000,
    VISION_IMAGE_QUALITY_ORIGINAL: None,
}
VISION_REQUEST_MODE_BATCHED = "batched"
VISION_REQUEST_MODE_SINGLE_PAGE = "single_page"
VISION_REQUEST_MODES = {
    VISION_REQUEST_MODE_BATCHED,
    VISION_REQUEST_MODE_SINGLE_PAGE,
}

SEMANTIC_TYPES = {
    "body",
    "heading",
    "table_cell",
    "map_label",
    "caption",
    "brand",
    "contact",
    "decorative",
    "vertical",
    "unknown",
}

FIT_STRATEGIES = {
    "default",
    "preserve_small",
    "tight",
    "shrink",
    "wrap",
    "single_line",
    "vertical",
}

FONT_WEIGHTS = {
    "auto",
    "normal",
    "bold",
}

TEXT_ALIGNMENTS = {
    "auto",
    "left",
    "center",
    "right",
    "justify",
}

QA_ACTIONS = {
    "none",
    "shrink_text",
    "wrap_text",
    "reduce_padding",
    "preserve_font_size",
    "set_bold",
    "set_normal",
    "align_left",
    "align_center",
    "align_right",
    "align_justify",
    "skip_render",
    "manual_review",
}


@dataclass
class VisionBlockUpdate:
    stable_id: str
    corrected_text: str
    semantic_type: str = "unknown"
    translate: bool = True
    fit_strategy: str = "default"
    font_weight: str = "auto"
    alignment: str = "auto"
    confidence: float = 0.0
    qa_flags: List[str] = field(default_factory=list)


@dataclass
class VisionQaIssue:
    stable_id: str
    issue_type: str
    severity: str
    action: str
    confidence: float = 0.0
    message: str = ""


def _noop_log(_message: str) -> None:
    return


def _normalize_request_mode(value: object) -> str:
    normalized = str(value or VISION_REQUEST_MODE_BATCHED).strip().lower()
    return (
        normalized
        if normalized in VISION_REQUEST_MODES
        else VISION_REQUEST_MODE_BATCHED
    )


def normalize_vision_image_quality(value: object) -> str:
    normalized = str(value or VISION_IMAGE_QUALITY_STABLE).strip().lower()
    return (
        normalized
        if normalized in VISION_IMAGE_QUALITIES
        else VISION_IMAGE_QUALITY_STABLE
    )


def _vision_image_max_pixels(value: object) -> Optional[int]:
    quality = normalize_vision_image_quality(value)
    return VISION_IMAGE_QUALITY_MAX_PIXELS[quality]


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, result))


def _response_text(response_obj: object) -> str:
    text_value = getattr(response_obj, "output_text", None)
    if isinstance(text_value, str) and text_value.strip():
        return text_value
    if isinstance(response_obj, dict):
        text_value = response_obj.get("output_text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value
        output = response_obj.get("output")
    else:
        output = getattr(response_obj, "output", None)
    fragments: List[str] = []
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    part_text = part.get("text")
                    if isinstance(part_text, str):
                        fragments.append(part_text)
                else:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str):
                        fragments.append(part_text)
    return "".join(fragments).strip()


def _image_to_data_url(
    image: "Image.Image",
    *,
    max_pixels: Optional[int] = 10_000_000,
) -> str:
    encoded = base64.b64encode(
        _image_to_png_bytes(image, max_pixels=max_pixels)
    ).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_to_png_bytes(
    image: "Image.Image",
    *,
    max_pixels: Optional[int] = 10_000_000,
) -> bytes:
    if Image is None:
        raise RuntimeError("Install Pillow for GPT vision PDF analysis")
    rgb = image.convert("RGB")
    pixels = max(1, rgb.width * rgb.height)
    if max_pixels and max_pixels > 0 and pixels > max_pixels:
        scale = (max_pixels / float(pixels)) ** 0.5
        new_size = (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))
        rgb = rgb.resize(new_size, Image.LANCZOS)
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


def _image_digest(image: "Image.Image") -> str:
    rgb = image.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


class _JsonCache:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._data: Optional[Dict[str, object]] = None

    def get(self, key: str) -> Optional[Dict[str, object]]:
        if not self.path:
            return None
        data = self._load()
        value = data.get(key)
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: Dict[str, object]) -> None:
        if not self.path:
            return
        data = self._load()
        data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def _load(self) -> Dict[str, object]:
        if self._data is not None:
            return self._data
        if not self.path or not self.path.exists():
            self._data = {}
            return self._data
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            raw = {}
        self._data = raw if isinstance(raw, dict) else {}
        return self._data


class _OpenAIVisionBase:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-5.5",
        base_url: Optional[str] = None,
        project: Optional[str] = None,
        reasoning_effort: Optional[str] = "medium",
        image_quality: str = VISION_IMAGE_QUALITY_HIGH,
        cache_path: Optional[Path] = None,
        client: Optional[object] = None,
        log: PdfLog = _noop_log,
    ) -> None:
        self.backend_name = "openai_api"
        self.model = model or "gpt-5.5"
        self.reasoning_effort = reasoning_effort or "medium"
        self.image_quality = normalize_vision_image_quality(image_quality)
        self.image_max_pixels = _vision_image_max_pixels(self.image_quality)
        self.cache = _JsonCache(cache_path)
        self.log = log
        self.client = client or self._init_client(
            api_key=api_key,
            base_url=base_url,
            project=project,
        )

    def _init_client(
        self,
        *,
        api_key: Optional[str],
        base_url: Optional[str],
        project: Optional[str],
    ) -> object:
        if not api_key and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI API key is required for Textract + GPT Vision PDF mode")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("Install openai>=2.41.1 for GPT vision PDF analysis") from exc
        kwargs: Dict[str, object] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if project:
            kwargs["project"] = project
        return OpenAI(**kwargs)

    def _call_responses(
        self,
        *,
        input_items: List[Dict[str, object]],
        schema_name: str,
        schema: Dict[str, object],
    ) -> Dict[str, object]:
        text_config: Dict[str, object] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
            "verbosity": "low",
        }
        kwargs: Dict[str, object] = {
            "model": self.model,
            "input": input_items,
            "text": text_config,
            "store": True,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**kwargs)  # type: ignore[attr-defined]
        message = _response_text(response)
        if not message:
            parsed = getattr(response, "output_parsed", None)
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError("OpenAI vision returned an empty response")
        data = json.loads(message)
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI vision returned a non-object JSON response")
        return data


class OpenAIVisionLayoutRefiner(_OpenAIVisionBase):
    """Use GPT vision to refine Textract block semantics without changing coordinates."""

    def refine_page(
        self,
        *,
        page_index: int,
        image: "Image.Image",
        blocks: Sequence[object],
    ) -> Dict[str, VisionBlockUpdate]:
        block_payload = [self._block_payload(block) for block in blocks if getattr(block, "stable_id", "")]
        if not block_payload:
            return {}
        cache_key = self._cache_key(f"layout-{self.image_quality}", image, block_payload)
        cached = self.cache.get(cache_key)
        if cached is None:
            data_url = _image_to_data_url(image, max_pixels=self.image_max_pixels)
            prompt = (
                "Analyze this scanned PDF page and refine OCR/layout metadata for the provided Textract blocks. "
                "Return JSON only. Use only stable_id values from the supplied block list. Do not invent coordinates. "
                "Correct obvious OCR text errors, classify semantic_type, decide whether each block should be translated, "
                "identify the visible source font_weight and paragraph alignment, and choose a fit_strategy for drawing "
                "translated text back into the same box. Preserve the source font size whenever the translated text can "
                "fit; do not request shrinking merely because a block is dense."
            )
            cached = self._call_responses(
                input_items=[
                    {
                        "role": "developer",
                        "content": (
                            "You are a PDF layout analyst for engineering catalogs. Preserve brands, model names, "
                            "phone numbers, emails, addresses, and decorative labels when they should not be translated."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_text",
                                "text": json.dumps(
                                    {
                                        "page_index": page_index,
                                        "page_width": image.width,
                                        "page_height": image.height,
                                        "blocks": block_payload,
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                            {"type": "input_image", "image_url": data_url, "detail": "high"},
                        ],
                    },
                ],
                schema_name="daru_pdf_layout_refinement",
                schema=self._layout_schema(),
            )
            self.cache.set(cache_key, cached)
        return self._parse_updates(cached, {item["stable_id"] for item in block_payload})

    def _block_payload(self, block: object) -> Dict[str, object]:
        bbox = getattr(block, "bbox", (0, 0, 0, 0))
        return {
            "stable_id": str(getattr(block, "stable_id", "") or ""),
            "block_type": str(getattr(block, "block_type", "") or ""),
            "text": str(getattr(block, "source_text", "") or ""),
            "bbox": [int(v) for v in bbox],
            "is_table": bool(getattr(block, "is_table", False)),
            "alignment": str(getattr(block, "alignment", "") or "left"),
            "font_weight": str(getattr(block, "font_weight", "") or "auto"),
            "confidence": float(getattr(block, "confidence", 0.0) or 0.0),
            "parent_block_type": str(getattr(block, "parent_block_type", "") or ""),
        }

    def _cache_key(self, prefix: str, image: "Image.Image", payload: object) -> str:
        identity = json.dumps(
            {
                "version": LAYOUT_SCHEMA_VERSION,
                "backend": getattr(self, "backend_name", "openai_api"),
                "prefix": prefix,
                "model": self.model,
                "image_quality": getattr(self, "image_quality", VISION_IMAGE_QUALITY_HIGH),
                "image": _image_digest(image),
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _parse_updates(
        self,
        data: Dict[str, object],
        allowed_ids: set[str],
    ) -> Dict[str, VisionBlockUpdate]:
        raw_blocks = data.get("blocks")
        if not isinstance(raw_blocks, list):
            return {}
        updates: Dict[str, VisionBlockUpdate] = {}
        for item in raw_blocks:
            if not isinstance(item, dict):
                continue
            stable_id = str(item.get("stable_id", "") or "").strip()
            if stable_id not in allowed_ids:
                continue
            corrected_text = str(item.get("corrected_text", "") or "")
            semantic_type = str(item.get("semantic_type", "unknown") or "unknown")
            if semantic_type not in SEMANTIC_TYPES:
                semantic_type = "unknown"
            fit_strategy = str(item.get("fit_strategy", "default") or "default")
            if fit_strategy not in FIT_STRATEGIES:
                fit_strategy = "default"
            font_weight = str(item.get("font_weight", "auto") or "auto")
            if font_weight not in FONT_WEIGHTS:
                font_weight = "auto"
            alignment = str(item.get("alignment", "auto") or "auto")
            if alignment not in TEXT_ALIGNMENTS:
                alignment = "auto"
            qa_flags_raw = item.get("qa_flags", [])
            qa_flags = [
                str(flag)[:64]
                for flag in qa_flags_raw
                if isinstance(flag, str) and flag.strip()
            ] if isinstance(qa_flags_raw, list) else []
            updates[stable_id] = VisionBlockUpdate(
                stable_id=stable_id,
                corrected_text=corrected_text,
                semantic_type=semantic_type,
                translate=bool(item.get("translate", True)),
                fit_strategy=fit_strategy,
                font_weight=font_weight,
                alignment=alignment,
                confidence=_coerce_float(item.get("confidence")),
                qa_flags=qa_flags,
            )
        return updates

    def _layout_schema(self) -> Dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "stable_id": {"type": "string"},
                            "corrected_text": {"type": "string"},
                            "semantic_type": {"type": "string", "enum": sorted(SEMANTIC_TYPES)},
                            "translate": {"type": "boolean"},
                            "fit_strategy": {"type": "string", "enum": sorted(FIT_STRATEGIES)},
                            "font_weight": {"type": "string", "enum": sorted(FONT_WEIGHTS)},
                            "alignment": {"type": "string", "enum": sorted(TEXT_ALIGNMENTS)},
                            "confidence": {"type": "number"},
                            "qa_flags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "stable_id",
                            "corrected_text",
                            "semantic_type",
                            "translate",
                            "fit_strategy",
                            "font_weight",
                            "alignment",
                            "confidence",
                            "qa_flags",
                        ],
                    },
                }
            },
            "required": ["blocks"],
        }


class OpenAIVisionQaReviewer(_OpenAIVisionBase):
    """Compare source/rendered pages and report deterministic layout issues."""

    def review_page(
        self,
        *,
        page_index: int,
        source_image: "Image.Image",
        rendered_image: "Image.Image",
        regions: Sequence[object],
    ) -> List[VisionQaIssue]:
        region_payload = [self._region_payload(region) for region in regions if getattr(region, "stable_id", "")]
        if not region_payload:
            return []
        cache_key = self._cache_key(
            f"qa-{self.image_quality}",
            source_image,
            rendered_image,
            region_payload,
        )
        cached = self.cache.get(cache_key)
        if cached is None:
            source_url = _image_to_data_url(
                source_image,
                max_pixels=self.image_max_pixels,
            )
            rendered_url = _image_to_data_url(
                rendered_image,
                max_pixels=self.image_max_pixels,
            )
            prompt = (
                "Compare the source scanned PDF page with the rendered translated page. "
                "Report only visible layout defects: clipped text, overlapping text, excessive text size, "
                "unnecessarily reduced text, incorrect bold/normal font weight, incorrect paragraph alignment, "
                "untranslated OCR artifacts, or text drawn over non-text artwork. Use stable_id from the supplied regions. "
                "When the translated text has enough room at the source font size, use preserve_font_size rather than "
                "shrink_text. Use set_bold/set_normal and align_left/align_center/align_right/align_justify for formatting "
                "mismatches."
            )
            cached = self._call_responses(
                input_items=[
                    {
                        "role": "developer",
                        "content": (
                            "You are a strict visual QA reviewer for localized PDF pages. "
                            "Prefer no issue when the page is visually acceptable."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_text",
                                "text": json.dumps(
                                    {
                                        "page_index": page_index,
                                        "regions": region_payload,
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                            {"type": "input_text", "text": "SOURCE PAGE"},
                            {"type": "input_image", "image_url": source_url, "detail": "high"},
                            {"type": "input_text", "text": "RENDERED TRANSLATED PAGE"},
                            {"type": "input_image", "image_url": rendered_url, "detail": "high"},
                        ],
                    },
                ],
                schema_name="daru_pdf_render_qa",
                schema=self._qa_schema(),
            )
            self.cache.set(cache_key, cached)
        return self._parse_issues(cached, {item["stable_id"] for item in region_payload})

    def _region_payload(self, region: object) -> Dict[str, object]:
        bbox = getattr(region, "bbox", (0, 0, 0, 0))
        return {
            "stable_id": str(getattr(region, "stable_id", "") or ""),
            "region_id": int(getattr(region, "region_id", 0) or 0),
            "source_text": str(getattr(region, "source_text", "") or ""),
            "translated_text": str(getattr(region, "translated_text", "") or ""),
            "bbox": [int(v) for v in bbox],
            "semantic_type": str(getattr(region, "semantic_type", "") or "unknown"),
            "fit_strategy": str(getattr(region, "fit_strategy", "") or "default"),
            "font_weight": str(getattr(region, "font_weight", "") or "auto"),
            "is_bold": bool(getattr(region, "is_bold", False)),
            "alignment": str(getattr(region, "alignment", "") or "left"),
            "source_font_size_estimate": int(
                getattr(region, "source_font_size_estimate", 0) or 0
            ),
            "rendered_font_size": int(
                getattr(region, "rendered_font_size", 0) or 0
            ),
        }

    def _cache_key(
        self,
        prefix: str,
        source_image: "Image.Image",
        rendered_image: "Image.Image",
        payload: object,
    ) -> str:
        identity = json.dumps(
            {
                "version": QA_SCHEMA_VERSION,
                "backend": getattr(self, "backend_name", "openai_api"),
                "prefix": prefix,
                "model": self.model,
                "image_quality": getattr(self, "image_quality", VISION_IMAGE_QUALITY_HIGH),
                "source": _image_digest(source_image),
                "rendered": _image_digest(rendered_image),
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _parse_issues(
        self,
        data: Dict[str, object],
        allowed_ids: set[str],
    ) -> List[VisionQaIssue]:
        raw_issues = data.get("issues")
        if not isinstance(raw_issues, list):
            return []
        issues: List[VisionQaIssue] = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            stable_id = str(item.get("stable_id", "") or "").strip()
            if stable_id and stable_id not in allowed_ids:
                continue
            action = str(item.get("action", "none") or "none")
            if action not in QA_ACTIONS:
                action = "manual_review"
            issues.append(
                VisionQaIssue(
                    stable_id=stable_id,
                    issue_type=str(item.get("issue_type", "unknown") or "unknown")[:64],
                    severity=str(item.get("severity", "low") or "low")[:32],
                    action=action,
                    confidence=_coerce_float(item.get("confidence")),
                    message=str(item.get("message", "") or "")[:240],
                )
            )
        return issues

    def _qa_schema(self) -> Dict[str, object]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "stable_id": {"type": "string"},
                            "issue_type": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "action": {"type": "string", "enum": sorted(QA_ACTIONS)},
                            "confidence": {"type": "number"},
                            "message": {"type": "string"},
                        },
                        "required": [
                            "stable_id",
                            "issue_type",
                            "severity",
                            "action",
                            "confidence",
                            "message",
                        ],
                    },
                }
            },
            "required": ["issues"],
        }


class CodexCliVisionLayoutRefiner(OpenAIVisionLayoutRefiner):
    """Refine Textract blocks through `codex exec` with an attached page image."""

    def __init__(
        self,
        *,
        cli_path: str = "",
        model: str = "gpt-5.5",
        reasoning_effort: str = "medium",
        timeout_seconds: int = 300,
        cache_path: Optional[Path] = None,
        request_mode: str = VISION_REQUEST_MODE_BATCHED,
        image_quality: str = VISION_IMAGE_QUALITY_STABLE,
        client: Optional[CodexCliStructuredClient] = None,
        log: PdfLog = _noop_log,
    ) -> None:
        self.backend_name = "codex_cli"
        self.model = model or "gpt-5.5"
        self.reasoning_effort = reasoning_effort or "medium"
        self.request_mode = _normalize_request_mode(request_mode)
        self.image_quality = normalize_vision_image_quality(image_quality)
        self.image_max_pixels = _vision_image_max_pixels(self.image_quality)
        self.cache = _JsonCache(cache_path)
        self.log = log
        self.client = client or CodexCliStructuredClient(
            cli_path=cli_path,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            timeout_seconds=timeout_seconds,
            log=log,
        )

    def refine_page(
        self,
        *,
        page_index: int,
        image: "Image.Image",
        blocks: Sequence[object],
    ) -> Dict[str, VisionBlockUpdate]:
        block_payload = [
            self._block_payload(block)
            for block in blocks
            if getattr(block, "stable_id", "")
        ]
        if not block_payload:
            return {}
        cache_key = self._cache_key(
            f"layout-{self.request_mode}-{self.image_quality}",
            image,
            block_payload,
        )
        cached = self.cache.get(cache_key)
        if cached is None:
            image_bytes = _image_to_png_bytes(
                image,
                max_pixels=self.image_max_pixels,
            )
            if self.request_mode == VISION_REQUEST_MODE_SINGLE_PAGE:
                cached = self._refine_page_single_request(
                    page_index=page_index,
                    image=image,
                    image_bytes=image_bytes,
                    block_payload=block_payload,
                )
                self.cache.set(cache_key, cached)
                return self._parse_updates(
                    cached,
                    {item["stable_id"] for item in block_payload},
                )
            batches = [
                block_payload[index:index + CODEX_LAYOUT_MAX_BLOCKS_PER_REQUEST]
                for index in range(0, len(block_payload), CODEX_LAYOUT_MAX_BLOCKS_PER_REQUEST)
            ]
            merged_blocks: List[object] = []
            for batch_index, batch in enumerate(batches):
                batch_cache_key = self._cache_key(
                    (
                        f"layout-{self.image_quality}-batch-"
                        f"{batch_index + 1}-{len(batches)}"
                    ),
                    image,
                    batch,
                )
                batch_result = self.cache.get(batch_cache_key)
                if batch_result is None:
                    payload = {
                        "page_index": page_index,
                        "page_width": image.width,
                        "page_height": image.height,
                        "batch_index": batch_index + 1,
                        "batch_count": len(batches),
                        "blocks": batch,
                    }
                    prompt = (
                        "The attached image is the full scanned PDF page. Analyze it as a "
                        "PDF layout specialist for engineering catalogs, but return metadata "
                        "only for the supplied batch of Textract blocks. Use only stable_id "
                        "values from this batch and do not invent or change coordinates. "
                        "Correct obvious OCR errors, classify semantic_type, decide whether "
                        "each block should be translated, identify the visible source "
                        "font_weight and paragraph alignment, and choose a fit_strategy for "
                        "drawing translated text back into the same box. Preserve the source "
                        "font size whenever the translated text can fit; do not request "
                        "shrinking merely because a block is dense. Preserve brands, model "
                        "names, phone numbers, emails, addresses, and decorative labels when "
                        "they should not be translated. Do not use tools, inspect other "
                        "files, or access the network. Return only the object required by the "
                        "supplied JSON Schema.\n\n"
                        + json.dumps(payload, ensure_ascii=False)
                    )
                    batch_result = self.client.run(
                        prompt=prompt,
                        schema=self._layout_schema(),
                        images=[("page.png", image_bytes)],
                    )
                    self.cache.set(batch_cache_key, batch_result)
                raw_blocks = batch_result.get("blocks")
                if isinstance(raw_blocks, list):
                    merged_blocks.extend(raw_blocks)
                self.log(
                    f"GPT Vision CLI: page {page_index + 1}, "
                    f"layout batch {batch_index + 1}/{len(batches)} completed"
                )
            cached = {"blocks": merged_blocks}
            self.cache.set(cache_key, cached)
        return self._parse_updates(
            cached,
            {item["stable_id"] for item in block_payload},
        )

    def _refine_page_single_request(
        self,
        *,
        page_index: int,
        image: "Image.Image",
        image_bytes: bytes,
        block_payload: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        payload = {
            "page_index": page_index,
            "page_width": image.width,
            "page_height": image.height,
            "request_mode": VISION_REQUEST_MODE_SINGLE_PAGE,
            "blocks": list(block_payload),
        }
        prompt = (
            "The attached image is the full scanned PDF page. Analyze it as a "
            "PDF layout specialist for engineering catalogs and return metadata "
            "for every supplied Textract block in this single request. Use only "
            "stable_id values from the supplied block list and do not invent or "
            "change coordinates. Correct obvious OCR errors, classify semantic_type, "
            "decide whether each block should be translated, identify the visible "
            "source font_weight and paragraph alignment, and choose a fit_strategy "
            "for drawing translated text back into the same box. Preserve the source "
            "font size whenever the translated text can fit; do not request shrinking "
            "merely because a block is dense. Preserve brands, model names, phone "
            "numbers, emails, addresses, and decorative labels when they should not "
            "be translated. Do not use tools, inspect other files, or access the "
            "network. Return only the object required by the supplied JSON Schema.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = self.client.run(
            prompt=prompt,
            schema=self._layout_schema(),
            images=[("page.png", image_bytes)],
        )
        self.log(
            f"GPT Vision CLI: page {page_index + 1}, "
            "single-page layout request completed"
        )
        return result


class CodexCliVisionQaReviewer(OpenAIVisionQaReviewer):
    """Review source and rendered PDF pages through `codex exec`."""

    def __init__(
        self,
        *,
        cli_path: str = "",
        model: str = "gpt-5.5",
        reasoning_effort: str = "medium",
        timeout_seconds: int = 300,
        cache_path: Optional[Path] = None,
        request_mode: str = VISION_REQUEST_MODE_BATCHED,
        image_quality: str = VISION_IMAGE_QUALITY_STABLE,
        client: Optional[CodexCliStructuredClient] = None,
        log: PdfLog = _noop_log,
    ) -> None:
        self.backend_name = "codex_cli"
        self.model = model or "gpt-5.5"
        self.reasoning_effort = reasoning_effort or "medium"
        self.request_mode = _normalize_request_mode(request_mode)
        self.image_quality = normalize_vision_image_quality(image_quality)
        self.image_max_pixels = _vision_image_max_pixels(self.image_quality)
        self.cache = _JsonCache(cache_path)
        self.log = log
        self.client = client or CodexCliStructuredClient(
            cli_path=cli_path,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            timeout_seconds=timeout_seconds,
            log=log,
        )

    def review_page(
        self,
        *,
        page_index: int,
        source_image: "Image.Image",
        rendered_image: "Image.Image",
        regions: Sequence[object],
    ) -> List[VisionQaIssue]:
        region_payload = [
            self._region_payload(region)
            for region in regions
            if getattr(region, "stable_id", "")
        ]
        if not region_payload:
            return []
        cache_key = self._cache_key(
            f"qa-{self.request_mode}-{self.image_quality}",
            source_image,
            rendered_image,
            region_payload,
        )
        cached = self.cache.get(cache_key)
        if cached is None:
            source_bytes = _image_to_png_bytes(
                source_image,
                max_pixels=self.image_max_pixels,
            )
            rendered_bytes = _image_to_png_bytes(
                rendered_image,
                max_pixels=self.image_max_pixels,
            )
            if self.request_mode == VISION_REQUEST_MODE_SINGLE_PAGE:
                cached = self._review_page_single_request(
                    page_index=page_index,
                    source_bytes=source_bytes,
                    rendered_bytes=rendered_bytes,
                    region_payload=region_payload,
                )
                self.cache.set(cache_key, cached)
                return self._parse_issues(
                    cached,
                    {item["stable_id"] for item in region_payload},
                )
            batches = [
                region_payload[index:index + CODEX_QA_MAX_REGIONS_PER_REQUEST]
                for index in range(0, len(region_payload), CODEX_QA_MAX_REGIONS_PER_REQUEST)
            ]
            merged_issues: List[object] = []
            for batch_index, batch in enumerate(batches):
                batch_cache_key = self._cache_key(
                    (
                        f"qa-{self.image_quality}-batch-"
                        f"{batch_index + 1}-{len(batches)}"
                    ),
                    source_image,
                    rendered_image,
                    batch,
                )
                batch_result = self.cache.get(batch_cache_key)
                if batch_result is None:
                    payload = {
                        "page_index": page_index,
                        "batch_index": batch_index + 1,
                        "batch_count": len(batches),
                        "regions": batch,
                    }
                    prompt = (
                        "Two images are attached in this exact order: (1) the full source "
                        "scanned PDF page and (2) the full rendered translated page. Compare "
                        "them as a strict visual QA reviewer, but report issues only for the "
                        "supplied batch of regions. Report visible layout defects: clipped "
                        "text, overlapping text, excessive or unnecessarily reduced text "
                        "size, incorrect bold/normal font weight, incorrect paragraph "
                        "alignment, untranslated OCR artifacts, or text drawn over non-text "
                        "artwork. Use only stable_id values from this batch. When text has "
                        "enough room at the source font size, use preserve_font_size. Use "
                        "set_bold/set_normal and the align_* actions for formatting "
                        "mismatches. Prefer no issue when the page is visually acceptable. "
                        "Do not use tools, inspect other files, or access the network. Return "
                        "only the object required by the supplied JSON Schema.\n\n"
                        + json.dumps(payload, ensure_ascii=False)
                    )
                    batch_result = self.client.run(
                        prompt=prompt,
                        schema=self._qa_schema(),
                        images=[
                            ("source.png", source_bytes),
                            ("rendered.png", rendered_bytes),
                        ],
                    )
                    self.cache.set(batch_cache_key, batch_result)
                raw_issues = batch_result.get("issues")
                if isinstance(raw_issues, list):
                    merged_issues.extend(raw_issues)
                self.log(
                    f"GPT Vision CLI: page {page_index + 1}, "
                    f"QA batch {batch_index + 1}/{len(batches)} completed"
                )
            cached = {"issues": merged_issues}
            self.cache.set(cache_key, cached)
        return self._parse_issues(
            cached,
            {item["stable_id"] for item in region_payload},
        )

    def _review_page_single_request(
        self,
        *,
        page_index: int,
        source_bytes: bytes,
        rendered_bytes: bytes,
        region_payload: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        payload = {
            "page_index": page_index,
            "request_mode": VISION_REQUEST_MODE_SINGLE_PAGE,
            "regions": list(region_payload),
        }
        prompt = (
            "Two images are attached in this exact order: (1) the full source "
            "scanned PDF page and (2) the full rendered translated page. Compare "
            "them as a strict visual QA reviewer and report issues for the supplied "
            "regions in this single request. Report visible layout defects: clipped "
            "text, overlapping text, excessive or unnecessarily reduced text size, "
            "incorrect bold/normal font weight, incorrect paragraph alignment, "
            "untranslated OCR artifacts, or text drawn over non-text artwork. Use "
            "only stable_id values from the supplied regions. When text has enough "
            "room at the source font size, use preserve_font_size. Use "
            "set_bold/set_normal and the align_* actions for formatting mismatches. "
            "Prefer no issue when the page is visually acceptable. Do not use tools, "
            "inspect other files, or access the network. Return only the object "
            "required by the supplied JSON Schema.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = self.client.run(
            prompt=prompt,
            schema=self._qa_schema(),
            images=[
                ("source.png", source_bytes),
                ("rendered.png", rendered_bytes),
            ],
        )
        self.log(
            f"GPT Vision CLI: page {page_index + 1}, "
            "single-page QA request completed"
        )
        return result
