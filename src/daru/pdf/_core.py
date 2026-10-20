"""PDF orchestration and compatibility facade.

Processing stages live in independent modules; they never import this facade.
"""

from __future__ import annotations

import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..config import ORIGINAL_FONT_VALUE as ORIGINAL_FONT_VALUE
from ..config import normalize_style_font as normalize_style_font
from ..translation.analysis import CodexAnalysisSession as CodexAnalysisSession
from ..translation.checkpoint import (
    TranslationCheckpointStore as TranslationCheckpointStore,
)
from ..translation.checkpoint import default_checkpoint_path as default_checkpoint_path
from ..translation.checkpoint import file_sha256 as file_sha256
from ..translation.engine import TranslationEngine as TranslationEngine
from ..translation.engine import chunked as chunked
from ..translation.job import check_cancelled as check_cancelled
from ..translation.options import TranslationOptions as TranslationOptions
from ..translation.progress import report_progress
from ..translation.prompts import SCANNED_PDF_SYSTEM_PROMPT as SCANNED_PDF_SYSTEM_PROMPT
from ..utils.paths import migrate_pdf_cache as migrate_pdf_cache
from .block_translation import BlockRegionTranslator as BlockRegionTranslator
from .block_translation import RegionTranslator as RegionTranslator
from .block_translation import _apply_cached_translations as _apply_cached_translations
from .block_translation import _apply_canonical_map as _apply_canonical_map
from .block_translation import (
    _apply_canonical_translations as _apply_canonical_translations,
)
from .block_translation import (
    _apply_stable_cached_translations as _apply_stable_cached_translations,
)
from .block_translation import _apply_translation_caches as _apply_translation_caches
from .block_translation import _collect_canonical_cache as _collect_canonical_cache
from .common import _noop_log as _noop_log
from .common import _require_dependency as _require_dependency
from .common import _safe_float as _safe_float
from .common import _safe_int as _safe_int
from .config import _CACHE_DIR as _CACHE_DIR
from .config import _LEGACY_CACHE_DIR as _LEGACY_CACHE_DIR
from .config import FIGURE_BLOCK_TYPES as FIGURE_BLOCK_TYPES
from .config import PDF_PROCESSING_TEXTRACT as PDF_PROCESSING_TEXTRACT
from .config import PDF_PROCESSING_TEXTRACT_VISION as PDF_PROCESSING_TEXTRACT_VISION
from .config import PDF_TYPE_NATIVE as PDF_TYPE_NATIVE
from .config import PDF_TYPE_SCANNED as PDF_TYPE_SCANNED
from .config import TEXTRACT_LOG_PATH as TEXTRACT_LOG_PATH
from .config import TRANSLATABLE_BLOCK_TYPES as TRANSLATABLE_BLOCK_TYPES
from .config import PdfProcessingConfig as PdfProcessingConfig
from .export import Exporter as Exporter
from .export import _image_is_grayscale as _image_is_grayscale
from .export import _pdf_jpeg_quality as _pdf_jpeg_quality
from .export import _prepare_pdf_page_image as _prepare_pdf_page_image
from .export import images_to_pdf as images_to_pdf
from .geometry import _bbox_intersection_area_wh as _bbox_intersection_area_wh
from .geometry import _bbox_intersection_ratio as _bbox_intersection_ratio
from .geometry import _bbox_iou_wh as _bbox_iou_wh
from .geometry import _block_content_bbox as _block_content_bbox
from .geometry import _clamp_bbox_to_page as _clamp_bbox_to_page
from .geometry import _collect_block_word_boxes as _collect_block_word_boxes
from .geometry import _median_float as _median_float
from .geometry import _union_bbox as _union_bbox
from .imaging import _boost_text_contrast as _boost_text_contrast
from .imaging import _estimate_background_color as _estimate_background_color
from .imaging import _estimate_text_color as _estimate_text_color
from .imaging import _sample_text_color as _sample_text_color
from .imaging import build_text_mask as build_text_mask
from .imaging import build_text_mask_from_word_boxes as build_text_mask_from_word_boxes
from .layers import canonical_text as _canonicalize_text_for_dedup
from .layers import load_layer_mappings as load_layer_mappings
from .layers import load_translation_canonicals as load_translation_canonicals
from .layers import load_translation_mapping as load_translation_mapping
from .layers import load_translation_stable_mapping as load_translation_stable_mapping
from .layers import save_translation_mapping as save_translation_mapping
from .models import BBox as BBox
from .models import BBoxWH as BBoxWH
from .models import BlockRegion as BlockRegion
from .models import DetectedSegment as DetectedSegment
from .models import LineRegion as LineRegion
from .models import MergedRegion as MergedRegion
from .models import PageImage as PageImage
from .models import PdfLog as PdfLog
from .models import PdfProcessingResult as PdfProcessingResult
from .models import RegionInfo as RegionInfo
from .models import TextBlock as TextBlock
from .models import TextractPageData as TextractPageData
from .models import WordBox as WordBox
from .ocr import LayoutAnalyzer as LayoutAnalyzer
from .ocr import RegionAnalyzer as RegionAnalyzer
from .ocr import RegionDetector as RegionDetector
from .ocr import TextPreprocessor as TextPreprocessor
from .ocr import build_blocks_from_tesseract as build_blocks_from_tesseract
from .ocr import run_ocr as run_ocr
from .pages import ImageSpool as ImageSpool
from .pages import PageStore as PageStore
from .pages import release_page as release_page
from .raster import PageLoader as PageLoader
from .raster import _detect_pdf_raster_dpi as _detect_pdf_raster_dpi
from .raster import _detect_poppler_path as _detect_poppler_path
from .raster import _effective_output_dpi as _effective_output_dpi
from .raster import pdf_to_images as pdf_to_images
from .regions import _EMAIL_RE as _EMAIL_RE
from .regions import _NEW_ITEM_RE as _NEW_ITEM_RE
from .regions import _PHONE_RE as _PHONE_RE
from .regions import _WRAPPED_LINE_END_RE as _WRAPPED_LINE_END_RE
from .regions import _assign_render_ids_for_page as _assign_render_ids_for_page
from .regions import _block_to_region_info as _block_to_region_info
from .regions import _bounded_block_bbox as _bounded_block_bbox
from .regions import _build_regions_from_blocks as _build_regions_from_blocks
from .regions import _disable_unchanged_blocks as _disable_unchanged_blocks
from .regions import _estimate_font_size_from_block as _estimate_font_size_from_block
from .regions import _harmonize_region_font_sizes as _harmonize_region_font_sizes
from .regions import _is_wrapped_text_continuation as _is_wrapped_text_continuation
from .regions import _iter_candidate_blocks as _iter_candidate_blocks
from .regions import _iter_translatable_blocks as _iter_translatable_blocks
from .regions import _join_wrapped_text as _join_wrapped_text
from .regions import _mark_nontranslatable_blocks as _mark_nontranslatable_blocks
from .regions import _merge_wrapped_text_blocks as _merge_wrapped_text_blocks
from .regions import _needs_large_heading_recovery as _needs_large_heading_recovery
from .regions import _nontranslatable_text_reason as _nontranslatable_text_reason
from .regions import _stable_block_id as _stable_block_id
from .regions import _starts_with_lowercase_letter as _starts_with_lowercase_letter
from .regions import (
    _trim_recovery_bbox_to_light_background as _trim_recovery_bbox_to_light_background,
)
from .regions import _uppercase_tokens as _uppercase_tokens
from .rendering import Renderer as Renderer
from .rendering import _font_line_advance as _font_line_advance
from .rendering import _FontLoader as _FontLoader
from .rendering import _measure_text as _measure_text
from .rendering import _split_word_to_width as _split_word_to_width
from .rendering import _wrap_text as _wrap_text
from .textract import TextractLayoutExtractor as TextractLayoutExtractor
from .textract import _load_textract_cache as _load_textract_cache
from .textract import _log_textract_exchange as _log_textract_exchange
from .vision import CodexCliVisionLayoutRefiner as CodexCliVisionLayoutRefiner
from .vision import CodexCliVisionQaReviewer as CodexCliVisionQaReviewer
from .vision import OpenAIVisionLayoutRefiner as OpenAIVisionLayoutRefiner
from .vision import OpenAIVisionQaReviewer as OpenAIVisionQaReviewer
from .vision import VisionBlockUpdate as VisionBlockUpdate
from .vision import VisionQaIssue as VisionQaIssue

try:  # pragma: no cover - compatibility binding
    from PIL import Image as Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


try:  # pragma: no cover - compatibility binding
    import fitz as fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

try:  # pragma: no cover - compatibility binding
    import boto3 as boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore

def _resolve_scanned_style_font(
    style_font: Optional[str],
    log: PdfLog,
) -> Optional[str]:
    resolved = normalize_style_font(style_font or "")
    if resolved == ORIGINAL_FONT_VALUE:
        log(
            "PDF: для отсканированной страницы невозможно надежно определить "
            "исходное семейство шрифта; используется Unicode-шрифт по умолчанию"
        )
        return None
    return resolved or None


def _compute_file_sha256(path: Path) -> str:
    return file_sha256(path)


# === Pipeline components ======================================================


def _apply_vision_block_updates(
    blocks: Sequence[BlockRegion],
    updates: Dict[str, VisionBlockUpdate],
) -> int:
    if not updates:
        return 0
    applied = 0
    for block in _iter_candidate_blocks(blocks):
        update = updates.get(block.stable_id)
        if update is None:
            continue
        corrected = str(update.corrected_text or "").strip()
        if corrected:
            block.corrected_text = corrected
            block.source_text = corrected
        block.semantic_type = update.semantic_type or "unknown"
        block.fit_strategy = update.fit_strategy or "default"
        if update.font_weight in {"auto", "normal", "bold"}:
            block.font_weight = update.font_weight
        if (
            not block.alignment_locked
            and update.alignment in {"left", "center", "right", "justify"}
        ):
            block.alignment = update.alignment
        block.vision_confidence = max(0.0, min(1.0, float(update.confidence or 0.0)))
        block.qa_flags = list(update.qa_flags)
        if not update.translate or block.semantic_type in {"brand", "contact", "decorative"}:
            block.should_translate = False
            block.render_enabled = False
        if block.semantic_type == "table_cell":
            block.is_table = True
        if block.semantic_type == "vertical" and block.fit_strategy == "default":
            block.fit_strategy = "vertical"
        applied += 1
    return applied


def _vision_cache_path(config: PdfProcessingConfig) -> Path:
    if config.vision_cache_path:
        return Path(config.vision_cache_path)
    return _CACHE_DIR / "vision_requests.json"


def _vision_mode_enabled(processing_mode: object) -> bool:
    return str(processing_mode or "").strip().lower() == PDF_PROCESSING_TEXTRACT_VISION


def _run_vision_layout_refinement(
    page_layouts: Sequence[TextractPageData],
    *,
    config: PdfProcessingConfig,
    openai_key: Optional[str],
    openai_base_url: Optional[str],
    openai_project: Optional[str],
    codex_cli_path: Optional[str],
    codex_timeout_seconds: int,
    log: PdfLog,
) -> None:
    if not page_layouts:
        return
    try:
        if config.vision_backend == "codex_cli":
            refiner = CodexCliVisionLayoutRefiner(
                cli_path=config.vision_codex_cli_path or codex_cli_path or "",
                model=config.vision_model or "gpt-5.5",
                reasoning_effort=config.vision_reasoning_effort or "medium",
                timeout_seconds=(
                    config.vision_codex_timeout_seconds
                    or codex_timeout_seconds
                ),
                cache_path=_vision_cache_path(config),
                request_mode=config.vision_request_mode,
                image_quality=config.vision_image_quality,
                log=log,
            )
        else:
            refiner = OpenAIVisionLayoutRefiner(
                api_key=config.vision_api_key or openai_key,
                model=config.vision_model or "gpt-5.5",
                base_url=config.vision_base_url or openai_base_url,
                project=config.vision_project or openai_project,
                reasoning_effort=config.vision_reasoning_effort or "medium",
                image_quality=config.vision_image_quality,
                cache_path=_vision_cache_path(config),
                log=log,
            )
    except Exception as exc:
        log(f"GPT Vision: layout refinement skipped ({exc})")
        return

    total = 0
    for layout in page_layouts:
        check_cancelled()
        try:
            updates = refiner.refine_page(
                page_index=layout.page.page_index,
                image=layout.page.image,
                blocks=list(_iter_candidate_blocks(layout.blocks)),
            )
        except Exception as exc:
            log(f"GPT Vision: page {layout.page.page_index + 1} layout refinement failed ({exc})")
            continue
        finally:
            release_page(layout.page)
        applied = _apply_vision_block_updates(layout.blocks, updates)
        total += applied
        log(f"GPT Vision: page {layout.page.page_index + 1} refined blocks: {applied}")
    if total:
        log(f"GPT Vision: total refined blocks: {total}")


def _apply_vision_qa_issues(
    regions: Sequence[RegionInfo],
    issues: Sequence[VisionQaIssue],
) -> bool:
    if not issues:
        return False
    regions_by_id: Dict[str, List[RegionInfo]] = {}
    for region in regions:
        if region.stable_id:
            regions_by_id.setdefault(region.stable_id, []).append(region)
    changed = False
    deterministic_actions = {
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
    }
    for issue in issues:
        target_regions = regions_by_id.get(issue.stable_id, [])
        for region in target_regions:
            flag = issue.issue_type or issue.action
            if flag and flag not in region.qa_flags:
                region.qa_flags.append(flag)
            if issue.action not in deterministic_actions or issue.confidence < 0.45:
                continue
            if issue.action == "wrap_text":
                region.fit_strategy = "wrap"
            elif issue.action == "reduce_padding":
                region.fit_strategy = "tight"
            elif issue.action == "preserve_font_size":
                region.fit_strategy = "tight"
                region.font_size_estimate = max(
                    region.font_size_estimate,
                    region.source_font_size_estimate or region.font_size_estimate,
                )
            elif issue.action == "set_bold":
                region.font_weight = "bold"
                region.is_bold = True
            elif issue.action == "set_normal":
                region.font_weight = "normal"
                region.is_bold = False
            elif issue.action.startswith("align_"):
                alignment = issue.action.removeprefix("align_")
                if alignment in {"left", "center", "right", "justify"}:
                    region.alignment = alignment
            elif issue.action == "shrink_text":
                region.fit_strategy = "shrink"
                region.font_size_estimate = max(
                    5, int(region.font_size_estimate * 0.85)
                )
            changed = True
    return changed


def _make_vision_qa_reviewer(
    *,
    config: PdfProcessingConfig,
    openai_key: Optional[str],
    openai_base_url: Optional[str],
    openai_project: Optional[str],
    codex_cli_path: Optional[str],
    codex_timeout_seconds: int,
    log: PdfLog,
) -> Optional[object]:
    try:
        if config.vision_backend == "codex_cli":
            return CodexCliVisionQaReviewer(
                cli_path=config.vision_codex_cli_path or codex_cli_path or "",
                model=config.vision_model or "gpt-5.5",
                reasoning_effort=config.vision_reasoning_effort or "medium",
                timeout_seconds=(
                    config.vision_codex_timeout_seconds
                    or codex_timeout_seconds
                ),
                cache_path=_vision_cache_path(config),
                request_mode=config.vision_request_mode,
                image_quality=config.vision_image_quality,
                log=log,
            )
        return OpenAIVisionQaReviewer(
            api_key=config.vision_api_key or openai_key,
            model=config.vision_model or "gpt-5.5",
            base_url=config.vision_base_url or openai_base_url,
            project=config.vision_project or openai_project,
            reasoning_effort=config.vision_reasoning_effort or "medium",
            image_quality=config.vision_image_quality,
            cache_path=_vision_cache_path(config),
            log=log,
        )
    except Exception as exc:
        log(f"GPT Vision: QA skipped ({exc})")
        return None


def process_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    *,
    config: Optional[PdfProcessingConfig] = None,
    log: Optional[PdfLog] = None,
    write_blurred_pdf: bool = True,
) -> PdfProcessingResult:
    """
    Convert a PDF into page images, detect text blocks with Textract, remove the original
    glyphs, and (optionally) save the blurred/clean PDF. This is the "analysis only"
    path — translation happens in translate_pdf().

    Raises descriptive exceptions on dependency or parsing issues.
    """
    logger = log or _noop_log
    cfg = (config or PdfProcessingConfig()).normalized()
    input_path = Path(input_pdf_path)
    output_path = Path(output_pdf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"PDF не найден: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError("Поддерживаются только файлы PDF")

    file_hash = _compute_file_sha256(input_path)
    loader = PageLoader(cfg, logger)
    renderer = Renderer(cfg, logger)
    textract = TextractLayoutExtractor(
        None,
        logger,
        config=cfg,
        log_path=TEXTRACT_LOG_PATH,
        file_hash=file_hash,
        input_name=str(input_path),
    )

    pages = loader.load(input_path)
    processed_images: List["Image.Image"] = []
    all_regions: List[RegionInfo] = []
    page_layouts: List[TextractPageData] = []

    for page in pages:
        logger(f"PDF: обработка страницы {page.page_index + 1}/{len(pages)}")
        page_layout = textract.analyze(page)
        page_layout.blocks = _merge_wrapped_text_blocks(page_layout.blocks, logger)
        _assign_render_ids_for_page(page_layout.blocks, page)
        page_layouts.append(page_layout)

    _mark_nontranslatable_blocks(page_layouts, logger)
    for page_layout in page_layouts:
        page = page_layout.page
        for block in _iter_translatable_blocks(page_layout.blocks):
            block.translated_text = block.source_text
        regions = _build_regions_from_blocks(page, page_layout.blocks, cfg, log=logger)
        all_regions.extend(regions)
        processed_image = renderer.render_page(page, regions)
        processed_images.append(processed_image)

    if write_blurred_pdf:
        output_dpi = _effective_output_dpi(cfg.dpi, pages, logger)
        Exporter(logger).to_pdf(
            processed_images,
            output_path,
            dpi=output_dpi,
            input_dpi=pages[0].dpi if pages else cfg.render_dpi,
            page_sizes_points=[page.physical_size_points for page in pages],
        )
        logger(f"PDF: файл сохранён: {output_path}")

    return PdfProcessingResult(
        blurred_pdf_path=output_path,
        regions=all_regions,
        processed_images=processed_images,
    )


def _translate_scanned_pdf(
    *,
    input_path: Path,
    output_path: Path,
    translator_name: str,
    source_lang: str,
    target_lang: str,
    log: PdfLog,
    pdf_type: str = PDF_TYPE_SCANNED,
    layer_json_path: Optional[Path] = None,
    processing_options: Optional[Dict[str, object]] = None,
    style_font: Optional[str] = None,
    deepl_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_project: Optional[str] = None,
    openai_temperature: float = 0.2,
    openai_reasoning_effort: Optional[str] = None,
    openai_verbosity: Optional[str] = None,
    openai_strict_mode: Optional[str] = None,
    openai_strict_value: Optional[float] = None,
    codex_cli_path: Optional[str] = None,
    codex_model: Optional[str] = None,
    codex_reasoning_effort: Optional[str] = None,
    codex_analysis_model: Optional[str] = None,
    codex_analysis_reasoning_effort: Optional[str] = None,
    codex_analysis_session: Optional[CodexAnalysisSession] = None,
    codex_timeout_seconds: int = 300,
    checkpoint_path: Optional[Path] = None,
    resume_policy: str = "auto",
) -> Dict[str, object]:
    """
    High-quality PDF translation pipeline driven purely by Textract layout blocks.
    """
    with ExitStack() as stack:
        workspace = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="daru-pages-")))
        options = TranslationOptions.from_mapping(locals(), system_prompt_template=SCANNED_PDF_SYSTEM_PROMPT)
        _ = pdf_type
        config = PdfProcessingConfig.from_options(processing_options)

        if layer_json_path is None:
            layer_json_path = _CACHE_DIR / f"{input_path.stem}.translation.json"

        file_hash = _compute_file_sha256(input_path)
        checkpoint_store = TranslationCheckpointStore(
            path=checkpoint_path or default_checkpoint_path(input_path, "pdf-scanned"),
            job_type="pdf-scanned",
            document_sha256=file_hash,
            source_lang=source_lang,
            target_lang=target_lang,
            translator=translator_name,
            settings_fingerprint=options.fingerprint("pdf-scanned"),
            resume_policy=resume_policy,
            log=log,
        )
        pdf_processing_mode = (processing_options.get("pdf_processing_mode", "textract") if processing_options else "textract") or "textract"
        use_vision_mode = _vision_mode_enabled(pdf_processing_mode)
        textract_client_override: Optional[object] = None
        if str(pdf_processing_mode).lower() in {PDF_PROCESSING_TEXTRACT, PDF_PROCESSING_TEXTRACT_VISION}:
            region_name = processing_options.get("textract_region") if processing_options else None
            access_key = processing_options.get("textract_access_key") if processing_options else None
            secret_key = processing_options.get("textract_secret_key") if processing_options else None
            session_token = processing_options.get("textract_session_token") if processing_options else None
            if access_key and secret_key:
                _require_dependency(boto3, "Установите boto3 (pip install boto3) для Textract")
                try:
                    textract_client_override = boto3.client(
                        "textract",
                        region_name=region_name or None,
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        aws_session_token=session_token or None,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Textract: не удалось создать клиент с заданными ключами: {exc}") from exc

        loader = PageLoader(config, log)
        resolved_style_font = _resolve_scanned_style_font(style_font, log)
        renderer = Renderer(
            config,
            log,
            style_font=resolved_style_font,
        )
        exporter = Exporter(log)
        textract = TextractLayoutExtractor(
            textract_client_override,
            log,
            config=config,
            log_path=TEXTRACT_LOG_PATH,
            file_hash=file_hash,
            input_name=str(input_path),
        )

        page_store = PageStore(workspace / "source")
        stack.callback(page_store.close)
        pages = page_store.pages
        page_layouts: List[TextractPageData] = []
        report_progress("ocr", 0)
        for page in loader.iter_pages(input_path):
            check_cancelled()
            try:
                stored_page = page_store.add(page)
                layout = textract.analyze(page)
                layout.blocks = _merge_wrapped_text_blocks(layout.blocks, log)
                _assign_render_ids_for_page(layout.blocks, page)
                layout.page = stored_page
                page_layouts.append(layout)
                report_progress("ocr", len(page_layouts))
            finally:
                page.image.close()

        if use_vision_mode:
            report_progress("analysis")
            _run_vision_layout_refinement(
                page_layouts,
                config=config,
                openai_key=openai_key,
                openai_base_url=openai_base_url,
                openai_project=openai_project,
                codex_cli_path=codex_cli_path,
                codex_timeout_seconds=codex_timeout_seconds,
                log=log,
            )

        _mark_nontranslatable_blocks(page_layouts, log)
        all_blocks: List[BlockRegion] = []
        for layout in page_layouts:
            check_cancelled()
            all_blocks.extend(list(_iter_translatable_blocks(layout.blocks)))

        if not all_blocks:
            raise RuntimeError("Не удалось извлечь текст из PDF/изображения")

        existing_map: Dict[Tuple[int, int], str] = {}
        existing_stable_map: Dict[str, str] = {}
        checkpoint_canonicals = checkpoint_store.text_map("pdf-scanned:canonical")
        existing_canonicals: Dict[str, str] = dict(checkpoint_canonicals)
        if checkpoint_canonicals:
            log(f"PDF: checkpoint loaded canonical translations: {len(checkpoint_canonicals)}")
        if resume_policy == "auto" and layer_json_path and layer_json_path.exists():
            layer = load_layer_mappings(layer_json_path, log, options.cache_identity(file_hash, "pdf-scanned"))
            # Render IDs are positional and can change after OCR/Vision refinement.
            # Automatic reuse requires a stable ID or the same source text.
            existing_stable_map = layer.by_stable_id
            layer_canonicals = layer.by_text
            for key, value in layer_canonicals.items():
                existing_canonicals.setdefault(key, value)
            cached_count = len(existing_stable_map) or len(existing_map)
            if cached_count:
                log(f"PDF: найден готовый слой перевода, используем без обращения к API ({cached_count} областей)")

        for layout in page_layouts:
            check_cancelled()
            _apply_translation_caches(
                layout.blocks,
                stable_map=existing_stable_map,
                legacy_render_map=existing_map,
                canonical_map=existing_canonicals,
            )

        backend_name = "cached-checkpoint" if checkpoint_canonicals else "cached-layer"
        canonical_cache = _collect_canonical_cache(all_blocks)
        missing_canonicals: Set[str] = set()
        pending_block_count = 0
        for block in all_blocks:
            if block.block_type not in TRANSLATABLE_BLOCK_TYPES:
                continue
            source_value = str(block.source_text or "").strip()
            if not source_value:
                continue
            canonical_value = _canonicalize_text_for_dedup(source_value)
            if canonical_value and canonical_value not in canonical_cache:
                pending_block_count += 1
                missing_canonicals.add(canonical_value)


        if missing_canonicals:
            translator = TranslationEngine(**options.engine_kwargs())
            all_source_texts = [
                str(b.source_text or "").strip()
                for b in all_blocks
                if b.block_type in TRANSLATABLE_BLOCK_TYPES and str(b.source_text or "").strip()
            ]
            translator.set_drawing_context(all_source_texts)
            backend_name = translator.backend_name()
            checkpoint_store.set_backend(backend_name)
            log(f"Инициализирован движок перевода: {backend_name}")

            log(
                f"Готовим {len(missing_canonicals)} уникальных блоков текста к переводу "
                f"(из {pending_block_count} без готового перевода)"
            )
            canonical_cache = BlockRegionTranslator(translator, log).translate(
                all_blocks,
                canonical_cache,
                checkpoint_store=checkpoint_store,
            )

        _apply_canonical_translations(all_blocks, canonical_cache)
        report_progress("apply", 0, len(page_layouts))
        for layout in page_layouts:
            check_cancelled()
            _disable_unchanged_blocks(layout.blocks)

        for block in all_blocks:
            if block.render_id is not None and block.translated_text:
                existing_map[(block.page_index, block.render_id)] = block.translated_text

        all_regions: List[RegionInfo] = []
        export_spool = ImageSpool(workspace / "output")
        layout_warnings: List[str] = []
        vision_reviewer = (
            _make_vision_qa_reviewer(
                config=config,
                openai_key=openai_key,
                openai_base_url=openai_base_url,
                openai_project=openai_project,
                codex_cli_path=codex_cli_path,
                codex_timeout_seconds=codex_timeout_seconds,
                log=log,
            )
            if use_vision_mode
            else None
        )
        editable_layer = None
        page_editable_regions: Dict[int, List[RegionInfo]] = {}
        if config.editable_text:
            from .editable import EditableTextLayer, supports_editable_rotation

            editable_layer = EditableTextLayer(resolved_style_font, log)
        for page_number, layout in enumerate(page_layouts, 1):
            check_cancelled()
            regions = _build_regions_from_blocks(layout.page, layout.blocks, config, log=log)
            all_regions.extend(regions)
            clean_background = renderer.clean_page(layout.page, regions)
            editable_regions = list(regions)
            raster_fallback_regions: List[RegionInfo] = []
            if editable_layer is not None:
                editable_regions = [
                    region for region in regions if supports_editable_rotation(region)
                ]
                raster_fallback_regions = [
                    region for region in regions if not supports_editable_rotation(region)
                ]
                page_editable_regions[layout.page.page_index] = editable_regions
                for region in raster_fallback_regions:
                    region.layout_warning = "raster_fallback_unsupported_rotation"
                    warning = (
                        f"page={layout.page.page_index + 1} stable_id={region.stable_id} "
                        f"reason=raster_fallback_unsupported_rotation "
                        f"rotation={region.text_orientation_deg:g}"
                    )
                    layout_warnings.append(warning)
                    log(f"PDF: {warning}")
            cleaned = (
                renderer.render_text_layer(clean_background, raster_fallback_regions)
                if raster_fallback_regions
                else clean_background
            )
            if vision_reviewer is not None and editable_layer is not None:
                rendered = editable_layer.render_preview(
                    cleaned,
                    layout.page,
                    editable_regions,
                )
            elif vision_reviewer is not None or not config.editable_text:
                rendered = renderer.render_text_layer(cleaned, regions)
            else:
                rendered = None
            if vision_reviewer is not None and regions:
                try:
                    qa_issues = vision_reviewer.review_page(
                        page_index=layout.page.page_index,
                        source_image=layout.page.image,
                        rendered_image=rendered,
                        regions=regions,
                    )
                except Exception as exc:
                    log(f"GPT Vision: page {layout.page.page_index + 1} QA failed ({exc})")
                    qa_issues = []
                for issue in qa_issues:
                    warning = (
                        f"page={layout.page.page_index + 1} stable_id={issue.stable_id} "
                        f"issue={issue.issue_type} action={issue.action}: {issue.message}"
                    )
                    layout_warnings.append(warning)
                if _apply_vision_qa_issues(regions, qa_issues):
                    log(f"GPT Vision: page {layout.page.page_index + 1} rerendered after QA adjustments")
                    if editable_layer is not None:
                        cleaned = (
                            renderer.render_text_layer(
                                clean_background,
                                raster_fallback_regions,
                            )
                            if raster_fallback_regions
                            else clean_background
                        )
                        rendered = editable_layer.render_preview(
                            cleaned,
                            layout.page,
                            editable_regions,
                        )
                    else:
                        rendered = renderer.render_text_layer(cleaned, regions)
            if config.editable_text:
                export_spool.append(cleaned)
            else:
                export_spool.append(rendered or renderer.render_text_layer(cleaned, regions))
            for region in regions:
                region.mask = None
            for image in (clean_background, cleaned, rendered):
                if image is not None:
                    image.close()
            release_page(layout.page)

            report_progress("apply", page_number, len(page_layouts))

        output_dpi = _effective_output_dpi(config.dpi, pages, log)
        page_overlay: Optional[Callable[[int, object], None]] = None
        export_images: Sequence["Image.Image"] = export_spool
        if editable_layer is not None:
            def insert_editable_page(index: int, pdf_page: object) -> None:
                page_image = pages[index]
                editable_layer.insert_page(
                    pdf_page,
                    page_image,
                    page_editable_regions.get(page_image.page_index, []),
                )

            page_overlay = insert_editable_page

        report_progress("export")
        exporter.to_pdf(
            export_images,
            output_path,
            dpi=output_dpi,
            input_dpi=pages[0].dpi if pages else config.render_dpi,
            page_sizes_points=[page.physical_size_points for page in pages],
            page_overlay=page_overlay,
        )
        if editable_layer is not None:
            for warning in editable_layer.warnings:
                layout_warnings.append(
                    f"page={warning.get('page')} stable_id={warning.get('stable_id')} "
                    f"reason={warning.get('reason')} scale={warning.get('scale')}"
                )

        if layer_json_path:
            save_translation_mapping(all_regions, layer_json_path, log, cache_identity=options.cache_identity(file_hash, "pdf-scanned"))
        log(f"PDF: файл сохранён: {output_path}")

        return {
            "output_path": output_path,
            "backend": backend_name,
            "pages": len(pages),
            "job_type": "pdf",
            "processing_mode": PDF_PROCESSING_TEXTRACT_VISION if use_vision_mode else "scanned",
            "native_pages": [],
            "ocr_pages": list(range(1, len(pages) + 1)),
            "ocr_performed": True,
            "editable_text": bool(config.editable_text),
            "translated_blocks": len(all_blocks),
            "layout_warnings": layout_warnings,
            "checkpoint_path": checkpoint_store.path,
        }


def translate_pdf(
    *,
    input_path: Path,
    output_path: Path,
    translator_name: str,
    source_lang: str,
    target_lang: str,
    log: PdfLog,
    pdf_type: str = PDF_TYPE_SCANNED,
    layer_json_path: Optional[Path] = None,
    processing_options: Optional[Dict[str, object]] = None,
    style_font: Optional[str] = None,
    deepl_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_project: Optional[str] = None,
    openai_temperature: float = 0.2,
    openai_reasoning_effort: Optional[str] = None,
    openai_verbosity: Optional[str] = None,
    openai_strict_mode: Optional[str] = None,
    openai_strict_value: Optional[float] = None,
    codex_cli_path: Optional[str] = None,
    codex_model: Optional[str] = None,
    codex_reasoning_effort: Optional[str] = None,
    codex_analysis_model: Optional[str] = None,
    codex_analysis_reasoning_effort: Optional[str] = None,
    codex_analysis_session: Optional[CodexAnalysisSession] = None,
    codex_timeout_seconds: int = 300,
    checkpoint_path: Optional[Path] = None,
    resume_policy: str = "auto",
) -> Dict[str, object]:
    """Route searchable PDFs to the native pipeline and scans to OCR/Textract."""

    check_cancelled()
    migrate_pdf_cache(_LEGACY_CACHE_DIR, _CACHE_DIR, log)
    if str(translator_name or "").strip().lower() == "codex":
        codex_analysis_session = codex_analysis_session or CodexAnalysisSession()

    normalized_type = str(pdf_type or PDF_TYPE_SCANNED).strip().lower()
    common_args = {
        "input_path": input_path,
        "output_path": output_path,
        "translator_name": translator_name,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "log": log,
        "layer_json_path": layer_json_path,
        "processing_options": processing_options,
        "style_font": style_font,
        "deepl_key": deepl_key,
        "openai_key": openai_key,
        "openai_model": openai_model,
        "openai_base_url": openai_base_url,
        "openai_project": openai_project,
        "openai_temperature": openai_temperature,
        "openai_reasoning_effort": openai_reasoning_effort,
        "openai_verbosity": openai_verbosity,
        "openai_strict_mode": openai_strict_mode,
        "openai_strict_value": openai_strict_value,
        "codex_cli_path": codex_cli_path,
        "codex_model": codex_model,
        "codex_reasoning_effort": codex_reasoning_effort,
        "codex_analysis_model": codex_analysis_model,
        "codex_analysis_reasoning_effort": codex_analysis_reasoning_effort,
        "codex_analysis_session": codex_analysis_session,
        "codex_timeout_seconds": codex_timeout_seconds,
        "checkpoint_path": checkpoint_path,
        "resume_policy": resume_policy,
    }
    if normalized_type != PDF_TYPE_NATIVE:
        return _translate_scanned_pdf(pdf_type=PDF_TYPE_SCANNED, **common_args)

    from .native import translate_native_pdf

    def translate_fallback_page(page_index: int) -> bytes:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Для гибридной обработки PDF требуется PyMuPDF>=1.27.2.2"
            ) from exc

        import tempfile

        log(f"PDF: запускаем OCR fallback для страницы {page_index + 1}")
        with tempfile.TemporaryDirectory(prefix="daru-pdf-fallback-") as temp_dir:
            temp_root = Path(temp_dir)
            fallback_input = temp_root / f"page-{page_index + 1}.pdf"
            fallback_output = temp_root / f"page-{page_index + 1}-translated.pdf"
            fallback_layer = temp_root / f"page-{page_index + 1}.translation.json"

            source_document = fitz.open(str(input_path))
            one_page = fitz.open()
            try:
                one_page.insert_pdf(
                    source_document,
                    from_page=page_index,
                    to_page=page_index,
                )
                one_page.save(str(fallback_input), garbage=4, deflate=True)
            finally:
                one_page.close()
                source_document.close()

            fallback_args = dict(common_args)
            fallback_args.update(
                {
                    "input_path": fallback_input,
                    "output_path": fallback_output,
                    "layer_json_path": fallback_layer,
                    "checkpoint_path": default_checkpoint_path(
                        input_path,
                        f"pdf-fallback-{page_index + 1}",
                    ),
                }
            )
            _translate_scanned_pdf(pdf_type=PDF_TYPE_SCANNED, **fallback_args)
            return fallback_output.read_bytes()

    native_args = dict(common_args)
    native_args.pop("processing_options", None)
    native_args["fallback_page_translator"] = translate_fallback_page
    return translate_native_pdf(**native_args)


if __name__ == "__main__":  # pragma: no cover - manual usage hint
    sample_input = Path("sample.pdf")
    sample_output = Path("sample_blurred.pdf")
    if sample_input.exists():
        print(f"Processing {sample_input} -> {sample_output}")
        process_pdf(str(sample_input), str(sample_output))
    else:
        print("Поместите sample.pdf рядом со скриптом для быстрой проверки")
