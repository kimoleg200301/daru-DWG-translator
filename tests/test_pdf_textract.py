from __future__ import annotations

import json
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest
from PIL import Image

from daru.pdf import _core, export
from daru.translation.checkpoint import TranslationCheckpointStore, file_sha256


def _geometry(left: float, top: float, width: float, height: float):
    return {
        "BoundingBox": {
            "Left": left,
            "Top": top,
            "Width": width,
            "Height": height,
        }
    }


def test_figure_cache_miss_initializes_client_after_cached_page(tmp_path, monkeypatch):
    requests = []
    initializations = []

    class Client:
        def analyze_document(self, **kwargs):
            requests.append(kwargs)
            return {"Blocks": []}

    extractor = _core.TextractLayoutExtractor(
        None, config=_core.PdfProcessingConfig(), log_path=tmp_path / "textract.log",
    )

    def init_client():
        initializations.append(True)
        return Client()

    monkeypatch.setattr(extractor, "_init_client", init_client)
    monkeypatch.setattr(
        extractor, "_find_cached_response",
        lambda index, metadata: {"Blocks": []} if "TABLES" in metadata["feature_types"] else None,
    )
    with Image.new("RGB", (160, 100), "white") as image:
        page = _core.PageImage(0, image, 160, 100, 150)
        extractor.analyze(page)
        assert initializations == []
        extractor._analyze_figure_region((20, 20, 140, 80), page, "figure")
    assert initializations == [True]
    assert len(requests) == 1
    assert requests[0]["FeatureTypes"] == ["LAYOUT", "SIGNATURES"]


def _word(block_id: str, text: str, left: float, top: float, width: float, height: float):
    return {
        "Id": block_id,
        "BlockType": "WORD",
        "Text": text,
        "Confidence": 99.0,
        "Geometry": _geometry(left, top, width, height),
    }


def _line(
    block_id: str,
    text: str,
    word_ids: list[str],
    left: float,
    top: float,
    width: float,
    height: float,
):
    return {
        "Id": block_id,
        "BlockType": "LINE",
        "Text": text,
        "Confidence": 98.0,
        "Geometry": _geometry(left, top, width, height),
        "Relationships": [{"Type": "CHILD", "Ids": word_ids}],
    }


def _extract_blocks(raw_blocks, *, config=None):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=config or _core.PdfProcessingConfig(),
        log_path=Path("missing-textract-cache.jsonl"),
    )
    by_id = {block["Id"]: block for block in raw_blocks}
    lines = extractor._build_lines(
        by_id,
        page_index=0,
        page_width=1000,
        page_height=1000,
    )
    return extractor._build_blocks(
        by_id,
        lines,
        set(),
        page_index=0,
        page_width=1000,
        page_height=1000,
        page_image=None,
        allow_figures=False,
    )


def test_textract_keeps_multiline_paragraph_as_one_block_by_default():
    raw = [
        _word("w1", "First", 0.1, 0.1, 0.08, 0.03),
        _word("w2", "line", 0.19, 0.1, 0.06, 0.03),
        _word("w3", "Second", 0.1, 0.15, 0.1, 0.03),
        _word("w4", "line", 0.21, 0.15, 0.06, 0.03),
        _line("l1", "First line", ["w1", "w2"], 0.1, 0.1, 0.15, 0.03),
        _line("l2", "Second line", ["w3", "w4"], 0.1, 0.15, 0.17, 0.03),
        {
            "Id": "layout",
            "BlockType": "LAYOUT_TEXT",
            "Geometry": _geometry(0.09, 0.09, 0.2, 0.1),
            "Relationships": [{"Type": "CHILD", "Ids": ["l1", "l2"]}],
        },
    ]

    blocks = list(_core._iter_candidate_blocks(_extract_blocks(raw)))

    assert _core.PdfProcessingConfig().textract_split_lines is False
    assert len(blocks) == 1
    assert blocks[0].source_text == "First line\nSecond line"
    assert len(blocks[0].lines) == 2


def test_table_cell_wins_over_overlapping_layout_text():
    raw = [
        _word("w1", "AFRICA", 0.1, 0.1, 0.1, 0.04),
        _line("l1", "AFRICA", ["w1"], 0.1, 0.1, 0.1, 0.04),
        {
            "Id": "cell",
            "BlockType": "CELL",
            "Geometry": _geometry(0.09, 0.09, 0.13, 0.06),
            "Relationships": [{"Type": "CHILD", "Ids": ["w1"]}],
        },
        {
            "Id": "layout",
            "BlockType": "LAYOUT_TEXT",
            "Geometry": _geometry(0.1, 0.1, 0.1, 0.04),
            "Relationships": [{"Type": "CHILD", "Ids": ["l1"]}],
        },
    ]

    blocks = list(_core._iter_candidate_blocks(_extract_blocks(raw)))

    assert len(blocks) == 1
    assert blocks[0].block_type == "CELL"
    assert blocks[0].source_text == "AFRICA"


def test_table_cells_suppress_multiline_layout_text_overlay():
    raw = [
        _word("w1", "Ceiling", 0.10, 0.10, 0.10, 0.03),
        _word("w2", "CD198A", 0.42, 0.10, 0.10, 0.03),
        _line("l1", "Ceiling", ["w1"], 0.10, 0.10, 0.10, 0.03),
        _line("l2", "CD198A", ["w2"], 0.42, 0.10, 0.10, 0.03),
        {
            "Id": "cell1",
            "BlockType": "CELL",
            "Geometry": _geometry(0.08, 0.08, 0.25, 0.08),
            "Relationships": [{"Type": "CHILD", "Ids": ["w1"]}],
        },
        {
            "Id": "cell2",
            "BlockType": "CELL",
            "Geometry": _geometry(0.38, 0.08, 0.25, 0.08),
            "Relationships": [{"Type": "CHILD", "Ids": ["w2"]}],
        },
        {
            "Id": "layout",
            "BlockType": "LAYOUT_TEXT",
            "Geometry": _geometry(0.08, 0.08, 0.55, 0.08),
            "Relationships": [{"Type": "CHILD", "Ids": ["l1", "l2"]}],
        },
    ]

    blocks = list(_core._iter_candidate_blocks(_extract_blocks(raw)))

    assert [block.block_type for block in blocks] == ["CELL", "CELL"]
    assert [block.source_text for block in blocks] == ["Ceiling", "CD198A"]


def test_layout_list_is_split_into_items_not_individual_wrapped_lines():
    raw = [
        _word("w1", "First", 0.1, 0.1, 0.08, 0.03),
        _word("w2", "continued", 0.13, 0.14, 0.12, 0.03),
        _word("w3", "Second", 0.1, 0.23, 0.1, 0.03),
        _line("l1", "- First item", ["w1"], 0.1, 0.1, 0.2, 0.03),
        _line("l2", "continued text", ["w2"], 0.13, 0.14, 0.2, 0.03),
        _line("l3", "- Second item", ["w3"], 0.1, 0.23, 0.2, 0.03),
        {
            "Id": "list",
            "BlockType": "LAYOUT_LIST",
            "Geometry": _geometry(0.09, 0.09, 0.25, 0.2),
            "Relationships": [{"Type": "CHILD", "Ids": ["l1", "l2", "l3"]}],
        },
    ]

    blocks = list(_core._iter_candidate_blocks(_extract_blocks(raw)))

    assert [block.source_text for block in blocks] == [
        "- First item\ncontinued text",
        "- Second item",
    ]


def test_figure_uses_main_textract_lines_without_second_request(tmp_path):
    class Client:
        def __init__(self):
            self.calls = 0

        def analyze_document(self, **_kwargs):
            self.calls += 1
            return {
                "Blocks": [
                    _word("w1", "ADVANCED", 0.1, 0.1, 0.2, 0.05),
                    _line("l1", "ADVANCED", ["w1"], 0.1, 0.1, 0.2, 0.05),
                    {
                        "Id": "layout",
                        "BlockType": "LAYOUT_TEXT",
                        "Geometry": _geometry(0.1, 0.1, 0.2, 0.05),
                        "Relationships": [{"Type": "CHILD", "Ids": ["l1"]}],
                    },
                    {
                        "Id": "figure",
                        "BlockType": "LAYOUT_FIGURE",
                        "Geometry": _geometry(0.05, 0.05, 0.4, 0.3),
                        "Relationships": [{"Type": "CHILD", "Ids": ["layout"]}],
                    },
                ]
            }

    client = Client()
    extractor = _core.TextractLayoutExtractor(
        client,
        log_path=tmp_path / "textract-cache.jsonl",
    )
    page = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (500, 500), "white"),
        width=500,
        height=500,
        dpi=300,
    )

    layout = extractor.analyze(page)
    blocks = list(_core._iter_candidate_blocks(layout.blocks))

    assert client.calls == 1
    assert [block.source_text for block in blocks] == ["ADVANCED"]
    assert blocks[0].parent_block_type == "FIGURE"


def test_figure_keeps_high_confidence_short_label_without_second_request(tmp_path):
    class Client:
        def __init__(self):
            self.calls = 0

        def analyze_document(self, **_kwargs):
            self.calls += 1
            return {
                "Blocks": [
                    _word("w1", "GLOBAL", 0.1, 0.1, 0.2, 0.05),
                    _word("w2", "AND", 0.1, 0.2, 0.1, 0.05),
                    _line("l1", "GLOBAL", ["w1"], 0.1, 0.1, 0.2, 0.05),
                    _line("l2", "AND", ["w2"], 0.1, 0.2, 0.1, 0.05),
                    {
                        "Id": "layout",
                        "BlockType": "LAYOUT_TEXT",
                        "Geometry": _geometry(0.1, 0.1, 0.2, 0.15),
                        "Relationships": [{"Type": "CHILD", "Ids": ["l1", "l2"]}],
                    },
                    {
                        "Id": "figure",
                        "BlockType": "LAYOUT_FIGURE",
                        "Geometry": _geometry(0.05, 0.05, 0.4, 0.3),
                        "Relationships": [{"Type": "CHILD", "Ids": ["layout"]}],
                    },
                ]
            }

    client = Client()
    extractor = _core.TextractLayoutExtractor(
        client,
        log_path=tmp_path / "textract-cache.jsonl",
    )
    page = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (500, 500), "white"),
        width=500,
        height=500,
        dpi=300,
    )

    layout = extractor.analyze(page)
    blocks = list(_core._iter_candidate_blocks(layout.blocks))

    assert client.calls == 1
    assert any(block.source_text == "AND" for block in blocks)
    assert all(block.parent_block_type == "FIGURE" for block in blocks)


def test_filter_skips_logo_email_phone_and_decorative_text():
    def layout(page_index: int, texts: list[tuple[str, str]]):
        blocks = [
            _core.BlockRegion(
                page_index=page_index,
                block_id=f"{page_index}-{idx}",
                block_type="TEXT",
                bbox=(10, 10 + idx * 20, 180, 28 + idx * 20),
                source_text=text,
                parent_block_type=parent,
            )
            for idx, (text, parent) in enumerate(texts)
        ]
        page = _core.PageImage(
            page_index=page_index,
            image=Image.new("RGB", (200, 200), "white"),
            width=200,
            height=200,
            dpi=300,
        )
        return _core.TextractPageData(page, {}, {}, blocks)

    first = layout(
        0,
        [
            ("HYUNDAI", "FIGURE"),
            ("HYUNDAI ELEVATOR", "FIGURE"),
            ("HYUNDAI ELEVATOR L", "TEXT"),
            ("ADVANCED TECHNOLOGY", "FIGURE"),
            ("support@example.com", "TEXT"),
            ("+ + + +", "FIGURE"),
            ("T. +7 777 123-45-67", "TEXT"),
        ],
    )
    second = layout(
        1,
        [
            ("HYUNDAI", "FIGURE"),
            ("HYUNDAI ELEVATOR", "FIGURE"),
            ("HYUNDAI ELEVATOR LUXEN", "TEXT"),
        ],
    )

    _core._mark_nontranslatable_blocks([first, second])

    states = {
        block.source_text: block.should_translate
        for block in first.blocks
    }
    assert states["HYUNDAI"] is False
    assert states["HYUNDAI ELEVATOR L"] is False
    assert states["ADVANCED TECHNOLOGY"] is True
    assert states["support@example.com"] is False
    assert states["+ + + +"] is False
    assert states["T. +7 777 123-45-67"] is False


def test_filter_skips_partial_logo_fragments_and_nearby_tagline():
    def block(page_index: int, idx: int, text: str, bbox, parent: str = "TEXT"):
        return _core.BlockRegion(
            page_index=page_index,
            block_id=f"{page_index}-{idx}",
            block_type="TEXT",
            bbox=bbox,
            source_text=text,
            parent_block_type=parent,
        )

    page0 = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (800, 800), "white"),
        width=800,
        height=800,
        dpi=300,
    )
    page1 = _core.PageImage(
        page_index=1,
        image=Image.new("RGB", (800, 800), "white"),
        width=800,
        height=800,
        dpi=300,
    )
    first = _core.TextractPageData(
        page0,
        {},
        {},
        [
            block(0, 0, "HYUNDAI ELEVATOR LUXEN", (100, 50, 320, 80), "FIGURE"),
            block(0, 1, "Premium Gea", (180, 500, 360, 530)),
            block(0, 2, "LUX", (160, 585, 380, 660)),
            block(0, 3, "Feature text", (500, 500, 650, 530)),
        ],
    )
    second = _core.TextractPageData(
        page1,
        {},
        {},
        [
            block(1, 0, "HYUNDAI ELEVATOR LUXEN", (100, 50, 320, 80), "FIGURE"),
        ],
    )

    _core._mark_nontranslatable_blocks([first, second])

    states = {candidate.source_text: candidate.should_translate for candidate in first.blocks}
    assert states["HYUNDAI ELEVATOR LUXEN"] is False
    assert states["LUX"] is False
    assert states["Premium Gea"] is False
    assert states["Feature text"] is True


def test_renderer_inpaints_only_masked_pixels():
    image = np.full((100, 260, 3), 255, dtype=np.uint8)
    cv2.line(image, (0, 88), (259, 88), (0, 0, 0), 2)
    cv2.putText(
        image,
        "SOURCE",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    roi = image.copy()
    mask = _core.build_text_mask_from_word_boxes(
        roi,
        [(15, 25, 210, 50)],
        offset_x=0,
        offset_y=0,
        dilation_kernel=3,
    )
    assert mask is not None
    region = _core.RegionInfo(
        page_index=0,
        region_id=0,
        bbox=(0, 0, image.shape[1], image.shape[0]),
        text_orientation_deg=0.0,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="SOURCE",
        translated_text=" ",
        mask=mask,
    )
    page = _core.PageImage(
        page_index=0,
        image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
        width=image.shape[1],
        height=image.shape[0],
        dpi=300,
    )

    rendered = np.array(_core.Renderer(_core.PdfProcessingConfig()).render_page(page, [region]))
    source_rgb = np.array(page.image)
    outside = mask == 0

    assert np.array_equal(rendered[outside], source_rgb[outside])
    assert rendered[mask > 0].mean() > source_rgb[mask > 0].mean()
    assert np.array_equal(rendered[88, :], source_rgb[88, :])


def test_translation_layer_v2_and_legacy_layer_are_both_readable(tmp_path):
    layer = tmp_path / "translation.json"
    region = _core.RegionInfo(
        page_index=0,
        region_id=7,
        bbox=(10, 20, 100, 30),
        text_orientation_deg=0.0,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="Safety",
        translated_text="Безопасность",
        stable_id="stable-123",
    )

    _core.save_translation_mapping([region], layer, lambda _message: None)
    payload = json.loads(layer.read_text(encoding="utf-8"))

    assert payload["version"] == 5
    assert payload["pages"][0]["blocks"][0]["semantic_type"] == "unknown"
    assert payload["pages"][0]["blocks"][0]["fit_strategy"] == "default"
    assert payload["pages"][0]["blocks"][0]["qa_flags"] == []
    assert _core.load_translation_stable_mapping(layer, lambda _message: None) == {
        "stable-123": "Безопасность"
    }
    assert _core.load_translation_mapping(layer, lambda _message: None) == {
        (0, 7): "Безопасность"
    }

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "version": 1,
                "pages": [
                    {
                        "page_index": 0,
                        "regions": [
                            {
                                "region_id": 3,
                                "text": "Door",
                                "translated_text": "Дверь",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert _core.load_translation_mapping(legacy, lambda _message: None) == {
        (0, 3): "Дверь"
    }


def test_block_region_translator_reuses_checkpoint_canonical_cache(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf-bytes")
    checkpoint = tmp_path / "checkpoint.json"
    store = TranslationCheckpointStore(
        path=checkpoint,
        job_type="pdf-scanned",
        document_sha256=file_sha256(source),
        source_lang="en",
        target_lang="ru",
        translator="fake",
    )

    class PrefixEngine:
        calls = 0

        def translate_many(self, texts):
            type(self).calls += 1
            return [f"ru:{text}" for text in texts]

    blocks = [
        _core.BlockRegion(
            page_index=0,
            block_id="a",
            block_type="TEXT",
            bbox=(0, 0, 10, 10),
            source_text="Safety",
        ),
        _core.BlockRegion(
            page_index=0,
            block_id="b",
            block_type="TEXT",
            bbox=(0, 20, 10, 30),
            source_text="Door",
        ),
    ]
    cache = _core.BlockRegionTranslator(PrefixEngine(), batch_size=1).translate(
        blocks,
        {},
        checkpoint_store=store,
    )

    assert PrefixEngine.calls == 2
    assert checkpoint.exists()
    assert cache[_core._canonicalize_text_for_dedup("Safety")] == "ru:Safety"

    loaded = TranslationCheckpointStore(
        path=checkpoint,
        job_type="pdf-scanned",
        document_sha256=file_sha256(source),
        source_lang="en",
        target_lang="ru",
    )

    class FailIfCalled:
        def translate_many(self, _texts):
            raise AssertionError("checkpoint should cover canonical cache")

    resumed_blocks = [
        _core.BlockRegion(
            page_index=0,
            block_id="a",
            block_type="TEXT",
            bbox=(0, 0, 10, 10),
            source_text="Safety",
        )
    ]
    _core.BlockRegionTranslator(FailIfCalled(), batch_size=1).translate(
        resumed_blocks,
        loaded.text_map("pdf-scanned:canonical"),
        checkpoint_store=loaded,
    )

    assert resumed_blocks[0].translated_text == "ru:Safety"


def test_unchanged_translation_is_not_cleaned_or_redrawn():
    block = _core.BlockRegion(
        page_index=0,
        block_id="brand",
        block_type="TEXT",
        bbox=(10, 10, 100, 30),
        source_text="LUXEN",
        translated_text="LUXEN",
    )

    _core._disable_unchanged_blocks([block])

    assert block.render_enabled is False


def test_export_at_render_dpi_preserves_pdf_page_size(tmp_path):
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    document = fitz.open()
    document.new_page(width=360, height=240)
    document.save(source)
    document.close()

    config = _core.PdfProcessingConfig(dpi=700, render_dpi=144).normalized()
    page = _core.PageLoader(config).load(source)[0]
    _core.Exporter().to_pdf([page.image], output, dpi=page.dpi)

    result = fitz.open(output)
    try:
        assert abs(result[0].rect.width - 360) < 1.0
        assert abs(result[0].rect.height - 240) < 1.0
    finally:
        result.close()


def test_near_white_text_color_is_darkened_on_white_background():
    assert _core._boost_text_contrast((215, 215, 215), (255, 255, 255)) == (155, 155, 155)
    assert _core._boost_text_contrast((255, 255, 255), (255, 255, 255)) == (155, 155, 155)


def test_large_non_figure_heading_gets_expanded_region():
    block = _core.BlockRegion(
        page_index=0,
        block_id="title",
        block_type="TITLE",
        bbox=(500, 100, 800, 240),
        source_text="FIXTUR\nDESIG",
        lines=[
            _core.LineRegion(
                page_index=0,
                line_id="l1",
                bbox=(500, 100, 800, 170),
                source_text="FIXTUR",
            ),
            _core.LineRegion(
                page_index=0,
                line_id="l2",
                bbox=(560, 180, 790, 240),
                source_text="DESIG",
            ),
        ],
        parent_block_type="TITLE",
        parent_bbox=(500, 100, 800, 240),
    )

    bbox = _core._bounded_block_bbox(
        block,
        [block],
        _core.PdfProcessingConfig().normalized(),
        page_width=1200,
        page_height=800,
    )

    assert bbox is not None
    assert bbox[0] < 500
    assert bbox[2] > 800


def test_table_cell_alignment_uses_word_boxes_not_cell_frame():
    left_cell = _core.BlockRegion(
        page_index=0,
        block_id="left",
        block_type="CELL",
        bbox=(100, 100, 500, 180),
        source_text="Left text",
        word_boxes=[(120, 120, 120, 30)],
        is_table=True,
        parent_bbox=(100, 100, 500, 180),
    )
    right_cell = _core.BlockRegion(
        page_index=0,
        block_id="right",
        block_type="CELL",
        bbox=(100, 200, 500, 280),
        source_text="Right text",
        word_boxes=[(350, 220, 120, 30)],
        is_table=True,
        parent_bbox=(100, 200, 500, 280),
    )
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=Path("missing-textract-cache.jsonl"),
    )

    assert extractor._infer_alignment(left_cell) == "left"
    assert extractor._infer_alignment(right_cell) == "right"


def test_output_dpi_is_independent_from_ocr_render_dpi():
    config = _core.PdfProcessingConfig(dpi=200, render_dpi=400).normalized()

    assert config.dpi == 200
    assert config.render_dpi == 400


def test_wrapped_textract_lines_merge_before_translation():
    first = _core.BlockRegion(
        page_index=0,
        block_id="first",
        block_type="TEXT",
        bbox=(100, 100, 800, 150),
        source_text="Install the guide rail at the",
        parent_block_id="figure",
    )
    second = _core.BlockRegion(
        page_index=0,
        block_id="second",
        block_type="TEXT",
        bbox=(105, 175, 780, 225),
        source_text="specified position.",
        parent_block_id="figure",
    )

    merged = _core._merge_wrapped_text_blocks([first, second])

    assert len(merged) == 1
    assert merged[0].source_text == "Install the guide rail at the specified position."
    assert merged[0].bbox == (100, 100, 800, 225)
    assert merged[0].fit_strategy == "wrap"


def test_separate_list_items_are_not_merged_as_wrapped_lines():
    first = _core.BlockRegion(
        page_index=0,
        block_id="first",
        block_type="TEXT",
        bbox=(100, 100, 800, 150),
        source_text="Preparation",
        parent_block_id="figure",
    )
    second = _core.BlockRegion(
        page_index=0,
        block_id="second",
        block_type="TEXT",
        bbox=(100, 175, 800, 225),
        source_text="a. Install the bracket",
        parent_block_id="figure",
    )

    assert len(_core._merge_wrapped_text_blocks([first, second])) == 2


def test_tall_body_text_is_not_treated_as_large_heading():
    body = _core.BlockRegion(
        page_index=0,
        block_id="body",
        block_type="TEXT",
        bbox=(100, 100, 900, 170),
        source_text="INSTALL THE GUIDE RAIL",
        lines=[
            _core.LineRegion(
                page_index=0,
                line_id="line",
                bbox=(100, 100, 900, 165),
                source_text="INSTALL THE GUIDE RAIL",
            )
        ],
        parent_block_type="TEXT",
        parent_bbox=(100, 100, 900, 170),
    )

    assert _core._needs_large_heading_recovery(body) is False


def test_tall_narrow_unrotated_block_stays_horizontal():
    block = _core.BlockRegion(
        page_index=0,
        block_id="narrow",
        block_type="TEXT",
        bbox=(40, 20, 110, 350),
        source_text="Narrow column",
        word_boxes=[(50, 30, 45, 28)],
        render_id=1,
        parent_bbox=(40, 20, 110, 350),
    )
    page = np.full((400, 200, 3), 255, dtype=np.uint8)

    region = _core._block_to_region_info(
        block,
        page,
        _core.PdfProcessingConfig().normalized(),
        page_width=200,
        page_height=400,
        candidates=[block],
    )

    assert region is not None
    assert region.is_vertical is False


def test_explicitly_rotated_block_uses_vertical_layout_without_tiny_font():
    page = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (120, 360), "white"),
        width=120,
        height=360,
        dpi=200,
    )
    region = _core.RegionInfo(
        page_index=0,
        region_id=1,
        bbox=(20, 20, 60, 300),
        text_orientation_deg=-90,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="Vertical label",
        translated_text="Vertical label",
        mask=np.zeros((300, 60), dtype=np.uint8),
        font_size_estimate=30,
        source_font_size_estimate=30,
        is_vertical=True,
    )

    rendered = _core.Renderer(_core.PdfProcessingConfig()).render_page(page, [region])

    assert region.rendered_font_size >= 28
    crop = np.asarray(rendered)[20:320, 20:80]
    dark_y, dark_x = np.where(np.any(crop < 245, axis=2))
    assert dark_x.size > 0
    assert np.ptp(dark_y) > np.ptp(dark_x) * 2


def test_near_vertical_rotation_is_clipped_to_region_bbox():
    page = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (260, 420), "white"),
        width=260,
        height=420,
        dpi=200,
    )
    region = _core.RegionInfo(
        page_index=0,
        region_id=1,
        bbox=(100, 60, 60, 300),
        text_orientation_deg=-78,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text="Near vertical label",
        translated_text="Near vertical label",
        mask=np.zeros((300, 60), dtype=np.uint8),
        font_size_estimate=28,
        source_font_size_estimate=28,
        is_vertical=True,
    )

    rendered = np.asarray(
        _core.Renderer(_core.PdfProcessingConfig()).render_page(page, [region])
    )
    outside = rendered.copy()
    outside[60:360, 100:160] = 255

    assert np.all(outside == 255)
    assert np.any(rendered[60:360, 100:160] < 245)


def test_font_estimate_prefers_word_height_over_inflated_line_box():
    line = _core.LineRegion(
        page_index=0,
        line_id="line-1",
        bbox=(10, 10, 300, 70),
        source_text="Normal body text",
        word_boxes=[(10, 20, 70, 23), (90, 19, 80, 24), (180, 18, 90, 25)],
    )
    block = _core.BlockRegion(
        page_index=0,
        block_id="block-1",
        block_type="TEXT",
        bbox=line.bbox,
        lines=[line],
        source_text=line.source_text,
    )

    assert _core._estimate_font_size_from_block(block) == 24


def test_vertical_font_estimate_uses_word_width_as_glyph_height():
    block = _core.BlockRegion(
        page_index=0,
        block_id="vertical",
        block_type="TEXT",
        bbox=(20, 20, 80, 320),
        source_text="Vertical label",
        word_boxes=[(30, 40, 28, 180), (31, 225, 30, 80)],
        rotation_deg=90,
    )

    assert _core._estimate_font_size_from_block(block) == 30


def test_semantic_vertical_font_estimate_uses_line_width_without_rotation():
    line = _core.LineRegion(
        page_index=0,
        line_id="semantic-vertical-line",
        bbox=(20, 20, 50, 220),
        source_text="Vertical label",
    )
    block = _core.BlockRegion(
        page_index=0,
        block_id="semantic-vertical",
        block_type="TEXT",
        bbox=line.bbox,
        lines=[line],
        source_text=line.source_text,
        semantic_type="vertical",
    )

    assert _core._estimate_font_size_from_block(block) == 30


def test_unrotated_tall_word_bbox_uses_minor_axis_for_font_size() -> None:
    block = _core.BlockRegion(
        page_index=0,
        block_id="tall-word-cell",
        block_type="CELL",
        bbox=(400, 2396, 681, 2776),
        source_text="In installation",
        word_boxes=[(517, 2700, 41, 42), (517, 2450, 43, 235)],
        is_table=True,
    )

    assert _core._estimate_font_size_from_block(block) == 43


def test_single_narrow_horizontal_glyph_keeps_its_height_estimate() -> None:
    block = _core.BlockRegion(
        page_index=0,
        block_id="single-i",
        block_type="TEXT",
        bbox=(10, 10, 30, 60),
        source_text="I",
        word_boxes=[(12, 15, 6, 40)],
    )

    assert _core._estimate_font_size_from_block(block) == 40


def test_compatible_body_font_sizes_are_harmonized_without_losing_roles():
    def region(
        region_id: int,
        size: int,
        *,
        semantic_type: str = "body",
        is_table_cell: bool = False,
        block_type: str = "TEXT",
    ):
        return _core.RegionInfo(
            page_index=0,
            region_id=region_id,
            bbox=(10, region_id * 130, 500, 120),
            text_orientation_deg=0,
            text_color=(0, 0, 0),
            background_color=(255, 255, 255),
            source_text=f"Region {region_id}",
            font_size_estimate=size,
            source_font_size_estimate=size,
            semantic_type=semantic_type,
            is_table_cell=is_table_cell,
            block_type=block_type,
        )

    body = [region(1, 59), region(2, 62), region(3, 64)]
    footnote = region(4, 32, semantic_type="footnote")
    heading = region(5, 82, semantic_type="heading")
    table_cell = region(6, 48, semantic_type="table_cell", is_table_cell=True)
    title = region(7, 64, semantic_type="unknown", block_type="TITLE")

    _core._harmonize_region_font_sizes([*body, footnote, heading, table_cell, title])

    assert [item.font_size_estimate for item in body] == [62, 62, 62]
    assert footnote.font_size_estimate == 32
    assert heading.font_size_estimate == 82
    assert table_cell.font_size_estimate == 48
    assert title.font_size_estimate == 64


def test_distinct_source_blocks_survive_same_translation_collision():
    page = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (220, 140), "white"),
        width=220,
        height=140,
        dpi=200,
    )
    blocks = [
        _core.BlockRegion(
            page_index=0,
            block_id="car",
            block_type="TEXT",
            bbox=(20, 30, 150, 80),
            source_text="car",
            translated_text="кабина",
            render_id=1,
        ),
        _core.BlockRegion(
            page_index=0,
            block_id="cabin",
            block_type="TEXT",
            bbox=(25, 32, 155, 82),
            source_text="cabin",
            translated_text="кабина",
            render_id=2,
        ),
    ]

    regions = _core._build_regions_from_blocks(
        page,
        blocks,
        _core.PdfProcessingConfig(),
    )

    assert {region.source_text for region in regions} == {"car", "cabin"}


def test_detects_native_dpi_of_full_page_scan(tmp_path):
    source = tmp_path / "scan.pdf"
    jpeg = tmp_path / "scan.jpg"
    Image.new("L", (1653, 2338), 240).save(jpeg, "JPEG", quality=85)
    document = fitz.open()
    page = document.new_page(width=595, height=841)
    page.insert_image(page.rect, filename=str(jpeg))
    document.save(source)
    document.close()

    assert abs((_core._detect_pdf_raster_dpi(source) or 0) - 200) <= 1


def test_native_scan_dpi_caps_only_final_export_resolution():
    page = _core.PageImage(
        page_index=0,
        image=Image.new("RGB", (3306, 4676), "white"),
        width=3306,
        height=4676,
        dpi=400,
        source_dpi=200,
    )

    assert _core._effective_output_dpi(400, [page]) == 200
    assert page.dpi == 400


def test_small_pdf_image_is_not_mistaken_for_full_page_scan(tmp_path):
    source = tmp_path / "hybrid.pdf"
    jpeg = tmp_path / "small.jpg"
    Image.new("L", (300, 200), 220).save(jpeg, "JPEG", quality=85)
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(fitz.Rect(50, 50, 250, 183), filename=str(jpeg))
    document.save(source)
    document.close()

    assert _core._detect_pdf_raster_dpi(source) is None


def test_mixed_raster_vector_pdf_does_not_get_global_dpi_cap(tmp_path):
    source = tmp_path / "mixed.pdf"
    jpeg = tmp_path / "page.jpg"
    Image.new("L", (1653, 2338), 235).save(jpeg, "JPEG", quality=85)
    document = fitz.open()
    for _ in range(4):
        page = document.new_page(width=595, height=841)
        page.insert_image(page.rect, filename=str(jpeg))
    document.new_page(width=595, height=841)
    document.save(source)
    document.close()

    assert _core._detect_pdf_raster_dpi(source) is None


def test_optimized_export_downsamples_grayscale_and_preserves_page_size(tmp_path):
    output = tmp_path / "optimized.pdf"
    image = Image.new("RGB", (400, 600), (230, 230, 230))
    _core.Exporter().to_pdf(
        [image],
        output,
        dpi=72,
        input_dpi=144,
        page_sizes_points=[(200.0, 300.0)],
    )

    document = fitz.open(output)
    try:
        assert abs(document[0].rect.width - 200.0) < 0.1
        assert abs(document[0].rect.height - 300.0) < 0.1
        embedded = document[0].get_images(full=True)
        assert len(embedded) == 1
        assert embedded[0][2:4] == (200, 300)
        assert fitz.Pixmap(document, embedded[0][0]).n == 1
    finally:
        document.close()


def test_optimized_export_preserves_real_color(tmp_path):
    output = tmp_path / "color.pdf"
    array = np.full((200, 200, 3), 230, dtype=np.uint8)
    array[40:160, 40:160] = (220, 30, 30)
    _core.Exporter().to_pdf(
        [Image.fromarray(array, "RGB")],
        output,
        dpi=100,
        input_dpi=100,
    )

    document = fitz.open(output)
    try:
        image_xref = document[0].get_images(full=True)[0][0]
        pixmap = fitz.Pixmap(document, image_xref)
        assert pixmap.n == 3
        red, green, blue = pixmap.pixel(pixmap.width // 2, pixmap.height // 2)[:3]
        assert red > green + 100
        assert red > blue + 100
    finally:
        document.close()


def test_optimized_scan_export_is_materially_smaller_than_legacy_rgb(tmp_path):
    rng = np.random.default_rng(7)
    scan = np.full((900, 600), 238, dtype=np.int16)
    scan += rng.integers(-9, 10, scan.shape, dtype=np.int16)
    scan[::45, 30:570] = 35
    scan[:, ::120] = 80
    scan = np.clip(scan, 0, 255).astype(np.uint8)
    rgb_scan = Image.fromarray(np.repeat(scan[:, :, None], 3, axis=2), "RGB")

    legacy = tmp_path / "legacy-rgb.pdf"
    rgb_scan.save(legacy, "PDF", resolution=144, quality=75)
    optimized = tmp_path / "optimized-gray.pdf"
    _core.Exporter().to_pdf(
        [rgb_scan],
        optimized,
        dpi=72,
        input_dpi=144,
        page_sizes_points=[(300.0, 450.0)],
    )

    assert optimized.stat().st_size < legacy.stat().st_size * 0.6


def test_export_replace_error_is_not_masked_and_temp_file_is_removed(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "locked.pdf"

    def fail_replace(_source, _destination):
        raise PermissionError("output is open")

    monkeypatch.setattr(export.os, "replace", fail_replace)

    with pytest.raises(PermissionError, match="output is open"):
        _core.Exporter().to_pdf(
            [Image.new("L", (100, 100), 240)],
            output,
            dpi=100,
            input_dpi=100,
        )

    assert not list(tmp_path.glob(".*.tmp.pdf"))


@pytest.mark.parametrize("layout_type", ["LAYOUT_TEXT", "LAYOUT_KEY_VALUE"])
def test_page_spanning_layout_text_is_split_into_aligned_zones(tmp_path, layout_type):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    lines = {
        f"line-{index}": _core.LineRegion(
            page_index=0,
            line_id=f"line-{index}",
            bbox=bbox,
            source_text=text,
        )
        for index, (bbox, text) in enumerate(
            [
                ((100, 100, 800, 130), "First paragraph line"),
                ((100, 140, 800, 170), "continues on this line"),
                ((100, 180, 780, 210), "and ends here."),
                ((600, 600, 850, 630), "Diagram label"),
                ((600, 640, 850, 670), "second label line"),
                ((600, 680, 850, 710), "label ending."),
            ]
        )
    }
    raw = {
        "Id": "layout",
        "BlockType": layout_type,
        "Geometry": _geometry(0.1, 0.1, 0.75, 0.61),
        "Relationships": [{"Type": "CHILD", "Ids": list(lines)}],
    }

    blocks = extractor._build_text_block(
        raw,
        {"layout": raw},
        lines,
        set(),
        set(),
        block_type="TEXT",
        page_index=0,
        page_width=1000,
        page_height=1000,
    )

    assert len(blocks) == 2
    assert [len(block.lines) for block in blocks] == [3, 3]
    assert max(block.bbox[3] - block.bbox[1] for block in blocks) < 150


def test_contiguous_large_paragraph_is_not_split(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    lines = {
        f"line-{index}": _core.LineRegion(
            page_index=0,
            line_id=f"line-{index}",
            bbox=(100, 100 + index * 40, 800, 130 + index * 40),
            source_text=f"Paragraph line {index}",
        )
        for index in range(6)
    }
    raw = {
        "Id": "layout",
        "BlockType": "LAYOUT_TEXT",
        "Geometry": _geometry(0.1, 0.1, 0.7, 0.23),
        "Relationships": [{"Type": "CHILD", "Ids": list(lines)}],
    }

    blocks = extractor._build_text_block(
        raw,
        {"layout": raw},
        lines,
        set(),
        set(),
        block_type="TEXT",
        page_index=0,
        page_width=1000,
        page_height=1000,
    )

    assert len(blocks) == 1
    assert len(blocks[0].lines) == 6


def test_interleaved_two_column_layout_is_split_into_two_columns(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    lines = {}
    for row in range(6):
        for column, x0 in (("left", 100), ("right", 600)):
            line_id = f"{column}-{row}"
            lines[line_id] = _core.LineRegion(
                page_index=0,
                line_id=line_id,
                bbox=(x0, 100 + row * 40, x0 + 250, 130 + row * 40),
                source_text=f"{column} column row {row}",
            )
    raw = {
        "Id": "layout-columns",
        "BlockType": "LAYOUT_TEXT",
        "Geometry": _geometry(0.1, 0.1, 0.75, 0.23),
        "Relationships": [{"Type": "CHILD", "Ids": list(lines)}],
    }

    blocks = extractor._build_text_block(
        raw,
        {"layout-columns": raw},
        lines,
        set(),
        set(),
        block_type="TEXT",
        page_index=0,
        page_width=1000,
        page_height=1000,
    )

    assert len(blocks) == 2
    assert sorted(len(block.lines) for block in blocks) == [6, 6]
    assert all((block.bbox[2] - block.bbox[0]) < 300 for block in blocks)


def test_table_cell_text_zones_do_not_cover_image_gap(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    block = _core.BlockRegion(
        page_index=0,
        block_id="cell",
        block_type="CELL",
        bbox=(100, 100, 900, 900),
        source_text="Top paragraph\nCaption below photo",
        lines=[
            _core.LineRegion(0, "top-1", (120, 130, 850, 165), "Top paragraph"),
            _core.LineRegion(0, "top-2", (120, 175, 850, 210), "continues here"),
            _core.LineRegion(0, "bottom", (120, 790, 700, 825), "Caption below photo"),
        ],
        is_table=True,
        parent_block_id="cell",
        parent_block_type="CELL",
        parent_bbox=(100, 100, 900, 900),
    )

    zones = extractor._split_table_cell_zones(block)

    assert len(zones) == 2
    assert zones[0].bbox[3] < 400
    assert zones[1].bbox[1] > 400
    assert all(zone.is_table for zone in zones)


def test_table_cell_separates_switch_labels_one_and_a_half_lines_apart(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    block = _core.BlockRegion(
        page_index=0,
        block_id="switch-cell",
        block_type="CELL",
        bbox=(100, 3000, 900, 3800),
        source_text="Lower limit switch\nLower ultimate\nlimit switch",
        lines=[
            _core.LineRegion(
                0,
                "lower-limit",
                (130, 3232, 700, 3283),
                "Lower limit switch",
            ),
            _core.LineRegion(
                0,
                "lower-ultimate",
                (130, 3385, 700, 3436),
                "Lower ultimate",
            ),
            _core.LineRegion(
                0,
                "limit-switch",
                (130, 3466, 700, 3517),
                "limit switch",
            ),
        ],
        is_table=True,
        parent_block_id="switch-cell",
        parent_block_type="CELL",
        parent_bbox=(100, 3000, 900, 3800),
    )

    zones = extractor._split_table_cell_zones(block)

    assert len(zones) == 2
    assert zones[0].source_text == "Lower limit switch"
    assert zones[1].source_text == "Lower ultimate\nlimit switch"


def test_table_cell_splits_marker_description_and_quantity_columns(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    word_boxes = [
        (1883, 2869, 31, 37),
        (1979, 2887, 74, 38),
        (2076, 2888, 135, 37),
        (2239, 2888, 41, 37),
        (2514, 2889, 24, 38),
        (2562, 2900, 72, 36),
    ]
    block = _core.BlockRegion(
        page_index=0,
        block_id="hardware-cell",
        block_type="CELL",
        bbox=(1823, 2820, 2641, 2948),
        source_text="B Flat washer 12 2 pcs.",
        lines=[
            _core.LineRegion(
                0,
                "hardware-line",
                (1883, 2869, 2634, 2936),
                "B Flat washer 12 2 pcs.",
                word_boxes=word_boxes,
            )
        ],
        word_boxes=word_boxes,
        is_table=True,
        parent_block_id="hardware-cell",
        parent_block_type="CELL",
        parent_bbox=(1823, 2820, 2641, 2948),
    )

    zones = extractor._split_table_cell_zones(block)

    assert [zone.source_text for zone in zones] == [
        "B",
        "Flat washer 12",
        "2 pcs.",
    ]
    assert [zone.alignment for zone in zones] == ["center", "left", "right"]
    assert all(
        zones[index].bbox[2] <= zones[index + 1].bbox[0]
        for index in range(len(zones) - 1)
    )


def test_table_cell_marker_split_tolerates_one_pixel_bbox_rounding(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    word_boxes = [
        (1893, 1019, 35, 38),
        (1975, 1036, 81, 38),
        (2079, 1037, 133, 37),
    ]
    block = _core.BlockRegion(
        page_index=0,
        block_id="rounded-marker-cell",
        block_type="CELL",
        bbox=(1825, 970, 2308, 1097),
        source_text="A Bolt M5x25",
        lines=[
            _core.LineRegion(
                0,
                "rounded-marker-line",
                (1893, 1019, 2212, 1074),
                "A Bolt M5x25",
                word_boxes=word_boxes,
            )
        ],
        word_boxes=word_boxes,
        is_table=True,
    )

    zones = extractor._split_table_cell_zones(block)

    assert [zone.source_text for zone in zones] == ["A", "Bolt M5x25"]


def test_table_cell_splits_offset_circle_marker_without_a_wide_gap(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    word_boxes = [
        (1834, 2197, 33, 39),
        (1909, 2213, 126, 48),
        (2057, 2215, 134, 38),
        (2221, 2216, 42, 37),
    ]
    block = _core.BlockRegion(
        page_index=0,
        block_id="offset-circle-marker-cell",
        block_type="CELL",
        bbox=(1777, 2153, 2267, 2278),
        source_text="C Spring washer 10",
        lines=[
            _core.LineRegion(
                0,
                "offset-circle-marker-line",
                (1834, 2197, 2263, 2263),
                "C Spring washer 10",
                word_boxes=word_boxes,
            )
        ],
        word_boxes=word_boxes,
        is_table=True,
        alignment="center",
    )

    zones = extractor._split_table_cell_zones(block)

    assert [zone.source_text for zone in zones] == ["C", "Spring washer 10"]
    assert [zone.alignment for zone in zones] == ["center", "left"]


def test_full_block_build_preserves_split_column_alignment(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    raw_blocks = [
        _word("marker", "C", 0.10, 0.10, 0.03, 0.04),
        _word("spring", "Spring", 0.17, 0.12, 0.12, 0.05),
        _word("washer", "washer", 0.31, 0.12, 0.13, 0.04),
        _word("ten", "10", 0.46, 0.12, 0.04, 0.04),
        {
            "Id": "offset-marker-cell",
            "BlockType": "CELL",
            "Confidence": 90.0,
            "Geometry": _geometry(0.05, 0.08, 0.50, 0.12),
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": ["marker", "spring", "washer", "ten"],
                }
            ],
        },
    ]

    blocks = extractor._build_blocks(
        {block["Id"]: block for block in raw_blocks},
        {},
        set(),
        page_index=0,
        page_width=1000,
        page_height=1000,
        page_image=None,
        allow_figures=False,
    )
    columns = {block.source_text: block for block in blocks}

    assert columns["C"].alignment == "center"
    assert columns["C"].alignment_locked is True
    assert columns["Spring washer 10"].alignment == "left"
    assert columns["Spring washer 10"].alignment_locked is True


def test_table_cell_does_not_split_leading_dimension_variable(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    word_boxes = [
        (120, 130, 24, 38),
        (152, 130, 20, 38),
        (180, 130, 45, 38),
        (233, 130, 20, 38),
    ]
    block = _core.BlockRegion(
        page_index=0,
        block_id="dimension-cell",
        block_type="CELL",
        bbox=(100, 100, 300, 190),
        source_text="A = 5 ±1",
        lines=[
            _core.LineRegion(
                0,
                "dimension-line",
                (120, 130, 253, 168),
                "A = 5 ±1",
                word_boxes=word_boxes,
            )
        ],
        word_boxes=word_boxes,
        is_table=True,
    )

    assert extractor._split_table_cell_zones(block) == [block]


def test_table_cell_does_not_split_ordinary_a_bolt_phrase(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    word_boxes = [
        (120, 130, 24, 38),
        (154, 130, 70, 38),
        (234, 130, 90, 38),
    ]
    block = _core.BlockRegion(
        page_index=0,
        block_id="ordinary-phrase-cell",
        block_type="CELL",
        bbox=(100, 100, 350, 190),
        source_text="A Bolt remains",
        lines=[
            _core.LineRegion(
                0,
                "ordinary-phrase-line",
                (120, 130, 324, 168),
                "A Bolt remains",
                word_boxes=word_boxes,
            )
        ],
        word_boxes=word_boxes,
        is_table=True,
    )

    assert extractor._split_table_cell_zones(block) == [block]


def test_low_confidence_merged_cell_uses_individual_child_cells(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(min_confidence=60),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    raw_blocks = [
        _word("guide", "Guide", 0.20, 0.30, 0.08, 0.04),
        _word("rail", "rail", 0.29, 0.30, 0.05, 0.04),
        {
            **_word("noise", "IT", 0.39, 0.50, 0.04, 0.04),
            "Confidence": 12.0,
        },
        _word("leveling", "Leveling", 0.20, 0.60, 0.10, 0.04),
        _word("frame", "frame", 0.31, 0.60, 0.08, 0.04),
        {
            "Id": "child-guide",
            "BlockType": "CELL",
            "Confidence": 29.0,
            "RowIndex": 1,
            "ColumnIndex": 1,
            "Geometry": _geometry(0.10, 0.25, 0.40, 0.15),
            "Relationships": [{"Type": "CHILD", "Ids": ["guide", "rail"]}],
        },
        {
            "Id": "child-frame",
            "BlockType": "CELL",
            "Confidence": 29.0,
            "RowIndex": 2,
            "ColumnIndex": 1,
            # Textract occasionally clips the visible word line at the bottom
            # of a low-confidence child CELL.
            "Geometry": _geometry(0.10, 0.58, 0.40, 0.05),
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": ["noise", "leveling", "frame"],
                }
            ],
        },
        {
            "Id": "merged",
            "BlockType": "MERGED_CELL",
            "Confidence": 29.0,
            "RowIndex": 1,
            "ColumnIndex": 1,
            "RowSpan": 2,
            "ColumnSpan": 1,
            "Geometry": _geometry(0.10, 0.25, 0.40, 0.45),
            "Relationships": [
                {
                    "Type": "CHILD",
                    "Ids": ["child-guide", "child-frame"],
                }
            ],
        },
    ]
    by_id = {block["Id"]: block for block in raw_blocks}

    blocks = extractor._build_blocks(
        by_id,
        {},
        set(),
        page_index=0,
        page_width=1000,
        page_height=1000,
        page_image=None,
        allow_figures=False,
    )

    texts = {block.source_text for block in blocks}
    assert "Guide rail" in texts
    assert "Leveling frame" in texts
    assert all("IT" not in text for text in texts)
    assert all(not ("Guide rail" in text and "Leveling frame" in text) for text in texts)
    frame = next(block for block in blocks if block.source_text == "Leveling frame")
    assert frame.word_boxes == [(200, 600, 100, 40), (310, 600, 80, 40)]
    assert frame.bbox[:3] == (100, 580, 500)
    assert frame.bbox[3] >= 640
    assert frame.parent_bbox == (100, 580, 500, 630)


def test_broad_composite_label_is_removed_when_child_labels_exist(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    broad = _core.BlockRegion(
        page_index=0,
        block_id="broad",
        block_type="TEXT",
        bbox=(100, 100, 500, 400),
        source_text="Traction machine\nTraction machine rack",
        parent_block_id="figure",
    )
    first = _core.BlockRegion(
        page_index=0,
        block_id="first",
        block_type="TEXT",
        bbox=(180, 130, 400, 190),
        source_text="Traction machine",
        parent_block_id="figure",
    )
    second = _core.BlockRegion(
        page_index=0,
        block_id="second",
        block_type="TEXT",
        bbox=(160, 280, 430, 340),
        source_text="Traction machine rack",
        parent_block_id="figure",
    )

    kept = extractor._deduplicate_blocks([broad, first, second])

    assert {block.block_id for block in kept} == {"first", "second"}


def test_composite_parent_is_kept_when_children_leave_real_text_uncovered(tmp_path):
    extractor = _core.TextractLayoutExtractor(
        object(),
        config=_core.PdfProcessingConfig(),
        log_path=tmp_path / "missing-textract-cache.jsonl",
    )
    broad = _core.BlockRegion(
        page_index=0,
        block_id="broad",
        block_type="TEXT",
        bbox=(100, 100, 500, 400),
        source_text="alpha beta gamma delta zeta",
        parent_block_id="figure",
    )
    first = _core.BlockRegion(
        page_index=0,
        block_id="first",
        block_type="TEXT",
        bbox=(150, 130, 430, 190),
        source_text="alpha beta gamma",
        parent_block_id="figure",
    )
    second = _core.BlockRegion(
        page_index=0,
        block_id="second",
        block_type="TEXT",
        bbox=(180, 220, 300, 270),
        source_text="delta",
        parent_block_id="figure",
    )

    kept = extractor._deduplicate_blocks([broad, first, second])

    assert {block.block_id for block in kept} == {"broad", "first", "second"}
