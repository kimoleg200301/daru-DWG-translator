from __future__ import annotations

import json
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from daru.pdf import _core
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

    assert payload["version"] == 3
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
