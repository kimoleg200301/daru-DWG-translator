from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image

from daru.pdf import _core, raster
from daru.pdf.editable import (
    EditableTextLayer,
    PageTransform,
    region_to_text_unit,
    supports_editable_rotation,
)


def _page_image(
    *,
    size: tuple[int, int] = (400, 600),
    points: tuple[float, float] = (200.0, 300.0),
) -> _core.PageImage:
    return _core.PageImage(
        page_index=0,
        image=Image.new("RGB", size, "white"),
        width=size[0],
        height=size[1],
        dpi=144,
        physical_size_points=points,
    )


def _region(
    region_id: int,
    bbox: tuple[int, int, int, int],
    text: str,
    *,
    content_bbox: tuple[int, int, int, int] | None = None,
    orientation: float = 0.0,
    vertical: bool = False,
) -> _core.RegionInfo:
    return _core.RegionInfo(
        page_index=0,
        region_id=region_id,
        bbox=bbox,
        text_orientation_deg=orientation,
        text_color=(0, 0, 0),
        background_color=(255, 255, 255),
        source_text=f"SOURCE-{region_id}",
        translated_text=text,
        mask=np.zeros((bbox[3], bbox[2]), dtype=np.uint8),
        font_size_estimate=24,
        source_font_size_estimate=24,
        stable_id=f"region-{region_id}",
        content_bbox=content_bbox,
        is_vertical=vertical,
    )


def _export_editable(
    output: Path,
    page: _core.PageImage,
    regions: list[_core.RegionInfo],
) -> EditableTextLayer:
    layer = EditableTextLayer()

    def overlay(_index: int, pdf_page: fitz.Page) -> None:
        layer.insert_page(pdf_page, page, regions)

    _core.Exporter().to_pdf(
        [page.image],
        output,
        dpi=144,
        input_dpi=144,
        page_sizes_points=[page.physical_size_points],
        page_overlay=overlay,
    )
    return layer


def _normalized_text(page: fitz.Page) -> str:
    return " ".join(page.get_text().replace("\xa0", " ").split())


def test_pdf_config_applies_known_options_without_coupling_output_and_ocr_dpi() -> None:
    config = _core.PdfProcessingConfig.from_options(
        {
            "dpi": 200,
            "render_dpi": 400,
            "editable_text": False,
            "unrelated_gui_option": "ignored",
        }
    )

    assert config.dpi == 200
    assert config.render_dpi == 400
    assert config.editable_text is False


def test_page_105_cleanup_box_keeps_visible_content_anchor() -> None:
    page = _page_image(size=(3306, 4673), points=(595.0, 841.0))
    region = _region(
        1,
        (0, 617, 3306, 185),
        "2. При скорости кабины 2,5 м/с",
        content_bbox=(372, 12, 2537, 161),
    )
    page_rect = fitz.Rect(0, 0, 595, 841)

    transform = PageTransform.from_page(page, page_rect)
    unit = region_to_text_unit(page, region, page_rect)

    assert transform.scale_x == pytest.approx(595 / 3306)
    assert transform.scale_y == pytest.approx(841 / 4673)
    assert unit.bbox[0] == pytest.approx(66.95, abs=0.1)
    assert unit.bbox[1] == pytest.approx(113.20, abs=0.1)
    assert unit.bbox[2] == pytest.approx(595.0, abs=0.1)


def test_table_cell_keeps_vertical_content_anchor() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 50, 300, 120),
        "Ячейка",
        content_bbox=(20, 70, 200, 20),
    )
    region.is_table_cell = True
    region.alignment = "center"

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.bbox == pytest.approx((20.0, 60.0, 170.0, 85.0))


def test_left_aligned_table_cell_keeps_horizontal_content_anchor() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 50, 300, 120),
        "Болт M12×45",
        content_bbox=(60, 20, 160, 30),
    )
    region.is_table_cell = True
    region.alignment = "left"
    region.block_type = "CELL"

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.bbox == pytest.approx((50.0, 35.0, 170.0, 85.0))


def test_table_description_near_marker_gets_a_small_safety_inset() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 50, 300, 120),
        "Пружинная шайба 10",
        content_bbox=(10, 20, 260, 30),
    )
    region.is_table_cell = True
    region.alignment = "left"
    region.block_type = "CELL"

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.bbox == pytest.approx((28.0, 35.0, 170.0, 85.0))


def test_right_aligned_table_cell_does_not_expand_over_left_neighbor() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 50, 300, 120),
        "Пружинная шайба 10",
        content_bbox=(60, 20, 160, 30),
    )
    region.is_table_cell = True
    region.alignment = "right"

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.bbox == pytest.approx((20.0, 35.0, 130.0, 85.0))


def test_sparse_table_callout_uses_content_frame_not_broad_cleanup_cell() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 50, 300, 120),
        "Рамка устройства точной остановки",
        content_bbox=(100, 60, 80, 20),
    )
    region.is_table_cell = True
    region.alignment = "left"

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.bbox == pytest.approx((70.0, 48.4, 117.2, 67.4), abs=0.1)


def test_realistic_sparse_callouts_stay_inside_frames_at_readable_scale() -> None:
    page_image = _page_image(size=(826, 1168), points=(595.0, 841.0))
    guide = _region(
        1,
        (153, 686, 184, 20),
        "Направляющая",
        content_bbox=(92, 3, 47, 10),
    )
    guide.font_size_estimate = guide.source_font_size_estimate = 10
    leveling = _region(
        2,
        (153, 737, 184, 41),
        "Рамка устройства точной остановки",
        content_bbox=(12, 26, 73, 12),
    )
    leveling.font_size_estimate = leveling.source_font_size_estimate = 12
    for region in (guide, leveling):
        region.is_table_cell = True
        region.alignment = "left"
        region.block_type = "CELL"

    document = fitz.open()
    pdf_page = document.new_page(width=595, height=841)
    layer = EditableTextLayer()
    layer.insert_page(pdf_page, page_image, [guide, leveling])

    for region in (guide, leveling):
        unit = region_to_text_unit(page_image, region, pdf_page.rect)
        rendered = fitz.Rect(region.rendered_bbox_points)
        frame = fitz.Rect(unit.bbox)
        assert region.vector_font_scale >= 0.65
        assert rendered.x0 >= frame.x0 - 0.5
        assert rendered.y0 >= frame.y0 - 0.5
        assert rendered.x1 <= frame.x1 + 0.5
        assert rendered.y1 <= frame.y1 + 0.5


def test_vertical_table_label_keeps_full_rotation_axis() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 50, 300, 120),
        "При монтаже",
        content_bbox=(100, 60, 40, 20),
        orientation=90,
        vertical=True,
    )
    region.is_table_cell = True
    region.alignment = "center"
    region.block_type = "CELL"

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.bbox == pytest.approx((20.0, 55.0, 170.0, 85.0))


def test_map_label_uses_nonexpanding_cell_layout() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 50, 160, 30),
        "Фотоэлектрический выключатель",
        content_bbox=(5, 3, 150, 24),
    )
    region.semantic_type = "map_label"

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.role == "CELL"
    assert unit.bbox == pytest.approx((22.5, 19.0, 100.0, 43.0))


def test_arbitrary_rotation_is_not_silently_snapped_to_a_quarter_turn() -> None:
    page = _page_image()
    region = _region(1, (40, 50, 300, 120), "Наклонная подпись", orientation=60)

    assert supports_editable_rotation(region) is False
    with pytest.raises(ValueError, match="quarter-turn rotation"):
        region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))


def test_pdf_rasterization_uses_cropbox_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_convert(*_args, **kwargs):
        received.update(kwargs)
        return [Image.new("RGB", (100, 150), "white")]

    monkeypatch.setattr(raster, "convert_from_path", fake_convert)
    monkeypatch.setattr(raster, "_detect_poppler_path", lambda: None)

    images = _core.pdf_to_images(Path("cropped.pdf"), 200, "png", lambda _m: None)

    assert len(images) == 1
    assert received["use_cropbox"] is True


def test_vector_fit_starts_from_preferred_size_not_legacy_preview_size() -> None:
    page = _page_image()
    region = _region(1, (40, 60, 320, 90), "Технический текст")
    region.font_size_estimate = 24
    region.source_font_size_estimate = 24
    region.rendered_font_size = 8

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.font_size == pytest.approx(12.0)


def test_body_ocr_line_wraps_reflow_but_list_boundaries_remain() -> None:
    page = _page_image()
    region = _region(
        1,
        (40, 60, 320, 160),
        "Первая строка\nпродолжение абзаца\n(1) Новый пункт\nа. Деталь",
    )

    unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))

    assert unit.translated_text == (
        "Первая строка продолжение абзаца\n(1) Новый пункт\nа. Деталь"
    )

    region.is_table_cell = True
    cell_unit = region_to_text_unit(page, region, fitz.Rect(0, 0, 200, 300))
    assert cell_unit.translated_text == region.translated_text


def test_extreme_vector_shrink_is_explicitly_reported() -> None:
    page_image = _page_image()
    region = _region(
        1,
        (20, 20, 45, 24),
        "Очень длинная техническая подпись, которая не помещается в узкую область",
    )
    region.font_size_estimate = 30
    document = fitz.open()
    try:
        pdf_page = document.new_page(width=200, height=300)
        warnings = EditableTextLayer().insert_page(pdf_page, page_image, [region])
    finally:
        document.close()

    assert warnings
    assert warnings[-1]["reason"] == "font_scaled_below_readable_minimum"
    assert warnings[-1]["scale"] < 0.5
    assert region.layout_warning == "font_scaled_below_readable_minimum"


def test_short_english_heading_expands_safely_for_longer_russian_word() -> None:
    page_image = _page_image(size=(3306, 4673), points=(595.0, 841.0))
    heading = _region(1, (375, 1290, 141, 50), "Инструменты")
    heading.source_text = "Tools"
    heading.font_size_estimate = 50
    heading.source_font_size_estimate = 50
    heading.semantic_type = "heading"
    heading.is_bold = True
    neighbor = _region(2, (800, 1290, 180, 50), "Соседняя подпись")
    neighbor.font_size_estimate = 50
    neighbor.source_font_size_estimate = 50
    document = fitz.open()
    try:
        pdf_page = document.new_page(width=595, height=841)
        EditableTextLayer().insert_page(
            pdf_page,
            page_image,
            [heading, neighbor],
        )
    finally:
        document.close()

    assert heading.vector_font_scale >= 0.82
    assert heading.rendered_bbox_points is not None
    neighbor_x0 = 800 * 595 / 3306
    assert heading.rendered_bbox_points[2] <= neighbor_x0


def test_vector_preview_uses_the_same_editable_overlay_geometry() -> None:
    page = _page_image()
    region = _region(1, (40, 60, 320, 90), "Предварительный просмотр")

    preview = np.asarray(EditableTextLayer().render_preview(page.image, page, [region]))

    assert np.any(preview[60:150, 40:360] < 180)
    outside = preview.copy()
    outside[60:150, 40:360] = 255
    assert np.all(outside >= 250)


def test_scanned_vector_pdf_is_searchable_and_background_has_no_baked_text(
    tmp_path: Path,
) -> None:
    output = tmp_path / "editable.pdf"
    page_image = _page_image()
    translated = "Редактируемый технический текст"
    region = _region(1, (40, 60, 320, 90), translated, content_bbox=(0, 0, 310, 70))

    layer = _export_editable(output, page_image, [region])

    document = fitz.open(output)
    try:
        page = document[0]
        extracted = _normalized_text(page)
        assert translated in extracted
        assert "SOURCE-1" not in extracted
        assert len(page.search_for("Редактируемый")) == 1
        assert len(page.get_images(full=True)) == 1
        fonts = page.get_fonts(full=True)
        assert any(font[2] == "Type0" and font[5] == "Identity-H" for font in fonts)

        image_xref = page.get_images(full=True)[0][0]
        background = fitz.Pixmap(document, image_xref)
        background_samples = np.frombuffer(background.samples, dtype=np.uint8)
        assert float(background_samples.mean()) > 250.0

        rendered = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        rendered_samples = np.frombuffer(rendered.samples, dtype=np.uint8)
        assert int(rendered_samples.min()) < 100
    finally:
        document.close()

    assert layer.inserted_regions == 1
    assert region.rendered_bbox_points is not None
    assert region.vector_font_scale > 0


def test_same_translation_in_two_regions_produces_two_search_hits(tmp_path: Path) -> None:
    output = tmp_path / "duplicates.pdf"
    page_image = _page_image()
    translated = "Метка"
    regions = [
        _region(1, (35, 60, 300, 70), translated),
        _region(2, (35, 220, 300, 70), translated),
    ]

    _export_editable(output, page_image, regions)

    document = fitz.open(output)
    try:
        assert len(document[0].search_for(translated)) == 2
    finally:
        document.close()


def test_scanned_vector_export_reuses_font_resources_across_pages(
    tmp_path: Path,
) -> None:
    output = tmp_path / "font-reuse.pdf"
    page_images: list[_core.PageImage] = []
    page_regions: list[list[_core.RegionInfo]] = []
    for page_index in range(3):
        page_image = _page_image()
        page_image.page_index = page_index
        region = _region(
            page_index + 1,
            (40, 60, 320, 90),
            f"Редактируемая страница {page_index + 1}",
        )
        region.page_index = page_index
        page_images.append(page_image)
        page_regions.append([region])
    layer = EditableTextLayer()

    def overlay(index: int, pdf_page: fitz.Page) -> None:
        layer.insert_page(pdf_page, page_images[index], page_regions[index])

    _core.Exporter().to_pdf(
        [page.image for page in page_images],
        output,
        dpi=144,
        input_dpi=144,
        page_sizes_points=[page.physical_size_points for page in page_images],
        page_overlay=overlay,
    )

    document = fitz.open(output)
    try:
        font_xrefs = {
            font[0]
            for page in document
            for font in page.get_fonts(full=True)
            if font[2] == "Type0"
        }
        assert len(font_xrefs) == 1
        assert all("Редактируемая страница" in _normalized_text(page) for page in document)
    finally:
        document.close()


def test_quarter_turn_scanned_text_is_searchable_and_vertical(tmp_path: Path) -> None:
    output = tmp_path / "vertical.pdf"
    page_image = _page_image()
    translated = "Вертикальная подпись"
    region = _region(
        1,
        (80, 80, 80, 360),
        translated,
        orientation=-90,
        vertical=True,
    )

    _export_editable(output, page_image, [region])

    document = fitz.open(output)
    try:
        page = document[0]
        assert translated in _normalized_text(page)
        text_block = next(
            block for block in page.get_text("dict")["blocks"] if block.get("type") == 0
        )
        lines = text_block["lines"]
        direction = lines[0]["dir"]
        assert abs(direction[0]) < 0.05
        assert abs(direction[1]) > 0.95
        hit = page.search_for(translated)[0]
        assert hit.height > hit.width
    finally:
        document.close()


def test_vector_overlay_failure_never_leaves_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "partial.pdf"
    images = [Image.new("RGB", (200, 300), "white") for _ in range(2)]

    def fail_on_second(index: int, _page: fitz.Page) -> None:
        if index == 1:
            raise RuntimeError("forced overlay failure")

    with pytest.raises(RuntimeError, match="forced overlay failure"):
        _core.Exporter().to_pdf(
            images,
            output,
            dpi=100,
            input_dpi=100,
            page_overlay=fail_on_second,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp.pdf"))


def test_identity_safe_cache_does_not_apply_stale_render_id_translation() -> None:
    block = _core.BlockRegion(
        page_index=3,
        block_id="new-page-105-zone",
        block_type="TEXT",
        bbox=(372, 629, 2909, 790),
        source_text="Newly split paragraph",
        render_id=3,
        stable_id="new-stable-id",
    )

    _core._apply_translation_caches(
        [block],
        stable_map={"different-stable-id": "safe cached translation"},
        legacy_render_map={(3, 3): "stale whole-page translation"},
        canonical_map={},
    )

    assert block.translated_text is None
