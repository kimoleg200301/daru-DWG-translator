"""Regressions for native manuals with diagrams inside large table cells."""

from types import SimpleNamespace

import fitz
import pytest

from daru.pdf import native


def _word_block(text, x0, x1, y, *, flags=0, direction=(1.0, 0.0)):
    span = {"text": text, "bbox": (x0, y, x1, y + 10), "origin": (x0, y + 8),
            "size": 10.0, "font": "Arial", "flags": flags, "color": 0}
    return {"type": 0, "lines": [{"dir": direction, "spans": [span]}]}


def _justified_fault_blocks():
    return [
        _word_block("No. 28 fault will be", 25, 135, 30),
        _word_block("reported", 25, 62, 47),
        _word_block("when", 87, 109, 47),
        _word_block("the", 125, 135, 47),
        _word_block("difference", 25, 73, 64),
        _word_block("between", 102, 135, 64),
        _word_block("encoder", 25, 64, 81),
        _word_block("absolute", 102, 135, 81),
        _word_block("position and calculated", 25, 135, 98),
        _word_block("position exceeds this set", 25, 135, 115),
        _word_block("value.", 25, 50, 132),
    ]


def _extract_blocks(monkeypatch, blocks, cells, *, drawings=()):
    page = SimpleNamespace(rect=fitz.Rect(0, 0, 500, 500),
                           get_drawings=lambda: list(drawings), get_image_info=lambda: [],
                           parent=None, get_xobjects=lambda: [])
    monkeypatch.setattr(native, "_native_text_blocks", lambda _: blocks)
    monkeypatch.setattr(native, "_table_cell_rects", lambda *_: cells)
    return native._extract_page(page, 0, "en").units


def test_justified_cell_sentence_recovers_words_across_pdf_blocks(monkeypatch):
    blocks = _justified_fault_blocks()
    neighbour = _word_block("Encoder fault", 165, 240, 81)
    cells = [fitz.Rect(20, 20, 140, 145), fitz.Rect(160, 20, 280, 145)]
    original_neighbour = native._units_from_block(
        neighbour, page_index=0, source_lang="en", median_font_size=10, cell_rects=cells,
    )[0]

    units = _extract_blocks(monkeypatch, blocks + [neighbour], cells)

    assert len(units) == 2
    sentence = units[0]
    assert sentence.source_text == (
        "No. 28 fault will be reported when the difference between encoder absolute "
        "position and calculated position exceeds this set value."
    )
    assert len(sentence.lines) == 7
    assert [span.bbox for span in sentence.style_runs] == [block["lines"][0]["spans"][0]["bbox"] for block in blocks]
    assert [span.origin for span in sentence.style_runs] == [block["lines"][0]["spans"][0]["origin"] for block in blocks]
    assert sentence.lines[1].origin == (25, 55)
    assert units[1].stable_id == original_neighbour.stable_id
    assert units[1].source_text == "Encoder fault"


def test_justified_recovery_keeps_heading_and_list_items_separate(monkeypatch):
    blocks = [_word_block("Fault diagnosis", 25, 135, 13, flags=16)] + _justified_fault_blocks()
    blocks.extend([_word_block("1. Check supply.", 25, 135, 149),
                   _word_block("2. Reset the drive.", 25, 135, 166)])

    units = _extract_blocks(monkeypatch, blocks, [fitz.Rect(20, 10, 140, 185)])

    assert len(units) == 4
    assert units[0].source_text == "Fault diagnosis"
    assert units[0].bold
    assert units[1].source_text.startswith("No. 28 fault will be reported when the")
    assert [unit.source_text for unit in units[2:]] == ["1. Check supply.", "2. Reset the drive."]


@pytest.mark.parametrize("cell", [(20, 20, 450, 200), (20, 20, 140, 400)])
def test_justified_recovery_does_not_collapse_large_diagram_cells(monkeypatch, cell):
    blocks = _justified_fault_blocks()

    units = _extract_blocks(monkeypatch, blocks, [fitz.Rect(cell)])

    assert len(units) == len(blocks)
    assert "the" in [unit.source_text for unit in units]


def test_justified_recovery_does_not_join_text_across_artwork(monkeypatch):
    blocks = _justified_fault_blocks()
    drawing = {"rect": fitz.Rect(25, 58, 135, 61), "items": [], "fill": (0, 0, 0)}

    units = _extract_blocks(monkeypatch, blocks, [fitz.Rect(20, 20, 140, 145)], drawings=[drawing])

    assert len(units) == len(blocks)


def test_neighbouring_multiword_captions_in_narrow_cell_are_not_justification(monkeypatch):
    blocks = [_word_block("Left guide", 25, 66, 30),
              _word_block("Right guide", 88, 135, 30),
              _word_block("Lower guide assembly", 25, 135, 47)]

    units = _extract_blocks(monkeypatch, blocks, [fitz.Rect(20, 20, 140, 100)])

    assert [unit.source_text for unit in units] == ["Left guide", "Right guide", "Lower guide assembly"]


def test_justified_recovery_does_not_enclose_unrepresented_numeric_text(monkeypatch):
    blocks = _justified_fault_blocks()
    blocks.append(_word_block("300", 75, 85, 47))

    units = _extract_blocks(monkeypatch, blocks, [fitz.Rect(20, 20, 140, 145)])

    assert len(units) == len(blocks) - 1
    assert "the" in [unit.source_text for unit in units]
    assert all(len(unit.lines) == 1 for unit in units)


def test_diagonal_page_watermark_does_not_prevent_cell_recovery(monkeypatch):
    blocks = _justified_fault_blocks()
    blocks.append(_word_block("品牌", -20, 450, 20, direction=(0.707, -0.707)))
    blocks[-1]["lines"][0]["spans"][0]["bbox"] = (-20, 20, 450, 300)

    units = _extract_blocks(monkeypatch, blocks, [fitz.Rect(20, 20, 140, 145)])

    assert len(units) == 1
    assert units[0].source_text.startswith("No. 28 fault will be reported when the")


def test_justified_recovery_leaves_rotated_labels_separate(monkeypatch):
    blocks = _justified_fault_blocks()
    for block in blocks:
        block["lines"][0]["dir"] = (0.0, -1.0)

    units = _extract_blocks(monkeypatch, blocks, [fitz.Rect(20, 20, 140, 145)])

    assert len(units) == len(blocks)
    assert all(unit.rotation == 90 for unit in units)


def test_padded_header_uses_visible_glyph_coordinates():
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 60), " " * 34 + "the elevator", fontname="hebo", fontsize=14)
        expected = page.search_for("the elevator")[0]
        unit = native._extract_page(page, 0, "en").units[0]
        assert unit.source_text == "the elevator"
        assert unit.bbox[0] == pytest.approx(expected.x0, abs=0.05)
        assert unit.lines[0].origin[0] == pytest.approx(expected.x0, abs=0.05)
        assert unit.bold


def test_table_paragraphs_keep_positions_around_diagram():
    with fitz.open() as document:
        page = document.new_page(width=440, height=500)
        for x in (30, 150, 410):
            page.draw_line((x, 40), (x, 470))
        for y in (40, 100, 470):
            page.draw_line((30, y), (410, y))
        for point, text in [((40, 65), "Operation"), ((160, 65), "Remarks"),
                            ((40, 280), "Guide alignment"), ((160, 130), "Inspect the rail before mounting."),
                            ((160, 400), "Tighten the bolts after alignment.")]:
            page.insert_text(point, text, fontsize=10)
        diagram = fitz.Rect(175, 180, 360, 350)
        page.draw_rect(diagram)
        page.draw_line((175, 180), (360, 350))
        model = native._extract_page(page, 0, "en")
        top = next(unit for unit in model.units if unit.source_text.startswith("Inspect"))
        bottom = next(unit for unit in model.units if unit.source_text.startswith("Tighten"))
        assert top.role == bottom.role == "CELL"
        assert top.bbox[3] < diagram.y0 < diagram.y1 < bottom.bbox[1]
        label = next(unit for unit in model.units if unit.source_text == "Guide alignment")
        assert label.bbox[1] > 250
        assert top.alignment == "left"
        assert not top.bold
        top.translated_text = "Проверьте направляющую до монтажа."
        bottom.translated_text = "После выверки затяните болты."
        native._render_native_pages(document, [model], None, lambda _: None)
        assert page.search_for("Проверьте")[0].y1 < diagram.y0
        assert page.search_for("После")[0].y0 > diagram.y1
        assert "Guide alignment" in page.get_text()


def test_inferred_cell_borders_do_not_slice_numbers_or_words(monkeypatch):
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((45, 60), "1.41 no", fontsize=11)
        page.insert_text((45, 90), "MLB16.1", fontsize=11)
        fake = SimpleNamespace(tables=[SimpleNamespace(row_count=2, col_count=2, rows=[
            SimpleNamespace(cells=[(30, 40, 52, 70), (52, 40, 110, 70)]),
            SimpleNamespace(cells=[(30, 70, 65, 100), (65, 70, 110, 100)]),
        ])])
        monkeypatch.setattr(fitz.Page, "find_tables", lambda _: fake)
        units = native._extract_page(page, 0, "en").units
        assert [unit.source_text for unit in units] == ["1.41 no", "MLB16.1"]
        assert all(unit.layout_bounds is None for unit in units)


def test_headings_and_numbered_steps_do_not_merge_into_prose():
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 50), "1.2 Safety precautions", fontsize=10, fontname="hebo")
        page.insert_text((40, 62), "Keep the working area clean.", fontsize=10)
        page.insert_text((40, 74), "1.3 Preparation of the site", fontsize=10, fontname="hebo")
        page.insert_text((40, 86), "1.3.1 Consultation with clients", fontsize=10, fontname="hebo")
        units = native._extract_page(page, 0, "en").units
        assert len(units) == 4
        assert [unit.bold for unit in units] == [True, False, True, True]


def test_wrapped_numbered_paragraph_reflows_without_hard_line_breaks():
    assert native._normalize_block_text([
        "(1) Disconnect the supply before", "working on the equipment.",
        "(2) Check the isolation.",
    ]) == "(1) Disconnect the supply before working on the equipment.\n(2) Check the isolation."


def test_toc_numbers_stay_in_the_original_column():
    with fitz.open() as document:
        page = document.new_page(width=450, height=250)
        page.insert_text((30, 50), "Safety procedures ........................................ 12", fontsize=10)
        model = native._extract_page(page, 0, "en")
        unit = model.units[0]
        assert unit.role == "TOC"
        right = unit.bbox[2]
        unit.translated_text = "Требования безопасности ........................................ 12"
        native._render_native_pages(document, [model], None, lambda _: None)
        number = page.search_for("12")[0]
        assert number.x1 == pytest.approx(right, abs=0.1)
        assert "Требования безопасности" in page.get_text().replace("\u00a0", " ")


def test_factory_sentence_is_translated_but_contact_label_is_protected():
    text = "Note: The buffer is not refueled at the factory."
    assert native._protect_text(text).value == text
    address = "Factory: 12 Main Street, London"
    assert native._protect_text(address).value == "[[DARU_P_A]]"


def test_explicit_regular_font_keeps_bold_and_italic_variants():
    paths = native._font_candidate_paths("times.ttf", "serif", True, True)
    assert paths[0].name == "timesbi.ttf"


def test_table_outline_does_not_block_free_interior_but_edges_do():
    with fitz.open() as document:
        page = document.new_page(width=400, height=300)
        page.draw_rect(fitz.Rect(20, 20, 350, 250))
        occupied = native._page_nontext_occupied(page)
        source = fitz.Rect(40, 50, 100, 70)
        assert native._expansion_is_free(fitz.Rect(40, 50, 180, 100), source, occupied)
        assert not native._expansion_is_free(fitz.Rect(40, 50, 360, 100), source, occupied)


def test_single_line_translation_keeps_font_size_and_baseline_in_free_space():
    with fitz.open() as document:
        page = document.new_page(width=500, height=200)
        page.draw_rect(fitz.Rect(20, 35, 480, 65), fill=(0.5, 0.7, 0.9))
        page.insert_text((30, 55), "Tools", fontsize=11, fontname="hebo")
        text_rect = page.search_for("Tools")[0]
        page.draw_rect(
            fitz.Rect(30, text_rect.y0 - 0.2, 480, text_rect.y1 + 0.2),
            color=None, fill=(0.5, 0.7, 0.9), overlay=False,
        )
        model = native._extract_page(page, 0, "en")
        unit = model.units[0]
        unit.translated_text = "Инструменты"
        native._render_native_pages(document, [model], None, lambda _: None)
        assert unit.font_scale == 1.0
        spans = [span for block in page.get_text("dict")["blocks"] if block["type"] == 0
                 for line in block["lines"] for span in line["spans"]]
        assert spans[0]["size"] == pytest.approx(11, abs=0.05)
        assert spans[0]["origin"][1] == pytest.approx(55, abs=0.05)


def test_clipped_callout_lines_are_fitted_together_in_their_frame():
    with fitz.open() as document:
        page = document.new_page(width=300, height=250)
        frame = fitz.Rect(40, 40, 125, 57)
        page.draw_rect(frame, color=(0, 0, 0), fill=(1, 1, 1))
        page.insert_text((50, 50), "Expansion", fontsize=9)
        page.insert_text((65, 62), "bolts", fontsize=9)
        model = native._extract_page(page, 0, "en")
        assert len(model.units) == 1
        unit = model.units[0]
        assert unit.source_text == "Expansion bolts"
        assert unit.framed_label
        unit.translated_text = "Распорные анкеры"
        native._render_native_pages(document, [model], None, lambda _: None)
        text = page.get_text().replace("\u00a0", " ").replace("\n", " ")
        assert "Распорные анкеры" in text
        for word in page.get_text("words"):
            assert (frame + (-0.5, -0.5, 0.5, 0.5)).contains(fitz.Rect(word[:4]))


def test_neighbouring_captions_in_a_large_cell_remain_separate():
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 60), "Left guide", fontsize=10)
        page.insert_text((115, 60), "Right guide", fontsize=10)
        blocks = native._native_text_blocks(page)
        block = {"lines": [line for b in blocks for line in b["lines"]]}
        units = native._units_from_block(block, page_index=0, source_lang="en", median_font_size=10,
                                        cell_rects=[fitz.Rect(20, 20, 400, 300)])
        assert [unit.source_text for unit in units] == ["Left guide", "Right guide"]


def test_paragraph_preserves_narrow_lines_beside_a_figure():
    with fitz.open() as document:
        page = document.new_page(width=500, height=300)
        figure = fitz.Rect(240, 40, 360, 62)
        page.draw_rect(figure)
        page.insert_text((40, 60), "Check the rail joint", fontsize=10)
        page.insert_text((40, 72), "Then inspect the entire surface and record all the measured dimensions.", fontsize=10)
        page.insert_text((40, 84), "Record the results in the maintenance log.", fontsize=10)
        model = native._extract_page(page, 0, "en")
        assert len(model.units) == 2
        first, second = model.units
        assert first.source_text == "Check the rail joint"
        assert first.layout_bounds[2] < figure.x0
        assert second.bbox[2] > figure.x1 - 20
        first.translated_text = "Проверьте стык направляющих"
        second.translated_text = "Проверьте поверхность, измерьте размеры и запишите результаты в журнал."
        native._render_native_pages(document, [model], None, lambda _: None)
        assert page.search_for("направляющих")[0].x1 < figure.x0


def test_direct_text_paths_defer_to_html_when_the_font_lacks_glyphs(monkeypatch):
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((40, 60), "Safety ................................ 12", fontsize=10)
        unit = native._extract_page(page, 0, "en").units[0]
        unit.translated_text = "Безопасность ................................ 12"
        fonts = native._build_font_bundle(native._FontResource(
            family="test", archive=None, font_face_css="", font=fitz.Font("helv"),
        ))
        monkeypatch.setattr(native, "_font_resource_supports_text", lambda *_: False)
        assert not native._insert_toc_unit(page, unit, fonts)
        assert not native._insert_single_line_unit(page, unit, fonts, [fitz.Rect(unit.bbox)])
        assert unit.rendered_bbox is None
