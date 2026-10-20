"""Regression tests using stored ezdxf entities, including block ownership."""

import ezdxf
import pytest

from daru.dxf.entities import (
    TextReplacer,
    get_dim_override_text,
    replace_all_exact,
    translate_text_keep_dim_and_mtext_controls,
)
from daru.dxf.extractor import extract_text_counts_and_types
from daru.dxf.pipeline import apply_translations


def test_layouts_and_block_definitions_are_counted_once():
    doc = ezdxf.new()
    doc.modelspace().add_text("Model label")
    doc.layouts.new("Sheet").add_text("Sheet label")
    block = doc.blocks.new("Valve details")
    block.add_text("Block label")
    for position in ((0, 0), (10, 10)):
        insertion = doc.modelspace().add_blockref(block.name, position)
        insertion.add_attrib("NAME", "Valve attribute")

    counts, types = extract_text_counts_and_types(doc)

    assert counts == {
        "Model label": 1, "Sheet label": 1, "Block label": 1, "Valve attribute": 2,
    }
    assert types["Block label"] == "TEXT"
    assert types["Valve attribute"] == "ATTRIB"


def test_translation_visits_layouts_blocks_and_attached_attributes_once(tmp_path):
    doc = ezdxf.new()
    model_label = doc.modelspace().add_text("A")
    paper_label = doc.layouts.new("Sheet").add_text("A")
    block = doc.blocks.new("Valve")
    block_label = block.add_text("A")
    nested_block = doc.blocks.new("Assembly")
    nested = nested_block.add_blockref("Valve", (0, 0)).add_attrib("NAME", "A")
    attribute = doc.modelspace().add_blockref("Assembly", (0, 0)).add_attrib("NAME", "A")

    apply_translations(doc, [("B", "C"), ("A", "B")])

    for entity in (model_label, paper_label, block_label, nested, attribute):
        assert entity.dxf.text == "B"
    output = tmp_path / "translated.dxf"
    doc.saveas(output)
    reloaded = ezdxf.readfile(output)
    counts, _ = extract_text_counts_and_types(reloaded)
    assert counts == {"B": 5}


def test_real_dimension_override_is_extracted_and_translated():
    doc = ezdxf.new()
    dimension = doc.modelspace().add_linear_dim(
        base=(0, 3), p1=(0, 0), p2=(4, 0), text="Length <> mm",
    ).dimension
    assert get_dim_override_text(dimension) == "Length <> mm"
    counts, types = extract_text_counts_and_types(doc)
    assert counts == {"Length <> mm": 1}
    assert types == {"Length <> mm": "DIMENSION"}

    apply_translations(doc, [("Length <> mm", "Длина <> мм")])

    assert dimension.dxf.text == "Длина <> мм"


def test_real_multileader_uses_shared_text_accessors():
    doc = ezdxf.new()
    builder = doc.modelspace().add_multileader_mtext()
    builder.set_content("Valve label")

    counts, types = extract_text_counts_and_types(doc)
    assert counts == {"Valve label": 1}
    assert types == {"Valve label": "MULTILEADER"}

    apply_translations(doc, [("Valve label", "Подпись клапана")])

    assert builder.multileader.get_mtext_content() == "Подпись клапана"


@pytest.mark.parametrize("pairs, expected", [
    ([], r"{\C1;Hello}\PWorld"),
    ([("Hello", "Привет"), ("World", "Мир")], r"{\C1;Привет}\PМир"),
    ([(r"{\C1;Hello}\PWorld", r"{\C1;Привет}\PМир")], r"{\C1;Привет}\PМир"),
    ([(r"{\C1;Hello}\PWorld", "Damaged formatting")], r"{\C1;Hello}\PWorld"),
])
def test_mtext_formatting_survives_application_and_save(tmp_path, pairs, expected):
    doc = ezdxf.new()
    mtext = doc.modelspace().add_mtext(r"{\C1;Hello}\PWorld")
    counts, _ = extract_text_counts_and_types(doc)
    assert counts == {r"{\C1;Hello}\PWorld": 1}

    apply_translations(doc, pairs)

    assert mtext.text == expected
    output = tmp_path / "formatted.dxf"
    doc.saveas(output)
    assert ezdxf.readfile(output).modelspace().query("MTEXT")[0].text == expected


def test_replacements_do_not_translate_previously_inserted_text():
    assert replace_all_exact("A B", [("A", "B"), ("B", "C")]) == "B C"


def test_formatted_mtext_matches_stripped_extraction_without_losing_spaces():
    doc = ezdxf.new()
    mtext = doc.modelspace().add_mtext(r"  {\C1;Hello}\PWorld  ")
    counts, _ = extract_text_counts_and_types(doc)
    source = next(iter(counts))

    apply_translations(doc, [(source, r"{\C1;Привет}\PМир")])

    assert mtext.text == r"  {\C1;Привет}\PМир  "


def test_longest_source_wins_and_regex_metacharacters_are_literal():
    replacer = TextReplacer([("A", "short"), ("A+B", "long"), ("B", "other")])
    assert replacer.translate("(A+B) A") == "(long) short"


def test_escaped_braces_stacks_and_unicode_controls_are_preserved():
    result = translate_text_keep_dim_and_mtext_controls(
        r"\{Hello\} \S1/2; \U+00B0",
        [("Hello", "Привет"), ("1", "one"), ("00B0", "degree")],
    )
    assert result == r"\{Привет\} \S1/2; \U+00B0"
