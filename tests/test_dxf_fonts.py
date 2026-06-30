"""DXF font-style behavior tests."""

from __future__ import annotations

import ezdxf

from daru.dxf import applier
from daru.dxf.pipeline import apply_translations


def test_original_font_preserves_compatible_style_and_falls_back_per_entity(monkeypatch):
    document = ezdxf.new()
    document.styles.new("Compatible", dxfattribs={"font": "compatible.ttf"})
    document.styles.new("Limited", dxfattribs={"font": "limited.ttf"})
    modelspace = document.modelspace()
    compatible = modelspace.add_text("Hello", dxfattribs={"style": "Compatible"})
    limited = modelspace.add_text("World", dxfattribs={"style": "Limited"})
    logs = []

    def fake_support(font_name, _text):
        if font_name == "compatible.ttf":
            return True, ""
        if font_name == "limited.ttf":
            return False, "missing_glyphs"
        if font_name == "Arial.ttf":
            return True, ""
        return False, "unavailable"

    monkeypatch.setattr(applier, "_font_supports_text", fake_support)

    apply_translations(
        document,
        [("Hello", "Привет"), ("World", "Мир")],
        style_font="original",
        log=logs.append,
    )

    assert compatible.dxf.style == "Compatible"
    assert limited.dxf.style.startswith("RU")
    assert document.styles.get(limited.dxf.style).dxf.font == "Arial.ttf"
    assert len(logs) == 1
    assert "не содержит всех символов" in logs[0]


def test_explicit_font_selection_does_not_leak_between_jobs():
    first = ezdxf.new()
    first_text = first.modelspace().add_text("Hello")
    apply_translations(first, [("Hello", "Привет")], style_font="Arial.ttf")

    second = ezdxf.new()
    second_text = second.modelspace().add_text("Hello")
    apply_translations(second, [("Hello", "Привет")], style_font="NotoSans-Regular.ttf")

    assert first.styles.get(first_text.dxf.style).dxf.font == "Arial.ttf"
    assert second.styles.get(second_text.dxf.style).dxf.font == "NotoSans-Regular.ttf"
    assert applier.STYLE_FONT == "Arial.ttf"
