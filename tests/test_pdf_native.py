from __future__ import annotations

import json
from pathlib import Path

import fitz

from daru.pdf import native
from daru.pdf import _core


class PrefixTranslationEngine:
    calls = 0

    def __init__(self, **_kwargs):
        self.context = []

    def set_document_context(self, texts, **_kwargs):
        self.context = list(texts)

    def backend_name(self):
        return "fake-native"

    def translate_many(self, texts):
        type(self).calls += 1
        return [f"Перевод: {text}" for text in texts]


class InvalidThenValidEngine(PrefixTranslationEngine):
    calls = 0

    def translate_many(self, texts):
        type(self).calls += 1
        if type(self).calls == 1:
            return ["Повреждённый перевод" for _text in texts]
        return [f"Перевод: {text}" for text in texts]


class FailIfCreatedEngine:
    def __init__(self, **_kwargs):
        raise AssertionError("cached translations must not initialize a translator")


def _save_native_pdf(path: Path, *, with_image_only_page: bool = False) -> None:
    document = fitz.open()
    page = document.new_page(width=500, height=300)
    page.draw_rect(fitz.Rect(20, 20, 480, 280), color=(0, 0, 1), width=2)
    page.insert_textbox(
        fitz.Rect(50, 50, 430, 110),
        "Safety device prevents elevator accidents.",
        fontsize=14,
        fontname="helv",
    )
    page.insert_textbox(
        fitz.Rect(50, 125, 430, 180),
        "Model STVF-5 uses 220 V.",
        fontsize=11,
        fontname="helv",
    )
    if with_image_only_page:
        second = document.new_page(width=500, height=300)
        second.draw_rect(fitz.Rect(30, 30, 470, 270), color=(1, 0, 0), fill=(0.9, 0.9, 0.9))
    document.save(path)
    document.close()


def _translate(
    input_path: Path,
    output_path: Path,
    *,
    layer_path: Path | None = None,
    fallback=None,
    style_font: str = "Arial.ttf",
    log=None,
    checkpoint_path: Path | None = None,
):
    return native.translate_native_pdf(
        input_path=input_path,
        output_path=output_path,
        translator_name="noop",
        source_lang="en",
        target_lang="ru",
        log=log or (lambda _message: None),
        layer_json_path=layer_path,
        style_font=style_font,
        fallback_page_translator=fallback,
        checkpoint_path=checkpoint_path,
    )


def test_native_pdf_preserves_vector_content_and_searchable_text(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    output = tmp_path / "translated.pdf"
    _save_native_pdf(source)
    PrefixTranslationEngine.calls = 0
    monkeypatch.setattr(native, "TranslationEngine", PrefixTranslationEngine)

    result = _translate(source, output)

    assert result["processing_mode"] == "native"
    assert result["native_pages"] == [1]
    assert result["ocr_pages"] == []
    assert result["ocr_performed"] is False
    assert PrefixTranslationEngine.calls >= 1

    document = fitz.open(output)
    try:
        text = document[0].get_text()
        assert "Перевод" in text
        assert "Safety device prevents elevator accidents." not in text
        assert len(document[0].get_drawings()) >= 1
    finally:
        document.close()


def test_native_pdf_original_mode_falls_back_for_base14_font(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    output = tmp_path / "translated.pdf"
    _save_native_pdf(source)
    logs = []
    monkeypatch.setattr(native, "TranslationEngine", PrefixTranslationEngine)

    _translate(source, output, style_font="original", log=logs.append)

    document = fitz.open(output)
    try:
        assert "Перевод" in document[0].get_text()
    finally:
        document.close()
    assert any("исходный шрифт" in message for message in logs)


def test_native_pdf_uses_page_level_fallback_only_for_image_page(tmp_path, monkeypatch):
    source = tmp_path / "mixed.pdf"
    output = tmp_path / "mixed-translated.pdf"
    _save_native_pdf(source, with_image_only_page=True)
    monkeypatch.setattr(native, "TranslationEngine", PrefixTranslationEngine)
    fallback_calls = []

    def fallback(page_index):
        fallback_calls.append(page_index)
        document = fitz.open()
        page = document.new_page(width=500, height=300)
        page.insert_text((40, 60), "OCR translated page", fontsize=14)
        data = document.tobytes()
        document.close()
        return data

    result = _translate(source, output, fallback=fallback)

    assert fallback_calls == [1]
    assert result["processing_mode"] == "hybrid"
    assert result["native_pages"] == [1]
    assert result["ocr_pages"] == [2]
    assert result["ocr_performed"] is True

    document = fitz.open(output)
    try:
        assert len(document) == 2
        assert "Перевод" in document[0].get_text()
        assert "OCR translated page" in document[1].get_text()
    finally:
        document.close()


def test_translation_layer_v2_reuses_exact_source_blocks(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    first_output = tmp_path / "first.pdf"
    second_output = tmp_path / "second.pdf"
    layer = tmp_path / "source.translation.json"
    _save_native_pdf(source)
    monkeypatch.setattr(native, "TranslationEngine", PrefixTranslationEngine)

    _translate(source, first_output, layer_path=layer)
    payload = json.loads(layer.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["document_sha256"]
    assert payload["native_pages"] == [1]
    assert payload["ocr_pages"] == []
    assert payload["pages"][0]["blocks"][0]["stable_id"]

    monkeypatch.setattr(native, "TranslationEngine", FailIfCreatedEngine)
    result = _translate(source, second_output, layer_path=layer)

    assert result["backend"] in {"cached-layer", "cached-checkpoint"}
    document = fitz.open(second_output)
    try:
        assert "Перевод" in document[0].get_text()
    finally:
        document.close()


def test_native_pdf_reuses_checkpoint_without_translator(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    first_output = tmp_path / "first.pdf"
    second_output = tmp_path / "second.pdf"
    checkpoint = tmp_path / "source.checkpoint.json"
    _save_native_pdf(source)
    monkeypatch.setattr(native, "TranslationEngine", PrefixTranslationEngine)

    _translate(source, first_output, checkpoint_path=checkpoint)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["job_type"] == "pdf-native"
    assert payload["blocks"]
    assert all(
        block["translated_text"] != block["source_text"]
        for block in payload["blocks"]
    )

    monkeypatch.setattr(native, "TranslationEngine", FailIfCreatedEngine)
    result = _translate(source, second_output, checkpoint_path=checkpoint)

    assert result["backend"] == "cached-checkpoint"
    document = fitz.open(second_output)
    try:
        assert "Перевод" in document[0].get_text()
    finally:
        document.close()


def test_legacy_layer_requires_exact_text_and_bbox(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    output = tmp_path / "translated.pdf"
    layer = tmp_path / "legacy.translation.json"
    _save_native_pdf(source)

    document = fitz.open(source)
    try:
        page = native._extract_page(document[0], 0, "en")
    finally:
        document.close()
    unit = page.units[0]
    layer.write_text(
        json.dumps(
            {
                "version": 1,
                "pages": [
                    {
                        "page_index": 0,
                        "blocks": [
                            {
                                "bbox": list(unit.bbox),
                                "text": unit.source_text,
                                "translated_text": "Legacy exact translation",
                            },
                            {
                                "bbox": [value + 1 for value in page.units[1].bbox],
                                "text": page.units[1].source_text,
                                "translated_text": "Must not be reused",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class TranslateUncachedOnlyEngine(PrefixTranslationEngine):
        def translate_many(self, texts):
            return [f"Fresh translation: {text}" for text in texts]

    monkeypatch.setattr(native, "TranslationEngine", TranslateUncachedOnlyEngine)
    _translate(source, output, layer_path=layer)

    document = fitz.open(output)
    try:
        text = document[0].get_text().replace("\u00a0", " ")
        assert "Legacy exact translation" in text
        assert "Must not be reused" not in text
        assert "Fresh translation" in text
    finally:
        document.close()


def test_invalid_placeholder_translation_is_retried_once(monkeypatch):
    unit = native._make_unit(
        page_index=0,
        bbox=(10.0, 10.0, 200.0, 40.0),
        role="TEXT",
        source_text="Model STVF-5 uses 220 V.",
        font_size=12,
        font_name="Arial",
        color=0,
        bold=False,
        rotation=0,
        alignment="left",
    )
    InvalidThenValidEngine.calls = 0
    monkeypatch.setattr(native, "TranslationEngine", InvalidThenValidEngine)

    backend = native._translate_units(
        [unit],
        {},
        translator_name="noop",
        source_lang="en",
        target_lang="ru",
        deepl_key=None,
        openai_key=None,
        openai_model=None,
        openai_base_url=None,
        openai_project=None,
        openai_temperature=0.2,
        openai_reasoning_effort=None,
        openai_verbosity=None,
        openai_strict_mode=None,
        openai_strict_value=None,
        log=lambda _message: None,
    )

    assert backend == "fake-native"
    assert InvalidThenValidEngine.calls == 2
    assert unit.translated_text == "Перевод: Model STVF-5 uses 220 V."


def test_table_cells_are_extracted_as_independent_units(tmp_path):
    source = tmp_path / "table.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=300)
    for x in (40, 200, 360):
        page.draw_line((x, 60), (x, 220), color=(0, 0, 0))
    for y in (60, 140, 220):
        page.draw_line((40, y), (360, y), color=(0, 0, 0))
    values = [
        (fitz.Rect(50, 75, 190, 125), "Item"),
        (fitz.Rect(210, 75, 350, 125), "Measure"),
        (fitz.Rect(50, 155, 190, 205), "Door"),
        (fitz.Rect(210, 155, 350, 205), "Check the sill"),
    ]
    for rect, text in values:
        page.insert_textbox(rect, text, fontsize=12)
    document.save(source)
    document.close()

    document = fitz.open(source)
    try:
        extracted = native._extract_page(document[0], 0, "en")
    finally:
        document.close()

    assert extracted.is_native
    cell_texts = [unit.source_text for unit in extracted.units if unit.role == "CELL"]
    assert {"Item", "Measure", "Door", "Check the sill"}.issubset(set(cell_texts))


def test_long_translation_records_layout_warning(tmp_path, monkeypatch):
    source = tmp_path / "small-box.pdf"
    output = tmp_path / "small-box-translated.pdf"
    document = fitz.open()
    page = document.new_page(width=220, height=120)
    page.insert_textbox(fitz.Rect(20, 20, 100, 42), "Short warning", fontsize=12)
    document.save(source)
    document.close()

    class LongTranslationEngine(PrefixTranslationEngine):
        def translate_many(self, texts):
            return [
                "Очень длинный перевод предупреждения " * 10
                for _text in texts
            ]

    monkeypatch.setattr(native, "TranslationEngine", LongTranslationEngine)
    result = _translate(source, output)

    assert result["layout_warnings"]
    assert result["layout_warnings"][0]["page"] == 1


def test_generated_html_css_has_balanced_braces():
    unit = native._make_unit(
        page_index=0,
        bbox=(10.0, 10.0, 100.0, 30.0),
        role="TEXT",
        source_text="Test",
        font_size=12,
        font_name="Arial",
        color=0,
        bold=False,
        rotation=0,
        alignment="left",
    )
    css = native._unit_css(unit, native._resolve_font_resource("Arial.ttf"))

    assert css.count("{") == css.count("}")


def test_original_font_resolver_uses_matching_embedded_font(monkeypatch):
    class FakeDocument:
        def get_page_fonts(self, _page_index, full=True):
            assert full
            return [(7, "ttf", "Type0", "Test Font Regular", "F1", "Identity-H", 0)]

    class FullCoverageFont:
        def has_glyph(self, _codepoint):
            return 1

    fallback = native._FontResource("Fallback", None, "", source_name="fallback.ttf")
    original = native._FontResource(
        "Original",
        None,
        "",
        font=FullCoverageFont(),
        source_name="test-font.ttf",
    )
    resolver = native._OriginalFontResolver(FakeDocument(), fallback, lambda _message: None)
    monkeypatch.setattr(resolver, "_resource_from_xref", lambda xref: original if xref == 7 else None)
    unit = native._make_unit(
        page_index=0,
        bbox=(10.0, 10.0, 100.0, 30.0),
        role="TEXT",
        source_text="Test",
        font_size=12,
        font_name="TestFont",
        color=0,
        bold=False,
        rotation=0,
        alignment="left",
    )
    unit.translated_text = "Перевод"

    try:
        assert resolver.resolve(unit) is original
    finally:
        resolver.close()


def test_original_font_resolver_falls_back_once_for_missing_glyphs(monkeypatch):
    class FakeDocument:
        def get_page_fonts(self, _page_index, full=True):
            assert full
            return [(9, "ttf", "Type0", "Latin Font", "F1", "Identity-H", 0)]

    class LatinOnlyFont:
        def has_glyph(self, codepoint):
            return 1 if codepoint < 128 else 0

    logs = []
    fallback = native._FontResource("Fallback", None, "", source_name="fallback.ttf")
    limited = native._FontResource(
        "Limited",
        None,
        "",
        font=LatinOnlyFont(),
        source_name="latin.ttf",
    )
    resolver = native._OriginalFontResolver(FakeDocument(), fallback, logs.append)
    monkeypatch.setattr(resolver, "_resource_from_xref", lambda xref: limited if xref == 9 else None)
    unit = native._make_unit(
        page_index=0,
        bbox=(10.0, 10.0, 100.0, 30.0),
        role="TEXT",
        source_text="Test",
        font_size=12,
        font_name="LatinFont",
        color=0,
        bold=False,
        rotation=0,
        alignment="left",
    )
    unit.translated_text = "Перевод"

    try:
        assert resolver.resolve(unit) is fallback
        assert resolver.resolve(unit) is fallback
    finally:
        resolver.close()

    assert len(logs) == 1
    assert "не содержит всех символов" in logs[0]


def test_scanned_original_font_uses_fallback_and_logs_warning():
    logs = []

    resolved = _core._resolve_scanned_style_font("Оригинал", logs.append)

    assert resolved is None
    assert len(logs) == 1
    assert "невозможно надежно определить" in logs[0]


def test_candidate_expansion_does_not_cross_vector_graphics():
    source = fitz.Rect(20, 20, 100, 40)
    footer_line = fitz.Rect(99, 15, 101, 60)

    candidates = native._candidate_rects(
        fitz.Rect(0, 0, 300, 200),
        source,
        "TEXT",
        [source, footer_line],
    )

    assert all(candidate.x1 <= 100 for candidate in candidates)


def test_ordinary_building_sentence_is_not_protected_as_an_address():
    source = "The elevator can be used in public buildings and offices."

    protected = native._protect_text(source)

    assert protected.value == source
    assert protected.replacements == {}


def test_distant_spans_on_one_line_are_separate_translation_units():
    block = {
        "type": 0,
        "lines": [
            {
                "dir": (1.0, 0.0),
                "spans": [
                    {
                        "bbox": (20.0, 20.0, 110.0, 34.0),
                        "text": "Operation Manual",
                        "size": 10.0,
                        "font": "Arial",
                        "color": 0,
                        "flags": 0,
                    },
                    {
                        "bbox": (360.0, 20.0, 445.0, 34.0),
                        "text": "Product Guide",
                        "size": 10.0,
                        "font": "Arial",
                        "color": 0,
                        "flags": 0,
                    },
                ],
            }
        ],
    }

    units = native._units_from_block(
        block,
        page_index=0,
        source_lang="en",
        median_font_size=10.0,
    )

    assert [unit.source_text for unit in units] == [
        "Operation Manual",
        "Product Guide",
    ]
    assert units[0].bbox[2] < units[1].bbox[0]


def test_zero_width_space_spans_are_preserved():
    line = {
        "dir": (1.0, 0.0),
        "spans": [
            {
                "bbox": (20.0, 20.0, 70.0, 34.0),
                "text": "Operation",
                "size": 10.0,
            },
            {
                "bbox": (70.0, 20.0, 70.0, 34.0),
                "text": " ",
                "size": 10.0,
            },
            {
                "bbox": (74.0, 20.0, 110.0, 34.0),
                "text": "Manual",
                "size": 10.0,
            },
        ],
    }

    segments = native._split_line_segments(line)

    assert len(segments) == 1
    assert segments[0]["text"] == "Operation Manual"


def test_rotated_text_keeps_quarter_turn(tmp_path, monkeypatch):
    source = tmp_path / "rotated.pdf"
    output = tmp_path / "rotated-translated.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=300)
    page.insert_text(
        (100, 250),
        "Vertical Label",
        fontsize=12,
        rotate=90,
    )
    document.save(source)
    document.close()

    class RotatedTranslationEngine(PrefixTranslationEngine):
        def translate_many(self, texts):
            return ["Translated Label" for _text in texts]

    monkeypatch.setattr(native, "TranslationEngine", RotatedTranslationEngine)
    _translate(source, output)

    document = fitz.open(output)
    try:
        extracted_text = document[0].get_text().replace("\u00a0", " ")
        assert "Translated Label" in extracted_text
        directions = [
            line.get("dir")
            for block in document[0].get_text("dict").get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
        ]
        assert (0.0, -1.0) in directions
    finally:
        document.close()


def test_public_pdf_router_honors_native_mode(tmp_path, monkeypatch):
    captured = {}

    def fake_native(**kwargs):
        captured.update(kwargs)
        return {"processing_mode": "native", "output_path": kwargs["output_path"]}

    def fail_scanned(**_kwargs):
        raise AssertionError("native mode must not use the full scanned pipeline")

    monkeypatch.setattr(native, "translate_native_pdf", fake_native)
    monkeypatch.setattr(_core, "_translate_scanned_pdf", fail_scanned)
    output = tmp_path / "translated.pdf"

    result = _core.translate_pdf(
        input_path=tmp_path / "source.pdf",
        output_path=output,
        translator_name="noop",
        source_lang="en",
        target_lang="ru",
        log=lambda _message: None,
        pdf_type="native",
        codex_cli_path="C:/tools/codex.cmd",
        codex_model="gpt-5.4-mini",
        codex_reasoning_effort="high",
        codex_analysis_model="gpt-5.5",
        codex_analysis_reasoning_effort="xhigh",
        codex_timeout_seconds=360,
    )

    assert result["processing_mode"] == "native"
    assert captured["output_path"] == output
    assert callable(captured["fallback_page_translator"])
    assert captured["codex_cli_path"] == "C:/tools/codex.cmd"
    assert captured["codex_model"] == "gpt-5.4-mini"
    assert captured["codex_reasoning_effort"] == "high"
    assert captured["codex_analysis_model"] == "gpt-5.5"
    assert captured["codex_analysis_reasoning_effort"] == "xhigh"
    assert captured["codex_timeout_seconds"] == 360


def test_hybrid_pdf_reuses_one_codex_analysis_session(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    document = fitz.open()
    document.new_page()
    document.save(source)
    document.close()
    captured = {}

    def fake_scanned(**kwargs):
        captured["fallback_session"] = kwargs["codex_analysis_session"]
        captured["fallback_model"] = kwargs["codex_analysis_model"]
        captured["fallback_effort"] = kwargs["codex_analysis_reasoning_effort"]
        kwargs["output_path"].write_bytes(b"translated-page")
        return {"output_path": kwargs["output_path"], "processing_mode": "scanned"}

    def fake_native(**kwargs):
        captured["native_session"] = kwargs["codex_analysis_session"]
        captured["native_model"] = kwargs["codex_analysis_model"]
        captured["native_effort"] = kwargs["codex_analysis_reasoning_effort"]
        captured["fallback_bytes"] = kwargs["fallback_page_translator"](0)
        return {"processing_mode": "hybrid", "output_path": kwargs["output_path"]}

    monkeypatch.setattr(native, "translate_native_pdf", fake_native)
    monkeypatch.setattr(_core, "_translate_scanned_pdf", fake_scanned)

    result = _core.translate_pdf(
        input_path=source,
        output_path=tmp_path / "translated.pdf",
        translator_name="codex",
        source_lang="en",
        target_lang="ru",
        log=lambda _message: None,
        pdf_type="native",
        codex_analysis_model="gpt-5.5",
        codex_analysis_reasoning_effort="high",
    )

    assert result["processing_mode"] == "hybrid"
    assert captured["native_session"] is captured["fallback_session"]
    assert captured["native_model"] == captured["fallback_model"] == "gpt-5.5"
    assert captured["native_effort"] == captured["fallback_effort"] == "high"
    assert captured["fallback_bytes"] == b"translated-page"
