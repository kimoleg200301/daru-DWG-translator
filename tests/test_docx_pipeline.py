"""Tests for direct OOXML DOCX translation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile

import pytest
from lxml import etree

from daru.docx import pipeline

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
W = f"{{{W_NS}}}"

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

DOCUMENT_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}">
  <w:body>
    <w:p>
      <w:r><w:t xml:space="preserve">Hello </w:t></w:r>
      <w:r><w:rPr><w:b/></w:rPr><w:t>world</w:t></w:r>
    </w:p>
    <w:p>
      <w:r>
        <w:t>Main rope</w:t>
        <w:tab/>
        <w:t>safety rope</w:t>
        <w:tab/>
        <w:t>balancing rope</w:t>
      </w:r>
    </w:p>
    <w:p><w:r><w:t>Уже переведено</w:t></w:r></w:p>
    <w:p><w:r><w:t>12345</w:t></w:r></w:p>
    <w:p><w:r><w:t>Visit https://example.com now</w:t></w:r></w:p>
    <w:p>
      <w:r><w:rPr><w:vanish/></w:rPr><w:t>Hidden secret</w:t></w:r>
      <w:del><w:r><w:delText>Deleted text</w:delText></w:r></w:del>
    </w:p>
    <w:p>
      <w:r><w:fldChar w:fldCharType="begin"/></w:r>
      <w:r><w:instrText> PAGE </w:instrText></w:r>
      <w:r><w:fldChar w:fldCharType="separate"/></w:r>
      <w:r><w:t>41</w:t></w:r>
      <w:r><w:fldChar w:fldCharType="end"/></w:r>
      <w:r><w:t>After field</w:t></w:r>
    </w:p>
    <w:p>
      <w:hyperlink r:id="rId5"><w:r><w:t>Open link</w:t></w:r></w:hyperlink>
    </w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Table text</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    <w:p>
      <w:r><w:drawing><w:txbxContent><w:p><w:r><w:t>Text box</w:t></w:r></w:p></w:txbxContent></w:drawing></w:r>
    </w:p>
    <w:sectPr/>
  </w:body>
</w:document>
"""

HEADER_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="{W_NS}">
  <w:p><w:r><w:t>Header text</w:t></w:r></w:p>
</w:hdr>
"""

FOOTER_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W_NS}">
  <w:p><w:r><w:t>Footer text</w:t></w:r></w:p>
</w:ftr>
"""

FOOTNOTES_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes xmlns:w="{W_NS}">
  <w:footnote w:id="1"><w:p><w:r><w:t>Footnote text</w:t></w:r></w:p></w:footnote>
</w:footnotes>
"""

ENDNOTES_XML = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:endnotes xmlns:w="{W_NS}">
  <w:endnote w:id="1"><w:p><w:r><w:t>Endnote text</w:t></w:r></w:p></w:endnote>
</w:endnotes>
"""


class FakeTranslationEngine:
    replacements = {
        "Hello": "Привет",
        "world": "мир",
        "Main rope": "Главный канат",
        "safety rope": "предохранительный канат",
        "balancing rope": "уравновешивающий канат",
        "Visit": "Посетите",
        "now": "сейчас",
        "After field": "После поля",
        "Open link": "Открыть ссылку",
        "Table text": "Текст таблицы",
        "Text box": "Текстовый блок",
        "Header text": "Текст заголовка",
        "Footer text": "Текст подвала",
        "Footnote text": "Текст сноски",
        "Endnote text": "Текст концевой сноски",
    }

    def __init__(self, **_kwargs):
        self.context = []

    def backend_name(self):
        return "fake"

    def set_document_context(self, texts, **_kwargs):
        self.context = list(texts)

    def translate_many(self, texts):
        translated = []
        for text in texts:
            value = text
            for source, target in self.replacements.items():
                value = value.replace(source, target)
            translated.append(value)
        return translated


class CorruptingMarkerEngine(FakeTranslationEngine):
    def translate_many(self, texts):
        translated = super().translate_many(texts)
        return [
            re.sub(r"\[\[/?DARU_FMT_\d+\]\]", "", value)
            if "[[DARU_FMT_" in value
            else value
            for value in translated
        ]


class FailingTranslationEngine(FakeTranslationEngine):
    def translate_many(self, texts):
        raise RuntimeError("provider failed")


class LeakingMarkerEngine(FakeTranslationEngine):
    received = []

    def translate_many(self, texts):
        self.__class__.received.extend(texts)
        translated = super().translate_many(texts)
        results = []
        for source, value in zip(texts, translated):
            if "[[DARU_FMT_" in source:
                results.append(f"[[DARU_FMT_0]]{value}[[/DARU_FMT_2]]")
            else:
                results.append(f"[[DARU_FMT_0]]{value}[[/DARU_FMT_0]]")
        return results


class RecordingTranslationEngine(FakeTranslationEngine):
    received = []

    def translate_many(self, texts):
        self.__class__.received.extend(texts)
        return super().translate_many(texts)


def _make_docx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", ROOT_RELS)
        archive.writestr("word/document.xml", DOCUMENT_XML)
        archive.writestr("word/header1.xml", HEADER_XML)
        archive.writestr("word/footer1.xml", FOOTER_XML)
        archive.writestr("word/footnotes.xml", FOOTNOTES_XML)
        archive.writestr("word/endnotes.xml", ENDNOTES_XML)
        archive.writestr("word/media/image1.png", b"unchanged-image")


def _read_part(path: Path, name: str) -> etree._Element:
    with ZipFile(path, "r") as archive:
        return etree.fromstring(archive.read(name))


def _all_text(root: etree._Element) -> str:
    return "|".join(root.xpath(".//w:t/text()", namespaces={"w": W_NS}))


def _paragraph_text_with_tabs(paragraph: etree._Element) -> str:
    values = []
    for element in paragraph.iter():
        if element.tag == f"{W}t":
            values.append(element.text or "")
        elif element.tag == f"{W}tab":
            values.append("\t")
    return "".join(values)


def test_translate_docx_preserves_structure_and_translates_visible_text(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.docx"
    output = tmp_path / "translated.docx"
    _make_docx(source)
    monkeypatch.setattr(pipeline, "TranslationEngine", FakeTranslationEngine)

    result = pipeline.translate_docx(
        input_path=source,
        output_path=output,
        translator_name="fake",
        source_lang="en",
        target_lang="ru",
    )

    document = _read_part(output, "word/document.xml")
    text = _all_text(document)
    assert "Привет" in text
    assert "мир" in text
    assert "Уже переведено" in text
    assert "https://example.com" in text
    assert "41" in text
    assert "После поля" in text
    assert "Открыть ссылку" in text
    assert "Текст таблицы" in text
    assert "Текстовый блок" in text
    assert "Hidden secret" in text
    assert document.find(f".//{W}b") is not None
    assert document.find(f".//{W}hyperlink").get(f"{{{R_NS}}}id") == "rId5"

    assert "Текст заголовка" in _all_text(_read_part(output, "word/header1.xml"))
    assert "Текст подвала" in _all_text(_read_part(output, "word/footer1.xml"))
    assert "Текст сноски" in _all_text(_read_part(output, "word/footnotes.xml"))
    assert "Текст концевой сноски" in _all_text(
        _read_part(output, "word/endnotes.xml")
    )
    with ZipFile(output, "r") as archive:
        assert archive.read("word/media/image1.png") == b"unchanged-image"

    assert result["backend"] == "fake"
    assert result["items_translated"] > 0
    assert result["items_skipped"] >= 2


def test_marker_corruption_uses_fragment_fallback(tmp_path, monkeypatch):
    source = tmp_path / "source.docx"
    output = tmp_path / "translated.docx"
    _make_docx(source)
    monkeypatch.setattr(pipeline, "TranslationEngine", CorruptingMarkerEngine)

    pipeline.translate_docx(
        input_path=source,
        output_path=output,
        translator_name="fake",
        source_lang="en",
        target_lang="ru",
    )

    document = _read_part(output, "word/document.xml")
    first_paragraph_text = "".join(
        document.xpath(".//w:body/w:p[1]//w:t/text()", namespaces={"w": W_NS})
    )
    assert first_paragraph_text == "Привет мир"
    assert document.find(f".//{W}b") is not None


def test_tab_separated_sentence_is_sent_as_one_translation_unit(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.docx"
    output = tmp_path / "translated.docx"
    _make_docx(source)
    RecordingTranslationEngine.received = []
    monkeypatch.setattr(pipeline, "TranslationEngine", RecordingTranslationEngine)

    pipeline.translate_docx(
        input_path=source,
        output_path=output,
        translator_name="fake",
        source_lang="en",
        target_lang="ru",
    )

    tab_requests = [
        text for text in RecordingTranslationEngine.received if "Main rope" in text
    ]
    assert len(tab_requests) == 1
    assert "safety rope" in tab_requests[0]
    assert "balancing rope" in tab_requests[0]
    assert "[[DARU_" not in tab_requests[0]
    assert tab_requests[0].count("\t") == 2

    document = _read_part(output, "word/document.xml")
    paragraphs = document.xpath(".//w:body/w:p", namespaces={"w": W_NS})
    translated = next(
        paragraph
        for paragraph in paragraphs
        if "Главный канат" in _paragraph_text_with_tabs(paragraph)
    )
    assert _paragraph_text_with_tabs(translated) == (
        "Главный канат\tпредохранительный канат\tуравновешивающий канат"
    )
    assert len(translated.xpath(".//w:tab", namespaces={"w": W_NS})) == 2


def test_damaged_markers_retry_whole_phrase_and_never_leak(tmp_path, monkeypatch):
    source = tmp_path / "source.docx"
    output = tmp_path / "translated.docx"
    _make_docx(source)
    LeakingMarkerEngine.received = []
    monkeypatch.setattr(pipeline, "TranslationEngine", LeakingMarkerEngine)
    logs = []

    pipeline.translate_docx(
        input_path=source,
        output_path=output,
        translator_name="fake",
        source_lang="en",
        target_lang="ru",
        log=logs.append,
    )

    assert "Hello world" in LeakingMarkerEngine.received
    assert any("повторяем цельные фразы" in message for message in logs)
    assert not any("пофрагментный режим" in message for message in logs)
    with ZipFile(output, "r") as archive:
        for name in archive.namelist():
            if name.endswith(".xml"):
                assert b"[[DARU_" not in archive.read(name)

    document = _read_part(output, "word/document.xml")
    first_paragraph_text = "".join(
        document.xpath(".//w:body/w:p[1]//w:t/text()", namespaces={"w": W_NS})
    )
    assert first_paragraph_text == "Привет мир"


def test_failure_does_not_replace_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "source.docx"
    output = tmp_path / "translated.docx"
    _make_docx(source)
    output.write_bytes(b"existing-output")
    monkeypatch.setattr(pipeline, "TranslationEngine", FailingTranslationEngine)

    with pytest.raises(RuntimeError, match="provider failed"):
        pipeline.translate_docx(
            input_path=source,
            output_path=output,
            translator_name="fake",
            source_lang="en",
            target_lang="ru",
        )

    assert output.read_bytes() == b"existing-output"


def test_translate_docx_resumes_from_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "source.docx"
    output = tmp_path / "translated.docx"
    checkpoint = tmp_path / "checkpoint.json"
    _make_docx(source)

    class FailingAfterFirstBatch(FakeTranslationEngine):
        calls = 0

        def translate_many(self, texts):
            type(self).calls += 1
            if type(self).calls > 1:
                raise RuntimeError("provider failed")
            return super().translate_many(texts)

    monkeypatch.setattr(pipeline, "TranslationEngine", FailingAfterFirstBatch)

    with pytest.raises(RuntimeError, match="provider failed"):
        pipeline.translate_docx(
            input_path=source,
            output_path=output,
            translator_name="fake",
            source_lang="en",
            target_lang="ru",
            checkpoint_path=checkpoint,
        )

    first_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert first_payload["blocks"][0]["namespace"] == "docx:encoded"
    checkpoint_source = first_payload["blocks"][0]["source_text"]

    class ResumeEngine(FakeTranslationEngine):
        received = []

        def translate_many(self, texts):
            type(self).received.extend(texts)
            return super().translate_many(texts)

    monkeypatch.setattr(pipeline, "TranslationEngine", ResumeEngine)

    pipeline.translate_docx(
        input_path=source,
        output_path=output,
        translator_name="fake",
        source_lang="en",
        target_lang="ru",
        checkpoint_path=checkpoint,
    )

    assert checkpoint_source not in ResumeEngine.received
    document = _read_part(output, "word/document.xml")
    text = _all_text(document)
    assert "Привет" in text
    assert "мир" in text


def test_source_and_output_must_differ(tmp_path):
    source = tmp_path / "source.docx"
    _make_docx(source)

    with pytest.raises(RuntimeError, match="должны отличаться"):
        pipeline.translate_docx(
            input_path=source,
            output_path=source,
            translator_name="noop",
        )


def test_docx_pipeline_passes_codex_settings_to_engine(tmp_path, monkeypatch):
    source = tmp_path / "source.docx"
    output = tmp_path / "translated.docx"
    _make_docx(source)
    captured = {}

    class CapturingEngine(FakeTranslationEngine):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__()

    monkeypatch.setattr(pipeline, "TranslationEngine", CapturingEngine)
    pipeline.translate_docx(
        input_path=source,
        output_path=output,
        translator_name="codex",
        codex_cli_path="C:/tools/codex.cmd",
        codex_model="gpt-5.4-mini",
        codex_reasoning_effort="medium",
        codex_analysis_model="gpt-5.5",
        codex_analysis_reasoning_effort="high",
        codex_timeout_seconds=240,
    )

    assert captured["provider"] == "codex"
    assert captured["codex_cli_path"] == "C:/tools/codex.cmd"
    assert captured["codex_model"] == "gpt-5.4-mini"
    assert captured["codex_reasoning_effort"] == "medium"
    assert captured["codex_analysis_model"] == "gpt-5.5"
    assert captured["codex_analysis_reasoning_effort"] == "high"
    assert captured["codex_timeout_seconds"] == 240
