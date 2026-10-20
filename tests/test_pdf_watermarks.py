"""Native PDF watermark Forms survive extraction and redaction cloning."""

import fitz
import pytest

from daru.pdf.watermarks import preserve_watermarks
from daru.pdf import native


def _watermark_form(document, page, *, text="CANNY", marked=True):
    regular = page.insert_font(fontname="helv")
    mono = page.insert_font(fontname="cour")
    xref = document.get_new_xref()
    metadata = "/PieceInfo << /ADBE_CompoundType << /Private /Watermark >> >>" if marked else ""
    document.update_object(xref, (
        "<< /Type /XObject /Subtype /Form /BBox [0 -12 200 35] "
        f"/Resources << /Font << /F0 {regular} 0 R /F1 {mono} 0 R >> >> {metadata} >>"
    ))
    document.update_stream(xref, (
        f"BT /F0 30 Tf 1 0 0 1 0 0 Tm ({text}) Tj "
        "/F1 18 Tf 1 0 0 1 110 0 Tm ( TRACE) Tj ET"
    ).encode("ascii"))
    return xref


def _paint_form(document, page, xref, *, name="Watermark1", x=60, y=70):
    opacity = document.get_new_xref()
    document.update_object(opacity, "<< /Type /ExtGState /ca 0.1 /CA 0.1 >>")
    kind, value = document.xref_get_key(page.xref, "Resources")
    resource = int(value.split()[0]) if kind == "xref" else page.xref
    prefix = "" if kind == "xref" else "Resources/"
    document.xref_set_key(resource, f"{prefix}XObject/{name}", f"{xref} 0 R")
    document.xref_set_key(resource, f"{prefix}ExtGState/{name}", f"{opacity} 0 R")
    content = document.get_new_xref()
    document.update_object(content, "<< >>")
    document.update_stream(content, (
        f"q /Artifact << /Subtype /Watermark >> BDC /{name} gs "
        f"0.707107 0.707107 -0.707107 0.707107 {x} {y} cm /{name} Do EMC Q"
    ).encode("ascii"))
    contents = page.get_contents() + [content]
    document.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{item} 0 R" for item in contents) + "]")


def _make_page(document):
    page = document.new_page(width=300, height=300)
    page.insert_text((30, 80), "Readable body text", fontsize=12)
    return page


def _watermark_trace(page):
    return [{key: value for key, value in trace.items() if key != "seqno"}
            for trace in page.get_texttrace() if trace["size"] >= 18]


def _redact_body(page):
    page.add_redact_annot(fitz.Rect(50, 40, 250, 270), fill=None)
    page.apply_redactions(images=0, graphics=0, text=0)


def _assert_no_temporary_markers(document):
    assert all("DaruWatermarkPreserve" not in document.xref_object(xref)
               for xref in range(1, document.xref_length()))


def test_explicit_watermark_is_hidden_for_extraction_then_restored():
    with fitz.open() as document:
        page = _make_page(document)
        xref = _watermark_form(document, page)
        _paint_form(document, page, xref)
        original = (document.xref_object(xref), document.xref_stream(xref))
        trace = _watermark_trace(page)

        with preserve_watermarks(page):
            assert "CANNY" not in page.get_text()
            assert "Readable body text" in page.get_text()
            assert not _watermark_trace(page)

        assert _watermark_trace(page) == trace
        assert (document.xref_object(xref), document.xref_stream(xref)) == original
        _assert_no_temporary_markers(document)


def test_redaction_clone_preserves_fonts_rotation_opacity_and_export():
    with fitz.open() as document:
        page = _make_page(document)
        xref = _watermark_form(document, page)
        _paint_form(document, page, xref)
        trace = _watermark_trace(page)

        with preserve_watermarks(page):
            _redact_body(page)
            assert not _watermark_trace(page)
            assert xref not in [item[0] for item in page.get_xobjects()]

        assert _watermark_trace(page) == trace
        assert {item["font"] for item in trace} == {"Helvetica", "Courier"}
        assert all(item["opacity"] == pytest.approx(0.1) for item in trace)
        assert all(item["dir"] == pytest.approx((0.707107, -0.707107)) for item in trace)
        _assert_no_temporary_markers(document)
        saved = document.tobytes(garbage=4, deflate=True, clean=True)
    with fitz.open(stream=saved, filetype="pdf") as restored:
        assert _watermark_trace(restored[0]) == trace
        _assert_no_temporary_markers(restored)


@pytest.mark.parametrize("redact", [False, True])
def test_watermark_restoration_runs_when_processing_raises(redact):
    with fitz.open() as document:
        page = _make_page(document)
        xref = _watermark_form(document, page)
        _paint_form(document, page, xref)
        trace = _watermark_trace(page)

        with pytest.raises(RuntimeError, match="processing failed"):
            with preserve_watermarks(page):
                if redact:
                    _redact_body(page)
                raise RuntimeError("processing failed")

        assert _watermark_trace(page) == trace
        assert document.xref_stream(xref)
        _assert_no_temporary_markers(document)


def test_multiple_watermarks_and_shared_forms_restore_independently():
    with fitz.open() as document:
        page = _make_page(document)
        shared = _watermark_form(document, page, text="FIRST")
        second = _watermark_form(document, page, text="SECOND")
        _paint_form(document, page, shared, name="Watermark1")
        _paint_form(document, page, second, name="Watermark2", x=100, y=110)
        _make_page(document)
        page, other = document[0], document[1]
        _paint_form(document, other, shared)
        expected = _watermark_trace(page)
        shared_expected = _watermark_trace(other)

        with preserve_watermarks(page):
            _redact_body(page)

        assert _watermark_trace(page) == expected
        assert _watermark_trace(document[1]) == shared_expected
        assert len(document[0].get_xobjects()) == 2
        _assert_no_temporary_markers(document)


def test_unmarked_forms_and_legitimate_translucent_text_are_kept_live():
    with fitz.open() as document:
        page = _make_page(document)
        unmarked = _watermark_form(document, page, marked=False)
        _paint_form(document, page, unmarked)
        page.insert_text((20, 30), "Translucent heading", fontsize=10, fill_opacity=0.1)
        expected = page.get_texttrace()

        with preserve_watermarks(page):
            assert page.get_texttrace() == expected
            assert "CANNY" in page.get_text()
            assert "Translucent heading" in page.get_text()

        assert page.get_texttrace() == expected
        _assert_no_temporary_markers(document)


def test_nested_preservation_restores_outer_state_and_final_watermark():
    with fitz.open() as document:
        page = _make_page(document)
        xref = _watermark_form(document, page)
        _paint_form(document, page, xref)
        expected = _watermark_trace(page)

        with preserve_watermarks(page):
            with preserve_watermarks(page):
                _redact_body(page)
            assert not _watermark_trace(page)

        assert _watermark_trace(page) == expected
        _assert_no_temporary_markers(document)


def test_native_translation_excludes_watermark_and_preserves_it_during_rendering():
    with fitz.open() as document:
        page = _make_page(document)
        page.insert_text((90, 185), "Check the rail.", fontsize=12)
        page.insert_text((25, 275), "Original nearby caption", fontsize=10)
        watermark = _watermark_form(document, page)
        _paint_form(document, page, watermark)
        expected = _watermark_trace(page)

        model = native._extract_page(page, 0, "en")
        assert all("CANNY" not in unit.source_text and "TRACE" not in unit.source_text for unit in model.units)
        body = next(unit for unit in model.units if unit.source_text == "Check the rail.")
        body.translated_text = "Проверьте направляющую."
        body.cached = True
        native._render_native_pages(document, [model], "Times New Roman", lambda _: None)

        assert _watermark_trace(page) == expected
        assert "Проверьте" in page.get_text()
        assert "Original nearby caption" in page.get_text()
        assert "Check the rail." not in page.get_text()
        _assert_no_temporary_markers(document)
        saved = document.tobytes(garbage=4, deflate=True, clean=True)
    with fitz.open(stream=saved, filetype="pdf") as restored:
        assert _watermark_trace(restored[0]) == expected
        assert "Проверьте" in restored[0].get_text()
        assert "Original nearby caption" in restored[0].get_text()
