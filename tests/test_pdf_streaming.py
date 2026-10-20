import fitz
import pytest
from PIL import Image, ImageDraw

from daru.pdf import _core, export, raster
from daru.pdf.pages import StoredPage
from daru.translation.progress import ProgressEvent, progress_scope


@pytest.mark.parametrize("fail_page", [None, 2, "render", "export"])
def test_scanned_pipeline_spools_pages_and_releases_pixels(
    tmp_path, monkeypatch, fail_page
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"synthetic fixture")
    output = tmp_path / "translated.pdf"
    tracked = []
    calls = []
    observed_loaded = []
    old_init = StoredPage.__init__

    def track(self, page, path):
        old_init(self, page, path)
        tracked.append(self)

    monkeypatch.setattr(StoredPage, "__init__", track)

    def pages(self, path):
        for index in range(4):
            image = Image.new("RGB", (600, 400), "white")
            ImageDraw.Draw(image).text(
                (35, 60), "Inspect the safety valve before operation.", fill="black"
            )
            yield _core.PageImage(index, image, 600, 400, 150, (288, 192))

    class Extractor:
        def __init__(self, *args, **kwargs):
            pass

        def analyze(self, page):
            if page.page_index == fail_page:
                raise RuntimeError("injected OCR failure")
            block = _core.BlockRegion(
                page.page_index,
                str(page.page_index),
                "TEXT",
                (30, 50, 550, 100),
                source_text="Inspect the safety valve before operation.",
            )
            return _core.TextractPageData(page, {}, {}, [block])

    class Engine:
        def __init__(self, **kwargs):
            calls.append(kwargs["target_lang"])

        def backend_name(self):
            return "test"

        def set_drawing_context(self, texts):
            assert len(texts) == 4  # Full document context survives streaming.

        def translate_many(self, texts):
            return ["Check the safety valve." for _ in texts]

    old_clean = _core.Renderer.clean_page

    def clean(self, page, regions):
        if fail_page == "render":
            raise RuntimeError("injected render failure")
        result = old_clean(self, page, regions)
        observed_loaded.append(sum(item._loaded is not None for item in tracked))
        return result

    monkeypatch.setattr(_core.PageLoader, "iter_pages", pages)
    monkeypatch.setattr(_core, "TextractLayoutExtractor", Extractor)
    monkeypatch.setattr(_core, "TranslationEngine", Engine)
    monkeypatch.setattr(_core.Renderer, "clean_page", clean)
    if fail_page == "export":

        def fail_export(*args, **kwargs):
            raise RuntimeError("injected export failure")

        monkeypatch.setattr(export, "images_to_pdf", fail_export)
    options = dict(
        input_path=source,
        output_path=output,
        translator_name="noop",
        source_lang="en",
        target_lang="ru",
        log=lambda _m: None,
        checkpoint_path=tmp_path / "checkpoint.json",
        layer_json_path=tmp_path / "layer.json",
        processing_options={"render_dpi": 150, "dpi": 150},
    )
    if fail_page is not None:
        with pytest.raises(RuntimeError, match="injected .* failure"):
            _core.translate_pdf(**options)
        assert not output.exists()
    else:
        events = []
        with progress_scope(events.append):
            result = _core.translate_pdf(**options)
        assert result["pages"] == 4
        assert ProgressEvent("ocr", 4) in events
        assert ProgressEvent("apply", 4, 4) in events
        assert events[-1] == ProgressEvent("export")
        assert not any(event.stage == "complete" for event in events)  # Only the job runner completes a job.
        assert max(observed_loaded) == 1
        with fitz.open(output) as doc:
            assert len(doc) == 4
            assert all(
                "Check the safety valve." in " ".join(page.get_text().split())
                for page in doc
            )
        _core.translate_pdf(**options)
        _core.translate_pdf(**dict(options, target_lang="fr"))
        assert calls == ["ru", "fr"]
    assert tracked
    assert all(item._loaded is None for item in tracked)
    assert all(not item.path.exists() for item in tracked)


def test_page_loader_requests_only_one_poppler_page(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    with fitz.open() as doc:
        for _ in range(3):
            doc.new_page(width=200, height=100)
        doc.save(source)
    calls = []

    def convert(path, **kwargs):
        calls.append((kwargs["first_page"], kwargs["last_page"], kwargs["use_cropbox"]))
        return [Image.new("RGB", (400, 200), "white")]

    monkeypatch.setattr(raster, "convert_from_path", convert)
    pages = _core.PageLoader(_core.PdfProcessingConfig()).iter_pages(source)
    first = next(pages)
    assert calls == [(1, 1, True)]
    first.image.close()
    for page in pages:
        assert page.physical_size_points == (200, 100)
        page.image.close()
    assert calls == [(1, 1, True), (2, 2, True), (3, 3, True)]
