import json
from types import SimpleNamespace

import ezdxf
import pytest

from daru.dxf import pipeline
from daru.pdf.layers import load_layer_mappings
from daru.translation.checkpoint import TranslationCheckpointStore
from daru.translation.options import TranslationOptions


def test_textract_cache_hit_does_not_initialize_aws_client(tmp_path, monkeypatch):
    from PIL import Image
    from daru.pdf import _core

    def unexpected_client(self):
        raise AssertionError("a valid OCR cache must work without AWS initialization")

    monkeypatch.setattr(
        _core.TextractLayoutExtractor, "_init_client", unexpected_client
    )
    extractor = _core.TextractLayoutExtractor(None, log_path=tmp_path / "missing")
    extractor._cached_entries = [
        {
            "page_index": 0,
            "image_sha256": "image",
            "response": {"Blocks": []},
            "feature_types": ("LAYOUT", "SIGNATURES", "TABLES"),
        }
    ]
    monkeypatch.setattr(
        extractor,
        "_encode_image_for_textract",
        lambda *args, **kwargs: (b"image", {"image_sha256": "image"}),
    )
    with Image.new("RGB", (100, 100), "white") as image:
        layout = extractor.analyze(_core.PageImage(0, image, 100, 100, 150))
        assert layout.blocks == []


def test_native_layer_bbox_fallback_cannot_cross_page_boundaries(tmp_path):
    from daru.pdf import native

    units = [
        SimpleNamespace(
            stable_id=f"new-{index}",
            page_index=index,
            source_text="Source",
            bbox=(10, 10, 100, 30),
        )
        for index in range(2)
    ]
    payload = {
        "version": 4,
        "document_sha256": "hash",
        "pages": [
            {
                "page_index": index,
                "blocks": [
                    {
                        "stable_id": f"old-{index}",
                        "text": "Source",
                        "bbox": [10, 10, 100, 30],
                        "translated_text": f"Translation {index}",
                    }
                ],
            }
            for index in range(2)
        ],
    }
    path = tmp_path / "layer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert native._load_translation_layer(
        path, "hash", units, lambda message: None
    ) == {
        "new-0": "Translation 0",
        "new-1": "Translation 1",
    }


@pytest.mark.parametrize(
    "file_hash, image_hash",
    [("other-file", "image"), ("file", "other-image"), ("file", None)],
)
def test_textract_cache_never_matches_byte_count_alone(tmp_path, file_hash, image_hash):
    from daru.pdf import _core

    extractor = _core.TextractLayoutExtractor(
        object(), log_path=tmp_path / "missing", file_hash="file"
    )
    extractor._cached_entries = [
        {
            "page_index": 0,
            "file_hash": file_hash,
            "image_sha256": image_hash,
            "image_bytes": 123,
            "feature_types": ("LAYOUT",),
            "response": {"Blocks": []},
        }
    ]
    assert (
        extractor._find_cached_response(
            0,
            {
                "image_sha256": "image",
                "image_bytes": 123,
                "feature_types": ("LAYOUT",),
            },
        )
        is None
    )


def test_textract_cache_loads_only_current_document_and_reuses_exact_image(tmp_path):
    from daru.pdf import _core

    path = tmp_path / "textract.jsonl"
    entries = [
        {
            "file_hash": key,
            "page_index": 0,
            "request": {"image_sha256": "image", "feature_types": ["LAYOUT"]},
            "response": {"data": {"Blocks": [{"Id": key}]}},
        }
        for key in ("other", "current")
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")
    extractor = _core.TextractLayoutExtractor(
        object(), log_path=path, file_hash="current"
    )
    assert len(extractor._cached_entries) == 1
    assert extractor._find_cached_response(
        0, {"image_sha256": "image", "feature_types": ("LAYOUT",)}
    ) == {
        "Blocks": [{"Id": "current"}],
    }


def test_native_pdf_layer_isolated_by_language_and_model(tmp_path, monkeypatch):
    import fitz
    from daru.pdf import native

    source = tmp_path / "source.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 100), "Inspect the safety valve before operation.")
        doc.save(source)
    calls = []

    class Engine:
        def __init__(self, **options):
            self.target = options["target_lang"]
            self.model = options["openai_model"]

        def set_document_context(self, *args, **kwargs):
            pass

        def backend_name(self):
            return "test"

        def translate_many(self, texts):
            calls.append((self.target, self.model))
            return [f"Translated {self.target} {self.model}" for _ in texts]

    monkeypatch.setattr(native, "TranslationEngine", Engine)
    output = tmp_path / "result.pdf"
    for language, model in [
        ("ru", "first"),
        ("fr", "first"),
        ("fr", "second"),
        ("fr", "second"),
    ]:
        native.translate_native_pdf(
            input_path=source,
            output_path=output,
            translator_name="chatgpt",
            source_lang="en",
            target_lang=language,
            openai_model=model,
            log=lambda message: None,
            layer_json_path=tmp_path / "layer.json",
            checkpoint_path=tmp_path / "checkpoint.json",
        )
        with fitz.open(output) as doc:
            assert f"Translated {language} {model}" in " ".join(
                doc[0].get_text().split()
            )
    assert calls == [("ru", "first"), ("fr", "first"), ("fr", "second")]


def test_dxf_automatic_cache_does_not_read_exported_csv(tmp_path, monkeypatch):
    source = tmp_path / "drawing.dxf"
    doc = ezdxf.new()
    doc.modelspace().add_text("Hello")
    doc.saveas(source)
    csv = tmp_path / "map_auto.csv"
    csv.write_text("text_en,text_ru\nHello,OLD LANGUAGE\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = []

    class Engine:
        def __init__(self, **options):
            self.target = options["target_lang"]

        def set_drawing_context(self, *args, **kwargs):
            pass

        def backend_name(self):
            return "test"

        def translate_many(self, texts):
            calls.append(self.target)
            return [self.target.upper() for _ in texts]

    monkeypatch.setattr(pipeline, "TranslationEngine", Engine)
    for lang in ("ru", "fr", "fr"):
        pipeline.translate_dxf(
            input_path=source,
            output_path=tmp_path / "result.dxf",
            target_lang=lang,
            translator_name="noop",
            map_path=csv,
            save_map=False,
            save_txt=False,
            checkpoint_path=tmp_path / "checkpoint.json",
        )
    assert calls == ["ru", "fr"]
    assert (
        ezdxf.readfile(tmp_path / "result.dxf").modelspace().query("TEXT")[0].dxf.text
        == "FR"
    )
    assert "OLD LANGUAGE" in csv.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "change",
    [
        {"target_lang": "fr"},
        {"source_lang": "de"},
        {"provider": "codex"},
        {"openai_model": "another-model"},
        {"system_prompt_template": "new instructions"},
    ],
)
def test_identity_changes_for_translation_semantics(change):
    original = {"provider": "chatgpt", "source_lang": "en", "target_lang": "ru"}
    first = TranslationOptions.from_mapping(original).cache_identity("hash", "cad")
    second = TranslationOptions.from_mapping(original | change).cache_identity(
        "hash", "cad"
    )
    assert first != second


def test_secrets_are_excluded_and_language_aliases_normalized():
    options = TranslationOptions.from_mapping(
        {"source_lang": "English", "target_lang": "Русский", "openai_key": "secret"}
    )
    assert options.target_lang == "ru"
    assert "secret" not in repr(options)
    assert "secret" not in json.dumps(options.cache_identity("hash", "cad"))


def test_checkpoint_requires_matching_settings_when_supplied(tmp_path):
    params = dict(
        path=tmp_path / "checkpoint.json",
        job_type="cad",
        document_sha256="hash",
        source_lang="en",
        target_lang="ru",
    )
    store = TranslationCheckpointStore(**params, settings_fingerprint="old")
    store.upsert(
        namespace="cad", block_id="1", source_text="Hello", translated_text="Привет"
    )
    store.save()
    assert TranslationCheckpointStore(**params, settings_fingerprint="new").blocks == []
    assert (
        len(TranslationCheckpointStore(**params, settings_fingerprint="old").blocks)
        == 1
    )


@pytest.mark.parametrize(
    "identity", [None, {"target_lang": "fr"}, {"document_sha256": "other"}]
)
def test_scanned_layer_rejects_missing_or_mismatched_identity(tmp_path, identity):
    path = tmp_path / "layer.json"
    path.write_text(
        json.dumps(
            {
                "cache_identity": identity,
                "pages": [
                    {
                        "page_index": 0,
                        "blocks": [
                            {
                                "stable_id": "id",
                                "text": "Hello",
                                "translated_text": "WRONG",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        load_layer_mappings(
            path, lambda _m: None, {"target_lang": "ru", "document_sha256": "hash"}
        ).by_text
        == {}
    )
