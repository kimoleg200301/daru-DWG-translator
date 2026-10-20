import json
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from daru.config import AppSettings, SettingsManager
from daru.docx.pipeline import _load_document_model, _write_atomic_docx
from daru.utils.paths import migrate_pdf_cache, user_data_dir


@pytest.mark.parametrize("data", [[], None, "settings", 42])
def test_settings_require_json_object(tmp_path, data):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert SettingsManager(path).data == AppSettings()


def test_settings_reject_wrong_types_nonfinite_numbers_and_method_names():
    settings = AppSettings.from_dict(
        {
            "to_dict": "corrupted",
            "save_map": "false",
            "source_lang": [],
            "target_lang": {},
            "openai_temperature": float("nan"),
            "pdf_dpi": -1,
            "codex_timeout_seconds": "120",
            "pdf_min_confidence": 500,
        }
    )
    assert callable(settings.to_dict)
    assert settings.save_map is True
    assert settings.source_lang == "en" and settings.target_lang == "ru"
    assert settings.openai_temperature == 0.2
    assert settings.pdf_dpi == 72 and settings.pdf_min_confidence == 100
    assert settings.codex_timeout_seconds == 120


def test_settings_failed_replace_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    settings = SettingsManager(path)
    settings.save()
    original = path.read_bytes()

    def fail(*args):
        raise OSError("simulated disk error")

    monkeypatch.setattr("daru.utils.io.os.replace", fail)
    settings.update(target_lang="fr")
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_utf8_settings_fall_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_bytes(b"\xff\xfeinvalid utf8")
    assert SettingsManager(path).data == AppSettings()


def test_user_data_directory_override_does_not_create_files(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    monkeypatch.setenv("DARU_DATA_DIR", str(root))
    assert user_data_dir() == root
    assert not root.exists()


def test_cache_migration_keeps_legacy_and_existing_files(tmp_path):
    legacy, current = tmp_path / "old", tmp_path / "new"
    legacy.mkdir()
    current.mkdir()
    (legacy / "textract_requests.log").write_text("OCR", encoding="utf-8")
    (legacy / "drawing.translation.json").write_text("old", encoding="utf-8")
    (legacy / "unrelated.json").write_text("private", encoding="utf-8")
    (current / "drawing.translation.json").write_text("current", encoding="utf-8")
    migrate_pdf_cache(legacy, current, lambda message: None)
    assert (legacy / "textract_requests.log").read_text() == "OCR"
    assert (current / "textract_requests.log").read_text() == "OCR"
    assert (current / "drawing.translation.json").read_text() == "current"
    assert {path.name for path in current.iterdir()} == {
        "drawing.translation.json",
        "textract_requests.log",
    }


def test_docx_binary_members_are_streamed_and_preserved(tmp_path, monkeypatch):
    source, output = tmp_path / "source.docx", tmp_path / "output.docx"
    media = b"binary drawing" * 100000
    xml = b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Inspect the valve</w:t></w:r></w:p></w:body></w:document>'
    with ZipFile(source, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
        archive.writestr("word/media/image.png", media)
    read = ZipFile.read

    def guard(self, name, *args, **kwargs):
        assert name != "word/media/image.png", "binary attachment was materialized"
        return read(self, name, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(ZipFile, "read", guard)
        originals, model = _load_document_model(source, "en")
        assert model.units
        assert set(originals) == {"word/document.xml"}
        _write_atomic_docx(source, output, originals, {})
    with ZipFile(output) as archive:
        assert archive.read("word/media/image.png") == media
        assert archive.read("word/document.xml") == xml
        assert archive.getinfo("word/media/image.png").compress_type == ZIP_DEFLATED
