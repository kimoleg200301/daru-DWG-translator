"""Tests for daru.config — AppSettings and SettingsManager."""

import json
from pathlib import Path

from daru.config import AppSettings, SettingsManager, LANGUAGE_CHOICES, TRANSLATOR_CHOICES


class TestAppSettings:
    def test_defaults(self):
        s = AppSettings()
        assert s.translator_name == "google"
        assert s.source_lang == "en"
        assert s.target_lang == "ru"
        assert s.output_format == "dwg"

    def test_to_dict_roundtrip(self):
        s = AppSettings(translator_name="deepl", target_lang="de")
        d = s.to_dict()
        assert d["translator_name"] == "deepl"
        assert d["target_lang"] == "de"
        restored = AppSettings.from_dict(d)
        assert restored.translator_name == "deepl"
        assert restored.target_lang == "de"

    def test_from_dict_ignores_unknown_keys(self):
        s = AppSettings.from_dict({"translator_name": "chatgpt", "unknown_key": 42})
        assert s.translator_name == "chatgpt"

    def test_from_dict_preserves_defaults(self):
        s = AppSettings.from_dict({"target_lang": "fr"})
        assert s.target_lang == "fr"
        assert s.source_lang == "en"  # default preserved


class TestSettingsManager:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "settings.json"
        mgr = SettingsManager(path=path)
        mgr.update(translator_name="deepl", target_lang="de")

        mgr2 = SettingsManager(path=path)
        assert mgr2.data.translator_name == "deepl"
        assert mgr2.data.target_lang == "de"

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        mgr = SettingsManager(path=path)
        assert mgr.data.translator_name == "google"  # defaults

    def test_load_corrupted_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        mgr = SettingsManager(path=path)
        assert mgr.data.translator_name == "google"  # falls back to defaults

    def test_update_only_known_keys(self, tmp_path):
        path = tmp_path / "settings.json"
        mgr = SettingsManager(path=path)
        mgr.update(translator_name="chatgpt", fake_key="ignored")
        assert mgr.data.translator_name == "chatgpt"
        assert not hasattr(mgr.data, "fake_key")


class TestConstants:
    def test_language_choices_not_empty(self):
        assert len(LANGUAGE_CHOICES) > 0
        assert "en" in LANGUAGE_CHOICES
        assert "ru" in LANGUAGE_CHOICES

    def test_translator_choices(self):
        assert "google" in TRANSLATOR_CHOICES
        assert "deepl" in TRANSLATOR_CHOICES
        assert "chatgpt" in TRANSLATOR_CHOICES
