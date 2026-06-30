"""Tests for daru.config — AppSettings and SettingsManager."""

import json
from pathlib import Path

from daru.config import (
    OPENAI_DEFAULT_BASE_URL,
    OPENAI_DEFAULT_MODEL,
    OPENAI_MODEL_CHOICES,
    ORIGINAL_FONT_VALUE,
    AppSettings,
    LANGUAGE_CHOICES,
    SettingsManager,
    TRANSLATOR_CHOICES,
    get_openai_model_profile,
    normalize_openai_base_url,
    normalize_style_font,
    normalize_translator_name,
)


class TestAppSettings:
    def test_defaults(self):
        s = AppSettings()
        assert s.translator_name == "google"
        assert s.source_lang == "en"
        assert s.target_lang == "ru"
        assert s.output_format == "dwg"
        assert s.openai_model == OPENAI_DEFAULT_MODEL
        assert s.openai_base_url == OPENAI_DEFAULT_BASE_URL
        assert not s.codex_enabled
        assert s.codex_model == "gpt-5.4-mini"
        assert s.codex_reasoning_effort == "low"
        assert s.codex_analysis_model == "gpt-5.5"
        assert s.codex_analysis_reasoning_effort == "high"
        assert s.codex_timeout_seconds == 300
        assert s.pdf_vision_backend == "openai_api"
        assert s.pdf_vision_model == "gpt-5.5"
        assert s.pdf_vision_reasoning_effort == "medium"
        assert s.pdf_vision_api_key == ""
        assert s.pdf_vision_base_url == OPENAI_DEFAULT_BASE_URL
        assert s.pdf_vision_project == ""
        assert s.pdf_vision_codex_cli_path == ""
        assert s.pdf_vision_codex_timeout_seconds == 300
        assert s.pdf_vision_request_mode == "batched"
        assert s.pdf_vision_image_quality == "stable"

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
        assert s.codex_analysis_model == "gpt-5.5"
        assert s.codex_analysis_reasoning_effort == "high"
        assert s.pdf_vision_model == "gpt-5.5"
        assert s.pdf_vision_backend == "openai_api"
        assert s.pdf_vision_request_mode == "batched"
        assert s.pdf_vision_image_quality == "stable"

    def test_from_dict_normalizes_pdf_vision_request_mode(self):
        assert (
            AppSettings.from_dict(
                {"pdf_vision_request_mode": "single_page"}
            ).pdf_vision_request_mode
            == "single_page"
        )
        assert (
            AppSettings.from_dict(
                {"pdf_vision_request_mode": "invalid"}
            ).pdf_vision_request_mode
            == "batched"
        )

    def test_from_dict_normalizes_pdf_vision_image_quality(self):
        assert (
            AppSettings.from_dict(
                {"pdf_vision_image_quality": "original"}
            ).pdf_vision_image_quality
            == "original"
        )
        assert (
            AppSettings.from_dict(
                {"pdf_vision_image_quality": "invalid"}
            ).pdf_vision_image_quality
            == "stable"
        )

    def test_from_dict_migrates_legacy_openai_settings(self):
        s = AppSettings.from_dict(
            {
                "openai_base_url": "https://api.openai.com/v1/responses",
                "openai_strict_mode": "effort",
                "openai_strict_value": 0.9,
            }
        )
        assert s.openai_base_url == OPENAI_DEFAULT_BASE_URL
        assert s.openai_reasoning_effort == "high"

    def test_from_dict_normalizes_localized_original_font(self):
        settings = AppSettings.from_dict({"style_font": "Оригинал"})

        assert settings.style_font == ORIGINAL_FONT_VALUE


class TestSettingsManager:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "settings.json"
        mgr = SettingsManager(path=path)
        mgr.update(
            translator_name="deepl",
            target_lang="de",
            codex_analysis_model="gpt-5.4",
            codex_analysis_reasoning_effort="xhigh",
        )

        mgr2 = SettingsManager(path=path)
        assert mgr2.data.translator_name == "deepl"
        assert mgr2.data.target_lang == "de"
        assert mgr2.data.codex_analysis_model == "gpt-5.4"
        assert mgr2.data.codex_analysis_reasoning_effort == "xhigh"

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
        assert TRANSLATOR_CHOICES == ["google", "deepl", "chatgpt", "codex", "noop"]

    def test_legacy_codex_translator_setting_enables_global_mode(self):
        settings = AppSettings.from_dict({"translator_name": "codex"})
        assert settings.codex_enabled
        assert settings.translator_name == "google"

    def test_legacy_google_translator_names_are_normalized(self):
        assert normalize_translator_name("deep_google") == "google"
        assert normalize_translator_name("googletrans") == "google"
        assert AppSettings.from_dict({"translator_name": "googletrans"}).translator_name == "google"

    def test_openai_models_are_current(self):
        assert OPENAI_MODEL_CHOICES[:6] == [
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.4-pro",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
        ]
        assert "gpt-5-chat-latest" not in OPENAI_MODEL_CHOICES
        assert "gpt-5-codex" not in OPENAI_MODEL_CHOICES

    def test_openai_model_profiles(self):
        latest = get_openai_model_profile("gpt-5.5")
        assert latest["reasoning_efforts"] == ("none", "low", "medium", "high", "xhigh")
        assert latest["supports_verbosity"]
        assert not latest["supports_temperature"]

        classic = get_openai_model_profile("gpt-4.1")
        assert not classic["reasoning_efforts"]
        assert not classic["supports_verbosity"]
        assert classic["supports_temperature"]

    def test_normalize_openai_base_url(self):
        assert normalize_openai_base_url("") == OPENAI_DEFAULT_BASE_URL
        assert (
            normalize_openai_base_url("https://api.openai.com/v1/responses")
            == OPENAI_DEFAULT_BASE_URL
        )
        assert (
            normalize_openai_base_url("https://proxy.example/v1/chat/completions")
            == "https://proxy.example/v1"
        )

    def test_normalize_style_font(self):
        assert normalize_style_font("Оригинал") == ORIGINAL_FONT_VALUE
        assert normalize_style_font("original") == ORIGINAL_FONT_VALUE
        assert normalize_style_font("Arial.ttf") == "Arial.ttf"
