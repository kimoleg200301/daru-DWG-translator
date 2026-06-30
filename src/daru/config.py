"""Application-wide constants, settings dataclass, and settings manager."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

try:
    from PySide6.QtWidgets import QComboBox
except ImportError:
    QComboBox = None  # type: ignore[assignment,misc]

SETTINGS_PATH = Path.home() / ".daru_gui_settings.json"

LANGUAGE_CHOICES = [
    "auto",
    "en",
    "ru",
    "de",
    "fr",
    "es",
    "it",
    "pl",
    "uk",
    "zh",
    "ja",
    "ko",
]

STYLE_FONT_CHOICES = [
    "Arial.ttf",
    "ArialUnicode.ttf",
    "Roboto-Regular.ttf",
    "NotoSans-Regular.ttf",
    "DejaVuSans.ttf",
]
ORIGINAL_FONT_VALUE = "original"
ORIGINAL_FONT_LABEL = "Оригинал"

OPENAI_DEFAULT_MODEL = "gpt-5.4-mini"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

OPENAI_MODEL_CHOICES = [
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

_LATEST_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")
_GPT51_REASONING_EFFORTS = ("none", "low", "medium", "high")
_PRO_REASONING_EFFORTS = ("medium", "high", "xhigh")
_LEGACY_GPT5_REASONING_EFFORTS = ("minimal", "low", "medium", "high")

OPENAI_MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "gpt-5.5": {
        "reasoning_efforts": _LATEST_REASONING_EFFORTS,
        "default_reasoning_effort": "medium",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5.5-pro": {
        "reasoning_efforts": _PRO_REASONING_EFFORTS,
        "default_reasoning_effort": "medium",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5.4": {
        "reasoning_efforts": _LATEST_REASONING_EFFORTS,
        "default_reasoning_effort": "none",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5.4-pro": {
        "reasoning_efforts": _PRO_REASONING_EFFORTS,
        "default_reasoning_effort": "medium",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5.4-mini": {
        "reasoning_efforts": _LATEST_REASONING_EFFORTS,
        "default_reasoning_effort": "low",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5.4-nano": {
        "reasoning_efforts": _LATEST_REASONING_EFFORTS,
        "default_reasoning_effort": "low",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5.2": {
        "reasoning_efforts": _LATEST_REASONING_EFFORTS,
        "default_reasoning_effort": "none",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5.1": {
        "reasoning_efforts": _GPT51_REASONING_EFFORTS,
        "default_reasoning_effort": "none",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5": {
        "reasoning_efforts": _LEGACY_GPT5_REASONING_EFFORTS,
        "default_reasoning_effort": "medium",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5-mini": {
        "reasoning_efforts": _LEGACY_GPT5_REASONING_EFFORTS,
        "default_reasoning_effort": "medium",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-5-nano": {
        "reasoning_efforts": _LEGACY_GPT5_REASONING_EFFORTS,
        "default_reasoning_effort": "medium",
        "supports_verbosity": True,
        "supports_temperature": False,
    },
    "gpt-4.1": {
        "reasoning_efforts": (),
        "default_reasoning_effort": "",
        "supports_verbosity": False,
        "supports_temperature": True,
    },
    "gpt-4.1-mini": {
        "reasoning_efforts": (),
        "default_reasoning_effort": "",
        "supports_verbosity": False,
        "supports_temperature": True,
    },
    "gpt-4o-mini": {
        "reasoning_efforts": (),
        "default_reasoning_effort": "",
        "supports_verbosity": False,
        "supports_temperature": True,
    },
}

OPENAI_REASONING_MODELS = {
    model for model, profile in OPENAI_MODEL_PROFILES.items() if profile["reasoning_efforts"]
}

OPENAI_BASE_URL_CHOICES = [OPENAI_DEFAULT_BASE_URL]


def get_openai_model_profile(model: str) -> Dict[str, Any]:
    model_key = (model or "").strip().lower()
    profile = OPENAI_MODEL_PROFILES.get(model_key)
    if profile is not None:
        return dict(profile)
    for base_model in sorted(OPENAI_MODEL_PROFILES, key=len, reverse=True):
        if model_key.startswith(f"{base_model}-"):
            return dict(OPENAI_MODEL_PROFILES[base_model])
    if model_key.startswith(("gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1")):
        return {
            "reasoning_efforts": _LATEST_REASONING_EFFORTS,
            "default_reasoning_effort": "medium",
            "supports_verbosity": True,
            "supports_temperature": False,
        }
    if model_key.startswith("gpt-5"):
        return {
            "reasoning_efforts": _LEGACY_GPT5_REASONING_EFFORTS,
            "default_reasoning_effort": "medium",
            "supports_verbosity": True,
            "supports_temperature": False,
        }
    return {
        "reasoning_efforts": (),
        "default_reasoning_effort": "",
        "supports_verbosity": False,
        "supports_temperature": True,
    }


def normalize_openai_base_url(value: str) -> str:
    normalized = (value or "").strip().rstrip("/")
    if not normalized:
        return OPENAI_DEFAULT_BASE_URL
    for endpoint in ("/chat/completions", "/responses"):
        if normalized.lower().endswith(endpoint):
            normalized = normalized[: -len(endpoint)].rstrip("/")
            break
    if normalized.lower() == "https://api.openai.com":
        return OPENAI_DEFAULT_BASE_URL
    return normalized

OUTPUT_FORMAT_CHOICES = ["dwg", "dxf"]

PDF_TYPE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("Отсканированный", "scanned"),
    ("Текстовый слой", "native"),
)

TRANSLATOR_CHOICES = [
    "google",
    "deepl",
    "chatgpt",
    "codex",
    "noop",
]

TRANSLATOR_ALIASES = {
    "deep_google": "google",
    "googletrans": "google",
}


def normalize_translator_name(value: str) -> str:
    normalized = (value or "google").strip().lower()
    return TRANSLATOR_ALIASES.get(normalized, normalized)


def normalize_style_font(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized.casefold() in {ORIGINAL_FONT_VALUE, ORIGINAL_FONT_LABEL.casefold()}:
        return ORIGINAL_FONT_VALUE
    return normalized


def populate_combo(
    combo: "QComboBox",
    options: Iterable[str],
    current: str = "",
    allow_empty: bool = False,
    editable: bool = True,
) -> None:
    combo.blockSignals(True)
    combo.clear()
    entries = list(options)
    seen = set()
    if allow_empty:
        combo.addItem("")
        seen.add("")
    for option in entries:
        if option and option not in seen:
            combo.addItem(option)
            seen.add(option)
    if current and current not in seen:
        combo.addItem(current)
        seen.add(current)
    combo.setEditable(editable)
    default_text = current or ("" if allow_empty else (entries[0] if entries else ""))
    combo.setCurrentText(default_text)
    combo.blockSignals(False)


@dataclass
class AppSettings:
    translator_name: str = "google"
    codex_enabled: bool = False
    codex_cli_path: str = ""
    codex_model: str = "gpt-5.4-mini"
    codex_reasoning_effort: str = "low"
    codex_analysis_model: str = "gpt-5.5"
    codex_analysis_reasoning_effort: str = "high"
    codex_timeout_seconds: int = 300
    source_lang: str = "en"
    target_lang: str = "ru"
    style_font: str = "DejaVuSans.ttf"
    save_pdf_layer: bool = True
    pdf_layer_path: str = ""
    pdf_dpi: int = 400
    pdf_min_confidence: int = 60
    pdf_blur_kernel: int = 23
    pdf_dilation_kernel: int = 3
    pdf_ocr_languages: str = "eng"
    pdf_processing_mode: str = "textract"
    pdf_vision_backend: str = "openai_api"
    pdf_vision_model: str = "gpt-5.5"
    pdf_vision_reasoning_effort: str = "medium"
    pdf_vision_api_key: str = ""
    pdf_vision_base_url: str = OPENAI_DEFAULT_BASE_URL
    pdf_vision_project: str = ""
    pdf_vision_codex_cli_path: str = ""
    pdf_vision_codex_timeout_seconds: int = 300
    pdf_vision_request_mode: str = "batched"
    pdf_vision_image_quality: str = "stable"
    textract_region: str = "us-east-1"
    textract_access_key: str = ""
    textract_secret_key: str = ""
    textract_session_token: str = ""
    deepl_key: str = ""
    openai_key: str = ""
    openai_model: str = OPENAI_DEFAULT_MODEL
    openai_base_url: str = OPENAI_DEFAULT_BASE_URL
    openai_project: str = ""
    openai_temperature: float = 0.2
    openai_reasoning_effort: str = "low"
    openai_verbosity: str = "low"
    openai_strict_mode: str = "verbosity"
    openai_strict_value: float = 0.5
    output_format: str = "dwg"
    save_map: bool = True
    save_txt: bool = True
    last_directory: str = str(Path.home())

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        defaults = cls()
        for key, value in data.items():
            if hasattr(defaults, key):
                setattr(defaults, key, value)

        defaults.translator_name = normalize_translator_name(defaults.translator_name)
        defaults.style_font = normalize_style_font(defaults.style_font)
        if defaults.translator_name == "codex" and "codex_enabled" not in data:
            defaults.codex_enabled = True
            defaults.translator_name = "google"
        defaults.openai_base_url = normalize_openai_base_url(defaults.openai_base_url)
        defaults.pdf_vision_base_url = normalize_openai_base_url(defaults.pdf_vision_base_url)
        pdf_vision_backend = str(
            defaults.pdf_vision_backend or "openai_api"
        ).strip().lower()
        defaults.pdf_vision_backend = (
            pdf_vision_backend
            if pdf_vision_backend in {"openai_api", "codex_cli"}
            else "openai_api"
        )
        pdf_vision_request_mode = str(
            defaults.pdf_vision_request_mode or "batched"
        ).strip().lower()
        defaults.pdf_vision_request_mode = (
            pdf_vision_request_mode
            if pdf_vision_request_mode in {"batched", "single_page"}
            else "batched"
        )
        pdf_vision_image_quality = str(
            defaults.pdf_vision_image_quality or "stable"
        ).strip().lower()
        defaults.pdf_vision_image_quality = (
            pdf_vision_image_quality
            if pdf_vision_image_quality in {"stable", "high", "original"}
            else "stable"
        )
        try:
            defaults.codex_timeout_seconds = max(10, int(defaults.codex_timeout_seconds))
        except (TypeError, ValueError):
            defaults.codex_timeout_seconds = 300
        try:
            defaults.pdf_vision_codex_timeout_seconds = max(
                10, int(defaults.pdf_vision_codex_timeout_seconds)
            )
        except (TypeError, ValueError):
            defaults.pdf_vision_codex_timeout_seconds = 300
        legacy_level = _legacy_openai_level(defaults.openai_strict_value)
        if "openai_reasoning_effort" not in data and defaults.openai_strict_mode == "effort":
            defaults.openai_reasoning_effort = legacy_level
        if "openai_verbosity" not in data and defaults.openai_strict_mode == "verbosity":
            defaults.openai_verbosity = legacy_level
        return defaults


def _legacy_openai_level(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.5
    if numeric <= 0.34:
        return "low"
    if numeric <= 0.67:
        return "medium"
    return "high"


class SettingsManager:
    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self.path = path
        self._settings = AppSettings()
        self.load()

    @property
    def data(self) -> AppSettings:
        return self._settings

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        self._settings = AppSettings.from_dict(raw)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(self._settings.to_dict(), fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        self.save()
