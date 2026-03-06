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

OPENAI_MODEL_CHOICES = [
    "gpt-5-chat-latest",
    "gpt-5-mini",
    "gpt-5-codex",
    "gpt-5-pro",
    "gpt-5",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]

OPENAI_REASONING_MODELS = {
    "gpt-5-chat-latest",
    "gpt-5-mini",
    "gpt-5-codex",
    "gpt-5-pro",
    "gpt-5",
}

OPENAI_BASE_URL_CHOICES = [
    "https://api.openai.com/v1",
]

OUTPUT_FORMAT_CHOICES = ["dwg", "dxf"]

PDF_TYPE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("Отсканированный", "scanned"),
    ("Не отсканированный", "native"),
)

TRANSLATOR_CHOICES = [
    "google",
    "deep_google",
    "googletrans",
    "deepl",
    "chatgpt",
    "noop",
]


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
    textract_region: str = "us-east-1"
    textract_access_key: str = ""
    textract_secret_key: str = ""
    textract_session_token: str = ""
    deepl_key: str = ""
    openai_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    openai_project: str = ""
    openai_temperature: float = 0.2
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
        defaults.__dict__.update(data)
        return defaults


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
