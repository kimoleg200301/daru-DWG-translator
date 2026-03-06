# Daru DWG Translator — Project Guide

## Overview

Daru is a Python application for automatic localization of DWG/DXF engineering drawings and PDF documents. It supports multiple translation backends (Google Translate, DeepL, OpenAI GPT-5) and provides both a PySide6 GUI and a CLI pipeline.

## Architecture

The project uses a `src/` layout with the main package at `src/daru/`.

### Package structure

- `daru.config` — `AppSettings` dataclass, `SettingsManager` (JSON persistence), UI constants (language/model/translator choices)
- `daru.utils.io` — CSV/TXT file I/O, path helpers (`ensure_parent`, `paths_equal`, `read_map_csv`, `write_map_csv`)
- `daru.utils.odafc` — ODA File Converter integration for DWG <-> DXF conversion
- `daru.utils.spinner` — CLI spinner and colorized output
- `daru.translation.engine` — `TranslationEngine` class with Google/DeepL/OpenAI backends, `chunked()`, text prepare/recover helpers
- `daru.translation.legacy` — `LegacyTranslationEngine` subclass
- `daru.dxf.entities` — Shared DXF entity helpers (deduplicated from extractor and applier): dimension text, mtext, table cells, mleader text, `ENTITY_TARGETS`
- `daru.dxf.extractor` — `extract_texts()` from DXF entities
- `daru.dxf.applier` — Apply translation pairs to DXF entities
- `daru.dxf.pipeline` — `translate_dxf()` full pipeline, CLI `parse_args()`/`main()`
- `daru.pdf._core` — Original monolithic PDF pipeline (~3300 lines): OCR, Textract, region detection, rendering, export
- `daru.pdf.pipeline` — Re-exports from `_core`
- `daru.pdf.config` — `PdfProcessingConfig`
- `daru.pdf.models` — PDF dataclasses (`WordBox`, `TextBlock`, `MergedRegion`, etc.)
- `daru.gui.app` — `main()` entry point (QApplication)
- `daru.gui.main_window` — `MainWindow` widget
- `daru.gui.settings_dialog` — API settings dialog
- `daru.gui.pdf_dialogs` — `PdfTextractSettingsDialog`, `PdfProcessingDialog`
- `daru.gui.worker` — `TranslateWorker` (QThread)

### Entry points

- `python daru_gui.py` or `python -m daru` — GUI
- `python auto_translate_dxf.py` — CLI for DXF/DWG
- After `pip install -e .`: `daru-gui`, `daru-translate`

### Key dependencies

PySide6, ezdxf, deep-translator, googletrans, deepl, openai, boto3, pymupdf, pdf2image, pytesseract, Pillow, opencv-python, easyocr

## Development

### Running tests

```bash
python -m pytest tests/ -v
```

Tests cover: module imports (smoke), config/settings, file I/O utils, DXF entity helpers, translation engine utilities.

### Code conventions

- Python 3.10+ with type hints
- Relative imports within the `daru` package (`from ..translation.engine import ...`)
- GUI uses Russian labels (the app UI is in Russian)
- Log messages in Russian
- `_core.py` in `daru.pdf` is the original monolithic module — treat with care, avoid large refactors without full test coverage
- DXF entity helpers are shared via `daru.dxf.entities` — do not duplicate between extractor and applier
- `TranslationEngine` handles provider initialization in `_init_translator()` with fallback chains
- OpenAI integration uses Responses API for GPT-5 models and Chat Completions API for older models

### Adding a new translation provider

1. Add provider name to `TRANSLATOR_CHOICES` in `daru/config.py`
2. Add initialization branch in `TranslationEngine._init_translator()` in `daru/translation/engine.py`
3. Implement `_<provider>_translate_batch()` method in the same class

### Adding a new DXF entity type

1. Add entity type string to `ENTITY_TARGETS` in `daru/dxf/entities.py`
2. Add extraction logic in `daru/dxf/extractor.py`
3. Add apply logic in `daru/dxf/applier.py`

## Important notes

- Root-level `daru_gui.py` and `auto_translate_dxf.py` are thin wrappers that add `src/` to `sys.path` and call into the package. All logic lives in `src/daru/`.
- The PDF pipeline (`_core.py`) is large and tightly coupled. It imports from `daru.translation.engine` directly. Gradual decomposition is planned but not yet done.
- ODA File Converter must be installed separately and available in PATH for DWG support.
- Settings are persisted to `~/.daru_gui_settings.json`.
