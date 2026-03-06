# Daru DWG Translator

Приложение для автоматической локализации DWG/DXF-чертежей и PDF-документов. Инструмент извлекает текст из исходного файла, переводит его выбранным движком и сохраняет результат в нужном формате. Доступны графический интерфейс на PySide6 и CLI-пайплайн.

## Возможности

- **DWG/DXF**: входной DWG автоматически конвертируется во временный DXF через ODA File Converter, а результат можно получить обратно в DWG или DXF.
- **PDF**: перевод накладывается поверх оригинальных страниц в виде редактируемых FreeText-слоев; дополнительно сохраняется JSON с координатами и стилями каждого блока. Поддерживаются Tesseract, EasyOCR и AWS Textract.
- **Движки перевода**: `google`, `deep_google`, `googletrans`, `deepl`, `chatgpt` (OpenAI) и `noop`.
- **OpenAI GPT-5**: поддержка Responses API с управлением строгостью через `verbosity`/`effort`.
- **Кэширование**: карта соответствий в CSV и TXT-файлы с оригиналами/переводами.
- **GUI**: полосатый лог, drag & drop файлов, настройки API, диалоги обработки PDF.

## Структура проекта

```
daru/
├── src/daru/                   # Основной пакет
│   ├── config.py               # AppSettings, SettingsManager, константы UI
│   ├── __main__.py             # python -m daru
│   ├── utils/
│   │   ├── io.py               # CSV/TXT чтение/запись, работа с путями
│   │   ├── odafc.py            # Интеграция с ODA File Converter
│   │   └── spinner.py          # CLI-спиннер и цветной вывод
│   ├── translation/
│   │   ├── engine.py           # TranslationEngine (Google/DeepL/OpenAI)
│   │   └── legacy.py           # LegacyTranslationEngine
│   ├── dxf/
│   │   ├── entities.py         # Общие хелперы для DXF-сущностей
│   │   ├── extractor.py        # Извлечение текстов из DXF
│   │   ├── applier.py          # Применение переводов к DXF
│   │   └── pipeline.py         # translate_dxf(), CLI точка входа
│   ├── pdf/
│   │   ├── _core.py            # Основная логика обработки PDF
│   │   ├── pipeline.py         # Re-exports из _core
│   │   ├── config.py           # PdfProcessingConfig
│   │   └── models.py           # Dataclasses (WordBox, TextBlock и др.)
│   └── gui/
│       ├── app.py              # main() — точка входа GUI
│       ├── main_window.py      # MainWindow
│       ├── settings_dialog.py  # Диалог настроек API
│       ├── pdf_dialogs.py      # Диалоги обработки PDF
│       └── worker.py           # TranslateWorker (QThread)
├── tests/                      # Тесты (pytest)
├── daru_gui.py                 # Обертка запуска GUI
├── auto_translate_dxf.py       # Обертка запуска CLI
├── pyproject.toml              # Метаданные проекта и entry points
└── requirements-gui.txt        # Зависимости
```

## Требования

- Python 3.10+
- Зависимости из `requirements-gui.txt`
- [ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter) в `PATH` для конверсии DWG <-> DXF. Можно задать переменные `ODA_FILE_CONVERTER` или `ODAFC_PATH`.
- Для PDF: Poppler и Tesseract. На macOS: `brew install poppler tesseract`.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-gui.txt
```

Или через pyproject.toml (editable install):

```bash
pip install -e .
```

## Запуск

### GUI

```bash
python daru_gui.py
# или
python -m daru
# или (после pip install -e .)
daru-gui
```

В интерфейсе перетащите DWG/DXF/PDF, выберите формат результата, заполните ключи API в настройках.

### CLI (DXF/DWG)

```bash
python auto_translate_dxf.py INPUT.dwg OUTPUT_ru.dwg \
    --translator chatgpt \
    --output-format dwg \
    --map-csv OUTPUT_map.csv \
    --extracted-txt OUTPUT_texts.txt \
    --translated-txt OUTPUT_texts_ru.txt \
    --openai-strict-mode verbosity \
    --openai-strict-value 0.6
```

Или после установки:

```bash
daru-translate INPUT.dwg OUTPUT_ru.dwg --translator google
```

### Флаги CLI

| Флаг | Описание |
|------|----------|
| `--translator` | Движок перевода (`google`, `deepl`, `chatgpt`, `noop`) |
| `--output-format` | Формат выхода: `dwg` (по умолчанию) или `dxf` |
| `--source-lang` | Исходный язык (по умолчанию `en`) |
| `--target-lang` | Целевой язык (по умолчанию `ru`) |
| `--map-csv` | Путь для CSV карты переводов |
| `--no-map` | Не создавать CSV |
| `--skip-txt` | Не создавать TXT |
| `--openai-project` | ID проекта OpenAI для логирования |
| `--openai-strict-mode` | `verbosity` или `effort` для GPT-5 |
| `--openai-strict-value` | Значение 0.0-1.0 |

## Настройка API

- **OpenAI**: задайте `OPENAI_API_KEY` (и при необходимости `OPENAI_BASE_URL`, `OPENAI_PROJECT`) в настройках GUI или через переменные окружения.
- **DeepL**: укажите `DEEPL_AUTH_KEY` или `DEEPL_API_KEY`.
- Все запросы к OpenAI выполняются через Responses API с `store=True` и попадают в [логи OpenAI](https://platform.openai.com/logs).

## Тестирование

```bash
pip install pytest
python -m pytest tests/ -v
```

## Сборка standalone

```bash
pip install pyinstaller
pyinstaller daru_gui.spec
```

Артефакты появятся в `dist/`.
