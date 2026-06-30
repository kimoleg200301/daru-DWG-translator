# Daru Document Translator

Приложение для автоматической локализации DWG/DXF-чертежей, PDF- и DOCX-документов. Инструмент извлекает текст из исходного файла, переводит его выбранным движком и сохраняет результат в нужном формате. Доступны графический интерфейс на PySide6 и CLI-пайплайн для DWG/DXF.

## Возможности

- **DWG/DXF**: входной DWG автоматически конвертируется во временный DXF через ODA File Converter, а результат можно получить обратно в DWG или DXF.
- **PDF**: перевод накладывается поверх оригинальных страниц в виде редактируемых FreeText-слоев; дополнительно сохраняется JSON с координатами и стилями каждого блока. Поддерживаются Tesseract, EasyOCR и AWS Textract.
- **DOCX**: прямой перевод текста OOXML без OCR с сохранением таблиц, стилей, изображений, гиперссылок, колонтитулов, текстовых блоков и служебных полей Word.
- **Движки перевода**: `google` (бесплатный Google через `deep-translator`), `deepl`, `chatgpt` (OpenAI API), `codex` (локальный Codex CLI) и `noop`.
- **OpenAI GPT-5**: Responses API с отдельными настройками `reasoning.effort` и `text.verbosity`.
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
│   ├── docx/
│   │   ├── segments.py         # OOXML-сегментация и сохранение форматирования
│   │   └── pipeline.py         # translate_docx()
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

В интерфейсе перетащите DWG/DXF/PDF/DOCX, выберите формат результата, заполните ключи API в настройках.

### CLI (DXF/DWG)

```bash
python auto_translate_dxf.py INPUT.dwg OUTPUT_ru.dwg \
    --translator chatgpt \
    --output-format dwg \
    --map-csv OUTPUT_map.csv \
    --extracted-txt OUTPUT_texts.txt \
    --translated-txt OUTPUT_texts_ru.txt \
    --openai-model gpt-5.4-mini \
    --openai-reasoning-effort low \
    --openai-verbosity low
```

Или после установки:

```bash
daru-translate INPUT.dwg OUTPUT_ru.dwg --translator google
```

### Флаги CLI

| Флаг | Описание |
|------|----------|
| `--translator` | Движок перевода (`google`, `deepl`, `chatgpt`, `codex`, `noop`) |
| `--output-format` | Формат выхода: `dwg` (по умолчанию) или `dxf` |
| `--source-lang` | Исходный язык (по умолчанию `en`) |
| `--target-lang` | Целевой язык (по умолчанию `ru`) |
| `--map-csv` | Путь для CSV карты переводов |
| `--no-map` | Не создавать CSV |
| `--skip-txt` | Не создавать TXT |
| `--openai-project` | ID проекта OpenAI для логирования |
| `--openai-reasoning-effort` | `none`, `minimal`, `low`, `medium`, `high` или `xhigh` в зависимости от модели |
| `--openai-verbosity` | `low`, `medium` или `high` для GPT-5 |
| `--codex-cli-path` | Явный путь к `codex.exe`, `codex.cmd` или другой команде Codex CLI |
| `--codex-model` | Модель для `codex exec`, по умолчанию `gpt-5.4-mini` |
| `--codex-reasoning-effort` | `low`, `medium`, `high` или `xhigh` |
| `--codex-analysis-model` | Модель преданализа, по умолчанию `gpt-5.5` |
| `--codex-analysis-reasoning-effort` | Reasoning преданализа, по умолчанию `high` |
| `--codex-timeout-seconds` | Таймаут одного пакетного вызова Codex CLI |

## Настройка API

- **OpenAI**: задайте `OPENAI_API_KEY` (и при необходимости `OPENAI_BASE_URL`, `OPENAI_PROJECT`) в настройках GUI или через переменные окружения. Для официального API базовый URL: `https://api.openai.com/v1` без `/responses`.
- **DeepL**: укажите `DEEPL_AUTH_KEY` или `DEEPL_API_KEY`.
- Все запросы к OpenAI выполняются через Responses API с `store=True` и попадают в [логи OpenAI](https://platform.openai.com/logs).

## Codex CLI без API

Режим Codex CLI использует отдельную авторизацию установленной команды `codex` и не передаёт сохранённые в Daru ключи OpenAI или DeepL дочернему процессу.

1. В GUI откройте «Настройки перевода» и нажмите «Установить/обновить Codex CLI» либо установите Codex CLI вручную.
2. После установки завершите авторизацию в браузере, открытом командой `codex login`.
3. Включите «Использовать Codex CLI» и нажмите «Проверить Codex CLI».

Кнопка установки выполняет `powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"`, а затем запускает `codex login`. Для этого нужны PowerShell и доступ в интернет; Node.js/npm не требуется. Путь к установленному `codex.exe` подставляется в поле настроек. Каждый пакет запускается через `codex exec` в одноразовом временном каталоге, с `read-only` sandbox и обязательной JSON Schema. Codex CLI не включается в PyInstaller-сборку Daru.

Перед переводом Codex формирует общий анализ документа отдельной моделью. В GUI анализ открывается в редактируемом окне: перевод начинается только после нажатия «Начать перевод», а «Отмена» завершает задачу без записи результата. Для CLI сформированный анализ принимается автоматически.

Для CAD CLI используйте:

```bash
daru-translate INPUT.dxf OUTPUT_ru.dxf \
    --translator codex \
    --codex-model gpt-5.4-mini \
    --codex-reasoning-effort low \
    --codex-analysis-model gpt-5.5 \
    --codex-analysis-reasoning-effort high \
    --codex-timeout-seconds 300
```

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
