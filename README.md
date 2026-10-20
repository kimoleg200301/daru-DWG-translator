# Daru Document Translator

Приложение для автоматической локализации DWG/DXF-чертежей, PDF- и DOCX-документов. Инструмент извлекает текст из исходного файла, переводит его выбранным движком и сохраняет результат в нужном формате. Доступны графический интерфейс на PySide6 и CLI-пайплайн для DWG/DXF.

## Возможности

- **DWG/DXF**: входной DWG автоматически конвертируется во временный DXF через ODA File Converter, а результат можно получить обратно в DWG или DXF.
- **PDF**: перевод сохраняется как редактируемый текст PDF; дополнительно сохраняется JSON с координатами и стилями каждого блока. Поддерживаются Tesseract, EasyOCR и AWS Textract.
- **DOCX**: прямой перевод текста OOXML без OCR с сохранением таблиц, стилей, изображений, гиперссылок, колонтитулов, текстовых блоков и служебных полей Word.
- **Движки перевода**: `google` (бесплатный Google через `deep-translator`), `deepl`, `chatgpt` (OpenAI API), `codex` (локальный Codex CLI) и `noop`.
- **OpenAI GPT-5**: Responses API с отдельными настройками `reasoning.effort` и `text.verbosity`.
- **Кэширование**: checkpoint и PDF-слои с проверкой исходного документа, языков и настроек перевода; экспорт карт в CSV и текстов в TXT.
- **GUI**: полосатый лог, drag & drop файлов, настройки API, диалоги обработки PDF и отмена задания.

## Структура проекта

```
daru/
├── src/daru/                   # Основной пакет
│   ├── config.py               # AppSettings, SettingsManager, константы UI
│   ├── __main__.py             # python -m daru
│   ├── utils/
│   │   ├── io.py               # CSV/TXT чтение/запись, работа с путями
│   │   ├── paths.py            # Пользовательский каталог данных и перенос кэша
│   │   ├── odafc.py            # Интеграция с ODA File Converter
│   │   └── spinner.py          # CLI-спиннер и цветной вывод
│   ├── translation/
│   │   ├── engine.py           # Совместимый фасад TranslationEngine, контекст документа
│   │   ├── providers/          # Адаптеры Google, DeepL, OpenAI и Codex CLI
│   │   ├── options.py          # Общие параметры и идентичность кэша
│   │   ├── contracts.py        # Контекст и проверка полноты ответов провайдеров
│   │   ├── job.py              # Кооперативная отмена задания
│   │   ├── progress.py         # Общие события этапов и счётчики элементов
│   │   ├── checkpoint.py       # Продолжение задания и совместимость с JSON
│   │   ├── checkpoint_sqlite.py # Транзакционная запись изменённых блоков
│   │   └── legacy.py           # LegacyTranslationEngine
│   ├── dxf/
│   │   ├── entities.py         # Общие хелперы для DXF-сущностей
│   │   ├── extractor.py        # Извлечение текстов из DXF
│   │   ├── applier.py          # Применение переводов к DXF
│   │   └── pipeline.py         # translate_dxf(), CLI точка входа
│   ├── pdf/
│   │   ├── _core.py            # Оркестрация PDF и совместимый фасад
│   │   ├── pipeline.py         # Re-exports из _core
│   │   ├── config.py           # Настройки PDF без загрузки OCR-библиотек
│   │   ├── raster.py           # Растеризация и последовательная загрузка страниц
│   │   ├── ocr.py              # Локальный OCR и анализ разметки
│   │   ├── textract.py         # AWS Textract и кэш ответов
│   │   ├── geometry.py         # Геометрия блоков и привязка текстовых рамок
│   │   ├── imaging.py          # Маски текста и оценка цветов
│   │   ├── regions.py          # Подготовка OCR-блоков к отрисовке
│   │   ├── block_translation.py # Перевод блоков и применение кэша
│   │   ├── rendering.py        # Очистка фона и растровое наложение текста
│   │   ├── export.py           # Атомарный экспорт PDF
│   │   ├── native.py           # Обработка встроенного текста PDF
│   │   ├── editable.py         # Редактируемый текст поверх сканированных страниц
│   │   ├── pages.py            # Временное хранение страниц без удержания всех пикселей
│   │   ├── layers.py           # Чтение и запись JSON-слоёв
│   │   └── models.py           # Независимые модели (WordBox, TextBlock и др.)
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
| `--map-csv` | Путь для экспорта CSV карты переводов |
| `--glossary-csv` | Явно выбранный CSV-глоссарий для текущего целевого языка |
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

## Продолжение и отмена заданий

Завершённые пакеты сохраняются в `~/.daru/checkpoints`. Кэш PDF находится в `~/.daru/cache/pdf`.
Новые checkpoint используют SQLite (`*.checkpoint.sqlite3`): каждый пакет записывает только изменённые блоки одной транзакцией.
Совместимый JSON с тем же именем автоматически переносится при продолжении; исходный JSON сохраняется.
Явно заданный путь с расширением `.json` сохраняет прежний формат. Сброс SQLite очищает сохранённые переводы и меняет поколение задания:
старый процесс не сможет дописать результаты после сброса, а оставшийся JSON не будет импортирован повторно.
Переменная окружения `DARU_DATA_DIR` меняет общий каталог данных (задайте её до запуска).
Известные файлы старого кэша `src/daru/pdf/cache` копируются при обработке PDF без удаления оригиналов и без замены уже существующих файлов.

Автоматическое продолжение проверяет SHA-256 документа, языковую пару, провайдера, модель и параметры перевода.
Старые checkpoint и слои без такой идентификации пропускаются. CSV из `--map-csv` больше не читается автоматически:
для подготовленного глоссария используйте `--glossary-csv glossary.csv` (столбцы `text_en,text_ru`, второй столбец содержит перевод на выбранный язык).
Изменение выбранного глоссария также сбрасывает совместимость checkpoint.

Кнопка «Отменить перевод» останавливает задание между страницами, пакетами и этапами.
Уже выполняющийся сетевой запрос или внешняя команда завершается либо достигает своего таймаута; следующие операции не запускаются.
При закрытии окна приложение запрашивает отмену и дожидается завершения потока. Ожидание подтверждения анализа прерывается сразу.
Строка состояния показывает этап обработки и число элементов текущего этапа. Успешное завершение сообщается после возврата конвейера;
завершение перевода текста ещё не означает завершение сохранения файла.

Для измерения стоимости checkpoint и проверки PDF без внешних API:

```powershell
python scripts/benchmark_checkpoints.py --blocks 10000 --format sqlite3
python scripts/benchmark_checkpoints.py --blocks 10000 --format json
python scripts/pdf_regression_fixture.py --output tmp/pdfs/layout-check
```

Библиотечный код может получать те же события через `daru.translation.progress.progress_scope(callback)`.
События не зависят от Qt; подписчик ограничен контекстом текущего задания.
Ошибка или неполный ответ провайдера не сохраняются как успешно переведённый пакет.

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
pip install -e ".[dev]"
python -m pytest tests/ -v
python -m ruff check src tests scripts
python -m vulture src tests --min-confidence 80
```

Замер локального применения DXF-переводов: `python scripts/benchmark_dxf_replacements.py`.
Он не включает обращения к API, файловый ввод/вывод и работу со шрифтами.

Замер пиковой памяти на синтетическом сканированном PDF без сетевых запросов:
`python scripts/benchmark_pdf_memory.py --pages 12`. Каждый размер документа запускайте отдельным процессом.
Основной перевод PDF обрабатывает изображения постранично с временными PNG; для них требуется свободное место во временном каталоге ОС.
Служебный `process_pdf()`, возвращающий список изображений вызывающему коду, сохраняет прежний контракт и удерживает этот список в памяти.

## Сборка standalone

```bash
pip install pyinstaller
pyinstaller daru_gui.spec
```

Артефакты появятся в `dist/`.
