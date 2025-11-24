# Daru DWG Translator

Приложение для автоматической локализации DWG/DXF-чертежей. Инструмент извлекает весь текст из исходного DWG (или DXF), переводит его выбранным движком и сохраняет результат в нужном формате (по умолчанию DWG). Доступны графический интерфейс на PySide6 и CLI-пайплайн, поддерживающие Google Translate, DeepL и OpenAI.

## Возможности

- Поддержка DWG и DXF: входной DWG автоматически конвертируется во временный DXF, а результат можно получить обратно в DWG или в DXF.
- Извлечение всех текстовых сущностей и просмотр их частот.
- Движки перевода: `google`, `deep_google`, `googletrans`, `deepl`, `chatgpt` (OpenAI) и `noop`.
- OpenAI GPT-5 модели с управлением строгостью через `verbosity`/`effort` вместо температуры.
- Сохранение карты соответствий в CSV и TXT-файлы с оригиналами/переводами.
- Перевод PDF накладывается поверх оригинальных страниц в виде редактируемых FreeText-слоёв; дополнительно сохраняется JSON с координатами и стилями каждого блока для ручной правки.
- Графический интерфейс с полосатым логом, поддержкой drag & drop и настройками API.

## Требования

- Python 3.9+
- Зависимости из `requirements-gui.txt`:
  - PySide6, pyinstaller, ezdxf, deep-translator, googletrans==4.0.0-rc1, deepl, openai
- Для перевода PDF/сканов понадобятся дополнительные Python-библиотеки (`pymupdf`, `pdf2image`, `pytesseract`, `PyPDF2`) и нативные утилиты OCR:
  - Установите библиотеки через `pip install -r requirements-gui.txt` (именно pip, macOS Homebrew не добавляет их в Python окружение приложения).
  - Установите Poppler и Tesseract для macOS: `brew install poppler tesseract`. Затем при необходимости укажите путь к Poppler в переменной окружения `POPPLER_PATH`.
- [ODA File Converter](https://www.opendesign.com/guestfiles/oda_file_converter) в `PATH` — именно он обеспечивает конверсию DWG ↔ DXF через `ezdxf.addons.odafc`. Можно также задать переменные среды `ODA_FILE_CONVERTER` или `ODAFC_PATH`, указывающие на `ODAFileConverter.exe`.

Установка зависимостей (рекомендуется виртуальное окружение):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-gui.txt
```

## Настройка API

- **OpenAI**: задайте `OPENAI_API_KEY` (и при необходимости `OPENAI_BASE_URL`) в настройках GUI или через переменные окружения. Для GPT-5 доступны параметры `verbosity` или `effort` с уровнем 0–100.
- Если ваш API-ключ привязан к конкретному проекту OpenAI, укажите его идентификатор через `OPENAI_PROJECT` (или в настройках GUI / флагом CLI). Это гарантирует, что запросы появятся в разделе [https://platform.openai.com/logs](https://platform.openai.com/logs).
- **DeepL**: укажите `DEEPL_AUTH_KEY`/`DEEPL_API_KEY` в настройках или окружении.
- Все запросы к OpenAI выполняются через Responses API с флагом `store=True`, поэтому они автоматически попадают в раздел [https://platform.openai.com/logs](https://platform.openai.com/logs). Если используется кастомный `OPENAI_BASE_URL`, проверяйте логи в соответствующем окружении или прокси.

## Запуск GUI

```bash
python3 daru_gui.py
```

В интерфейсе перетащите DWG/DXF, выберите формат результата (по умолчанию DWG), заполните опциональные пути для CSV/TXT и сохраните ключи API. Лог отображает ход обработки с обновлением статуса и цветовым выделением финальных сообщений.

## CLI-пайплайн

Пример обработки с сохранением в DWG:

```bash
python3 auto_translate_dxf.py INPUT.dwg OUTPUT_ru.dwg \
    --translator chatgpt \
    --output-format dwg \
    --map-csv OUTPUT_map.csv \
    --extracted-txt OUTPUT_texts.txt \
    --translated-txt OUTPUT_texts_ru.txt \
    --openai-strict-mode verbosity \
    --openai-strict-value 0.6
```

Полезные флаги:

- `--translator` — выбор движка перевода.
- `--output-format` — целевой формат (`dwg` по умолчанию, можно указать `dxf`).
- `--openai-project` — передача идентификатора проекта OpenAI, чтобы запросы попадали в нужный раздел логов.
- `--no-map`, `--skip-txt` — отключают генерацию CSV и TXT.
- `--openai-strict-mode` / `--openai-strict-value` — контроль strictness для GPT-5.

## Сборка standalone

PyInstaller-спека уже настроена. Перед сборкой убедитесь, что ODA File Converter доступен в `PATH`, затем выполните:

```bash
pyinstaller daru_gui.spec
```

Собранные артефакты появятся в папке `dist/`.

## Полезные файлы

- `auto_translation.py` — ядро движков перевода и нормализации текста.
- `auto_translate_dxf.py` — пайплайн DWG/DXF → перевод → DWG/DXF.
- `apply_dxf_translation.py` — применение перевода к DXF.
- `extract_dxf_texts.py` — извлечение и сортировка текстов.

## Обратная связь

CLI и GUI выводят подробные сообщения об ошибках. При сбоях убедитесь, что ключи API заданы, ODA File Converter установлен и путь к нему доступен.
