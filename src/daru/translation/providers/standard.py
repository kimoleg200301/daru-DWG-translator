"""Google HTTP and DeepL adapters with explicit failure semantics."""

import json
import urllib.parse
import urllib.request
from typing import Any, Sequence

from ..batching import chunked
from ..contracts import TranslationBatchError, validate_translations
from ..job import check_cancelled


def extract_google_translation(payload: Any) -> str | None:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
        return None
    parts = [
        segment[0]
        for segment in payload[0]
        if isinstance(segment, list) and segment and isinstance(segment[0], str)
    ]
    return "".join(parts) or None


class GoogleHttpProvider:
    def __init__(self, source_lang: str, target_lang: str) -> None:
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, texts: Sequence[str]) -> list[str]:
        params = {
            "client": "gtx",
            "sl": self.source_lang.lower(),
            "tl": self.target_lang.lower(),
            "dt": "t",
        }
        result = []
        for text in texts:
            check_cancelled()
            query = urllib.parse.urlencode({**params, "q": text})
            url = f"https://translate.googleapis.com/translate_a/single?{query}"
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                check_cancelled()
                raise TranslationBatchError(f"Ошибка перевода Google: {exc}") from exc
            result.append(extract_google_translation(payload))
        return validate_translations(texts, result, "Google")


class DeepLProvider:
    def __init__(self, client: Any, source_lang: str, target_lang: str) -> None:
        self.client = client
        self.source = None if source_lang.lower() == "auto" else source_lang.upper()
        self.target = target_lang.upper()

    def translate(self, texts: Sequence[str]) -> list[str]:
        result = []
        for batch in chunked(texts, 40):
            check_cancelled()
            try:
                values = self.client.translate_text(
                    list(batch), target_lang=self.target, source_lang=self.source
                )
            except Exception as exc:
                check_cancelled()
                raise TranslationBatchError(f"Ошибка перевода DeepL: {exc}") from exc
            if not isinstance(values, list):
                values = [values]
            result.extend(
                validate_translations(
                    batch, [getattr(value, "text", None) for value in values], "DeepL"
                )
            )
        return result
