"""
Legacy translation engine that keeps the previous behaviour of sending
OpenAI payloads without chunking or retry logic. Use this as a fallback
when the default engine becomes too defensive for specific jobs.
"""

from auto_translation import TranslationEngine as _TranslationEngine


class LegacyTranslationEngine(_TranslationEngine):
    """Drop-in replacement without message splitting."""

    def _split_openai_payload(self, text: str, limit: int = 0):  # type: ignore[override]
        return [text]


def legacy_translate(texts, **kwargs):
    engine = LegacyTranslationEngine(**kwargs)
    return engine.translate_many(texts)
