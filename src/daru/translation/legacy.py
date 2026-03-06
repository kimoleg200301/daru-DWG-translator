"""Legacy translation engine without message splitting."""

from .engine import TranslationEngine as _TranslationEngine


class LegacyTranslationEngine(_TranslationEngine):
    """Drop-in replacement without message splitting."""

    def _split_openai_payload(self, text: str, limit: int = 0):  # type: ignore[override]
        return [text]


def legacy_translate(texts, **kwargs):
    engine = LegacyTranslationEngine(**kwargs)
    return engine.translate_many(texts)
