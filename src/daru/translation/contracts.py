"""Provider-independent batch and document context contracts."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass
class TranslationContext:
    originals: Sequence[str] = ()
    document_text: str | None = None
    label: str = "DOCUMENT"
    entity_types: Mapping[str, str] | None = None
    glossary: dict[str, str] = field(default_factory=dict)


class BatchProvider(Protocol):
    def translate(self, texts: Sequence[str]) -> list[str]: ...


class TranslationBatchError(RuntimeError):
    """A failed or incomplete batch must never enter a successful checkpoint."""


def validate_translations(
    sources: Sequence[str], values: Any, provider: str
) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise TranslationBatchError(f"{provider}: ожидался список переводов")
    if len(values) != len(sources):
        raise TranslationBatchError(
            f"{provider}: неверное количество фрагментов: {len(values)} вместо {len(sources)}"
        )
    for index, (source, value) in enumerate(zip(sources, values)):
        if not isinstance(value, str) or (source.strip() and not value.strip()):
            raise TranslationBatchError(
                f"{provider}: пустой или некорректный перевод фрагмента {index + 1}"
            )
    return list(values)


def translation_mapping(
    payload: Any, expected: Mapping[str, str], provider: str
) -> dict[str, str]:
    items = payload.get("translations") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise TranslationBatchError(f"{provider}: отсутствует список translations")
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise TranslationBatchError(
                f"{provider}: некорректный идентификатор перевода"
            )
        key = item["id"]
        if key not in expected or key in result:
            raise TranslationBatchError(
                f"{provider}: лишний или повторный идентификатор {key}"
            )
        result[key] = item.get("text")
    if result.keys() != expected.keys():
        raise TranslationBatchError(f"{provider}: получены не все переводы пакета")
    validate_translations(
        list(expected.values()), [result[key] for key in expected], provider
    )
    return result
