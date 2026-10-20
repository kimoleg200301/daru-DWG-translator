"""Shared DXF entity access helpers — deduplicated from extractor and applier."""

import re
from typing import Iterable, Iterator, List, Tuple


def get_dim_override_text(dim) -> str:
    try:
        t = dim.dxf.text if hasattr(dim, "dxf") else dim.get_text()
        return t if t and t.strip() != "<>" else ""
    except Exception:
        return ""


def set_dim_override_text(dim, s: str):
    try:
        if s and s.strip():
            if hasattr(dim, "dxf"):
                dim.dxf.text = s
            else:
                dim.set_text(s)
    except Exception:
        pass


def safe_mtext_text(e) -> str:
    # plain_text() discards inline formatting even when no translation is made.
    for attr in ("text", "plain_text"):
        try:
            v = getattr(e, attr)
            v = v() if callable(v) else v
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    return ""


def safe_set_mtext(e, s: str):
    try:
        e.text = s
    except Exception:
        pass


def safe_table_cell_text(tbl, r, c) -> str:
    for meth in ("text_cell_content", "get_cell_text", "get_text"):
        try:
            fn = getattr(tbl, meth)
            v = fn(r, c)
            if isinstance(v, str):
                return v
        except Exception:
            pass
    return ""


def safe_table_set_text(tbl, r, c, s: str):
    for meth in ("set_text_cell_content", "set_cell_text", "set_text"):
        try:
            fn = getattr(tbl, meth)
            fn(r, c, s)
            return
        except Exception:
            pass


def safe_get_mleader_text(e) -> str:
    for meth in ("get_mtext_content", "get_mtext", "mtext"):
        try:
            v = getattr(e, meth)
            v = v() if callable(v) else v
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    return ""


def safe_set_mleader_text(e, s: str):
    for meth in ("set_mtext_content", "set_mtext", "set_text", "set_mleader_text"):
        try:
            fn = getattr(e, meth)
            fn(s)
            return
        except Exception:
            pass


def protect_dim(text: str):
    return text.split("<>"), "<>"


_MTEXT_CONTROLS = re.compile(
    r"(<>|\\(?:[AaCcFfHhQqSsTtWwp][^;\\{}]*;|U\+[0-9A-Fa-f]{4}|[A-Za-z~{}\\])|[{}])"
)


class TextReplacer:
    """Reuse a translation index and replace source text without cascading."""

    def __init__(self, pairs: Iterable[Tuple[str, str]]) -> None:
        self._translations: dict[str, str] = {}
        for source, translated in pairs:
            if source:
                self._translations.setdefault(source, translated)
        self._pattern: re.Pattern[str] | None = None

    def replace(self, text: str) -> str:
        if text in self._translations:
            return self._translations[text]
        if not text or not self._translations:
            return text
        if self._pattern is None:
            sources = sorted(self._translations, key=len, reverse=True)
            self._pattern = re.compile("|".join(re.escape(source) for source in sources))
        return self._pattern.sub(lambda match: self._translations[match.group()], text)

    def translate(self, text: str) -> str:
        translated = self._translations.get(text)
        if translated is None:
            # Extraction indexes stripped text; retain the entity's margins.
            translated = self._translations.get(text.strip())
            if translated is not None:
                leading = len(text) - len(text.lstrip())
                end = len(text.rstrip())
                translated = text[:leading] + translated + text[end:]
        # Whole MTEXT and dimension translations are safe when their controls
        # are preserved. Otherwise only translate the visible text segments.
        if translated is not None and _MTEXT_CONTROLS.findall(text) == _MTEXT_CONTROLS.findall(translated):
            return translated
        segments = _MTEXT_CONTROLS.split(text)
        for index in range(0, len(segments), 2):
            segments[index] = self.replace(segments[index])
        return "".join(segments)


TranslationPairs = List[Tuple[str, str]] | TextReplacer


def replace_all_exact(text: str, pairs: TranslationPairs) -> str:
    replacer = pairs if isinstance(pairs, TextReplacer) else TextReplacer(pairs)
    return replacer.replace(text)


def translate_text_keep_dim_and_mtext_controls(s: str, pairs: TranslationPairs) -> str:
    if not s:
        return s
    replacer = pairs if isinstance(pairs, TextReplacer) else TextReplacer(pairs)
    return replacer.translate(s)


ENTITY_TARGETS = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "DIMENSION", "MULTILEADER", "MLEADER", "LEADER", "TABLE"}


def iter_text_entities(layout) -> Iterator:
    """Visit stored text, including attributes owned by block references."""
    for entity in layout:
        entity_type = entity.dxftype()
        if entity_type in ENTITY_TARGETS:
            yield entity
        elif entity_type in {"INSERT", "MINSERT"}:
            yield from entity.attribs


def iter_block_definitions(doc) -> Iterator:
    """Modelspace and paperspace also occur in doc.blocks; skip those copies."""
    for block in doc.blocks:
        if not block.is_any_layout:
            yield block
