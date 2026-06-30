"""OOXML text segmentation and formatting-preserving replacement helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from lxml import etree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W = f"{{{W_NS}}}"
XML_SPACE = f"{{{XML_NS}}}space"

TEXT_PART_PATTERN = re.compile(
    r"word/(?:document|header\d+|footer\d+|footnotes|endnotes)\.xml$"
)
TOKEN_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\w+|[^\w]+",
    re.UNICODE,
)
IDENTIFIER_PATTERN = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9._/-]+")

MARKER_OPEN = "[[DARU_FMT_{index}]]"
MARKER_CLOSE = "[[/DARU_FMT_{index}]]"
SERVICE_MARKER_PATTERN = re.compile(
    r"\[\[[^\]\r\n]*DARU[^\]\r\n]*(?:\]\]|$)",
    re.IGNORECASE,
)
BARE_SERVICE_MARKER_PATTERN = re.compile(
    r"/?DARU_(?:FMT|TAB)_[A-Za-z0-9_/-]*",
    re.IGNORECASE,
)
SERVICE_MARKER_BYTES_PATTERN = re.compile(
    rb"DARU_(?:FMT|TAB)_",
    re.IGNORECASE,
)

_VISIBLE_SEPARATORS = {
    f"{W}br",
    f"{W}cr",
    f"{W}noBreakHyphen",
    f"{W}softHyphen",
    f"{W}sym",
}
_HIDDEN_CONTAINERS = {
    f"{W}del",
    f"{W}moveFrom",
}
_SEMANTIC_RUN_PROPERTIES = {
    "rStyle",
    "rFonts",
    "b",
    "bCs",
    "i",
    "iCs",
    "u",
    "strike",
    "dstrike",
    "outline",
    "shadow",
    "emboss",
    "imprint",
    "caps",
    "smallCaps",
    "color",
    "sz",
    "szCs",
    "highlight",
    "vertAlign",
    "effect",
    "shd",
    "bdr",
}


@dataclass
class ParsedXmlPart:
    """Parsed XML part plus pending text-node replacements."""

    name: str
    root: etree._Element
    replacements: Dict[etree._Element, List[Tuple[int, int, str]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class TextSlice:
    """A source substring backed by a concrete ``w:t`` node."""

    node: Optional[etree._Element]
    start: int
    end: int
    text: str
    style_key: Tuple[object, ...]
    is_tab: bool = False


@dataclass
class FormatSegment:
    """Adjacent text slices that share meaningful Word run formatting."""

    slices: List[TextSlice]

    @property
    def text(self) -> str:
        return "".join(item.text for item in self.slices)

    @property
    def is_tab(self) -> bool:
        return all(item.is_tab for item in self.slices)

    def encoded_text(self, index: int) -> str:
        if self.is_tab:
            return f"[[DARU_TAB_{index}_{len(self.text)}]]"
        return self.text


@dataclass
class TranslationUnit:
    """One source-language range translated as a coherent unit."""

    part_name: str
    source_text: str
    segments: List[FormatSegment]

    @property
    def has_tabs(self) -> bool:
        return any(segment.is_tab for segment in self.segments)

    @property
    def encoded_text(self) -> str:
        if self.has_tabs:
            return self.source_text
        if len(self.segments) == 1:
            return self.source_text
        chunks: List[str] = []
        for index, segment in enumerate(self.segments):
            chunks.append(MARKER_OPEN.format(index=index))
            chunks.append(segment.encoded_text(index))
            chunks.append(MARKER_CLOSE.format(index=index))
        return "".join(chunks)


@dataclass
class DocumentModel:
    """Mutable parsed DOCX text model."""

    parts: Dict[str, ParsedXmlPart]
    units: List[TranslationUnit]
    skipped_items: int


@dataclass(frozen=True)
class _TextNodeEntry:
    node: Optional[etree._Element]
    text: str
    style_key: Tuple[object, ...]
    is_tab: bool = False


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    kind: str


def is_text_part(name: str) -> bool:
    return bool(TEXT_PART_PATTERN.fullmatch(name))


def parse_xml_part(name: str, data: bytes) -> ParsedXmlPart:
    parser = etree.XMLParser(
        remove_blank_text=False,
        resolve_entities=False,
        no_network=True,
        recover=False,
    )
    try:
        root = etree.fromstring(data, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise RuntimeError(f"Некорректная XML-часть DOCX: {name}: {exc}") from exc
    return ParsedXmlPart(name=name, root=root)


def build_document_model(parts: Dict[str, ParsedXmlPart], source_lang: str) -> DocumentModel:
    units: List[TranslationUnit] = []
    skipped_items = 0

    for part_name in sorted(parts, key=_part_sort_key):
        part = parts[part_name]
        for paragraph in part.root.iter(f"{W}p"):
            chunks = _collect_paragraph_chunks(paragraph)
            for entries in chunks:
                combined = "".join(entry.text for entry in entries)
                if not combined.strip():
                    continue
                ranges = _source_language_ranges(combined, source_lang)
                if not ranges:
                    skipped_items += 1
                    continue
                for start, end in ranges:
                    translation_unit = _build_translation_unit(
                        part_name,
                        entries,
                        start,
                        end,
                    )
                    if translation_unit is not None:
                        units.append(translation_unit)

    return DocumentModel(parts=parts, units=units, skipped_items=skipped_items)


def decode_formatted_translation(
    unit: TranslationUnit,
    translated: str,
) -> Optional[List[str]]:
    """Return translated text for every formatting segment."""

    if unit.has_tabs:
        return decode_plain_translation(unit, translated)
    if len(unit.segments) == 1:
        return [strip_service_markers(translated)]

    values: List[str] = []
    position = 0
    for index in range(len(unit.segments)):
        opening = MARKER_OPEN.format(index=index)
        closing = MARKER_CLOSE.format(index=index)
        if not translated.startswith(opening, position):
            return None
        content_start = position + len(opening)
        content_end = translated.find(closing, content_start)
        if content_end < 0:
            return None
        values.append(strip_service_markers(translated[content_start:content_end]))
        position = content_end + len(closing)
    if position != len(translated):
        return None
    for index, (segment, value) in enumerate(zip(unit.segments, values)):
        if segment.is_tab and value != segment.encoded_text(index):
            return None
    return values


def decode_plain_translation(
    unit: TranslationUnit,
    translated: str,
) -> List[str]:
    """Distribute a marker-free translation over existing OOXML segments."""

    cleaned = strip_service_markers(translated)
    if not unit.has_tabs:
        values = [""] * len(unit.segments)
        first_text_index = next(
            (
                index
                for index, segment in enumerate(unit.segments)
                if not segment.is_tab
            ),
            None,
        )
        if first_text_index is not None:
            values[first_text_index] = cleaned
        return values

    groups = _segment_groups_around_tabs(unit.segments)
    expected_tabs = max(0, len(groups) - 1)
    if cleaned.count("\t") == expected_tabs:
        translated_groups = cleaned.split("\t")
    else:
        translated_groups = _split_text_by_weights(
            cleaned.replace("\t", " "),
            [
                sum(len(unit.segments[index].text.strip()) for index in group)
                for group in groups
            ],
        )

    values = [""] * len(unit.segments)
    for index, segment in enumerate(unit.segments):
        if segment.is_tab:
            values[index] = segment.text
    for group, translated_group in zip(groups, translated_groups):
        if group:
            values[group[0]] = translated_group.strip()
    return values


def strip_service_markers(text: str) -> str:
    cleaned = SERVICE_MARKER_PATTERN.sub("", text or "")
    return BARE_SERVICE_MARKER_PATTERN.sub("", cleaned)


def segment_needs_translation(text: str, source_lang: str) -> bool:
    return bool(_source_language_ranges(text, source_lang))


def schedule_unit_replacement(
    model: DocumentModel,
    unit: TranslationUnit,
    translated_segments: Sequence[str],
) -> None:
    if len(translated_segments) != len(unit.segments):
        raise RuntimeError("Количество переведенных DOCX-сегментов не совпадает с исходным")

    part = model.parts[unit.part_name]
    for segment, translated in zip(unit.segments, translated_segments):
        if segment.is_tab:
            continue
        translated = strip_service_markers(translated)
        first = True
        for text_slice in segment.slices:
            if text_slice.node is None:
                continue
            replacement = translated if first else ""
            part.replacements.setdefault(text_slice.node, []).append(
                (text_slice.start, text_slice.end, replacement)
            )
            first = False


def apply_scheduled_replacements(model: DocumentModel) -> Dict[str, bytes]:
    modified_parts: Dict[str, bytes] = {}
    for part_name, part in model.parts.items():
        if not part.replacements:
            continue
        for node, replacements in part.replacements.items():
            original = node.text or ""
            ordered = sorted(replacements, key=lambda item: item[0])
            cursor = 0
            chunks: List[str] = []
            for start, end, replacement in ordered:
                if start < cursor or end < start or end > len(original):
                    raise RuntimeError("Пересекающиеся или некорректные замены текста DOCX")
                chunks.append(original[cursor:start])
                chunks.append(replacement)
                cursor = end
            chunks.append(original[cursor:])
            updated = "".join(chunks)
            node.text = updated
            if updated[:1].isspace() or updated[-1:].isspace():
                node.set(XML_SPACE, "preserve")
            else:
                node.attrib.pop(XML_SPACE, None)

        modified_parts[part_name] = etree.tostring(
            part.root,
            encoding="UTF-8",
            xml_declaration=True,
        )
        if SERVICE_MARKER_BYTES_PATTERN.search(modified_parts[part_name]):
            raise RuntimeError(
                "Служебный маркер DOCX остался в подготовленном документе"
            )
    return modified_parts


def _segment_groups_around_tabs(
    segments: Sequence[FormatSegment],
) -> List[List[int]]:
    groups: List[List[int]] = [[]]
    for index, segment in enumerate(segments):
        if segment.is_tab:
            groups.append([])
        else:
            groups[-1].append(index)
    return groups


def _split_text_by_weights(text: str, weights: Sequence[int]) -> List[str]:
    if not weights:
        return []
    if len(weights) == 1:
        return [text]

    normalized_weights = [max(1, weight) for weight in weights]
    total_weight = sum(normalized_weights)
    boundaries = [match.end() for match in re.finditer(r"\s+", text)]
    cuts: List[int] = []
    previous = 0
    cumulative = 0
    for weight in normalized_weights[:-1]:
        cumulative += weight
        target = int(len(text) * cumulative / total_weight)
        candidates = [boundary for boundary in boundaries if boundary > previous]
        cut = min(candidates, key=lambda value: abs(value - target)) if candidates else target
        cut = max(previous, min(len(text), cut))
        cuts.append(cut)
        previous = cut

    parts: List[str] = []
    start = 0
    for cut in cuts:
        parts.append(text[start:cut])
        start = cut
    parts.append(text[start:])
    return parts


def _part_sort_key(name: str) -> Tuple[int, str]:
    if name == "word/document.xml":
        return (0, name)
    if "/header" in name:
        return (1, name)
    if "/footer" in name:
        return (2, name)
    if name == "word/footnotes.xml":
        return (3, name)
    if name == "word/endnotes.xml":
        return (4, name)
    return (5, name)


def _collect_paragraph_chunks(
    paragraph: etree._Element,
) -> List[List[_TextNodeEntry]]:
    chunks: List[List[_TextNodeEntry]] = []
    current: List[_TextNodeEntry] = []
    field_depth = 0

    def boundary() -> None:
        nonlocal current
        if current:
            chunks.extend(_split_non_inline_tabs(current))
            current = []

    def visit(element: etree._Element) -> None:
        nonlocal field_depth
        tag = element.tag

        if element is not paragraph and tag == f"{W}p":
            return
        if tag in _HIDDEN_CONTAINERS:
            boundary()
            return
        if tag == f"{W}fldSimple":
            boundary()
            return
        if tag == f"{W}r" and _run_is_hidden(element):
            boundary()
            return
        if tag == f"{W}fldChar":
            field_type = element.get(f"{W}fldCharType", "")
            if field_type == "begin":
                boundary()
                field_depth += 1
            elif field_type == "end":
                field_depth = max(0, field_depth - 1)
                boundary()
            elif field_type == "separate":
                boundary()
            return
        if tag == f"{W}tab":
            if field_depth == 0:
                current.append(
                    _TextNodeEntry(
                        node=None,
                        text="\t",
                        style_key=("__DARU_TAB__",),
                        is_tab=True,
                    )
                )
            return
        if tag in _VISIBLE_SEPARATORS:
            boundary()
            return
        if tag == f"{W}t":
            if field_depth == 0 and element.text:
                current.append(
                    _TextNodeEntry(
                        node=element,
                        text=element.text,
                        style_key=_style_key(element, paragraph),
                        is_tab=False,
                    )
                )
            return

        for child in element:
            visit(child)

    visit(paragraph)
    boundary()
    return chunks


def _split_non_inline_tabs(
    entries: Sequence[_TextNodeEntry],
) -> List[List[_TextNodeEntry]]:
    chunks: List[List[_TextNodeEntry]] = []
    current: List[_TextNodeEntry] = []

    for index, entry in enumerate(entries):
        if entry.is_tab and not _tab_connects_words(entries, index):
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(entry)

    if current:
        chunks.append(current)
    return chunks


def _tab_connects_words(
    entries: Sequence[_TextNodeEntry],
    tab_index: int,
) -> bool:
    left = "".join(entry.text for entry in entries[:tab_index] if not entry.is_tab)
    right = "".join(
        entry.text for entry in entries[tab_index + 1 :] if not entry.is_tab
    )
    return _nearest_significant_token_is_word(left, reverse=True) and (
        _nearest_significant_token_is_word(right, reverse=False)
    )


def _nearest_significant_token_is_word(text: str, *, reverse: bool) -> bool:
    tokens = list(TOKEN_PATTERN.finditer(text))
    iterable = reversed(tokens) if reverse else iter(tokens)
    for match in iterable:
        value = match.group(0)
        if not any(char.isalnum() for char in value):
            continue
        return any(char.isalpha() for char in value) and not any(
            char.isdigit() for char in value
        )
    return False


def _run_is_hidden(run: etree._Element) -> bool:
    run_properties = run.find(f"{W}rPr")
    if run_properties is None:
        return False
    return any(
        _on_off_property_enabled(run_properties.find(f"{W}{name}"))
        for name in ("vanish", "webHidden")
    )


def _on_off_property_enabled(element: Optional[etree._Element]) -> bool:
    if element is None:
        return False
    value = (element.get(f"{W}val") or "true").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _style_key(
    text_node: etree._Element,
    paragraph: etree._Element,
) -> Tuple[object, ...]:
    run = _nearest_ancestor(text_node, f"{W}r", paragraph)
    hyperlink = _nearest_ancestor(text_node, f"{W}hyperlink", paragraph)
    properties: List[bytes] = []
    if run is not None:
        run_properties = run.find(f"{W}rPr")
        if run_properties is not None:
            for child in run_properties:
                local_name = etree.QName(child).localname
                if local_name in _SEMANTIC_RUN_PROPERTIES:
                    properties.append(etree.tostring(child, method="c14n"))
    return (
        id(hyperlink) if hyperlink is not None else None,
        tuple(properties),
    )


def _nearest_ancestor(
    node: etree._Element,
    tag: str,
    stop: etree._Element,
) -> Optional[etree._Element]:
    parent = node.getparent()
    while parent is not None and parent is not stop:
        if parent.tag == tag:
            return parent
        parent = parent.getparent()
    return None


def _source_language_ranges(text: str, source_lang: str) -> List[Tuple[int, int]]:
    tokens = _classify_tokens(text, source_lang)
    ranges: List[Tuple[int, int]] = []
    region: List[_Token] = []

    def flush() -> None:
        nonlocal region
        source_tokens = [token for token in region if token.kind == "source"]
        if source_tokens:
            start = source_tokens[0].start
            end = source_tokens[-1].end
            candidate = text[start:end]
            if _is_meaningful_translation_candidate(candidate):
                ranges.append((start, end))
        region = []

    for token in tokens:
        if token.kind in {"foreign", "protected"}:
            flush()
        else:
            region.append(token)
    flush()
    return ranges


def _classify_tokens(text: str, source_lang: str) -> List[_Token]:
    result: List[_Token] = []
    source_groups = _source_script_groups(source_lang)
    auto_mode = (source_lang or "").strip().lower() in {"", "auto"}

    for match in TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        if _is_url_or_email(value):
            kind = "protected"
        else:
            letter_groups = {
                group
                for char in value
                if char.isalpha()
                for group in [_character_script_group(char)]
                if group is not None
            }
            if not letter_groups:
                kind = "neutral"
            elif auto_mode or letter_groups.issubset(source_groups):
                kind = "source"
            else:
                kind = "foreign"
        result.append(_Token(match.start(), match.end(), kind))
    return result


def _source_script_groups(source_lang: str) -> set[str]:
    language = (source_lang or "").strip().lower().split("-", 1)[0]
    if language in {"ru", "uk", "be", "bg", "sr", "mk"}:
        return {"CYRILLIC"}
    if language == "zh":
        return {"HAN"}
    if language == "ja":
        return {"HAN", "HIRAGANA", "KATAKANA"}
    if language == "ko":
        return {"HANGUL", "HAN"}
    if language in {"en", "de", "fr", "es", "it", "pl", "pt", "nl", "cs", "sk"}:
        return {"LATIN"}
    return {"LATIN", "CYRILLIC", "HAN", "HIRAGANA", "KATAKANA", "HANGUL"}


def _character_script_group(char: str) -> Optional[str]:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return None
    if "LATIN" in name:
        return "LATIN"
    if "CYRILLIC" in name:
        return "CYRILLIC"
    if "HIRAGANA" in name:
        return "HIRAGANA"
    if "KATAKANA" in name:
        return "KATAKANA"
    if "HANGUL" in name:
        return "HANGUL"
    if "CJK UNIFIED IDEOGRAPH" in name or "IDEOGRAPH" in name:
        return "HAN"
    return "OTHER"


def _is_url_or_email(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered.startswith(("http://", "https://", "www."))
        or ("@" in value and "." in value.rsplit("@", 1)[-1])
    )


def _is_meaningful_translation_candidate(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 2:
        return False
    if _is_url_or_email(stripped):
        return False
    if IDENTIFIER_PATTERN.fullmatch(stripped):
        return False
    if re.fullmatch(r"[^\W\d_]\.?", stripped, re.UNICODE):
        return False
    if stripped.isupper() and len(stripped) <= 3 and " " not in stripped:
        return False
    return True


def _build_translation_unit(
    part_name: str,
    entries: Sequence[_TextNodeEntry],
    range_start: int,
    range_end: int,
) -> Optional[TranslationUnit]:
    slices: List[TextSlice] = []
    offset = 0
    for entry in entries:
        node_start = offset
        node_end = offset + len(entry.text)
        overlap_start = max(range_start, node_start)
        overlap_end = min(range_end, node_end)
        if overlap_start < overlap_end:
            local_start = overlap_start - node_start
            local_end = overlap_end - node_start
            slices.append(
                TextSlice(
                    node=entry.node,
                    start=local_start,
                    end=local_end,
                    text=entry.text[local_start:local_end],
                    style_key=entry.style_key,
                    is_tab=entry.is_tab,
                )
            )
        offset = node_end

    if not slices:
        return None

    format_segments: List[FormatSegment] = []
    for text_slice in slices:
        if (
            format_segments
            and format_segments[-1].slices[-1].style_key == text_slice.style_key
        ):
            format_segments[-1].slices.append(text_slice)
        else:
            format_segments.append(FormatSegment(slices=[text_slice]))

    source_text = "".join(item.text for item in slices)
    return TranslationUnit(
        part_name=part_name,
        source_text=source_text,
        segments=format_segments,
    )


__all__ = [
    "DocumentModel",
    "FormatSegment",
    "ParsedXmlPart",
    "TranslationUnit",
    "apply_scheduled_replacements",
    "build_document_model",
    "decode_formatted_translation",
    "decode_plain_translation",
    "is_text_part",
    "parse_xml_part",
    "schedule_unit_replacement",
    "segment_needs_translation",
    "strip_service_markers",
]
