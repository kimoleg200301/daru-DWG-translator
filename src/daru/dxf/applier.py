"""Apply translation pairs to DXF entities."""

import argparse
import csv
import re
import sys
import unicodedata
from typing import Callable, Dict, List, Optional, Set, Tuple

import ezdxf
from ezdxf.fonts import fonts

from ..config import ORIGINAL_FONT_VALUE, normalize_style_font
from .entities import (
    ENTITY_TARGETS,
    get_dim_override_text,
    safe_get_mleader_text,
    safe_mtext_text,
    safe_set_mleader_text,
    safe_set_mtext,
    safe_table_cell_text,
    safe_table_set_text,
    set_dim_override_text,
    translate_text_keep_dim_and_mtext_controls,
)

STYLE_NAME = "RU"
STYLE_FONT = "Arial.ttf"
FALLBACK_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "ArialUnicode.ttf",
    "NotoSans-Regular.ttf",
    "Arial.ttf",
    "SegoeUI.ttf",
)


def _strip_count_prefix(s: str) -> str:
    m = re.match(r"^\s*\[\d+\]\s*(.*)$", s.strip())
    return m.group(1) if m else s.strip()


def load_map_from_csv(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = [c.lower() for c in r.fieldnames or []]
        c_en = "text_en" if "text_en" in cols else (r.fieldnames[0] if r.fieldnames else None)
        c_ru = "text_ru" if "text_ru" in cols else (r.fieldnames[1] if r.fieldnames and len(r.fieldnames) > 1 else None)
        if not c_en or not c_ru:
            raise ValueError("CSV должен содержать две колонки: text_en,text_ru")
        for row in r:
            en = (row.get(c_en) or "").strip()
            ru = (row.get(c_ru) or "").strip()
            if en:
                pairs.append((en, ru))
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


def load_map_from_txt_pair(en_txt: str, ru_txt: str) -> List[Tuple[str, str]]:
    with open(en_txt, "r", encoding="utf-8", errors="ignore") as f:
        en_lines = [ln.rstrip("\r\n") for ln in f]
    with open(ru_txt, "r", encoding="utf-8", errors="ignore") as f:
        ru_lines = [ln.rstrip("\r\n") for ln in f]
    en_clean = [_strip_count_prefix(x) for x in en_lines if x.strip()]
    ru_clean = [_strip_count_prefix(x) for x in ru_lines if x.strip()]
    n = min(len(en_clean), len(ru_clean))
    pairs = list(zip(en_clean[:n], ru_clean[:n]))
    seen: Dict[str, str] = {}
    for en, ru in pairs:
        if en not in seen:
            seen[en] = ru
    pairs = [(k, v) for k, v in seen.items()]
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


def ensure_ru_style(doc, style_font: str = STYLE_FONT, style_name: str = STYLE_NAME) -> str:
    styles = doc.styles
    if style_name not in styles:
        styles.new(style_name, dxfattribs={"font": style_font})
    else:
        try:
            styles.get(style_name).dxf.font = style_font
        except Exception:
            pass
    return style_name


def _text_codepoints(text: str) -> Set[int]:
    return {
        ord(char)
        for char in str(text or "")
        if not char.isspace() and not unicodedata.category(char).startswith("C")
    }


def _font_supports_text(font_name: str, text: str) -> Tuple[bool, str]:
    if not font_name or not fonts.font_manager.has_font(font_name):
        return False, "unavailable"
    codepoints = _text_codepoints(text)
    if not codepoints:
        return True, ""
    suffix = str(font_name).lower()
    try:
        if suffix.endswith((".shx", ".shp")):
            cache = fonts.font_manager.get_shapefile_glyph_cache(font_name)
            available = set(cache.font.shapes)
        elif suffix.endswith(".lff"):
            cache = fonts.font_manager.get_lff_glyph_cache(font_name)
            available = set(cache.font)
        else:
            cmap = fonts.font_manager.get_ttf_font(font_name).getBestCmap() or {}
            available = set(cmap)
    except Exception:
        return False, "unavailable"
    return (True, "") if codepoints.issubset(available) else (False, "missing_glyphs")


def _style_font_name(doc, style_name: str) -> str:
    try:
        style = doc.styles.get(style_name)
    except Exception:
        return ""
    font_name = str(style.dxf.get("font", "") or "").strip()
    if font_name:
        return font_name
    try:
        family, italic, bold = style.get_extended_font_data()
        if family:
            face = fonts.font_manager.find_best_match(
                family=family,
                italic=italic,
                weight=700 if bold else 400,
            )
            if face is not None:
                return fonts.find_font_file_name(face)
    except Exception:
        pass
    return ""


class FontStyleResolver:
    """Resolve a text style per entity without mutating module-global state."""

    def __init__(
        self,
        doc,
        style_font: Optional[str],
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.doc = doc
        self.style_font = normalize_style_font(style_font or STYLE_FONT) or STYLE_FONT
        self.original_mode = self.style_font == ORIGINAL_FONT_VALUE
        self.log = log or (lambda _message: None)
        self._fallback_styles: Dict[str, str] = {}
        self._warned: Set[Tuple[str, str]] = set()

    def style_for_entity(self, entity, translated_text: str) -> str:
        if not self.original_mode:
            return self._fallback_style(self.style_font)

        original_style = str(entity.dxf.get("style", "Standard") or "Standard")
        font_name = _style_font_name(self.doc, original_style)
        supported, reason = _font_supports_text(font_name, translated_text)
        if supported:
            return original_style

        fallback_font = self._find_fallback_font(translated_text)
        self._warn_once(font_name or original_style, reason, fallback_font)
        return self._fallback_style(fallback_font)

    def _find_fallback_font(self, text: str) -> str:
        for font_name in FALLBACK_FONT_CANDIDATES:
            supported, _reason = _font_supports_text(font_name, text)
            if supported:
                return font_name
        return STYLE_FONT

    def _fallback_style(self, font_name: str) -> str:
        cached = self._fallback_styles.get(font_name)
        if cached:
            return cached
        if not self.original_mode:
            resolved = ensure_ru_style(self.doc, font_name)
            self._fallback_styles[font_name] = resolved
            return resolved
        style_name = STYLE_NAME
        suffix = 2
        while style_name in self.doc.styles:
            style_name = f"{STYLE_NAME}_{suffix}"
            suffix += 1
        resolved = ensure_ru_style(self.doc, font_name, style_name)
        self._fallback_styles[font_name] = resolved
        return resolved

    def _warn_once(self, font_name: str, reason: str, fallback_font: str) -> None:
        key = (font_name.casefold(), reason)
        if key in self._warned:
            return
        self._warned.add(key)
        if reason == "missing_glyphs":
            detail = "не содержит всех символов перевода"
        else:
            detail = "недоступен в системе"
        self.log(
            f"CAD: исходный шрифт «{font_name}» {detail}; "
            f"используется {fallback_font}"
        )


def process_entity(
    e,
    pairs: List[Tuple[str, str]],
    ru_style: str,
    style_resolver: Optional[FontStyleResolver] = None,
):
    dxft = e.dxftype()

    if dxft == "TEXT":
        t = e.dxf.text or ""
        new_t = translate_text_keep_dim_and_mtext_controls(t, pairs)
        e.dxf.text = new_t
        target_style = style_resolver.style_for_entity(e, new_t) if style_resolver else ru_style
        if target_style:
            e.dxf.style = target_style

    elif dxft == "MTEXT":
        t = safe_mtext_text(e)
        new_t = translate_text_keep_dim_and_mtext_controls(t, pairs)
        safe_set_mtext(e, new_t)
        target_style = style_resolver.style_for_entity(e, new_t) if style_resolver else ru_style
        if target_style:
            try:
                e.dxf.style = target_style
            except Exception:
                pass

    elif dxft in ("ATTRIB", "ATTDEF"):
        t = e.dxf.text or ""
        new_t = translate_text_keep_dim_and_mtext_controls(t, pairs)
        e.dxf.text = new_t
        target_style = style_resolver.style_for_entity(e, new_t) if style_resolver else ru_style
        if target_style:
            try:
                e.dxf.style = target_style
            except Exception:
                pass

    elif dxft == "DIMENSION":
        t = get_dim_override_text(e)
        if t:
            set_dim_override_text(e, translate_text_keep_dim_and_mtext_controls(t, pairs))

    elif dxft in ("MULTILEADER", "MLEADER", "LEADER"):
        t = safe_get_mleader_text(e)
        if t:
            safe_set_mleader_text(e, translate_text_keep_dim_and_mtext_controls(t, pairs))

    elif dxft == "TABLE":
        try:
            rows, cols = e.nrows, e.ncols
            for r in range(rows):
                for c in range(cols):
                    val = safe_table_cell_text(e, r, c)
                    if isinstance(val, str) and val.strip():
                        safe_table_set_text(e, r, c, translate_text_keep_dim_and_mtext_controls(val, pairs))
        except Exception:
            pass


def walk_layout(
    layout,
    pairs: List[Tuple[str, str]],
    ru_style: str,
    style_resolver: Optional[FontStyleResolver] = None,
):
    for e in layout:
        if e.dxftype() in ENTITY_TARGETS:
            process_entity(e, pairs, ru_style, style_resolver)


def walk_blocks(
    doc,
    pairs: List[Tuple[str, str]],
    ru_style: str,
    style_resolver: Optional[FontStyleResolver] = None,
):
    for block in doc.blocks:
        for e in block:
            if e.dxftype() in ENTITY_TARGETS:
                process_entity(e, pairs, ru_style, style_resolver)


def parse_args():
    p = argparse.ArgumentParser(description="Apply RU translation to DXF using CSV or TXT mapping.")
    p.add_argument("input_dxf", help="Входной DXF")
    p.add_argument("mapping", help="map.csv (text_en,text_ru) ИЛИ translated.txt (русский TXT)")
    p.add_argument("output_dxf", help="Выходной DXF (русский)")
    p.add_argument("--source-en", help="Исходный EN TXT ([count] text). Обязателен, если mapping=*.txt без EN.")
    p.add_argument(
        "--style-font",
        help=(
            f"TTF/SHX шрифт для стиля {STYLE_NAME} или original "
            f"(по умолчанию {STYLE_FONT})"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    style_font = args.style_font or STYLE_FONT

    mapping_path = args.mapping.lower()
    if mapping_path.endswith(".csv"):
        pairs = load_map_from_csv(args.mapping)
    elif mapping_path.endswith(".txt"):
        if not args.source_en:
            print("Ошибка: для TXT-перевода нужен исходный английский TXT через --source-en <path>")
            sys.exit(2)
        pairs = load_map_from_txt_pair(args.source_en, args.mapping)
    else:
        print("mapping должен быть .csv или .txt")
        sys.exit(2)

    if not pairs:
        print("Не удалось загрузить пары переводов.")
        sys.exit(3)

    doc = ezdxf.readfile(args.input_dxf)
    resolver = FontStyleResolver(doc, style_font, print)
    walk_layout(doc.modelspace(), pairs, "", resolver)
    for layout in doc.layouts:
        if layout.name != "Model":
            walk_layout(layout, pairs, "", resolver)
    walk_blocks(doc, pairs, "", resolver)
    doc.saveas(args.output_dxf)
    print("Saved:", args.output_dxf)


if __name__ == "__main__":
    main()
