"""Apply translation pairs to DXF entities."""

import argparse
import csv
import re
import sys
from typing import Dict, List, Tuple

import ezdxf

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


def ensure_ru_style(doc) -> str:
    styles = doc.styles
    if STYLE_NAME not in styles:
        styles.new(STYLE_NAME, dxfattribs={"font": STYLE_FONT})
    else:
        try:
            styles.get(STYLE_NAME).dxf.font = STYLE_FONT
        except Exception:
            pass
    return STYLE_NAME


def process_entity(e, pairs: List[Tuple[str, str]], ru_style: str):
    dxft = e.dxftype()

    if dxft == "TEXT":
        t = e.dxf.text or ""
        e.dxf.text = translate_text_keep_dim_and_mtext_controls(t, pairs)
        if ru_style:
            e.dxf.style = ru_style

    elif dxft == "MTEXT":
        t = safe_mtext_text(e)
        new_t = translate_text_keep_dim_and_mtext_controls(t, pairs)
        safe_set_mtext(e, new_t)
        if ru_style:
            try:
                e.dxf.style = ru_style
            except Exception:
                pass

    elif dxft in ("ATTRIB", "ATTDEF"):
        t = e.dxf.text or ""
        e.dxf.text = translate_text_keep_dim_and_mtext_controls(t, pairs)
        if ru_style:
            try:
                e.dxf.style = ru_style
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


def walk_layout(layout, pairs: List[Tuple[str, str]], ru_style: str):
    for e in layout:
        if e.dxftype() in ENTITY_TARGETS:
            process_entity(e, pairs, ru_style)


def walk_blocks(doc, pairs: List[Tuple[str, str]], ru_style: str):
    for block in doc.blocks:
        for e in block:
            if e.dxftype() in ENTITY_TARGETS:
                process_entity(e, pairs, ru_style)


def parse_args():
    p = argparse.ArgumentParser(description="Apply RU translation to DXF using CSV or TXT mapping.")
    p.add_argument("input_dxf", help="Входной DXF")
    p.add_argument("mapping", help="map.csv (text_en,text_ru) ИЛИ translated.txt (русский TXT)")
    p.add_argument("output_dxf", help="Выходной DXF (русский)")
    p.add_argument("--source-en", help="Исходный EN TXT ([count] text). Обязателен, если mapping=*.txt без EN.")
    p.add_argument("--style-font", help=f"TTF шрифт для стиля {STYLE_NAME} (по умолчанию {STYLE_FONT})")
    return p.parse_args()


def main():
    args = parse_args()
    global STYLE_FONT
    if args.style_font:
        STYLE_FONT = args.style_font

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
    ru_style = ensure_ru_style(doc)
    walk_layout(doc.modelspace(), pairs, ru_style)
    for layout in doc.layouts:
        if layout.name != "Model":
            walk_layout(layout, pairs, ru_style)
    walk_blocks(doc, pairs, ru_style)
    doc.saveas(args.output_dxf)
    print("Saved:", args.output_dxf)


if __name__ == "__main__":
    main()
