"""Extract and count text entities from DXF documents."""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import ezdxf

from .entities import (
    ENTITY_TARGETS,
    get_dim_override_text,
    safe_mtext_text,
    safe_table_cell_text,
)


def record_text(bag: List[Tuple[str, str, str]], kind: str, detail: str, text: str) -> None:
    if not isinstance(text, str):
        return
    normalized = text.strip()
    if not normalized:
        return
    bag.append((kind, detail, text))


def process_entity(e, bag: List[Tuple[str, str, str]], prefix: str = "") -> None:
    dxft = e.dxftype()
    kind = prefix or dxft

    if dxft == "TEXT":
        record_text(bag, kind, "", e.dxf.text or "")
        return

    if dxft == "MTEXT":
        record_text(bag, kind, "", safe_mtext_text(e))
        return

    if dxft in ("ATTRIB", "ATTDEF"):
        record_text(bag, kind, getattr(e.dxf, "tag", ""), e.dxf.text or "")
        return

    if dxft == "DIMENSION":
        record_text(bag, kind, "", get_dim_override_text(e))
        return

    if dxft in ("MULTILEADER", "MLEADER", "LEADER"):
        for meth in ("get_mtext", "mtext"):
            try:
                v = getattr(e, meth)
                v = v() if callable(v) else v
                if isinstance(v, str) and v.strip():
                    record_text(bag, kind, "", v)
                    break
            except Exception:
                continue
        return

    if dxft == "TABLE":
        try:
            rows, cols = e.nrows, e.ncols
            for r in range(rows):
                for c in range(cols):
                    val = safe_table_cell_text(e, r, c)
                    if val:
                        table_kind = f"{prefix}:TABLE" if prefix else "TABLE"
                        record_text(bag, table_kind, f"r{r}c{c}", val)
        except Exception:
            pass
        return

    if dxft in ("INSERT", "MINSERT"):
        try:
            for attrib in getattr(e, "attribs", []):
                record_text(bag, f"{kind}:ATTRIB", getattr(attrib.dxf, "tag", ""), getattr(attrib.dxf, "text", ""))
        except Exception:
            pass
        return


def walk_layout(layout, bag):
    for e in layout:
        if e.dxftype() in ENTITY_TARGETS or e.dxftype() in ("INSERT", "MINSERT"):
            process_entity(e, bag)


def walk_blocks(doc, bag):
    for block in doc.blocks:
        name = block.name
        prefix = f"BLOCK:{name}" if name else "BLOCK"
        for e in block:
            if e.dxftype() in ENTITY_TARGETS or e.dxftype() in ("INSERT", "MINSERT"):
                process_entity(e, bag, prefix)


def collect_text_bag(doc) -> List[Tuple[str, str, str]]:
    bag: List[Tuple[str, str, str]] = []
    walk_layout(doc.modelspace(), bag)
    for layout in doc.layouts:
        if layout.name != "Model":
            walk_layout(layout, bag)
    walk_blocks(doc, bag)
    return bag


def build_frequency(bag: Iterable[Tuple[str, str, str]]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for _, _, txt in bag:
        key = (txt or "").strip()
        if key:
            freq[key] = freq.get(key, 0) + 1
    return freq


def sort_frequency(freq: Dict[str, int]) -> List[Tuple[str, int]]:
    return sorted(freq.items(), key=lambda x: (-x[1], x[0]))


def write_csv(freq: Dict[str, int], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["count", "text_en"])
        for text, count in sort_frequency(freq):
            w.writerow([count, text])


def write_json(freq: Dict[str, int], path: Path) -> None:
    items = [{"text_en": text, "count": count} for text, count in sort_frequency(freq)]
    with path.open("w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)


def write_txt(freq: Dict[str, int], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for text, count in sort_frequency(freq):
            f.write(f"[{count}] {text}\n")


def build_entity_types(bag: Iterable[Tuple[str, str, str]]) -> Dict[str, str]:
    """Map each unique text to its dominant DXF entity type (e.g. TEXT, MTEXT, TABLE)."""
    type_counts: Dict[str, Dict[str, int]] = {}
    for kind, _detail, txt in bag:
        key = (txt or "").strip()
        if not key:
            continue
        base_type = kind.split(":")[0] if ":" in kind else kind
        if base_type.startswith("BLOCK"):
            base_type = kind.split(":")[-1] if ":" in kind else "BLOCK"
        bucket = type_counts.setdefault(key, {})
        bucket[base_type] = bucket.get(base_type, 0) + 1
    result: Dict[str, str] = {}
    for text, counts in type_counts.items():
        result[text] = max(counts, key=counts.get)  # type: ignore[arg-type]
    return result


def extract_text_counts(doc) -> Dict[str, int]:
    return build_frequency(collect_text_bag(doc))


def extract_text_counts_and_types(doc) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Return (frequency_map, entity_type_map) for all texts in the document."""
    bag = collect_text_bag(doc)
    return build_frequency(bag), build_entity_types(bag)


def extract_texts(inp: str) -> Dict[str, int]:
    doc = ezdxf.readfile(inp)
    return extract_text_counts(doc)


def main(inp, out_csv="extracted_texts.csv", out_json="extracted_texts.json", out_txt="extracted_texts.txt"):
    doc = ezdxf.readfile(inp)
    freq = extract_text_counts(doc)
    write_csv(freq, Path(out_csv))
    write_json(freq, Path(out_json))
    write_txt(freq, Path(out_txt))
    print(f"OK: {out_csv}, {out_json}, {out_txt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m daru.dxf.extractor input.dxf [out_csv] [out_json] [out_txt]")
        sys.exit(1)
    main(sys.argv[1], *sys.argv[2:5])
