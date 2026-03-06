"""File I/O utilities: CSV/TXT read/write, path helpers."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def ensure_parent(path: Path) -> None:
    if path and path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def paths_equal(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        return a == b


def write_translated_txt(path: Path, items: List[Tuple[str, int]], translations: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for (text, count), translated in zip(items, translations):
            f.write(f"[{count}] {translated}\n")


def write_map_csv(path: Path, english: List[str], russian: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text_en", "text_ru"])
        for en, ru in zip(english, russian):
            writer.writerow([en, ru])


def read_map_csv(path: Optional[Path]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path:
        return cache
    try:
        if not path.exists():
            return cache
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                en = (row.get("text_en") or "").strip()
                ru = row.get("text_ru") or ""
                if en:
                    cache[en] = ru
    except Exception:
        return {}
    return cache
