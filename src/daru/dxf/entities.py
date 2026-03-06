"""Shared DXF entity access helpers — deduplicated from extractor and applier."""

import re
from typing import List, Tuple


def get_dim_override_text(dim) -> str:
    try:
        t = dim.get_text()
        return t if t and t.strip() != "<>" else ""
    except Exception:
        return ""


def set_dim_override_text(dim, s: str):
    try:
        if s and s.strip():
            dim.set_text(s)
    except Exception:
        pass


def safe_mtext_text(e) -> str:
    for attr in ("plain_text", "text"):
        try:
            v = getattr(e, attr)
            v = v() if callable(v) else v
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    try:
        v = e.text
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
    for meth in ("get_mtext", "mtext"):
        try:
            v = getattr(e, meth)
            v = v() if callable(v) else v
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    return ""


def safe_set_mleader_text(e, s: str):
    for meth in ("set_mtext", "set_text", "set_mleader_text"):
        try:
            fn = getattr(e, meth)
            fn(s)
            return
        except Exception:
            pass


def protect_dim(text: str):
    return text.split("<>"), "<>"


def replace_all_exact(text: str, pairs: List[Tuple[str, str]]) -> str:
    out = text
    for en, ru in pairs:
        if en:
            out = out.replace(en, ru)
    return out


def translate_text_keep_dim_and_mtext_controls(s: str, pairs: List[Tuple[str, str]]) -> str:
    if not s:
        return s
    chunks, sep = protect_dim(s)
    out_chunks = []
    for ch in chunks:
        segments = re.split(r"(\\[A-Za-z][^\\]*)", ch)
        for i, seg in enumerate(segments):
            if not seg or re.fullmatch(r"\\[A-Za-z][^\\]*", seg):
                continue
            segments[i] = replace_all_exact(seg, pairs)
        out_chunks.append("".join(segments))
    return sep.join(out_chunks)


ENTITY_TARGETS = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF", "DIMENSION", "MULTILEADER", "MLEADER", "LEADER", "TABLE"}
