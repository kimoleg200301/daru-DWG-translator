"""Common lossless text splitting and provider batch grouping."""

from typing import Dict, Iterable, List, Sequence, Tuple

OPENAI_SAFE_TEXT = 100000


def chunked(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    if size <= 0:
        raise ValueError("Размер пакета должен быть положительным")
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _group_split_jobs_into_batches(
    split_jobs: List[Tuple[str, int, int, str]],
    part_counts: Dict[int, int],
    batch_size: int,
) -> List[List[Tuple[str, int, int, str]]]:
    """Group split_jobs into batches ensuring all parts of one text stay together."""
    by_orig: Dict[int, List[Tuple[str, int, int, str]]] = {}
    seen_order: List[int] = []
    for job in split_jobs:
        orig_idx = job[1]
        if orig_idx not in by_orig:
            seen_order.append(orig_idx)
        by_orig.setdefault(orig_idx, []).append(job)

    batches: List[List[Tuple[str, int, int, str]]] = []
    current: List[Tuple[str, int, int, str]] = []
    for orig_idx in seen_order:
        group = by_orig[orig_idx]
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def split_payload(text: str, limit: int = OPENAI_SAFE_TEXT) -> List[str]:
    if len(text) <= limit or limit <= 0:
        return [text]
    segments: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + limit, length)
        split_pos = end
        if end < length:
            candidates: List[int] = []
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                candidates.append(newline_pos + 1)
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                candidates.append(space_pos + 1)
            for char in ".!?;:,":
                pos = text.rfind(char, start, end)
                if pos > start:
                    candidates.append(pos + 1)
            if candidates:
                split_pos = max(candidates)
        piece = text[start:split_pos]
        if not piece:
            piece = text[start:end]
            split_pos = end
            if not piece:
                break
        segments.append(piece)
        start = split_pos
    return segments if segments else [text]
